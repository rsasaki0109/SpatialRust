"""Post-fit stage attribution from hash-bound saved native registration reports."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from compare_3dmatch_cases import validate_cases, validate_saved_receipt
from evaluate_pose_reference import validate_reference
from evaluate_redwood_logs import squared_information_error
from redwood_reference import load_records


def stage_scores(report, truth, information):
    stages = [('initial', report['initial_transform_source_to_target'])] + [
        (stage['name'], stage['transform_source_to_target']) for stage in report['stages']]
    if not stages or not np.array_equal(stages[-1][1], report['transform_source_to_target']):
        raise ValueError('final stage does not match the saved output pose')
    rows = []
    for name, matrix in stages:
        row = dict(stage=name, correct=False, score_squared=None)
        try:
            score = squared_information_error(truth, np.asarray(matrix), information)
            row.update(score_squared=score, correct=score <= .04)
        except ValueError as error:
            row['error'] = str(error)
        rows.append(row)
    return rows


def transition(initial_correct, final_correct):
    return ('retained' if final_correct else 'lost') if initial_correct else (
        'recovered' if final_correct else 'unrecovered')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    study_bytes, manifest_bytes = args.study.read_bytes(), args.cases.read_bytes()
    study, manifest = json.loads(study_bytes), json.loads(manifest_bytes)
    if study.get('schema') != 'spatialrust.3dmatch-live-comparison.v1' or study.get('cases_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        parser.error('study and case manifest binding differs')
    cases = {case['case_id']: case for case in validate_cases(manifest)}
    if len(study['cases']) != len(cases) or {case['case_id'] for case in study['cases']} != set(cases):
        parser.error('study case IDs differ from manifest')
    rows, bindings = [], []
    for case in study['cases']:
        if case['status'] != 'completed':
            continue
        original = cases[case['case_id']]
        reference_path = Path(original['reference_file'])
        reference_bytes = reference_path.read_bytes()
        reference = json.loads(reference_bytes)
        validate_reference(reference)
        comparison_bytes = Path(case['comparison_file']).read_bytes()
        if hashlib.sha256(comparison_bytes).hexdigest() != case['comparison_sha256']:
            raise ValueError('saved comparison bytes changed')
        comparison = json.loads(comparison_bytes)
        validate_saved_receipt(comparison, reference, reference_bytes, study['seeds'], study['ransac_iterations'])
        truth, gt_hash = load_records(reference_path.parent / 'gt.log', project_rotation=True)
        information, info_hash = load_records(reference_path.parent / 'gt.info', 6)
        if gt_hash != reference['provenance']['gt_log_sha256'] or info_hash != reference['provenance']['gt_info_sha256']:
            raise ValueError('reference metadata bytes changed')
        pair = tuple(original['pair'])
        np.testing.assert_array_equal(reference['transform_source_to_target'], truth[pair]['matrix'])
        bindings.append(dict(case_id=case['case_id'], reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                             gt_log_sha256=gt_hash, gt_info_sha256=info_hash,
                             comparison_sha256=case['comparison_sha256']))
        for row in comparison['rows']:
            if row['method'] != 'spatialrust' or row['status'] != 'success':
                continue
            scores = stage_scores(row['report'], truth[pair]['original_matrix'], information[pair]['matrix'])
            rows.append(dict(case_id=case['case_id'], seed=row['seed'], stages=scores,
                             transition=transition(scores[0]['correct'], scores[-1]['correct'])))
    counts = {name: sum(row['transition'] == name for row in rows)
              for name in ('retained', 'lost', 'recovered', 'unrecovered')}
    receipt = dict(schema='spatialrust.3dmatch-stage-diagnosis.v1',
                   study_sha256=hashlib.sha256(study_bytes).hexdigest(),
                   cases_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   evaluator_source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('evaluate_redwood_logs.py', 'redwood_reference.py',
                                    'prepare_numpy_reference.py', 'evaluate_pose_reference.py')},
                   registration_executed=False, planned_native_rows=study['summary']['spatialrust']['planned'],
                   diagnosed_rows=len(rows), transitions=counts, bindings=bindings, rows=rows)
    table = ''.join('<tr><td>' + html.escape(row['case_id']) + '</td><td>' + str(row['seed'])
                    + '</td><td>' + row['transition'] + '</td><td>'
                    + html.escape(json.dumps(row['stages'])) + '</td></tr>' for row in rows)
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Saved stage diagnosis</title>'
                '<h1>Initialization versus refinement</h1><p>Reference enters only post-fit evaluation. '
                'Correct means publisher information score squared ≤0.04. Stage transitions locate loss '
                'or recovery; they do not isolate the physical cause or select an output for deployment.</p><pre>'
                + html.escape(json.dumps(counts, indent=2)) + '</pre><table><tr><th>Case</th><th>Seed</th>'
                '<th>Transition</th><th>Stage scores</th></tr>' + table + '</table></html>')
    serialized = json.dumps(receipt, indent=2, allow_nan=False)
    args.output_dir.mkdir()
    (args.output_dir / 'diagnosis.json').write_text(serialized)
    (args.output_dir / 'report.html').write_text(rendered)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
