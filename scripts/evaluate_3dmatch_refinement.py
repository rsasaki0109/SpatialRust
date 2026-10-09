"""Score fixed-initialization trim replays after fitting; verify paired controls."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import platform

import numpy as np

from evaluate_pose_reference import validate_reference
from evaluate_redwood_logs import squared_information_error
from redwood_reference import load_records


def evaluate_replay(study, comparison, truth, information):
    if study.get('schema') != 'spatialrust.public-refinement-study.v1' or comparison.get('schema') != 'spatialrust.public-global-comparison.v1':
        raise ValueError('unsupported replay or comparison schema')
    if study['native_sha256'] != comparison['native_sha256'] or study['input_file_sha256'] != comparison['input_file_sha256']:
        raise ValueError('replay native or input bindings differ')
    native = [row for row in comparison['rows'] if row['method'] == 'spatialrust']
    if len(native) != len(comparison['seeds']) or {row['seed'] for row in native} != set(comparison['seeds']) or any(row['status'] != 'success' or row['report'].get('initial_transform_supplied') is not False for row in native):
        raise ValueError('paired replay evaluation requires all native baseline outputs')
    baseline = {row['seed']: row['report'] for row in native}
    fractions = {row['trim_fraction'] for row in study['rows']}
    if 1. not in fractions or any(not np.isfinite(value) or not 0 < value <= 1 for value in fractions):
        raise ValueError('require valid trim fractions including the untrimmed control')
    slots = [(row['seed'], row['trim_fraction']) for row in study['rows']]
    if len(slots) != len(set(slots)) or set(slots) != {(seed, fraction) for seed in baseline for fraction in fractions}:
        raise ValueError('replay omitted or duplicated paired rows')
    rows = []
    for row in study['rows']:
        report, original = row['report'], baseline[row['seed']]
        if report.get('input_file_sha256') != study['input_file_sha256'] or report.get('initial_transform_supplied') is not True:
            raise ValueError('replay row bindings or initialization provenance differ')
        np.testing.assert_array_equal(report['initial_transform_source_to_target'], original['initial_transform_source_to_target'])
        if len(report['schedule']) != len(original['schedule']):
            raise ValueError('refinement schedule count differs')
        for stage, control in zip(report['schedule'], original['schedule']):
            for key in ('leaf', 'max_distance', 'iterations', 'convergence'):
                if stage.get(key) != control.get(key):
                    raise ValueError('refinement schedule differs beyond trimming')
        if len(report['stages']) != len(original['stages']):
            raise ValueError('refinement stage count differs')
        for stage, control in zip(report['stages'], original['stages']):
            for key in ('leaf_metres', 'max_correspondence_distance_metres', 'convergence_criteria'):
                if stage[key] != control[key]:
                    raise ValueError('refinement controls differ beyond trimming')
            if stage['trim_fraction'] != row['trim_fraction']:
                raise ValueError('reported trim differs from actual stage configuration')
        if row['trim_fraction'] == 1.:
            np.testing.assert_array_equal(report['transform_source_to_target'], original['transform_source_to_target'])
        try:
            score = squared_information_error(truth, np.asarray(report['transform_source_to_target']), information)
            result = dict(score_squared=score, correct=score <= .04)
        except ValueError as error:
            result = dict(score_squared=None, correct=False, error=str(error))
        rows.append(dict(seed=row['seed'], trim_fraction=row['trim_fraction'], **result))
    summary = {str(value): dict(planned=len(baseline), correct=sum(row['correct'] for row in rows if row['trim_fraction'] == value))
               for value in sorted(fractions, reverse=True)}
    return dict(summary=summary, rows=rows, exact_untrimmed_controls=len(baseline),
                truth_selected_trim=False, deployment_policy_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'comparison', 'reference', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    data = {name: getattr(args, name).read_bytes() for name in ('study', 'comparison', 'reference')}
    study, comparison, reference = (json.loads(data[name]) for name in ('study', 'comparison', 'reference'))
    validate_reference(reference)
    if reference['length_unit'] != 'm':
        raise ValueError('official-fragment replay requires metre units')
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    if study['comparison_sha256'] != digests['comparison'] or study['reference_sha256'] != digests['reference'] or comparison['reference_sha256'] != digests['reference']:
        raise ValueError('study/comparison/reference byte bindings differ')
    if reference['input_file_sha256'] != comparison['input_file_sha256']:
        raise ValueError('reference input binding differs')
    truth, gt_hash = load_records(args.reference.parent / 'gt.log', project_rotation=True)
    information, info_hash = load_records(args.reference.parent / 'gt.info', 6)
    provenance = reference['provenance']
    if gt_hash != provenance['gt_log_sha256'] or info_hash != provenance['gt_info_sha256']:
        raise ValueError('reference GT/information bytes changed')
    pair = tuple(sorted([provenance['source_fragment_id'], provenance['target_fragment_id']]))
    if provenance['source_fragment_id'] != pair[1]:
        raise ValueError('require original publisher j-to-i convention')
    np.testing.assert_array_equal(reference['transform_source_to_target'], truth[pair]['matrix'])
    result = evaluate_replay(study, comparison, truth[pair]['original_matrix'], information[pair]['matrix'])
    result.update(schema='spatialrust.3dmatch-refinement-evaluation.v1', input_artifact_sha256=digests,
                  gt_log_sha256=gt_hash, gt_info_sha256=info_hash, registration_executed=False,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions=dict(numpy=np.__version__, python=platform.python_version()),
                  evaluator_source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('evaluate_redwood_logs.py', 'redwood_reference.py',
                                   'prepare_numpy_reference.py', 'evaluate_pose_reference.py')})
    serialized = json.dumps(result, indent=2, allow_nan=False)
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired refinement evaluation</title>'
                '<h1>Fixed-initialization trimming</h1><p>Reference is evaluated after fitting. '
                'Exact untrimmed controls and unchanged stage gates are required. This selected-pair '
                'experiment does not establish a default trim or benchmark recall/precision.</p><pre>'
                + html.escape(serialized) + '</pre></html>')
    args.output_dir.mkdir()
    (args.output_dir / 'evaluation.json').write_text(serialized)
    (args.output_dir / 'report.html').write_text(rendered)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
