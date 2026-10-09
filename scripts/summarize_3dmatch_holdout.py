"""Summarize preregistered paired interventions, retaining planned failures."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from compare_3dmatch_cases import execute
from evaluate_3dmatch_refinement import evaluate_replay
from evaluate_redwood_logs import squared_information_error
from redwood_reference import load_records
import numpy as np

from validate_3dmatch_holdout_plan import verify_files


def paired_summary(pairs, planned):
    if len(pairs) > planned:
        raise ValueError('evaluated rows exceed planned denominator')
    return dict(planned=planned, evaluated=len(pairs), correct=sum(b for a, b in pairs),
                gained=sum(not a and b for a, b in pairs), lost=sum(a and not b for a, b in pairs),
                unavailable=planned-len(pairs))


def summarize_primary(plan, baseline, refinement, stages):
    expected_cases = [case['case_id'] for case in plan['cases']]
    expected = {(case, seed) for case in expected_cases for seed in plan['seeds']}
    if baseline.get('schema') != 'spatialrust.3dmatch-live-comparison.v1' or baseline['cases_manifest_sha256'] != plan['cases_manifest_sha256']:
        raise ValueError('baseline schema or manifest binding differs')
    if baseline['seeds'] != plan['seeds'] or baseline['ransac_iterations'] != plan['ransac_iterations']:
        raise ValueError('baseline sampler differs')
    if [case['case_id'] for case in baseline['cases']] != expected_cases:
        raise ValueError('baseline case membership or order differs')
    labels, original = {}, {}
    for case in baseline['cases']:
        if case['status'] == 'case_error':
            continue
        if case['status'] != 'completed' or case['native_sha256'] != plan['native_sha256'] or case['open3d_native_sha256'] != plan['open3d_native_sha256']:
            raise ValueError('baseline status or native differs')
        controls = {key: value for key, value in case['controls'].items() if key != 'thread_environment'}
        if controls != plan['comparison_controls']:
            raise ValueError('baseline controls differ from frozen plan')
        native = [row for row in case['rows'] if row['method'] == 'spatialrust']
        if len(native) != len(plan['seeds']) or {r['seed'] for r in native} != set(plan['seeds']):
            raise ValueError('native seed slots omitted or duplicated')
        for row in native:
            key = case['case_id'], row['seed']
            if row['status'] == 'success':
                labels[key] = row['publisher_correct']
                original[key] = row['report']
            elif row['status'] != 'error':
                raise ValueError('unknown native row status')
    if refinement.get('schema') != 'spatialrust.3dmatch-refinement-batch.v1' or [c['case_id'] for c in refinement['cases']] != expected_cases:
        raise ValueError('refinement schema or case membership differs')
    trim_pairs = []
    for case in refinement['cases']:
        if case['planned_seeds'] != len(plan['seeds']):
            raise ValueError('refinement planned seed count differs')
        if case['status'] == 'case_error':
            continue
        if case['status'] != 'completed':
            raise ValueError('unknown refinement status')
        rows = case['evaluation']['rows']
        slots = {(r['seed'], r['trim_fraction']): r for r in rows}
        wanted = {(seed, fraction) for seed in plan['seeds'] for fraction in plan['primary_rules']['trim_fractions']}
        if len(slots) != len(rows) or set(slots) != wanted:
            raise ValueError('refinement row slots differ')
        for seed in plan['seeds']:
            key = case['case_id'], seed
            if key not in labels or slots[seed, 1.]['correct'] != labels[key]:
                raise ValueError('paired trim control differs from baseline')
            trim_pairs.append((labels[key], slots[seed, .8]['correct']))
    if stages.get('schema') != 'spatialrust.stage-selection.v1' or stages['leaf'] != .05 or stages['gate'] != .05:
        raise ValueError('stage selection schema or geometry differs')
    if plan['primary_rules']['motion_limit_metres'] not in stages['motion_limits_metres']:
        raise ValueError('primary motion rule missing')
    seen, motion_pairs = set(), []
    for row in stages['rows']:
        key = row['case_id'], row['seed']
        if key not in expected or key in seen:
            raise ValueError('unknown or duplicated stage row')
        seen.add(key)
        if row['status'] == 'error':
            if key in labels:
                raise ValueError('missing successful baseline stage')
            continue
        if row['status'] != 'success' or key not in original:
            raise ValueError('stage row has no successful baseline')
        records = row['stages']
        if records[0]['pose'] != original[key]['initial_transform_source_to_target'] or records[-1]['pose'] != original[key]['transform_source_to_target']:
            raise ValueError('stage endpoints differ from baseline')
        if records[-1]['correct'] != labels[key]:
            raise ValueError('stage final correctness differs from baseline')
        limit = plan['primary_rules']['motion_limit_metres']
        selected = 0 if records[-1]['motion_from_initial']['source_centroid_displacement_metres'] > limit else len(records)-1
        if row['selected'][f'motion_{limit:g}m'] != selected:
            raise ValueError('motion selection differs from frozen rule')
        motion_pairs.append((labels[key], records[selected]['correct']))
    if not set(labels) <= seen:
        raise ValueError('successful baseline stage rows omitted')
    return dict(baseline=paired_summary([(v, v) for v in labels.values()], len(expected)),
                trim_0_8=paired_summary(trim_pairs, len(expected)),
                motion_0_2m=paired_summary(motion_pairs, len(expected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'cases', 'baseline', 'refinement', 'stages', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    data = {name: getattr(args, name).read_bytes() for name in ('plan', 'cases', 'baseline', 'refinement', 'stages')}
    plan, manifest, baseline, refinement, stages = (json.loads(data[name]) for name in data)
    verify_files(plan, manifest, data['cases'], Path(__file__).resolve().parents[1])
    summary = summarize_primary(plan, baseline, refinement, stages)
    case_map = {case['case_id']: case for case in manifest['cases']}
    # Re-evaluate saved baseline and replay poses; aggregate labels are not trusted alone.
    for case in baseline['cases']:
        if case['status'] == 'completed':
            checked = execute(case_map[case['case_id']], Path(case['comparison_file']).parent,
                              plan['seeds'], plan['ransac_iterations'], 1, reuse=True)
            if checked != case:
                raise ValueError('baseline aggregate differs from bound comparison')
    # Bind all stage inputs and each replay's checkpoint artifacts as well as aggregates.
    for path, expected in stages['input_and_helper_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('stage input/helper changed')
    for case in refinement['cases']:
        if case['status'] != 'completed':
            continue
        directory = args.refinement.parent / case['case_id']
        for name, expected in case['artifact_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError('refinement checkpoint artifact changed')
        if json.loads((directory/'evaluation/evaluation.json').read_bytes()) != case['evaluation']:
            raise ValueError('refinement aggregate differs from checkpoint')
        raw = json.loads((directory/'replay/study.json').read_bytes())
        original = case_map[case['case_id']]
        baseline_case = next(c for c in baseline['cases'] if c['case_id'] == case['case_id'])
        comparison = json.loads(Path(baseline_case['comparison_file']).read_bytes())
        reference = Path(original['reference_file'])
        poses, _ = load_records(reference.parent/'gt.log', project_rotation=True)
        info, _ = load_records(reference.parent/'gt.info', 6)
        pair = tuple(original['pair'])
        checked = evaluate_replay(raw, comparison, poses[pair]['original_matrix'], info[pair]['matrix'])
        if any(case['evaluation'][key] != value for key, value in checked.items()):
            raise ValueError('replay accuracy differs from independent recomputation')
    for row in stages['rows']:
        if row['status'] != 'success':
            continue
        original = case_map[row['case_id']]
        reference = Path(original['reference_file'])
        poses, _ = load_records(reference.parent/'gt.log', project_rotation=True)
        info, _ = load_records(reference.parent/'gt.info', 6)
        pair = tuple(original['pair'])
        for stage in row['stages']:
            try:
                score = squared_information_error(poses[pair]['original_matrix'], np.asarray(stage['pose']), info[pair]['matrix'])
            except ValueError:
                score = None
            if stage['score_squared'] != score or stage['correct'] is not (score is not None and score <= .04):
                raise ValueError('stage accuracy differs from independent recomputation')
    output = dict(schema='spatialrust.3dmatch-holdout-result.v1', summary=summary,
                  artifact_sha256={name: hashlib.sha256(value).hexdigest() for name, value in data.items()},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  registration_executed=False, primary_rules=plan['primary_rules'],
                  limits='Held-out fitting scenes within one benchmark; selected positive pairs and repeated seeds. No default change.')
    bars = ''.join(f'<text x="5" y="{30+i*45}">{html.escape(name)}</text><rect x="140" y="{15+i*45}" '
        f'width="{240*v["correct"]/v["planned"]}" height="24" fill="#2563eb"/>'
        f'<text x="390" y="{30+i*45}">{v["correct"]}/{v["planned"]}</text>' for i, (name,v) in enumerate(summary.items()))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 160" role="img" aria-label="Frozen held-out primary rules">'+bars+'</svg>'
    args.output_dir.mkdir()
    (args.output_dir/'result.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    (args.output_dir/'comparison.svg').write_text(svg, encoding='utf-8')
    (args.output_dir/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen scene holdout</title>'
        '<h1>Frozen scene holdout</h1><p>'+html.escape(output['limits'])+'</p>'+svg+'<pre>'
        +html.escape(json.dumps(summary, indent=2))+'</pre>', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
