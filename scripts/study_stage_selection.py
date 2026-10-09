"""Test reference-free protection of saved initial poses against refinement drift.

Exploratory fixed geometry rules on previously examined cases; no registration.
"""
import argparse
import html
import importlib
import json
import math
from pathlib import Path

import numpy as np
import spatialrust as sr

from compare_3dmatch_cases import execute, validate_cases
from evaluate_pose_reference import rigid
from evaluate_redwood_logs import squared_information_error
from redwood_reference import load_records
from study_global_candidate_selection import digest, rank_key, support


def saved_stages(report):
    stages = [('initial', report['initial_transform_source_to_target'])] + [
        (stage['name'], stage['transform_source_to_target']) for stage in report['stages']]
    if len(stages) < 2 or len({name for name, _ in stages}) != len(stages):
        raise ValueError('require distinct initial and refinement stage names')
    if not np.array_equal(stages[-1][1], report['transform_source_to_target']):
        raise ValueError('final stage differs from saved output')
    return [(name, rigid(pose)) for name, pose in stages]


def pose_motion(initial, final, centroid):
    initial, final = rigid(initial), rigid(final)
    centroid = np.asarray(centroid, dtype=np.float64)
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError('require finite source centroid')
    relative = initial[:3, :3].T @ final[:3, :3]
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    angle = math.degrees(math.atan2(float(np.linalg.norm(skew)) / 2,
                                  float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    with np.errstate(over='ignore', invalid='ignore'):
        displacement = (final[:3, :3] - initial[:3, :3]) @ centroid + final[:3, 3] - initial[:3, 3]
    length = math.hypot(*map(float, displacement))
    if not math.isfinite(length):
        raise ValueError('centroid displacement overflows')
    return dict(rotation_degrees=angle, source_centroid_displacement_metres=length)


def motion_selection(stages, limit):
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError('motion limit must be finite and positive')
    if len(stages) < 2:
        raise ValueError('require initial and final stages')
    displacement = stages[-1]['motion_from_initial']['source_centroid_displacement_metres']
    if not math.isfinite(displacement) or displacement < 0:
        raise ValueError('invalid centroid displacement')
    return 0 if displacement > limit else len(stages)-1


def summarize(rows, planned):
    summary = {}
    evaluated = [row for row in rows if row['status'] == 'success']
    rules = list(evaluated[0]['selected']) if evaluated else ['initial', 'final', 'forward', 'balanced']
    if len(evaluated) > planned or any(set(row['selected']) != set(rules) for row in evaluated):
        raise ValueError('inconsistent planned rows or selection rules')
    for rule in rules:
        pairs = [(row['stages'][-1]['correct'], row['stages'][row['selected'][rule]]['correct']) for row in evaluated]
        summary[rule] = dict(planned=planned, evaluated=len(pairs), correct=sum(b for a, b in pairs),
                             gained_vs_final=sum(not a and b for a, b in pairs),
                             lost_vs_final=sum(a and not b for a, b in pairs), unavailable=planned-len(pairs))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    cases = validate_cases(json.loads(args.cases.read_bytes()))
    native = Path(importlib.import_module('spatialrust.spatialrust').__file__)
    helpers = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        'study_global_candidate_selection.py', 'compare_3dmatch_cases.py', 'evaluate_pose_reference.py',
        'evaluate_redwood_logs.py', 'redwood_reference.py', 'prepare_numpy_reference.py')]
    hashes = {str(path.resolve()): digest(path) for path in helpers + [native, args.cases]}
    rows, planned = [], 0
    leaf, gate = .05, .05
    # Fixed multiples of the fine/coarse correspondence gates, not fitted thresholds.
    motion_limits = [.05, .1, .2, .4]
    for case in cases:
        path = args.comparisons / case['case_id'] / 'comparison.json'
        hashes[str(path.resolve())] = digest(path)
        receipt = json.loads(path.read_bytes())
        if receipt.get('schema') != 'spatialrust.public-global-comparison.v1' or receipt['native_sha256'] != hashes[str(native.resolve())]:
            raise ValueError('unsupported comparison or changed native')
        seeds = receipt['seeds']
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 16 or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError('require distinct bounded integer seeds')
        reference = Path(case['reference_file'])
        for p in (reference, reference.parent / 'gt.log', reference.parent / 'gt.info'):
            hashes[str(p.resolve())] = digest(p)
        if hashes[str(reference.resolve())] != receipt['reference_sha256']:
            raise ValueError('reference changed since fitting')
        clouds = []
        for role in ('source', 'target'):
            p = Path(case[role])
            hashes[str(p.resolve())] = digest(p)
            if hashes[str(p.resolve())] != receipt['input_file_sha256'][role]:
                raise ValueError('changed input cloud')
            cloud = sr.read(str(p))
            if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
                raise ValueError('require finite geometry')
            if role == 'source':
                centroid = cloud.xyz().mean(axis=0, dtype=np.float64)
            clouds.append(sr.voxel_downsample(cloud, leaf, 'cpu'))
        source, target = clouds
        source_index, target_index = (sr.DistanceSupportIndex(c) for c in clouds)
        planned += len(receipt['seeds'])
        for row in receipt['rows']:
            if row['method'] != 'spatialrust':
                continue
            item = dict(case_id=case['case_id'], seed=row['seed'], status=row['status'])
            if row['status'] == 'success':
                report = row['report']
                if report['input_file_sha256'] != receipt['input_file_sha256']:
                    raise ValueError('report input binding differs')
                stages = saved_stages(report)
                item['stages'] = [dict(name=name, pose=pose.tolist(), metrics=dict(
                    forward=support(target_index, sr.apply_transform(source, pose.astype(np.float32)), gate),
                    reverse=support(source_index, sr.apply_transform(target, np.linalg.inv(pose).astype(np.float32)), gate)),
                    motion_from_initial=pose_motion(stages[0][1], pose, centroid)) for name, pose in stages]
                item['selected'] = dict(initial=0, final=len(stages)-1)
                item['selected'].update({rule: max(range(len(stages)), key=lambda i: rank_key(item['stages'][i]['metrics'], rule))
                                         for rule in ('forward', 'balanced')})
                item['selected'].update({f'motion_{limit:g}m': motion_selection(item['stages'], limit)
                                         for limit in motion_limits})
            rows.append(item)
    # All geometry selections are fixed before any ground-truth parsing/scoring.
    for case in cases:
        path = args.comparisons / case['case_id'] / 'comparison.json'
        receipt = json.loads(path.read_bytes())
        execute(case, path.parent, receipt['seeds'], receipt['ransac_iterations'], 1, reuse=True)
        reference = Path(case['reference_file'])
        poses, _ = load_records(reference.parent / 'gt.log', project_rotation=True)
        information, _ = load_records(reference.parent / 'gt.info', 6)
        pair = tuple(case['pair'])
        for row in rows:
            if row['case_id'] != case['case_id'] or row['status'] != 'success':
                continue
            for stage in row['stages']:
                stage.update(correct=False, score_squared=None)
                try:
                    score = squared_information_error(poses[pair]['original_matrix'], np.array(stage['pose']), information[pair]['matrix'])
                    stage.update(correct=score <= .04, score_squared=score)
                except ValueError as error:
                    stage['score_error'] = str(error)
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('inputs, native or helpers changed during study')
    summary = summarize(rows, planned)
    output = dict(schema='spatialrust.stage-selection.v1', leaf=leaf, gate=gate, motion_limits_metres=motion_limits,
                  summary=summary, rows=rows,
                  input_and_helper_sha256=hashes, registration_executed=False, reference_used_for_selection=False,
                  limits='Exploratory previously examined cases; proximity and pose motion do not certify correctness.')
    table = ''.join('<tr><td>' + html.escape(r['case_id']) + '</td><td>' + str(r['seed']) + '</td><td>'
                    + html.escape(json.dumps(r.get('selected'))) + '</td><td>'
                    + html.escape(json.dumps(r.get('stages'))) + '</td></tr>' for r in rows)
    bars = ''.join(f'<text x="5" y="{30+i*40}">{name}</text><rect x="100" y="{15+i*40}" '
                   f'width="{280*v["correct"]/planned}" height="22" fill="#2563eb"/>'
                   f'<text x="390" y="{30+i*40}">{v["correct"]}/{planned}</text>' for i, (name, v) in enumerate(summary.items()))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 {len(summary)*40+20}" role="img" aria-label="Correct stage selections">'+bars+'</svg>'
    args.output_dir.mkdir()
    (args.output_dir / 'study.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    (args.output_dir / 'selection.svg').write_text(svg, encoding='utf-8')
    (args.output_dir / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Stage selection</title>'
        '<h1>Can proximity protect an initializer?</h1><p>'+html.escape(output['limits'])
        +'</p><p>Motion rules keep the initial pose when final source-centroid displacement exceeds '
        'the declared limit; otherwise keep the final pose. Limits are fixed correspondence-gate multiples. '
        'This can suppress both harmful drift and useful correction; no default change is established.</p>'+svg+'<pre>'
        +html.escape(json.dumps(summary, indent=2))+'</pre><table><tr><th>Pair</th><th>Seed</th><th>Selections</th>'
        '<th>Exact stage metrics, motion and post-fit accuracy</th></tr>'+table+'</table>', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
