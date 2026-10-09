"""Reference-free selection of saved global poses, followed by publisher scoring.

Exploratory analysis on previously inspected positive pairs; no new fitting.
"""
import argparse
import hashlib
import html
import importlib
import json
import math
from pathlib import Path

import numpy as np
import spatialrust as sr

from compare_3dmatch_cases import execute, validate_cases
from evaluate_pose_reference import rigid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_key(metrics, rule):
    """Only geometry metrics enter ranking; exact ties keep candidate order."""
    forward, reverse = metrics['forward'], metrics['reverse']
    for metric in (forward, reverse):
        fraction, rmse = metric['fraction'], metric['rmse']
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('invalid support fraction')
        if rmse is not None and (isinstance(rmse, bool) or not isinstance(rmse, (int, float)) or not math.isfinite(rmse) or rmse < 0):
            raise ValueError('invalid support RMSE')
        if (fraction == 0) != (rmse is None):
            raise ValueError('zero support must have undefined RMSE')
    if rule == 'forward':
        return forward['fraction'], -(math.inf if forward['rmse'] is None else forward['rmse'])
    if rule != 'balanced':
        raise ValueError('unknown selection rule')
    a, b = forward['fraction'], reverse['fraction']
    harmonic = 0 if a + b == 0 else 2 * a * b / (a + b)
    rmse = math.inf if forward['rmse'] is None or reverse['rmse'] is None else max(forward['rmse'], reverse['rmse'])
    return harmonic, -rmse


def select(candidates, rule, methods):
    eligible = [i for i, row in enumerate(candidates) if row['status'] == 'success' and row['method'] in methods]
    return max(eligible, key=lambda i: rank_key(candidates[i]['metrics'], rule)) if eligible else None


def support(index, query, gate):
    count, fraction, rmse = index.support(query, gate)
    return dict(count=count, fraction=fraction, rmse=rmse if count else None)


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
    helpers = [Path(__file__), Path(__file__).with_name('compare_3dmatch_cases.py'),
               Path(__file__).with_name('evaluate_pose_reference.py'),
               Path(__file__).with_name('evaluate_redwood_logs.py'), Path(__file__).with_name('redwood_reference.py')]
    hashes = {str(p.resolve()): digest(p) for p in helpers + [native, args.cases]}
    # Fixed sampling and ranking rules; no reference pose or accuracy label enters selection.
    leaf, gate = .05, .05
    results = []
    for case in cases:
        path = args.comparisons / case['case_id'] / 'comparison.json'
        hashes[str(path.resolve())] = digest(path)
        receipt = json.loads(path.read_bytes())
        if receipt.get('schema') != 'spatialrust.public-global-comparison.v1' or receipt['native_sha256'] != hashes[str(native.resolve())]:
            raise ValueError('unsupported comparison or changed native build')
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
            cloud = sr.voxel_downsample(sr.read(str(p)), leaf, 'cpu')
            if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
                raise ValueError('require finite nonempty coarse geometry')
            clouds.append(cloud)
        source, target = clouds
        source_index, target_index = (sr.DistanceSupportIndex(c) for c in clouds)
        candidates = []
        for row in receipt['rows']:
            item = dict(method=row['method'], seed=row['seed'], status=row['status'])
            if row['status'] == 'success':
                if row['report']['input_file_sha256'] != receipt['input_file_sha256']:
                    raise ValueError('candidate input binding differs')
                pose = rigid(row['report']['transform_source_to_target'])
                item['metrics'] = dict(
                    forward=support(target_index, sr.apply_transform(source, pose.astype(np.float32)), gate),
                    reverse=support(source_index, sr.apply_transform(target, np.linalg.inv(pose).astype(np.float32)), gate))
            candidates.append(item)
        selections = {f'{pool}_{rule}': select(candidates, rule, methods)
                      for pool, methods in [('spatialrust', ('spatialrust',)), ('open3d', ('open3d',)),
                                            ('pooled', ('spatialrust', 'open3d'))]
                      for rule in ('forward', 'balanced')}
        results.append(dict(case_id=case['case_id'], scene=case['scene'], candidates=candidates,
                            selected_indices=selections, comparison_file=str(path.resolve()),
                            coarse_points=dict(source=len(source), target=len(target))))
    # All candidate selections are fixed before reference evaluation begins.
    for case, result in zip(cases, results):
        receipt = json.loads(Path(result['comparison_file']).read_bytes())
        evaluated = execute(case, Path(result['comparison_file']).parent,
                            receipt['seeds'], receipt['ransac_iterations'], 1, reuse=True)
        for candidate, row in zip(result['candidates'], evaluated['rows']):
            candidate.update(publisher_correct=row['publisher_correct'],
                             publisher_score_squared=row.get('publisher_score_squared'))
        result['selected_correct'] = {name: index is not None and result['candidates'][index]['publisher_correct']
                                      for name, index in result['selected_indices'].items()}
        result['oracle_available'] = {pool: any(c['publisher_correct'] for c in result['candidates'] if c['method'] in methods)
                                     for pool, methods in [('spatialrust', ('spatialrust',)), ('open3d', ('open3d',)),
                                                           ('pooled', ('spatialrust', 'open3d'))]}
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('inputs, native or helpers changed during study')
    summary = {name: dict(planned=len(cases), correct=sum(r['selected_correct'][name] for r in results),
                         unavailable=sum(r['selected_indices'][name] is None for r in results),
                         oracle_available=sum(r['oracle_available'][name.rsplit('_', 1)[0]] for r in results))
               for name in results[0]['selected_indices']}
    output = dict(schema='spatialrust.global-candidate-selection.v1', leaf=leaf, gate=gate,
                  summary=summary, cases=results, input_and_helper_sha256=hashes,
                  registration_executed=False, reference_used_for_selection=False,
                  limits='Exploratory previously examined positive pairs; oracle is post-fit diagnosis only; no speed ranking.')
    rows = ''.join('<tr><td>' + html.escape(r['case_id']) + '</td><td>' + html.escape(name)
                   + '</td><td>' + str(index) + '</td><td>' + str(r['selected_correct'][name]) + '</td></tr>'
                   for r in results for name, index in r['selected_indices'].items())
    bars = ''.join(f'<text x="5" y="{30 + i * 40}" font-size="13">{html.escape(name)}</text>'
                   f'<rect x="180" y="{15 + i * 40}" width="{240 * values["correct"] / values["planned"]}" height="22" fill="#2563eb"/>'
                   f'<text x="430" y="{30 + i * 40}" font-size="13">{values["correct"]}/{values["planned"]}</text>'
                   for i, (name, values) in enumerate(summary.items()))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 490 260" role="img" aria-label="Publisher correct selected poses out of twelve pairs">' + bars + '</svg>'
    args.output_dir.mkdir()
    (args.output_dir / 'selection.svg').write_text(svg, encoding='utf-8')
    (args.output_dir / 'study.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    (args.output_dir / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Global candidate selection</title>'
        '<h1>Reference-free candidate selection</h1><p>' + html.escape(output['limits'])
        + '</p><p>Forward support versus harmonic forward/reverse support, then gated RMSE; '
        'ties keep saved candidate order. All selections precede reference evaluation.</p><pre>'
        + html.escape(json.dumps(summary, indent=2)) + '</pre>' + svg + '<table><tr><th>Pair</th><th>Rule</th>'
        '<th>Candidate index</th><th>Publisher correct</th></tr>' + rows + '</table>', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
