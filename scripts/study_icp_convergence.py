"""Controlled failure study: stopping thresholds versus synthetic pose recovery.

No algorithm selection uses truth. This executes SpatialRust only, comparing
stopping policies on identical pairs/priors; it is not an external-library ranking.
"""
import argparse
import hashlib
import html
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[1]
CASES = ('clean_volume', 'noise_volume', 'replace_outliers', 'partial_source', 'symmetric_ring')
POLICIES = ('defaults', 'scaled_transform', 'budget_only')


def z_rotation(degrees):
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])


def fixture(case, seed, scale):
    rng = np.random.default_rng(seed)
    target = rng.uniform(-1, 1, (400, 3))
    clean = target.copy()
    if case == 'noise_volume':
        clean += np.random.default_rng(seed+100).normal(0, .005, clean.shape)
    elif case == 'replace_outliers':
        clean += np.random.default_rng(seed+100).normal(0, .005, clean.shape)
        clean[-80:] = np.random.default_rng(seed+200).uniform(-1.5, 1.5, (80, 3))
    elif case == 'partial_source':
        clean = clean[:240]
    elif case == 'symmetric_ring':
        angles = np.arange(180)*2*np.pi/180
        target = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(180)))
        clean = target.copy()
    elif case != 'clean_volume':
        raise ValueError('unknown case')
    # Generate source from the known target-frame geometry. Truth is used only
    # to generate controlled priors and assess outputs, never to choose a policy.
    rotation = z_rotation(20)
    translation = np.array([.4, -.2, .1])*scale
    target = (target*scale).astype(np.float32)
    source = (clean*scale @ rotation.T + translation).astype(np.float32)
    truth = np.eye(4)
    truth[:3, :3] = rotation.T
    truth[:3, 3] = -rotation.T @ translation
    return source, target, truth


def pose_errors(pose, truth, scale):
    u, _, vt = np.linalg.svd(pose[:3, :3])
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    cosine = np.clip((np.trace(rotation @ truth[:3, :3].T)-1)/2, -1, 1)
    return float(np.degrees(np.arccos(cosine))), float(np.linalg.norm(pose[:3, 3]-truth[:3, 3])/scale)


def run_case(case, seed, scale, initial_error, policy, iterations):
    source_xyz, target_xyz, truth = fixture(case, seed, scale)
    perturb = np.eye(4)
    perturb[:3, :3] = z_rotation(initial_error)
    perturb[:3, 3] = np.array([.03, -.02, .01])*scale
    prior = (perturb @ truth).astype(np.float32)
    source, target = sr.PointCloud.from_xyz(source_xyz), sr.PointCloud.from_xyz(target_xyz)
    seeded = sr.apply_transform(source, prior)
    thresholds = (dict(translation_epsilon=1e-8, rotation_epsilon=1e-8, fitness_epsilon=1e-6)
                  if policy == 'defaults' else
                  dict(translation_epsilon=1e-4*scale, rotation_epsilon=1e-4, fitness_epsilon=0)
                  if policy == 'scaled_transform' else
                  dict(translation_epsilon=0, rotation_epsilon=0, fitness_epsilon=0))
    row = dict(case=case, seed=seed, coordinate_scale=scale, initial_error_degrees=initial_error,
               policy=policy, thresholds=thresholds, source_points=len(source), target_points=len(target),
               correspondence_distance=.3*scale, evaluation_distance=.05*scale)
    started = time.perf_counter()
    try:
        trace = sr.register_icp_diagnostics(seeded, target, .3*scale, iterations, **thresholds)
    except ValueError as error:
        row.update(status='error', error=str(error), seconds=time.perf_counter()-started,
                   recovered_generating_pose=False, false_convergence=False)
        return row
    seconds = time.perf_counter()-started
    result = trace.result
    pose = result.transform().astype(np.float64) @ prior.astype(np.float64)
    rotation_error, translation_error = pose_errors(pose, truth, scale)
    recovered = rotation_error < 1 and translation_error < .01
    aligned = sr.apply_transform(source, pose.astype(np.float32))
    count, fraction, rmse = sr.distance_gated_support(aligned, target, .05*scale)
    _, reverse, _ = sr.distance_gated_support(target, aligned, .05*scale)
    history = [dict(iteration=h.iteration, correspondences=h.correspondences,
                    rematched=h.evaluated_correspondences, fitness=h.fitness,
                    fitness_change=h.fitness_change, translation_delta=h.translation_delta,
                    rotation_delta_radians=h.rotation_delta_radians) for h in trace.history]
    row.update(status='success', error=None, seconds=seconds, iterations=result.iterations,
               converged=result.converged, stop_reason=trace.stop_reason,
               rotation_error_degrees=rotation_error, translation_error_in_base_units=translation_error,
               recovered_generating_pose=recovered, false_convergence=result.converged and not recovered,
               supported_points=count, forward_support=fraction, reverse_support=reverse,
               normalized_gated_rmse=None if rmse is None else rmse/scale,
               transform_source_to_target=pose.tolist(), history=history)
    return row


def render(results):
    overview = []
    for scale, policy in sorted(set((r['coordinate_scale'], r['policy']) for r in results)):
        group = [r for r in results if (r['coordinate_scale'], r['policy']) == (scale, policy)]
        good = sum(r['recovered_generating_pose'] for r in group)
        false = sum(r['false_convergence'] for r in group)
        other = len(group)-good-false
        rectangles = []
        position = 0
        for count, color in zip((good, false, other), ('#15803d', '#c2410c', '#64748b')):
            width = 100*count/len(group)
            rectangles.append(f'<rect x="{position:.6g}" width="{width:.6g}" height="8" fill="{color}"/>')
            position += width
        label = f'{good} recovered, {false} stopped without generating pose, {other} remaining'
        overview.append(f'<tr><td>{scale:g}</td><td>{html.escape(policy)}</td><td>'
                        f'<svg viewBox="0 0 100 8" role="img" aria-label="{label}" style="width:240px;height:20px">'
                        + ''.join(rectangles) + f'</svg></td><td>{good}/{len(group)}</td><td>{false}/{len(group)}</td>'
                        f'<td>{other}/{len(group)}</td></tr>')
    rows = []
    for case, scale, angle, policy in sorted(set((r['case'], r['coordinate_scale'], r['initial_error_degrees'], r['policy']) for r in results)):
        group = [r for r in results if (r['case'], r['coordinate_scale'], r['initial_error_degrees'], r['policy']) == (case, scale, angle, policy)]
        good = sum(r['recovered_generating_pose'] for r in group)
        false = sum(r['false_convergence'] for r in group)
        completed = [r for r in group if r['status'] == 'success']
        iteration_label = ', '.join(str(r['iterations']) for r in completed)
        reason_label = ', '.join(sorted(set(r['stop_reason'] for r in completed)))
        support = ', '.join(f'{r["forward_support"]:.1%}' for r in completed)
        rows.append(f'<tr><td>{html.escape(case)}</td><td>{scale:g}</td><td>{angle:g}</td><td>{html.escape(policy)}</td>'
                    f'<td><meter min="0" max="{len(group)}" value="{good}"></meter> {good}/{len(group)}</td>'
                    f'<td>{false}/{len(group)}</td><td>{iteration_label}</td><td>{reason_label}</td><td>{support}</td></tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>ICP stopping and failure study</title><style>body{font:15px system-ui;padding:1rem}td,th{padding:.4rem;text-align:left}table{border-collapse:collapse}tr:nth-child(even){background:#f1f5f9}</style>'
            '<h1>ICP stopping and failure study</h1><p>Matched conditions, different stopping policies. '
            'Recovered means rotation error below 1 degree and translation error below 0.01 base units. '
            'Truth is used for controlled fixture generation and output assessment, never policy selection.</p>'
            '<p>Coordinate scale rescales the same geometry and noise. Defaults keep absolute thresholds unchanged; '
            'scaled_transform scales translation tolerance, keeps angular tolerance fixed, and disables absolute fitness. '
            'budget_only disables both stopping tests. False convergence means a stopping threshold was met '
            'but the generating pose was not recovered. A symmetric ring has multiple equivalent poses: this counts '
            'disagreement with its generating pose, not observable physical error. Ring rows use one fixture.</p>'
            '<h2>Overview of this controlled matrix</h2><p>Green: generating pose recovered; orange: '
            'a stopping criterion was met without recovering it; gray: remaining runs. These counts depend '
            'on this chosen mixture of environments and prior errors. Disabling stopping makes the orange '
            'count zero by definition, without making incorrect poses correct.</p>'
            '<table><tr><th>Scale</th><th>Policy</th><th>Outcome fractions</th><th>Recovered</th>'
            '<th>Stopped without recovery</th><th>Remaining</th></tr>' + ''.join(overview) + '</table>'
            '<h2>Environment and prior breakdown</h2>'
            '<table><tr><th>Environment</th><th>Scale</th><th>Prior error (degrees)</th><th>Policy</th>'
            '<th>Recovered</th><th>False convergence</th><th>Iterations</th><th>Stop reasons</th><th>Forward support</th></tr>'
            + ''.join(rows) + '</table><p>Synthetic fixed-seed study of SpatialRust point-to-point ICP. '
            'No external library execution, no universal ranking, and no measured real-world accuracy. '
            'Times are individual observations, not controlled comparative benchmarks. Detailed per-update data are in study.json.</p></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--scales', type=float, nargs='+', default=[.001, 1, 1000])
    parser.add_argument('--angles', type=float, nargs='+', default=[0, 20, 40])
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20 or not 1 <= args.iterations <= 1000:
        parser.error('seeds must be 1..20 and iterations 1..1000')
    if (not all(math.isfinite(v) and 1e-6 <= v <= 1e6 for v in args.scales)
            or not all(math.isfinite(v) and 0 <= v <= 180 for v in args.angles)):
        parser.error('scales must be 1e-6..1e6 and angles 0..180, all finite')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for case in CASES:
        seeds = range(1 if case == 'symmetric_ring' else args.seeds)
        for seed, scale, angle, policy in itertools.product(seeds, args.scales, args.angles, POLICIES):
            results.append(run_case(case, seed, scale, angle, policy, args.iterations))
        print(f'{case}: {len(results)} total cases completed', file=sys.stderr, flush=True)
    inputs = [Path(__file__), ROOT/'crates/spatialrust-registration/src/icp.rs', ROOT/'crates/spatialrust-registration/src/kabsch.rs']
    native = Path(importlib.import_module('spatialrust.spatialrust').__file__)
    receipt = dict(schema='spatialrust.icp-convergence-study.v1', seeds=args.seeds,
                   scales=args.scales, initial_errors=args.angles, max_iterations=args.iterations,
                   policies=list(POLICIES), case_count=len(results), results=results,
                   native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                   numpy_version=np.__version__, limits='Synthetic SpatialRust-only controlled prior study; ring one fixture; no universal ranking or accuracy claim.')
    (args.output_dir/'study.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output_dir/'report.html').write_text(render(results), encoding='utf-8')
    print(json.dumps(dict(cases=len(results), recovered=sum(r['recovered_generating_pose'] for r in results),
                         false_convergence=sum(r['false_convergence'] for r in results)), indent=2))


if __name__ == '__main__':
    main()
