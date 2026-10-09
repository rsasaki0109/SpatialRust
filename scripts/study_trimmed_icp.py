"""Paired ICP trimming study: outlier rejection versus loss of useful pairs.

Runs SpatialRust only. Ground truth evaluates outputs and labels synthetic
membership; it never participates in nearest-pair ranking or pose selection.
"""
import argparse
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'crates/spatialrust-py/examples'))
from align_point_clouds import rigid_matrix
from study_icp_convergence import pose_errors, z_rotation

CASES = ('clean', 'noise', 'near_outliers', 'random_replacements', 'partial_source', 'symmetric_ring')


def fixture(case, seed):
    target = np.random.default_rng(seed).uniform(-1, 1, (400, 3))
    clean = target.copy()
    labels = np.ones(400, dtype=bool)
    if case in ('noise', 'near_outliers', 'random_replacements', 'partial_source'):
        clean += np.random.default_rng(10000+seed).normal(0, .005, clean.shape)
    if case in ('near_outliers', 'random_replacements'):
        rng = np.random.default_rng(20000+seed)
        clean[-80:] = rng.uniform(-1.5, 1.5, (80, 3))
        if case == 'near_outliers':
            clean[-80:, 0] = rng.uniform(1.2, 2, 80)
        labels[-80:] = False
    elif case == 'partial_source':
        clean, labels = clean[:240], labels[:240]
    elif case == 'symmetric_ring':
        angles = np.arange(180)*2*np.pi/180
        target = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(180)))
        clean, labels = target.copy(), np.ones(180, dtype=bool)
    rotation, translation = z_rotation(60), np.array([5., -2., .5])
    source = (clean @ rotation.T + translation).astype(np.float32)
    truth = np.eye(4)
    truth[:3, :3], truth[:3, 3] = rotation.T, -rotation.T @ translation
    return source, target.astype(np.float32), truth, labels


def evaluate(source, target, seeded, prior, truth, labels, first_distances, gate, fraction, iterations):
    # Assess which synthetic members the distance-only rule initially retains.
    # Labels do not enter the native estimator or this ranking.
    native_gate = np.float32(gate)
    eligible = np.flatnonzero(first_distances <= native_gate*native_gate)
    count = math.floor(len(eligible)*fraction)
    ranked = eligible[np.lexsort((eligible, first_distances[eligible]))][:count]
    row = dict(first_update_gated_points=len(eligible), first_update_retained_points=count,
               first_update_retained_inliers=int(labels[ranked].sum()),
               first_update_retained_outliers=int((~labels[ranked]).sum()),
               source_points=len(source), target_points=len(target), recovered=False,
               pose_ambiguity_known=len(target) == 180)
    try:
        trace = sr.register_icp_diagnostics(seeded, target, gate, iterations, trim_fraction=fraction,
            translation_epsilon=1e-4, rotation_epsilon=1e-4, fitness_epsilon=0)
    except ValueError as error:
        row.update(status='error', error=str(error))
        return row
    result = trace.result
    # Exact ties follow source order for trimming, independent of target NN ties.
    assert trace.history[0].correspondences == count
    pose = result.transform().astype(np.float64) @ prior.astype(np.float64)
    rotation_error, translation_error = pose_errors(pose, truth, 1)
    aligned = sr.apply_transform(source, pose.astype(np.float32))
    support_count, support, rmse = sr.distance_gated_support(aligned, target, .05)
    _, reverse, _ = sr.distance_gated_support(target, aligned, .05)
    row.update(status='success', error=None, recovered=rotation_error < 1 and translation_error < .01,
               rotation_error_degrees=rotation_error, translation_error_metres=translation_error,
               iterations=result.iterations, converged=result.converged, stop_reason=trace.stop_reason,
               full_gated_fitness=result.fitness, forward_supported_points=support_count,
               forward_support=support, reverse_support=reverse, evaluation_distance=.05,
               gated_rmse_metres=rmse, final_retained_estimator_pairs=trace.history[-1].correspondences,
               final_full_gated_pairs=trace.history[-1].evaluated_correspondences)
    return row


def render(rows):
    tables = []
    for case, angle, gate in sorted(set((r['case'], r['initial_error_degrees'], r['gate']) for r in rows)):
        cells = []
        for fraction in (1., .8, .5):
            group = [r for r in rows if (r['case'], r['initial_error_degrees'], r['gate'], r['trim_fraction']) == (case, angle, gate, fraction)]
            good = sum(r['recovered'] for r in group)
            retention = ', '.join(f'{r["first_update_retained_inliers"]}/{r["first_update_retained_outliers"]}' for r in group)
            cells.append(f'<td><meter min="0" max="{len(group)}" value="{good}"></meter> {good}/{len(group)}'
                         f'<br><small>initial true/false kept: {retention}</small></td>')
        tables.append(f'<tr><td>{html.escape(case)}</td><td>{angle:g}</td><td>{gate:g}</td>' + ''.join(cells) + '</tr>')
    changes = []
    paired = {}
    for row in rows:
        paired.setdefault((row['case'], row['seed'], row['initial_error_degrees'], row['gate']), {})[row['trim_fraction']] = row
    for fraction in (.8, .5):
        improved = sum(not g[1.]['recovered'] and g[fraction]['recovered'] for g in paired.values())
        worsened = sum(g[1.]['recovered'] and not g[fraction]['recovered'] for g in paired.values())
        changes.append(f'<li>Retain {fraction:.0%}: {improved} paired recoveries gained; {worsened} lost versus all pairs.</li>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Trimmed ICP failure study</title><style>body{font:15px system-ui;padding:1rem}td,th{padding:.5rem;text-align:left}tr:nth-child(even){background:#f1f5f9}small{color:#475569}</style>'
            '<h1>Trimmed ICP failure study</h1><p>Same points, prior, distance gate and stopping policy; '
            'only the retained fraction changes. Recovery means rotation error below 1 degree and translation '
            'error below .01 m relative to the synthetic generating pose. No truth-based pose selection.</p><ul>'
            + ''.join(changes) + '</ul><p>Initial true/false kept counts show which labeled synthetic members '
            'survive distance ranking on the first update. Larger residuals may belong to useful correct points '
            'when the prior is poor. A geometric inlier is not necessarily paired with its generating target point.</p>'
            '<table><tr><th>Environment</th><th>Prior error (degrees)</th><th>Gate (m)</th><th>All pairs</th>'
            '<th>Retain 80%</th><th>Retain 50%</th></tr>' + ''.join(tables) + '</table>'
            '<p>Support and RMSE in study.json evaluate the full original source at .05 m, without shrinking '
            'the denominator after trimming. Native fitness evaluates all rematched gated pairs. '
            'The symmetric ring has multiple equivalent poses and only one fixture; failure to recover its '
            'generating pose is not observable physical error. Synthetic SpatialRust-only study, not an external '
            'library ranking or a guarantee for real sensors. Output directory is exclusive.</p></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20 or not 1 <= args.iterations <= 1000:
        parser.error('seeds must be 1..20 and iterations 1..1000')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    for case in CASES:
        for seed in range(1 if case == 'symmetric_ring' else args.seeds):
            source_xyz, target_xyz, truth, labels = fixture(case, seed)
            hashes[f'{case}/{seed}'] = dict(source=hashlib.sha256(source_xyz.tobytes()).hexdigest(),
                                           target=hashlib.sha256(target_xyz.tobytes()).hexdigest())
            source, target = sr.PointCloud.from_xyz(source_xyz), sr.PointCloud.from_xyz(target_xyz)
            for angle in (0, 20, 40):
                perturb = np.eye(4)
                perturb[:3, :3], perturb[:3, 3] = z_rotation(angle), [.03, -.02, .01]
                prior = rigid_matrix(perturb @ truth)
                seeded = sr.apply_transform(source, prior)
                differences = seeded.xyz()[:, None, :] - target_xyz[None, :, :]
                first_distances = (differences*differences).sum(axis=2).min(axis=1)
                for gate, fraction in itertools.product((.3, .6, 1.2), (1., .8, .5)):
                    row = evaluate(source, target, seeded, prior, truth, labels, first_distances, gate, fraction, args.iterations)
                    row.update(case=case, seed=seed, initial_error_degrees=angle, gate=gate, trim_fraction=fraction)
                    rows.append(row)
        print(f'{case}: {len(rows)} cumulative runs', file=sys.stderr, flush=True)
    native = Path(sr.__file__)
    if native.suffix != '.so':
        binaries = sorted(native.parent.glob('*.so'))
        if len(binaries) != 1:
            raise RuntimeError('cannot identify native extension')
        native = binaries[0]
    sources = [Path(__file__), ROOT/'scripts/study_icp_convergence.py', ROOT/'crates/spatialrust-registration/src/icp.rs',
               ROOT/'crates/spatialrust-registration/src/kabsch.rs', ROOT/'crates/spatialrust-py/examples/align_point_clouds.py',
               ROOT/'crates/spatialrust-py/src/lib.rs']
    report = dict(schema='spatialrust.trimmed-icp-study.v1', seeds=args.seeds, max_iterations=args.iterations,
                  fixture_sha256=hashes, rows=rows, numpy_version=np.__version__,
                  native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  limitations='Synthetic SpatialRust-only paired fraction study; ring one fixture; no universal or real-sensor accuracy claim.')
    (args.output_dir/'study.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output_dir/'report.html').write_text(render(rows), encoding='utf-8')
    print(json.dumps(dict(runs=len(rows), recovered=sum(r['recovered'] for r in rows),
                         failures=sum(r['status']=='error' for r in rows)), indent=2))


if __name__ == '__main__':
    main()
