"""Fixed-seed ICP initialization experiment; regenerates synthetic artifacts."""
import json
import argparse
import math
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evaluation-distance', type=float, help='fixed diagnostic distance; defaults to each ICP gate')
parser.add_argument('--output-dir', type=Path, default=REPO / 'target/gate-study')
parser.add_argument('--noise-std', type=float, default=0, help='source Gaussian noise standard deviation per axis, metres')
source_changes = parser.add_mutually_exclusive_group()
source_changes.add_argument('--source-outlier-fraction', type=float, default=0, help='fraction of source points replaced by nearby nonmatching points')
source_changes.add_argument('--source-delete-fraction', type=float, default=0, help='fraction of final source points deleted after noise, for paired overlap control')
args = parser.parse_args()
if args.evaluation_distance is not None and (not math.isfinite(args.evaluation_distance) or args.evaluation_distance <= 0):
    parser.error('evaluation-distance must be finite and positive')
if not math.isfinite(args.noise_std) or args.noise_std < 0:
    parser.error('noise-std must be finite and nonnegative')
if not math.isfinite(args.source_outlier_fraction) or not 0 <= args.source_outlier_fraction < 1:
    parser.error('source-outlier-fraction must be in [0, 1)')
if not math.isfinite(args.source_delete_fraction) or not 0 <= args.source_delete_fraction < 1:
    parser.error('source-delete-fraction must be in [0, 1)')
deleted_count = int(400 * args.source_delete_fraction)
if 400 - deleted_count < 3:
    parser.error('deletion must retain at least three source points')
ROOT = args.output_dir
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'crates/spatialrust-py/examples'))
from align_point_clouds import align_files

def rotation(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def matrix(r, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m

def write(path, xyz):
    header = f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA ascii\n'
    path.write_text(header + '\n'.join(' '.join(format(float(x), '.9g') for x in p) for p in xyz) + '\n')

def errors(estimated, truth):
    u, _, vh = np.linalg.svd(estimated[:3, :3])
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(r @ truth[:3, :3].T) - 1) / 2, -1, 1))))
    translation = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    return angle, translation

rows = []
for seed in range(5):
    target = np.random.default_rng(seed).uniform(-1, 1, (400, 3))
    r, t = rotation(60), np.array([5., -2., .5])
    source = target @ r.T + t
    # Separate perturbation RNG keeps target geometry identical across conditions.
    disturbance = np.random.default_rng(10000 + seed)
    outlier_rng = np.random.default_rng(20000 + seed)
    outlier_count = int(len(source) * args.source_outlier_fraction)
    if outlier_count:
        outliers = outlier_rng.uniform(-1, 1, (outlier_count, 3))
        outliers[:, 0] = outlier_rng.uniform(1.2, 2, outlier_count)
        source[-outlier_count:] = outliers @ r.T + t
    if args.noise_std:
        source += disturbance.normal(0, args.noise_std, source.shape)
    if deleted_count:
        source = source[:-deleted_count]
    truth = matrix(r.T, -r.T @ t)
    sp, tp = ROOT / f'source-{seed}.pcd', ROOT / f'target-{seed}.pcd'
    write(sp, source)
    write(tp, target)
    for gate in [.05, .15, .3, .6, 1.2]:
        degrees = 40
        perturbation = matrix(rotation(degrees), np.array([.03, -.02, .01]))
        prior = perturbation @ truth
        row = dict(seed=seed, perturbation_degrees=degrees, leaf_metres=.025,
                   gate_metres=gate, iterations_per_stage=100,
                   perturbation_translation_metres=[.03, -.02, .01],
                   perturbation_convention='target-frame perturbation @ correct inverse')
        try:
            _, d = align_files(sp, tp, leaf=.025, max_distance=gate,
                               iterations=100, initial_transform=prior,
                               evaluation_distance=args.evaluation_distance)
            angle, translation = errors(np.asarray(d['transform_source_to_target']), truth)
            row.update(rotation_error_degrees=angle, translation_error_metres=translation,
                       success=angle < 1 and translation < .01,
                       converged=d['converged'],
                       forward_fraction=d['aligned_support']['query_fraction'],
                       reverse_fraction=d['aligned_reverse_support']['query_fraction'],
                       rmse_metres=d['aligned_support']['gated_rmse_metres'], diagnostics=d)
        except Exception as exc:
            row.update(success=False, error_type=type(exc).__name__, error=str(exc))
        rows.append(row)
summary = []
for gate in [.05, .15, .3, .6, 1.2]:
    group = [row for row in rows if row['gate_metres'] == gate]
    completed = [row for row in group if 'converged' in row]
    summary.append(dict(gate_metres=gate, trials=len(group),
                        successes=sum(row['success'] for row in group),
                        converged=sum(row['converged'] for row in completed),
                        error_count=len(group)-len(completed),
                        rotation_errors=[row['rotation_error_degrees'] for row in completed],
                        translation_errors=[row['translation_error_metres'] for row in completed],
                        forward_support=[row['forward_fraction'] for row in completed],
                        reverse_support=[row['reverse_fraction'] for row in completed],
                        gated_rmse_metres=[row['rmse_metres'] for row in completed]))
(ROOT / 'results.json').write_text(json.dumps(dict(evaluation_distance_metres=args.evaluation_distance,
    noise_std_metres=args.noise_std, source_outlier_fraction_requested=args.source_outlier_fraction,
    source_outlier_count=int(400 * args.source_outlier_fraction), source_points=400-deleted_count,
    source_delete_fraction_requested=args.source_delete_fraction, source_deleted_count=deleted_count,
    outlier_generation='target-frame x in [1.2,2], y/z in [-1,1]; replace final source points before noise',
    rows=rows, summary=summary), indent=2, allow_nan=False) + '\n')
table = []
for item in summary:
    support = item['forward_support']
    span = f'{min(support):.1%}–{max(support):.1%}' if support else 'No completed runs'
    success = item['successes'] / item['trials']
    table.append(f'<tr><td>{item["gate_metres"]:g}</td><td>{item["successes"]}/{item["trials"]}</td>'
                 f'<td>{item["converged"]}/{item["trials"]}</td><td>{span}</td>'
                 f'<td><svg viewBox="0 0 100 10" role="img" aria-label="Correct poses {success:.0%}">'
                 f'<rect width="100" height="10" fill="#eee"/><rect width="{100*success:g}" height="10" fill="#0369a1"/></svg></td></tr>')
evaluation = f'{args.evaluation_distance:g} m (fixed)' if args.evaluation_distance is not None else 'matches each ICP gate'
(ROOT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ICP gate study</title>'
    '<style>body{font:16px system-ui;max-width:950px;margin:2rem auto}td,th{padding:.6rem;text-align:left}svg{width:150px}</style>'
    f'<h1>ICP gate study</h1><p>Evaluation gate: {evaluation}. Five fixed synthetic seeds per search gate.</p>'
    f'<p>Source noise per axis: {args.noise_std:g} m; replaced outliers: {int(400 * args.source_outlier_fraction)}/400 points.</p>'
    f'<p>Deleted source points: {deleted_count}/400; retained source points: {400-deleted_count}.</p>'
    '<table><thead><tr><th>ICP gate (m)</th><th>Correct poses</th><th>Converged</th><th>Forward support range</th><th>Correct pose fraction</th></tr></thead><tbody>'
    + ''.join(table) + '</tbody></table><p>Correctness uses known generating poses: rotation error &lt; 1 degree and translation error &lt; 0.01 m. '
    'Convergence and proximity support do not certify pose. Synthetic uniform geometry; this is not a general gate recommendation.</p></html>', encoding='utf-8')
print(json.dumps(summary, indent=2))
