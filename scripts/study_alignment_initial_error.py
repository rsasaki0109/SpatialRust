"""Fixed-seed ICP initialization experiment; regenerates synthetic artifacts."""
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'target/prior-basin'
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
    path.write_text(header + '\n'.join(' '.join(format(float(x), '.9g') for x in p) for p in xyz) + '\n', encoding='utf-8')

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
    truth = matrix(r.T, -r.T @ t)
    sp, tp = ROOT / f'source-{seed}.pcd', ROOT / f'target-{seed}.pcd'
    write(sp, source)
    write(tp, target)
    for degrees in [0, 3, 10, 20, 40]:
        perturbation = matrix(rotation(degrees), np.array([.03, -.02, .01]))
        prior = perturbation @ truth
        row = dict(seed=seed, perturbation_degrees=degrees, leaf_metres=.025,
                   gate_metres=.15, iterations_per_stage=100,
                   perturbation_translation_metres=[.03, -.02, .01],
                   perturbation_convention='target-frame perturbation @ correct inverse')
        try:
            _, d = align_files(sp, tp, leaf=.025, max_distance=.15,
                               iterations=100, initial_transform=prior)
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
for degrees in [0, 3, 10, 20, 40]:
    group = [row for row in rows if row['perturbation_degrees'] == degrees]
    completed = [row for row in group if 'converged' in row]
    summary.append(dict(perturbation_degrees=degrees, trials=len(group),
                        successes=sum(row['success'] for row in group),
                        converged=sum(row['converged'] for row in completed),
                        error_count=len(group)-len(completed),
                        rotation_errors=[row['rotation_error_degrees'] for row in completed],
                        translation_errors=[row['translation_error_metres'] for row in completed],
                        forward_support=[row['forward_fraction'] for row in completed]))
(ROOT / 'results.json').write_text(json.dumps(dict(rows=rows, summary=summary), indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
