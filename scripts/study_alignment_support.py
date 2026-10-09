import importlib.util
import json
from pathlib import Path
import numpy as np
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'target' / 'alignment-study'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('alignment_example', ROOT / 'crates/spatialrust-py/examples/align_point_clouds.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

def run_case(name, seed, source, target, truth):
    folder = OUT / f'{name}-{seed}'
    folder.mkdir(exist_ok=True)
    paths = [folder / 'source.pcd', folder / 'target.pcd']
    for path, xyz in zip(paths, (source, target)):
        sr.write(str(path), sr.PointCloud.from_xyz(np.asarray(xyz, dtype=np.float32)))
    _, report = example.align_files(*paths, leaf=.04, max_distance=.12, iterations=60)
    pose = np.asarray(report['transform_source_to_target'])
    # Project f32 roundoff onto SO(3) before measuring the geodesic angle.
    u, _, vt = np.linalg.svd(pose[:3,:3])
    correction = np.diag([1,1,np.linalg.det(u @ vt)])
    measured_rotation = u @ correction @ vt
    cosine = np.clip((np.trace(measured_rotation @ truth[:3,:3].T)-1)/2, -1, 1)
    result = dict(case=name, seed=seed, translation_error_metres=float(np.linalg.norm(pose[:3,3]-truth[:3,3])),
                  rotation_error_degrees=float(np.degrees(np.arccos(cosine))), diagnostics=report)
    (folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result

results = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    target = rng.uniform(-1,1,(400,3)).astype(np.float32)
    shift = np.array([.025,-.018,.012], dtype=np.float32)
    truth = np.eye(4); truth[:3,3] = -shift
    results.append(run_case('baseline', seed, target+shift, target, truth))
    results.append(run_case('partial_60_percent', seed, target[:240]+shift, target, truth))
    outliers = rng.uniform(3,4,(160,3)).astype(np.float32)
    results.append(run_case('source_outliers_40_percent', seed, np.vstack([target[:240]+shift,outliers]), target, truth))
    # 180-fold ring symmetry: 20 degrees is exactly 10 vertex spacings.
    angles = np.arange(180) * 2*np.pi/180
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(180)], axis=1)
    theta = np.deg2rad(20)
    rotation = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    truth = np.eye(4); truth[:3,:3] = rotation.T
    results.append(run_case('symmetric_ring_20_degrees', seed, ring@rotation.T, ring, truth))
summary = dict(schema='spatialrust.synthetic-alignment-study.v1',
    scope='Synthetic fixed-seed diagnostic study; no external library comparison or real sensor data. Ring pose is intrinsically ambiguous from geometry alone.',
    seeds=list(range(5)), results=results)
(OUT / 'study.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
for case in sorted(set(r['case'] for r in results)):
    rows = [r for r in results if r['case']==case]
    print(case, json.dumps(dict(translation_error_max=max(r['translation_error_metres'] for r in rows),
        rotation_error_max=max(r['rotation_error_degrees'] for r in rows),
        forward_support=[r['diagnostics']['aligned_support']['query_fraction'] for r in rows],
        reverse_support=[r['diagnostics']['aligned_reverse_support']['query_fraction'] for r in rows],
        rmse_max=max(r['diagnostics']['aligned_support']['gated_rmse_metres'] for r in rows),
        converged=[r['diagnostics']['converged'] for r in rows])))
