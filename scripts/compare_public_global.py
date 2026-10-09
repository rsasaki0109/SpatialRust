"""Compare no-prior global registration on a hash-bound public reference pair.

Shared input coordinates, coarse points, seeds, gates and iteration caps;
feature neighborhoods, RANSAC samplers and ICP numerics differ. No speed ranking.
"""
import argparse
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import open3d as o3d
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/spatialrust-py/examples'))
from align_global import align_global_clouds
from align_point_clouds import read_bound_cloud
from evaluate_pose_reference import evaluate, validate_reference


def open3d_global(source, target, seed, budget):
    # Common SpatialRust voxel points avoid a second library's voxel-grid origin.
    coarse = [sr.voxel_downsample(cloud, .1, 'cpu').xyz() for cloud in (source, target)]
    prepared, features = [], []
    for xyz in coarse:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        cloud.orient_normals_towards_camera_location(np.zeros(3))
        features.append(o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=.5, max_nn=100)))
        prepared.append(cloud)
    o3d.utility.random.seed(seed)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        *prepared, *features, True, .2,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.2)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(budget, .99))
    if result.fitness <= 0:
        raise ValueError('Open3D global initialization has no retained correspondence')
    pose = result.transformation
    for leaf, gate in [(.05, .2), (None, .05)]:
        clouds = [cloud if leaf is None else sr.voxel_downsample(cloud, leaf, 'cpu') for cloud in (source, target)]
        clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c.xyz().astype(np.float64))) for c in clouds]
        pose = o3d.pipelines.registration.registration_icp(
            *clouds, gate, pose, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0, relative_rmse=0, max_iteration=50)).transformation
    return dict(schema_version='spatialrust.python-alignment.v1', initial_transform_supplied=False,
                transform_source_to_target=pose.tolist())


def run(source, target, fingerprints, seeds, budget):
    rows = []
    schedule = [dict(leaf=.05, max_distance=.2, iterations=50), dict(leaf=None, max_distance=.05, iterations=50)]
    target_index = sr.DistanceSupportIndex(target)
    for seed in seeds:
        for method in ('spatialrust', 'open3d'):
            row = dict(seed=seed, method=method, status='error')
            started = time.perf_counter()
            try:
                if method == 'spatialrust':
                    _, report = align_global_clouds(source, target, schedule, seeds=[seed], ransac_iterations=budget)
                else:
                    report = open3d_global(source, target, seed, budget)
                elapsed = time.perf_counter() - started
                report['input_file_sha256'] = dict(fingerprints)
                aligned = sr.apply_transform(source, np.array(report['transform_source_to_target'], np.float32))
                count, fraction, rmse = target_index.support(aligned, .05)
                row.update(status='success', report=report, elapsed_seconds=elapsed,
                           common_support=dict(count=count, fraction=fraction, rmse=rmse))
            except (ValueError, RuntimeError) as error:
                row.update(error=str(error), elapsed_seconds=time.perf_counter() - started)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 8, 9])
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not 1 <= len(args.seeds) <= 16 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**31 for s in args.seeds):
        parser.error('require 1..16 distinct signed-32-bit-compatible nonnegative seeds')
    if not 1 <= args.iterations <= 1000000:
        parser.error('iterations must be 1..1000000')
    if any(os.environ.get(key) != '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        parser.error('set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    validate_reference(reference)
    if reference['length_unit'] != 'm':
        parser.error('this comparison uses metre-based gates; reference must declare m')
    source, source_hash = read_bound_cloud(args.source)
    target, target_hash = read_bound_cloud(args.target)
    fingerprints = dict(source=source_hash, target=target_hash)
    if fingerprints != reference['input_file_sha256']:
        parser.error('input pair does not match reference hashes')
    for cloud in (source, target):
        if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
            parser.error('require at least three finite points per cloud')
        size = len(sr.voxel_downsample(cloud, .1, 'cpu'))
        if not 3 <= size <= 5000:
            parser.error('shared coarse cloud must contain 3..5000 points')
    native = list(Path(sr.__file__).parent.glob('*.so'))
    if len(native) != 1:
        parser.error('expected one native extension')
    native_hash = hashlib.sha256(native[0].read_bytes()).hexdigest()
    open3d_native = Path(importlib.import_module('open3d.cpu.pybind').__file__)
    open3d_hash = hashlib.sha256(open3d_native.read_bytes()).hexdigest()
    algorithm_paths = [Path(__file__).with_name('evaluate_pose_reference.py')] + [
        Path(__file__).resolve().parents[1] / 'crates/spatialrust-py/examples' / name
        for name in ('align_global.py', 'align_multiscale.py', 'align_point_clouds.py')]
    algorithm_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in algorithm_paths}
    source_bytes = Path(__file__).read_bytes()
    rows = run(source, target, fingerprints, args.seeds, args.iterations)
    # Reference pose is used only after all algorithms finish.
    for row in rows:
        if row['status'] == 'success':
            row['pose_accuracy'] = evaluate(row['report'], reference)
    if hashlib.sha256(native[0].read_bytes()).hexdigest() != native_hash or Path(__file__).read_bytes() != source_bytes:
        raise ValueError('native or runner changed during comparison')
    if hashlib.sha256(open3d_native.read_bytes()).hexdigest() != open3d_hash or any(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in algorithm_hashes.items()
    ):
        raise ValueError('Open3D native or algorithm helpers changed during comparison')
    receipt = dict(schema='spatialrust.public-global-comparison.v1',
                   versions=dict(spatialrust=sr.__version__, open3d=o3d.__version__, numpy=np.__version__, python=platform.python_version()),
                   reference_sha256=hashlib.sha256(reference_bytes).hexdigest(), reference_provenance=reference['provenance'],
                   native_sha256=native_hash, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                   open3d_native_sha256=open3d_hash, algorithm_source_sha256=algorithm_hashes,
                   input_file_sha256=fingerprints, seeds=args.seeds, ransac_iterations=args.iterations,
                   controls=dict(coarse_leaf=.1, feature_radius=.5, normal_neighbors=20, global_gate=.2,
                                 stages=[dict(leaf=.05, gate=.2, max_iterations=50), dict(leaf=None, gate=.05, max_iterations=50)],
                                 evaluation_gate=.05, thread_environment={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
                   rows=rows)
    table = []
    for row in rows:
        if row['status'] == 'success':
            error = row['pose_accuracy']
            cells = f'<td>{error["rotation_error_degrees"]:.6g}°</td><td>{error["translation_error"]:.6g} m</td><td>{row["common_support"]["fraction"]:.6g}</td>'
        else:
            cells = '<td colspan="3">' + html.escape(row['error']) + '</td>'
        table.append(f'<tr><td>{row["method"]}</td><td>{row["seed"]}</td>{cells}</tr>')
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Public reference comparison</title>'
                '<h1>No-prior global registration</h1><p>Same full and coarse point coordinates, seeds and distance gates. '
                'Feature neighborhoods, samplers, numeric precision and stopping differ. No truth-based pose selection. '
                'Timing is recorded for diagnosis, not a speed ranking. Declared reference provenance is not independent sensor certification.</p>'
                '<table><tr><th>Method</th><th>Seed</th><th>Rotation error</th><th>Translation error</th><th>Proximity fraction</th></tr>'
                + ''.join(table) + '</table></html>')
    serialized = json.dumps(receipt, indent=2, allow_nan=False)
    args.output_dir.mkdir()
    try:
        (args.output_dir / 'comparison.json').write_text(serialized, encoding='utf-8')
        (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Completed {len(rows)} rows; failures={sum(r["status"] == "error" for r in rows)}')


if __name__ == '__main__':
    main()
