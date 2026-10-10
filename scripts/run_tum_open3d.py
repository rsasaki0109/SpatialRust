"""Separate reference-free Open3D baseline with the committed TUM geometry/budgets.

Uses the same native projection and common CPU voxel points. Open3D requests
the full iteration budget with zero relative stopping thresholds; these differ
from native absolute stopping criteria. No speed/equivalent-stopping claim.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from run_tum_sequence import compose_camera_pose, generate
from tum_sequence_reference import file_sha256


def align_open3d(current, previous, controls):
    import open3d as o3d
    pose, stages = np.eye(4), []
    for stage in controls['stages']:
        moved = sr.apply_transform(current, pose.astype(np.float32))
        query = sr.voxel_downsample(moved, stage['leaf_m'], 'cpu')
        target = sr.voxel_downsample(previous, stage['leaf_m'], 'cpu')
        if min(len(query), len(target)) < 3:
            raise ValueError('fewer than three common voxel points')
        clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c.xyz().astype(np.float64))) for c in (query, target)]
        result = o3d.pipelines.registration.registration_icp(
            *clouds, stage['gate_m'], np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=0., relative_rmse=0., max_iteration=stage['iterations']))
        if len(result.correspondence_set) < 3:
            raise ValueError('Open3D stage has fewer than three retained correspondences')
        pose = compose_camera_pose(result.transformation, pose)
        stages.append(dict(requested_iteration_cap=stage['iterations'], actual_updates_observable=False,
                           relative_fitness_threshold=0., relative_rmse_threshold=0.,
                           query_points=len(query), target_points=len(target),
                           fitness=result.fitness, inlier_rmse_m=result.inlier_rmse,
                           previous_from_current=pose.tolist()))
    count, fraction, rmse = sr.distance_gated_support(
        sr.apply_transform(current, pose.astype(np.float32)), previous, controls['stages'][-1]['gate_m'])
    if count < 3 or fraction < controls['min_support_fraction'] or rmse is None:
        raise ValueError('Open3D lacks fixed common support: count={}, fraction={}, rmse={}'.format(count, fraction, rmse))
    return pose, dict(stages=stages, support_count=count, support_fraction=fraction, support_rmse_m=rmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).resolve().parents[1]/'benchmarks/tum-fr1-xyz-plan.json'
    baseline = json.loads((args.prepared_dir/'plan.json').read_bytes())
    if baseline.get('prior_native_plan_sha256') != file_sha256(original):
        raise ValueError('baseline must bind the original pre-acquisition native plan')
    native_plan = json.loads(original.read_bytes())
    if any(baseline[key] != native_plan[key] for key in ('calibration', 'controls', 'limits')):
        raise ValueError('baseline geometry/frame/budget controls must match the original plan')
    native = Path(importlib.import_module('open3d.cpu.pybind').__file__)
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                      pair_aligner=align_open3d, method='open3d_cpu', additional_bindings=(native,))
    print(json.dumps(dict(method=result['method'], planned=result['planned_poses'],
                          generated=result['generated_poses'], estimates_sha256=result['estimates_sha256'])))


if __name__ == '__main__':
    main()
