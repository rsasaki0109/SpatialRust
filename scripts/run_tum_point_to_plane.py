"""Explicit native point-to-plane trajectory study; the default ICP is unchanged."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from run_tum_sequence import compose_camera_pose, generate
from tum_sequence_reference import file_sha256, validate_plan


ESTIMATOR = dict(method='spatialrust_point_to_plane_cpu', normal_neighbors=20,
                 transformation_epsilon=1e-8, fitness_epsilon=1e-6,
                 min_correspondences=6)


def validate_plane_plan(plan, original):
    """Bind unchanged input geometry/budgets and the exposed native defaults."""
    validate_plan(plan)
    if plan.get('prior_native_plan_sha256') != file_sha256(original):
        raise ValueError('point-to-plane plan must bind the original native protocol')
    native = json.loads(Path(original).read_bytes())
    if any(plan[key] != native[key] for key in ('calibration', 'controls', 'limits')):
        raise ValueError('geometry/frame/budget controls must match original protocol')
    if plan.get('estimator') != ESTIMATOR:
        raise ValueError('unsupported point-to-plane normal or convergence settings')
    return plan


def align_point_to_plane(current, previous, controls):
    pose, stages = np.eye(4), []
    for stage in controls['stages']:
        moved = sr.apply_transform(current, pose.astype(np.float32))
        query = sr.voxel_downsample(moved, stage['leaf_m'], 'cpu')
        target = sr.voxel_downsample(previous, stage['leaf_m'], 'cpu')
        if min(len(query), len(target)) < ESTIMATOR['normal_neighbors']:
            raise ValueError('too few common voxel points for fixed normal estimation')
        result = sr.register_point_to_plane(query, target, stage['gate_m'],
                                            stage['iterations'], ESTIMATOR['normal_neighbors'])
        pose = compose_camera_pose(result.transform(), pose)
        stages.append(dict(iterations=result.iterations, converged=result.converged,
                           stopping_cause_observable=False, query_points=len(query),
                           target_points=len(target), normal_neighbors=ESTIMATOR['normal_neighbors'],
                           point_to_plane_fitness=result.fitness,
                           previous_from_current=pose.tolist()))
    count, fraction, rmse = sr.distance_gated_support(
        sr.apply_transform(current, pose.astype(np.float32)), previous, controls['stages'][-1]['gate_m'])
    if count < 3 or fraction < controls['min_support_fraction'] or rmse is None:
        raise ValueError('point-to-plane lacks fixed common support: count={}, fraction={}, rmse={}'.format(count, fraction, rmse))
    return pose, dict(stages=stages, support_count=count, support_fraction=fraction, support_rmse_m=rmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).resolve().parents[1]/'benchmarks/tum-fr1-xyz-plan.json'
    validate_plane_plan(json.loads((args.prepared_dir/'plan.json').read_bytes()), original)
    native = list(Path(sr.__file__).parent.glob('*.so')) + list(Path(sr.__file__).parent.glob('*.pyd'))
    if len(native) != 1:
        raise ValueError('require one installed native extension')
    receipt = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                       pair_aligner=align_point_to_plane, method=ESTIMATOR['method'],
                       additional_bindings=(native[0],))
    print(json.dumps(dict(method=receipt['method'], planned=receipt['planned_poses'],
                          generated=receipt['generated_poses'], estimates_sha256=receipt['estimates_sha256'])))


if __name__ == '__main__':
    main()
