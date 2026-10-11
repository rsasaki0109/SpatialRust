"""Closed-form point-to-point coarse alignment, then native plane refinement."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from run_tum_adaptive_keyframe import ESTIMATOR as OVERLAP_ESTIMATOR, OverlapKeyframeTracker, validate_overlap_plan
from run_tum_sequence import compose_camera_pose, generate


ESTIMATOR = dict(OVERLAP_ESTIMATOR, method='spatialrust_hybrid_keyframe_cpu',
                 stage_objectives=['point_to_point', 'point_to_plane'],
                 coarse_translation_epsilon=1e-8, coarse_rotation_epsilon=1e-8,
                 coarse_fitness_epsilon=1e-6, coarse_trim_fraction=1.)


def validate_hybrid_plan(plan, original):
    if plan.get('estimator') != ESTIMATOR:
        raise ValueError('unsupported fixed hybrid objectives or convergence controls')
    validate_overlap_plan(dict(plan, estimator=OVERLAP_ESTIMATOR), original)
    return plan


def align_hybrid(current, previous, controls):
    if len(controls['stages']) != 2:
        raise ValueError('hybrid requires its fixed coarse/fine schedule')
    pose, stages = np.eye(4), []
    for index, stage in enumerate(controls['stages']):
        moved = sr.apply_transform(current, pose.astype(np.float32))
        query = sr.voxel_downsample(moved, stage['leaf_m'], 'cpu')
        target = sr.voxel_downsample(previous, stage['leaf_m'], 'cpu')
        if min(len(query), len(target)) < (3 if index == 0 else ESTIMATOR['normal_neighbors']):
            raise ValueError('too few common voxel points for the hybrid stage')
        if index == 0:
            trace = sr.register_icp_diagnostics(query, target, stage['gate_m'], stage['iterations'],
                translation_epsilon=ESTIMATOR['coarse_translation_epsilon'],
                rotation_epsilon=ESTIMATOR['coarse_rotation_epsilon'],
                fitness_epsilon=ESTIMATOR['coarse_fitness_epsilon'],
                trim_fraction=ESTIMATOR['coarse_trim_fraction'])
            result = trace.result
            stopping = dict(stop_reason=trace.stop_reason, stopping_cause_observable=True)
        else:
            result = sr.register_point_to_plane(query, target, stage['gate_m'], stage['iterations'],
                                                ESTIMATOR['normal_neighbors'])
            stopping = dict(stopping_cause_observable=False)
        pose = compose_camera_pose(result.transform(), pose)
        stages.append(dict(stopping, objective=ESTIMATOR['stage_objectives'][index],
                           iterations=result.iterations, converged=result.converged,
                           query_points=len(query), target_points=len(target),
                           native_fitness=result.fitness, previous_from_current=pose.tolist()))
    count, fraction, rmse = sr.distance_gated_support(
        sr.apply_transform(current, pose.astype(np.float32)), previous, controls['stages'][-1]['gate_m'])
    if count < 3 or fraction < controls['min_support_fraction'] or rmse is None:
        raise ValueError('hybrid lacks fixed common support: count={}, fraction={}, rmse={}'.format(count, fraction, rmse))
    return pose, dict(stages=stages, support_count=count, support_fraction=fraction, support_rmse_m=rmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).resolve().parents[1]/'benchmarks/tum-fr1-xyz-plan.json'
    validate_hybrid_plan(json.loads((args.prepared_dir/'plan.json').read_bytes()), original)
    native = list(Path(sr.__file__).parent.glob('*.so')) + list(Path(sr.__file__).parent.glob('*.pyd'))
    if len(native) != 1:
        raise ValueError('require one installed native extension')
    tracker = OverlapKeyframeTracker(pair_aligner=align_hybrid)
    helpers = [Path(__file__)] + [Path(__file__).with_name(name) for name in
                                  ('run_tum_keyframe.py','run_tum_point_to_plane.py')]
    receipt = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                       pair_aligner=tracker.align_pair, method=ESTIMATOR['method'],
                       additional_bindings=helpers+[native[0]])
    print(json.dumps(dict(method=receipt['method'], planned=receipt['planned_poses'],
                          generated=receipt['generated_poses'], estimates_sha256=receipt['estimates_sha256'])))


if __name__ == '__main__':
    main()
