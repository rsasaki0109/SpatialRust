"""Freeze native RGB-D image-tracking/PnP poses without using reference motion."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from evaluate_pose_reference import rigid
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, validate_plan


ESTIMATOR = 'spatialrust_rgbd_pnp_cpu'
ORIGINAL_PLAN = Path(__file__).resolve().parents[1]/'benchmarks/tum-fr1-xyz-plan.json'
DEFAULT_CONTROLS = dict(max_features=300, quality_level=.01, min_distance=10.,
                        image_margin=40, window_radius=5, pyramid_levels=3,
                        max_iterations=30, epsilon=.001, min_eigenvalue=.0001,
                        max_forward_backward_error=.75, pnp_threshold=2.,
                        min_tracks=20, min_inliers=15, min_inlier_fraction=.5)


def validate_rgbd_plan(plan):
    validate_plan(plan)
    original = json.loads(ORIGINAL_PLAN.read_text(encoding='utf-8'))
    expected_controls = dict(original['controls'], rgbd_odometry=DEFAULT_CONTROLS)
    if plan.get('prior_native_plan_sha256') != file_sha256(ORIGINAL_PLAN) or plan['controls'] != expected_controls or any(plan[key] != original[key] for key in ('calibration', 'limits')):
        raise ValueError('RGB-D calibration, support, geometry and limits must retain the original protocol')
    if plan.get('estimator') != ESTIMATOR or plan['controls'].get('rgbd_odometry') != DEFAULT_CONTROLS:
        raise ValueError('require the fixed native RGB-D tracking/PnP protocol')
    acceptance = plan.get('acceptance', {})
    for name, expected in dict(minimum_generated_fraction=1., minimum_evaluated_fraction=.95,
                               maximum_first_pose_ate_rmse_m=.1,
                               maximum_one_second_translation_rpe_rmse_m=.05).items():
        if type(acceptance.get(name)) not in (int, float) or acceptance[name] != expected:
            raise ValueError('RGB-D study must retain the fixed complete/coverage/accuracy limits')
    return plan


def align_rgbd(previous_depth, previous_gray, current_gray, camera, controls):
    """Native PnP maps previous -> current; invert once for the existing world chain."""
    settings = controls.get('rgbd_odometry')
    if settings != DEFAULT_CONTROLS:
        raise ValueError('RGB-D settings differ from the fixed protocol')
    keypoints = sr.shi_tomasi_keypoints(previous_gray, max_corners=settings['max_features'],
                                      quality_level=settings['quality_level'],
                                      min_distance=settings['min_distance'], block_size=3, gradient_size=3)
    points = np.array([[point.x, point.y] for point in keypoints], dtype=np.float64).reshape(-1, 2)
    margin = settings['image_margin']
    interior = ((points[:, 0] >= margin) & (points[:, 0] < camera['width']-margin) &
                (points[:, 1] >= margin) & (points[:, 1] < camera['height']-margin))
    points = points[interior]
    # Match Rust's positive-coordinate round(), not NumPy's half-even rint().
    pixels = np.floor(points + .5).astype(np.int64)
    depths = previous_depth[pixels[:, 1], pixels[:, 0]].astype(np.float64)*camera['depth_unit_m']
    points = np.ascontiguousarray(points[(depths >= controls['min_depth_m']) &
                                       (depths <= controls['max_depth_m'])])
    if len(points) < settings['min_tracks']:
        raise ValueError('RGB-D has only {} valid-depth keypoints; require {}'.format(len(points), settings['min_tracks']))
    following, tracked, roundtrip = sr.track_points_forward_backward(
        previous_gray, current_gray, points, window_radius=settings['window_radius'],
        pyramid_levels=settings['pyramid_levels'], max_iterations=settings['max_iterations'],
        epsilon=settings['epsilon'], min_eigenvalue=settings['min_eigenvalue'],
        max_forward_backward_error=settings['max_forward_backward_error'])
    source, target = np.ascontiguousarray(points[tracked]), np.ascontiguousarray(following[tracked])
    if len(source) < settings['min_tracks']:
        raise ValueError('RGB-D accepted {}/{} forward/backward tracks; require {}'.format(len(source), len(points), settings['min_tracks']))
    metres = (previous_depth.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
    rotation, translation, inliers, rejected = sr.estimate_rgbd_odometry(
        metres, source, target, camera['fx'], camera['fy'], camera['cx'], camera['cy'],
        depth_scale=1., threshold=settings['pnp_threshold'])
    count = int(np.count_nonzero(inliers))
    if rejected or len(inliers) != len(source) or count < settings['min_inliers'] or count/len(source) < settings['min_inlier_fraction']:
        raise ValueError('RGB-D PnP has {}/{} inliers and {} rejected depths'.format(count, len(source), rejected))
    current_from_previous = np.eye(4)
    current_from_previous[:3, :3] = rotation
    current_from_previous[:3, 3] = translation
    rigid(current_from_previous)
    previous_from_current = np.eye(4)
    previous_from_current[:3, :3] = rotation.T
    previous_from_current[:3, 3] = -rotation.T @ translation
    return rigid(previous_from_current), dict(
        objective='native Shi-Tomasi, forward/backward pyramidal LK and metric PnP RANSAC',
        valid_depth_keypoints=len(points), accepted_tracks=len(source), inlier_count=count,
        inlier_fraction=count/len(source), rejected_depth_count=rejected,
        max_accepted_roundtrip_px=float(np.max(roundtrip[tracked])),
        source_pixels=source.tolist(), target_pixels=target.tolist(), inliers=inliers.tolist(),
        current_from_previous=current_from_previous.tolist(),
        previous_from_current=previous_from_current.tolist(), settings=settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    validate_rgbd_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                      rgbd_aligner=align_rgbd, method=ESTIMATOR, additional_bindings=[__file__, ORIGINAL_PLAN])
    print('Frozen RGB-D {}/{} poses; estimates SHA-256 {}'.format(
        result['generated_poses'], result['planned_poses'], result['estimates_sha256']))


if __name__ == '__main__':
    main()
