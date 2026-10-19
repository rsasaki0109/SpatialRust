"""Freeze native direct RGB-D poses from both adjacent grayscale/depth frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from dense_rgbd_protocol import CONTROLS, ESTIMATOR, PARENTS, validate_dense_plan
from evaluate_pose_reference import rigid
from run_tum_sequence import generate


def align_dense_rgbd(previous_depth, current_depth, previous_gray, current_gray, camera, controls):
    return _align_dense_rgbd(previous_depth, current_depth, previous_gray, current_gray,
                             camera, controls, expected_final_support_only=False)


def _align_dense_rgbd(previous_depth, current_depth, previous_gray, current_gray, camera, controls,
                     *, expected_final_support_only):
    if controls.get('dense_rgbd_odometry') != CONTROLS:
        raise ValueError('dense RGB-D controls differ from the fixed protocol')
    source = (previous_depth.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
    target = (current_depth.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
    rotation, translation, report = sr.estimate_dense_rgbd_odometry(
        previous_gray, source, current_gray, target,
        camera['fx'], camera['fy'], camera['cx'], camera['cy'], initial=np.eye(4), **CONTROLS)
    if type(report.get('final_support_only', False)) is not bool or report.get('final_support_only', False) != expected_final_support_only:
        raise ValueError('native dense support policy does not match this fixed experiment')
    valid = int(np.count_nonzero((source >= CONTROLS['min_depth']) & (source <= CONTROLS['max_depth'])))
    if (valid <= 0 or type(report.get('valid_source_pixels')) is not int or report['valid_source_pixels'] != valid
            or type(report.get('correspondences')) is not int
            or not CONTROLS['min_correspondences'] <= report['correspondences'] <= valid
            or report['correspondences']/valid < CONTROLS['min_correspondence_fraction']
            or any(type(report.get(key)) not in (int, float) or not np.isfinite(report[key]) or report[key] < 0
                   for key in ('correspondence_fraction', 'photometric_rmse', 'depth_rmse_m'))
            or abs(report['correspondence_fraction']-report['correspondences']/valid) > 1e-12):
        raise ValueError('dense RGB-D does not preserve complete valid-source support')
    current_from_previous = np.eye(4)
    current_from_previous[:3, :3], current_from_previous[:3, 3] = rotation, translation
    rigid(current_from_previous)
    previous_from_current = np.eye(4)
    previous_from_current[:3, :3] = rotation.T
    previous_from_current[:3, 3] = -rotation.T @ translation
    return rigid(previous_from_current), dict(report,
        objective='native dense normalized photometric and projective metric depth residuals',
        current_from_previous=current_from_previous.tolist(),
        previous_from_current=previous_from_current.tolist(), dense_settings=CONTROLS,
        initialization='identity per adjacent RGB-D pair; no reference read')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = validate_dense_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
        dense_rgbd_aligner=align_dense_rgbd, method=ESTIMATOR,
        additional_bindings=[__file__, Path(__file__).with_name('dense_rgbd_protocol.py'),
                             Path(__file__).with_name('wide_rgbd_protocol.py'),
                             Path(__file__).with_name('audit_tum_dense_rgbd.py'), PARENTS[plan['sequence']]])
    print('Frozen dense RGB-D {}/{} poses; estimates SHA-256 {}'.format(
        result['generated_poses'], result['planned_poses'], result['estimates_sha256']))


if __name__ == '__main__':
    main()
