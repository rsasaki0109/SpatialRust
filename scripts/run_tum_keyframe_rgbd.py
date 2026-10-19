"""Track fixed keyframe depth landmarks with native sparse LK and metric PnP."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from evaluate_pose_reference import rigid
from run_tum_rgbd_odometry import DEFAULT_CONTROLS, validate_rgbd_plan
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256


ESTIMATOR = 'spatialrust_keyframe_rgbd_pnp_cpu'
PARENT_PLAN = Path(__file__).resolve().parents[1] / 'benchmarks/tum-fr1-xyz-rgbd-pnp-plan.json'
KEYFRAME_CONTROLS = dict(interval_frames=30, retention='pnp_inliers')


def validate_keyframe_rgbd_plan(plan):
    parent = json.loads(PARENT_PLAN.read_text(encoding='utf-8'))
    if plan.get('sequence') not in ('rgbd_dataset_freiburg1_xyz','rgbd_dataset_freiburg1_rpy'):
        raise ValueError('require one of the two fixed development sequences')
    if plan.get('estimator') != ESTIMATOR or plan.get('prior_rgbd_plan_sha256') != file_sha256(PARENT_PLAN):
        raise ValueError('require the fixed parent RGB-D protocol identity')
    expected = dict(parent['controls'], rgbd_keyframe=KEYFRAME_CONTROLS)
    if plan.get('controls') != expected or type(plan['controls']['rgbd_keyframe']['interval_frames']) is not int:
        raise ValueError('keyframe controls and original support limits are fixed')
    if any(type(plan['controls']['rgbd_odometry'][key]) is not type(value) for key, value in DEFAULT_CONTROLS.items()):
        raise ValueError('RGB-D control types must retain the fixed protocol')
    baseline = dict(plan, estimator=parent['estimator'], controls=parent['controls'])
    validate_rgbd_plan(baseline)
    return plan


def invert_rigid(pose):
    pose = rigid(pose)
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return rigid(inverse)


class KeyframeRgbdTracker:
    """Hold one source depth and original pixel IDs while tracking last-frame pixels.

    Small intervals support synthetic tests. The executable fixes the interval
    to 30 before fitting. A failed update loses the connected chain permanently.
    """
    def __init__(self, interval_frames=KEYFRAME_CONTROLS['interval_frames']):
        if type(interval_frames) is not int or not 1 <= interval_frames <= 1000:
            raise ValueError('require a bounded integer keyframe interval')
        self.interval_frames = interval_frames
        self.pose_ordinal = 0
        self.keyframe_ordinal = 0
        self.keyframe_depth = None
        self.keyframe_pixels = None
        self.previous_pixels = None
        self.previous_gray = None
        self.previous_from_keyframe = np.eye(4)
        self.lost = False

    def seed(self, depth, gray, camera, controls):
        settings = DEFAULT_CONTROLS
        keypoints = sr.shi_tomasi_keypoints(gray, max_corners=settings['max_features'],
            quality_level=settings['quality_level'], min_distance=settings['min_distance'],
            block_size=3, gradient_size=3)
        pixels = np.array([[p.x, p.y] for p in keypoints], dtype=np.float64).reshape(-1, 2)
        margin = settings['image_margin']
        inside = ((pixels[:, 0] >= margin) & (pixels[:, 0] < camera['width']-margin) &
                  (pixels[:, 1] >= margin) & (pixels[:, 1] < camera['height']-margin))
        pixels = pixels[inside]
        indices = np.floor(pixels + .5).astype(np.int64)
        z = depth[indices[:, 1], indices[:, 0]].astype(np.float64) * camera['depth_unit_m']
        pixels = np.ascontiguousarray(pixels[(z >= controls['min_depth_m']) & (z <= controls['max_depth_m'])])
        if len(pixels) < settings['min_tracks']:
            raise ValueError('keyframe has insufficient valid-depth features')
        metres = np.array(depth.astype(np.float64) * camera['depth_unit_m'], dtype=np.float32, copy=True)
        return metres, pixels

    def align(self, previous_depth, previous_gray, current_gray, camera, controls):
        if self.lost:
            raise ValueError('keyframe RGB-D tracking lost; resets forbidden')
        try:
            settings = controls.get('rgbd_odometry')
            if settings != DEFAULT_CONTROLS or controls.get('rgbd_keyframe') != KEYFRAME_CONTROLS:
                raise ValueError('keyframe RGB-D controls differ from the fixed protocol')
            if self.previous_gray is not None and previous_gray is not self.previous_gray:
                raise ValueError('previous image is not the last accepted frame')
            # Promote the previous frame because this explicit callback owns its
            # depth. No current/future depth or reference pose is requested.
            promote = self.keyframe_depth is None or self.pose_ordinal-self.keyframe_ordinal == self.interval_frames
            if promote:
                depth, originals = self.seed(previous_depth, previous_gray, camera, controls)
                tracked_pixels = originals
                previous_from_keyframe = np.eye(4)
                keyframe_ordinal = self.pose_ordinal
            else:
                depth, originals = self.keyframe_depth, self.keyframe_pixels
                tracked_pixels = self.previous_pixels
                previous_from_keyframe = self.previous_from_keyframe
                keyframe_ordinal = self.keyframe_ordinal
            if len(tracked_pixels) < settings['min_tracks']:
                raise ValueError('insufficient surviving keyframe tracks')
            following, valid, roundtrip = sr.track_points_forward_backward(
                previous_gray, current_gray, tracked_pixels,
                window_radius=settings['window_radius'], pyramid_levels=settings['pyramid_levels'],
                max_iterations=settings['max_iterations'], epsilon=settings['epsilon'],
                min_eigenvalue=settings['min_eigenvalue'],
                max_forward_backward_error=settings['max_forward_backward_error'])
            source, target = np.ascontiguousarray(originals[valid]), np.ascontiguousarray(following[valid])
            if len(source) < settings['min_tracks']:
                raise ValueError('insufficient accepted keyframe forward/backward tracks')
            rotation, translation, inliers, rejected = sr.estimate_rgbd_odometry(
                depth, source, target, camera['fx'], camera['fy'], camera['cx'], camera['cy'],
                depth_scale=1., threshold=settings['pnp_threshold'])
            count = int(np.count_nonzero(inliers))
            if rejected or len(inliers) != len(source) or count < settings['min_inliers'] or count/len(source) < settings['min_inlier_fraction']:
                raise ValueError('keyframe PnP lacks fixed inlier/depth support')
            current_from_keyframe = np.eye(4)
            current_from_keyframe[:3, :3] = rotation
            current_from_keyframe[:3, 3] = translation
            rigid(current_from_keyframe)
            previous_from_current = rigid(previous_from_keyframe @ invert_rigid(current_from_keyframe))
            ordinal = self.pose_ordinal + 1
            report = dict(objective='persistent keyframe depth/pixels, last-frame native LK and metric PnP',
                keyframe_pose_ordinal=keyframe_ordinal, current_pose_ordinal=ordinal,
                keyframe_promoted_from_previous=promote, keyframe_interval_frames=self.interval_frames,
                accepted_tracks=len(source), inlier_count=count, inlier_fraction=count/len(source),
                rejected_depth_count=rejected, max_accepted_roundtrip_px=float(np.max(roundtrip[valid])),
                source_pixels=source.tolist(), target_pixels=target.tolist(), inliers=inliers.tolist(),
                previous_from_keyframe=previous_from_keyframe.tolist(),
                current_from_keyframe=current_from_keyframe.tolist(), previous_from_current=previous_from_current.tolist(),
                settings=settings, keyframe_settings=KEYFRAME_CONTROLS)
            # Commit state after all native/rigid checks. The outer generator
            # also keeps the original full-cloud support and lost-chain gate.
            self.keyframe_depth, self.keyframe_ordinal = depth, keyframe_ordinal
            self.keyframe_pixels = np.ascontiguousarray(source[inliers])
            self.previous_pixels = np.ascontiguousarray(target[inliers])
            self.previous_from_keyframe = current_from_keyframe.copy()
            self.pose_ordinal, self.previous_gray = ordinal, current_gray
            return previous_from_current, report
        except (ValueError, RuntimeError, OSError):
            self.lost = True
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    validate_keyframe_rgbd_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    tracker = KeyframeRgbdTracker()
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                      rgbd_aligner=tracker.align, method=ESTIMATOR,
                      additional_bindings=[__file__, Path(__file__).with_name('run_tum_rgbd_odometry.py'), PARENT_PLAN])
    print('Frozen keyframe RGB-D {}/{} poses; estimates SHA-256 {}'.format(
        result['generated_poses'], result['planned_poses'], result['estimates_sha256']))


if __name__ == '__main__':
    main()
