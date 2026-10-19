"""Use locally stable metric depth landmarks and input-count keyframe promotion."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_tum_keyframe_rgbd import KeyframeRgbdTracker, validate_keyframe_rgbd_plan
from run_tum_rgbd_odometry import DEFAULT_CONTROLS
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256
import spatialrust as sr


ESTIMATOR = 'spatialrust_adaptive_keyframe_rgbd_pnp_cpu'
CONTROLS = dict(maximum_candidate_features=900, maximum_retained_features=300,
                depth_patch_radius=1, absolute_deviation_depth_counts=100,
                relative_deviation_denominator=50, minimum_retained_tracks=60)
PLAN_ROOT = Path(__file__).resolve().parents[1] / 'benchmarks'
FIXED_PLANS = {name: PLAN_ROOT / ('tum-fr1-' + short + '-keyframe-rgbd-pnp-plan.json')
               for name,short in [('rgbd_dataset_freiburg1_xyz','xyz'),('rgbd_dataset_freiburg1_rpy','rpy')]}


def validate_adaptive_rgbd_plan(plan):
    path = FIXED_PLANS.get(plan.get('sequence'))
    if path is None or plan.get('estimator') != ESTIMATOR or plan.get('prior_fixed_keyframe_plan_sha256') != file_sha256(path):
        raise ValueError('require the fixed scene-specific keyframe protocol identity')
    fixed = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(fixed['controls'], rgbd_adaptive_keyframe=CONTROLS)
    if plan.get('controls') != expected or any(type(plan['controls']['rgbd_adaptive_keyframe'][key]) is not int for key in CONTROLS):
        raise ValueError('adaptive input-quality and original support controls are fixed')
    validate_keyframe_rgbd_plan(dict(plan, estimator=fixed['estimator'], controls=fixed['controls']))
    return plan


class AdaptiveRgbdTracker(KeyframeRgbdTracker):
    def __init__(self):
        super().__init__()
        self.seed_statistics = {}

    def seed(self, depth, gray, camera, controls):
        if camera['depth_unit_m'] != .0002 or controls['min_depth_m'] != .1 or controls['max_depth_m'] != 5.:
            raise ValueError('depth consensus requires the fixed registered-depth count convention')
        points = sr.shi_tomasi_keypoints(gray, max_corners=CONTROLS['maximum_candidate_features'],
            quality_level=DEFAULT_CONTROLS['quality_level'], min_distance=DEFAULT_CONTROLS['min_distance'],
            block_size=3, gradient_size=3)
        pixels = np.array([[p.x,p.y] for p in points], dtype=np.float64).reshape(-1,2)
        margin = DEFAULT_CONTROLS['image_margin']
        inside = ((pixels[:,0]>=margin) & (pixels[:,0]<camera['width']-margin) &
                  (pixels[:,1]>=margin) & (pixels[:,1]<camera['height']-margin))
        pixels = pixels[inside]
        if not len(pixels):
            raise ValueError('no interior depth-consensus keypoints')
        indices = np.floor(pixels+.5).astype(np.int64)
        patch = np.stack([depth[indices[:,1]+dy,indices[:,0]+dx]
                          for dy in (-1,0,1) for dx in (-1,0,1)],axis=1).astype(np.int64)
        # Nine integer samples have an integer median. Compare deviation in
        # counts with exact products, before the single float32 metric cast.
        median = np.sort(patch,axis=1)[:,4]
        deviation = np.max(np.abs(patch-median[:,None]),axis=1)
        divisor = CONTROLS['relative_deviation_denominator']
        admitted = np.all((patch>=500) & (patch<=25000),axis=1)
        admitted &= deviation*divisor <= np.maximum(CONTROLS['absolute_deviation_depth_counts']*divisor,median)
        selected = np.flatnonzero(admitted)[:CONTROLS['maximum_retained_features']]
        self.seed_statistics = dict(detected_candidates=len(points), interior_candidates=len(pixels),
                                    depth_consistent_candidates=int(admitted.sum()), retained_features=len(selected))
        if len(selected)<DEFAULT_CONTROLS['min_tracks']:
            raise ValueError('only {} stable-depth keypoints; require {}'.format(len(selected),DEFAULT_CONTROLS['min_tracks']))
        metres = np.array(depth.astype(np.float64)*camera['depth_unit_m'],dtype=np.float32,copy=True)
        chosen = indices[selected]
        metres[chosen[:,1],chosen[:,0]] = (median[selected].astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
        return metres,np.ascontiguousarray(pixels[selected])

    def align(self, previous_depth, previous_gray, current_gray, camera, controls):
        if self.lost:
            return super().align(previous_depth,previous_gray,current_gray,camera,controls)
        if controls.get('rgbd_adaptive_keyframe') != CONTROLS:
            self.lost = True
            raise ValueError('adaptive keyframe settings differ from the fixed protocol')
        retained = 0 if self.previous_pixels is None else len(self.previous_pixels)
        early = (self.keyframe_depth is not None and retained<CONTROLS['minimum_retained_tracks']
                 and self.pose_ordinal-self.keyframe_ordinal<self.interval_frames)
        saved_depth = self.keyframe_depth
        if early:
            # Request promotion from the last accepted frame. Parent state is
            # committed only after all native/rigid checks; keep world chaining.
            self.keyframe_depth = None
        try:
            transform,report = super().align(previous_depth,previous_gray,current_gray,camera,controls)
        except (ValueError,RuntimeError,OSError):
            if early:self.keyframe_depth = saved_depth
            raise
        report.update(adaptive_settings=CONTROLS, input_retained_tracks=retained,
                      early_keyframe_promotion=early, source_depth_consensus=dict(self.seed_statistics))
        return transform,report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    plan = validate_adaptive_rgbd_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    tracker = AdaptiveRgbdTracker()
    result = generate(args.prepared_dir,args.manifest_sha256,args.output_dir,
        rgbd_aligner=tracker.align,method=ESTIMATOR,additional_bindings=[__file__,
        Path(__file__).with_name('run_tum_keyframe_rgbd.py'),Path(__file__).with_name('run_tum_rgbd_odometry.py'),
        FIXED_PLANS[plan['sequence']],PLAN_ROOT/'tum-fr1-xyz-rgbd-pnp-plan.json',
        PLAN_ROOT/'tum-fr1-xyz-plan.json'])
    print('Frozen adaptive RGB-D {}/{} poses; estimates SHA-256 {}'.format(result['generated_poses'],result['planned_poses'],result['estimates_sha256']))


if __name__ == '__main__':
    main()
