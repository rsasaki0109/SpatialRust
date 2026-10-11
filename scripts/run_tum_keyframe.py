"""Fixed-interval native keyframe tracking; no reference input or tracking reset."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from run_tum_point_to_plane import ESTIMATOR as PLANE_ESTIMATOR, align_point_to_plane
from run_tum_sequence import compose_camera_pose, generate
from tum_sequence_reference import file_sha256, validate_plan


ESTIMATOR = dict(PLANE_ESTIMATOR, method='spatialrust_keyframe_point_to_plane_cpu',
                 keyframe_interval_frames=30,
                 initialization='previous_estimated_pose_in_active_keyframe',
                 promotion='fixed_successful_pose_interval',
                 tracking_reset=False)
ACCEPTANCE = dict(minimum_generated_fraction=1., minimum_evaluated_fraction=.95,
                  maximum_first_pose_ate_rmse_m=.1,
                  maximum_one_second_translation_rpe_rmse_m=.05)


def validate_keyframe_plan(plan, original):
    validate_plan(plan)
    if plan.get('prior_native_plan_sha256') != file_sha256(original):
        raise ValueError('keyframe plan must bind the original native protocol')
    native = json.loads(Path(original).read_bytes())
    if any(plan[key] != native[key] for key in ('calibration', 'controls', 'limits')):
        raise ValueError('keyframe geometry/frame/budget must match original protocol')
    if plan.get('estimator') != ESTIMATOR:
        raise ValueError('unsupported fixed keyframe estimator settings')
    acceptance = plan.get('acceptance')
    if not isinstance(acceptance, dict) or any(acceptance.get(k) != v for k, v in ACCEPTANCE.items()):
        raise ValueError('keyframe accuracy/coverage limits are fixed before execution')
    return plan


class KeyframeTracker:
    """Register to one retained keyframe, returning current-to-previous deltas.

    Ordinals count successful poses from the sole identity anchor, not image
    indices. An initial decode failure therefore cannot mislabel a keyframe.
    The smaller intervals accepted by this helper support analytic tests; the
    executable study validates and always uses the fixed 30-frame protocol.
    """
    def __init__(self, interval_frames=ESTIMATOR['keyframe_interval_frames']):
        if type(interval_frames) is not int or not 1 <= interval_frames <= 1000:
            raise ValueError('require a bounded integer keyframe interval')
        self.interval_frames = interval_frames
        self.keyframe = None
        self.previous = None
        self.keyframe_from_previous = np.eye(4)
        self.keyframe_ordinal = 0
        self.pose_ordinal = 0
        self.lost = False

    def align_pair(self, current, previous, controls):
        if self.lost:
            raise ValueError('keyframe tracking was lost; resets forbidden')
        try:
            if self.keyframe is None:
                self.keyframe, self.previous = previous, previous
            if previous is not self.previous:
                raise ValueError('previous frame is not the last accepted cloud')
            initial = self.keyframe_from_previous.copy()
            # Zero displacement from the previously estimated pose. No velocity,
            # future frame, reference pose or optimized world gauge is supplied.
            moved = sr.apply_transform(current, initial.astype(np.float32))
            correction, report = align_point_to_plane(moved, self.keyframe, controls)
            keyframe_from_current = compose_camera_pose(correction, initial)
            previous_from_current = compose_camera_pose(np.linalg.inv(initial), keyframe_from_current)
            ordinal = self.pose_ordinal + 1
            promote = ordinal - self.keyframe_ordinal == self.interval_frames
            stages = [{**{k: v for k, v in s.items() if k != 'previous_from_current'},
                       'keyframe_correction': s['previous_from_current']} for s in report['stages']]
            report = dict(report, stages=stages, registration_target='active_keyframe',
                          keyframe_pose_ordinal=self.keyframe_ordinal,
                          current_pose_ordinal=ordinal, keyframe_promoted=promote,
                          keyframe_interval_frames=self.interval_frames,
                          initial_keyframe_from_current=initial.tolist(),
                          keyframe_from_current=keyframe_from_current.tolist(),
                          previous_from_current=previous_from_current.tolist())
            # Publish state only after geometric support and both rigid transforms
            # validate. A failed update cannot promote a keyframe or reset motion.
            self.pose_ordinal, self.previous = ordinal, current
            if promote:
                self.keyframe, self.keyframe_ordinal = current, ordinal
                self.keyframe_from_previous = np.eye(4)
            else:
                self.keyframe_from_previous = keyframe_from_current
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
    original = Path(__file__).resolve().parents[1]/'benchmarks/tum-fr1-xyz-plan.json'
    validate_keyframe_plan(json.loads((args.prepared_dir/'plan.json').read_bytes()), original)
    native = list(Path(sr.__file__).parent.glob('*.so')) + list(Path(sr.__file__).parent.glob('*.pyd'))
    if len(native) != 1:
        raise ValueError('require one installed native extension')
    tracker = KeyframeTracker()
    receipt = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                       pair_aligner=tracker.align_pair, method=ESTIMATOR['method'],
                       additional_bindings=(Path(__file__).with_name('run_tum_point_to_plane.py'), native[0]))
    print(json.dumps(dict(method=receipt['method'], planned=receipt['planned_poses'],
                          generated=receipt['generated_poses'], estimates_sha256=receipt['estimates_sha256'])))


if __name__ == '__main__':
    main()
