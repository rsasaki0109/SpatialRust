"""Input-only overlap-margin keyframe promotion, without fallback or reset."""
import argparse
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from run_tum_keyframe import ESTIMATOR as FIXED_ESTIMATOR, KeyframeTracker, validate_keyframe_plan
from run_tum_sequence import generate


ESTIMATOR = dict(FIXED_ESTIMATOR, method='spatialrust_overlap_keyframe_point_to_plane_cpu',
                 promotion='fixed_interval_or_low_geometric_overlap',
                 promotion_support_fraction=.75)


def validate_overlap_plan(plan, original):
    if plan.get('estimator') != ESTIMATOR:
        raise ValueError('unsupported fixed overlap-promotion settings')
    validate_keyframe_plan(dict(plan, estimator=FIXED_ESTIMATOR), original)
    return plan


class OverlapKeyframeTracker:
    """Promote only an already accepted pose; never retry an unsupported pose."""
    def __init__(self):
        self.tracker = KeyframeTracker()

    def align_pair(self, current, previous, controls):
        delta, report = self.tracker.align_pair(current, previous, controls)
        cause = 'interval' if report['keyframe_promoted'] else 'retained'
        if not report['keyframe_promoted'] and report['support_fraction'] < ESTIMATOR['promotion_support_fraction']:
            # The base tracker has already checked the unchanged 50% acceptance
            # criterion. Promotion reserves overlap for the next image, without
            # looking ahead, changing the world gauge or inventing a valid pose.
            self.tracker.keyframe = current
            self.tracker.keyframe_ordinal = self.tracker.pose_ordinal
            self.tracker.keyframe_from_previous = np.eye(4)
            report = dict(report, keyframe_promoted=True)
            cause = 'geometric_overlap_margin'
        return delta, dict(report, keyframe_promotion_cause=cause,
                           promotion_support_fraction=ESTIMATOR['promotion_support_fraction'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original = Path(__file__).resolve().parents[1]/'benchmarks/tum-fr1-xyz-plan.json'
    validate_overlap_plan(json.loads((args.prepared_dir/'plan.json').read_bytes()), original)
    native = list(Path(sr.__file__).parent.glob('*.so')) + list(Path(sr.__file__).parent.glob('*.pyd'))
    if len(native) != 1:
        raise ValueError('require one installed native extension')
    tracker = OverlapKeyframeTracker()
    receipt = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
                       pair_aligner=tracker.align_pair, method=ESTIMATOR['method'],
                       additional_bindings=(Path(__file__).with_name('run_tum_point_to_plane.py'),
                                            Path(__file__).with_name('run_tum_keyframe.py'), native[0]))
    print(json.dumps(dict(method=receipt['method'], planned=receipt['planned_poses'],
                          generated=receipt['generated_poses'], estimates_sha256=receipt['estimates_sha256'])))


if __name__ == '__main__':
    main()
