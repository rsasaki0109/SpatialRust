"""Freeze dense poses with an explicit previous-increment seed and no resets."""
import argparse
import json
from pathlib import Path

import numpy as np

from causal_dense_rgbd_protocol import ESTIMATOR, validate_causal_dense_plan
from evaluate_pose_reference import rigid
from run_tum_dense_rgbd import _align_dense_rgbd
from run_tum_sequence import generate


class PreviousMotionDenseRgbd:
    """One owned 4x4 seed per run; no images, reference data or pose history."""

    def __init__(self):
        self._previous = np.eye(4)
        self._successful_pairs = 0

    def align(self, previous_depth, current_depth, previous_gray, current_gray, camera, controls):
        initial = self._previous.copy()
        inverse, report = _align_dense_rgbd(previous_depth, current_depth, previous_gray,
            current_gray, camera, controls, expected_final_support_only=True, initial=initial)
        candidate = rigid(report['current_from_previous']).copy()
        report.update(initialization='previous returned relative increment; first pair identity; no rescaling/reference/reset',
            initial_seed_policy='previous-relative-increment',
            initial_previous_pair_index=self._successful_pairs if self._successful_pairs else None,
            initial_current_from_previous=initial.tolist())
        # The generator permanently stops on any later full-cloud/world/IO failure.
        # Thus an unaccepted chain increment cannot seed another generator call.
        self._previous = candidate
        self._successful_pairs += 1
        return inverse, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    validate_causal_dense_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    aligner = PreviousMotionDenseRgbd()
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
        dense_rgbd_aligner=aligner.align, method=ESTIMATOR,
        additional_bindings=[__file__, Path(__file__).with_name('run_tum_dense_rgbd.py'),
            Path(__file__).with_name('causal_dense_rgbd_protocol.py'),
            Path(__file__).with_name('dense_rgbd_protocol.py'),
            Path(__file__).with_name('wide_rgbd_protocol.py'),
            Path(__file__).with_name('audit_tum_dense_rgbd.py')])
    print('Frozen causal dense RGB-D {}/{} poses; estimates SHA-256 {}'.format(
        result['generated_poses'], result['planned_poses'], result['estimates_sha256']))


if __name__ == '__main__':
    main()
