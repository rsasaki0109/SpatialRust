"""Freeze full native Gaussian3 fitting timelines with original raw admission."""
import argparse
import copy
import importlib
import json
from pathlib import Path

import numpy as np

from evaluate_pose_reference import rigid
from gaussian_dense_rgbd_protocol import ESTIMATOR, POLICY, validate_gaussian_dense_plan
from run_tum_dense_rgbd import _align_dense_rgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, require_sha256
from wide_rgbd_protocol import exact_json


class PreviousMotionGaussianRgbd:
    """One owned preceding increment; no reference, image cache or reset."""

    def __init__(self):
        self._previous = np.eye(4)
        self._successful_pairs = 0

    def align(self, previous_depth, current_depth, previous_gray, current_gray, camera, controls):
        if exact_json(controls.get('dense_fitting_preprocessing')) != exact_json(POLICY):
            raise ValueError('Gaussian3 fitting policy is fixed')
        initial = self._previous.copy()
        inverse, report = _align_dense_rgbd(previous_depth,current_depth,previous_gray,
            current_gray,camera,controls,expected_final_support_only=True,initial=initial,
            preprocessing=POLICY['preprocessing'])
        candidate = rigid(report['current_from_previous']).copy()
        report.update(initial_seed_policy='previous-relative-increment',
            initial_previous_pair_index=self._successful_pairs if self._successful_pairs else None,
            initial_current_from_previous=initial.tolist(),
            objective='native Gaussian3 fitting; final diagnostics use original raw images',
            fitting_policy=copy.deepcopy(POLICY))
        self._previous = candidate
        self._successful_pairs += 1
        return inverse, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--native-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    validate_gaussian_dense_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    native=Path(importlib.import_module('spatialrust.spatialrust').__file__)
    if file_sha256(native)!=require_sha256(args.native_sha256):
        raise ValueError('require the separately frozen native Gaussian3 module')
    adapter=PreviousMotionGaussianRgbd()
    result=generate(args.prepared_dir,args.manifest_sha256,args.output_dir,
        dense_rgbd_aligner=adapter.align,method=ESTIMATOR,additional_bindings=[__file__,
            Path(__file__).with_name('run_tum_dense_rgbd.py'),
            Path(__file__).with_name('gaussian_dense_rgbd_protocol.py'),
            Path(__file__).with_name('audit_tum_dense_rgbd.py')])
    print('Frozen native Gaussian3 {}/{} poses; SHA-256 {}'.format(
        result['generated_poses'],result['planned_poses'],result['estimates_sha256']))


if __name__=='__main__':main()
