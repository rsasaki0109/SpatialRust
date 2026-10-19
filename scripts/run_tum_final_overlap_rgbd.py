"""Freeze the explicitly revised final-acceptance dense RGB-D solver."""
import argparse
import json
from pathlib import Path

from dense_rgbd_protocol import FINAL_ESTIMATOR, validate_final_overlap_plan
from run_tum_dense_rgbd import _align_dense_rgbd
from run_tum_sequence import generate


def align_final_overlap_rgbd(previous_depth, current_depth, previous_gray, current_gray, camera, controls):
    return _align_dense_rgbd(previous_depth, current_depth, previous_gray, current_gray,
        camera, controls, expected_final_support_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    validate_final_overlap_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
        dense_rgbd_aligner=align_final_overlap_rgbd, method=FINAL_ESTIMATOR,
        additional_bindings=[__file__, Path(__file__).with_name('run_tum_dense_rgbd.py'),
            Path(__file__).with_name('dense_rgbd_protocol.py'),
            Path(__file__).with_name('wide_rgbd_protocol.py'),
            Path(__file__).with_name('audit_tum_dense_rgbd.py')])
    print('Frozen final-acceptance dense RGB-D {}/{} poses; estimates SHA-256 {}'.format(
        result['generated_poses'], result['planned_poses'], result['estimates_sha256']))


if __name__ == '__main__':
    main()
