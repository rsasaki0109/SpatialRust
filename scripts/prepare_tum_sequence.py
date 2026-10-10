"""Prepare every RGB/depth frame from one hash-bound official TUM sequence."""
import argparse
from pathlib import Path

from tum_sequence_reference import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.archive, args.archive_sha256, args.plan, args.output_dir)
    print('Prepared {} depth / {} RGB frames; {} depth frames have no RGB association'.format(
        result['depth_frames'], result['rgb_frames'], result['rgb_unmatched_depth_frames']))


if __name__ == '__main__':
    main()
