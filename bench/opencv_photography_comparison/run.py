"""OpenCV correctness receipt for Epic 108 panorama composition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import emit_report, environment, make_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    height, width = 32, 64
    x = np.arange(width, dtype=np.uint8)[None, :]
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[..., 0] = x + 80
    source[..., 1] = 20
    target = np.zeros_like(source)
    target[..., 1] = 40
    target[..., 2] = 160
    homography = np.array([[1.0, 0.0, 24.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    panorama, origin_x, origin_y = sr.stitch_panorama_pair(source, target, homography)
    cv_warp = cv2.warpPerspective(source, homography, (width + 24, height))
    source_only_error = int(np.max(np.abs(
        panorama[:, width:, :].astype(np.int16) - cv_warp[:, width:, :].astype(np.int16)
    )))

    cast = np.zeros((2, 2, 3), dtype=np.uint8)
    cast[...] = [40, 80, 120]
    balanced = sr.gray_world_white_balance_image(cast)
    channel_means = balanced.mean(axis=(0, 1))
    balance_spread = float(channel_means.max() - channel_means.min())
    expected_shape = [height, width + 24, 3]
    status = "pass" if (
        list(panorama.shape) == expected_shape and origin_x == 0 and origin_y == 0
        and source_only_error == 0 and balance_spread <= 1.0
    ) else "fail"
    emit_report(make_report(
        suite="opencv-photography", kind="correctness", status=status,
        environment_receipt=environment(
            opencv_version=cv2.__version__, spatialrust_version=sr.__version__),
        results={
            "panorama_shape": list(panorama.shape),
            "origin": [origin_x, origin_y],
            "opencv_source_only_max_error": source_only_error,
            "balanced_channel_mean_spread": balance_spread,
            "thresholds": {"warp_max_error": 0, "channel_mean_spread": 1.0},
        },
    ), args.output)


if __name__ == "__main__":
    main()
