"""Dense-flow correctness comparison for Epic 106 video recognition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import emit_report, environment, make_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    height, width = 96, 128
    dx, dy = 3.0, 2.0
    yy, xx = np.indices((height, width), dtype=np.int32)
    previous = ((xx * 17 + yy * 29 + xx * yy * 3) % 251).astype(np.uint8)
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    next_frame = cv2.warpAffine(
        previous,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    spatialrust_flow = sr.dense_flow_image(previous, next_frame, search_radius=4)
    opencv_flow = cv2.calcOpticalFlowFarneback(
        previous,
        next_frame,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=5,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    region = np.s_[20:-20, 20:-20]
    spatialrust_median = np.nanmedian(spatialrust_flow[region], axis=(0, 1))
    opencv_median = np.median(opencv_flow[region], axis=(0, 1))
    expected = np.array([dx, dy])
    spatialrust_error = float(np.max(np.abs(spatialrust_median - expected)))
    opencv_error = float(np.max(np.abs(opencv_median - expected)))
    cross_error = float(np.max(np.abs(spatialrust_median - opencv_median)))
    status = "pass" if spatialrust_error <= 0.01 and cross_error <= 0.5 else "fail"
    report = make_report(
        suite="opencv-video",
        kind="correctness",
        status=status,
        environment_receipt=environment(
            opencv_version=cv2.__version__, spatialrust_version=sr.__version__
        ),
        results={
            "translation": [dx, dy],
            "spatialrust_median_flow": spatialrust_median.tolist(),
            "opencv_median_flow": opencv_median.tolist(),
            "spatialrust_max_error": spatialrust_error,
            "opencv_max_error": opencv_error,
            "cross_max_error": cross_error,
            "thresholds": {"spatialrust_max_error": 0.01, "cross_max_error": 0.5},
        },
    )
    emit_report(report, args.output)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
