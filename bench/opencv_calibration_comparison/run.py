"""OpenCV correctness comparison for Epic 105 camera calibration contracts."""

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
    camera_points = np.array(
        [
            [x * 0.07, y * 0.06, 1.5 + 0.025 * abs(x + y)]
            for y in range(-4, 5)
            for x in range(-5, 6)
        ],
        dtype=np.float64,
    )
    expected_intrinsics = np.array([720.0, 715.0, 640.0, 360.0])
    camera_matrix = np.array(
        [[720.0, 0.0, 640.0], [0.0, 715.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pixels, _ = cv2.projectPoints(
        camera_points, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(5)
    )
    pinhole = sr.calibrate_pinhole_camera(
        camera_points, pixels[:, 0, :], 1280, 720
    )
    intrinsics_error = float(
        np.max(np.abs(np.asarray(pinhole[:4]) - expected_intrinsics))
    )

    expected_fisheye = np.array([0.025, -0.003, 0.0004, -0.00002])
    theta = np.linspace(0.06, 1.2, 20, dtype=np.float64)
    rays = np.column_stack((np.tan(theta), np.zeros_like(theta), np.ones_like(theta)))
    fisheye_pixels, _ = cv2.fisheye.projectPoints(
        rays[:, None, :],
        np.zeros(3),
        np.zeros(3),
        np.eye(3, dtype=np.float64),
        expected_fisheye,
    )
    distorted_radius = fisheye_pixels[:, 0, 0]
    fisheye = sr.calibrate_fisheye_angles(theta, distorted_radius)
    fisheye_error = float(
        np.max(np.abs(np.asarray(fisheye[:4]) - expected_fisheye))
    )
    status = "pass" if intrinsics_error <= 1e-8 and fisheye_error <= 1e-8 else "fail"
    report = make_report(
        suite="opencv-calibration",
        kind="correctness",
        status=status,
        environment_receipt=environment(
            opencv_version=cv2.__version__, spatialrust_version=sr.__version__
        ),
        results={
            "pinhole_max_parameter_error": intrinsics_error,
            "pinhole_rms_pixels": pinhole[4],
            "fisheye_max_coefficient_error": fisheye_error,
            "fisheye_rms_normalized_radius": fisheye[4],
            "thresholds": {
                "pinhole_max_parameter_error": 1e-8,
                "fisheye_max_coefficient_error": 1e-8,
            },
        },
    )
    emit_report(report, args.output)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
