"""OpenCV correctness receipt for Epic 107 metric RGB-D odometry."""

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
    width, height = 160, 120
    fx, fy, cx, cy = 180.0, 175.0, 80.0, 60.0
    camera = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    depth = np.full((height, width), np.nan, dtype=np.float32)
    source, target, objects = [], [], []
    expected = np.array([0.08, -0.025, 0.04], dtype=np.float64)
    for index, (x, y) in enumerate(
        (x, y) for y in range(20, 101, 16) for x in range(24, 137, 16)
    ):
        z = 1.3 + 0.025 * index
        point = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z])
        moved = point + expected
        source.append((x, y))
        target.append((fx * moved[0] / moved[2] + cx, fy * moved[1] / moved[2] + cy))
        objects.append(point)
        depth[y, x] = z
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    objects = np.asarray(objects, dtype=np.float64)
    source = np.vstack((source, [[5.0, 5.0]]))
    target = np.vstack((target, [[5.0, 5.0]]))

    sr_rotation, sr_translation, sr_inliers, rejected = sr.estimate_rgbd_odometry(
        depth, source, target, fx, fy, cx, cy, threshold=0.1
    )
    ok, rvec, cv_translation, cv_inliers = cv2.solvePnPRansac(
        objects, target[:-1], camera, None, iterationsCount=2000,
        reprojectionError=0.1, confidence=0.99, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        raise RuntimeError("OpenCV solvePnPRansac failed")
    cv_rotation, _ = cv2.Rodrigues(rvec)
    translation_error = float(np.max(np.abs(sr_translation - expected)))
    cross_translation_error = float(np.max(np.abs(sr_translation - cv_translation[:, 0])))
    cross_rotation_error = float(np.max(np.abs(sr_rotation - cv_rotation)))
    status = "pass" if (
        translation_error <= 1e-5 and cross_translation_error <= 1e-5
        and cross_rotation_error <= 1e-5 and rejected == 1
        and int(np.count_nonzero(sr_inliers)) == len(objects)
    ) else "fail"
    emit_report(make_report(
        suite="opencv-odometry", kind="correctness", status=status,
        environment_receipt=environment(
            opencv_version=cv2.__version__, spatialrust_version=sr.__version__),
        results={
            "translation_max_error": translation_error,
            "cross_translation_max_error": cross_translation_error,
            "cross_rotation_max_error": cross_rotation_error,
            "spatialrust_inliers": int(np.count_nonzero(sr_inliers)),
            "opencv_inliers": int(len(cv_inliers)),
            "rejected_depth_count": rejected,
            "thresholds": {"pose_max_error": 1e-5},
        },
    ), args.output)


if __name__ == "__main__":
    main()
