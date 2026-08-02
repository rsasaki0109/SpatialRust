"""Reproducible linear and Gaussian Soft-NMS comparison with OpenCV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import emit_report, environment, make_report, timed_pair


PROFILES = {
    "small_100": (100, 50),
    "medium_1000": (1_000, 20),
    "yolo_8400": (8_400, 8),
}
METHODS = {
    "linear": cv2.dnn.SOFT_NMSMETHOD_SOFTNMS_LINEAR,
    "gaussian": cv2.dnn.SOFT_NMSMETHOD_SOFTNMS_GAUSSIAN,
}
SCORE_TOLERANCE = 4.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_profiles = [name.strip() for name in args.profiles.split(",") if name.strip()]
    selected_methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown_profiles = sorted(set(selected_profiles) - PROFILES.keys())
    unknown_methods = sorted(set(selected_methods) - METHODS.keys())
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {', '.join(unknown_profiles)}")
    if unknown_methods:
        raise ValueError(f"unknown methods: {', '.join(unknown_methods)}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not hasattr(cv2.dnn, "softNMSBoxes"):
        raise RuntimeError("OpenCV build does not expose dnn.softNMSBoxes")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    rng = np.random.default_rng(129)
    results: dict[str, object] = {}
    for profile in selected_profiles:
        count, repeats = PROFILES[profile]
        origins = rng.integers(0, 600, size=(count, 2), dtype=np.int32)
        sizes = rng.integers(5, 100, size=(count, 2), dtype=np.int32)
        boxes_xywh = np.column_stack((origins, sizes)).astype(np.int32)
        boxes_xyxy = np.column_stack((origins, origins + sizes)).astype(np.float32)
        scores = rng.random(count, dtype=np.float32)
        profile_results: dict[str, object] = {}

        for method_name in selected_methods:
            opencv_method = METHODS[method_name]

            def opencv_soft_nms() -> tuple[np.ndarray, np.ndarray]:
                return cv2.dnn.softNMSBoxes(
                    boxes_xywh,
                    scores,
                    0.25,
                    0.5,
                    0,
                    0.5,
                    opencv_method,
                )

            def spatialrust_soft_nms() -> tuple[list[int], list[float]]:
                return sr.soft_nms(boxes_xyxy, scores, 0.25, 0.5, method_name, 0.5)

            expected_scores, expected_indices = opencv_soft_nms()
            actual_indices, actual_scores = spatialrust_soft_nms()
            actual_indices_array = np.asarray(actual_indices, dtype=np.int64)
            actual_scores_array = np.asarray(actual_scores, dtype=np.float32)
            indices_exact = bool(np.array_equal(expected_indices, actual_indices_array))
            score_max_error = float(
                np.max(np.abs(expected_scores - actual_scores_array))
            ) if expected_scores.size else 0.0
            if not indices_exact:
                raise AssertionError(f"{profile}/{method_name} Soft-NMS index mismatch")
            if score_max_error > SCORE_TOLERANCE:
                raise AssertionError(
                    f"{profile}/{method_name} score error {score_max_error} "
                    f"> {SCORE_TOLERANCE}"
                )

            _, _, opencv_timing, spatialrust_timing = timed_pair(
                opencv_soft_nms,
                spatialrust_soft_nms,
                warmup=args.warmup,
                repeats=repeats,
                seed=131,
                min_sample_time_ms=5.0,
            )
            opencv_ms = float(opencv_timing["median"])
            spatialrust_ms = float(spatialrust_timing["median"])
            profile_results[method_name] = {
                "kept_count": len(actual_indices),
                "indices_exact": indices_exact,
                "score_max_absolute_error": score_max_error,
                "score_tolerance": SCORE_TOLERANCE,
                "opencv": opencv_timing,
                "spatialrust": spatialrust_timing,
                "spatialrust_speedup": opencv_ms / spatialrust_ms,
                "faster_implementation": (
                    "spatialrust" if spatialrust_ms < opencv_ms else "opencv"
                ),
            }
        results[profile] = {
            "box_count": count,
            "score_threshold": 0.25,
            "iou_threshold": 0.5,
            "sigma": 0.5,
            "methods": profile_results,
        }

    receipt = environment(
        opencv_version=cv2.__version__, spatialrust_version=sr.__version__
    )
    receipt["opencv_threads"] = cv2.getNumThreads()
    receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-soft-nms-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "Python API call returning updated scores and kept indices",
                "paired_interleaved": True,
                "input_seed": 129,
                "random_order_seed": 131,
                "minimum_sample_time_ms": 5.0,
                "box_format": {
                    "opencv": "xywh int32 NumPy array",
                    "spatialrust": "corresponding xyxy float32 NumPy array",
                },
                "thread_policy": "library defaults; OpenCV thread count recorded",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
