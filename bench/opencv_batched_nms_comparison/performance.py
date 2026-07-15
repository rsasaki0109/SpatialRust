"""Reproducible class-aware batched NMS comparison with OpenCV."""

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
    "multi_class_1000": (1_000, 20, 30),
    "yolo_8400": (8_400, 80, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = sorted(set(selected) - PROFILES.keys())
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not hasattr(cv2.dnn, "NMSBoxesBatched"):
        raise RuntimeError("OpenCV build does not expose dnn.NMSBoxesBatched")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    rng = np.random.default_rng(119)
    results: dict[str, object] = {}
    for profile in selected:
        count, class_count, repeats = PROFILES[profile]
        centers = rng.uniform(0.0, 640.0, size=(count, 2)).astype(np.float32)
        sizes = rng.uniform(5.0, 120.0, size=(count, 2)).astype(np.float32)
        boxes_xyxy = np.empty((count, 4), dtype=np.float32)
        boxes_xyxy[:, :2] = centers - sizes * 0.5
        boxes_xyxy[:, 2:] = centers + sizes * 0.5
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        scores = rng.random(count, dtype=np.float32)
        class_ids_cv = rng.integers(0, class_count, count, dtype=np.int32)
        class_ids_sr = class_ids_cv.astype(np.int64)

        def opencv_batched_nms() -> np.ndarray:
            return np.asarray(
                cv2.dnn.NMSBoxesBatched(
                    boxes_xywh, class_ids=class_ids_cv, scores=scores,
                    score_threshold=0.25, nms_threshold=0.5,
                )
            ).reshape(-1)

        def spatialrust_batched_nms() -> np.ndarray:
            return sr.batched_nms(boxes_xyxy, scores, class_ids_sr, 0.25, 0.5)

        expected = opencv_batched_nms().astype(np.int64, copy=False)
        actual = spatialrust_batched_nms()
        exact = bool(np.array_equal(expected, actual))
        if not exact:
            raise AssertionError(f"{profile} batched NMS index mismatch")

        _, _, opencv_timing, spatialrust_timing = timed_pair(
            opencv_batched_nms,
            spatialrust_batched_nms,
            warmup=args.warmup,
            repeats=repeats,
            seed=121,
            min_sample_time_ms=1.0,
        )
        opencv_ms = float(opencv_timing["median"])
        spatialrust_ms = float(spatialrust_timing["median"])
        results[profile] = {
            "box_count": count,
            "class_count": class_count,
            "kept_count": int(actual.size),
            "score_threshold": 0.25,
            "iou_threshold": 0.5,
            "indices_exact": exact,
            "opencv": opencv_timing,
            "spatialrust": spatialrust_timing,
            "spatialrust_speedup": opencv_ms / spatialrust_ms,
            "faster_implementation": (
                "spatialrust" if spatialrust_ms < opencv_ms else "opencv"
            ),
        }

    receipt = environment(
        opencv_version=cv2.__version__, spatialrust_version=sr.__version__
    )
    receipt["opencv_threads"] = cv2.getNumThreads()
    receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-batched-nms-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "Python API call returning globally score-ordered kept indices",
                "paired_interleaved": True,
                "input_seed": 119,
                "random_order_seed": 121,
                "minimum_sample_time_ms": 1.0,
                "box_format": {
                    "opencv": "xywh float32 NumPy array",
                    "spatialrust": "xyxy float32 NumPy array",
                },
                "class_id_format": {
                    "opencv": "int32 NumPy array",
                    "spatialrust": "int64 NumPy array",
                },
                "thread_policy": "library defaults; OpenCV thread count recorded",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
