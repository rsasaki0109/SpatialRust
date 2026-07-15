"""Reproducible structured-mask connected-components comparison with OpenCV."""

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
    "vga": (640, 480, 40),
    "1080p": (1920, 1080, 30),
    "4k": (3840, 2160, 20),
}
PATTERNS = ("segmentation_blobs", "document_lines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--patterns", default=",".join(PATTERNS))
    parser.add_argument("--warmup", type=int, default=8)
    return parser.parse_args()


def make_mask(width: int, height: int, pattern: str) -> np.ndarray:
    x = np.arange(width, dtype=np.int32)[None, :]
    y = np.arange(height, dtype=np.int32)[:, None]
    if pattern == "segmentation_blobs":
        foreground = (x % 97 < 23) & (y % 83 < 19)
    elif pattern == "document_lines":
        foreground = (y % 32 < 3) & (x % 211 > 8) & (x % 211 < 190)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return foreground.astype(np.uint8) * np.uint8(255)


def assert_sauf_compatible(
    mask: np.ndarray, connectivity: int, context: str
) -> tuple[int, np.ndarray, np.ndarray]:
    count, expected_labels, expected_stats, _ = (
        cv2.connectedComponentsWithStatsWithAlgorithm(
            mask, connectivity, cv2.CV_32S, cv2.CCL_SAUF
        )
    )
    actual_labels, actual_stats = sr.connected_components_image(
        mask, connectivity=connectivity
    )
    labels_exact = bool(np.array_equal(actual_labels, expected_labels))
    actual_areas = np.asarray([value[1] for value in actual_stats], dtype=np.int64)
    expected_areas = expected_stats[1:count, cv2.CC_STAT_AREA].astype(np.int64)
    areas_exact = bool(np.array_equal(actual_areas, expected_areas))
    actual_boxes = np.asarray([value[2] for value in actual_stats], dtype=np.float64)
    expected_boxes = expected_stats[1:count, :4].astype(np.float64)
    expected_boxes[:, 2] += expected_boxes[:, 0]
    expected_boxes[:, 3] += expected_boxes[:, 1]
    boxes_exact = bool(np.array_equal(actual_boxes, expected_boxes))
    if not labels_exact or not areas_exact or not boxes_exact:
        raise AssertionError(
            f"{context} mismatch: labels={labels_exact}, "
            f"areas={areas_exact}, boxes={boxes_exact}"
        )
    return count, expected_labels, expected_stats


def validate_randomized_cases(cases_per_connectivity: int = 160) -> int:
    rng = np.random.default_rng(20_260_715)
    checked = 0
    for connectivity in (4, 8):
        for case in range(cases_per_connectivity):
            height = int(rng.integers(1, 90))
            width = int(rng.integers(1, 120))
            if case % 4 == 0:
                density = float(rng.uniform(0.01, 0.8))
                mask = (rng.random((height, width)) < density).astype(np.uint8) * 255
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                for _ in range(int(rng.integers(1, 25))):
                    y0 = int(rng.integers(height))
                    y1 = int(rng.integers(y0 + 1, height + 1))
                    x0 = int(rng.integers(width))
                    x1 = int(rng.integers(x0 + 1, width + 1))
                    mask[y0:y1, x0:x1] = int(rng.integers(1, 256))
            assert_sauf_compatible(mask, connectivity, f"random/{connectivity}/{case}")
            checked += 1
    return checked


def main() -> None:
    args = parse_args()
    profiles = [name.strip() for name in args.profiles.split(",") if name.strip()]
    patterns = [name.strip() for name in args.patterns.split(",") if name.strip()]
    unknown_profiles = sorted(set(profiles) - PROFILES.keys())
    unknown_patterns = sorted(set(patterns) - set(PATTERNS))
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {', '.join(unknown_profiles)}")
    if unknown_patterns:
        raise ValueError(f"unknown patterns: {', '.join(unknown_patterns)}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not hasattr(cv2, "connectedComponentsWithStatsWithAlgorithm"):
        raise RuntimeError("OpenCV build does not expose explicit CCL algorithms")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    randomized_cases = validate_randomized_cases()

    results: dict[str, object] = {}
    for profile in profiles:
        width, height, repeats = PROFILES[profile]
        profile_results: dict[str, object] = {}
        for pattern in patterns:
            mask = make_mask(width, height, pattern)

            def opencv_components() -> tuple[object, ...]:
                return cv2.connectedComponentsWithStatsWithAlgorithm(
                    mask, 8, cv2.CV_32S, cv2.CCL_SAUF
                )

            def spatialrust_components() -> tuple[object, ...]:
                return sr.connected_components_image(mask, connectivity=8)

            count, _, _ = assert_sauf_compatible(mask, 8, f"{profile}/{pattern}")
            labels_exact = True
            areas_exact = True
            boxes_exact = True

            _, _, opencv_timing, spatialrust_timing = timed_pair(
                opencv_components,
                spatialrust_components,
                warmup=args.warmup,
                repeats=repeats,
                seed=173,
                min_sample_time_ms=20.0,
            )
            opencv_ms = float(opencv_timing["median"])
            spatialrust_ms = float(spatialrust_timing["median"])
            profile_results[pattern] = {
                "width": width,
                "height": height,
                "connectivity": 8,
                "component_count": int(count - 1),
                "labels_exact": labels_exact,
                "areas_exact": areas_exact,
                "bounding_boxes_exact": boxes_exact,
                "opencv": opencv_timing,
                "spatialrust": spatialrust_timing,
                "spatialrust_speedup": opencv_ms / spatialrust_ms,
                "faster_implementation": (
                    "spatialrust" if spatialrust_ms < opencv_ms else "opencv"
                ),
            }
        results[profile] = profile_results

    receipt = environment(
        opencv_version=cv2.__version__, spatialrust_version=sr.__version__
    )
    receipt["opencv_threads"] = cv2.getNumThreads()
    receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-connected-components-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "Python API call returning labels and foreground statistics",
                "paired_interleaved": True,
                "random_order_seed": 173,
                "minimum_sample_time_ms": 20.0,
                "opencv_algorithm": "CCL_SAUF",
                "mask_values": "packed uint8; 0 background, 255 foreground",
                "randomized_correctness_seed": 20_260_715,
                "randomized_correctness_cases": randomized_cases,
                "thread_policy": "library defaults; OpenCV thread count recorded",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
