"""Reproducible fused Sobel L1 magnitude comparison with OpenCV."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import emit_report, environment, make_report, timed_pair


PROFILES = {
    "1080p": (1920, 1080, 20),
    "4k": (3840, 2160, 14),
    "8k": (7680, 4320, 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def opencv_l1_allocate(image: np.ndarray, zero: np.ndarray) -> np.ndarray:
    gradient_x, gradient_y = cv2.spatialGradient(
        image, ksize=3, borderType=cv2.BORDER_REFLECT_101
    )
    absolute_x = cv2.absdiff(gradient_x, zero)
    absolute_y = cv2.absdiff(gradient_y, zero)
    return cv2.add(absolute_x, absolute_y)


def validate_randomized_cases() -> int:
    rng = np.random.default_rng(116)
    checked = 0
    for case in range(300):
        height = int(rng.integers(1, 120))
        width = int(rng.integers(1, 160))
        image = rng.integers(0, 256, (height, width), dtype=np.uint8)
        if case % 3 == 0:
            image = image[:, ::-1]
        packed = np.ascontiguousarray(image)
        zero = np.zeros((height, width), dtype=np.int16)
        expected = opencv_l1_allocate(packed, zero)
        actual = sr.sobel_l1_magnitude_image(image)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"random case {case} is not bit-exact")
        checked += 1
    return checked


def main() -> None:
    args = parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = sorted(set(profiles) - PROFILES.keys())
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(os.cpu_count() or 1)

    randomized_cases = validate_randomized_cases()
    rng = np.random.default_rng(20_260_716)
    results: dict[str, object] = {}
    for profile in profiles:
        width, height, repeats = PROFILES[profile]
        image = rng.integers(0, 256, (height, width), dtype=np.uint8)
        zero = np.zeros((height, width), dtype=np.int16)
        opencv_dx = np.empty((height, width), dtype=np.int16)
        opencv_dy = np.empty((height, width), dtype=np.int16)
        opencv_abs_x = np.empty((height, width), dtype=np.int16)
        opencv_abs_y = np.empty((height, width), dtype=np.int16)
        opencv_out = np.empty((height, width), dtype=np.int16)
        spatialrust_out = np.empty((height, width), dtype=np.int16)

        def opencv_allocate() -> np.ndarray:
            return opencv_l1_allocate(image, zero)

        def spatialrust_allocate() -> np.ndarray:
            return sr.sobel_l1_magnitude_image(image)

        def opencv_reuse() -> np.ndarray:
            cv2.spatialGradient(
                image,
                opencv_dx,
                opencv_dy,
                3,
                cv2.BORDER_REFLECT_101,
            )
            cv2.absdiff(opencv_dx, zero, opencv_abs_x)
            cv2.absdiff(opencv_dy, zero, opencv_abs_y)
            return cv2.add(opencv_abs_x, opencv_abs_y, opencv_out)

        def spatialrust_reuse() -> np.ndarray:
            return sr.sobel_l1_magnitude_image(image, spatialrust_out)

        expected = opencv_allocate()
        actual = spatialrust_allocate()
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{profile} allocated output is not bit-exact")
        if opencv_reuse() is not opencv_out:
            raise AssertionError("OpenCV did not return its caller-owned output")
        if spatialrust_reuse() is not spatialrust_out:
            raise AssertionError("SpatialRust did not return its caller-owned output")
        if not np.array_equal(opencv_out, expected) or not np.array_equal(
            spatialrust_out, expected
        ):
            raise AssertionError(f"{profile} reused output is not bit-exact")

        _, _, opencv_timing, spatialrust_timing = timed_pair(
            opencv_allocate,
            spatialrust_allocate,
            warmup=args.warmup,
            repeats=repeats,
            seed=116,
            min_sample_time_ms=20.0,
        )
        _, _, opencv_reuse_timing, spatialrust_reuse_timing = timed_pair(
            opencv_reuse,
            spatialrust_reuse,
            warmup=args.warmup,
            repeats=repeats,
            seed=2116,
            min_sample_time_ms=20.0,
        )
        opencv_ms = float(opencv_timing["median"])
        spatialrust_ms = float(spatialrust_timing["median"])
        opencv_reuse_ms = float(opencv_reuse_timing["median"])
        spatialrust_reuse_ms = float(spatialrust_reuse_timing["median"])
        results[profile] = {
            "width": width,
            "height": height,
            "operation": "abs(Sobel X) + abs(Sobel Y)",
            "kernel_size": 3,
            "border": "reflect101",
            "dtype": "int16",
            "exact": True,
            "opencv_stages": ["spatialGradient", "absdiff X", "absdiff Y", "add"],
            "spatialrust_stages": ["fused Sobel L1"],
            "opencv": opencv_timing,
            "spatialrust": spatialrust_timing,
            "spatialrust_speedup": opencv_ms / spatialrust_ms,
            "faster_implementation": (
                "spatialrust" if spatialrust_ms < opencv_ms else "opencv"
            ),
            "opencv_reuse": opencv_reuse_timing,
            "spatialrust_reuse": spatialrust_reuse_timing,
            "spatialrust_reuse_speedup": opencv_reuse_ms / spatialrust_reuse_ms,
            "faster_reuse_implementation": (
                "spatialrust"
                if spatialrust_reuse_ms < opencv_reuse_ms
                else "opencv"
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
        suite="opencv-fused-sobel-l1-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "allocated and caller-owned-output Python API calls",
                "paired_interleaved": True,
                "minimum_sample_time_ms": 20.0,
                "input": "seeded packed random uint8 grayscale",
                "randomized_correctness_cases": randomized_cases,
                "thread_policy": "logical CPU count for OpenCV; Rayon default for SpatialRust",
                "accuracy": "bit-exact int16 L1 magnitude",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
