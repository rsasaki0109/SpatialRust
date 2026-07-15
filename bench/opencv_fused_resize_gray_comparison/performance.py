"""Reproducible OpenCV resize+gray versus SpatialRust fused comparison."""

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
    "1080p_to_540p": (1920, 1080, 960, 540, 32),
    "4k_to_1080p": (3840, 2160, 1920, 1080, 20),
    "8k_to_4k": (7680, 4320, 3840, 2160, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def opencv_pipeline(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)


def validate_randomized_cases() -> tuple[int, int]:
    rng = np.random.default_rng(1153)
    max_error = 0
    for case in range(300):
        height = int(rng.integers(2, 101))
        width = int(rng.integers(2, 141))
        output_height = int(rng.integers(1, 81))
        output_width = int(rng.integers(1, 101))
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        if case % 3 == 0:
            image = image[:, ::-1]
        actual = sr.resize_rgb_to_gray_image(image, output_width, output_height)
        unfused = sr.rgb_to_gray_image(
            sr.resize_image(image, output_width, output_height)
        )
        if not np.array_equal(actual, unfused):
            raise AssertionError(f"random case {case} differs from unfused SpatialRust")
        expected = opencv_pipeline(
            np.ascontiguousarray(image), output_width, output_height
        )
        error = int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max())
        if error > 2:
            raise AssertionError(f"random case {case} max error {error} exceeds 2")
        max_error = max(max_error, error)
    return 300, max_error


def main() -> None:
    args = parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = sorted(set(profiles) - PROFILES.keys())
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(os.cpu_count() or 1)

    randomized_cases, randomized_max_error = validate_randomized_cases()
    rng = np.random.default_rng(20_260_716)
    results: dict[str, object] = {}
    for profile in profiles:
        width, height, output_width, output_height, repeats = PROFILES[profile]
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        opencv_rgb = np.empty((output_height, output_width, 3), dtype=np.uint8)
        opencv_out = np.empty((output_height, output_width), dtype=np.uint8)
        spatialrust_out = np.empty_like(opencv_out)

        def opencv_allocate() -> np.ndarray:
            return opencv_pipeline(image, output_width, output_height)

        def spatialrust_allocate() -> np.ndarray:
            return sr.resize_rgb_to_gray_image(image, output_width, output_height)

        def opencv_reuse() -> np.ndarray:
            cv2.resize(
                image,
                (output_width, output_height),
                dst=opencv_rgb,
                interpolation=cv2.INTER_LINEAR,
            )
            return cv2.cvtColor(opencv_rgb, cv2.COLOR_RGB2GRAY, dst=opencv_out)

        def spatialrust_reuse() -> np.ndarray:
            return sr.resize_rgb_to_gray_image(
                image, output_width, output_height, out=spatialrust_out
            )

        expected = opencv_allocate()
        actual = spatialrust_allocate()
        unfused = sr.rgb_to_gray_image(
            sr.resize_image(image, output_width, output_height)
        )
        if not np.array_equal(actual, unfused):
            raise AssertionError(f"{profile} differs from unfused SpatialRust")
        error = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        max_error = int(error.max())
        if max_error > 2:
            raise AssertionError(f"{profile} max error {max_error} exceeds 2")
        if opencv_reuse() is not opencv_out or spatialrust_reuse() is not spatialrust_out:
            raise AssertionError("caller-owned output identity was not preserved")

        _, _, opencv_timing, spatialrust_timing = timed_pair(
            opencv_allocate,
            spatialrust_allocate,
            warmup=args.warmup,
            repeats=repeats,
            seed=1153,
            min_sample_time_ms=20.0,
        )
        _, _, opencv_reuse_timing, spatialrust_reuse_timing = timed_pair(
            opencv_reuse,
            spatialrust_reuse,
            warmup=args.warmup,
            repeats=repeats,
            seed=2153,
            min_sample_time_ms=20.0,
        )
        opencv_ms = float(opencv_timing["median"])
        spatialrust_ms = float(spatialrust_timing["median"])
        opencv_reuse_ms = float(opencv_reuse_timing["median"])
        spatialrust_reuse_ms = float(spatialrust_reuse_timing["median"])
        results[profile] = {
            "input_dimensions": [width, height],
            "output_dimensions": [output_width, output_height],
            "operation": "bilinear RGB8 resize followed by BT.601 gray",
            "max_absolute_error": max_error,
            "exact_fraction": float((error == 0).mean()),
            "spatialrust_unfused_exact": True,
            "opencv": opencv_timing,
            "spatialrust": spatialrust_timing,
            "spatialrust_speedup": opencv_ms / spatialrust_ms,
            "faster_implementation": "spatialrust" if spatialrust_ms < opencv_ms else "opencv",
            "opencv_reuse": opencv_reuse_timing,
            "spatialrust_reuse": spatialrust_reuse_timing,
            "spatialrust_reuse_speedup": opencv_reuse_ms / spatialrust_reuse_ms,
            "faster_reuse_implementation": (
                "spatialrust" if spatialrust_reuse_ms < opencv_reuse_ms else "opencv"
            ),
        }

    receipt = environment(opencv_version=cv2.__version__, spatialrust_version=sr.__version__)
    receipt["opencv_threads"] = cv2.getNumThreads()
    receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-fused-resize-gray-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "allocated and caller-owned-output Python API pipelines",
                "paired_interleaved": True,
                "minimum_sample_time_ms": 20.0,
                "input": "seeded packed random uint8 RGB",
                "randomized_correctness_cases": randomized_cases,
                "randomized_max_absolute_error": randomized_max_error,
                "accuracy": "exact versus SpatialRust unfused; OpenCV max error <= 2",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
