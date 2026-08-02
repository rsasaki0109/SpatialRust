"""Reproducible packed RGB8-to-gray comparison with OpenCV."""

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
    "vga": (640, 480, 48),
    "1080p": (1920, 1080, 32),
    "4k": (3840, 2160, 20),
    "8k": (7680, 4320, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def validate_randomized_cases() -> tuple[int, int]:
    rng = np.random.default_rng(1151)
    checked = 0
    max_error = 0
    for case in range(300):
        height = int(rng.integers(1, 101))
        width = int(rng.integers(1, 141))
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        if case % 3 == 0:
            image = image[:, ::-1]
        expected = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
        actual = sr.rgb_to_gray_image(image)
        error = int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max())
        if error > 1:
            raise AssertionError(f"random case {case} max error {error} exceeds 1")
        max_error = max(max_error, error)
        checked += 1
    return checked, max_error


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
        width, height, repeats = PROFILES[profile]
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        opencv_out = np.empty((height, width), dtype=np.uint8)
        spatialrust_out = np.empty_like(opencv_out)

        def opencv_allocate() -> np.ndarray:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        def spatialrust_allocate() -> np.ndarray:
            return sr.rgb_to_gray_image(image)

        def opencv_reuse() -> np.ndarray:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dst=opencv_out)

        def spatialrust_reuse() -> np.ndarray:
            return sr.rgb_to_gray_image(image, out=spatialrust_out)

        expected = opencv_allocate()
        actual = spatialrust_allocate()
        error = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        max_error = int(error.max())
        if max_error > 1:
            raise AssertionError(f"{profile} max error {max_error} exceeds 1")
        if opencv_reuse() is not opencv_out or spatialrust_reuse() is not spatialrust_out:
            raise AssertionError("caller-owned output identity was not preserved")
        if not np.array_equal(opencv_out, expected) or not np.array_equal(
            spatialrust_out, actual
        ):
            raise AssertionError(f"{profile} reuse output differs from allocated output")

        _, _, opencv_timing, spatialrust_timing = timed_pair(
            opencv_allocate,
            spatialrust_allocate,
            warmup=args.warmup,
            repeats=repeats,
            seed=1151,
            min_sample_time_ms=20.0,
        )
        _, _, opencv_reuse_timing, spatialrust_reuse_timing = timed_pair(
            opencv_reuse,
            spatialrust_reuse,
            warmup=args.warmup,
            repeats=repeats,
            seed=2151,
            min_sample_time_ms=20.0,
        )
        opencv_ms = float(opencv_timing["median"])
        spatialrust_ms = float(spatialrust_timing["median"])
        opencv_reuse_ms = float(opencv_reuse_timing["median"])
        spatialrust_reuse_ms = float(spatialrust_reuse_timing["median"])
        results[profile] = {
            "width": width,
            "height": height,
            "operation": "packed RGB uint8 to gray uint8",
            "coefficients": "OpenCV-compatible BT.601 Q14",
            "max_absolute_error": max_error,
            "exact_fraction": float((error == 0).mean()),
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
        suite="opencv-packed-rgb8-to-gray-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "allocated and caller-owned-output Python API calls",
                "paired_interleaved": True,
                "minimum_sample_time_ms": 20.0,
                "input": "seeded packed random uint8 RGB",
                "randomized_correctness_cases": randomized_cases,
                "randomized_max_absolute_error": randomized_max_error,
                "thread_policy": "logical CPU count for OpenCV; Rayon default for SpatialRust",
                "accuracy": "maximum absolute uint8 error <= 1",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
