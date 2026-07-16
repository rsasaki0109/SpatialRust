"""Reproducible fused Sobel-to-binary-mask comparison with OpenCV."""

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
    "vga": (640, 480, 24),
    "1080p": (1920, 1080, 16),
    "4k": (3840, 2160, 10),
}
THRESHOLD = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def opencv_allocate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    signed = cv2.Sobel(
        image,
        cv2.CV_16S,
        dx,
        dy,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    )
    absolute = cv2.convertScaleAbs(signed)
    return cv2.threshold(absolute, THRESHOLD, 255, cv2.THRESH_BINARY)[1]


def validate_randomized_cases() -> int:
    rng = np.random.default_rng(119)
    checked = 0
    for case in range(300):
        height = int(rng.integers(1, 120))
        width = int(rng.integers(1, 160))
        image = rng.integers(0, 256, (height, width), dtype=np.uint8)
        if case % 3 == 0:
            image = image[:, ::-1]
        packed = np.ascontiguousarray(image)
        dx, dy = ((1, 0), (0, 1))[case & 1]
        expected = opencv_allocate(packed, dx, dy)
        actual = sr.sobel_threshold_image(image, dx, dy, THRESHOLD)
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
        signed = np.empty((height, width), dtype=np.int16)
        absolute = np.empty((height, width), dtype=np.uint8)
        opencv_out = np.empty((height, width), dtype=np.uint8)
        spatialrust_out = np.empty((height, width), dtype=np.uint8)

        def cv_allocate() -> np.ndarray:
            return opencv_allocate(image, 1, 0)

        def sr_allocate() -> np.ndarray:
            return sr.sobel_threshold_image(image, 1, 0, THRESHOLD)

        def cv_reuse() -> np.ndarray:
            cv2.Sobel(
                image,
                cv2.CV_16S,
                1,
                0,
                signed,
                3,
                1.0,
                0.0,
                cv2.BORDER_REFLECT_101,
            )
            cv2.convertScaleAbs(signed, absolute)
            return cv2.threshold(
                absolute, THRESHOLD, 255, cv2.THRESH_BINARY, opencv_out
            )[1]

        def sr_reuse() -> np.ndarray:
            return sr.sobel_threshold_image(
                image, 1, 0, THRESHOLD, out=spatialrust_out
            )

        expected = cv_allocate()
        if not np.array_equal(sr_allocate(), expected):
            raise AssertionError(f"{profile} allocated output is not bit-exact")
        if cv_reuse() is not opencv_out or sr_reuse() is not spatialrust_out:
            raise AssertionError(f"{profile} caller-owned output identity failed")
        if not np.array_equal(opencv_out, expected) or not np.array_equal(
            spatialrust_out, expected
        ):
            raise AssertionError(f"{profile} reused output is not bit-exact")

        _, _, cv_timing, sr_timing = timed_pair(
            cv_allocate,
            sr_allocate,
            warmup=args.warmup,
            repeats=repeats,
            seed=119,
            min_sample_time_ms=20.0,
        )
        _, _, cv_reuse_timing, sr_reuse_timing = timed_pair(
            cv_reuse,
            sr_reuse,
            warmup=args.warmup,
            repeats=repeats,
            seed=2119,
            min_sample_time_ms=20.0,
        )
        cv_ms = float(cv_timing["median"])
        sr_ms = float(sr_timing["median"])
        cv_reuse_ms = float(cv_reuse_timing["median"])
        sr_reuse_ms = float(sr_reuse_timing["median"])
        results[profile] = {
            "width": width,
            "height": height,
            "operation": "abs(Sobel X) > 96 binary mask",
            "kernel_size": 3,
            "border": "reflect101",
            "exact": True,
            "opencv_stages": ["Sobel CV_16S", "convertScaleAbs", "threshold"],
            "spatialrust_stages": ["fused Sobel threshold"],
            "opencv": cv_timing,
            "spatialrust": sr_timing,
            "spatialrust_speedup": cv_ms / sr_ms,
            "faster_implementation": "spatialrust" if sr_ms < cv_ms else "opencv",
            "opencv_reuse": cv_reuse_timing,
            "spatialrust_reuse": sr_reuse_timing,
            "spatialrust_reuse_speedup": cv_reuse_ms / sr_reuse_ms,
            "faster_reuse_implementation": (
                "spatialrust" if sr_reuse_ms < cv_reuse_ms else "opencv"
            ),
            "spatialrust_reuse_vs_opencv_allocate_speedup": cv_ms / sr_reuse_ms,
            "faster_spatialrust_reuse_vs_opencv_allocate": (
                "spatialrust" if sr_reuse_ms < cv_ms else "opencv"
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
        suite="opencv-fused-sobel-threshold-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "timing_scope": "allocated and caller-owned-output Python API calls",
                "paired_interleaved": True,
                "minimum_sample_time_ms": 20.0,
                "input": "seeded packed random uint8 grayscale",
                "threshold": THRESHOLD,
                "randomized_correctness_cases": randomized_cases,
                "accuracy": "bit-exact binary mask for alternating X/Y derivatives",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
