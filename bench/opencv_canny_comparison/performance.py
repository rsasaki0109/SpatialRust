"""Reproducible OpenCV versus allocation-light SpatialRust Canny comparison."""

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
    "vga": (640, 480, 30),
    "1080p": (1920, 1080, 20),
    "4k": (3840, 2160, 12),
}
PATTERNS = ("document-lines", "sensor-noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--patterns", default=",".join(PATTERNS))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def make_image(width: int, height: int, pattern: str, seed: int) -> np.ndarray:
    if pattern == "sensor-noise":
        return np.random.default_rng(seed).integers(0, 256, (height, width), dtype=np.uint8)
    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(20, height, 80):
        cv2.line(image, (10, y), (width - 11, y), 255, 2)
    for x in range(60, width, 320):
        cv2.rectangle(image, (x, 35), (min(width - 1, x + 90), min(height - 1, 105)), 160, 2)
    return image


def validate_randomized_cases() -> int:
    rng = np.random.default_rng(118)
    for case in range(300):
        height = int(rng.integers(1, 97))
        width = int(rng.integers(1, 129))
        image = rng.integers(0, 256, (height, width), dtype=np.uint8)
        expected = cv2.Canny(image, 80.0, 160.0, apertureSize=3, L2gradient=True)
        actual = sr.canny_image(image, 80.0, 160.0, aperture_size=3, l2_gradient=True)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"random Canny case {case} differs from OpenCV")
    return 300


def main() -> None:
    args = parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    patterns = [value.strip() for value in args.patterns.split(",") if value.strip()]
    if unknown := sorted(set(profiles) - PROFILES.keys()):
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    if unknown := sorted(set(patterns) - set(PATTERNS)):
        raise ValueError(f"unknown patterns: {', '.join(unknown)}")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(os.cpu_count() or 1)

    randomized_cases = validate_randomized_cases()
    results: dict[str, object] = {}
    for profile in profiles:
        width, height, repeats = PROFILES[profile]
        for pattern in patterns:
            image = make_image(width, height, pattern, 20_260_716)
            opencv_out = np.empty_like(image)
            spatialrust_out = np.empty_like(image)
            workspace = sr.CannyWorkspace()

            def opencv_allocate() -> np.ndarray:
                return cv2.Canny(image, 80.0, 160.0, apertureSize=3, L2gradient=True)

            def spatialrust_allocate() -> np.ndarray:
                return sr.canny_image(
                    image, 80.0, 160.0, aperture_size=3, l2_gradient=True
                )

            def opencv_reuse() -> np.ndarray:
                return cv2.Canny(
                    image, 80.0, 160.0, opencv_out, apertureSize=3, L2gradient=True
                )

            def spatialrust_reuse() -> np.ndarray:
                return sr.canny_image(
                    image,
                    80.0,
                    160.0,
                    aperture_size=3,
                    l2_gradient=True,
                    out=spatialrust_out,
                    workspace=workspace,
                )

            if not np.array_equal(opencv_allocate(), spatialrust_allocate()):
                raise AssertionError(f"{profile}/{pattern} differs from OpenCV")
            if opencv_reuse() is not opencv_out or spatialrust_reuse() is not spatialrust_out:
                raise AssertionError("caller-owned output identity was not preserved")
            _, _, cv_alloc, sr_alloc = timed_pair(
                opencv_allocate,
                spatialrust_allocate,
                warmup=args.warmup,
                repeats=repeats,
                seed=118,
                min_sample_time_ms=20.0,
            )
            _, _, cv_reuse, sr_reuse = timed_pair(
                opencv_reuse,
                spatialrust_reuse,
                warmup=args.warmup,
                repeats=repeats,
                seed=1118,
                min_sample_time_ms=20.0,
            )
            cv_alloc_ms = float(cv_alloc["median"])
            sr_alloc_ms = float(sr_alloc["median"])
            cv_reuse_ms = float(cv_reuse["median"])
            sr_reuse_ms = float(sr_reuse["median"])
            results[f"{profile}/{pattern}"] = {
                "dimensions": [width, height],
                "pattern": pattern,
                "accuracy": "bit exact",
                "opencv_allocate": cv_alloc,
                "spatialrust_allocate": sr_alloc,
                "spatialrust_allocate_speedup": cv_alloc_ms / sr_alloc_ms,
                "opencv_reuse": cv_reuse,
                "spatialrust_reuse": sr_reuse,
                "spatialrust_reuse_speedup": cv_reuse_ms / sr_reuse_ms,
            }

    receipt = environment(opencv_version=cv2.__version__, spatialrust_version=sr.__version__)
    receipt["opencv_threads"] = cv2.getNumThreads()
    receipt["opencv_opencl_enabled"] = bool(hasattr(cv2, "ocl") and cv2.ocl.useOpenCL())
    report = make_report(
        suite="opencv-canny-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "operation": "Canny 3x3, thresholds 80/160, L2 gradient",
                "paired_interleaved": True,
                "minimum_sample_time_ms": 20.0,
                "randomized_bit_exact_cases": randomized_cases,
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
