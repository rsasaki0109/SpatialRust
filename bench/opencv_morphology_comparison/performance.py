"""Reproducible rectangular morphology comparison with OpenCV."""

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
    "vga": (640, 480, 30),
    "1080p": (1920, 1080, 20),
    "4k": (3840, 2160, 12),
}
KERNELS = (5, 511)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--kernels", default=",".join(map(str, KERNELS)))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def validate_randomized_cases() -> int:
    rng = np.random.default_rng(117)
    operations = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }
    checked = 0
    for case in range(140):
        height = int(rng.integers(1, 80))
        width = int(rng.integers(1, 100))
        kernel_width = int(rng.integers(1, 18))
        kernel_height = int(rng.integers(1, 18))
        iterations = int(rng.integers(0, 3))
        image = rng.integers(0, 256, (height, width), dtype=np.uint8)
        kernel = np.ones((kernel_height, kernel_width), dtype=np.uint8)
        for name, code in operations.items():
            expected = cv2.morphologyEx(
                image,
                code,
                kernel,
                iterations=iterations,
                borderType=cv2.BORDER_REPLICATE,
            )
            actual = sr.morphology_image(
                image, name, kernel_width, kernel_height, "rect", iterations
            )
            if not np.array_equal(actual, expected):
                raise AssertionError(f"random case {case} failed for {name}")
            checked += 1
    return checked


def main() -> None:
    args = parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    kernels = [int(value) for value in args.kernels.split(",") if value.strip()]
    unknown = sorted(set(profiles) - PROFILES.keys())
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    if any(value < 1 for value in kernels):
        raise ValueError("kernel sizes must be positive")
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    randomized_cases = validate_randomized_cases()
    rng = np.random.default_rng(20_260_715)
    results: dict[str, object] = {}
    for profile in profiles:
        width, height, repeats = PROFILES[profile]
        image = rng.integers(0, 256, (height, width), dtype=np.uint8)
        profile_results: dict[str, object] = {}
        for kernel_size in kernels:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            opencv_out = np.empty_like(image)
            spatialrust_out = np.empty_like(image)
            spatialrust_workspace = sr.MorphologyWorkspace()

            def opencv_open() -> np.ndarray:
                return cv2.morphologyEx(
                    image,
                    cv2.MORPH_OPEN,
                    kernel,
                    borderType=cv2.BORDER_REPLICATE,
                )

            def spatialrust_open() -> np.ndarray:
                return sr.morphology_image(
                    image, "open", kernel_size, kernel_size, "rect", 1
                )

            def opencv_open_reuse() -> np.ndarray:
                return cv2.morphologyEx(
                    image,
                    cv2.MORPH_OPEN,
                    kernel,
                    dst=opencv_out,
                    borderType=cv2.BORDER_REPLICATE,
                )

            def spatialrust_open_reuse() -> np.ndarray:
                return sr.morphology_image(
                    image,
                    "open",
                    kernel_size,
                    kernel_size,
                    "rect",
                    1,
                    out=spatialrust_out,
                    workspace=spatialrust_workspace,
                )

            expected = opencv_open()
            actual = spatialrust_open()
            if not np.array_equal(actual, expected):
                raise AssertionError(f"{profile}/{kernel_size} is not bit-exact")
            if opencv_open_reuse() is not opencv_out:
                raise AssertionError("OpenCV did not return its caller-owned output")
            if spatialrust_open_reuse() is not spatialrust_out:
                raise AssertionError("SpatialRust did not return its caller-owned output")
            if not np.array_equal(opencv_out, expected) or not np.array_equal(
                spatialrust_out, expected
            ):
                raise AssertionError(f"{profile}/{kernel_size} reuse is not bit-exact")
            _, _, opencv_timing, spatialrust_timing = timed_pair(
                opencv_open,
                spatialrust_open,
                warmup=args.warmup,
                repeats=repeats,
                seed=117 + kernel_size,
                min_sample_time_ms=20.0,
            )
            _, _, opencv_reuse_timing, spatialrust_reuse_timing = timed_pair(
                opencv_open_reuse,
                spatialrust_open_reuse,
                warmup=args.warmup,
                repeats=repeats,
                seed=2117 + kernel_size,
                min_sample_time_ms=20.0,
            )
            opencv_ms = float(opencv_timing["median"])
            spatialrust_ms = float(spatialrust_timing["median"])
            opencv_reuse_ms = float(opencv_reuse_timing["median"])
            spatialrust_reuse_ms = float(spatialrust_reuse_timing["median"])
            profile_results[str(kernel_size)] = {
                "width": width,
                "height": height,
                "kernel_width": kernel_size,
                "kernel_height": kernel_size,
                "operation": "open",
                "iterations": 1,
                "border": "replicate",
                "exact": True,
                "opencv": opencv_timing,
                "spatialrust": spatialrust_timing,
                "spatialrust_speedup": opencv_ms / spatialrust_ms,
                "faster_implementation": (
                    "spatialrust" if spatialrust_ms < opencv_ms else "opencv"
                ),
                "opencv_reuse": opencv_reuse_timing,
                "spatialrust_reuse": spatialrust_reuse_timing,
                "spatialrust_reuse_speedup": opencv_reuse_ms
                / spatialrust_reuse_ms,
                "faster_reuse_implementation": (
                    "spatialrust"
                    if spatialrust_reuse_ms < opencv_reuse_ms
                    else "opencv"
                ),
                "spatialrust_workspace_capacity": spatialrust_workspace.capacity,
                "spatialrust_workspace_workers": spatialrust_workspace.worker_capacity,
                "spatialrust_workspace_line_capacity": spatialrust_workspace.line_capacity,
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
        suite="opencv-rectangular-morphology-performance",
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
                "thread_policy": "library defaults; OpenCV thread count recorded",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
