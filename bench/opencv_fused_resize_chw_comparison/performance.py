"""Reproducible OpenCV blob versus SpatialRust fused resize-to-CHW comparison."""

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
    "1080p_to_640": (1920, 1080, 640, 640, 32),
    "4k_to_640": (3840, 2160, 640, 640, 20),
    "4k_to_720p": (3840, 2160, 1280, 720, 16),
}
SCALE = 1.0 / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--warmup", type=int, default=6)
    return parser.parse_args()


def opencv_blob(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.dnn.blobFromImage(
        image,
        scalefactor=SCALE,
        size=(width, height),
        mean=(0.0, 0.0, 0.0),
        swapRB=False,
        crop=False,
    )[0]


def validate_randomized_cases() -> tuple[int, float]:
    rng = np.random.default_rng(1154)
    max_error = 0.0
    for case in range(300):
        height = int(rng.integers(2, 101))
        width = int(rng.integers(2, 141))
        output_height = int(rng.integers(1, 81))
        output_width = int(rng.integers(1, 101))
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        if case % 3 == 0:
            image = image[:, ::-1]
        actual = sr.resize_normalize_image_chw(image, output_width, output_height)
        unfused = sr.normalize_image_chw(
            sr.resize_image(image, output_width, output_height)
        )
        if not np.array_equal(actual, unfused):
            raise AssertionError(f"random case {case} differs from unfused SpatialRust")
        expected = opencv_blob(
            np.ascontiguousarray(image), output_width, output_height
        )
        error = float(np.max(np.abs(expected - actual), initial=0.0))
        if error > SCALE + 1e-7:
            raise AssertionError(f"random case {case} max error {error} exceeds 1/255")
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
        spatialrust_out = np.empty((3, output_height, output_width), dtype=np.float32)

        def opencv_allocate() -> np.ndarray:
            return opencv_blob(image, output_width, output_height)

        def spatialrust_allocate() -> np.ndarray:
            return sr.resize_normalize_image_chw(image, output_width, output_height)

        def spatialrust_reuse() -> np.ndarray:
            return sr.resize_normalize_image_chw(
                image, output_width, output_height, out=spatialrust_out
            )

        expected = opencv_allocate()
        actual = spatialrust_allocate()
        unfused = sr.normalize_image_chw(
            sr.resize_image(image, output_width, output_height)
        )
        if not np.array_equal(actual, unfused):
            raise AssertionError(f"{profile} differs from unfused SpatialRust")
        error = np.abs(expected - actual)
        max_error = float(np.max(error, initial=0.0))
        if max_error > SCALE + 1e-7:
            raise AssertionError(f"{profile} max error {max_error} exceeds 1/255")
        if spatialrust_reuse() is not spatialrust_out:
            raise AssertionError("caller-owned output identity was not preserved")
        if not np.array_equal(spatialrust_out, actual):
            raise AssertionError(f"{profile} reuse output differs from allocation")

        _, _, opencv_timing, spatialrust_timing = timed_pair(
            opencv_allocate,
            spatialrust_allocate,
            warmup=args.warmup,
            repeats=repeats,
            seed=1154,
            min_sample_time_ms=20.0,
        )
        _, _, opencv_reuse_reference, spatialrust_reuse_timing = timed_pair(
            opencv_allocate,
            spatialrust_reuse,
            warmup=args.warmup,
            repeats=repeats,
            seed=2154,
            min_sample_time_ms=20.0,
        )
        opencv_ms = float(opencv_timing["median"])
        spatialrust_ms = float(spatialrust_timing["median"])
        opencv_reuse_ms = float(opencv_reuse_reference["median"])
        spatialrust_reuse_ms = float(spatialrust_reuse_timing["median"])
        results[profile] = {
            "input_dimensions": [width, height],
            "output_dimensions": [output_width, output_height],
            "operation": "bilinear RGB8 resize, float scale, CHW pack",
            "max_absolute_error": max_error,
            "exact_fraction": float((error == 0.0).mean()),
            "spatialrust_unfused_exact": True,
            "opencv_blob": opencv_timing,
            "spatialrust": spatialrust_timing,
            "spatialrust_speedup": opencv_ms / spatialrust_ms,
            "opencv_blob_reuse_reference": opencv_reuse_reference,
            "spatialrust_reuse": spatialrust_reuse_timing,
            "spatialrust_reuse_speedup": opencv_reuse_ms / spatialrust_reuse_ms,
        }

    receipt = environment(opencv_version=cv2.__version__, spatialrust_version=sr.__version__)
    receipt["opencv_threads"] = cv2.getNumThreads()
    receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-fused-resize-normalize-chw-performance",
        kind="performance",
        status="pass",
        environment_receipt=receipt,
        results={
            "methodology": {
                "opencv_reference": "cv2.dnn.blobFromImage",
                "timing_scope": "allocated calls; SpatialRust caller-owned output also compared to OpenCV allocation",
                "paired_interleaved": True,
                "minimum_sample_time_ms": 20.0,
                "scale": SCALE,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
                "randomized_correctness_cases": randomized_cases,
                "randomized_max_absolute_error": randomized_max_error,
                "accuracy": "exact versus SpatialRust unfused; OpenCV max error <= 1/255",
            },
            "profiles": results,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
