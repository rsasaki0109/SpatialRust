"""Performance comparison for Epic 103 reusable CPU vision paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import emit_report, environment, make_report, timed


PROFILES = {
    "vga": (640, 480, 20),
    "1080p": (1920, 1080, 8),
    "4k": (3840, 2160, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profiles", default="vga,1080p,4k")
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def measurement(
    workload: str,
    implementation: str,
    mode: str,
    width: int,
    height: int,
    timing: dict[str, object],
) -> dict[str, object]:
    return {
        "workload": workload,
        "implementation": implementation,
        "mode": mode,
        "width": width,
        "height": height,
        "timing": timing,
    }


def main() -> None:
    args = parse_args()
    selected = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = sorted(set(selected) - PROFILES.keys())
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    rng = np.random.default_rng(103)
    measurements: list[dict[str, object]] = []
    correctness: dict[str, object] = {}
    speedups: dict[str, object] = {}

    for profile in selected:
        width, height, repeats = PROFILES[profile]
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        output_width, output_height = width // 2, height // 2

        resize_cv_out = np.empty((output_height, output_width, 3), dtype=np.uint8)
        resize_sr_out = np.empty_like(resize_cv_out)
        resize_cv = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        resize_sr = sr.resize_image(image, output_width, output_height, interpolation="bilinear")
        resize_error = int(
            np.max(np.abs(resize_cv.astype(np.int16) - resize_sr.astype(np.int16)))
        )
        correctness[f"{profile}_resize_max_u8_error"] = resize_error
        if resize_error > 1:
            raise AssertionError(f"{profile} resize error {resize_error} > 1")

        _, cv_resize_alloc = timed(
            lambda: cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR),
            warmup=args.warmup,
            repeats=repeats,
        )
        _, sr_resize_alloc = timed(
            lambda: sr.resize_image(image, output_width, output_height, interpolation="bilinear"),
            warmup=args.warmup,
            repeats=repeats,
        )
        _, cv_resize_reuse = timed(
            lambda: cv2.resize(
                image,
                (output_width, output_height),
                dst=resize_cv_out,
                interpolation=cv2.INTER_LINEAR,
            ),
            warmup=args.warmup,
            repeats=repeats,
        )
        _, sr_resize_reuse = timed(
            lambda: sr.resize_image(
                image,
                output_width,
                output_height,
                interpolation="bilinear",
                out=resize_sr_out,
            ),
            warmup=args.warmup,
            repeats=repeats,
        )
        np.testing.assert_array_equal(resize_sr_out, resize_sr)

        gray_cv_out = np.empty((height, width), dtype=np.uint8)
        gray_sr_out = np.empty_like(gray_cv_out)
        gray_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_sr = sr.rgb_to_gray_image(image)
        gray_error = int(np.max(np.abs(gray_cv.astype(np.int16) - gray_sr.astype(np.int16))))
        correctness[f"{profile}_rgb_to_gray_max_u8_error"] = gray_error
        if gray_error > 1:
            raise AssertionError(f"{profile} gray error {gray_error} > 1")
        _, cv_gray_alloc = timed(
            lambda: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            warmup=args.warmup,
            repeats=repeats,
        )
        _, sr_gray_alloc = timed(
            lambda: sr.rgb_to_gray_image(image), warmup=args.warmup, repeats=repeats
        )
        _, cv_gray_reuse = timed(
            lambda: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dst=gray_cv_out),
            warmup=args.warmup,
            repeats=repeats,
        )
        _, sr_gray_reuse = timed(
            lambda: sr.rgb_to_gray_image(image, out=gray_sr_out),
            warmup=args.warmup,
            repeats=repeats,
        )
        np.testing.assert_array_equal(gray_sr_out, gray_sr)

        chw_sr_out = np.empty((3, height, width), dtype=np.float32)
        blob_cv = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0, size=(width, height), swapRB=False, crop=False
        )[0]
        chw_sr = sr.normalize_image_chw(image)
        chw_error = float(np.max(np.abs(blob_cv - chw_sr)))
        correctness[f"{profile}_ai_preprocess_max_f32_error"] = chw_error
        if chw_error > 1e-6:
            raise AssertionError(f"{profile} AI preprocess error {chw_error} > 1e-6")
        _, cv_chw_alloc = timed(
            lambda: cv2.dnn.blobFromImage(
                image, scalefactor=1.0 / 255.0, size=(width, height), swapRB=False, crop=False
            ),
            warmup=args.warmup,
            repeats=repeats,
        )
        _, sr_chw_alloc = timed(
            lambda: sr.normalize_image_chw(image), warmup=args.warmup, repeats=repeats
        )
        _, sr_chw_reuse = timed(
            lambda: sr.normalize_image_chw(image, out=chw_sr_out),
            warmup=args.warmup,
            repeats=repeats,
        )
        np.testing.assert_allclose(chw_sr_out, chw_sr, atol=0.0, rtol=0.0)

        rows = (
            ("resize_bilinear", "opencv", "allocate", cv_resize_alloc),
            ("resize_bilinear", "spatialrust", "allocate", sr_resize_alloc),
            ("resize_bilinear", "opencv", "reuse", cv_resize_reuse),
            ("resize_bilinear", "spatialrust", "reuse", sr_resize_reuse),
            ("rgb_to_gray", "opencv", "allocate", cv_gray_alloc),
            ("rgb_to_gray", "spatialrust", "allocate", sr_gray_alloc),
            ("rgb_to_gray", "opencv", "reuse", cv_gray_reuse),
            ("rgb_to_gray", "spatialrust", "reuse", sr_gray_reuse),
            ("ai_preprocess", "opencv", "allocate", cv_chw_alloc),
            ("ai_preprocess", "spatialrust", "allocate", sr_chw_alloc),
            ("ai_preprocess", "spatialrust", "reuse", sr_chw_reuse),
        )
        measurements.extend(
            measurement(workload, implementation, mode, width, height, timing)
            for workload, implementation, mode, timing in rows
        )
        speedups[profile] = {
            "resize_allocate": cv_resize_alloc["median"] / sr_resize_alloc["median"],
            "resize_reuse": cv_resize_reuse["median"] / sr_resize_reuse["median"],
            "rgb_to_gray_allocate": cv_gray_alloc["median"] / sr_gray_alloc["median"],
            "rgb_to_gray_reuse": cv_gray_reuse["median"] / sr_gray_reuse["median"],
            "ai_preprocess_allocate": cv_chw_alloc["median"] / sr_chw_alloc["median"],
            "ai_preprocess_reuse_vs_opencv_allocate": cv_chw_alloc["median"]
            / sr_chw_reuse["median"],
        }

    environment_receipt = environment(
        opencv_version=cv2.__version__, spatialrust_version=sr.__version__
    )
    environment_receipt["opencv_threads"] = cv2.getNumThreads()
    environment_receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-vision-performance",
        kind="performance",
        status="pass",
        environment_receipt=environment_receipt,
        results={
            "correctness": correctness,
            "speedup_vs_opencv": speedups,
            "measurements": measurements,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
