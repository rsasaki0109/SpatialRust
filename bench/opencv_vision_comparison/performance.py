"""Performance comparison for Epic 103 reusable CPU vision paths."""

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
    "vga": (640, 480, 20),
    "1080p": (1920, 1080, 8),
    "4k": (3840, 2160, 3),
}
MIN_SAMPLE_TIME_MS = 5.0


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
    pixels = width * height
    return {
        "workload": workload,
        "implementation": implementation,
        "mode": mode,
        "width": width,
        "height": height,
        "timing": timing,
        "throughput_megapixels_per_second": pixels
        / float(timing["median"])
        / 1000.0,
    }


def numerical_accuracy(
    reference: np.ndarray, actual: np.ndarray, peak: float
) -> dict[str, float | None]:
    """Return scale-aware error metrics over identical array layouts."""

    if reference.shape != actual.shape:
        raise AssertionError(f"shape mismatch: {reference.shape} != {actual.shape}")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    mse = float(np.mean(difference * difference))
    reference_l2 = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return {
        "max_absolute_error": float(np.max(absolute, initial=0.0)),
        "mean_absolute_error": float(np.mean(absolute)),
        "root_mean_square_error": mse**0.5,
        "relative_l2_error": float(np.linalg.norm(difference.ravel())) / max(reference_l2, 1e-30),
        "exact_fraction": float(np.mean(difference == 0.0)),
        "psnr_db": None if mse == 0.0 else 20.0 * np.log10(peak / mse**0.5),
    }


def binary_accuracy(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference_edge = reference != 0
    actual_edge = actual != 0
    true_positive = int(np.count_nonzero(reference_edge & actual_edge))
    false_positive = int(np.count_nonzero(~reference_edge & actual_edge))
    false_negative = int(np.count_nonzero(reference_edge & ~actual_edge))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
        "intersection_over_union": true_positive
        / max(true_positive + false_positive + false_negative, 1),
        "disagreement_fraction": float(np.mean(reference_edge != actual_edge)),
    }


def speed_comparison(
    opencv_timing: dict[str, object], spatialrust_timing: dict[str, object]
) -> dict[str, float | str]:
    median_ratio = float(opencv_timing["median"]) / float(spatialrust_timing["median"])
    return {
        "opencv_ms": float(opencv_timing["median"]),
        "spatialrust_ms": float(spatialrust_timing["median"]),
        "spatialrust_speedup": median_ratio,
        "faster_implementation": "spatialrust" if median_ratio > 1.0 else "opencv",
        "p95_ratio": float(opencv_timing["p95"]) / float(spatialrust_timing["p95"]),
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
    accuracy: dict[str, object] = {}
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
        accuracy[f"{profile}_resize_bilinear"] = numerical_accuracy(
            resize_cv, resize_sr, 255.0
        )
        if resize_error > 1:
            raise AssertionError(f"{profile} resize error {resize_error} > 1")

        _, _, cv_resize_alloc, sr_resize_alloc = timed_pair(
            lambda: cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR),
            lambda: sr.resize_image(image, output_width, output_height, interpolation="bilinear"),
            warmup=args.warmup,
            repeats=repeats,
            seed=103,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )
        _, _, cv_resize_reuse, sr_resize_reuse = timed_pair(
            lambda: cv2.resize(
                image,
                (output_width, output_height),
                dst=resize_cv_out,
                interpolation=cv2.INTER_LINEAR,
            ),
            lambda: sr.resize_image(
                image,
                output_width,
                output_height,
                interpolation="bilinear",
                out=resize_sr_out,
            ),
            warmup=args.warmup,
            repeats=repeats,
            seed=104,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )
        np.testing.assert_array_equal(resize_sr_out, resize_sr)

        gray_cv_out = np.empty((height, width), dtype=np.uint8)
        gray_sr_out = np.empty_like(gray_cv_out)
        gray_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_sr = sr.rgb_to_gray_image(image)
        gray_error = int(np.max(np.abs(gray_cv.astype(np.int16) - gray_sr.astype(np.int16))))
        correctness[f"{profile}_rgb_to_gray_max_u8_error"] = gray_error
        accuracy[f"{profile}_rgb_to_gray"] = numerical_accuracy(gray_cv, gray_sr, 255.0)
        if gray_error > 1:
            raise AssertionError(f"{profile} gray error {gray_error} > 1")
        _, _, cv_gray_alloc, sr_gray_alloc = timed_pair(
            lambda: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            lambda: sr.rgb_to_gray_image(image),
            warmup=args.warmup,
            repeats=repeats,
            seed=105,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )
        _, _, cv_gray_reuse, sr_gray_reuse = timed_pair(
            lambda: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dst=gray_cv_out),
            lambda: sr.rgb_to_gray_image(image, out=gray_sr_out),
            warmup=args.warmup,
            repeats=repeats,
            seed=106,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )
        np.testing.assert_array_equal(gray_sr_out, gray_sr)

        chw_sr_out = np.empty((3, height, width), dtype=np.float32)
        blob_cv = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0, size=(width, height), swapRB=False, crop=False
        )[0]
        chw_sr = sr.normalize_image_chw(image)
        chw_error = float(np.max(np.abs(blob_cv - chw_sr)))
        correctness[f"{profile}_ai_preprocess_max_f32_error"] = chw_error
        accuracy[f"{profile}_ai_preprocess"] = numerical_accuracy(blob_cv, chw_sr, 1.0)
        if chw_error > 1e-6:
            raise AssertionError(f"{profile} AI preprocess error {chw_error} > 1e-6")
        _, _, cv_chw_alloc, sr_chw_alloc = timed_pair(
            lambda: cv2.dnn.blobFromImage(
                image, scalefactor=1.0 / 255.0, size=(width, height), swapRB=False, crop=False
            ),
            lambda: sr.normalize_image_chw(image),
            warmup=args.warmup,
            repeats=repeats,
            seed=107,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )
        _, _, cv_chw_reuse_baseline, sr_chw_reuse = timed_pair(
            lambda: cv2.dnn.blobFromImage(
                image, scalefactor=1.0 / 255.0, size=(width, height), swapRB=False, crop=False
            ),
            lambda: sr.normalize_image_chw(image, out=chw_sr_out),
            warmup=args.warmup,
            repeats=repeats,
            seed=108,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )
        np.testing.assert_allclose(chw_sr_out, chw_sr, atol=0.0, rtol=0.0)

        gaussian_cv = cv2.GaussianBlur(
            image, (5, 5), 1.2, sigmaY=1.2, borderType=cv2.BORDER_REFLECT_101
        )
        gaussian_sr = sr.gaussian_blur_image(image, 5, 5, 1.2, 1.2)
        gaussian_accuracy = numerical_accuracy(gaussian_cv, gaussian_sr, 255.0)
        accuracy[f"{profile}_gaussian_blur"] = gaussian_accuracy
        if gaussian_accuracy["max_absolute_error"] > 2.0:
            raise AssertionError(
                f"{profile} Gaussian max error {gaussian_accuracy['max_absolute_error']} > 2"
            )
        _, _, cv_gaussian, sr_gaussian = timed_pair(
            lambda: cv2.GaussianBlur(
                image, (5, 5), 1.2, sigmaY=1.2, borderType=cv2.BORDER_REFLECT_101
            ),
            lambda: sr.gaussian_blur_image(image, 5, 5, 1.2, 1.2),
            warmup=args.warmup,
            repeats=repeats,
            seed=109,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )

        sobel_cv = cv2.Sobel(
            gray_cv, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101
        )
        sobel_sr = sr.sobel_image(gray_cv, 1, 0, kernel_size=3)
        sobel_accuracy = numerical_accuracy(sobel_cv, sobel_sr, 2040.0)
        accuracy[f"{profile}_sobel_x"] = sobel_accuracy
        if sobel_accuracy["max_absolute_error"] > 1e-4:
            raise AssertionError(
                f"{profile} Sobel max error {sobel_accuracy['max_absolute_error']} > 1e-4"
            )
        _, _, cv_sobel, sr_sobel = timed_pair(
            lambda: cv2.Sobel(
                gray_cv, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101
            ),
            lambda: sr.sobel_image(gray_cv, 1, 0, kernel_size=3),
            warmup=args.warmup,
            repeats=repeats,
            seed=110,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )

        morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphology_cv = cv2.morphologyEx(
            gray_cv, cv2.MORPH_OPEN, morphology_kernel, borderType=cv2.BORDER_REPLICATE
        )
        morphology_sr = sr.morphology_image(
            gray_cv, "open", 5, 5, shape="rectangle", iterations=1
        )
        morphology_accuracy = numerical_accuracy(morphology_cv, morphology_sr, 255.0)
        accuracy[f"{profile}_morphology_open"] = morphology_accuracy
        if morphology_accuracy["max_absolute_error"] != 0.0:
            raise AssertionError(f"{profile} morphology is not exact")
        _, _, cv_morphology, sr_morphology = timed_pair(
            lambda: cv2.morphologyEx(
                gray_cv, cv2.MORPH_OPEN, morphology_kernel, borderType=cv2.BORDER_REPLICATE
            ),
            lambda: sr.morphology_image(
                gray_cv, "open", 5, 5, shape="rectangle", iterations=1
            ),
            warmup=args.warmup,
            repeats=repeats,
            seed=111,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )

        canny_cv = cv2.Canny(gray_cv, 80.0, 160.0, apertureSize=3, L2gradient=True)
        canny_sr = sr.canny_image(
            gray_cv, 80.0, 160.0, aperture_size=3, l2_gradient=True
        )
        canny_accuracy = binary_accuracy(canny_cv, canny_sr)
        accuracy[f"{profile}_canny"] = canny_accuracy
        if canny_accuracy["f1"] < 0.80:
            raise AssertionError(f"{profile} Canny F1 {canny_accuracy['f1']} < 0.80")
        _, _, cv_canny, sr_canny = timed_pair(
            lambda: cv2.Canny(gray_cv, 80.0, 160.0, apertureSize=3, L2gradient=True),
            lambda: sr.canny_image(
                gray_cv, 80.0, 160.0, aperture_size=3, l2_gradient=True
            ),
            warmup=args.warmup,
            repeats=repeats,
            seed=112,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )

        distance_mask = np.where(gray_cv > 96, 255, 0).astype(np.uint8)
        distance_cv = cv2.distanceTransform(
            distance_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        distance_sr = sr.distance_transform_edt(distance_mask)
        distance_accuracy = numerical_accuracy(
            distance_cv, distance_sr, float(np.hypot(width, height))
        )
        accuracy[f"{profile}_distance_transform_edt"] = distance_accuracy
        if distance_accuracy["max_absolute_error"] > 1e-5:
            raise AssertionError(
                f"{profile} distance-transform max error "
                f"{distance_accuracy['max_absolute_error']} > 1e-5"
            )
        _, _, cv_distance, sr_distance = timed_pair(
            lambda: cv2.distanceTransform(
                distance_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            ),
            lambda: sr.distance_transform_edt(distance_mask),
            warmup=args.warmup,
            repeats=repeats,
            seed=113,
            min_sample_time_ms=MIN_SAMPLE_TIME_MS,
        )

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
            ("gaussian_blur", "opencv", "allocate", cv_gaussian),
            ("gaussian_blur", "spatialrust", "allocate", sr_gaussian),
            ("sobel_x", "opencv", "allocate", cv_sobel),
            ("sobel_x", "spatialrust", "allocate", sr_sobel),
            ("morphology_open", "opencv", "allocate", cv_morphology),
            ("morphology_open", "spatialrust", "allocate", sr_morphology),
            ("canny", "opencv", "allocate", cv_canny),
            ("canny", "spatialrust", "allocate", sr_canny),
            ("distance_transform_edt", "opencv", "allocate", cv_distance),
            ("distance_transform_edt", "spatialrust", "allocate", sr_distance),
        )
        measurements.extend(
            measurement(workload, implementation, mode, width, height, timing)
            for workload, implementation, mode, timing in rows
        )
        speedups[profile] = {
            "resize_allocate": speed_comparison(cv_resize_alloc, sr_resize_alloc),
            "resize_reuse": speed_comparison(cv_resize_reuse, sr_resize_reuse),
            "rgb_to_gray_allocate": speed_comparison(cv_gray_alloc, sr_gray_alloc),
            "rgb_to_gray_reuse": speed_comparison(cv_gray_reuse, sr_gray_reuse),
            "ai_preprocess_allocate": speed_comparison(cv_chw_alloc, sr_chw_alloc),
            "ai_preprocess_reuse_vs_opencv_allocate": speed_comparison(
                cv_chw_reuse_baseline, sr_chw_reuse
            ),
            "gaussian_blur": speed_comparison(cv_gaussian, sr_gaussian),
            "sobel_x": speed_comparison(cv_sobel, sr_sobel),
            "morphology_open": speed_comparison(cv_morphology, sr_morphology),
            "canny": speed_comparison(cv_canny, sr_canny),
            "distance_transform_edt": speed_comparison(cv_distance, sr_distance),
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
            "methodology": {
                "timing_scope": "Python API call, including output allocation according to mode",
                "paired_interleaved": True,
                "random_order_seed_base": 103,
                "minimum_sample_time_ms": MIN_SAMPLE_TIME_MS,
                "gc_disabled_during_samples": True,
                "input_seed": 103,
                "throughput_basis": "input_pixels",
                "thread_policy": "library defaults; exact OpenCV thread count recorded in environment",
            },
            "correctness": correctness,
            "accuracy_metrics": accuracy,
            "speedup_vs_opencv": speedups,
            "measurements": measurements,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
