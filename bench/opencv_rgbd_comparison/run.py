"""Numerical and timing comparison with OpenCV rgbd.depthTo3d.

Gates SpatialRust processing-speed wins on:
1. dense HxWx3 XYZ alloc + into vs ``cv.rgbd.depthTo3d``
2. colored PointCloud vs OpenCV depthTo3d + NumPy mask/color gather
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import (
    emit_report,
    environment,
    make_report,
    percentile,
    timed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--groups", type=int, default=5)
    return parser.parse_args()


def summarize_groups(
    group_medians_ms: list[float], raw_samples_ms: list[float], args: argparse.Namespace
) -> dict[str, object]:
    return {
        "unit": "ms",
        "warmup_per_group": args.warmup,
        "repeats_per_group": args.repeats,
        "groups": args.groups,
        "median": statistics.median(group_medians_ms),
        "p95": percentile(raw_samples_ms, 0.95),
        "min": min(raw_samples_ms),
        "max": max(raw_samples_ms),
        "group_medians": group_medians_ms,
        "samples": raw_samples_ms,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.groups < 1:
        raise ValueError("warmup must be non-negative; repeats and groups must be positive")
    if not hasattr(cv2, "rgbd"):
        raise RuntimeError("OpenCV rgbd module missing; install opencv-contrib-python<5")

    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    height, width = 240, 320
    yy, xx = np.mgrid[:height, :width]
    depth = (1.0 + xx * 0.001 + yy * 0.0005).astype(np.float32)
    depth[::31, ::29] = np.nan
    color = np.empty((height, width, 3), dtype=np.uint8)
    color[..., 0] = xx % 256
    color[..., 1] = yy % 256
    color[..., 2] = 127
    fx, fy, cx, cy = 280.0, 282.0, 159.5, 119.5
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    cv_points = cv2.rgbd.depthTo3d(depth, intrinsics)
    sr_dense = sr.depth_to_xyz(depth, fx, fy, cx, cy)
    sr_cloud = sr.rgbd_to_point_cloud(depth, color, fx, fy, cx, cy)
    mask = np.isfinite(depth) & (depth > 0)
    expected = cv_points[mask].astype(np.float32)
    actual = sr_cloud.xyz()
    if expected.shape != actual.shape:
        raise AssertionError(
            f"shape mismatch: OpenCV={expected.shape}, SpatialRust cloud={actual.shape}"
        )
    max_error_cloud = float(np.max(np.abs(expected - actual)))
    max_error_dense = float(np.nanmax(np.abs(cv_points.astype(np.float32) - sr_dense)))
    if max_error_cloud > 1e-5:
        raise AssertionError(f"cloud XYZ error {max_error_cloud:.3e} exceeds 1e-5 m")
    if max_error_dense > 1e-5:
        raise AssertionError(f"dense XYZ error {max_error_dense:.3e} exceeds 1e-5 m")

    out_cv = np.empty((height, width, 3), dtype=np.float32)
    out_sr = np.empty((height, width, 3), dtype=np.float32)

    def opencv_colored_cloud():
        pts = cv2.rgbd.depthTo3d(depth, intrinsics)
        valid = np.isfinite(depth) & (depth > 0)
        return pts[valid], color[valid]

    alloc_ratios = []
    into_ratios = []
    cloud_ratios = []
    cv_alloc_samples = []
    sr_alloc_samples = []
    cv_into_samples = []
    sr_into_samples = []
    cv_cloud_samples = []
    sr_cloud_samples = []
    cv_alloc_raw = []
    sr_alloc_raw = []
    cv_into_raw = []
    sr_into_raw = []
    cv_cloud_raw = []
    sr_cloud_raw = []
    for _ in range(args.groups):
        _, cv_alloc_stats = timed(
            lambda: cv2.rgbd.depthTo3d(depth, intrinsics),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _, sr_alloc_stats = timed(
            lambda: sr.depth_to_xyz(depth, fx, fy, cx, cy),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _, cv_into_stats = timed(
            lambda: cv2.rgbd.depthTo3d(depth, intrinsics, out_cv),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _, sr_into_stats = timed(
            lambda: sr.depth_to_xyz(depth, fx, fy, cx, cy, out=out_sr),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _, cv_cloud_stats = timed(
            opencv_colored_cloud, warmup=args.warmup, repeats=args.repeats
        )
        _, sr_cloud_stats = timed(
            lambda: sr.rgbd_to_point_cloud(depth, color, fx, fy, cx, cy),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        cv_alloc = float(cv_alloc_stats["median"])
        sr_alloc = float(sr_alloc_stats["median"])
        cv_into = float(cv_into_stats["median"])
        sr_into = float(sr_into_stats["median"])
        cv_cloud = float(cv_cloud_stats["median"])
        sr_cloud_s = float(sr_cloud_stats["median"])
        cv_alloc_raw.extend(cv_alloc_stats["samples"])
        sr_alloc_raw.extend(sr_alloc_stats["samples"])
        cv_into_raw.extend(cv_into_stats["samples"])
        sr_into_raw.extend(sr_into_stats["samples"])
        cv_cloud_raw.extend(cv_cloud_stats["samples"])
        sr_cloud_raw.extend(sr_cloud_stats["samples"])
        cv_alloc_samples.append(cv_alloc)
        sr_alloc_samples.append(sr_alloc)
        cv_into_samples.append(cv_into)
        sr_into_samples.append(sr_into)
        cv_cloud_samples.append(cv_cloud)
        sr_cloud_samples.append(sr_cloud_s)
        alloc_ratios.append(cv_alloc / sr_alloc)
        into_ratios.append(cv_into / sr_into)
        cloud_ratios.append(cv_cloud / sr_cloud_s)

    cv_alloc = statistics.median(cv_alloc_samples)
    sr_alloc = statistics.median(sr_alloc_samples)
    cv_into = statistics.median(cv_into_samples)
    sr_into = statistics.median(sr_into_samples)
    cv_cloud = statistics.median(cv_cloud_samples)
    sr_cloud_s = statistics.median(sr_cloud_samples)
    alloc_ratio = statistics.median(alloc_ratios)
    into_ratio = statistics.median(into_ratios)
    cloud_ratio = statistics.median(cloud_ratios)
    print(f"points: {len(actual)}")
    print(f"max dense XYZ error: {max_error_dense:.3e} m")
    print(f"max cloud XYZ error: {max_error_cloud:.3e} m")
    print(f"OpenCV depthTo3d (alloc):             {cv_alloc:.3f} ms")
    print(f"SpatialRust depth_to_xyz (alloc):     {sr_alloc:.3f} ms  ({alloc_ratio:.2f}× vs OpenCV)")
    print(f"OpenCV depthTo3d (into):              {cv_into:.3f} ms")
    print(f"SpatialRust depth_to_xyz (into out=): {sr_into:.3f} ms  ({into_ratio:.2f}× vs OpenCV)")
    print(f"OpenCV depth+mask+color:              {cv_cloud:.3f} ms")
    print(f"SpatialRust rgbd_to_point_cloud:      {sr_cloud_s:.3f} ms  ({cloud_ratio:.2f}× vs OpenCV)")
    if alloc_ratio < 1.0 or into_ratio < 1.0 or cloud_ratio < 1.0:
        raise SystemExit(
            "SpatialRust slower than OpenCV on a gated path "
            f"(alloc {alloc_ratio:.2f}×, into {into_ratio:.2f}×, cloud {cloud_ratio:.2f}×)"
        )

    measurements = [
        {
            "workload": workload,
            "implementation": implementation,
            "mode": mode,
            "width": width,
            "height": height,
            "timing": summarize_groups(group_medians, raw_samples, args),
        }
        for workload, implementation, mode, group_medians, raw_samples in (
            ("depth_to_xyz", "opencv", "allocate", cv_alloc_samples, cv_alloc_raw),
            ("depth_to_xyz", "spatialrust", "allocate", sr_alloc_samples, sr_alloc_raw),
            ("depth_to_xyz", "opencv", "reuse", cv_into_samples, cv_into_raw),
            ("depth_to_xyz", "spatialrust", "reuse", sr_into_samples, sr_into_raw),
            ("rgbd_to_point_cloud", "opencv", "allocate", cv_cloud_samples, cv_cloud_raw),
            ("rgbd_to_point_cloud", "spatialrust", "allocate", sr_cloud_samples, sr_cloud_raw),
        )
    ]
    environment_receipt = environment(
        opencv_version=cv2.__version__, spatialrust_version=sr.__version__
    )
    environment_receipt["opencv_threads"] = cv2.getNumThreads()
    environment_receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-rgbd-performance",
        kind="performance",
        status="pass",
        environment_receipt=environment_receipt,
        results={
            "correctness": {
                "points": len(actual),
                "max_dense_xyz_error_m": max_error_dense,
                "max_cloud_xyz_error_m": max_error_cloud,
            },
            "speedup_vs_opencv": {
                "depth_to_xyz_allocate": alloc_ratio,
                "depth_to_xyz_reuse": into_ratio,
                "rgbd_to_point_cloud_allocate": cloud_ratio,
            },
            "measurements": measurements,
        },
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
