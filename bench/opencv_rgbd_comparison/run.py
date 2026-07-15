"""Numerical and timing comparison with OpenCV rgbd.depthTo3d.

Gates SpatialRust processing-speed wins on:
1. dense HxWx3 XYZ alloc + into vs ``cv.rgbd.depthTo3d``
2. colored PointCloud vs OpenCV depthTo3d + NumPy mask/color gather
"""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np
import spatialrust as sr


def timed(call, *, warmup: int = 25, repeats: int = 100):
    for _ in range(warmup):
        call()
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        values.append(time.perf_counter() - start)
    return result, statistics.median(values)


def main() -> None:
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
    for _ in range(5):
        _, cv_alloc = timed(lambda: cv2.rgbd.depthTo3d(depth, intrinsics), warmup=8, repeats=40)
        _, sr_alloc = timed(lambda: sr.depth_to_xyz(depth, fx, fy, cx, cy), warmup=8, repeats=40)
        _, cv_into = timed(lambda: cv2.rgbd.depthTo3d(depth, intrinsics, out_cv), warmup=8, repeats=40)
        _, sr_into = timed(
            lambda: sr.depth_to_xyz(depth, fx, fy, cx, cy, out=out_sr), warmup=8, repeats=40
        )
        _, cv_cloud = timed(opencv_colored_cloud, warmup=8, repeats=40)
        _, sr_cloud_s = timed(
            lambda: sr.rgbd_to_point_cloud(depth, color, fx, fy, cx, cy), warmup=8, repeats=40
        )
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
    print(f"OpenCV depthTo3d (alloc):             {cv_alloc * 1e3:.3f} ms")
    print(f"SpatialRust depth_to_xyz (alloc):     {sr_alloc * 1e3:.3f} ms  ({alloc_ratio:.2f}× vs OpenCV)")
    print(f"OpenCV depthTo3d (into):              {cv_into * 1e3:.3f} ms")
    print(f"SpatialRust depth_to_xyz (into out=): {sr_into * 1e3:.3f} ms  ({into_ratio:.2f}× vs OpenCV)")
    print(f"OpenCV depth+mask+color:              {cv_cloud * 1e3:.3f} ms")
    print(f"SpatialRust rgbd_to_point_cloud:      {sr_cloud_s * 1e3:.3f} ms  ({cloud_ratio:.2f}× vs OpenCV)")
    if alloc_ratio < 1.0 or into_ratio < 1.0 or cloud_ratio < 1.0:
        raise SystemExit(
            "SpatialRust slower than OpenCV on a gated path "
            f"(alloc {alloc_ratio:.2f}×, into {into_ratio:.2f}×, cloud {cloud_ratio:.2f}×)"
        )


if __name__ == "__main__":
    main()
