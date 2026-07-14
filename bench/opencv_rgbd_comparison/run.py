"""Numerical and timing comparison with OpenCV rgbd.depthTo3d."""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np
import spatialrust as sr


def timed(call, repeats: int = 20):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        values.append(time.perf_counter() - start)
    return result, statistics.median(values)


def main() -> None:
    if not hasattr(cv2, "rgbd"):
        raise RuntimeError("OpenCV rgbd module missing; install opencv-contrib-python")

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

    cv_points, cv_seconds = timed(lambda: cv2.rgbd.depthTo3d(depth, intrinsics))
    sr_cloud, sr_seconds = timed(
        lambda: sr.rgbd_to_point_cloud(depth, color, fx, fy, cx, cy)
    )
    mask = np.isfinite(depth) & (depth > 0)
    expected = cv_points[mask].astype(np.float32)
    actual = sr_cloud.xyz()
    if expected.shape != actual.shape:
        raise AssertionError(f"shape mismatch: OpenCV={expected.shape}, SpatialRust={actual.shape}")
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-5:
        raise AssertionError(f"maximum XYZ error {max_error:.3e} exceeds 1e-5 m")

    print(f"points: {len(actual)}")
    print(f"max XYZ error: {max_error:.3e} m")
    print(f"SpatialRust median: {sr_seconds * 1e3:.3f} ms")
    print(f"OpenCV median:      {cv_seconds * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
