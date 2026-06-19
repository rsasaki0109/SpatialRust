#!/usr/bin/env python3
"""Times Open3D point-cloud operations on a PCD file.

Prints `operation,seconds,output_points` lines on stdout so run.sh can compare
the results with SpatialRust's bench_ops example.
"""
from __future__ import annotations

import argparse
import sys
import time

import open3d as o3d


def seconds_since(start: float) -> float:
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", help="Input PCD file")
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(args.cloud)
    if cloud.is_empty():
        raise SystemExit(f"failed to read non-empty point cloud from {args.cloud}")
    print(f"loaded {len(cloud.points)} points from {args.cloud}", file=sys.stderr)

    start = time.perf_counter()
    downsampled = cloud.voxel_down_sample(voxel_size=0.05)
    print(f"voxel_downsample,{seconds_since(start):.4f},{len(downsampled.points)}")

    normals_cloud = o3d.geometry.PointCloud(cloud)
    start = time.perf_counter()
    normals_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10),
        fast_normal_computation=True,
    )
    print(f"normal_estimation,{seconds_since(start):.4f},{len(normals_cloud.normals)}")

    start = time.perf_counter()
    statistical_cleaned, _ = cloud.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.0)
    print(
        "statistical_outlier_removal,"
        f"{seconds_since(start):.4f},{len(statistical_cleaned.points)}"
    )

    start = time.perf_counter()
    radius_cleaned, _ = cloud.remove_radius_outlier(nb_points=4, radius=0.1)
    print(f"radius_outlier_removal,{seconds_since(start):.4f},{len(radius_cleaned.points)}")


if __name__ == "__main__":
    main()
