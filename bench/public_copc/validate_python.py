#!/usr/bin/env python3
"""Validate the SpatialRust Python pipeline on the public PCL table scan."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import spatialrust as sr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "target" / "bench-data" / "table_scene_lms400.pcd"


def rigid(yaw_degrees: float, translation: list[float]) -> np.ndarray:
    radians = np.radians(yaw_degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(
        [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float32)]
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--leaf", type=float, default=0.02)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(
            f"{args.input} is missing; run: python bench/public_copc/run.py --fetch-only"
        )

    started = time.perf_counter()
    cloud = sr.read(str(args.input))
    cleaned = sr.statistical_outlier_removal(cloud, k_neighbors=16, std_mul=2.0)
    downsampled = sr.voxel_downsample(cleaned, leaf_size=args.leaf, policy="auto")
    segmentation = sr.dbscan(downsampled, eps=args.leaf * 5.0, min_points=8)

    registration_leaf = max(args.leaf * 5.0, 0.1)
    target = sr.voxel_downsample(cleaned, leaf_size=registration_leaf, policy="auto")
    target_xyz = target.xyz()
    misalignment = rigid(20.0, [0.6, -0.4, 0.05])
    source_xyz = apply(misalignment, target_xyz)
    source = sr.PointCloud.from_xyz(source_xyz)
    truth = np.linalg.inv(misalignment)

    global_result = sr.register_fpfh_ransac(
        source,
        target,
        feature_radius=registration_leaf * 5.0,
        max_correspondence_distance=registration_leaf * 1.5,
        ransac_iterations=8000,
    )
    global_error = float(np.abs(global_result.transform() - truth).max())
    coarse = sr.PointCloud.from_xyz(apply(global_result.transform(), source_xyz))
    icp_result = sr.register_icp(coarse, target, registration_leaf * 2.0, 80)
    refined_transform = icp_result.transform() @ global_result.transform()
    refined_error = float(np.abs(refined_transform - truth).max())

    assert len(cloud) == 460_400
    assert 0 < len(cleaned) < len(cloud)
    assert 0 < len(downsampled) < len(cleaned)
    assert segmentation.cluster_count >= 1
    assert global_result.converged
    assert global_error < 0.2
    assert icp_result.converged
    assert refined_error < 1e-4

    receipt = {
        "spatialrust_version": sr.__version__,
        "dataset": str(args.input.resolve()),
        "dataset_bytes": args.input.stat().st_size,
        "dataset_sha256": sha256(args.input),
        "source_points": len(cloud),
        "cleaned_points": len(cleaned),
        "removed_points": len(cloud) - len(cleaned),
        "downsampled_points": len(downsampled),
        "cluster_count": segmentation.cluster_count,
        "noise_points": segmentation.noise_count,
        "registration_points": len(target),
        "fpfh_converged": global_result.converged,
        "fpfh_max_transform_error": global_error,
        "icp_converged": icp_result.converged,
        "refined_max_transform_error": refined_error,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
