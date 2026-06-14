#!/usr/bin/env python3
"""Generates a deterministic synthetic scan and writes it as a PCD file.

Used by the SpatialRust-vs-PCL benchmark so both libraries process the exact
same input. The scene is a room (floor + walls + furniture) plus speckle, which
exercises voxel downsampling, normal estimation, and outlier removal.
"""
from __future__ import annotations

import argparse

import numpy as np

import spatialrust as sr


def synth(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Scale per-surface counts to roughly hit the requested total.
    unit = max(n_points // 20, 1)

    def plane(n, fx, fy, fz):
        a = rng.uniform(0, 5, n)
        b = rng.uniform(0, 5, n)
        return np.column_stack([fx(a, b), fy(a, b), fz(a, b)]).astype(np.float32)

    nz = lambda a: rng.normal(0, 0.004, len(a))
    floor = plane(unit * 8, lambda a, b: a, lambda a, b: b, lambda a, b: nz(a))
    wall_x = plane(unit * 4, lambda a, b: nz(a), lambda a, b: a, lambda a, b: b)
    wall_y = plane(unit * 4, lambda a, b: a, lambda a, b: nz(a), lambda a, b: b)

    def blob(center, n, scale):
        return (np.asarray(center, np.float32) + rng.normal(0, scale, (n, 3))).astype(np.float32)

    furniture = np.vstack([blob([1.5, 1.2, 0.4], unit * 2, 0.1), blob([3.5, 3.0, 0.7], unit * 2, 0.12)])
    speckle = rng.uniform([-0.5, -0.5, -0.5], [5.5, 5.5, 3.0], (unit, 3)).astype(np.float32)
    return np.vstack([floor, wall_x, wall_y, furniture, speckle]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/tmp/bench_cloud.pcd")
    args = parser.parse_args()

    pts = synth(args.points, args.seed)
    cloud = sr.PointCloud.from_xyz(pts)
    sr.write(args.out, cloud)
    print(f"wrote {len(cloud):,} points -> {args.out}")


if __name__ == "__main__":
    main()
