#!/usr/bin/env python3
"""End-to-end SpatialRust pipeline from Python.

Synthesizes a scan-like room (dominant floor plane + furniture-like objects),
runs the native-Rust MVP pipeline (voxel downsample -> normals -> RANSAC plane
-> Euclidean clustering), prints a reproducible summary, and optionally writes a
labeled point cloud and a top-down PNG preview.

Usage:
    python examples/segment_room.py
    python examples/segment_room.py --out labeled.las --png room.png

Requires only NumPy. The PNG step additionally uses Matplotlib if installed.
"""

from __future__ import annotations

import argparse

import numpy as np

import spatialrust as sr


def synthesize_room(seed: int = 0) -> np.ndarray:
    """Builds a (N, 3) float32 cloud: a 6x6 m floor plus five objects."""
    rng = np.random.default_rng(seed)

    def blob(center, n, scale):
        return (np.asarray(center, np.float32) + rng.normal(0, scale, (n, 3))).astype(np.float32)

    floor = np.column_stack(
        [rng.uniform(0, 6, 9000), rng.uniform(0, 6, 9000), rng.normal(0, 0.01, 9000)]
    ).astype(np.float32)

    objects = np.vstack(
        [
            blob([1.2, 1.2, 0.45], 700, 0.10),  # chair-ish
            blob([4.4, 1.5, 0.75], 900, 0.12),  # table-ish
            blob([2.8, 4.5, 0.60], 600, 0.09),  # box
            blob([5.0, 5.0, 1.10], 800, 0.14),  # cabinet
            blob([0.8, 4.8, 0.30], 400, 0.07),  # small object
        ]
    )

    return np.vstack([floor, objects]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leaf-size", type=float, default=0.08)
    parser.add_argument("--cluster-tolerance", type=float, default=0.30)
    parser.add_argument("--min-cluster-size", type=int, default=40)
    parser.add_argument("--plane-distance", type=float, default=0.05,
                        help="RANSAC inlier distance for the dominant plane [m]")
    parser.add_argument("--policy", default="auto", choices=["auto", "cpu", "cpu-single"])
    parser.add_argument("--out", default=None, help="write labeled cloud (e.g. labeled.las)")
    parser.add_argument("--png", default=None, help="write a top-down PNG preview")
    args = parser.parse_args()

    pts = synthesize_room(args.seed)
    cloud = sr.PointCloud.from_xyz(pts)
    print(f"SpatialRust {sr.__version__}")
    print(f"input cloud      : {len(cloud):>7,} points")

    result = sr.run_pipeline(
        cloud,
        leaf_size=args.leaf_size,
        cluster_tolerance=args.cluster_tolerance,
        min_cluster_size=args.min_cluster_size,
        plane_distance=args.plane_distance,
        policy=args.policy,
    )

    nx, ny, nz = result.plane_normal
    print(f"downsampled      : {len(result.downsampled):>7,} points (leaf={args.leaf_size} m)")
    print(f"dominant plane   : normal=({nx:+.2f}, {ny:+.2f}, {nz:+.2f}), inliers={result.plane_inliers:,}")
    print(f"clusters         : {result.cluster_count}  sizes={result.cluster_sizes}")
    print(f"labeled output   : {len(result.output):>7,} points")

    if args.out:
        sr.write(args.out, result.output)
        print(f"wrote labeled cloud -> {args.out}")

    if args.png:
        _save_png(result, args.png)


def _save_png(result, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG (pip install matplotlib)")
        return

    from matplotlib.colors import ListedColormap

    # SpatialRust neon-on-slate palette, matching the README visuals.
    palette = ["#38bdf8", "#f97316", "#22c55e", "#a855f7",
               "#f472b6", "#facc15", "#2dd4bf", "#60a5fa"]
    bg = "#0f172a"

    xyz = result.output.xyz()
    labels = result.labels()
    colors = ListedColormap(palette)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.scatter(
        xyz[:, 0], xyz[:, 1],
        c=labels % len(palette), cmap=colors, vmin=0, vmax=len(palette) - 1,
        s=10, linewidths=0, alpha=0.95,
    )
    ax.set_aspect("equal")
    ax.set_title(
        f"SpatialRust — {result.cluster_count} clusters from a Python call",
        color="#e2e8f0", fontsize=12, pad=12,
    )
    ax.set_xlabel("x [m]", color="#94a3b8")
    ax.set_ylabel("y [m]", color="#94a3b8")
    ax.tick_params(colors="#475569")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    print(f"wrote preview -> {path}")


if __name__ == "__main__":
    main()
