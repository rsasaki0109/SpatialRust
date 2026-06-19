#!/usr/bin/env python3
"""SpatialRust as a point-cloud preprocessing front-end for learned models.

Runs the native point-cloud pipeline on one scan and emits the tensors a model
would consume:

  load -> Statistical Outlier Removal -> unit-sphere normalize -> Farthest Point
  Sampling -> {voxel occupancy grid, LiDAR range image, k-NN graph edge_index}

With --png it renders a four-panel figure of the representations.

Usage:
    python examples/ml_preprocess.py
    python examples/ml_preprocess.py --input scan.pcd --png ml.png

Requires NumPy; the PNG step additionally uses Matplotlib.
"""
from __future__ import annotations

import argparse

import numpy as np

import spatialrust as sr

BG, FG = "#0f172a", "#e2e8f0"


def synth(seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    blob = lambda c, n, s: (np.asarray(c, np.float32) + r.normal(0, s, (n, 3))).astype(np.float32)
    return np.vstack([
        blob([0.0, 0.0, 0.0], 2500, [0.30, 0.30, 0.30]),
        blob([2.5, 0.3, 0.0], 1800, [0.08, 0.08, 0.60]),
        blob([1.0, 2.5, 0.0], 2000, [0.50, 0.12, 0.18]),
        blob([3.2, 2.8, 0.0], 1500, [0.22, 0.22, 0.05]),
        r.uniform([-1, -1, -1], [5.5, 3.8, 1.2], (500, 3)).astype(np.float32),  # speckle
    ]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="scan file (PCD/PLY/LAS/COPC)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048, help="FPS target point count")
    parser.add_argument("--knn", type=int, default=16)
    parser.add_argument("--png", default=None)
    args = parser.parse_args()

    print(f"SpatialRust {sr.__version__}")
    if args.input:
        cloud = sr.read(args.input)
        print(f"loaded             : {len(cloud):,} points from {args.input}")
    else:
        cloud = sr.PointCloud.from_xyz(synth(args.seed))
        print(f"synthesized scan   : {len(cloud):,} points")

    # 1. Clean and normalize into the unit sphere (canonical ML normalization).
    cleaned = sr.statistical_outlier_removal(cloud, 16, 2.0)
    normalized = sr.normalize_unit_sphere(cleaned)
    print(f"after SOR+normalize: {len(normalized):,} points in the unit sphere")

    # 2. Farthest Point Sampling to a fixed token count (PointNet++ style).
    sampled = sr.farthest_point_sampling(normalized, args.samples)
    pts = sampled.xyz()
    print(f"after FPS          : {len(sampled):,} points (target {args.samples})")

    # 3. Build the model-ready representations.
    occ, origin, vsize = sr.voxelize(sampled, voxel_size=0.06, mode="occupancy")
    rimg = sr.range_image(sampled, width=256, height=64, fov_up_deg=20.0, fov_down_deg=-20.0)
    edge_index = sr.knn_graph(sampled, args.knn)

    print("\nmodel-ready tensors:")
    print(f"  points        : {pts.shape}      float32   (N, 3)")
    print(f"  voxel grid    : {occ.shape}  float32   (nz, ny, nx) occupancy")
    print(f"  range image   : {rimg.shape}     float32   (H, W) depth")
    print(f"  edge_index    : {edge_index.shape}    int32     (2, E) PyG graph")

    if args.png:
        _save_png(pts, occ, rimg, edge_index, args.png)


def _save_png(pts, occ, rimg, edge_index, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG (pip install matplotlib)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=130)
    fig.patch.set_facecolor(BG)

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=12)
        ax.tick_params(colors="#475569")
        for s in ax.spines.values():
            s.set_color("#334155")

    # (a) FPS-sampled points, colored by height.
    ax = axes[0, 0]
    ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", s=5, linewidths=0)
    ax.set_aspect("equal")
    style(ax, f"1. SOR + normalize + FPS  ({len(pts)} pts)")

    # (b) voxel occupancy: max-projection down the z axis.
    ax = axes[0, 1]
    ax.imshow(occ.max(axis=0), origin="lower", cmap="magma")
    style(ax, f"2. Voxel occupancy  {occ.shape} (z-max)")

    # (c) range image.
    ax = axes[1, 0]
    ax.imshow(rimg, aspect="auto", cmap="cividis")
    style(ax, f"3. LiDAR range image  {rimg.shape}")

    # (d) k-NN graph edges.
    ax = axes[1, 1]
    src, dst = edge_index
    seg = np.stack([pts[src, :2], pts[dst, :2]], axis=1)
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(seg, colors="#38bdf8", linewidths=0.15, alpha=0.5))
    ax.scatter(pts[:, 0], pts[:, 1], c="#f97316", s=3, linewidths=0)
    ax.set_aspect("equal")
    ax.autoscale()
    style(ax, f"4. k-NN graph  edge_index {edge_index.shape}")

    fig.suptitle("SpatialRust: point cloud -> model-ready tensors", color=FG, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    print(f"wrote preview -> {path}")


if __name__ == "__main__":
    main()
