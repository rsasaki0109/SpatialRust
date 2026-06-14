#!/usr/bin/env python3
"""Renders the animated GIFs used in the SpatialRust README.

Produces two GIFs under docs/assets/:
  - clusters_rotating.gif : a multi-object scene rotated and colored by DBSCAN.
  - voxelize_rotating.gif  : the same scene voxelized into an occupancy grid.

Requires NumPy, Matplotlib, and Pillow.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import spatialrust as sr

BG = "#0f172a"
FG = "#e2e8f0"


def synth(seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    blob = lambda c, n, s: (np.asarray(c, np.float32) + r.normal(0, s, (n, 3))).astype(np.float32)
    return np.vstack([
        blob([0.0, 0.0, 0.0], 1500, [0.30, 0.30, 0.30]),   # sphere
        blob([2.5, 0.3, 0.0], 1200, [0.08, 0.08, 0.60]),   # pillar
        blob([1.0, 2.5, 0.0], 1400, [0.50, 0.12, 0.18]),   # slab
        blob([3.2, 2.8, 0.0], 1000, [0.22, 0.22, 0.05]),   # disk
        blob([4.5, 1.2, 0.0], 900, [0.15, 0.15, 0.15]),    # ball
        r.uniform([-1, -1, -1], [5.5, 3.8, 1.2], (300, 3)).astype(np.float32),  # speckle
    ]).astype(np.float32)


def style_3d(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 0.5))
    except Exception:
        pass


def clusters_gif(path, frames, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cloud = sr.PointCloud.from_xyz(synth())
    clean = sr.statistical_outlier_removal(cloud, 16, 2.0)
    small = sr.voxel_downsample(clean, 0.06, "auto")
    seg = sr.dbscan(small, eps=0.3, min_points=8)
    pts = small.xyz()
    labels = seg.labels()
    noise = labels < 0

    fig = plt.figure(figsize=(5, 5), dpi=80)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")

    def draw(i):
        ax.clear()
        ax.scatter(pts[noise, 0], pts[noise, 1], pts[noise, 2], c="#475569", s=3, alpha=0.4)
        p = pts[~noise]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=labels[~noise] % 20, cmap="tab20", s=6)
        style_3d(ax, f"DBSCAN: {seg.cluster_count} clusters")
        ax.view_init(elev=28, azim=i * 360 / frames)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


def voxelize_gif(path, frames, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cloud = sr.PointCloud.from_xyz(synth())
    grid, origin, vsize = sr.voxelize(cloud, voxel_size=0.35, mode="occupancy")
    # grid is (nz, ny, nx); matplotlib voxels wants (nx, ny, nz).
    occ = np.transpose(grid, (2, 1, 0)) > 0
    colors = np.empty(occ.shape + (4,), dtype=float)
    colors[occ] = (0.22, 0.74, 0.93, 0.9)  # cyan

    fig = plt.figure(figsize=(5, 5), dpi=80)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")

    def draw(i):
        ax.clear()
        ax.voxels(occ, facecolors=colors, edgecolor=(1, 1, 1, 0.12))
        style_3d(ax, f"Voxelized: {int(occ.sum())} voxels")
        ax.view_init(elev=30, azim=i * 360 / frames)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.normpath(os.path.join(here, "..", "..", "..", "docs", "assets"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--outdir", default=assets)
    args = parser.parse_args()

    clusters_gif(os.path.join(args.outdir, "clusters_rotating.gif"), args.frames, args.fps)
    voxelize_gif(os.path.join(args.outdir, "voxelize_rotating.gif"), args.frames, args.fps)


if __name__ == "__main__":
    main()
