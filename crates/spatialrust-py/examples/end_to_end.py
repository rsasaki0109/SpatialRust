#!/usr/bin/env python3
"""End-to-end SpatialRust demo: clean -> cluster -> globally register -> refine.

Runs a realistic point-cloud workflow with the SpatialRust Python bindings:

  1. Load a scan (``--input``) or synthesize a noisy room with speckle outliers.
  2. Statistical Outlier Removal to drop scanner speckle.
  3. Voxel downsample.
  4. DBSCAN clustering (with explicit noise labeling).
  5. Make a second scan by applying a large, unknown misalignment, then recover
     it with FPFH + RANSAC global registration (no initial guess) followed by
     ICP refinement -- the standard coarse-to-fine pipeline.

With ``--png`` it renders a four-panel figure of each stage.

Usage:
    python examples/end_to_end.py
    python examples/end_to_end.py --input scan.pcd --png demo.png

``--input`` accepts any format SpatialRust reads (PCD/PLY/LAS/COPC). Real LiDAR
scans (e.g. the KITTI odometry set, or any room scan exported to PLY) work
directly; the synthetic fallback keeps the demo runnable with no data on hand.

Requires NumPy; the PNG step additionally uses Matplotlib.
"""

from __future__ import annotations

import argparse

import numpy as np

import spatialrust as sr


def synthesize_scan(seed: int = 0) -> np.ndarray:
    """A scene of five spatially separated objects of distinct shapes (sphere,
    pillar, slab, disk, ball) plus stray speckle. The separation gives DBSCAN
    clean clusters, and the varied shapes give feature-based global registration
    distinctive geometry to lock onto."""
    rng = np.random.default_rng(seed)

    def blob(center, n, scale):
        return (np.asarray(center, np.float32) + rng.normal(0, scale, (n, 3))).astype(np.float32)

    objects = [
        blob([0.0, 0.0, 0.0], 1500, [0.30, 0.30, 0.30]),  # sphere
        blob([2.5, 0.3, 0.0], 1200, [0.08, 0.08, 0.60]),  # tall pillar
        blob([1.0, 2.5, 0.0], 1400, [0.50, 0.12, 0.18]),  # elongated slab
        blob([3.2, 2.8, 0.0], 1000, [0.22, 0.22, 0.05]),  # flat disk
        blob([4.5, 1.2, 0.0], 900, [0.15, 0.15, 0.15]),  # small ball
    ]

    # Stray speckle scattered through the volume -- the outliers SOR removes.
    speckle = rng.uniform([-1.0, -1.0, -1.0], [5.5, 3.8, 1.2], (400, 3)).astype(np.float32)

    return np.vstack([*objects, speckle]).astype(np.float32)


def rigid(yaw_deg: float, t) -> np.ndarray:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    m[:3, 3] = np.asarray(t, np.float32)
    return m


def apply(transform: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.c_[pts, np.ones(len(pts), np.float32)]
    return (h @ transform.T)[:, :3].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="scan file (PCD/PLY/LAS/COPC)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaw", type=float, default=20.0, help="misalignment yaw [deg]")
    parser.add_argument("--leaf", type=float, default=0.05, help="voxel leaf size")
    parser.add_argument("--png", default=None, help="write a four-panel preview")
    args = parser.parse_args()

    print(f"SpatialRust {sr.__version__}")

    # 1. Load or synthesize.
    if args.input:
        cloud = sr.read(args.input)
        raw = cloud.xyz()
        print(f"loaded            : {len(cloud):,} points from {args.input}")
    else:
        raw = synthesize_scan(args.seed)
        cloud = sr.PointCloud.from_xyz(raw)
        print(f"synthesized scan  : {len(cloud):,} points (incl. 400 speckle outliers)")

    # 2. Statistical Outlier Removal.
    cleaned = sr.statistical_outlier_removal(cloud, k_neighbors=16, std_mul=2.0)
    cleaned_xyz = cleaned.xyz()
    print(f"after SOR         : {len(cleaned):,} points ({len(cloud) - len(cleaned):,} removed)")

    # 3. Voxel downsample.
    small = sr.voxel_downsample(cleaned, leaf_size=args.leaf, policy="auto")
    print(f"after downsample  : {len(small):,} points (leaf={args.leaf})")

    # 4. DBSCAN clustering.
    seg = sr.dbscan(small, eps=args.leaf * 5.0, min_points=8)
    print(f"DBSCAN            : {seg.cluster_count} clusters, {seg.noise_count} noise points")

    # 5. Coarse-to-fine registration on a misaligned copy. Global feature
    #    matching is brute-force in feature space, so register a coarser cloud.
    reg_leaf = max(args.leaf * 2.0, 0.1)
    target = sr.voxel_downsample(cleaned, leaf_size=reg_leaf, policy="auto")
    target_xyz = target.xyz()
    misalign = rigid(args.yaw, [0.6, -0.4, 0.05])
    source_xyz = apply(misalign, target_xyz)
    source = sr.PointCloud.from_xyz(source_xyz)
    inv_truth = np.linalg.inv(misalign)
    print(f"registration cloud: {len(target):,} points (leaf={reg_leaf:.2f})")

    def err(T):
        return float(np.abs(T - inv_truth).max())

    glob = sr.register_fpfh_ransac(source, target, feature_radius=reg_leaf * 5.0,
                                   max_correspondence_distance=reg_leaf * 1.5,
                                   ransac_iterations=8000)
    print(f"FPFH global       : converged={glob.converged} max|T-T*|={err(glob.transform()):.4f} (coarse)")

    # Refine from the global estimate: pre-transform the source, then run ICP.
    coarse_xyz = apply(glob.transform(), source_xyz)
    coarse = sr.PointCloud.from_xyz(coarse_xyz)
    icp = sr.register_icp(coarse, target, reg_leaf * 2.0, 80)
    refined_T = icp.transform() @ glob.transform()
    aligned_xyz = apply(refined_T, source_xyz)
    print(f"+ ICP refine      : converged={icp.converged} max|T-T*|={err(refined_T):.4f}")

    if args.png:
        _save_png(raw, cleaned_xyz, small.xyz(), seg, target_xyz, source_xyz, aligned_xyz, args.png)


def _save_png(raw, cleaned, small, seg, target, source, aligned, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG (pip install matplotlib)")
        return

    bg = "#0f172a"
    labels = seg.labels()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=140)
    fig.patch.set_facecolor(bg)

    def style(ax, title):
        ax.set_facecolor(bg)
        ax.set_aspect("equal")
        ax.set_title(title, color="#e2e8f0", fontsize=12)
        ax.tick_params(colors="#475569")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # (a) raw scan with outliers highlighted.
    ax = axes[0, 0]
    ax.scatter(raw[:, 0], raw[:, 1], c="#64748b", s=2, linewidths=0, alpha=0.5)
    style(ax, f"1. Raw scan ({len(raw):,} pts, with speckle)")

    # (b) cleaned + downsampled.
    ax = axes[0, 1]
    ax.scatter(cleaned[:, 0], cleaned[:, 1], c="#38bdf8", s=2, linewidths=0, alpha=0.5)
    style(ax, f"2. SOR + downsample ({len(small):,} pts)")

    # (c) DBSCAN clusters (noise in grey).
    ax = axes[1, 0]
    if labels is not None:
        noise = labels < 0
        ax.scatter(small[noise, 0], small[noise, 1], c="#475569", s=3, linewidths=0, alpha=0.6)
        pts = small[~noise]
        lab = labels[~noise]
        ax.scatter(pts[:, 0], pts[:, 1], c=lab % 20, cmap="tab20", s=4, linewidths=0)
    style(ax, f"3. DBSCAN ({seg.cluster_count} clusters)")

    # (d) registration before/after.
    ax = axes[1, 1]
    ax.scatter(target[:, 0], target[:, 1], c="#38bdf8", s=3, linewidths=0, alpha=0.5, label="target")
    ax.scatter(source[:, 0], source[:, 1], c="#f97316", s=3, linewidths=0, alpha=0.3, label="source")
    ax.scatter(aligned[:, 0], aligned[:, 1], c="#22c55e", s=3, linewidths=0, alpha=0.6, label="aligned")
    leg = ax.legend(loc="upper right", facecolor=bg, edgecolor="#334155", labelcolor="#e2e8f0")
    leg.get_frame().set_alpha(0.8)
    style(ax, "4. FPFH global + ICP refine")

    fig.suptitle("SpatialRust end-to-end pipeline", color="#e2e8f0", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    print(f"wrote preview -> {path}")


if __name__ == "__main__":
    main()
