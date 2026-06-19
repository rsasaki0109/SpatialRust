#!/usr/bin/env python3
"""Two-scan registration demo using the SpatialRust Python bindings.

Loads a scan or synthesizes a small fallback scene, makes a second "scan" by
applying a known rigid misalignment, then aligns it back with each registration
backend (point-to-plane ICP, GICP, NDT) and reports the recovered error.
Optionally renders a before/after top-down preview.

Usage:
    python examples/register_scans.py
    python examples/register_scans.py --input scan.pcd
    python examples/register_scans.py --png registration.png

Requires NumPy; the PNG step additionally uses Matplotlib.
"""

from __future__ import annotations

import argparse

import numpy as np

import spatialrust as sr


def synthesize_room(seed: int = 0) -> np.ndarray:
    """A room with a floor, two perpendicular walls, and a few objects."""
    rng = np.random.default_rng(seed)

    def grid(n, fa, fb, fc):
        a = rng.uniform(0, 4, n)
        b = rng.uniform(0, 4, n)
        return np.column_stack([fa(a, b), fb(a, b), fc(a, b)]).astype(np.float32)

    floor = grid(6000, lambda a, b: a, lambda a, b: b, lambda a, b: rng.normal(0, 0.004, len(a)))
    wall_x = grid(3000, lambda a, b: rng.normal(0, 0.004, len(a)), lambda a, b: a, lambda a, b: b)
    wall_y = grid(3000, lambda a, b: a, lambda a, b: rng.normal(0, 0.004, len(a)), lambda a, b: b)

    def blob(center, n, scale):
        return (np.asarray(center, np.float32) + rng.normal(0, scale, (n, 3))).astype(np.float32)

    objects = np.vstack([blob([1.2, 1.0, 0.4], 500, 0.08), blob([2.8, 2.4, 0.7], 500, 0.10)])
    return np.vstack([floor, wall_x, wall_y, objects]).astype(np.float32)


def rigid(yaw_deg: float, t: np.ndarray) -> np.ndarray:
    th = np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    m[:3, 3] = t
    return m


def apply(transform: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.c_[pts, np.ones(len(pts), np.float32)]
    return (h @ transform.T)[:, :3].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="scan file (PCD/PLY/LAS/COPC)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaw", type=float, default=6.0, help="misalignment yaw [deg]")
    parser.add_argument("--leaf", type=float, default=0.05, help="input voxel leaf size")
    parser.add_argument("--png", default=None, help="write a before/after preview")
    args = parser.parse_args()

    print(f"SpatialRust {sr.__version__}")
    if args.input:
        target_cloud = sr.voxel_downsample(sr.read(args.input), args.leaf, "auto")
        target_np = target_cloud.xyz()
        print(f"loaded target     : {len(target_cloud):,} points from {args.input} (leaf={args.leaf})")
    else:
        target_np = synthesize_room(args.seed)
        print("target fixture    : generated room fallback")
    misalign = rigid(args.yaw, np.array([0.2, -0.15, 0.05], np.float32))
    source_np = apply(misalign, target_np)

    target = sr.PointCloud.from_xyz(target_np)
    source = sr.PointCloud.from_xyz(source_np)
    print(f"target/source     : {len(target):,} points each")
    print(f"applied misalign  : yaw={args.yaw}deg, t=(0.20, -0.15, 0.05)")

    methods = {
        "icp": lambda: sr.register_icp(source, target, 1.0, 60),
        "point_to_plane": lambda: sr.register_point_to_plane(source, target, 1.0, 60),
        "gicp": lambda: sr.register_gicp(source, target, 1.0, 60),
        "ndt": lambda: sr.register_ndt(source, target, 0.3, 60),
    }

    results = {}
    inv_truth = np.linalg.inv(misalign)
    for name, fn in methods.items():
        r = fn()
        t = r.transform()
        # residual of recovered transform vs the true inverse misalignment
        err = float(np.abs(t - inv_truth).max())
        results[name] = (r, t)
        print(f"{name:16}: iters={r.iterations:>3} converged={r.converged} "
              f"fitness={r.fitness:.2e} max|T-T*|={err:.4f}")

    best = min(results, key=lambda k: np.abs(results[k][1] - inv_truth).max())
    print(f"best by transform error: {best}")

    if args.png:
        _save_png(target_np, source_np, apply(results[best][1], source_np), best, args.png)


def _save_png(target, source, aligned, best, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG (pip install matplotlib)")
        return

    bg, tgt_c, src_c = "#0f172a", "#38bdf8", "#f97316"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=140)
    fig.patch.set_facecolor(bg)
    for ax, (title, moved) in zip(axes, [("Before", source), (f"After ({best})", aligned)]):
        ax.set_facecolor(bg)
        ax.scatter(target[:, 0], target[:, 1], c=tgt_c, s=2, linewidths=0, alpha=0.5, label="target")
        ax.scatter(moved[:, 0], moved[:, 1], c=src_c, s=2, linewidths=0, alpha=0.5, label="source")
        ax.set_aspect("equal")
        ax.set_title(title, color="#e2e8f0", fontsize=13)
        ax.tick_params(colors="#475569")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        leg = ax.legend(loc="upper right", facecolor=bg, edgecolor="#334155", labelcolor="#e2e8f0")
        leg.get_frame().set_alpha(0.8)
    fig.suptitle("SpatialRust scan registration", color="#e2e8f0", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, facecolor=bg)
    print(f"wrote preview -> {path}")


if __name__ == "__main__":
    main()
