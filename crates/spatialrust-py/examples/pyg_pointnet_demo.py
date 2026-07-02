#!/usr/bin/env python3
"""Epic 63: SpatialRust -> PyTorch Geometric end-to-end demo.

Builds model-ready tensors with SpatialRust, wraps them in a PyG ``Data`` object,
and runs a tiny two-layer GCN forward pass (random weights, no training).

Usage:
    python examples/pyg_pointnet_demo.py
    python examples/pyg_pointnet_demo.py --input ../../target/bench-data/table_scene_lms400.pcd

Requires NumPy. For the GCN step also install PyTorch + PyG:

    pip install torch torch-geometric
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import spatialrust as sr


def synth(seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    blob = lambda c, n, s: (np.asarray(c, np.float32) + r.normal(0, s, (n, 3))).astype(np.float32)
    return np.vstack(
        [
            blob([0.0, 0.0, 0.0], 1200, [0.25, 0.25, 0.25]),
            blob([2.0, 0.4, 0.0], 900, [0.08, 0.08, 0.50]),
            blob([1.0, 2.0, 0.0], 800, [0.40, 0.10, 0.15]),
        ]
    ).astype(np.float32)


def preprocess(cloud: sr.PointCloud, samples: int, knn: int) -> tuple[np.ndarray, np.ndarray]:
    cleaned = sr.statistical_outlier_removal(cloud, 12, 2.0)
    normalized = sr.normalize_unit_sphere(cleaned)
    sampled = sr.farthest_point_sampling(normalized, samples)
    pts = sampled.xyz().astype(np.float32)
    edge_index = sr.knn_graph(sampled, knn).astype(np.int64)
    return pts, edge_index


def run_gcn(pts: np.ndarray, edge_index: np.ndarray) -> None:
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
    except ImportError as exc:
        print("PyTorch / PyG not installed; preprocessing only.", file=sys.stderr)
        print(f"  ({exc})", file=sys.stderr)
        print("  pip install torch torch-geometric", file=sys.stderr)
        return

    device = torch.device("cpu")
    data = Data(
        pos=torch.from_numpy(pts).to(device),
        edge_index=torch.from_numpy(edge_index).to(device),
    )

    class TinyGcn(torch.nn.Module):
        def __init__(self, hidden: int = 32) -> None:
            super().__init__()
            self.conv1 = GCNConv(3, hidden)
            self.conv2 = GCNConv(hidden, 16)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

    model = TinyGcn().to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.pos, data.edge_index)

    print("\nPyG forward pass:")
    print(f"  Data.pos         : {tuple(data.pos.shape)}")
    print(f"  Data.edge_index  : {tuple(data.edge_index.shape)}")
    print(f"  GCN output       : {tuple(out.shape)}  (per-point 16-d embedding)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="PCD/PLY/LAS/COPC path")
    parser.add_argument("--samples", type=int, default=1024, help="FPS target count")
    parser.add_argument("--knn", type=int, default=16, help="k for knn_graph")
    args = parser.parse_args()

    print(f"SpatialRust {sr.__version__}")
    if args.input:
        cloud = sr.read(args.input)
        print(f"loaded : {len(cloud):,} points from {args.input}")
    else:
        cloud = sr.PointCloud.from_xyz(synth())
        print(f"synth  : {len(cloud):,} points")

    pts, edge_index = preprocess(cloud, args.samples, args.knn)
    print("\nSpatialRust tensors:")
    print(f"  points     : {pts.shape} float32")
    print(f"  edge_index : {edge_index.shape} int64  (PyG convention)")

    run_gcn(pts, edge_index)


if __name__ == "__main__":
    main()
