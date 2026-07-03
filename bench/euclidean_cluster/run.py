#!/usr/bin/env python3
"""CPU vs GPU Euclidean clustering benchmark on the public PCL sample.

Runs `bench_euclidean_cluster` on table_scene_lms400.pcd plane outliers (MVP path)
or the full cloud and prints CSV rows for cpu/gpu latency.

Usage:
    python bench/euclidean_cluster/run.py
    python bench/euclidean_cluster/run.py --mvp-leaf 0.05 --repeat 5
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FETCH = ROOT / "bench" / "pcl_comparison" / "fetch_public_cloud.py"
FEATURES = (
    "segment-euclidean,segment-euclidean-gpu,segment-ransac-plane,"
    "io-pcd,gpu-wgpu,filter-voxel,feature-normal"
)


def cargo_exe() -> str:
    found = shutil.which("cargo")
    if found:
        return found
    fallback = Path(os.environ.get("USERPROFILE", Path.home())) / ".cargo" / "bin" / "cargo.exe"
    if fallback.is_file():
        return str(fallback)
    fallback_unix = Path.home() / ".cargo" / "bin" / "cargo"
    if fallback_unix.is_file():
        return str(fallback_unix)
    raise RuntimeError("cargo not found in PATH")


def run(cmd: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    if cmd and cmd[0] == "cargo":
        cmd = [cargo_exe(), *cmd[1:]]
    print("+", " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mvp-leaf", type=float, default=0.05)
    parser.add_argument("--full-cloud", action="store_true", help="cluster full cloud instead of MVP plane outliers")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    subprocess.run([sys.executable, str(FETCH)], cwd=ROOT, check=True)

    pcd = args.input or (ROOT / "target" / "bench-data" / "table_scene_lms400.pcd")
    if not pcd.is_file():
        raise SystemExit(f"input not found: {pcd}")

    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "spatialrust",
        "--example",
        "bench_euclidean_cluster",
        "--features",
        FEATURES,
        "--",
        str(pcd),
        "--tolerance",
        str(args.tolerance),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    if not args.full_cloud:
        cmd.extend(["--mvp-leaf", str(args.mvp_leaf)])

    completed = run(cmd)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    print("backend,seconds,cluster_count,point_count")
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
