#!/usr/bin/env python3
"""Run the SpatialRust-vs-Open3D comparison benchmark."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union


CommandArg = Union[str, os.PathLike]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def example_path(root: Path) -> Path:
    name = "bench_ops.exe" if os.name == "nt" else "bench_ops"
    return root / "target" / "release" / "examples" / name


def cargo_path() -> Union[Path, str]:
    cargo = shutil.which("cargo")
    if cargo is not None:
        return cargo

    local = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    if local.exists():
        return local

    return "cargo"


def read_csv(text: str) -> Dict[str, Tuple[str, str]]:
    rows: Dict[str, Tuple[str, str]] = {}
    for row in csv.reader(text.splitlines()):
        if len(row) != 3:
            continue
        op, seconds, output_points = row
        rows[op] = (seconds, output_points)
    return rows


def run(command: List[CommandArg], **kwargs) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print(f"== {printable} ==", file=sys.stderr)
    return subprocess.run(command, check=True, text=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points", nargs="?", type=int, default=200_000)
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python interpreter with spatialrust, numpy, and open3d installed",
    )
    args = parser.parse_args()

    root = repo_root()
    python = args.python or venv_python(root)
    pcd = Path(tempfile.gettempdir()) / "bench_cloud.pcd"

    run(
        [
            python,
            root / "bench" / "pcl_comparison" / "gen_cloud.py",
            "--points",
            str(args.points),
            "--out",
            pcd,
        ]
    )

    run(
        [
            cargo_path(),
            "build",
            "--release",
            "--manifest-path",
            root / "Cargo.toml",
            "-p",
            "spatialrust",
            "--example",
            "bench_ops",
            "--features",
            "mvp,filter-outlier",
        ],
        stdout=subprocess.DEVNULL,
    )

    sr = run([example_path(root), pcd], capture_output=True).stdout
    open3d = run(
        [python, Path(__file__).resolve().parent / "open3d_bench.py", pcd],
        capture_output=True,
    ).stdout

    sr_rows = read_csv(sr)
    open3d_rows = read_csv(open3d)

    print()
    print(f"{'operation':<30} {'SpatialRust(s)':>14} {'Open3D(s)':>14} {'speedup':>10}")
    print(f"{'------------------------------':<30} {'--------------':>14} {'--------------':>14} {'----------':>10}")
    for op, (sr_t, _) in sr_rows.items():
        if op in open3d_rows:
            open3d_t = open3d_rows[op][0]
            speedup = f"{float(open3d_t) / float(sr_t):.2f}x" if float(sr_t) > 0 else "n/a"
        else:
            open3d_t = "n/a"
            speedup = "n/a"
        print(f"{op:<30} {sr_t:>14} {open3d_t:>14} {speedup:>10}")


if __name__ == "__main__":
    main()
