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
    parser.add_argument(
        "points",
        nargs="?",
        type=int,
        default=None,
        help="Legacy shorthand for --synthetic-points",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing PCD file to benchmark instead of the public PCL sample",
    )
    parser.add_argument(
        "--synthetic-points",
        type=int,
        default=None,
        help="Generate a deterministic synthetic cloud with this many requested points",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python interpreter with spatialrust, numpy, and open3d installed",
    )
    args = parser.parse_args()

    root = repo_root()
    python = args.python or venv_python(root)
    if args.input is not None and (args.points is not None or args.synthetic_points is not None):
        parser.error("--input cannot be combined with synthetic point-count arguments")

    if args.input is not None:
        pcd = args.input
        print(f"== using input cloud {pcd} ==", file=sys.stderr)
    else:
        synthetic_points = args.synthetic_points if args.synthetic_points is not None else args.points
        if synthetic_points is not None:
            pcd = Path(tempfile.gettempdir()) / f"bench_cloud_{synthetic_points}.pcd"
            run(
                [
                    python,
                    root / "bench" / "pcl_comparison" / "gen_cloud.py",
                    "--points",
                    str(synthetic_points),
                    "--out",
                    pcd,
                ]
            )
        else:
            sys.path.insert(0, str(root / "bench" / "pcl_comparison"))
            from fetch_public_cloud import ensure_public_cloud

            pcd = ensure_public_cloud()

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
