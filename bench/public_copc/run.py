#!/usr/bin/env python3
"""Epic 60: validate SpatialRust COPC + MVP on public datasets.

1. Fetch the public PCL table_scene_lms400.pcd sample (460k points).
2. Run the Rust integration test that writes COPC, applies bounds/resolution
   queries, and executes the MVP pipeline.
3. Optionally (--http) run the ignored HTTP Autzen COPC smoke test.

Usage:
    python bench/public_copc/run.py
    python bench/public_copc/run.py --fetch-only
    python bench/public_copc/run.py --http
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


def cargo_exe() -> str:
    found = shutil.which("cargo")
    if found:
        return found
    fallback = Path(os.environ.get("USERPROFILE", "")) / ".cargo" / "bin" / "cargo.exe"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("cargo not found in PATH; install Rust from https://rustup.rs")


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    if cmd and cmd[0] == "cargo":
        cmd = [cargo_exe(), *cmd[1:]]
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only", action="store_true", help="download the public PCD only")
    parser.add_argument("--http", action="store_true", help="also run HTTP Autzen COPC smoke test")
    parser.add_argument("--release", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", help="use debug build instead of release")
    args = parser.parse_args()

    run([sys.executable, str(FETCH)])

    if args.fetch_only:
        return

    profile = [] if args.debug else ["--release"]
    run(
        [
            "cargo",
            "test",
            "-p",
            "spatialrust",
            "--features",
            "mvp",
            "--test",
            "mvp_public_copc",
            *profile,
            "--",
            "--nocapture",
            "mvp_public_pcd_copc_bounds_resolution_and_pipeline",
        ]
    )

    if args.http:
        run(
            [
                "cargo",
                "test",
                "-p",
                "spatialrust",
                "--features",
                "mvp,mvp-http",
                "--test",
                "mvp_public_copc",
                *profile,
                "--",
                "--ignored",
                "--nocapture",
                "mvp_http_autzen_copc_bounds_smoke",
            ]
        )


if __name__ == "__main__":
    main()
