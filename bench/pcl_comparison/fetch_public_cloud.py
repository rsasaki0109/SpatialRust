#!/usr/bin/env python3
"""Fetches the public PCL table_scene_lms400 PCD sample for benchmarks."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional


PUBLIC_PCL_URL = (
    "https://raw.githubusercontent.com/PointCloudLibrary/data/master/"
    "tutorials/table_scene_lms400.pcd"
)
PUBLIC_PCL_FILE = "table_scene_lms400.pcd"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output() -> Path:
    return repo_root() / "target" / "bench-data" / PUBLIC_PCL_FILE


def ensure_public_cloud(out: Optional[Path] = None, force: bool = False) -> Path:
    path = out or default_output()
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"using cached public PCL sample -> {path}", file=sys.stderr)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(PUBLIC_PCL_URL, headers={"User-Agent": "SpatialRust benchmark"})
    with urllib.request.urlopen(request) as response, path.open("wb") as file:
        file.write(response.read())
    print(f"downloaded public PCL sample -> {path}", file=sys.stderr)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=default_output())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_public_cloud(args.out, args.force)


if __name__ == "__main__":
    main()
