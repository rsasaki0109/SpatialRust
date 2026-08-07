#!/usr/bin/env python3
"""Times PDAL point-cloud operations on a PCD file.

Prints `operation,seconds,output_points` lines on stdout so run.sh can compare
the results with SpatialRust's bench_ops example. Requires `pdal` on PATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def seconds_since(start: float) -> float:
    return time.perf_counter() - start


def run_pipeline(pipeline: dict) -> int:
    encoded = json.dumps(pipeline)
    proc = subprocess.run(
        ["pdal", "pipeline", "--stdin"],
        input=encoded.encode(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode())
    result = json.loads(proc.stdout.decode() or b"{}")
    # Point count of the final stage.
    try:
        return int(result["metadata"]["metadata"]["_runtime"]["total_read_points"])
    except (KeyError, TypeError):
        return 0


def make_pipeline(cloud: str, stages: list[dict]) -> dict:
    reader: dict = {"type": "readers.pcd", "filename": cloud}
    writer: dict = {"type": "writers.null"}
    return {"pipeline": [reader, *stages, writer]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", help="Input PCD file")
    args = parser.parse_args()

    print(f"loaded {args.cloud}", file=sys.stderr)

    # Voxel-grid downsample (cell 0.05). PDAL's nearest-voxel-center filter.
    start = time.perf_counter()
    count = run_pipeline(
        make_pipeline(args.cloud, [{"type": "filters.voxelcenternearest", "cell": 0.05}])
    )
    print(f"voxel_downsample,{seconds_since(start):.4f},{count}")

    # Translate XYZ by +1 m on each axis (spatial transform without reprojection).
    start = time.perf_counter()
    count = run_pipeline(
        make_pipeline(
            args.cloud,
            [{"type": "filters.transformation", "matrix": "1 0 0 1 0 1 0 1 0 0 1 1 0 0 0 1"}],
        )
    )
    print(f"translate_xyz,{seconds_since(start):.4f},{count}")

    # Reprojection WGS84 geocentric → geographic (EPSG:4978 -> EPSG:4979).
    start = time.perf_counter()
    count = run_pipeline(
        make_pipeline(
            args.cloud,
            [{"type": "filters.reprojection", "in_srs": "EPSG:4978", "out_srs": "EPSG:4979"}],
        )
    )
    print(f"reprojection,{seconds_since(start):.4f},{count}")


if __name__ == "__main__":
    main()
