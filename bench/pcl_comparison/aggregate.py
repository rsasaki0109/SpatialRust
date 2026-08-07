"""Aggregate and validate point-cloud comparison receipts.

Collects PCL/PDAL/Open3D comparison receipts, verifies each against the
`spatialrust.pointcloud-comparison.v1` contract, rejects duplicate or
conflicting workloads, and writes one aggregate report. Stdlib-only so it can
gate CI without installing any comparison library.

Usage:
    python bench/pcl_comparison/aggregate.py \
      --receipts bench/pcl_comparison/receipt-*.json \
      [--output target/pointcloud-aggregate.json]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from report import emit_report, load_report, make_report, validate_report

SUPPORTED_SUITES = {"pcl_comparison", "pdal_comparison", "open3d_comparison"}


def collect(receipt_paths: list[Path]) -> list[dict[str, object]]:
    reports = []
    for path in receipt_paths:
        reports.append(load_report(path))
    return reports


def aggregate(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("at least one receipt is required")

    seen: dict[str, dict[str, object]] = {}
    for report in reports:
        suite = report["suite"]
        if suite not in SUPPORTED_SUITES:
            raise ValueError(f"unsupported suite {suite}")
        operations = report.get("results", {}).get("operations", [])
        if not isinstance(operations, list):
            raise ValueError(f"suite {suite} results.operations must be a list")
        for operation in operations:
            operation_id = operation.get("id")
            if not isinstance(operation_id, str) or not operation_id:
                raise ValueError(f"suite {suite} has an operation without an id")
            if operation_id in seen:
                raise ValueError(f"duplicate workload {operation_id} in {suite}")
            seen[operation_id] = {
                "id": operation_id,
                "suite": suite,
                "spatialrust_seconds": operation.get("spatialrust_seconds"),
                "library_seconds": operation.get(
                    {"pcl_comparison": "pcl_seconds", "pdal_comparison": "pdal_seconds", "open3d_comparison": "open3d_seconds"}[suite]
                ),
                "speedup": operation.get("speedup"),
                "output_points_sr": operation.get("output_points_sr"),
                "output_points_library": operation.get(
                    {"pcl_comparison": "output_points_pcl", "pdal_comparison": "output_points_pdal", "open3d_comparison": "output_points_open3d"}[suite]
                ),
            }

    environment = reports[0]["environment"]
    return make_report(
        suite="pointcloud-conformance-aggregate",
        kind="aggregate",
        status="pass",
        environment_receipt=environment,
        results={
            "suite_count": len(reports),
            "workload_count": len(seen),
            "workloads": list(seen.values()),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipts",
        nargs="+",
        help="one or more receipt JSON paths or glob patterns",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def expand(pattern: str) -> list[Path]:
    paths = [Path(match) for match in glob.glob(pattern)]
    return [path for path in paths if path.exists()]


def main() -> None:
    args = parse_args()
    patterns = args.receipts or [
        "bench/pcl_comparison/receipt-*.json",
        "bench/pdal_comparison/receipt-*.json",
        "bench/open3d_comparison/receipt-*.json",
    ]
    receipt_paths: list[Path] = []
    for pattern in patterns:
        receipt_paths.extend(expand(pattern))
    receipt_paths = list(dict.fromkeys(receipt_paths))
    receipt_paths.sort()
    if not receipt_paths:
        raise SystemExit("no receipt files found")

    reports = collect(receipt_paths)
    result = aggregate(reports)
    emit_report(result, output=args.output)


if __name__ == "__main__":
    main()
