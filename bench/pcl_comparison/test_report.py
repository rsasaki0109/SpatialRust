#!/usr/bin/env python3
"""Contract tests for the point-cloud comparison report schema.

Deliberately stdlib-only so CI can gate the report contract without PCL/PDAL.
Run with:
    python bench/pcl_comparison/test_report.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from report import (  # noqa: E402
    SCHEMA_VERSION,
    emit_report,
    environment,
    load_report,
    make_report,
    percentile,
    timing_statistics,
    validate_report,
)


def test_schema_version_is_canonical():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "spatialrust.pointcloud-benchmark-manifest.v1"
    assert SCHEMA_VERSION == "spatialrust.pointcloud-comparison.v1"


def test_manifest_has_canonical_profiles_and_workloads():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    assert {"small", "full"} <= set(manifest["profiles"])
    ids = [w["id"] for w in manifest["workloads"]]
    assert "voxel_downsample" in ids
    assert "normal_estimation" in ids
    assert "statistical_outlier_removal" in ids
    assert "radius_outlier_removal" in ids
    assert "translate_xyz" in ids
    assert "reprojection" in ids


def test_valid_report_passes():
    report = make_report(
        suite="pcl_comparison",
        kind="performance",
        status="pass",
        environment_receipt=environment(
            pcl_version="1.15.1",
            pdal_version=None,
            open3d_version=None,
            spatialrust_version="1.2.0",
        ),
        results={"workloads": []},
    )
    assert validate_report(report) == []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        emit_report(report, output=path)
        assert load_report(path)["suite"] == "pcl_comparison"


def test_missing_environment_key_fails():
    report = make_report(
        suite="pdal_comparison",
        kind="performance",
        status="pass",
        environment_receipt=environment(
            pcl_version=None,
            pdal_version="2.6",
            open3d_version=None,
            spatialrust_version="1.2.0",
        ),
        results={},
    )
    del report["environment"]["pcl_version"]
    errors = validate_report(report)
    assert any("missing" in error for error in errors)
    assert any("pcl_version" in error for error in errors)


def test_wrong_schema_version_fails():
    report = make_report(
        suite="pcl_comparison",
        kind="performance",
        status="pass",
        environment_receipt=environment(
            pcl_version="1.15.1",
            pdal_version=None,
            open3d_version=None,
            spatialrust_version="1.2.0",
        ),
        results={},
    )
    report["schema_version"] = "spatialrust.opencv-comparison.v1"
    assert any("schema_version" in error for error in validate_report(report))


def test_non_finite_values_fail():
    report = make_report(
        suite="pcl_comparison",
        kind="performance",
        status="pass",
        environment_receipt=environment(
            pcl_version="1.15.1",
            pdal_version=None,
            open3d_version=None,
            spatialrust_version="1.2.0",
        ),
        results={"mean": float("inf")},
    )
    assert any("must be finite" in error for error in validate_report(report))


def test_timing_statistics_and_percentile():
    stats = timing_statistics([10.0, 11.0, 12.0, 13.0, 14.0], warmup=2)
    assert stats["median"] == 12.0
    assert stats["mean"] == 12.0
    assert abs(stats["p95"] - 13.8) < 1e-9
    assert stats["samples"] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert abs(percentile([1.0, 2.0], 0.5) - 1.5) < 1e-9
    try:
        percentile([], 0.5)
        raise AssertionError("empty samples must raise")
    except ValueError:
        pass


def main() -> int:
    tests = [
        ("schema_version_is_canonical", test_schema_version_is_canonical),
        ("manifest_profiles_and_workloads", test_manifest_has_canonical_profiles_and_workloads),
        ("valid_report_passes", test_valid_report_passes),
        ("missing_environment_key_fails", test_missing_environment_key_fails),
        ("wrong_schema_version_fails", test_wrong_schema_version_fails),
        ("non_finite_values_fail", test_non_finite_values_fail),
        ("timing_statistics_and_percentile", test_timing_statistics_and_percentile),
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"ok - {name}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL - {name}: {error}")
    if failures:
        print(f"\n{failures} contract test(s) failed")
        return 1
    print("\nall point-cloud report contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
