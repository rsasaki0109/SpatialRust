from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from report import (
    SCHEMA_VERSION,
    load_report,
    make_report,
    percentile,
    timed,
    timed_pair,
    validate_report,
)


class ReportContractTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 0.95), 1.95)

    def test_timed_retains_raw_samples(self) -> None:
        result, timing = timed(lambda: 42, warmup=1, repeats=3)
        self.assertEqual(result, 42)
        self.assertEqual(timing["unit"], "ms")
        self.assertEqual(len(timing["samples"]), 3)
        self.assertGreaterEqual(timing["p95"], timing["min"])
        self.assertIn("median_absolute_deviation", timing)
        self.assertIn("coefficient_of_variation", timing)
        self.assertEqual(timing["batch_size"], 1)

    def test_timed_pair_interleaves_and_retains_each_sample_set(self) -> None:
        calls: list[str] = []

        def left() -> str:
            calls.append("left")
            return "L"

        def right() -> str:
            calls.append("right")
            return "R"

        left_result, right_result, left_timing, right_timing = timed_pair(
            left, right, warmup=1, repeats=4, seed=7
        )
        self.assertEqual((left_result, right_result), ("L", "R"))
        self.assertEqual(len(left_timing["samples"]), 4)
        self.assertEqual(len(right_timing["samples"]), 4)
        self.assertEqual(calls.count("left"), 5)
        self.assertEqual(calls.count("right"), 5)

    def test_timed_pair_batches_short_calls(self) -> None:
        _, _, left_timing, right_timing = timed_pair(
            lambda: 1, lambda: 2, warmup=0, repeats=2, min_sample_time_ms=0.1
        )
        self.assertGreater(int(left_timing["batch_size"]), 1)
        self.assertGreater(int(right_timing["batch_size"]), 1)

    def test_report_round_trip(self) -> None:
        report = make_report(
            suite="test",
            kind="correctness",
            status="pass",
            environment_receipt={
                "platform": "test-os",
                "machine": "test-machine",
                "logical_cpu_count": 1,
                "python_version": "3.test",
                "opencv_version": "4.test",
                "spatialrust_version": "1.test",
            },
            results={"metric": 0},
        )
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(validate_report(report), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(load_report(path), report)

    def test_missing_environment_fields_are_rejected(self) -> None:
        report = make_report(
            suite="test",
            kind="performance",
            status="pass",
            environment_receipt={},
            results=[],
        )
        self.assertIn("environment missing platform", validate_report(report))

    def test_non_finite_values_are_rejected(self) -> None:
        report = make_report(
            suite="test",
            kind="performance",
            status="pass",
            environment_receipt={
                "platform": "test-os",
                "machine": "test-machine",
                "logical_cpu_count": 1,
                "python_version": "3.test",
                "opencv_version": "4.test",
                "spatialrust_version": "1.test",
            },
            results={"invalid": math.inf},
        )
        self.assertIn("report.results.invalid must be finite", validate_report(report))

    def test_manifest_reserves_representative_profiles_and_workloads(self) -> None:
        manifest_path = Path(__file__).with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["profiles"]), {"vga", "1080p", "4k"})
        workloads = {entry["id"] for entry in manifest["workloads"]}
        statistics = set(manifest["required_statistics"])
        self.assertGreaterEqual(len(workloads), 10)
        self.assertIn("rgbd_to_voxel", workloads)
        self.assertIn("ai_preprocess", workloads)
        self.assertIn("nms", workloads)
        self.assertIn("batched_nms", workloads)
        self.assertIn("soft_nms", workloads)
        self.assertIn("connected_components", workloads)
        self.assertIn("coefficient_of_variation", statistics)
        self.assertIn("median_absolute_deviation", statistics)
        self.assertIn("batch_size", statistics)


if __name__ == "__main__":
    unittest.main()
