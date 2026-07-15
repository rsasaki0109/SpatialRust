from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report import (
    SCHEMA_VERSION,
    load_report,
    make_report,
    percentile,
    timed,
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

    def test_manifest_reserves_representative_profiles_and_workloads(self) -> None:
        manifest_path = Path(__file__).with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["profiles"]), {"vga", "1080p", "4k"})
        workloads = {entry["id"] for entry in manifest["workloads"]}
        self.assertGreaterEqual(len(workloads), 10)
        self.assertIn("rgbd_to_voxel", workloads)
        self.assertIn("ai_preprocess", workloads)


if __name__ == "__main__":
    unittest.main()
