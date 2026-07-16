from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from report import COMPONENT_STAGES, SCHEMA_VERSION, validate_receipt


def valid_receipt() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-07-16T00:00:00+00:00",
        "host": {"platform": "test", "machine": "x86_64", "processor": "cpu", "logical_cpu_count": 8},
        "thread_policy": {"mode": "default", "worker_count": 8},
        "workload": {
            "id": "resize_bilinear",
            "profile": "vga",
            "width": 640,
            "height": 480,
            "allocation_mode": "reuse",
            "batch_policy": "adaptive-minimum-20ms",
        },
        "stages": [
            {
                "name": name,
                "status": "measured",
                "timing": {"median_ms": 1.0, "p95_ms": 1.1, "samples": 10, "batch_size": 1},
            }
            if name in {"python_conversion", "allocation", "native_kernel"}
            else {"name": name, "status": "not_applicable", "reason": "CPU baseline"}
            for name in COMPONENT_STAGES
        ],
        "metrics": {
            "mpix_per_second": 307.2,
            "ns_per_pixel": 1_000_000.0 / (640 * 480),
            "bytes_allocated": 640 * 480 * 3,
            "peak_workspace_bytes": 0,
        },
    }


class ComponentReceiptTests(unittest.TestCase):
    def test_valid_contract_and_manifest(self) -> None:
        self.assertEqual(validate_receipt(valid_receipt()), [])
        manifest = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["profiles"]), {"vga", "1080p", "4k"})
        self.assertEqual(set(manifest["required_component_stages"]), set(COMPONENT_STAGES))
        self.assertEqual(len(manifest["workloads"]), 8)
        for workload in manifest["workloads"]:
            self.assertIn("native_criterion", workload)
            self.assertIn("bench", workload["native_criterion"])
            self.assertIn("group", workload["native_criterion"])

    def test_missing_duplicate_and_non_finite_values_fail_closed(self) -> None:
        missing = valid_receipt()
        missing["stages"] = missing["stages"][:-1]
        self.assertIn("stages must contain each component exactly once", validate_receipt(missing))

        duplicate = valid_receipt()
        duplicate["stages"][1]["name"] = "python_conversion"
        errors = validate_receipt(duplicate)
        self.assertIn("stage names must be unique", errors)

        non_finite = valid_receipt()
        non_finite["metrics"]["mpix_per_second"] = math.inf
        self.assertIn("metrics.mpix_per_second must be finite", validate_receipt(non_finite))

    def test_derived_throughput_must_match_native_timing(self) -> None:
        receipt = copy.deepcopy(valid_receipt())
        receipt["metrics"]["ns_per_pixel"] = 99.0
        self.assertIn(
            "metrics.ns_per_pixel disagrees with native median",
            validate_receipt(receipt),
        )

        receipt = copy.deepcopy(valid_receipt())
        receipt["metrics"]["mpix_per_second"] = 1.0
        self.assertIn(
            "metrics.mpix_per_second disagrees with native median",
            validate_receipt(receipt),
        )

    def test_counts_and_percentiles_are_strict(self) -> None:
        receipt = copy.deepcopy(valid_receipt())
        receipt["stages"][0]["timing"]["samples"] = 0
        receipt["stages"][0]["timing"]["p95_ms"] = 0.5
        errors = validate_receipt(receipt)
        self.assertIn(
            "stage python_conversion.timing.samples must be a positive integer",
            errors,
        )
        self.assertIn(
            "stage python_conversion.timing.p95_ms must be at least median_ms",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
