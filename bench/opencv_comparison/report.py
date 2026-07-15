"""Machine-readable report contract for OpenCV comparison harnesses.

This module deliberately depends only on the Python standard library so its
contract tests can run without building SpatialRust or installing OpenCV.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, TypeVar


SCHEMA_VERSION = "spatialrust.opencv-comparison.v1"
T = TypeVar("T")


def environment(*, opencv_version: str, spatialrust_version: str) -> dict[str, object]:
    """Return the minimum environment receipt required for publication."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "opencv_version": opencv_version,
        "spatialrust_version": spatialrust_version,
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    samples = sorted(values)
    if not samples:
        raise ValueError("at least one timing sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    index = (len(samples) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(samples) - 1)
    fraction = index - lower
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction


def timed(
    call: Callable[[], T], *, warmup: int, repeats: int
) -> tuple[T, dict[str, object]]:
    """Benchmark a callable and return its last result plus timing statistics."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        call()
    samples_ms: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return result, {
        "unit": "ms",
        "warmup": warmup,
        "repeats": repeats,
        "median": statistics.median(samples_ms),
        "p95": percentile(samples_ms, 0.95),
        "min": min(samples_ms),
        "max": max(samples_ms),
        "samples": samples_ms,
    }


def make_report(
    *,
    suite: str,
    kind: str,
    status: str,
    environment_receipt: dict[str, object],
    results: object,
) -> dict[str, object]:
    if kind not in {"correctness", "performance", "aggregate"}:
        raise ValueError(f"unsupported report kind: {kind}")
    if status not in {"pass", "fail"}:
        raise ValueError(f"unsupported report status: {status}")
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "kind": kind,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_receipt,
        "results": results,
    }


def emit_report(report: dict[str, object], output: Path | None = None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def validate_report(report: object) -> list[str]:
    """Validate the stable structural contract without third-party packages."""

    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    required = {
        "schema_version",
        "suite",
        "kind",
        "status",
        "generated_at",
        "environment",
        "results",
    }
    missing = sorted(required - report.keys())
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("kind") not in {"correctness", "performance", "aggregate"}:
        errors.append("kind must be correctness, performance, or aggregate")
    if report.get("status") not in {"pass", "fail"}:
        errors.append("status must be pass or fail")
    environment_value = report.get("environment")
    if not isinstance(environment_value, dict):
        errors.append("environment must be an object")
    else:
        for key in (
            "platform",
            "machine",
            "logical_cpu_count",
            "python_version",
            "opencv_version",
            "spatialrust_version",
        ):
            if key not in environment_value:
                errors.append(f"environment missing {key}")
    return errors


def load_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    errors = validate_report(value)
    if errors:
        raise ValueError(f"invalid report {path}: {'; '.join(errors)}")
    return value
