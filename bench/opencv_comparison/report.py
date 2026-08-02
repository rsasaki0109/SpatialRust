"""Machine-readable report contract for OpenCV comparison harnesses.

This module deliberately depends only on the Python standard library so its
contract tests can run without building SpatialRust or installing OpenCV.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
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
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = call()
            samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    finally:
        if gc_enabled:
            gc.enable()
    return result, timing_statistics(samples_ms, warmup=warmup)


def timing_statistics(
    samples_ms: list[float], *, warmup: int, batch_size: int = 1
) -> dict[str, object]:
    """Summarize raw millisecond samples with robust dispersion statistics."""

    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    median = statistics.median(samples_ms)
    deviations = [abs(value - median) for value in samples_ms]
    mean = statistics.fmean(samples_ms)
    stdev = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "unit": "ms",
        "warmup": warmup,
        "repeats": len(samples_ms),
        "batch_size": batch_size,
        "mean": mean,
        "median": median,
        "p95": percentile(samples_ms, 0.95),
        "min": min(samples_ms),
        "max": max(samples_ms),
        "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean > 0.0 else 0.0,
        "median_absolute_deviation": statistics.median(deviations),
        "samples": samples_ms,
    }


def timed_pair(
    left: Callable[[], T],
    right: Callable[[], T],
    *,
    warmup: int,
    repeats: int,
    seed: int = 0,
    min_sample_time_ms: float = 0.0,
) -> tuple[T, T, dict[str, object], dict[str, object]]:
    """Time two implementations in randomized interleaved order.

    Interleaving reduces systematic thermal, boost-clock, and first/second
    implementation bias. Short calls may be batched to reduce timer noise.
    Garbage collection is disabled only while sampling.
    """

    if warmup < 0 or repeats < 1 or min_sample_time_ms < 0.0:
        raise ValueError(
            "warmup and min_sample_time_ms must be non-negative; repeats must be positive"
        )
    left_result: T | None = None
    right_result: T | None = None
    for _ in range(warmup):
        left_result = left()
        right_result = right()

    def batch_size_for(call: Callable[[], T]) -> int:
        if min_sample_time_ms == 0.0:
            return 1
        batch_size = 1
        while batch_size < 16_384:
            start = time.perf_counter_ns()
            for _ in range(batch_size):
                call()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            if elapsed_ms >= min_sample_time_ms:
                return batch_size
            batch_size *= 2
        return batch_size

    left_batch_size = batch_size_for(left)
    right_batch_size = batch_size_for(right)
    left_samples: list[float] = []
    right_samples: list[float] = []
    rng = random.Random(seed)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            calls = [
                ("left", left, left_batch_size),
                ("right", right, right_batch_size),
            ]
            rng.shuffle(calls)
            for name, call, batch_size in calls:
                start = time.perf_counter_ns()
                for _ in range(batch_size):
                    result = call()
                elapsed = (time.perf_counter_ns() - start) / 1_000_000.0 / batch_size
                if name == "left":
                    left_result = result
                    left_samples.append(elapsed)
                else:
                    right_result = result
                    right_samples.append(elapsed)
    finally:
        if gc_enabled:
            gc.enable()
    return (
        left_result,  # type: ignore[return-value]
        right_result,  # type: ignore[return-value]
        timing_statistics(left_samples, warmup=warmup, batch_size=left_batch_size),
        timing_statistics(right_samples, warmup=warmup, batch_size=right_batch_size),
    )


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
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid report: {'; '.join(errors)}")
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def validate_report(report: object) -> list[str]:
    """Validate the stable structural contract without third-party packages."""

    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]

    def find_non_finite(value: object, path: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path} must be finite")
        elif isinstance(value, dict):
            for key, child in value.items():
                find_non_finite(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                find_non_finite(child, f"{path}[{index}]")

    find_non_finite(report, "report")
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
