"""Strict component timing receipt contract for the Vision 2 baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path


SCHEMA_VERSION = "spatialrust.vision-component-timing.v1"
COMPONENT_STAGES = (
    "python_conversion",
    "allocation",
    "native_kernel",
    "upload",
    "execution",
    "readback",
)
THREAD_MODES = {"single", "default"}
STAGE_STATUSES = {"measured", "not_applicable"}


def validate_receipt(receipt: object) -> list[str]:
    """Return all structural and finite-number contract violations."""

    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["receipt must be a JSON object"]

    def require_finite(value: object, path: str, *, non_negative: bool = True) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path} must be numeric")
        elif not math.isfinite(float(value)):
            errors.append(f"{path} must be finite")
        elif non_negative and value < 0:
            errors.append(f"{path} must be non-negative")

    def require_positive_int(value: object, path: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{path} must be a positive integer")

    if receipt.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("generated_at", "host", "thread_policy", "workload", "stages", "metrics"):
        if key not in receipt:
            errors.append(f"missing {key}")

    host = receipt.get("host")
    if not isinstance(host, dict):
        errors.append("host must be an object")
    else:
        for key in ("platform", "machine", "processor", "logical_cpu_count"):
            if key not in host:
                errors.append(f"host missing {key}")
        require_positive_int(host.get("logical_cpu_count"), "host.logical_cpu_count")

    thread_policy = receipt.get("thread_policy")
    if not isinstance(thread_policy, dict):
        errors.append("thread_policy must be an object")
    else:
        if thread_policy.get("mode") not in THREAD_MODES:
            errors.append("thread_policy.mode must be single or default")
        require_positive_int(thread_policy.get("worker_count"), "thread_policy.worker_count")

    workload = receipt.get("workload")
    pixels = None
    if not isinstance(workload, dict):
        errors.append("workload must be an object")
    else:
        for key in ("id", "profile", "width", "height", "allocation_mode", "batch_policy"):
            if key not in workload:
                errors.append(f"workload missing {key}")
        require_positive_int(workload.get("width"), "workload.width")
        require_positive_int(workload.get("height"), "workload.height")
        if isinstance(workload.get("width"), int) and isinstance(workload.get("height"), int):
            pixels = workload["width"] * workload["height"]
            if pixels <= 0:
                errors.append("workload dimensions must be positive")

    stages = receipt.get("stages")
    if not isinstance(stages, list):
        errors.append("stages must be an array")
    else:
        by_name = {
            stage.get("name"): stage
            for stage in stages
            if isinstance(stage, dict) and isinstance(stage.get("name"), str)
        }
        if set(by_name) != set(COMPONENT_STAGES):
            errors.append("stages must contain each component exactly once")
        if len(stages) != len(by_name):
            errors.append("stage names must be unique")
        for name, stage in by_name.items():
            status = stage.get("status")
            if status not in STAGE_STATUSES:
                errors.append(f"stage {name} has invalid status")
            if status == "measured":
                timing = stage.get("timing")
                if not isinstance(timing, dict):
                    errors.append(f"stage {name} timing must be an object")
                else:
                    for key in ("median_ms", "p95_ms"):
                        require_finite(timing.get(key), f"stage {name}.timing.{key}")
                    for key in ("samples", "batch_size"):
                        require_positive_int(timing.get(key), f"stage {name}.timing.{key}")
                    median_ms = timing.get("median_ms")
                    p95_ms = timing.get("p95_ms")
                    if (
                        isinstance(median_ms, (int, float))
                        and not isinstance(median_ms, bool)
                        and isinstance(p95_ms, (int, float))
                        and not isinstance(p95_ms, bool)
                        and math.isfinite(float(median_ms))
                        and math.isfinite(float(p95_ms))
                        and p95_ms < median_ms
                    ):
                        errors.append(f"stage {name}.timing.p95_ms must be at least median_ms")
            elif status == "not_applicable" and not stage.get("reason"):
                errors.append(f"stage {name} requires a not-applicable reason")

    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")
    else:
        for key in (
            "mpix_per_second",
            "ns_per_pixel",
            "bytes_allocated",
            "peak_workspace_bytes",
        ):
            require_finite(metrics.get(key), f"metrics.{key}")
        if pixels and isinstance(metrics.get("ns_per_pixel"), (int, float)):
            native = next(
                (
                    stage
                    for stage in stages
                    if isinstance(stage, dict) and stage.get("name") == "native_kernel"
                ),
                None,
            ) if isinstance(stages, list) else None
            if isinstance(native, dict) and isinstance(native.get("timing"), dict):
                median_ms = native["timing"].get("median_ms")
                if isinstance(median_ms, (int, float)) and math.isfinite(float(median_ms)):
                    expected = median_ms * 1_000_000.0 / pixels
                    if not math.isclose(float(metrics["ns_per_pixel"]), expected, rel_tol=1e-9):
                        errors.append("metrics.ns_per_pixel disagrees with native median")
                    expected_mpix = pixels / (median_ms * 1000.0) if median_ms > 0 else math.inf
                    measured_mpix = metrics.get("mpix_per_second")
                    if (
                        isinstance(measured_mpix, (int, float))
                        and not isinstance(measured_mpix, bool)
                        and not math.isclose(float(measured_mpix), expected_mpix, rel_tol=1e-9)
                    ):
                        errors.append("metrics.mpix_per_second disagrees with native median")
    return errors


def emit_receipt(receipt: dict[str, object], output: Path) -> None:
    """Validate and write a deterministic strict JSON receipt."""

    errors = validate_receipt(receipt)
    if errors:
        raise ValueError("invalid component timing receipt: " + "; ".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_receipt(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_receipt(value)
    if errors:
        raise ValueError(f"invalid receipt {path}: " + "; ".join(errors))
    return value
