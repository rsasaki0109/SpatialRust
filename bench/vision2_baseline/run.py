"""Measure the Vision 2 component baseline with explicit ownership stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Callable

from report import COMPONENT_STAGES, SCHEMA_VERSION, emit_receipt


PROFILES = {
    "vga": (640, 480),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
MIN_SAMPLE_NS = 20_000_000
SAMPLES = 10


def measure(operation: Callable[[], object]) -> dict[str, float | int]:
    """Return per-call latency using an adaptively sized timed batch."""

    for _ in range(3):
        operation()
    batch_size = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            operation()
        elapsed = time.perf_counter_ns() - start
        if elapsed >= MIN_SAMPLE_NS or batch_size >= 1_048_576:
            break
        batch_size *= max(2, min(16, (MIN_SAMPLE_NS + elapsed - 1) // max(elapsed, 1)))

    samples_ms: list[float] = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        for _ in range(batch_size):
            operation()
        samples_ms.append((time.perf_counter_ns() - start) / batch_size / 1_000_000.0)
    ordered = sorted(samples_ms)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return {
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[p95_index],
        "samples": SAMPLES,
        "batch_size": batch_size,
    }


def measured_stage(name: str, operation: Callable[[], object]) -> dict[str, object]:
    return {"name": name, "status": "measured", "timing": measure(operation)}


def cpu_only_stage(name: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "not_applicable",
        "reason": "CPU baseline performs no device upload, execution, or readback",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="vga")
    parser.add_argument("--thread-mode", choices=("single", "default"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rayon_threads = os.environ.get("RAYON_NUM_THREADS")
    if args.thread_mode == "single" and rayon_threads != "1":
        raise SystemExit("single mode requires RAYON_NUM_THREADS=1 before process startup")
    if args.thread_mode == "default" and rayon_threads is not None:
        raise SystemExit("default mode requires RAYON_NUM_THREADS to be unset")

    # Import after validating the process-wide Rayon policy.
    import numpy as np
    import spatialrust as sr

    width, height = PROFILES[args.profile]
    output_width = width // 2
    output_height = height // 2
    rng = np.random.default_rng(0x112)
    backing = rng.integers(0, 256, (height, width * 2, 3), dtype=np.uint8)
    python_view = backing[:, ::2, :]
    image = np.ascontiguousarray(python_view)
    output = np.empty((output_height, output_width, 3), dtype=np.uint8)

    stages = [
        measured_stage("python_conversion", lambda: np.ascontiguousarray(python_view)),
        measured_stage(
            "allocation",
            lambda: np.empty((output_height, output_width, 3), dtype=np.uint8),
        ),
        measured_stage(
            "native_kernel",
            lambda: sr.resize_image(
                image,
                output_width,
                output_height,
                interpolation="bilinear",
                out=output,
            ),
        ),
        cpu_only_stage("upload"),
        cpu_only_stage("execution"),
        cpu_only_stage("readback"),
    ]
    assert [stage["name"] for stage in stages] == list(COMPONENT_STAGES)
    native_ms = stages[2]["timing"]["median_ms"]
    pixels = width * height
    logical_cpus = os.cpu_count() or 1
    worker_count = 1 if args.thread_mode == "single" else logical_cpus
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine() or "unknown",
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": logical_cpus,
        },
        "thread_policy": {"mode": args.thread_mode, "worker_count": worker_count},
        "workload": {
            "id": "resize_bilinear",
            "profile": args.profile,
            "width": width,
            "height": height,
            "output_width": output_width,
            "output_height": output_height,
            "allocation_mode": "caller_owned_output",
            "batch_policy": "adaptive-minimum-20ms",
        },
        "stages": stages,
        "metrics": {
            "mpix_per_second": pixels / (native_ms * 1000.0),
            "ns_per_pixel": native_ms * 1_000_000.0 / pixels,
            "bytes_allocated": output.nbytes,
            "peak_workspace_bytes": 0,
        },
    }
    emit_receipt(receipt, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
