"""Run and aggregate the canonical OpenCV comparison suites."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from report import emit_report, load_report, make_report


ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "calibration": ROOT / "bench" / "opencv_calibration_comparison" / "run.py",
    "vision": ROOT / "bench" / "opencv_vision_comparison" / "run.py",
    "vision-performance": ROOT
    / "bench"
    / "opencv_vision_comparison"
    / "performance.py",
    "rgbd": ROOT / "bench" / "opencv_rgbd_comparison" / "run.py",
    "video": ROOT / "bench" / "opencv_video_comparison" / "run.py",
    "odometry": ROOT / "bench" / "opencv_odometry_comparison" / "run.py",
    "photography": ROOT / "bench" / "opencv_photography_comparison" / "run.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "target" / "opencv-comparison"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
    reports = []
    for name, script in selected.items():
        output = args.output_dir / f"{name}.json"
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        reports.append(load_report(output))
    aggregate = make_report(
        suite="opencv-comparison",
        kind="aggregate",
        status="pass",
        environment_receipt=dict(reports[0]["environment"]),
        results={"reports": [report["suite"] for report in reports]},
    )
    emit_report(aggregate, args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
