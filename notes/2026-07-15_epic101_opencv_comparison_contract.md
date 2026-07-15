# Epic 101: OpenCV comparison contract

Date: 2026-07-15

Epic 101 establishes the evidence contract for the OpenCV-outcome program. It
does not broaden production dependencies or claim that every reserved workload
already beats OpenCV.

## Delivered

- `spatialrust.opencv-comparison.v1` JSON envelope for correctness,
  performance, and aggregate reports.
- Environment receipt with platform, architecture, logical CPU count, Python,
  OpenCV, SpatialRust, OpenCV thread count, and OpenCL state.
- Performance timing receipt with all raw repetitions, group medians, overall
  median, p95, min/max, dimensions, implementation, and allocate/reuse mode.
- Canonical VGA/1080p/4K manifest with twelve initial competitive workloads.
- Aggregate runner for the existing vision correctness and RGB-D performance
  suites.
- Standard-library contract tests wired into the Python CI job.

## Verification

Commands run from `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
.venv\Scripts\python.exe bench/opencv_comparison/test_report.py
.venv\Scripts\python.exe -m py_compile bench/opencv_comparison/report.py bench/opencv_comparison/run.py bench/opencv_vision_comparison/run.py bench/opencv_rgbd_comparison/run.py
.venv\Scripts\python.exe bench/opencv_comparison/run.py --output-dir target/opencv-comparison/epic101-final
```

The contract suite passed five tests. The aggregate run passed the complete
vision correctness suite and all three existing RGB-D speed gates using
OpenCV 4.10.0 and SpatialRust 1.0.0 on the recorded Windows machine. The final
local medians showed approximately 1.48x for dense XYZ allocation, 2.06x for
reused dense XYZ output, and 20.02x for colored RGB-D point-cloud construction.
These values are a dated local receipt, not universal performance claims.

Generated JSON remains under `target/` and is not committed. Future published
claims must cover every applicable profile from the manifest and preserve the
full environment and timing receipt.
