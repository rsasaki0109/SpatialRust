# OpenCV vision comparison

This deterministic harness compares SpatialRust's Python-visible CPU vision
primitives with OpenCV: linear, median, and bilateral filters; Sobel, Scharr,
Laplacian, Gaussian pyramids, morphology, thresholds, histograms, CLAHE, integral images,
and Canny across 3/5/7 Sobel apertures and L1/L2 gradients;
Harris, Shi–Tomasi, and FAST-9/16 keypoint coordinates/order (plus exact FAST scores);
Hamming/L2 brute-force nearest matches; ORB keypoint repeatability and descriptor layout;
homography transfer residuals, PnP translation vs OpenCV, and StereoBM center disparity
on a synthetic textured pair; four resize filters; RGB-to-gray/HSV conversion;
bilinear remap; NMS; and connected-component areas.

From the repository root, after installing the editable Python extension:

```powershell
$env:PYTHONPATH=(Resolve-Path '.\.venv\Lib\site-packages').Path
python bench\opencv_vision_comparison\run.py
```

The command prints an Epic 101 machine-readable JSON report and exits non-zero
when a documented numerical tolerance is exceeded. Pass `--output PATH` to
retain the report. OpenCV is comparison/test tooling only; it is not a Rust
runtime dependency. The shared report contract and workload registry are in
[`../opencv_comparison`](../opencv_comparison/README.md).

The performance suite measures allocate/reuse bilinear resize, RGB-to-gray,
AI CHW preprocessing, Gaussian blur, Sobel X, morphology open, and Canny at
VGA, 1080p, and 4K. OpenCV and SpatialRust calls are seeded and interleaved;
short calls are batched to reduce timer noise. Reports preserve raw samples,
mean/median/p95, standard deviation, coefficient of variation, median absolute
deviation, batch size, throughput, detailed accuracy, and the environment/thread
receipt:

```powershell
python bench\opencv_vision_comparison\performance.py `
  --output target\opencv-comparison\vision-performance.json
```

The suite reports the faster implementation per workload. It intentionally
does not impose a blanket "SpatialRust faster than OpenCV" gate: optimized
OpenCV image kernels are expected to lead many scalar CPU paths.
