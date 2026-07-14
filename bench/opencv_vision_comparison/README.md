# OpenCV vision comparison

This deterministic harness compares SpatialRust's Python-visible CPU vision
primitives with OpenCV: linear, median, and bilateral filters; Sobel, Scharr,
Laplacian, Gaussian pyramids, morphology, thresholds, histograms, CLAHE, integral images,
and Canny across 3/5/7 Sobel apertures and L1/L2 gradients;
Harris, Shi–Tomasi, and FAST-9/16 keypoint coordinates/order (plus exact FAST scores);
Hamming/L2 brute-force nearest matches; ORB keypoint repeatability and descriptor layout;
four resize filters; RGB-to-gray/HSV conversion;
bilinear remap; NMS; and connected-component areas.

From the repository root, after installing the editable Python extension:

```powershell
$env:PYTHONPATH=(Resolve-Path '.\.venv\Lib\site-packages').Path
python bench\opencv_vision_comparison\run.py
```

The command prints a JSON report and exits non-zero when a documented numerical
tolerance is exceeded. OpenCV is comparison/test tooling only; it is not a Rust
runtime dependency.
