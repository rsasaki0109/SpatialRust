# OpenCV vision comparison

This deterministic harness compares SpatialRust's Python-visible CPU vision
primitives with OpenCV: four resize filters, RGB-to-gray/HSV conversion,
bilinear remap, NMS, and connected-component areas.

From the repository root, after installing the editable Python extension:

```powershell
$env:PYTHONPATH=(Resolve-Path '.\.venv\Lib\site-packages').Path
python bench\opencv_vision_comparison\run.py
```

The command prints a JSON report and exits non-zero when a documented numerical
tolerance is exceeded. OpenCV is comparison/test tooling only; it is not a Rust
runtime dependency.
