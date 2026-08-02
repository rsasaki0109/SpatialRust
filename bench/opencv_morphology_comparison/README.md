# OpenCV rectangular morphology comparison

This focused harness compares bit-exact grayscale rectangular opening through
the public Python APIs in both allocated and caller-owned-output/workspace
modes. It covers the common 5×5 case and a 511×511
background-estimation/document workload where the window-area-independent
SpatialRust engine can overtake OpenCV on large images.
The centered 5×5 Replicate case uses a fixed-comparison row-major path, while
other sizes, anchors, and borders retain the window-area-independent engine.

```powershell
python bench/opencv_morphology_comparison/performance.py `
  --output target/opencv-morphology-performance.json
```

OpenCL is disabled, input is seeded packed random `uint8`, calls are paired and
interleaved, and every timing is gated by exact output equality. The report is
a machine-specific receipt, not a blanket speed guarantee.
