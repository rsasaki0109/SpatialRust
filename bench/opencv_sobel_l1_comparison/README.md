# OpenCV fused Sobel L1 comparison

This harness compares exact 3×3 grayscale Sobel L1 magnitude,
`abs(Gx) + abs(Gy)`, through public Python APIs. OpenCV uses its paired
`spatialGradient` primitive followed by two `absdiff` stages and `add`;
SpatialRust fuses the same integer operation into one source traversal and one
output write.

```powershell
python bench/opencv_sobel_l1_comparison/performance.py `
  --output target/opencv-sobel-l1-performance.json
```

OpenCL is disabled, OpenCV receives the logical CPU count, input is seeded
packed random `uint8`, and allocate/reuse calls are paired and interleaved.
Every timing is gated by exact `int16` equality plus 300 randomized cases. The
result is a machine-specific fused-workload receipt, not a claim that the
standalone SpatialRust paired-gradient primitive beats OpenCV
`spatialGradient`.
