# OpenCV fused Sobel threshold comparison

This harness compares an exact binary edge mask built from a first-order 3x3
Sobel response. OpenCV uses `Sobel(CV_16S)`, `convertScaleAbs`, then
`threshold(THRESH_BINARY)`. SpatialRust fuses the same steps into one
three-row-ring operation and one `uint8` output.

```powershell
python bench/opencv_sobel_threshold_comparison/performance.py `
  --output target/opencv-sobel-threshold-performance.json
```

OpenCL is disabled, inputs are seeded packed `uint8`, and allocate/reuse calls
are paired and interleaved. Timings are gated by exact pixels for both X and Y
derivatives across 300 randomized cases. Packed NumPy input is borrowed without
a copy; non-contiguous input is explicitly packed. Results are workload- and
host-specific.
