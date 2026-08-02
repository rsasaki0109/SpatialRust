# OpenCV specialized Gaussian comparison

This harness compares 5×5 RGB `uint8` Gaussian blur with sigma 1.2 and
Reflect101 borders through public Python APIs. It measures both allocated and
caller-owned output calls at 1080p, 4K, and 8K.

```powershell
python bench/opencv_gaussian_comparison/performance.py `
  --output target/opencv-gaussian-performance.json
```

OpenCL is disabled, OpenCV receives the logical CPU count, and paired timings
use seeded packed input. The timing gate is backed by 300 randomized 3×3,
5×5, and 7×7 cases, including non-contiguous inputs, with maximum absolute
error limited to two `uint8` levels. This receipt reports standalone Gaussian
honestly; it does not imply that SpatialRust already leads OpenCV on this
kernel.
