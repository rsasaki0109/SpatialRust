# OpenCV packed RGB8-to-gray comparison

This focused harness compares public Python APIs for packed RGB `uint8` to
gray `uint8` conversion. It measures allocated and caller-owned output calls
at VGA, 1080p, 4K, and 8K.

```powershell
.\.venv\Scripts\python.exe bench/opencv_rgb_gray_comparison/performance.py `
  --output target/opencv-rgb-gray-performance.json
```

OpenCL is disabled, OpenCV receives the logical CPU count, and paired timings
use seeded random input. Three hundred arbitrary-size cases, including
non-contiguous inputs, gate the Q14 BT.601 path at a maximum absolute error of
one `uint8` level.
