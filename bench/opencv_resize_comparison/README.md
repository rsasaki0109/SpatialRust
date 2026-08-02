# OpenCV packed RGB8 resize comparison

This focused harness compares public Python APIs for packed RGB `uint8`
bilinear resize at exactly half width and half height. It measures both
allocated and caller-owned output calls at VGA, 1080p, 4K, and 8K.

```powershell
.\.venv\Scripts\python.exe bench/opencv_resize_comparison/performance.py `
  --output target/opencv-resize-performance.json
```

OpenCL is disabled, OpenCV receives the logical CPU count, and paired timings
use seeded random input. The canonical half-scale output must be bit-exact.
Three hundred arbitrary-dimension cases, including non-contiguous inputs, gate
the planned fixed-point path at a maximum absolute error of one `uint8` level.
