# OpenCV fused resize-to-gray comparison

This harness compares a canonical two-stage OpenCV pipeline (`resize` followed
by `cvtColor`) with SpatialRust's single-pass bilinear RGB resize-to-gray API.
The profiles are camera-pyramid half reductions from 1080p, 4K, and 8K.

```powershell
.\.venv\Scripts\python.exe bench/opencv_fused_resize_gray_comparison/performance.py `
  --output target/opencv-fused-resize-gray-performance.json
```

Allocated and caller-owned outputs are measured separately with paired,
interleaved samples. Randomized dimensions and non-contiguous inputs verify
exact parity with SpatialRust's unfused plan and bound OpenCV disagreement.
