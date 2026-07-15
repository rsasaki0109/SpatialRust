# OpenCV fused resize-normalize-CHW comparison

This harness compares OpenCV's integrated `cv2.dnn.blobFromImage` against
SpatialRust's fused bilinear RGB resize, float normalization, and CHW packing.
It covers common 640×640 and 1280×720 model-input shapes.

```powershell
.\.venv\Scripts\python.exe bench/opencv_fused_resize_chw_comparison/performance.py `
  --output target/opencv-fused-resize-chw-performance.json
```

OpenCL is disabled and paired timings measure SpatialRust allocated and
caller-owned outputs separately against the OpenCV integrated call. Three
hundred arbitrary-size cases, including non-contiguous inputs, require exact
SpatialRust fused/unfused parity and bound OpenCV float disagreement.
