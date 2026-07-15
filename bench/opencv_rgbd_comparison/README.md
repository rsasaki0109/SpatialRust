# OpenCV RGB-D comparison

Compares SpatialRust RGB-D conversion against OpenCV `cv.rgbd.depthTo3d` on a
deterministic depth image and intrinsics.

Gates (median of repeated trials; OpenCL disabled):

1. Dense `H×W×3` XYZ — alloc and fill-into vs OpenCV
2. Colored point cloud — `rgbd_to_point_cloud` vs OpenCV depthTo3d + NumPy
   mask/color gather

```bash
pip install maturin numpy opencv-contrib-python
cd crates/spatialrust-py
maturin develop --release
cd ../..
python bench/opencv_rgbd_comparison/run.py
```

Exits non-zero when XYZ error exceeds `1e-5` m or any gated path is slower
than OpenCV.
