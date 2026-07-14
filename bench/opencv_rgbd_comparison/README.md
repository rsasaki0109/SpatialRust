# OpenCV RGB-D comparison

This harness compares SpatialRust `rgbd_to_point_cloud` against OpenCV's
`cv.rgbd.depthTo3d` on the same deterministic depth image and intrinsics.

The OpenCV wheel must include the contrib `rgbd` module:

```bash
pip install maturin numpy opencv-contrib-python
cd crates/spatialrust-py
maturin develop --release
cd ../..
python bench/opencv_rgbd_comparison/run.py
```

The command exits non-zero when valid-point masks differ or maximum XYZ error
exceeds `1e-5` meters. It also reports median runtime for both implementations.
