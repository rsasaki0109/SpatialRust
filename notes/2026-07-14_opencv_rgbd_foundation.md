# OpenCV-oriented RGB-D foundation

Date: 2026-07-14

## Delivered

- `spatialrust-image`: typed packed images and validated strided views
- `spatialrust-camera`: pinhole projection/unprojection, Brown–Conrady
  distortion, depth to XYZ, and aligned RGB-D to XYZRGB
- `camera-rgbd` meta-crate feature and full MVP integration test
- NumPy/Python `rgbd_to_point_cloud` API, type stub, contract test, and demo
- OpenCV `cv2.rgbd.depthTo3d` numerical/timing comparison harness
- Criterion CPU benchmark for a 640x480 RGB-D frame

## Local validation

Windows, release build, Python 3.12.10:

| Check | Result |
| --- | --- |
| OpenCV comparison points | 76,704 |
| Maximum XYZ difference | `5.960e-08 m` |
| SpatialRust Python median | `2.065 ms` |
| OpenCV Python median | `0.264 ms` |
| Native SpatialRust 640x480 RGB-D | `6.47–6.86 ms` |

The native benchmark processes 307,200 colored points. The OpenCV comparison
uses a 320x240 depth frame and includes Python API boundary costs for both
libraries. Results are machine-specific and should be rerun before publishing.

## Reproduction

```powershell
cargo test -p spatialrust-camera
cargo test -p spatialrust --features mvp,camera-rgbd --test rgbd_pipeline
cargo bench -p spatialrust-camera --bench rgbd
python bench/opencv_rgbd_comparison/run.py
python crates/spatialrust-py/examples/rgbd_pipeline.py
```
