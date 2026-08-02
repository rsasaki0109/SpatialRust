# AI-ready image and vision foundation (Epics 75–79)

Date: 2026-07-14

## Delivered

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-image` now provides
  checked mutable strided views and ROI/subviews, planar and interleaved layouts,
  and explicit color/range/alpha metadata.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision` contains
  independently gated resize, preprocess, warp, detection, dense-map, and
  spatial bridge modules. CPU ownership and camera/point-cloud conversion are
  explicit; there are no hidden device copies.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py\src\lib.rs` and
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py\spatialrust.pyi`
  expose the AI preprocessing/post-processing and PointMap bridge to NumPy.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py\examples\vision_ai_pipeline.py`
  runs letterbox/CHW, NMS, mask components/RLE, PointMap conversion, and the MVP
  point-cloud pipeline in one executable example.

## Verification

- Rust vision unit tests: 22 passed.
- Generated property tests: 3 passed (identity resize, both RLE orders, IoU laws).
- Python binding tests: 48 passed; `mypy.stubtest` passed.
- All seven `spatialrust-vision` features and all eight meta-crate `vision-*`
  features built separately with default features disabled.
- `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_vision_comparison\run.py`
  passed against OpenCV 4.13.0: resize max uint8 error 0/1/1/0 for
  nearest/bilinear/bicubic/area, gray/HSV max error 1, remap error 0, and exact
  NMS indices/component areas.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\benches\preprocess.rs`
  measured 1280x720 RGB letterbox plus normalization/CHW packing to 640x640 at
  12.075–13.621 ms (Criterion interval on this development machine).
