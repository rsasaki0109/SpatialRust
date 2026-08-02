# Fused resize-normalize-CHW acceleration

This Epic 115C slice adds safe allocating and caller-owned APIs that fuse Q11
bilinear RGB resize, `f32` scaling/normalization, and planar CHW packing. The
output is written directly in model-input layout without an intermediate HWC
image. Arbitrary scale, per-channel mean, and non-zero finite standard
deviation remain explicit parameters.

## Correctness

- Fused output is bit-exact with `BilinearResizeU8Plan::resize` followed by
  `pack_chw` for arbitrary dimensions and normalization parameters.
- Packed and strided input plus caller-owned output paths are covered in Rust.
- Three hundred seeded arbitrary-size Python cases include non-contiguous
  input and retain exact SpatialRust fused/unfused parity.
- Against OpenCV 4.13 `dnn.blobFromImage`, maximum float disagreement is
  `0.003921628` (one `u8` level after scaling by `1/255`).

## Native caller-owned medians

| Input → model shape | Unfused | Fused | Improvement |
| --- | ---: | ---: | ---: |
| 1920×1080 → 640×640 | 1.270 ms | 1.106 ms | 1.15× |
| 3840×2160 → 640×640 | 1.385 ms | 1.239 ms | 1.12× |
| 3840×2160 → 1280×720 | 2.587 ms | 2.345 ms | 1.10× |

## OpenCV 4.13 Python medians

OpenCL was disabled and OpenCV used 12 threads. OpenCV's reference is the
integrated `cv2.dnn.blobFromImage` call, not a slower NumPy composition.

| Input → model shape | OpenCV allocate | SpatialRust allocate | Allocate result | SpatialRust reuse | Reuse vs OpenCV allocate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920×1080 → 640×640 | 3.570 ms | 1.617 ms | **SpatialRust 2.21×** | 1.114 ms | **SpatialRust 3.56×** |
| 3840×2160 → 640×640 | 4.272 ms | 2.117 ms | **SpatialRust 2.02×** | 1.420 ms | **SpatialRust 3.02×** |
| 3840×2160 → 1280×720 | 8.359 ms | 3.592 ms | **SpatialRust 2.33×** | 2.472 ms | **SpatialRust 3.48×** |

The focused harness at
`C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_fused_resize_chw_comparison`
emits the complete environment, dispersion, and raw paired samples.
