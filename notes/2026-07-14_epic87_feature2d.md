# Epic 87 Feature2D completion record

Date: 2026-07-14 (Asia/Tokyo)

## Delivered contracts

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\feature2d.rs`
  owns validated keypoint, descriptor, feature-set, and match data contracts.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\corners.rs`
  implements Harris, Shi–Tomasi, and FAST-9/16 on packed or strided CPU views.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\orb.rs`
  implements multi-scale FAST selection, intensity-centroid orientation, and a
  stable fixed-seed rotated BRIEF 256-bit descriptor.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\matcher.rs`
  implements deterministic Hamming and L2 nearest matching with ratio,
  cross-check, and maximum-distance filters.

The SpatialRust BRIEF sampling table is deliberately stable but is not the
private learned OpenCV table. Interoperable distance semantics and detector
repeatability are therefore tested independently from descriptor bit identity.

## Verification

- `cargo test -p spatialrust-vision --all-features`: 64 unit tests and 8
  generated property tests passed.
- `cargo clippy -p spatialrust-vision --all-features --all-targets -- -D warnings` passed.
- Python binding suite: 63 passed, 1 optional ONNX test skipped.
- `python -m mypy.stubtest spatialrust --ignore-missing-stub` passed.
- `C:\Users\rsasa\Workspace\SpatialRust\bench\opencv_vision_comparison\run.py`
  passed against OpenCV 4.13.0. FAST raw/NMS, Harris, and Shi–Tomasi coordinates
  match exactly. Hamming nearest matches match exactly; L2 train indices match
  exactly with maximum distance error `4.76837158203125e-7`. ORB returned 200
  keypoints in both implementations with 35% SpatialRust-to-OpenCV coordinate
  repeatability within two pixels.

## Scalar CPU baseline

Criterion release benchmark on Intel Core i7-9750H, Windows 11 Insider,
rustc 1.92.0. Intervals are 95% estimates from 10 samples.

| Operation | 640p | 1080p | 4K |
| --- | ---: | ---: | ---: |
| FAST-9/16 | 158.83 ms | 842.61 ms | 3.2482 s |
| ORB, max 500 | 747.49 ms | 2.7778 s | 11.634 s |

These timings are the correctness-first scalar baseline, not a performance
target. They identify FAST scoring, pyramid construction, and Gaussian
descriptor preparation as candidates for SIMD/parallel and Epic 89 GPU work.
