# Epic 88 geometry / motion / stereo completion record

Date: 2026-07-15 (Asia/Tokyo)

## Delivered contracts

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\geometry.rs`
  owns correspondences, camera matrix, projective models, absolute/relative
  pose, and robust-estimation result types.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\multiview.rs`
  implements normalized DLT and deterministic RANSAC for H/F/E plus triangulation
  and essential pose recovery.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\pnp.rs`
  implements DLT-initialized PnP with Gauss–Newton refine and six-point RANSAC.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\optical_flow.rs`
  implements sparse pyramidal Lucas–Kanade tracking without a Feature2D dependency.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\stereo.rs`
  implements stereo rig/rectify maps, SAD BM, and disparity depth/XYZ reproject
  into packed `Image` buffers (no `dense` coupling).

## Verification

- `cargo test -p spatialrust-vision --features geometry --lib`
- Property tests under `full` include noisy homography and PnP recovery.
- Python bindings expose `estimate_homography_ransac`, `solve_pnp`, and
  `stereo_block_match` with stubs and binding tests.
- OpenCV comparison documents residual and translation tolerances rather than
  bit-identical matrices; StereoBM center disparity is checked on a synthetic
  textured pair.

## Scalar CPU baseline

Criterion `geometry` bench covers PnP/homography correspondence counts and
640p/1080p/4K sparse LK plus StereoBM. Timings are correctness-first baselines
for later Epic 89 acceleration.
