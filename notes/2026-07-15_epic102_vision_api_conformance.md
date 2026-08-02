# Epic 102: Vision API and cross-platform conformance

Date: 2026-07-15

Epic 102 freezes the SpatialRust Vision 1.x foundation without prematurely
freezing every algorithm or backend. `StabilityRegistry::vision_v1_surface()`
is the machine-readable authority: image ownership and views, pinhole/RGB-D
camera entry points, vision errors/borders, resize/filter entry points,
detection/dense data, and Feature2D data contracts are stable. Geometry,
stereo, optical flow, AI adapters, and `GpuImage` remain provisional.

## Conformance

The `vision_api_v1` integration test composes a strided zero-copy ROI, resize,
filtering, camera project/unproject, NMS, and the stability registry through the
public meta-crate surface. The dedicated CI job runs the following commands on
Linux, Windows, and macOS:

```text
cargo test -p spatialrust-image
cargo test -p spatialrust-camera
cargo test -p spatialrust-vision --no-default-features --features full
cargo test -p spatialrust --no-default-features --features platform,camera-rgbd,vision-full --test vision_api_v1
```

Local Windows verification passed 8 image tests, 5 camera tests, 79 vision unit
tests, 10 vision property tests, 7 platform tests, and the public API contract.
The CI matrix is the authoritative evidence for the other operating systems.
