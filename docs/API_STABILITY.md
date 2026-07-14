# API stability (Epic 62)

Design version: **v0.1 → v1.0 candidate**

This document tracks which public APIs are intended to remain stable for the
SpatialRust **1.0.0** release. Algorithm crates may evolve behind feature flags
until their individual 1.0 milestones.

## Stability tiers

| Tier | Meaning |
| --- | --- |
| **Stable** | Semver-compatible; breaking changes only in major releases |
| **Provisional** | Public but may change before 1.0 |
| **Unstable** | Feature-gated or explicitly experimental |

## `spatialrust-core` (freeze first)

### Stable for 1.0

| Symbol | Notes |
| --- | --- |
| `PointCloud`, `PointCloudBuilder` | Primary data container |
| `PointBuffer`, `PointBufferSet` | Columnar storage |
| `PointSchema`, `PointField`, `FieldSemantic`, `StandardSchemas`, `DType` | Schema model |
| `HasPositions3`, `HasNormals3`, `HasIntensity` | Capability traits |
| `SpatialAlgorithm` | Algorithm trait boundary |
| `ExecutionPolicy` | `Auto` / `Cpu` / `CpuSingle` / `Gpu` |
| `SpatialError`, `SpatialResult` | Error surface |
| `SpatialMetadata`, `FrameId`, `Timestamp` | Frame metadata |
| `Device`, `DeviceKind`, `CpuDevice` | Device tagging (CUDA is enum-only until backend lands) |

### Provisional

| Symbol | Notes |
| --- | --- |
| `SpatialTensor`, `SpatialTensorChunk` | Provisional chunked views over `PointCloud` (`spatial_tensor()`) |
| `AoSoAXyzChunk`, `SpatialTensorChunk::pack_xyz*` | Provisional interleaved chunk packing (`tensor-aoso`) |

### Rules (unchanged at 1.0)

- No IO, GPU impl, ROS2, or AI runtimes in `spatialrust-core`
- `#![deny(unsafe_code)]` on core

## `spatialrust` meta crate

### Stable for 1.0

| Area | Symbols |
| --- | --- |
| IO | `read_point_cloud_file`, `write_point_cloud_file` |
| COPC | `read_copc_file`, `read_copc_file_with_query`, `CopcBounds`, `CopcQuery`, `CopcFileInfo`, `CopcWriterParams` |
| Transform | `bounding_box`, `centroid`, `apply_transform`, `normalize_unit_sphere`, `merge_clouds` |
| Pipeline | `MvpPipeline`, `MvpPipelineConfig`, `MvpPipelineResult` |

### Provisional

| Area | Notes |
| --- | --- |
| MVP CLI flags | `--bounds`, `--resolution`, `--repeat` may gain aliases |
| HTTP COPC (`mvp-http`) | URL IO is stable; timeout/retry policy may change |
| Image/camera (`image`, `camera-rgbd`) | Typed image, calibration, distortion, and RGB-D APIs are provisional |
| Vision (`vision-*`) | CPU preprocessing, warp, detection, masks, and dense spatial bridges are provisional |

## Algorithm crates

Each algorithm crate follows:

```
spatialrust-<area> / feature-<name>
```

| Crate | 1.0 status | Notes |
| --- | --- | --- |
| `spatialrust-math` | Stable primitives | `Vec3`, `Mat4`, `Isometry3` |
| `spatialrust-image` | Provisional | Packed ownership and strided CPU views; no hidden device transfers |
| `spatialrust-camera` | Provisional | Pinhole/Brown–Conrady and RGB-D conversion |
| `spatialrust-vision` | Provisional | Feature-gated CPU image algorithms and explicit point-cloud bridges |
| `spatialrust-search` | Stable with features | KD-tree behind `search-kdtree`; **chunked query traits** and **`search-parallel`** provisional |
| `spatialrust-filtering` | Provisional | GPU thresholds may move |
| `spatialrust-features` | Provisional | Normal GPU path still tuning |
| `spatialrust-segmentation` | Provisional | RANSAC configs may extend; **GPU plane scoring** behind `segment-ransac-plane-gpu` |
| `spatialrust-registration` | Provisional | New backends (TEASER++, etc.) expected |
| `spatialrust-gpu` | Provisional | `WgpuRuntime`, `GpuBufferPool` upload/recycle API stable; kernel APIs still tuning |
| `spatialrust-py` | Stable user surface | Stubs enforced by `mypy.stubtest`; new vision functions remain provisional with the Rust APIs |

## Explicitly out of 1.0 scope

- `spatialrust-ros2` (not started)
- `spatialrust-ai` / ONNX / DLPack export (not started)
- `gpu-cuda` backend (feature placeholder only)
- `SpatialTensor` chunked views (provisional API in `spatialrust-core`)

## Deprecation policy (from 1.0 onward)

1. Deprecate in minor release (`#[deprecated]` + CHANGELOG)
2. Remove no sooner than next major release
3. Migration notes in CHANGELOG and rustdoc

## v1.0.0 release checklist

- [x] This document reviewed; all **Stable** items covered by tests
- [x] `cargo test --workspace` green (run before tag)
- [x] `cargo test -p spatialrust --features mvp --test mvp_public_copc` green
- [x] Python stubtest green (CI `python-bindings` job)
- [x] Public COPC harness documented in `bench/public_copc/`
- [x] CHANGELOG 1.0 section with breaking-change policy
