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
| Image (`image`) | Packed/planar ownership, metadata, regions, and strided view contracts are stable |
| Camera (`camera`, `camera-rgbd`) | Pinhole/Brown-Conrady models and explicit RGB-D conversion entry points are stable |
| Image IO (`image-io-*`) | Bounded codecs, typed decoded pixels, and source metadata are provisional |
| AI (`ai-*`) | Backend/session, named dynamic I/O, copy policy, I/O binding, mock backend, and ONNX Runtime adapter APIs are provisional |
| Vision (`vision-*`) | Base errors/borders, resize/filter entry points, detection/dense data contracts, and Feature2D data contracts are stable; geometry, stereo, optical flow, and AI adapters remain provisional |
| Tensor (`tensor-*`) | Dtype/layout/device ownership, typed host storage, external host owner, and DLPack APIs are provisional |
| Records (`records`) | Versioned `SpatialRecord`, schema compatibility/migration, and chunked record streams are provisional |
| Arrow (`arrow-*`) | Arrow C Data/Stream/Device bridges for point clouds are provisional |
| Sync (`sync`, `sync-mcap`) | Clock domains, frame graphs, stamped records, and deterministic episode replay are provisional |
| Mapping (`mapping`) | Trajectories, relative motion estimators, pose graphs, and loop-closure candidates are provisional |
| Scene (`scene`, `scene-gaussian`) | TSDF/surfel/mesh reconstruction and Gaussian scene containers are provisional |
| Semantic (`semantic`) | Embeddings, open-vocab labels, fusion, and semantic search are provisional |
| Episode (`episode`) | Embodied episode schemas, annotations, augmentation, eval, and provenance are provisional |
| Runtime (`runtime`, `runtime-ros2`) | Bounded pipelines, tracing, diagnostics, and ROS 2 adapter contracts are provisional |
| Interchange (`interchange-*`) | glTF JSON and OpenUSD stage adapter contracts are provisional |
| Distribute (`distribute`) | Partition graphs, backpressure, and named transfers are provisional |
| Platform (`platform`) | API stability registry, conformance reports, security checklists, and LTS policy are provisional |
| GPU image (`gpu-image`) | Texture-backed `GpuImage`, upload/readback, receipts, pooling, and image compute kernels are provisional through the Vision 1.0 gate |

## Algorithm crates

Each algorithm crate follows:

```
spatialrust-<area> / feature-<name>
```

| Crate | 1.0 status | Notes |
| --- | --- | --- |
| `spatialrust-math` | Stable primitives | `Vec3`, `Mat4`, `Isometry3` |
| `spatialrust-image` | Stable | Packed/planar ownership, metadata, regions, and strided CPU views; no hidden device transfers |
| `spatialrust-image-io` | Provisional | Standard codecs by default; TIFF/OpenEXR independently gated |
| `spatialrust-tensor` | Provisional | Generic tensor descriptors, explicit CPU ownership, image/spatial bridges, and feature-gated DLPack major-version 1 ABI |
| `spatialrust-ai` | Provisional | Runtime-independent session contract; ONNX Runtime CPU and hardware providers are independently gated |
| `spatialrust-records` | Provisional | Versioned records, schema evolution, chunked host streams; Arrow-free |
| `spatialrust-arrow` | Provisional | Arrow C Data/Stream/Device adapters; optional features only |
| `spatialrust-sync` | Provisional | Sensor clocks, frame graphs, stamped records, deterministic replay; MCAP file codecs gated |
| `spatialrust-mapping` | Provisional | Trajectories, synthetic odometry traits, pose graphs, loop-closure candidates |
| `spatialrust-scene` | Provisional | TSDF, surfels, meshes; Gaussian containers + CPU soft-splat behind `gaussian` |
| `spatialrust-semantic` | Provisional | Embeddings, entities, multimodal fusion/search |
| `spatialrust-episode` | Provisional | Episode schema, annotation, augmentation, eval, provenance |
| `spatialrust-runtime` | Provisional | Bounded pipelines/tracing/diagnostics; ROS 2 adapters gated |
| `spatialrust-interchange` | Provisional | glTF JSON mesh bridge; USDA ASCII OpenUSD stage adapter |
| `spatialrust-distribute` | Provisional | Partition graphs, topo schedules, backpressure queues, named measurable transfers |
| `spatialrust-platform` | Provisional | Stability registry, conformance summaries, security checklist, LTS policy, performance budgets, release gate |
| `spatialrust-camera` | Stable foundation | Pinhole/Brown–Conrady and named RGB-D conversion entry points are stable; mono/stereo/fisheye/hand-eye/BA calibration contracts are additive and provisional |
| `spatialrust-vision` | Stable foundation | Errors, borders, resize/filter entry points, reusable resize/gray/normalize/CHW outputs, detection/dense and Feature2D data contracts are stable; geometry/stereo/flow/AI adapters remain provisional |
| `spatialrust-search` | Stable with features | KD-tree behind `search-kdtree`; **chunked query traits** and **`search-parallel`** provisional |
| `spatialrust-filtering` | Provisional | GPU thresholds may move |
| `spatialrust-features` | Provisional | Normal GPU path still tuning |
| `spatialrust-segmentation` | Provisional | RANSAC configs may extend; **GPU plane scoring** behind `segment-ransac-plane-gpu` |
| `spatialrust-registration` | Provisional | New backends (TEASER++, etc.) expected |
| `spatialrust-gpu` | Provisional | `WgpuRuntime`, `GpuBufferPool`; `GpuImage` / image kernels behind `gpu-image`; voxel/AoSoA kernel APIs still tuning |
| `spatialrust-py` | Stable user surface | Stubs enforced by `mypy.stubtest`; new vision functions remain provisional with the Rust APIs |

## Explicitly out of 1.0 scope

- `spatialrust-ros2` (not started)
- `gpu-cuda` backend (feature placeholder only)
- `SpatialTensor` chunked views (provisional API in `spatialrust-core`)

## Deprecation policy (from 1.0 onward)

The machine-readable freeze list for the stable vision foundation is
`StabilityRegistry::vision_v1_surface()`. The
`vision_api_v1` compile-and-behavior test must remain green on Linux, Windows,
and macOS. A symbol marked stable there follows the deprecation policy below;
group-level provisional entries may evolve behind their existing feature flag.

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
