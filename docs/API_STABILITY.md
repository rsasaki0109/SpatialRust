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
| `SpatialRuntime`, `CpuRuntime` | Backend identity and execution-policy compatibility boundary |
| `ExecutionReceipt`, `ExecutionOutput<T>` | Requested/resolved policy, stage, and transfer accounting boundary |

`ExecutionPolicy::Auto` is the only policy that permits an algorithm to choose
CPU fallback. `ExecutionPolicy::Gpu(DeviceKind)` is an explicit backend
request; unsupported backends and invalid device targets must return an error
instead of silently running on CPU. The GPU stage crates are being migrated to
this contract incrementally.

### Provisional

| Symbol | Notes |
| --- | --- |
| `SpatialTensor`, `SpatialTensorChunk` | Provisional chunked views over `PointCloud` (`spatial_tensor()`) |
| `AoSoAXyzChunk`, `SpatialTensorChunk::pack_xyz*` | Provisional interleaved chunk packing (`tensor-aoso`) |
| `TransferDirection`, `TransferStats` | Provisional transfer accounting shared by execution backends |

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
| Pipeline | `MvpPipeline`, `MvpPipelineConfig`, `MvpPipelineResult`, `MvpPipelineReceipt` |

### Provisional

| Area | Notes |
| --- | --- |
| MVP CLI flags | `--bounds`, `--resolution`, `--repeat` may gain aliases |
| HTTP COPC (`mvp-http`) | URL IO is stable; timeout/retry policy may change |
| Image (`image`) | Packed/planar ownership, metadata, regions, and strided view contracts are stable |
| Camera (`camera`, `camera-rgbd`) | Pinhole/Brown-Conrady models and explicit RGB-D conversion entry points are stable |
| Image IO (`image-io-*`) | Bounded codecs, typed decoded pixels, and source metadata are provisional |
| AI (`ai-*`) | Backend/session, named dynamic I/O, copy policy, I/O binding, mock backend, and ONNX Runtime adapter APIs are provisional |
| Vision (`vision-*`) | Base errors/borders, resize/filter entry points, detection/dense data contracts, and Feature2D data contracts are stable; geometry, stereo, optical flow, odometry, photography, video, and AI adapters remain provisional |
| Tensor (`tensor-*`) | Dtype/layout/device ownership, typed host storage, external host owner, and DLPack APIs are provisional |
| Records (`records`, `records-receipt-json`) | The 1.2 memory/options/cancellation/receipt and bounded source/sink/chunk contracts are stable; schema migration and concrete prefetch/recycling implementations remain provisional |
| Streaming IO/pipeline/CLI | Format adapters, spool implementation, chunk operations, `StreamingPipeline`, CLI flags, and Python iterator are additive and provisional |
| Arrow (`arrow-*`) | Arrow C Data/Stream/Device bridges for point clouds are provisional |
| Sync (`sync`, `sync-mcap`) | Clock domains, frame graphs, stamped records, and deterministic episode replay are provisional |
| Mapping (`mapping`) | Trajectories, relative motion estimators, pose graphs, loop closure, and feature-gated vision-odometry bridges are provisional |
| Scene (`scene`, `scene-gaussian`) | TSDF/surfel/mesh reconstruction and Gaussian scene containers are provisional |
| Semantic (`semantic`) | Embeddings, open-vocab labels, fusion, and semantic search are provisional |
| Episode (`episode`) | Embodied episode schemas, annotations, augmentation, eval, and provenance are provisional |
| Runtime (`runtime`, `runtime-graph`, `runtime-ros2`) | Bounded pipelines/graphs, fusion schedules, transfer receipts, tracing, diagnostics, and ROS 2 adapters are provisional |
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
| `spatialrust-records` | Stable bounded foundation | `MemoryBudget`, `MemoryTracker`, `CancellationToken`, `StreamOptions`, `StreamingReceipt`, `SpatialRecordChunk`, and bounded source/sink traits are stable for 1.2; schema evolution and concrete adapters remain provisional; Arrow-free |
| `spatialrust-arrow` | Provisional | Arrow C Data/Stream/Device adapters; optional features only |
| `spatialrust-sync` | Provisional | Sensor clocks, frame graphs, stamped records, deterministic replay; MCAP file codecs gated |
| `spatialrust-mapping` | Provisional | Trajectories, odometry traits, pose graphs, loop closure, and explicit vision motion bridges |
| `spatialrust-scene` | Provisional | TSDF, surfels, meshes; Gaussian containers + CPU soft-splat behind `gaussian` |
| `spatialrust-semantic` | Provisional | Embeddings, entities, multimodal fusion/search |
| `spatialrust-episode` | Provisional | Episode schema, annotation, augmentation, eval, provenance |
| `spatialrust-runtime` | Provisional | Bounded pipelines and execution graphs, explicit transfer receipts, tracing/diagnostics; ROS 2 adapters gated |
| `spatialrust-ros2` | Provisional | Read-only rosbag2 SQLite PointCloud2 CDR source; native executors and custom message bindings remain separate |
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

- native `rclrs` execution and custom ROS 2 message bindings
- `gpu-cuda` backend (feature placeholder only)
- `SpatialTensor` chunked views (provisional API in `spatialrust-core`)

## Deprecation policy (from 1.0 onward)

The machine-readable freeze list for the stable vision foundation is
`StabilityRegistry::vision_v1_surface()`. The
`vision_api_v1` compile-and-behavior test must remain green on Linux, Windows,
and macOS. A symbol marked stable there follows the deprecation policy below;
group-level provisional entries may evolve behind their existing feature flag.
Every Vision 1 release candidate must additionally pass `Vision1ReleaseGate`,
which requires named cross-platform/test/audit evidence, fixed performance
measurements, OpenCV receipts, runnable examples, and migration-policy
acknowledgement.

Vision 2 keeps the Vision 1 stable surface unchanged and adds provisional plan,
workspace, fused preprocessing, and GPU-resident entries through
`StabilityRegistry::vision_v2_surface()`. A Vision 2 candidate must pass
`Vision2ReleaseGate`, including three-OS conformance, native/Python
allocate/reuse budgets, explicit resource and transfer measurements, generated
documentation, dated receipts, and the `vision-2` migration policy.

SpatialRust 1.2 freezes the additive bounded record foundation through
`StabilityRegistry::bounded_streaming_v1_2_surface()`. Concrete format
adapters, spool implementation, chunk algorithms, pipeline builder, CLI flags,
and Python iterator remain provisional behind named features. Every 1.2
candidate must pass `Streaming12ReleaseGate`, including three-OS conformance,
memory/spill/cleanup/copy/transfer/determinism/file-handle budgets, all five
Epic receipts, the runnable example, and the `bounded-streaming-1.2` migration
policy. `SpatialTensor` remains provisional and is not an out-of-core source.

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
