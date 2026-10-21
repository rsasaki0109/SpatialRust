# SpatialRust development roadmap

This document is the canonical registry for active Epic identifiers,
dependencies, scope, and completion gates. New Epic numbers must be reserved
here before implementation begins.

## Numbering

Historical work used parallel point-cloud/GPU and image-planning tracks, which
caused identifiers 75–79 to appear in both sets of notes. Those historical note
titles remain unchanged. Canonical cross-project numbering resumes at Epic 83,
after the GPU-resident frame work recorded through Epic 82.

## Long-term 2D → AI → 3D program

| Epic | Status | Depends on | Deliverable |
| --- | --- | --- | --- |
| 83 | Complete | Image foundation | `spatialrust-image-io`: bounded stream/memory codecs and metadata |
| 84 | Complete | 83 | Shared CPU imgproc kernels, filters, morphology, thresholds, histograms, Canny, pyramids |
| 85 | Complete | 83–84 | `spatialrust-tensor`, DLPack 1.x versioned ABI, explicit copy/device semantics |
| 86 | Complete | 85 | `spatialrust-ai`, backend traits, ONNX Runtime CPU and explicit I/O binding |
| 87 | Complete | 84 | Feature2D data model, corners, FAST/ORB, descriptors and matching |
| 88 | Complete | 84, 87 | Camera geometry, robust multiview estimation, motion and stereo |
| 89 | Complete | 84–85 | Explicit `GpuImage` upload/readback and chainable wgpu vision kernels |
| 90 | Complete | 86, 88–89 | Model adapters and image → AI → point-cloud end-to-end pipelines |

Dependency flow:

```text
image-io -> CPU imgproc -> tensor/DLPack -> ONNX inference
                    \-> Feature2D -> camera geometry/motion
CPU imgproc + tensor/DLPack -> explicit wgpu vision
ONNX + geometry + wgpu vision -> model adapters and 2D-to-3D demos
```

## North star after Epic 90: perception to spatial intelligence

SpatialRust's long-term goal is to become the Rust-native data plane and
execution framework that turns synchronized sensor streams into queryable,
replayable, and actionable spatial worlds. A user should be able to move from
capture to geometry, AI inference, mapping, semantic understanding, simulation,
and robot action without replacing the core data model or accepting hidden
host/device copies.

Epic 83–90 is the foundation program. The following identifiers are reserved
for its successor program; their implementation scope is refined only after the
foundation contracts they depend on are stable.

| Epic | Status | Depends on | Long-term outcome |
| --- | --- | --- | --- |
| 91 | Complete | 85, 90 | Spatial records and streams: schema evolution, chunked/out-of-core execution, Arrow C Data/Stream/Device interoperability |
| 92 | Complete | 88, 91 | Sensor-time and frame graph: calibrated multimodal synchronization, deterministic replay, MCAP integration |
| 93 | Complete | 87–88, 92 | Localization and mapping: visual/RGB-D/lidar odometry, pose graphs, loop closure, relocalization |
| 94 | Complete | 89, 93 | Scene reconstruction: TSDF, surfels, meshes, and a feature-gated Gaussian scene representation and renderer |
| 95 | Complete | 90–94 | Semantic spatial intelligence: open-vocabulary detections, embeddings on spatial entities, multimodal fusion and search |
| 96 | Complete | 91–95 | Embodied-AI data workflows: episodes, annotation, augmentation, evaluation, model provenance and reproducible replay |
| 97 | Complete | 92–96 | Production robotics runtime: ROS 2 type adaptation/negotiation, bounded pipelines, tracing and failure diagnostics |
| 98 | Complete | 94–97 | Scene and digital-twin interchange through dedicated glTF and OpenUSD adapters |
| 99 | Complete | 89, 91, 97 | Explicit edge/distributed execution: graph partitioning, backpressure and named device/network transfers |
| 100 | Complete | 91–99 | Platform stability milestone: API compatibility, conformance suites, security audits, performance budgets and LTS policy |

Success is measured by end-to-end capabilities rather than crate count:

1. Record a synchronized camera/depth/lidar/IMU episode and replay it
   deterministically through the same bounded execution graph.
2. Produce geometry, trajectories, semantic entities, uncertainty, and model
   provenance in one versioned spatial schema.
3. Share host and device data through explicit, testable ownership boundaries;
   every unavoidable copy is named and measurable.
4. Run the same safe public pipeline on desktop, robot, and edge targets while
   heavy runtimes remain optional dedicated features.
5. Export runtime assets through glTF and composed digital-twin scenes through
   OpenUSD without making either format a dependency of `spatialrust-core`.

The successor Goal is active. Epic 91 establishes versioned records and Arrow
bridges; Epics 92–100 proceed in dependency order with per-Epic delivery slices.

## 3D Tiles 1.1 point-cloud tileset program (Epic 147)

The OGC 3D Tiles 1.1 standard is the de-facto streaming contract for massive
3D point data in web, desktop, and digital-twin runtimes. SpatialRust already
owns the COPC bounds/LOD substrate and a WebGPU/WASM viewer; exporting a
deterministic, bounded `tileset.json` + `pnts` tile set lets any 3D Tiles
consumer stream SpatialRust point clouds without a potree/PDAL rewrite. The
tileset stays dependency-light in `spatialrust-interchange` and never depends
on `spatialrust-core`, a GPU backend, or serde.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 147A | Complete | `pnts` binary codec: header, feature-table JSON, POSITION/RGB/RTC_CENTER, byte-aligned padding, round-trip decode | `tiles3d` |
| 147B | Complete | `tileset.json` model: box bounding volumes, geometric error, refine, content URIs; strict JSON parse/serialize | `tiles3d` |
| 147C | Complete | Deterministic octree tileset builder from interleaved positions with point budgets, per-tile RTC_CENTER, and write receipt | `tiles3d` |
| 147D | Complete | Facade `interchange-tiles3d`, runnable example, FEATURE_MATRIX/CHANGELOG/notes | facade |
| 147E | Complete | Bounded COPC → 3D Tiles exporter: `CopcNodeReader` per-node hierarchy walk in `spatialrust-io` plus `export_copc_tileset` in `spatialrust-interchange`, with LAS color preserved as 8-bit `pnts` RGB | `tiles3d-copc` |
| 147F | Complete | Python `export_tiles3d` / `export_copc_tiles3d` bindings with typed stubs and smoke tests | Python tiles3d surface |

## Point-cloud conformance program (Epic 148)

SpatialRust already proves image correctness and speed against OpenCV with
dated, honest receipts (Epics 101–111). Epic 148 does the same for point
clouds: reproducible PCL / PDAL comparisons on identical public clouds, with a
versioned workload manifest, environment receipts, and per-operation winner
reporting. It extends the existing `bench/pcl_comparison` and
`bench/open3d_comparison` harnesses and adds a PDAL runner; every published
number names the workload, machine, and library versions. PCL remains
comparison tooling only and never enters a production feature.

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 148A | Complete | Versioned point-cloud benchmark manifest (profiles, statistics, workloads) and stdlib-only report contract | `bench/pcl_comparison/manifest.json`, `report.py`, `test_report.py` |
| 148B | Complete | PDAL runner with matching filters/operations on the identical cloud | `bench/pdal_comparison/` |
| 148C | Complete | Unify PCL/PDAL/Open3D comparison receipts and aggregate runner with fail-closed checks | `bench/pcl_comparison/aggregate.py` + tests |
| 148D | Complete | Dated honest comparison receipt, docs, and README updates | `receipt-2026-08-07.json`, note |

## Arrow canonical interchange program (Epic 149)

SpatialRust already owns the audited Arrow C Data / Stream / Device substrate
(91C/91D). Epic 149 promotes Arrow to the canonical cross-language zero-copy
interchange: Python objects expose `__arrow_c_array__` / `__arrow_c_stream__`
so PyArrow, pandas, and DuckDB consume SpatialRust records without copying, and
Rust round-trips stay audited and explicit.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 149A | Complete | Python `__arrow_c_array__` on `PyPointCloud` via the existing C Data export, with PyArrow round-trip test | `arrow-c-data` |
| 149B | Complete | `__arrow_c_stream__` on `PyPointCloudStream` for bounded record streaming | `arrow-c-stream` |
| 149C | Complete | Facade docs, FEATURE_MATRIX/CHANGELOG/notes | docs |

## Real AI semantic meaning program (Epic 150)

Epic 95 established semantic entities and embeddings with a deterministic mock
profile. Epic 150 connects real ONNX inference to spatial semantic entities:
point-cloud entity features are embedded through a model session, indexed in
the existing `SemanticSearchIndex`, and searchable by open-vocabulary label.
Heavy runtimes stay behind dedicated features; tests use a tiny deterministic
ONNX model committed as fixture bytes so no external download is required.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 150A | Complete | `OnnxEntityEmbedder`: entity feature tensors → ONNX embedding via an existing `ModelSession`, with explicit copy policy | `semantic`, `ai-onnxruntime` |
| 150B | Complete | Fixture ONNX model bytes + embedder correctness tests against the mock/real boundary | fixture, tests |
| 150C | Complete | Facade wiring, Python binding, FEATURE_MATRIX/CHANGELOG/notes | facade |

The embedder never loads a model itself; it consumes an already-open session so
backend identity, device placement, and copy policy stay explicit. The fixture
model maps a fixed-dimension feature vector to a fixed-dimension embedding so
round trips are exact and no learned weights are implied. The facade
`semantic-model` feature wires the embedder with the existing AI/ONNX surface.



Each slice lands as one reviewable PR. The manifest reserves VGA-class and
full-size cloud profiles and at least the operations both libraries implement
(voxel, normals, SOR, radius outlier removal, and PDAL-oriented IO/reprojection).


The builder splits octants in a fixed bit order and writes one `pnts` payload
per BFS tile id; leaf geometric error is zero and internal errors halve each
level. Public APIs take plain `&[f32]`/`&[u8]` slices so the codec stays
independent of `spatialrust-core`. Data is always owned host memory and no
hidden host/device copy is introduced. The COPC exporter (147E) walks a COPC
file's octree hierarchy one node at a time, re-centers each node to its own
`RTC_CENTER`, and never materializes the whole cloud.

## Program invariants

- `spatialrust-core` remains independent of image codecs and AI runtimes.
- Codec, ONNX, CUDA, TensorRT, DirectML, and similar dependencies are opt-in
  features in dedicated crates.
- CPU/GPU transfers are named, explicit operations. Production APIs do not
  silently migrate data or read GPU results back to the host.
- Public APIs are safe. `unsafe` is restricted to audited FFI and GPU boundaries.
- Data models and capability contracts land before broad algorithm families.

## Completion gates for every Epic

1. Correctness tests for supported dtypes, strided ROI input, degenerate sizes,
   and invalid input.
2. Property or fuzz tests for parsers and correctness-critical transforms.
3. Numerical comparison with an authoritative implementation such as OpenCV,
   DLPack consumers, or ONNX Runtime reference output.
4. CPU/GPU benchmarks at representative 640p, 1080p, and 4K sizes where the
   operation is performance-sensitive.
5. Each feature builds alone with default features disabled; the workspace
   default does not acquire optional heavy runtimes.
6. Python bindings and type stubs for user-facing workflows.
7. Rustdoc, architecture, API-stability, changelog, and reproducibility notes.

## Epic 83 acceptance criteria

- Decode PNG, JPEG, and PNM from paths, arbitrary readers, and memory bytes.
- Keep TIFF and OpenEXR behind independent features.
- Enforce compressed-input, width, height, decoded-pixel, and allocation limits.
- Preserve source format, sample/color type, and Exif orientation metadata;
  optionally apply orientation to decoded pixels.
- Encode supported owned image variants to paths, seekable writers, and bytes.
- Test exact lossless round trips, bounded failure, orientation transforms,
  malformed input, feature-alone builds, and rustdoc.

## Epic 84 delivery slices

Epic 84 extends `spatialrust-vision` without introducing a second image owner or
an implicit CPU/GPU runtime. Work lands in dependency order:

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 84A | Complete | Shared border sampling, validated kernels, correlation/filter2D, separable filters, box and Gaussian blur | `imgproc-filter` |
| 84B | Complete | Median and bilateral filters, Sobel/Scharr/Laplacian, Gaussian pyramids | `imgproc-filter` |
| 84C | Complete | Structuring elements, erode/dilate, open/close/gradient/top-hat/black-hat | `imgproc-morphology` |
| 84D | Complete | Fixed/adaptive/Otsu thresholds, histograms, equalization/CLAHE, integral images | `imgproc-analysis` |
| 84E | Complete | Non-maximum suppression and hysteresis-based Canny edge detection | `imgproc-canny` |

The shared filter contract follows the established image-processing convention
that filter2D performs correlation unless callers explicitly reverse a kernel.
Multi-channel inputs are processed independently and every neighborhood API
requires an explicit border mode. Existing `warp::BorderMode` remains source
compatible while its sampling contract moves to a shared module.

Epic 84 is complete when every slice supports strided ROI input, rejects empty
or invalid kernels deterministically, has property tests for degenerate images,
matches documented OpenCV behavior within per-operation tolerances, and ships
feature-alone builds, Python bindings/stubs, rustdoc, and 640p/1080p/4K
benchmarks for the performance-sensitive kernels.

## Epic 85 delivery slices

Epic 85 introduces a runtime-independent tensor crate. It does not rename or
replace `spatialrust-core::SpatialTensor`, which remains the chunked point-cloud
view used by existing algorithms.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 85A | Complete | Dtype, shape, signed element strides, byte offset, device, owned/borrowed CPU storage | `tensor` |
| 85B | Complete | Zero-copy packed/planar image and point-field bridges plus explicit packing copies | `tensor-image`, `tensor-spatial` |
| 85C | Complete | Audited DLPack major-version 1 managed-tensor import/export with minor-version checks | `tensor-dlpack` |
| 85D | Complete | Python `__dlpack__`, `__dlpack_device__`, NumPy/PyTorch interoperability | Python tensor bindings |

Host byte slices are only exposed for host-accessible devices. Backend device
copies remain named operations owned by backend crates. DLPack exchange uses
the versioned managed-tensor ABI and makes ownership/deleter transfer explicit.

## Epic 86 delivery slices

Epic 86 isolates inference runtimes from tensor metadata and the workspace
default build. Copy permission and device placement are part of each run or
binding request rather than backend side effects.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 86A | Complete | Runtime-independent backend/session, model metadata, named dynamic I/O, copy policy, and binding contracts | `ai` |
| 86B | Complete | ONNX Runtime CPU EP, session options, typed input/output conversion, dynamic model metadata | `ai-onnxruntime` |
| 86C | Complete | Zero-copy typed CPU inputs, runtime-retained outputs, caller-preallocated outputs, and output-to-input chaining | `ai-onnxruntime` |
| 86D | Complete | Python session API, Python ONNX Runtime numerical comparison, stubs, and 640p/1080p/4K binding benchmark | Python `onnxruntime` feature |

CUDA, TensorRT, and DirectML remain separately compiled provider features; no
provider is selected implicitly. The current optional `ort` 2.0.0-rc.12
adapter requires Rust 1.88, while default and runtime-independent `ai` builds
retain the workspace MSRV because they do not resolve or compile `ort`.

Raw byte allocations for multi-byte elements are never cast into backend tensor
pointers. Callers use typed constructors or authorize an explicit copy. Bound
ONNX Runtime CPU outputs retain their runtime allocation behind
`HostTensorStorage`, allowing the output to become another bound input without
an intermediate host allocation.

## Epic 87 delivery slices

Feature2D keeps keypoint metadata and descriptor representation independent of
any detector. Binary and float descriptor matrices carry their distance
semantics explicitly, and matching never changes device placement.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 87A | Complete | `Keypoint2`, checked binary/float `DescriptorBuffer`, paired `FeatureSet2`, bounded `FeatureMatch` | `vision-feature2d` |
| 87B | Complete | Harris, Shi–Tomasi, and exact FAST-9/16 coordinates, scores, and non-maximum suppression | `vision-feature2d` |
| 87C | Complete | Multi-scale oriented FAST plus stable 256-bit rotated BRIEF, brute-force Hamming/L2 matching, ratio/cross-check filters | `vision-feature2d` |
| 87D | Complete | Python NumPy workflow, OpenCV comparison, property tests, and 640p/1080p/4K Criterion coverage | Python vision bindings |

SpatialRust ORB uses a documented fixed-seed BRIEF table, not OpenCV's private
learned table, so descriptor bits are stable within SpatialRust but do not claim
OpenCV bit identity. Detector repeatability and BFMatcher distance compatibility
are measured separately. The initial scalar CPU implementation establishes the
contract and correctness baseline; Epic 89 may accelerate it without changing
host/device transfer semantics.

## Epic 88 delivery slices

Geometry stays independent of `feature2d` and `dense`. Multiview models and
absolute pose share one robust-estimation contract. Stereo remaps are returned as
caller-owned maps for explicit `warp::remap`; disparity reprojects to packed
`Image` buffers rather than dense-map wrappers.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 88A | Complete | `PointCorrespondence2`, `CameraMatrix3`, projective models, robust options, pose/triangulation result types | `vision-geometry` |
| 88B | Complete | Normalized DLT + deterministic RANSAC for H/F/E; triangulation; essential pose disambiguation | `vision-geometry` |
| 88C | Complete | EPnP-class PnP with iterative refine and RANSAC; sparse pyramidal Lucas–Kanade tracks | `vision-geometry` |
| 88D | Complete | Stereo rig, rectify maps, block-matching disparity, depth/XYZ reproject; Python; OpenCV comparison; Criterion | Python + `vision-geometry` |

Essential/pose and StereoBM comparisons document residual and disparity tolerances
rather than claiming bit-identical OpenCV matrices. Scalar CPU is the correctness
baseline; Epic 89 may accelerate kernels without changing host/device semantics.

## Epic 89 delivery slices

GPU images live in `spatialrust-gpu` behind `gpu-image`. CPU `spatialrust-vision`
remains the numerical baseline. Kernel APIs take and return `GpuImage` and never
imply host transfers; only named upload/readback move bytes across the host/device
boundary. Epic 89 initially used one `u32` per component for WGSL clarity;
Epic 104 supersedes that internal representation with pooled `rgba8uint`
textures without changing the explicit transfer contract.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 89A | Complete | `GpuImage` ownership, packed/`ImageView` upload with named stride packing, explicit readback, receipt bytes, recycle, cross-runtime rejection | `gpu-image` |
| 89B | Complete | Device-resident `copy_gpu_image` chain with mid-chain `device_to_host_bytes == 0` | `gpu-image` |
| 89C | Complete | `rgb_to_gray_gpu` (BT.601 fixed-point) and gray `box_blur_gpu` with clamp/replicate borders | `gpu-image` |
| 89D | Complete | Facade `gpu-image` flag, headless CPU comparison tests, Criterion upload/chain bench, CHANGELOG and notes | `gpu-image` |

Epic 89 completes when `upload → gray → box_blur → readback` records a single H2D
and a single D2H, feature-alone builds succeed without `gpu-aoso-staging`, and
CPU reference residuals stay within documented tolerances.

## Epic 90 delivery slices

Model adapters stay runtime-light: `spatialrust-ai` owns mock/ONNX backends,
`spatialrust-vision` owns image↔tensor and tensor→dense-map helpers, and the
facade wires a documented image → infer → unproject smoke path. Default builds
do not pull ONNX; `MockInferenceBackend` is always available with `ai`.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 90A | Complete | `MockProfile` / `MockInferenceBackend` and `ModelSource::Mock` for deterministic depth without weights | `ai` (default-safe) |
| 90B | Complete | Letterbox + NCHW prep (`rgb_u8_to_nchw_f32`, `planar_f32_to_nchw`) | `vision-ai-adapters` |
| 90C | Complete | Tensor → `DepthMap` / `BinaryMask` / `Detection` decode helpers | `vision-ai-adapters` |
| 90D | Complete | Facade `ai-vision-pipeline` E2E (mock depth → XYZ → MVP), ROADMAP/CHANGELOG/notes | `ai-vision-pipeline` |

Epic 90 completes when an RGB image can be letterboxed into contiguous NCHW,
run through mock inference with explicit output-copy permission, decoded to a
`DepthMap`, and unprojected to a point cloud that feeds MVP without enabling
`onnxruntime`.

## Epic 91 delivery slices

Versioned spatial records stay in `spatialrust-records` (Arrow-free). Arrow C
Data/Stream/Device live in `spatialrust-arrow` behind independent features.
`spatialrust-core` remains free of Arrow FFI and schema evolution APIs.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 91A | Complete | `SchemaId`/`SchemaVersion`/`SchemaDescriptor`, compatibility reports, `SpatialRecord`, and validated `RecordProvenance` lineage | `records` |
| 91B | Complete | `SpatialRecordSource`/`Sink`, `MemoryChunkSource`/`Sink`, migrate with fill/drop policy | `records` |
| 91C | Complete | Arrow C Data export/import for `PointCloud` struct columns | `arrow-c-data` |
| 91D | Complete | Arrow C Stream over record sources; CPU Arrow C Device array export/import | `arrow-c-stream`, `arrow-c-device` |

Epic 91 completes when a point cloud can be split into versioned records, migrated
across compatible schema minors, and round-tripped through Arrow C Data without
pulling Arrow into `spatialrust-core`.

## Epic 92–100 delivery slices (activated)

Epics 92–100 have concrete crates and facade flags. Heavy native toolchains
(`rclrs` / libusd / Hydra GPU path) remain install-time optional. Portable
deepenings ship without those SDKs: ROS 2 CDR PointCloud2 + loopback
(`runtime-ros2`), USDA ASCII mesh interchange (`interchange-openusd`), CPU
Gaussian soft-splat rendering (`scene-gaussian`), plus file MCAP XYZ codecs and
TSDF marching tetrahedra.

| Epic | Status | Delivered substrate |
| --- | --- | --- |
| 92 | Complete | `spatialrust-sync` clocks, frame graph, MemoryEpisode replay |
| 93 | Complete | `spatialrust-mapping` trajectories, pose graph, synthetic odometry |
| 94 | Complete | `spatialrust-scene` TSDF/surfel/mesh + Gaussian CPU soft-splat (`gaussian`) |
| 95 | Complete | `spatialrust-semantic` embeddings, fusion, search |
| 96 | Complete | `spatialrust-episode` episode/annotation/augment/eval/provenance |
| 97 | Complete | `spatialrust-runtime` bounded pipeline/trace/diagnostics + ROS 2 CDR/loopback (`ros2`) |
| 98 | Complete | `spatialrust-interchange` glTF JSON + USDA ASCII OpenUSD adapter |
| 99 | Complete | `spatialrust-distribute` partitions, topo order, backpressure queues, named transfers |
| 100 | Complete | `spatialrust-platform` stability/conformance/security/LTS + release gate/perf budgets |

## Epic 92 delivery slices

Sensor time and frame graphs live in `spatialrust-sync`. Default builds use
in-memory episodes; enable `mcap` / facade `sync-mcap` for XYZ stamped-record
file codecs via the Foxglove `mcap` crate (no compression codecs by default).

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 92A | Complete | `ClockId`/`ClockDomain`, `SyncQuality`, `StampedTime` | `sync` |
| 92B | Complete | `FrameGraph` / `FrameEdge` with inverse-aware lookup | `sync` |
| 92C | Complete | Topic channels + `MemoryEpisode` index; file MCAP XYZ round-trip | `sync`, `sync-mcap` |
| 92D | Complete | `DeterministicReplayer` with nearest-topic bundling | `sync` |
| 92E | Complete | Bounded `MemoryEpisodeBuilder` with record/point/allocated-byte admission | `sync` |
| 92F | Complete | `SpatialRecord` frame transformation preserving columns, metadata, and provenance; bounded rosbag2 sync preview | `sync`, `rosbag2-sqlite` |

Epic 92 completes when stamped multimodal records can be indexed deterministically,
collected under explicit episode limits, bundled within a sync window, and
transformed across a calibrated frame graph without dropping record lineage.
The rosbag2 preview treats PointCloud2 header stamps as one external ROS time
domain and reports that assumption explicitly; it does not claim clock
calibration. Optional `sync-mcap` write/read path covers XYZ-only stamped clouds
today.

## Epic 93 delivery slices

Localization contracts live in `spatialrust-mapping`. Full visual/lidar odometry
pipelines grow behind later algorithmic slices; Epic 93 lands the pose trajectory
and pose-graph substrate first.

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 93A | Complete | `Trajectory` / `StampedPose` with interpolation | `mapping` |
| 93B | Complete | `RelativeMotionEstimator` + `SyntheticOdometry` | `mapping` |
| 93C | Complete | `PoseGraph` relative edges and root localization | `mapping` |
| 93D | Complete | Loop-closure candidate search by translation distance | `mapping` |
| 93E | Complete | Bounded topic-prefix scan odometry with generic matcher and optional ICP adapter | `mapping`, `mapping-scan-icp` |

Epic 93 completes when stamped poses can be buffered, differenced into relative
motion, and localized on a pose graph with loop-closure candidates without
pulling ROS 2 or MCAP file codecs.

## Epic 94–100 delivery slices

| Epic | Feature flags | Substrate crates |
| --- | --- | --- |
| 94 | `scene`, `scene-gaussian` | `spatialrust-scene` |
| 95 | `semantic` | `spatialrust-semantic` |
| 96 | `episode` | `spatialrust-episode` |
| 97 | `runtime`, `runtime-ros2` | `spatialrust-runtime` |
| 98 | `interchange-gltf`, `interchange-openusd` | `spatialrust-interchange` |
| 99 | `distribute` | `spatialrust-distribute` |
| 100 | `platform` | `spatialrust-platform` |

## Epic 94 delivery slices

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 94A | Complete | Direct `PointCloud` column integration into TSDF with explicit sensor origin and no temporary interleave | `scene` |
| 94B | Complete | Explicit sensor-to-volume pose integration without transformed-cloud allocation | `scene` |

## Epic 95 delivery slices

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 95A | Complete | `SpatialRecordEntity` centroid/label/embedding adapter with copied provenance and frame/timestamp metadata | `semantic` |

## Epic 99 delivery slices

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 99A | Complete | `PartitionGraph` / `ExecutionPartition` with deterministic topo order | `distribute` |
| 99B | Complete | `BackpressurePolicy` + `BoundedTransferQueue` admissions | `distribute` |
| 99C | Complete | `NamedTransfer` / `TransferPlan` / `TransferLedger` (measurable copies) | `distribute` |

## Epic 100 delivery slices

| Slice | Status | Scope | Feature |
| --- | --- | --- | --- |
| 100A | Complete | `StabilityRegistry` + north-star surface seed | `platform` |
| 100B | Complete | `ConformanceReport` statuses/counts/summary | `platform` |
| 100C | Complete | `SecurityChecklist` baseline + mark helpers | `platform` |
| 100D | Complete | `LtsPolicy` / `SupportWindow` for 1.x (18+6 months) | `platform` |
| 100E | Complete | `PerformanceBudgetReport` + `ReleaseGate` aggregation | `platform` |

Facade convenience flag `north-star` enables the Epic 91–100 substrate stack
without ONNX/ROS2 native executors. Portable OpenUSD ASCII, CPU Gaussian
rendering, and ROS 2 CDR codecs are available behind `interchange-openusd`,
`scene-gaussian`, and `runtime-ros2`. Linking `rclrs` / libusd remains deferred
to install-time toolchains.

The integration feature `north-star-e2e` (`north-star` + `ai-vision-pipeline` +
`sync-mcap` + `runtime-ros2`) runs
`crates/spatialrust/tests/north_star_pipeline.rs` and example `north_star_demo`:
RGB → mock depth → episode → MCAP XYZ round-trip → ROS 2 CDR loopback →
TSDF/mesh → glTF JSON + USDA ASCII → Gaussian CPU soft-splat →
`ReleaseGate` (stability/conformance/security/LTS/perf budgets).

## OpenCV-outcome program (Epics 101–111)

This program does not attempt API-count parity with OpenCV. It targets the
Rust-native spatial workloads where typed ownership, explicit transfers, and a
single image-to-world dataflow can provide a measurable advantage. Each Epic
uses the standard completion gates above and lands as one reviewable PR.

| Epic | Status | Depends on | Outcome |
| --- | --- | --- | --- |
| 101 | Complete | 83–90 | Reproducible OpenCV correctness/performance contract, workload manifest, environment receipts, and aggregate runner |
| 102 | Complete | 101 | Stabilize the image/camera/vision 1.0 contract and cross-platform conformance |
| 103 | Complete | 101–102 | SIMD/parallel CPU kernel dispatch, reusable outputs, and measured allocation control |
| 104 | Complete | 89, 101–103 | Texture-backed GPU Image v2 and device-resident resize/filter/edge/morphology chains |
| 105 | Complete | 88, 101–102 | Mono/stereo/fisheye/hand-eye calibration and bundle-adjustment contracts |
| 106 | Complete | 92, 101–105 | Dense flow, tracking, background modeling, and feature-gated video stream adapters |
| 107 | Complete | 93, 101–106 | Stronger local features, robust tracking, and visual/RGB-D odometry integration |
| 108 | Complete | 101–107 | Feature-gated computational photography and panorama stitching |
| 109 | Complete | 97, 99, 101–108 | Bounded spatial execution graph with fusion, backpressure, and named transfer receipts |
| 110 | Complete | 100, 101–109 | SpatialRust Vision 1.0 conformance, audits, performance budgets, examples, and migration policy |
| 111 | Complete | 101, 103, 110 | Bias-resistant OpenCV speed methodology, robust timing dispersion, and workload-specific accuracy metrics |

### Epic 101 acceptance criteria

- One versioned JSON envelope distinguishes correctness, performance, and
  aggregate reports.
- Performance results retain raw samples, median, p95, min/max, warmup/repeat
  policy, input dimensions, implementation, and allocation/reuse mode.
- Every report identifies OS/platform, architecture, CPU count, Python,
  OpenCV, and SpatialRust versions; published GPU results additionally record
  adapter/backend data in the suite-specific result.
- The canonical manifest reserves VGA, 1080p, and 4K profiles and at least ten
  image, geometry, RGB-D, AI-adapter, and spatial end-to-end workloads.
- Existing vision correctness and RGB-D performance gates emit the contract;
  an aggregate command runs either or both suites and validates their reports.
- Contract tests require only the Python standard library and run in CI. OpenCV
  remains comparison tooling and never enters a production feature.

Epics 103–104 may add internal dispatch and fusion, but public CPU APIs never
perform implicit device transfers. Epics 105–109 keep codecs, ONNX, ROS 2,
CUDA, and external video runtimes in dedicated additive features.

### Epic 102 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 102A | Complete | Machine-readable stable/provisional image, camera, and vision surface | `StabilityRegistry::vision_v1_surface()` |
| 102B | Complete | Stable ownership, stride, camera, filter, detection entry-point contract | `vision_api_v1` integration test |
| 102C | Complete | Feature-complete image/camera/vision tests on Linux, Windows, and macOS | `vision-platform-conformance` CI matrix |
| 102D | Complete | API stability policy, ROADMAP, CHANGELOG, and reproducibility note | release documentation |

Epic 102 freezes data ownership and the common algorithm entry surface, not
every algorithm implementation. Geometry, stereo, optical flow, AI adapters,
and `GpuImage` remain explicitly provisional. Stable entries may gain faster
internal dispatch without changing ownership, error, stride, or transfer
semantics. Completion requires the dedicated three-OS CI matrix and the full
`spatialrust-vision/full` property suite to pass.

### Epic 103 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 103A | Complete | Allocation audit and caller-owned image/planar outputs | `resize_into`, `rgb_to_gray_into`, `normalize_into`, `pack_chw_into` |
| 103B | Complete | Safe size-aware parallel dispatch for planar AI packing | scoped channel workers with scalar small-image fallback |
| 103C | Complete | Rust and NumPy reusable-output contracts | strided Rust tests and Python `out=` identity tests |
| 103D | Complete | VGA/1080p/4K correctness and allocation/reuse measurements against OpenCV | `opencv-vision-performance` report |

Epic 103 does not claim blanket CPU kernel superiority. The comparison receipt
records OpenCV's SIMD advantage for resize and RGB-to-gray, while SpatialRust's
typed RGB-to-CHW path is faster on every canonical profile. On the reference
Windows host, reusable SpatialRust CHW measured 8.54x, 13.07x, and 16.11x faster
than allocating `cv2.dnn.blobFromImage` at VGA, 1080p, and 4K. Public CPU APIs
accept explicit caller storage and never perform an implicit device transfer.

### Epic 104 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 104A | Complete | Replace component-expanded storage buffers with pooled `rgba8uint` 2D textures | four physical bytes/pixel and explicit texture upload/readback |
| 104B | Complete | Texture-resident copy, RGB-to-gray, and box-filter migration | existing CPU parity tests with zero mid-chain D2H |
| 104C | Complete | Chainable nearest resize, Sobel magnitude, erosion, and dilation | known-pixel GPU tests and transfer-stage receipt |
| 104D | Complete | Runtime adapter/backend identity, explicit synchronization, per-device pipeline caches, and steady-state texture pool | `WgpuAdapterInfo`, `wait_idle`, recycle/acquire pool |
| 104E | Complete | VGA/1080p/4K synchronized Criterion coverage | upload, gray+blur, and five-stage resident chain groups |

The reference low-power adapter measured the five-stage resident chain at
0.963 ms (VGA), 3.504 ms (1080p), and 13.523 ms (4K), with explicit device
synchronization. A chain receipt contains one upload, named device stages, no
mid-chain readback, and one readback only when the caller requests host data.

### Epic 105 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 105A | Complete | Shared robust solver options and RMS/max/iteration receipts | `CalibrationOptions`, `CalibrationReport` |
| 105B | Complete | Robust mono intrinsics and Kannala–Brandt4 fisheye fitting | synthetic outlier and angle-polynomial recovery tests |
| 105C | Complete | Stereo and hand-eye transforms with supplied-rotation translation solves | 3D alignment and `AX = XB` residual tests |
| 105D | Complete | Sparse fixed-camera point bundle adjustment | multi-view numerical-Jacobian convergence test |
| 105E | Complete | Calibration workload coverage and provisional API registration | 100/1000 observation and 100-point/3-view Criterion groups |

Calibration solvers live in `spatialrust-camera`, use small deterministic dense
normal equations, and introduce no native optimizer dependency. Supplied
rotations are checked for finite, right-handed orthonormal form. The first BA
contract intentionally fixes calibrated camera poses and refines world points;
joint pose/intrinsics optimization remains additive and provisional.

### Epic 106 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 106A | Complete | Dense deterministic integer block flow with invalid-border semantics | translated-texture test and OpenCV Farneback comparison |
| 106B | Complete | Adaptive single-Gaussian foreground modeling | new-object mask/ratio sequence test |
| 106C | Complete | Same-class IoU track lifecycle with monotonic IDs | confirmation, association, miss, and expiry test |
| 106D | Complete | Pull-based timestamped stream adapter contract | `VideoFrameSource` and deterministic `MemoryVideoSource` |
| 106E | Complete | Dedicated features, Python flow binding, and benchmark coverage | `vision-video[-adapters]`, QQVGA/QVGA Criterion |

Core video algorithms stay runtime-free in `spatialrust-vision/video`. Codec,
camera, and network integrations implement `VideoFrameSource` behind additive
features; frames carry sequence/time explicitly and remain owned host images.
No adapter may hide a GPU transfer.

### Epic 107 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 107A | Complete | Deterministic strongest-per-cell local-feature distribution | grid ordering and response test; 4096-keypoint Criterion |
| 107B | Complete | Bidirectional pyramidal-LK consistency filtering | per-track forward/backward errors and explicit threshold |
| 107C | Complete | Scale-ambiguous calibrated monocular odometry | essential RANSAC, cheirality recovery, explicit caller scale |
| 107D | Complete | Metric RGB-D odometry from source depth and pixel tracks | depth filtering, PnP RANSAC, synthetic metric translation test |
| 107E | Complete | Mapping/Python integration and OpenCV receipt | `mapping-vision-odometry`, Python binding, `solvePnPRansac` parity |

The vision layer reports source-to-target motion and never invents monocular
scale. `spatialrust-mapping` accepts scale explicitly for monocular estimates
and preserves metric RGB-D translation. Invalid source depths are counted,
not silently filled or copied to another device.

### Epic 108 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 108A | Complete | Deterministic RGB gray-world white balance | channel-mean equality test and Python binding |
| 108B | Complete | Aligned well-exposedness fusion | middle-gray preference test and VGA/3-exposure Criterion |
| 108C | Complete | Bounded pairwise panorama canvas and origin receipt | translated pair geometry and pixel-budget rejection |
| 108D | Complete | Bilinear source warp and edge-distance feather blending | overlap/non-overlap known-pixel tests |
| 108E | Complete | Homography estimation composition and OpenCV comparison | RANSAC entry point and zero-error `warpPerspective` receipt |

Photography remains a runtime-free `vision-photography` feature. Inputs must
share dimensions/metadata where alignment requires it, panorama allocations are
checked against a caller-visible pixel ceiling, and no codec or GPU transfer is
performed implicitly.

### Epic 109 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 109A | Complete | Typed stateful operators and deterministic DAG compilation | duplicate/missing endpoint and cycle rejection tests |
| 109B | Complete | Same-device linear fusion schedule | `decode + gray` fused while GPU inference stays separate |
| 109C | Complete | Soft/hard bounded source admission | watermark counters and hard-rejection test |
| 109D | Complete | Mandatory named cross-device edges | missing-transfer compile rejection |
| 109E | Complete | Per-run stage/fusion/transfer receipts and workload | 1024-byte upload ledger and eight-stage Criterion |

`spatialrust-runtime/execution-graph` reuses placement, watermark, and transfer
contracts from `spatialrust-distribute`. Fusion never crosses a placement
boundary or an explicitly named transfer. Values remain owned, and the runtime
does not infer or execute a host/device copy on behalf of an operator.

### Epic 110 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 110A | Complete | Mandatory cross-platform Rust, property, Python, OpenCV, GPU-transfer, and unsafe cases | `Vision1ReleaseGate::required_conformance_cases()` |
| 110B | Complete | Fixed VGA/1080p CPU, 4K GPU, and explicit-copy ceilings | typed microsecond/byte measurements and denial tests |
| 110C | Complete | Required OpenCV receipts from Epics 101 and 105–108 | seven suite identifiers checked for presence |
| 110D | Complete | Runnable CPU workflow and release-receipt examples | `vision_1_cpu`, `vision_1_release_gate`, three-OS CI |
| 110E | Complete | OpenCV-to-SpatialRust migration and stability policy | `docs/VISION_1_MIGRATION.md` and `vision-1` acknowledgement |

Vision 1.0 freezes the stable foundation listed in `API_STABILITY.md`; additive
geometry, odometry, photography, video, GPU, and runtime surfaces remain
provisional behind their named features. The release gate denies missing or
skipped mandatory evidence, absent performance samples, over-budget samples,
unsatisfied security audits, missing examples/comparison receipts, and an
unacknowledged migration policy.

### Epic 111 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 111A | Complete | Seeded interleaved pairs and adaptive batching for short calls | `timed_pair` contract tests |
| 111B | Complete | Mean/median/p95, standard deviation, CV, MAD, raw samples, and throughput | v1 additive timing fields |
| 111C | Complete | Scale-aware numerical and binary edge accuracy | MAE/RMSE/relative-L2/PSNR and F1/IoU |
| 111D | Complete | Resize, gray, CHW, Gaussian, Sobel, morphology, and Canny at VGA/1080p/4K | dated Epic 111 receipt |
| 111E | Complete | Strict finite JSON and honest per-workload winner reporting | report contract tests and comparison docs |

Epic 111 does not claim blanket superiority. On the dated Windows reference
host, SpatialRust leads AI CHW preprocessing while OpenCV leads the measured
general image kernels. Results are hardware receipts, not portable guarantees.

## Vision 2 performance program (Epics 112–120)

This program turns the Epic 111 evidence into a systematic optimization track.
It covers native CPU kernels, Python end-to-end calls, explicit GPU-resident
chains, allocation behavior, and release gates. It does not promise blanket
OpenCV superiority: every published result names the workload, hardware,
backend, allocation mode, and accuracy contract.

| Epic | Status | Depends on | Outcome |
| --- | --- | --- | --- |
| 112 | Complete | 111 | Attribute native kernel, allocation, Python conversion, and transfer costs with reproducible throughput and memory receipts |
| 113 | Complete | 112 | Caller-owned outputs and reusable workspaces for multi-stage CPU vision without hidden copies |
| 114 | Complete | 112–113 | Safe size-aware CPU dispatch for packed fast paths, strided fallbacks, and bounded row/tile parallelism |
| 115 | Complete | 113–114 | Accelerated resize and color conversion with precomputed sampling plans and fused preprocessing experiments |
| 116 | Complete | 113–115 | Accelerated separable Gaussian and Sobel engine with cached kernels and shared gradient passes |
| 117 | Complete | 113–116 | Sliding-window morphology engine with exact OpenCV comparison and generic-mask fallback |
| 118 | Complete | 113–117 | Fused Canny fast path that avoids public intermediates unless explicitly requested |
| 119 | Complete | 104, 115–118 | Explicit upload-once GPU-resident vision chain with no intermediate readback |
| 120 | Complete | 112–119 | Vision 2 cross-platform correctness, speed, memory, allocation, and transfer release gate |

Each Epic lands as one reviewable PR using implement → test → commit → PR →
merge. Stable Vision 1 ownership and error contracts remain compatible;
reusable workspace surfaces are additive. CPU APIs do not choose a GPU or copy
to one implicitly, and GPU receipts must retain named upload/readback stages.

### Epic 112 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 112A | Complete | Separate Python conversion, allocation, native kernel, upload, execution, and readback time | versioned component timing receipt; CPU-only transfer stages are explicit N/A |
| 112B | Complete | Report MPix/s, ns/pixel, bytes allocated, peak workspace, and batch policy | strict finite JSON contract tests |
| 112C | Complete | Add native Criterion counterparts for every OpenCV vision workload | eight-workload VGA/1080p/4K matched manifest |
| 112D | Complete | Record single-thread and default-thread CPU modes | six host and thread-policy receipts |
| 112E | Complete | Publish bottleneck attribution without changing kernels | `notes/2026-07-16_vision2_baseline.md` and Pages update |

### Epic 113 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 113A | Complete | `*_into` entry points for Gaussian, Sobel, morphology, and Canny | packed/strided identity and padding tests |
| 113B | Complete | Explicit reusable scratch storage for multi-pass algorithms | Gaussian/morphology/Canny steady-state capacity and allocation receipts |
| 113C | Complete | Validate dimensions, metadata, overlap, and channel contracts | negative, strided, metadata, and property tests |
| 113D | Complete | Reuse outputs through Python `out=` where supported | Gaussian/Sobel/morphology/Canny object-identity and numerical tests |
| 113E | Complete | Exact EDT caller-owned output and explicit reusable scratch | Rust/Python identity, capacity, and brute-force tests |

### Epic 114 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 114A | Complete | Shared small-image scalar and large-image row/tile dispatch policy | exact 100,000/262,144/1,000,000 threshold tests |
| 114B | Complete | Packed `u8` one/three-channel and `f32` internal fast-path selection | shared packed selector, dispatch receipt, and fallback parity |
| 114C | Complete | Preserve generic components, channels, strides, and borders as safe fallbacks | full 139-test unit and 13-test property suites |
| 114D | Complete | Bound worker creation and temporary memory | deterministic worker bounds and scratch ownership receipt |
| 114E | Complete | Exact EDT binary-row fast path, tiled transpose, and bounded pool dispatch | VGA/1080p/4K Criterion and OpenCV receipt |
| 114F | Complete | Cache EDT parabola heights and balance column tasks for dense masks | exact OpenCV parity and 4K Python reuse win |
| 114G | Complete | Cache NMS box geometry and avoid packed Python score copies | exact OpenCV index parity and 100/1,000/8,400-candidate wins |
| 114H | Complete | Bucket class-aware NMS keeps and expose one-call Python batched NMS | exact OpenCV parity and 26.38×/97.25× wins |
| 114I | Complete | One-pass active-set Soft-NMS selection, disjoint IoU exit, and borrowed Python scores | exact indices, bounded scores, and 3.42×–7.40× wins |
| 114J | Complete | Run-length union-find connected components with borrowed/non-zero Python masks | exact SAUF labels/stats and 2.17×–3.61× structured-mask wins |

### Epic 115 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 115A | Complete | Precompute resize source coordinates and interpolation coefficients | reusable Q11 bilinear plan, shape/stride/padding tests |
| 115B | Complete | Packed bilinear/nearest/area and RGB-to-gray fast paths | bit-exact planned nearest/area parity over 300 cases; VGA/1080p/4K RGB8 Criterion receipt |
| 115C | Complete | Evaluate resize+gray and resize+CHW fusion without changing standalone APIs | bit-exact unfused parity; resize+gray wins 1.12× at 1080p→540p and resize+CHW wins 2.02×–2.33× against OpenCV allocation |
| 115D | Complete | Improve current SpatialRust throughput by at least 5x on one canonical large profile | native reuse improved 47.8× at 1080p and 37.3× at 4K; VGA Python reuse is 1.10× faster than OpenCV |

### Epic 116 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 116A | Complete | Reuse separable-filter intermediates and cache validated Gaussian kernels | workspace capacity and output-reuse tests |
| 116B | Complete | Split border handling from contiguous interior loops | five Gaussian border modes plus strided-view tests |
| 116C | Complete | Specialized 3x3/5x5/7x7 Gaussian and paired Sobel X/Y passes | Q15 7×7 workspace path; 300-case OpenCV max error 2/255; paired Sobel exact |
| 116D | Complete | Improve Gaussian by at least 10x and Sobel by at least 5x on one canonical large profile | 5×5 Gaussian 20.7× at 1080p and 26.7× at 4K; dated native timing receipt |
| 116E | Complete | Remove standalone 3×3 Sobel's generic `f64` intermediate and fuse common absolute-threshold consumers | exact direct/absolute/mask APIs; direct Sobel beats OpenCV 1.88×/2.03× at 1080p/4K; fused masks win 2.95×–8.68× |
| 116F | Complete | Counter the remaining standalone Gaussian gap with band-local horizontal/vertical execution | exact canonical OpenCV parity; 1.70×–1.80× Python allocation improvement at 1080p/4K |

### Epic 117 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 117A | Complete | Separable sliding min/max for rectangular elements | generic-reference property tests and OpenCV receipt |
| 117B | Complete | Packed rectangular `u8` dispatch and generic Cross/Ellipse/Diamond/custom-mask fallback | shape, border, stride, iteration, and anchor parity |
| 117C | Complete | Caller-owned output and reusable ping-pong/worker scratch for iterations and composite operations | capacity, alias, stride, object-identity, and OpenCV reuse receipt |
| 117D | Complete | Improve morphology by at least 20x on one canonical large profile | 43.8× 4K 5×5 baseline improvement; bit-exact 511×511 OpenCV wins |
| 117E | Complete | Specialize centered 5×5 Replicate morphology without large-window transpose overhead | 6.6×–31.8× gap reduction; 1.22× OpenCV win at 1080p reuse; 980 exact randomized operations |

### Epic 118 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 118A | Complete | Compute paired gradients, magnitude, and direction with shared traversal | fast-path/intermediate bit-exact parity tests |
| 118B | Complete | Ring-buffer suppression and reusable hysteresis queue | parallel three-row magnitude rings, reusable state/stack, and allocated-byte receipt |
| 118C | Complete | Keep inspectable intermediates opt-in while making `canny()` allocation-light | `canny_into`, strided output padding, Python output identity, and workspace capacity tests |
| 118D | Complete | Improve Canny by at least 5x on one canonical large profile | 11.92× native 4K document-line improvement; bit-exact OpenCV parity; 1.42× OpenCV win |
| 118E | Complete | Counter dense sensor-noise hysteresis cost without regressing sparse documents | weak-candidate frontier; bit-exact parity; 2.59×/2.75× OpenCV reuse wins at 1080p/4K |

### Epic 119 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 119A | Complete | Chain resize, gray, blur, edge, morphology, and AI packing on explicit GPU images | seven named stages through resident CHW storage |
| 119B | Complete | One caller-requested upload, no intermediate readback, optional final readback | exact upload ledger and post-readback denial tests |
| 119C | Complete | Reuse textures and pipelines in steady state | four initialized pipelines and stable warmed texture/buffer pools |
| 119D | Complete | Compare CPU, GPU round-trip, and GPU-resident modes separately | synchronized VGA/1080p/4K Criterion receipt |

### Epic 120 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 120A | Complete | Cross-platform correctness and API compatibility matrix | Linux/Windows/macOS CI runs Vision 2 gate tests and receipt example |
| 120B | Complete | Native and Python allocate/reuse performance budgets | typed fail-closed 1080p RGB-to-gray measurements |
| 120C | Complete | Peak memory, allocation count, thread policy, and GPU transfer budgets | nine typed release measurements with overrun denial tests |
| 120D | Complete | Generated algorithm/performance documentation and migration guidance | Pages-generated receipt plus README/migration links |
| 120E | Complete | Vision 2 release gate and runnable receipt example | aggregate missing/skip/duplicate/budget denial tests |

### Vision 2 video pipeline E2E demo

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| Video E2E | Complete | Deterministic frame generation/loading → dense optical flow → object detection → native multi-object tracking in Rust and Python | 12 byte-identical PGM frames, 11 exact bidirectional flow checks, stable IDs 1/2, committed GIF |
| Real-data E2E | Complete | Public PCL scan → PCD/COPC bounds+LOD/MVP and Python clean/cluster/register/refine; remote Autzen COPC bounds query | 460,400 local points, 889,058-point non-empty HTTP ROI, shared connection pool measured 1.61× against one pre-change run |

The improvement thresholds above compare against the checked Epic 112
SpatialRust baseline on the same host; they are not claims against every OpenCV
build. Accuracy gates remain workload-specific: resize/gray/Gaussian retain
their documented bounded error, Sobel and morphology retain exact comparison,
and Canny retains binary precision, recall, F1, and IoU requirements.

## SpatialRust 1.2 bounded-streaming program (Epics 121–126)

SpatialRust 1.2 targets one user outcome: process point clouds larger than host
memory through a deterministic, bounded stream without changing the stable
in-memory `spatialrust-core` surface or hiding host/device transfers.
`SpatialTensor` remains a provisional borrowed view over one materialized
`PointCloud`; versioned stream contracts live in `spatialrust-records`, format
adapters stay feature-gated in `spatialrust-io`, and execution composition lives
outside core.

| Epic | Status | Depends on | Deliverable |
| --- | --- | --- | --- |
| 121 | Complete | 91, 100, 111 | Hard tracked-memory limits, cooperative cancellation, versioned execution receipts, and canonical scale/chunk workloads |
| 122 | Complete | 121 | Backward-compatible bounded record sources/sinks with chunk identity, deterministic ordering, prefetch, and buffer reuse |
| 123 | Complete | 122 | Local/HTTP COPC plus PCD/PLY/LAS/LAZ streaming adapters and bounded temporary spool contracts |
| 124 | Complete | 122–123 | Chunk-safe crop/transform/reductions and deterministic global voxel aggregation with explicit spill |
| 125 | Complete | 123–124 | Composable Rust pipeline, CLI, Python iterator, cancellation, and reproducible end-to-end receipt |
| 126 | Complete | 121–125 | Linux/Windows/macOS conformance, memory/copy budgets, documentation, migration notes, and the 1.2 release gate |

### Epic 121 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 121A | Complete | Positive chunk and hard memory-budget options plus deterministic ordering policy | invalid-configuration tests and stable defaults |
| 121B | Complete | Concurrent fail-closed memory reservations and cooperative cancellation | contention, overflow-denial, drop-release, and clone-visibility tests |
| 121C | Complete | Versioned strict JSON receipt for points, chunks, bytes, phases, spill, peak tracked memory, and named transfers | round-trip, unknown-field, version, and counter-overflow denial tests |
| 121D | Complete | Canonical 1M/10M/100M × 16K/64K/256K workload manifest and runnable synthetic receipt | `streaming_receipt` example and dated implementation receipt |

### Epic 122 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 122A | Complete | Leased `SpatialRecordChunk` identity, global point offset, optional finite XYZ bounds, and tracked column-capacity reservation | chunk continuity, bounds, lifetime, and drop-release tests |
| 122B | Complete | Additive bounded source/sink traits and adapters for the existing synchronous traits | legacy source/sink round trip with no signature changes |
| 122C | Complete | Deterministic single-worker prefetch with count backpressure, cancellation, and fail-closed concurrent-memory admission | ordered delivery, cancellation, and insufficient-budget denial tests |
| 122D | Complete | Source-owned buffer-set recycling with safe ownership extraction | one-allocation steady-state test and `bounded_record_stream` example |

### Epic 123 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 123A | Complete | True sequential PCD/PLY/LAS/LAZ sources plus exact-count sinks, without whole-cloud materialization | binary and ASCII chunk identity, memory peak, round-trip, short-count, and cancellation tests |
| 123B | Complete | Deterministic local and HTTP-range COPC source plus source-driven COPC writer | local/HTTP E2E, stable node ordering, bounds/LOD reuse, and exact point-count tests |
| 123C | Complete | Pre-allocation decode reservations, fixed-size ASCII records, and explicit bounded temporary spool contracts | insufficient-memory/spill preflight, extent denial, cleanup, and no-source-progress tests |
| 123D | Complete | Feature isolation, migration documentation, and runnable bounded format conversion | feature-alone checks, `STREAMING_IO.md`, and `bounded_pcd_to_ply` example |

### Epic 124 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 124A | Complete | Shared-budget chunk crop and affine position/normal transform with contiguous output identity | crop/transform boundary, identity, and peak-memory tests |
| 124B | Complete | Single-pass finite bounds, count, and compensated centroid reduction | cross-chunk and non-finite reduction tests |
| 124C | Complete | Fixed-width external voxel runs sorted by voxel key and global source offset | output equality across input chunk and run sizes plus attribute centroid tests |
| 124D | Complete | Bounded disk extent, run/file-handle limit, merge memory, cancellation, and public feature surface | spill/run denial, cleanup, cancellation-release, root API, docs, and example gates |

### Epic 125 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 125A | Complete | Type-erased Rust builder and metered pull iterator over the Epic 124 operations | crop composition, cancellation, and receipt-counter tests |
| 125B | Complete | Local/HTTP input CLI with open-ended LAS/LAZ output and Ctrl-C cancellation | real PCD → crop → LAS subprocess E2E |
| 125C | Complete | Python iterator backed by the same Rust workflow with cancellation and live receipt JSON | extension compile gate, stubs, and wheel smoke test |
| 125D | Complete | Reproducible workflow documentation and dated implementation receipt | `STREAMING_PIPELINE.md` and Epic 125 receipt |

### Epic 126 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 126A | Complete | Linux/Windows/macOS records, all-format IO, pipeline, CLI, and gate conformance | dedicated three-OS CI matrix |
| 126B | Complete | Fail-closed memory, spill, cleanup, copy, transfer, determinism, and file-handle budgets | typed overrun and relational-limit denial tests |
| 126C | Complete | Stable bounded record contract and provisional adapter/workflow registry | `bounded_streaming_v1_2_surface()` |
| 126D | Complete | Additive migration guidance, limitations, and explicit Python/device ownership | `STREAMING_MIGRATION.md` |
| 126E | Complete | Runnable aggregate release decision and canonical receipt | `streaming_1_2_release_gate` and `STREAMING_RELEASE_RECEIPT.md` |

### SpatialRust 1.2 exclusions

- Native ROS 2/rclrs integration, CUDA, SLAM, and reconstruction expansion.
- A fully GPU-resident 3D pipeline or an implicit `ExecutionPolicy::Auto`
  device crossover.
- Cross-chunk normal estimation and global clustering.
- Stabilization or redesign of the provisional `SpatialTensor` API.

## SpatialRust Visual program (Epics 133–140)

The Visual program makes point clouds, algorithm results, reconstructed scenes,
and synchronized sensor data inspectable without introducing UI or rendering
dependencies into `spatialrust-core`. Host/device transfers remain explicit,
named, and measurable. The base renderer starts with materialized data; bounded
COPC/index-driven LOD follows the Epics 127–132 contracts.

| Epic | Status | Depends on | Deliverable |
| --- | --- | --- | --- |
| 133 | Complete | 0, 100 | Backend-independent borrowed geometry, camera, style, layer, residency, and explicit transfer-receipt contracts in `spatialrust-viz` |
| 134 | Complete | 89, 133 | Headless wgpu point/line/triangle renderer with explicit upload, color maps, depth, picking, screenshots, and reusable GPU resources |
| 135 | Complete | 134 | Native viewer MVP with orbit/pan/zoom, layer inspector, point attributes, and algorithm-debug overlays for normals, voxels, planes, clusters, and registration |
| 136 | Complete | 92–94, 135 | Mesh, surfel, Gaussian, trajectory, pose-graph, camera-frustum, RGB-D, and semantic scene inspection |
| 137 | Complete | 127–132, 134 | Bounded COPC/index-driven frustum and LOD streaming with cancellation, progressive refinement, and strict memory/upload/point budgets |
| 138 | Complete | 134–137 | WebAssembly/WebGPU viewer with portable scene state, browser input, and bounded remote data access |
| 139 | Complete | 135–138 | Python and Jupyter adapters using the same viewer state and explicit ownership/transfer contracts |
| 140 | Complete | 133–139 | Cross-platform headless image conformance, native/Web/Python smoke tests, performance receipts, documentation, migration guidance, and Visual release gate |

### Visual delivery slices

| Slice | Scope | Required evidence |
| --- | --- | --- |
| 133A | Borrowed SoA positions, RGB/scalar attributes, lines, and indexed triangles | zero-copy pointer identity, length, index, and mismatch-denial tests |
| 133B | Validated projection, camera, color, point style, unique layer ordering, and core capability adapter | invalid finite/range/style tests and feature-alone build |
| 133C | Explicit host/device residency and ordered transfer receipt | direction/residency denial and checked byte-overflow tests |
| 134A | Explicit wgpu upload and device-resident geometry handles | exact upload ledger, runtime identity, recycling, and no-hidden-readback tests |
| 134B | Point, line, and triangle render passes with depth and stable color maps | deterministic headless fixtures and shader validation |
| 134C | Picking, camera fit, and caller-requested RGBA screenshot readback | exact synthetic IDs, bounds, transfer, and row-padding tests |
| 135A | Native window, camera controls, drag/drop, layer visibility, style controls, and attribute inspector | scripted input/state tests plus native smoke test |
| 135B | Normal, voxel, plane, cluster, correspondence, bounds, and search-radius overlays | canonical algorithm fixtures and stable overlay identity |
| 136A | Mesh, surfel, Gaussian, trajectory, pose graph, frustum, and semantic adapters | source-identity and geometry-count parity tests |
| 136B | Synchronized RGB/depth/cloud timeline and projection inspection | deterministic timestamp/frame alignment fixtures |
| 136C | Lineage-preserving `SpatialRecordEntity` semantic viewer adapter | source identity/count, confidence, and generated-byte receipt tests |
| 137A | Camera-driven bounded LOD request planner with hysteresis and cancellation | deterministic selection across traversal order and camera jitter |
| 137B | Leased chunk upload, GPU eviction, progressive refinement, and hard budgets | memory/upload/point/in-flight denial, cancellation, and cleanup receipts |
| 138A | WASM/WebGPU renderer and serializable viewer state | browser smoke test and native/Web headless parity |
| 138B | Bounded HTTP range source and browser interaction | remote fixture, cancellation, cache, and request-budget tests |
| 139A | Python viewer state, explicit NumPy borrowing/copying, and native launch | wheel smoke tests and ownership-lifetime tests |
| 139B | Jupyter widget transport and Web viewer embedding | notebook execution and state round-trip tests |
| 140A | Linux/Windows/macOS renderer and viewer conformance | strict headless images, transfer ledgers, and native smoke matrix |
| 140B | Web/Python/Jupyter conformance, docs, migration, and aggregate release decision | fail-closed Visual release gate and committed receipt |

### Epic 134 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 134A | Complete | Exact position/RGB/scalar/index upload ledger, renderer runtime identity, bounded buffer recycling, and wrong-runtime denial tests |
| 134B | Complete | Deterministic point/line/triangle headless rendering, depth, point-size quads, RGB/scalar color modes, and six cached pipeline variants |
| 134C | Complete | Exact point IDs, perspective bounds fit, tightly packed RGBA readback, row-padding removal, runtime/bounds denial, and transfer receipts |

### Epic 135 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 135A | Complete | Feature-gated winit shell, deterministic orbit/pan/zoom/focus reducer, resize and validated drag/drop, stable layer visibility/style state, and point-attribute inspector |
| 135B | Complete | Owned normal, voxel, plane, cluster, correspondence, bounds, and search-radius fixtures with stable IDs, exact geometry counts, borrowed layer conversion, and malformed-input denial |

### Epic 136 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 136A | Complete | Mesh zero-copy pointer identity plus surfel, Gaussian RGB/opacity, trajectory, pose-graph, calibrated frustum, and semantic centroid adapters with exact source/output/generated-byte receipts |
| 136B | Complete | Borrowed RGB/depth/cloud frames, bounded timestamp skew, ordered nearest-frame selection, exact dimension/payload validation, calibrated pixel unprojection, and invalid-depth handling |
| 136C | Complete | Direct `SpatialRecordEntity` slice adaptation retains wrapper source identity while materializing only renderer XYZ/confidence columns with explicit generated-byte accounting |

### Epic 137 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 137A | Complete | Validated deterministic hierarchy, perspective/orthographic frustum selection, screen-error enter/exit hysteresis, traversal-order and camera-jitter stability, obsolete request cancellation, and resident-ancestor progressive display |
| 137B | Complete | Hard point/host/GPU/upload/in-flight budgets, records-backed RAII chunk leases, cooperative cancellation, protected deterministic LRU eviction, exact upload/cleanup receipts, and selected-bounds COPC query adaptation |

### Epic 138 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 138A | Complete | Strict versioned viewer-state JSON, shared browser input reducer, async WebGPU runtime construction, same-backend 64×64 pixel parity, wasm32 `wasm,webgpu` cross-check, and executable browser smoke fixture |
| 138B | Complete | AbortController-backed 206 Range fetch with exact Content-Length/body validation, deterministic request/byte admission, cancellation, exact-range cache hits, bounded LRU eviction, and JS/WASM copy receipts |

### Epic 139 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 139A | Complete | abi3 wheel import, strict shared viewer-state round-trip, shared reducer input, validated native launch receipt, retained contiguous NumPy SoA pointer identity/lifetime, explicit AoS copy isolation, and exact byte receipts |
| 139B | Complete | AnyWidget transport with Rust state validation, exact source/origin/version checks, Web embed handshake, Python round-trip tests, and executable nbclient notebook smoke |

### Epic 140 progress

| Slice | Status | Evidence |
| --- | --- | --- |
| 140A | Complete | Fail-closed whole-image 64×64 RGBA hash, exact geometry/frame/readback ledgers, mandatory headless adapter, and Linux/Windows/macOS renderer/viewer matrix |
| 140B | Complete | wasm32 plus real-browser smoke, Python 3.8/current and executable Jupyter notebook coverage, Visual guide and `visual-1` migration policy, typed fresh-evidence budgets, and committed allowed receipt |

### Visual completion gates

- `spatialrust-viz` builds without wgpu, windowing, image codecs, ROS 2, ONNX,
  CUDA, Python, or browser dependencies.
- No render or viewer API performs an implicit CPU/GPU crossover. Every upload,
  readback, and device-to-device copy appears in a checked receipt.
- Headless fixtures define portable correctness for geometry, depth, color,
  picking, and overlays; tolerance and adapter identity are recorded.
- Streaming display never materializes the full source and fails before
  exceeding memory, point, upload-byte, or in-flight chunk limits.
- Native, Web, Python, and Jupyter surfaces round-trip one versioned viewer-state
  contract and preserve layer identity and camera state.
- The Visual release gate fails closed on missing, skipped, duplicate, stale, or
  over-budget evidence.

### Visual exclusions

- Rendering or UI types in `spatialrust-core`.
- An automatic CPU/GPU placement policy or an unrecorded staging copy.
- Mandatory native windowing, browser, Python, ROS 2, CUDA, or codec dependencies
  in the default workspace feature set.
- Claims of universal interactive frame-rate parity across adapters or hardware.

## Operational v1.3 program (Epics 141–146)

The foundation and Visual programs are complete. The next program validates the
same contracts against external rosbag2 data without moving sensor or derived
artifacts into the repository. Its canonical input, storage roots, current
evidence, and fail-closed acceptance gates live in
[`docs/REAL_DATA_ACCEPTANCE.md`](REAL_DATA_ACCEPTANCE.md).

| Epic | Status | Depends on | Outcome |
| --- | --- | --- | --- |
| 141 | Active | 91–100, 133–140 | External-data acceptance contract, canonical rosbag2 baseline, and reproducible run identity |
| 142 | Active | 141 | External SSD result-root preflight, run-scoped manifests, resumable checkpoints, and cleanup |
| 143 | Active | 141–142 | Full rosbag2 → records → sync → odometry → TSDF → semantic → Viewer/glTF/OpenUSD E2E |
| 144 | Complete | 142–143 | Memory/transfer/latency budgets, benchmark receipts, failure recovery, and deterministic rerun comparison |
| 145 | Active | 143–144 | ROS 2 publish, edge/distributed partition execution, and optional AI runtime boundaries |
| 146 | Active | 141–145 | API stability, Python/docs/CI updates, security review, and v1.3 release receipt |

### Epic 146 operational acceptance slices

| Slice | Status | Scope | Evidence required |
| --- | --- | --- | --- |
| 146A | Complete: workflow; motion accuracy not accepted | Native forward/backward image tracking and explicit Python RGB-D/PnP adapter; both candidate and point-cloud baseline retain every public-data frame and independent scoring | 160 native/753 Python passes, stubtest/Clippy/WASM compile; both 798/798 generations but both fail unchanged 0.10m ATE limit; `docs/TUM_RGBD_MOTION_VALIDATION.md`; canonical Autoware calibration remains open |
| 146B | Complete: selected Windows hardware and Linux software paths | Fixed full public RGB-D frames on integrated/discrete Vulkan, independent image/geometry references and explicit transfer/resource receipts | All six Windows hardware and six Linux software cases pass fixed gates; two Rust cases on both OSes/thirteen Python cases and Clippy; `docs/PUBLIC_GPU_RGBD_CONFORMANCE.md`; three development frames do not prove broad portability |
| 146C | Complete: workflows; motion not accepted | Persistent depth landmarks, local depth consensus and input-count promotion; checked hard-link input reuse | Fixed xyz 90/798, quality xyz 240/798 and both rpy candidates 2/722; all failures/independent audits preserved; 835 unique local Python cases and 33,203 source-landmark checks; `docs/TUM_KEYFRAME_RGBD_VALIDATION.md` |
| 146D | Complete: numerical fix; motion not accepted | Preserve selected robust-PnP hypotheses through failing or worse final refinement | Parent native/wheel failures reproduced; 166 native/13 properties/837 Python passes. Adaptive xyz reaches 798/798 but ATE 0.118821m fails; stateless xyz 0.122263m and both rpy 2/722 also fail. All four audits, 288,068 landmark rows and 32 typed IO checks pass; `docs/PNP_CONSENSUS_VALIDATION.md` |
| 146E | Complete: fixed capture workflow; motion not accepted | Explicit radius-10/four-level LK, parent-bound image reuse and unchanged original defaults/gates | XYZ both 798/798 but ATE 0.114557/0.144424m fails; rpy 2/722 and 59/722. Four score audits, 318,435 landmark checks, 32 typed IO audits and 868 Python passes; same-row native/OpenCV PnP/LK diagnostics remain below support gate; `docs/WIDE_RGBD_CAPTURE_VALIDATION.md` |
| 146F | Complete: isolated attempt and auditing; hardware not accepted | Isolated Mesa Dozen Vulkan-to-D3D12 bridge in WSL2; reuse the original executable, inputs and acceptance gates | Both preferences fail default adapter admission before all six planned frames. Capability-only probe sees one discrete, conformance-zero adapter; no gate override or operation success claimed. Complete failures and data-only receipt checks retained; `docs/WSL_HARDWARE_GPU_CONFORMANCE.md` |
| 146G | Implemented; xyz accepted, rpy open | Checked explicit source/target RGB-D frames, bounded native projective photometric/depth estimation and Python ownership; both strict and revised full timelines frozen | Revised xyz 798/798, ATE 0.090314m and 1s RPE 0.025987m pass unchanged gates; rpy 40/722 fails; four independent projective/trajectory audits, derivative/pose/schema tests and source/wheel/storage receipts; `docs/DENSE_RGBD_ODOMETRY_VALIDATION.md` |
| 146H | Implemented; full tracking, rpy ATE open | Explicit previous returned dense relative pose as the next seed; same frozen solver, numeric controls and final gates | XYZ 798/798, ATE 0.089993m/RPE 0.026591m passes; rpy 722/722 and RPE 0.048145m passes but ATE 0.114793m fails. All 1,518 causal/projective pairs, both score audits, 963 Python passes and storage/source receipts; `docs/CAUSAL_DENSE_RGBD_VALIDATION.md` |
| 146I | Implemented; Linux GL device creation open | Typed runtime options preserve primary defaults and allow explicit GL; reject uncompiled GL/browser fallback and adapters without compute; fixed public image/point data, unchanged gates | Windows both raw classes pass 6/6; Linux 0/6 observed, both device-creation failures preserved and diagnosed as internal GLSL dynamic-array admission. Native 14, report 34, Python 973 passes/3 skips, Clippy/WASM and bounded storage receipts; `docs/EXPLICIT_GLES_GPU_VALIDATION.md` |
| 146J | Implemented; operation passes, raw-class coverage open | Same frozen executable/input with privately staged Mesa Gallium D3D12/surfaceless GL; no system installation, version override or numerical change | All 6 image/point cases pass, max centroid error 3.818e-6m; both raw Other classes remain unaccepted for two-class hardware coverage. Preserve all prior failures, 698 hash entries/8 links, bounded E storage; `docs/ISOLATED_GALLIUM_GLES_VALIDATION.md` |
| 146K | Implemented; external controls pass, native rpy accuracy open | Pinned installed Open3D hybrid RGB-D with owned causal seed/images; preserve native outputs and original raw projective/full-cloud/trajectory gates | Complete 798/722-frame controls pass: xyz ATE/RPE 0.051174/0.023110m, rpy 0.099684/0.048494m; all 1,518 independent seed/projective pairs and both score audits pass. Native rpy remains at ATE 0.114793m. Strict LF/CRLF parent identity, Python 1,018 passes/3 skips, no image copies, 337MB new output and 1.43GB passing scratch reclaimed; `docs/OPEN3D_DENSE_RGBD_COMPARISON.md` |
| 146L | Implemented; xyz passes, rpy incomplete | Explicit typed Gaussian3 fitting policy, raw default/final statistics unchanged, bounded deterministic row tiles | XYZ 798/798 ATE/RPE 0.082755/0.025186m passes; rpy 349/722 fails at raw support 0.497388 < 0.5, all 373 errors retained. Independent 1,145 causal/raw pairs and both score audits; native 13, Python 1,043 passes/3 skips, Clippy/WASM, exact five-case raw regression and bounded reused E storage; `docs/GAUSSIAN_DENSE_RGBD_VALIDATION.md` |
| 146M | Implemented; xyz passes, rpy accuracy open | Preserve Gaussian3 gray inputs and raw final admission; retain original fitting depth and all numeric/trajectory controls | Complete 798/798 xyz and 722/722 rpy; xyz ATE/RPE 0.091755/0.027265m passes, rpy 0.120216/0.048803m fails ATE. Independent 1,518 causal/raw pairs and both score audits, native 15/Python 1,069 passes and 3 skips, exact ten-case raw/joint regression; 316MB results, 128MB final new build/cache, zero image copies, all prior variants retained; `docs/GRAY_GAUSSIAN_DENSE_RGBD_VALIDATION.md` |
| 146N | Implemented; xyz passes, rpy accuracy open | Explicit gray-only fitting normalization to mean 0.5 using the native initial correspondence set; original depth/final raw admission and all gates unchanged | Complete 798/798 xyz and 722/722 rpy; xyz ATE/RPE 0.086962/0.025649m passes, rpy 0.130595/0.048722m fails ATE. Independent 1,518 causal/raw pairs and both score audits; native 20/Python 1,103 passes and 3 skips, exact 15-case old/new selector regression; 317MB results, 25MB final new package/cache, 560MB passing scratch reclaimed and zero image copies; `docs/PAIR_MEAN_DENSE_RGBD_VALIDATION.md` |
| 146O | Complete: operator audit; native rpy accuracy open | Fixed synthetic pyramid/pixel/missing-depth/derivative cases and actual installed CPU reference APIs; no production estimator change | Confirm different coarse gray values (23/255 vs 27/255), 7x5 lower grids (4x3 vs 3x2) and missing-depth/border lookup rules. All twelve native Jacobian entries match independent analytics; reference residual/weights observed, runtime Jacobian unavailable. Native 21/Python 1,134 passes and 3 existing skips, 116KB results, 5.6MB retained build/cache, 438MB passing scratch reclaimed, zero sensor copies/downloads and all prior artifacts preserved; `docs/DENSE_SAMPLING_AUDIT.md` |
| 146P | Implemented; xyz passes, rpy accuracy open | Complete opt-in rounded-pixel/Sobel fitting model with original raw final admission, typed CPU ownership and preflighted deterministic rows | Complete 798/798 xyz and 722/722 rpy; ATE/RPE 0.050258/0.023051m passes xyz, 0.100327/0.048503m fails rpy ATE. All 1,518 independent causal/raw pairs and both score audits pass. Native 28/Python 1,171 passes with 3 existing skips, exact 20-case legacy regression, Clippy/WASM and wheel checks; 322MB results, 46MB retained build/cache, 641MB passing scratch reclaimed, zero sensor copies/downloads; `docs/ROUNDED_SOBEL_DENSE_RGBD_VALIDATION.md` |
| 146Q | Complete study; robustness open | Fixed native candidate and source identities before preparation; both full trajectories frozen before independent scoring | plant 1139/1139 poses, ATE/RPE 0.077810/0.037764m passes; room 1360/1360 poses, ATE/RPE 0.379777/0.058405m fails. All 2497 successful causal/raw pairs and both scores audited, every failure/gap retained. Python 1,229 passes/3 old skips, native 28/workflow 12; production trees unchanged. Retain 3.624GB new owned trees, reclaim 1.202GB passing scratch; development rpy/full 80% remain open; `docs/FROZEN_HELDOUT_RGBD_VALIDATION.md` |
| 146R | Complete: storage inventory | Complete immutable logical counts and acquisition preflight; never ignore denied/stat/vanished/non-regular subtrees | Real restricted omission reproduced; new counter stops, privileged three-root total 3,624,257,697 bytes matches independent walk. 19 new IO/77 focused/1,248 full Python passes and 3 old skips; 388 JSON/36 native artifacts preserved. Retain 106KB metadata, reclaim 642MB passing scratch, zero new sensor payload/build/score; `docs/REGULAR_FILE_INVENTORY.md` |
| 146S | Complete: synthetic precision/update audit | Six fixed synthetic cases, independent normal systems/proper left retractions and actual frozen CPU one-step APIs | Native/reference predictions agree to <=1.11e-16; actual pose difference <=4.95e-7 and 24 public Gaussian byte checks match. Runtime Jacobians remain unobserved. Preserve six initial adapter failures; fix only tuple assembly, reuse reference; 42 math/1,290 full Python passes and 3 old skips. Retain 877KB, reclaim 1.12GB passing scratch, no new native/sensor/real-score work; `docs/DENSE_PRECISION_AUDIT.md` |
| 146T | Implemented; xyz/rpy/plant pass, room open | Explicit f32-intermediate/XYZ model, checked f64 solve and proper left Euler increments; original raw final gates and all legacy choices retained | All 4,019 development poses generated and 4,015 raw/causal pairs/four scores independently audited. Rpy ATE/RPE 0.099679/0.048503m now passes; room 0.378653/0.058375m fails both. Six synthetic reference pairs <=1.11e-16, 25 exact legacy cases, native 35/Python 1,310 passes with three old skips and Clippy/WASM/wheel checks. Retain 3.352GB logical trees/cache including 2.503GB hard-link views, reclaim 723MB passing scratch, zero sensor copies/downloads; full 80% remains open; `docs/REFERENCE_PRECISION_DENSE_RGBD_VALIDATION.md` |
| 146U | Complete: full room drift audit; robustness open | Metadata-only comparison of both frozen full trajectories with original first gauges, 1s pairs, eight chronological bins and independent rigid/score/raw evidence | Per model preserve 1,360 rows, 1,359 reference matches, 1,357 adjacent and 1,328/1,330 1s pairs. Models differ by 2.008mm/0.0602 degrees RMS, while new room error remains 0.378653m/12.803 degrees. Fixed bin 170–339 contributes 37.18% of 1s translation SSE; associations remain descriptive. 41 final validation/1,310 SDK passes plus 3 old skips, new dedicated CI; all production crates/437 JSON/39 native artifacts preserved. Retain 15.3MB and reclaim 561MB passing scratch, zero image/archive reads or new fits; `docs/ROOM_DRIFT_AUDIT.md` |
| 146V | Implemented; fixed Huber development comparison | Explicit scalar Huber IRLS on the isolated reference-precision pipeline; original raw final gates and all six old choices retained | 4,019 poses and 4,015 causal/raw pairs independently audited; original passing scenes: xyz. Native 42/Python 1,337 passes with three old skips, 30 exact old-selector cases, six independent NumPy cases <=6.51e-17 and Clippy/WASM/wheel checks. Retain 3.371GB logical trees/cache with hard-linked inputs; reclaim 723MB passing scratch, zero sensor copies/downloads; full 80% remains open; `docs/HUBER_DENSE_RGBD_VALIDATION.md` |
| 146W | Complete: full room public reference diagnostic | One frozen CPU hybrid policy on all 1,360 original frames, independent raw/causal/score audit and complete native comparison; seven bounded source-model cases stay distinct from runtime pairs | Reference ATE/RPE 0.378945/0.058388m; original reference acceptance false, native failure retained. All 1,359 successful causal/raw pairs and 1,360 planned comparison rows preserved; 37 new/1,374 SDK passes plus three old skips and frozen wheel validation. Preserve 477 JSON/42 native artifacts, reclaim 696MB passing scratch, zero payload copies/downloads/native builds/cache growth; full 80% remains open; `docs/ROOM_REFERENCE_AUDIT.md` |
| 146X | Complete: rooted propagation substrate; drift correction open | Complete SE3 propagation from a fixed root, explicit node-from-anchor algebra, indexed iterative traversal and atomic finite/tree validation | Original independent contract tests: 4 pass/12 fail; now all 16 pass, plus scan direction case. Mapping 22 default/24 all-feature passes, north-star integration, Clippy/WASM/format; dedicated Linux/Windows/macOS CI. Preserve 493 JSON/42 frozen artifacts, zero sensor copies/downloads/new real fits; core/math/GPU/Python/dependencies unchanged. Operational 80% remains open; `docs/POSE_GRAPH_ROOTED_LOCALIZATION.md` |
| 146Y | Complete: explicit mapping sensor transport; physical acceptance open | Invert current-from-first scan poses into a fixed body gauge, conjugate front motion through shared body extrinsics, finite-check transforms and constrain native output paths | Original coordinates 5 pass/10 fail; existing Windows path failure and prospective path baseline preserved. Twelve shared coordinate/finite cases plus one path case; full-bag 15/E2E 17 passes, ROS2/mapping/calibration/Clippy/format and Linux/Windows/macOS CI. Preserve 493 JSON/42 frozen artifacts; zero data copies/downloads/new real fits. Operational 80% remains open; `docs/MAPPING_SENSOR_TRANSPORT.md` |
| 146Z | Complete: incremental retained scan admission; identity/RSS separate | Shared episode gate covers finished scans and pending owned chunks across both sensors, pre-copy capacity checks, bounded assembly requests and fail-fast source errors | Preserved per-topic drain baseline: 23 pass/9 fail (not unchanged whole-program scheduling); now 37 full-bag/17 E2E passes, 22 new IO/capacity cases, SQLite malformed-tail proof, records/sync/Clippy and three OS CI. Preserve 493 JSON/42 frozen artifacts; no real-data acquisition/copies/fits. Operational 80% remains open; `docs/SCAN_COLLECTION_BUDGET.md` |
| 146AA | Complete: fail-closed scan/pose ambiguity admission; physical identity open | Match selected SQLite message/group counts, reject duplicate corrected times per topic before odometry, validate exact trajectory cardinality and complete stamps before TSDF | Preserved filter/zip baseline: 41 pass/13 fail; now 54 full-bag/17 E2E passes with 17 new synthetic IO/association cases, existing controls/Clippy and three OS CI. Preserve 493 JSON/42 frozen artifacts; cache growth 15MB and zero real-data reads/acquisition/copies/fits. Operational 80% remains open; `docs/SCAN_IDENTITY_ADMISSION.md` |
| 146AB | Complete: portable mapping receipt consistency; unique fusion coverage open | Bind completed source cardinality/bounds, front trajectory totals/topic/frame/clock and volume root across existing receipt fields; retain consistent partial diagnostics and visit semantics | Original mapping subset: 14 pass/15 fail; 26 new cases, viewer default/serde/all-feature 68/73/81 passes, full-bag/E2E 54/17, controls/Clippy and three OS CI. Preserve 493 JSON/42 frozen artifacts; cache growth 933MB under fixed cap, zero real-data reads/acquisition/copies/fits. Operational 80% remains open; `docs/MAPPING_RECEIPT_CONSISTENCY.md` |
| 146AC | Complete: bounded fusion submission ledger; physical coverage open | Bind verified input/calibration, immutable episode indices, exact replay associations and successful TSDF visits; reconcile counters and stream an atomic no-clobber coverage artifact | Original replayer characterization 13 pass; 33 new ledger cases and 100 full-bag passes, synthetic SQLite publication/blocked controls, related checks/Clippy and three OS CI. Fixed 65,536 records/8MiB metadata/32MiB JSON, preserved 493 JSON/42 frozen artifacts and zero real-data reads/acquisition/copies/fits. Operational 80% remains open; `docs/FUSION_COVERAGE.md` |
| 146AD | Complete: bounded canonical fusion readback; physical acceptance open | Compare persisted producer bytes with independently completed coverage through fixed buffers before parent success; link exact size and existing checksum | Original producer 100 pass; 28 new IO/schema/binding controls and 128 full-bag passes, related checks/Clippy and three OS CI. Account for 48KiB fixed readback buffers within the unchanged 8MiB ledger cap and 32MiB JSON plus one probe. Preserve 493 JSON/42 frozen artifacts and zero real-data reads/acquisition/copies/fits. Operational 80% remains open; `docs/FUSION_READBACK.md` |
| 146AE | Complete: TSDF disjoint support admission; physical accuracy open | Intersect raw inclusive support-cell ranges with the grid before clamping; preserve overlapping field math and submitted cloud counts | Original all-feature baseline 10 pass/18 fail; 21 new controls, scene default/all 26/28 pass, 729 independent dyadic enumerations and 343 frozen-parent update comparisons. Related mapping/Clippy and three OS CI; preserve 493 JSON/42 frozen artifacts with zero real-data reads/acquisition/copies/fits. Operational 80% remains open; `docs/TSDF_SUPPORT_ADMISSION.md` |
| 146AF | Complete: fallible TSDF grid construction; physical acceptance open | Reject non-finite origins/centers and unrepresentable field capacities before allocator access; reserve both fields before initialization using existing errors | Original all-feature baseline 36 pass/9 fail; 17 new controls and scene default/all 43/45 pass, unchanged integration/meshing/public types. Related mapping/Clippy and three-OS scene CI; preserve 493 JSON/42 frozen artifacts with zero real-data reads/acquisition/copies/fits. Operational 80% remains open; `docs/TSDF_GRID_ADMISSION.md` |
| 146AG | Reserved: marching-tetrahedra surface topology | Correct six quadrilateral cycles and one three-inside winding; preserve interpolation, cube split and TSDF integration | Prospective bounds in `benchmarks/mesh-topology-plan.json`; independent all-case edge/area/winding and tiny volume plane controls, preserved parent failures and three-OS scene CI. Physical accuracy remains open |

The operational maturity target is 80% on the explicitly requested real-data,
multi-GPU/OS and public-release criteria. Historical engineering estimates in
`docs/MATURITY.md` use a different subjective scale and cannot by themselves
prove this target. Configured jobs, fixture-only success and a generation count
without accurate reference motion do not establish operational acceptance.

### Epic 141 acceptance slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 141A | Complete | Canonical external input snapshot, topic counts, output hashes, storage policy, and fail-closed downstream gates | [`docs/REAL_DATA_ACCEPTANCE.md`](REAL_DATA_ACCEPTANCE.md) |
| 141B | Complete | Automated absolute-root/free-space preflight and local checksum manifest validation | `storage-preflight`, `DatasetManifest::validate_local_files`, and rosbag2 CLI smoke |
| 141C | Complete: publisher configuration; physical/clock gates open | Public canonical reacquisition, publisher-bound front/rear configuration and native partial admission | 1,535 headers and six pinned source files verified; native graph ready, missing clock blocks overall, wrong source rejected; 801 Python passes (35 new); `docs/PUBLIC_AUTOWARE_FRAME_EVIDENCE.md` |

### Epic 142 progress

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 142A | Complete | Atomic run-scoped stage checkpoint, complete-run resume verification, and narrow temp cleanup | `rosbag2_e2e --resume`, external `142a-checkpoint-smoke` receipt |
| 142B | Complete | Persisted bounded XYZ/XYZI ingest episode and summary for partial-stage resume, with manifest-tracked survivors | `--stop-after ingest` → `--resume`, external `142b-ingest-resume-smoke-v2` receipt/manifest |

### Epic 143 progress

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 143A | Complete | Bounded external rosbag2 → records → sync → ICP → TSDF → semantic → Viewer/glTF vertical smoke | `rosbag2_e2e`, external receipt/manifest under `v1-3/143a-e2e-smoke` |
| 143B | Active | Calibration/frame readiness gate, source-bound TF inventory, then calibration-aware full-bag mapping, semantic quality, and interchange acceptance | external readiness and SSD survey receipts are fail-closed; separate-source TF fixture is diagnostic; canonical clock/frame artifacts and full-run receipt pending |

### Epic 144 progress

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 144A | Complete | Versioned stage timing, bounded memory observations, explicit transfer counters, and fresh-source baseline | Receipt version 2 under `v1-3/144-performance-fresh` |
| 144B | Complete | Three-run variance, bounded-smoke stage budgets, failure-safe non-overwriting comparison, and deterministic hash report | `rosbag2_e2e_compare`, external `144-comparison-smoke-v2` receipt |

### Epic 145 progress

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 145A | Complete | Portable Spatial Studio state and one-screen source-bound dashboard for layers, timeline, calibration/TF gates, and explicit pipeline metrics | `spatialrust-viewer::StudioState`, `rosbag2_studio`, external `145a-spatial-studio-v2` JSON/HTML/manifest |
| 145B | Complete | TF/Calibration Observatory for artifact status, clock diagnostics, source-bound edges, graph topology, and fail-closed composition admission | `spatialrust-viewer::CalibrationObservatoryState`, `rosbag2_calibration_observatory`, external `145b-tf-calibration-observatory` JSON/HTML/manifest |
| 145C | Complete | One-command bounded deterministic rosbag2 replay with exact input identity, portable trace state, static dashboard, and checksummed manifest; replay readiness remains separate from mapping/calibration admission | `spatialrust-viewer::ReplayDemoState`, `rosbag2_replay_demo`, external `145c-one-command-replay-demo` JSON/HTML/manifest |
| 145D | Complete | Source-bound TSDF/glTF Map Diff with decoded stable-index displacement metrics, topology/hash checks, 16×16 spatial heatmap, and fail-closed source/frame/calibration gates | `spatialrust-viewer::MapDiffState`, `rosbag2_map_diff`, external `145d-map-diff-v2` JSON/HTML/manifest |
| 145E | Complete | AI Semantic Overlay with deterministic mock inference, explicit CPU tensor transfer receipt, RGB class palette, bounded source-index predictions, canvas dashboard, and fail-closed source/frame/calibration gates | `spatialrust-viewer::SemanticOverlayState`, `rosbag2_semantic_overlay`, external `145e-ai-semantic-overlay` JSON/HTML/manifest |
| 145F | Complete | Portable glTF/USD Digital Twin with byte-identical canonical glTF, explicit ASCII USDA companion/reference layer, optional source-bound semantic attachment, polished dashboard, and fail-closed source/frame/calibration gates | `spatialrust-viewer::DigitalTwinState`, `rosbag2_digital_twin`, external `145f-digital-twin` JSON/HTML/manifest plus negative validation probe |
| 145G | Complete | Source-bound Dataset Health Dashboard aggregating canonical identity, topic inventory, mesh/readiness checks, 145A–145F stage health, storage receipts, and fail-closed mapping admission | `spatialrust-viewer::DatasetHealthState`, `rosbag2_dataset_health`, external `145g-dataset-health-v2` JSON/HTML/manifest plus wrong-source negative probe |
| 145H | Complete | Source-bound ROS 2 PointCloud2 Live Publish Bridge with explicit topic/frame mapping, bounded deterministic packets, CPU CDR loopback round-trip receipt, transport counters, dashboard, and fail-closed source/frame/calibration gates | `spatialrust-viewer::LivePublishState`, `rosbag2_live_publish`, external `145h-live-publish-v2` JSON/HTML/manifest plus wrong-source validation probe |
| 145I | Complete | Source-bound edge-to-host partition execution receipt consuming live-publish packets with deterministic PartitionGraph topology, named explicit-copy transfers, bounded queue/backpressure counters, dashboard, and fail-closed source/upstream/calibration gates | `spatialrust-viewer::EdgePartitionState`, `rosbag2_edge_partition`, external `145i-edge-partition-v2` JSON/HTML/manifest plus wrong-source validation probe |
| 145J-A | Complete | Bounded interactive Mission Cockpit joining source-indexed packet samples, timeline playback, point selection/measurement, and edge-to-host execution graph while preserving source/frame/calibration gates | `spatialrust-viewer::MissionCockpitState`, `rosbag2_mission_cockpit`, external `145j-mission-cockpit-v2` JSON/HTML/manifest plus wrong-source validation probe |
| 145J-B | Complete | Explicit source-bound clock and front/rear extrinsic evidence manifest, quality/path validation, fail-closed registration receipt, and Mission Cockpit integration; real calibration artifacts remain an external prerequisite | `spatialrust-viewer::CalibrationEvidenceState`, `rosbag2_calibration_evidence`, external `145j-calibration-evidence-v2` JSON/HTML/manifest plus wrong-source validation probe |
| 145K-A | Complete | Source-bound bounded full-bag PointCloud2 ingest, chunk reassembly, registered clock correction, root-to-front/rear frame application, ICP pose graph, TSDF/glTF receipt, and Mission Cockpit mapping-state gate; canonical admission remains blocked without real calibration artifacts | `spatialrust-viewer::FullBagMappingState`, `rosbag2_full_bag_mapping`, external `145k-full-bag-mapping-v3` JSON/HTML/manifest and `145k-mission-cockpit-v3` integration receipt |

## Parallel image and point-cloud execution follow-up

Image and point-cloud execution share the goals of bounded parallel work,
retained device data, explicit transfer receipts, and measured stage costs.
These focused slices do not change automatic GPU placement or the operational
v1.3 admission gates.

| Slice | Status | Deliverable | Acceptance |
| --- | --- | --- | --- |
| Frame transfers | Complete | Partition source points without unused centroid reduction/readback; account for cell-count synchronization and attribute copies | source-point/CPU partition parity across chunk sizes, planar normals, empty frames, rebuild/reuse receipts, and GPU image-chain regressions; `notes/2026-10-11_gpu_frame_transfer_accounting.md` |
| Shared GPU timing | Complete | Optional bounded pass timestamps shared by image chains and point-cloud frames, with explicit native collection | output/receipt parity, feature fallback, runtime ownership, capacity/failure checks, completed-work comparison; `notes/2026-10-11_gpu_stage_timing.md` |
| Point-cloud compaction | Complete | Hierarchical work-efficient prefix scan in one submission | GPU/CPU parity through 1,000,003 elements, repeated keys crossing hierarchy boundaries, device index order, exact query capacity, and isolated completed-work comparison; `notes/2026-10-11_parallel_image_point_execution.md` |
| Unused GPU voxel counters | Complete | Drop discarded per-point atomics/explicit clear; four-byte compatibility storage replaces 4*N bytes; exact starts-derived counts and original transfers/layout | 380 native tests; 50,785,230 raw/guard words, 24,036,610 complete parent/CPU values, fresh/warm/zero-budget pools and profiling; Clippy/CPU/WASM; two balanced surveys, write 9 favorable / 14 inconclusive / 1 adverse, resident 6 favorable / 18 inconclusive / 0 adverse, full 3 favorable / 21 inconclusive / 0 adverse; all 72 rows retained. `notes/2026-10-17_unused_gpu_voxel_counts.md` |
| GPU voxel boundary/index lifetime reuse | Complete | Owned indices temporarily store mark flags; scan consumes them before prefix-based boundary recovery and final index overwrite; save 4*N-4 private bytes with distinct per-pass bindings | 383 native tests; 80,211,910 phased raw/guard words, 24,036,610 complete parent/CPU values, fresh/warm/zero-budget pools and profiling; Clippy/CPU/WASM; two balanced surveys, write 18 favorable / 6 inconclusive / 0 adverse, resident 2 favorable / 22 inconclusive / 0 adverse, full 1 favorable / 23 inconclusive / 0 adverse; all 72 rows retained. `notes/2026-10-17_reused_gpu_voxel_boundary_storage.md` |
| Fused GPU voxel boundary scan | Complete | Generate exact key boundaries in first scan block; remove one mark pass/dispatch and 8*N source-level flag bytes, preserve scratch/output/transfer contracts and update profile capacity | 393 native tests; 68,052,310 block and 67,944,740 hierarchy words/guards, 24,036,610 complete parent/CPU values; exact frame capacities/zero/warm/budget/profiling checks; Clippy/CPU/WASM; two balanced surveys, GPU 24 favorable / 0 inconclusive / 0 adverse, resident 11 favorable / 13 inconclusive / 0 adverse, full 2 favorable / 22 inconclusive / 0 adverse; all 72 rows retained. `notes/2026-10-17_fused_gpu_voxel_boundary_scan.md` |
| Fused GPU voxel radix entry construction | Complete | Read keys directly and construct source indices in a private first-scatter shader; remove build pass/submission/uniform/group and intermediate write, preserve original radix shader, stable order and scratch/transfers | 397 native tests, complete frozen parent/CPU/raw intermediate guards, threshold/tails/ownership/concurrency/preflight and CPU/WASM; all 240 phased measurement rows in `notes/2026-10-17_fused_gpu_voxel_radix_build.md` |
| CPU-key voxel entry upload | Complete | Prepare source indices in existing clamp loop; upload owned sort entries directly, remove 16*N device entry allocation and singleton setup; unchanged GPU-key route, shaders, scratch and explicit transfers | 401 native tests, complete frozen CPU entry points, signed/raw/retained/concurrent/pool/profile guards and CPU/WASM; paired prep 0 favorable / 21 inconclusive / 3 adverse, resident 6 favorable / 17 inconclusive / 1 adverse, full 6 favorable / 18 inconclusive / 0 adverse; all 72 rows in `notes/2026-10-18_uploaded_cpu_voxel_entries.md` |
| Bounded CPU voxel entry preparation | Complete | At least 524,288 native points use caller-inclusive at most four workers and one exact-sized safe zeroed Vec; stable global indices and original small/one-worker/WASM collector; unchanged GPU work/explicit transfers | 406 native tests, frozen parent/CPU/raw/capacity/guards/concurrency/pool/profiling and CPU/WASM checks; large-input prep 16 favorable / 0 inconclusive / 0 adverse, resident 15 favorable / 1 inconclusive / 0 adverse, full 16 favorable / 0 inconclusive / 0 adverse; all 104 paired rows and every host sketch size in `notes/2026-10-18_parallel_cpu_voxel_preparation.md` |
| CPU image analysis | Complete | Bounded parallel thresholding and local u8 histograms under `imgproc-analysis` | Exact 1/2/6-worker strided/ROI parity, floating bit patterns, padding exclusion, small/empty/single-row fallbacks, and before/after plus one-worker measurements; `notes/2026-10-11_parallel_image_point_execution.md` |
| Temporary storage | Complete | Private submitted scratch leases plus byte-bounded buffer/texture retention with explicit controls and stats | repeated-frame allocation/retained-byte measurements, CPU/GPU parity with stale scratch and live outputs, concurrent submissions, abandoned encoders, resolution/size churn and zero/shrinking budgets; `notes/2026-10-11_bounded_gpu_temporary_storage.md` |
| CPU contrast enhancement | Complete | Bounded parallel CLAHE tile/row work and equalization lookup application with shared horizontal geometry | Exact 1/2/6-worker ROI, metadata, odd-grid, clipping, rounding and degenerate-image parity; checked grid allocation; before/after and one-worker Criterion comparison; `notes/2026-10-11_parallel_contrast_enhancement.md` |
| Bounded CPU point queries | Complete | Group original radius/k-NN query chunks into balanced, available-CPU-bounded scoped workers with worker-sized dispatch metadata | Unsorted serial/append parity, exact callback boundaries, deterministic 1/2/6-worker and tiny-chunk limits, minimal feature/WASM checks and completed-work comparison; `notes/2026-10-11_bounded_parallel_point_queries.md` |
| Bounded CPU KD-tree construction | Complete | Explicit parallel constructor, exact preallocated preorder ranges and unchanged partitions; resident GPU GICP host preparation | Frozen/layout/query-bit checks with 1/2/6 workers, 216 exact GICP diagnostics conditions, feature/facade/WASM checks and completed tree/registration comparisons; `notes/2026-10-13_parallel_kdtree_build.md` |
| GPU image histogram | Complete | Exact two-pass logical-channel histograms with bounded private partials, resident bin ownership, explicit native readback and optional shared timestamps | Packed/strided/ROI/channel and hot-bin CPU parity, stale scratch/live output/concurrency ownership, limit/overflow checks, GPU regressions and completed-work/transfer measurements; `notes/2026-10-11_resident_gpu_image_histograms.md` |
| GPU histogram CDF | Complete | Work-efficient inclusive 256-bin integer scan into an independently owned resident result, with explicit readback and one-pass profiling | Exact u32::MAX counts, all logical channels/ROI and grayscale chains, source recycling/live output/concurrency ownership, cumulative receipts and completed-work placement comparisons; `notes/2026-10-11_resident_gpu_image_cdf.md` |
| GPU histogram equalization | Complete | Resident grayscale histogram/CDF/lookup/application with CPU-compatible binary64 rounding and bounded temporary ownership | Constant-byte/metadata/ROI and RGB-gray chain parity, 2,309,120 independent LUT comparisons through maximum counters, pool reuse/concurrency, five-pass timing and completed-work placement measurements; `notes/2026-10-11_resident_gpu_histogram_equalization.md` |
| GPU adaptive contrast | Complete | Resident Reflect101 tile histograms, clipping/lookup, GPU-generated geometry and CPU-compatible binary64 interpolation | Odd grids/ROI/constants/clip limits and 65,536 arithmetic/geometry rows plus 66,049 interpolation boundary pixels, stale scratch/live outputs/concurrency, three-stage timestamps and placement/transfer measurements; `notes/2026-10-11_resident_gpu_clahe.md` |
| GPU Gaussian filtering | Complete | Resident 1..=4-channel separable 3/5/7 kernels with Q8/Q15 CPU-compatible weights, wide integer final sums, five borders and two optional pass timestamps | 3,240 CPU byte-exact conditions, extreme sigmas/constant mass/large grid strides, ROI/metadata/receipts/ownership/concurrency/capacity checks and completed-call timing; `notes/2026-10-13_gpu_gaussian_blur.md` |
| GPU median filtering | Complete | Exact resident 1..=4-channel 3x3/5x5 integer selection with replicate/zero borders, one pass and independently lazy kernel pipelines | Exhaustive 3x3 binary and independent sorted-byte network checks, 896 CPU byte-exact image conditions, ROI/metadata/padding/receipt/ownership/concurrency/limit checks and completed CPU/full-GPU/resident measurements; `notes/2026-10-14_gpu_median_filter.md` |
| Packed GPU Gaussian Q8 scratch | Complete | Pair exact 16-bit horizontal sums per pixel, reducing two/four-channel scratch requests by half and three-channel requests by one third, with old Q15/guards and actual work receipts | 213 image/voxel/facade regressions; 42,997,770 raw parent/CPU words and 88,570,575 complete bytes; bounds/guards/odd halves/profile/cache/ownership, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_gaussian_q8_scratch.md` |
| Paired GPU Gaussian Q8 reads | Complete | Read each packed vertical word once for two exact channel sums; source load expressions per tap C to ceil(C/2), with Q15/one-channel and all API/resource/transfer boundaries retained | 348 native tests; 57,330,360 raw parent/u64-CPU physical and 46,830,700 complete-parent/CPU bytes; all borders/dirty halves/grid strides, Clippy/CPU/WASM; two paired surveys favor all 16 affected vertical-pass rows and 11/16 resident rows, with full-call/control conditions retained. `notes/2026-10-17_paired_gaussian_q8_vertical.md` |
| Split GPU Gaussian Q15 sums | Complete | Two bounded base-65536 digit sums replace general per-tap wide products/carries, with exact final half-up rounding for all Q15 channels; keep API/resource/transfer boundaries | 350 native tests; 97,850,720 raw parent/u64-CPU physical and 60,210,900 complete-parent/CPU bytes; all channels/borders/mixed axes/rounding/grid strides, Clippy/CPU/WASM; two paired surveys, Q15 pass 13 favorable / 2 inconclusive / 1 adverse; all resident/full/Q8 controls retained. `notes/2026-10-17_split_gaussian_q15_sums.md` |
| Symmetric GPU Gaussian horizontal taps | Complete | Exact shared weights for mirrored pixels in packed/unpacked horizontal stages; weighted terms 3/5/7 to 2/3/4, same borders, storage, API and transfers | 374 native tests; 3,132 quantizers, 86,465,776 raw parent/u64/guard words and 60,216,300 complete-parent/CPU bytes; Clippy/CPU/WASM; two initial and two final balanced surveys, final horizontal 22 favorable / 4 inconclusive / 0 adverse, resident 13 favorable / 13 inconclusive / 0 adverse, full 0 favorable / 24 inconclusive / 2 adverse; all 156 rows retained by phase. `notes/2026-10-17_symmetric_gpu_gaussian_horizontal.md` |
| Symmetric GPU Gaussian vertical taps | Complete | Share mirrored weights in packed Q8 and separately bounded Q15 digit sums; terms 3/5/7 to 2/3/4, exact bytes and execution contracts | 377 native tests; 50,112 arithmetic fixtures, 174,291,200 raw parent/u64 and 60,216,300 complete-parent/CPU bytes; Clippy/CPU/WASM; two balanced surveys, vertical 18 favorable / 5 inconclusive / 3 adverse, resident 13 favorable / 12 inconclusive / 1 adverse, full 4 favorable / 22 inconclusive / 0 adverse; all 78 rows retained. `notes/2026-10-17_symmetric_gpu_gaussian_vertical.md` |
| Shared GPU Gaussian Q15 rows | Complete | Two-/four-lane exact channel sums share mapped rows/weights; original Q8/mono shaders and resource/transfer contracts, lazy optional vertical caches with explicit cold-cost limit | 410 native tests, complete frozen parent/CPU/u64 raw/physical/large-stride checks and lazy-cache/resource/profiler/concurrency/CPU/WASM; affected pass 20 favorable / 0 inconclusive / 0 adverse, resident 17 favorable / 3 inconclusive / 0 adverse, full 12 favorable / 8 inconclusive / 0 adverse; all 78 final rows and 20 exploratory GPU fixtures in `notes/2026-10-18_shared_gpu_gaussian_q15_rows.md` |
| Batched GPU voxel radix controls | Complete | One aligned immutable control buffer replaces 24 source creation requests when bounded device layout fits; static bindings and original fallback, no shader/pass/payload changes, explicit padding cost | 415 native tests, all-slice GPU/parent/CPU/stability/ownership/profiler/pool/fallback/CPU/WASM checks; setup 20 favorable / 0 inconclusive / 0 adverse, resident 15 favorable / 5 inconclusive / 0 adverse, full 5 favorable / 15 inconclusive / 0 adverse; all 60 comparisons in `notes/2026-10-18_batched_gpu_voxel_radix_controls.md` |
| Reused GPU box blur windows | Complete | Same-row adjacent horizontal sums share outgoing/incoming byte loads; exact one-tap path, original row crossings/tails, policy/resources/passes/transfers | 419 native tests, complete parent/u64-prefix/brute/physical/guard/ownership/profiler/pool/CPU/WASM checks; packed pass 21 favorable / 1 inconclusive / 0 adverse, resident 15 favorable / 7 inconclusive / 0 adverse, full 5 favorable / 17 inconclusive / 0 adverse; all 78 final and 36 exploratory rows in `notes/2026-10-18_reused_gpu_box_blur_windows.md` |
| Batched GPU voxel metadata readback | Complete | One checked <=64MiB staging allocation/submit/map/wait replaces three sequential readers; exact copy bytes, original serial fallback, signed outputs and final ordering; higher staging peak | 424 native tests, parent/CPU/limit/alignment/cap-fallback/ownership/concurrency/CPU/WASM checks; 18/18 nonempty readback and 18/18 full comparisons favorable; all 40 rows including empty zero-resolution/control results in `notes/2026-10-18_batched_gpu_voxel_metadata_readback.md` |
| Parallel GPU voxel metadata keys | Complete | Exact mapped signed-key conversion uses <=4 native workers at 524,288 cells; one final vector with extra safe initialization writes, original small/WASM/serial fallback, unchanged GPU transfers | 429 native tests, signed/guard/capacity/parent/CPU/threshold/map-lifetime/concurrency/CPU/WASM checks; decode: 12 favorable / 0 inconclusive / 0 adverse / 0 unresolved; readback: 10 favorable / 0 inconclusive / 0 adverse / 0 unresolved; full: 7 favorable / 3 inconclusive / 0 adverse / 0 unresolved; all 72 final and 14 exploratory rows in `notes/2026-10-18_parallel_gpu_voxel_metadata_keys.md` |
| Shared GPU box blur vertical windows | Complete | Adjacent row windows share exact outgoing/incoming u16 samples with scalar u32 sums; one-tap loads, odd-row ownership, original host policy/resources/transfers | 434 native tests, raw-u32/u64/parent/CPU/maximum-kernel/ownership/guard/profiler/pool/CPU/WASM checks; pass: 29 favorable / 2 inconclusive / 1 adverse / 0 unresolved; resident: 18 favorable / 14 inconclusive / 0 adverse / 0 unresolved; full: 6 favorable / 26 inconclusive / 0 adverse / 0 unresolved; all 108 final and 48 exploratory rows in `notes/2026-10-18_shared_gpu_box_blur_vertical_windows.md` |
| Shared wide GPU box blur vertical windows | Complete | Adjacent row windows share exact outgoing/incoming full-width u32 samples with scalar u32 sums; one-tap loads, odd-row ownership, original host policy/resources/transfers | 439 native tests, raw-u32/u64/parent/CPU/maximum-kernel/ownership/guard/profiler/pool/CPU/WASM checks; pass: 30 favorable / 1 inconclusive / 3 adverse / 0 unresolved; resident: 15 favorable / 19 inconclusive / 0 adverse / 0 unresolved; full: 5 favorable / 27 inconclusive / 2 adverse / 0 unresolved; all 114 final and 24 exploratory rows in `notes/2026-10-18_shared_wide_gpu_box_blur_vertical_windows.md` |
| Shared wide GPU box blur horizontal windows | Complete | Exact adjacent u32 sums for >=65,536 pixels, original scalar schedule below cutoff, row-crossing fallback and unique odd tails; host policy/resources/transfers retained | 444 native tests, full-u32/u64/parent/CPU/threshold/guard/ownership/CPU/WASM checks; paired pass: 28 favorable / 0 inconclusive / 0 adverse / 0 unresolved; resident: 28 favorable / 0 inconclusive / 0 adverse / 0 unresolved; full: 22 favorable / 5 inconclusive / 1 adverse / 0 unresolved; all 126 final and 48 exploratory rows, including fallback/original controls and adverse results, in `notes/2026-10-18_shared_wide_gpu_box_blur_horizontal_windows.md` |
| Direct parallel neighborhood row decoding | Deferred after measurement | Keep direct-row candidate only in native fixtures; production retains the parent decoder after nonrepeatable full-call gains and adverse CPU controls | 61 final native executions; bitwise/error/worker-boundary/remap/concurrency/ownership/receipts/downstream/facade/CPU/WASM; all 160 native/6 exploratory comparisons in `notes/2026-10-19_direct_parallel_neighborhood_rows.md` |
| Count-only GPU neighborhood readback | Complete | Exact OR validation and bulk owned empty rows/counts for capacity-zero large readbacks; unchanged fallback, GPU work and transfers | 58 native executions; counts/bits/errors/CPU oracles/concurrency/ownership/downstream/facade/CPU/WASM; all 92 native/28 exploratory comparisons in `notes/2026-10-19_gpu_neighborhood_count_readback.md` |
| Parallel GPU neighborhood readback decoding | Complete | Bounded native CPU decode of completed maps; original row bits/counts/validation, serial fallbacks, ownership and explicit transfers | 55 native executions; raw bits, density/region/worker/cutoff/errors/concurrency/CPU oracles/downstream/facade/CPU/WASM; all 86 final/78 initial comparisons in `notes/2026-10-19_parallel_gpu_neighborhood_readback.md` |
| Capacity-specialized GPU neighborhoods | Complete | Exact nearest candidate extents 1/8/16/64; lazy nearest/radius programs, original traversal/ties/schema/owned outputs/transfers | 75 native executions; all-k parent/CPU/raw/guards/cache/errors/concurrency/downstream/facade/CPU/WASM; all 120 warm/12 fresh-cache comparisons in `notes/2026-10-19_specialized_gpu_neighborhood_capacity.md` |
| GPU box-blur column prefixes | Complete | Exact interleaved u32 scans for bounded tall unpacked columns; whole-column overflow/capability guard, seventh lazy program, original scratch/passes/receipts/fallbacks | 191 native executions; parent/u64/raw/RGBA/canaries/limits/cache/resources/concurrency/ownership, CPU/WASM; all 108 warm/12 fresh-cache comparisons in `notes/2026-10-19_gpu_box_blur_column_prefixes.md` |
| GPU box-blur row prefixes | Complete | Exact workgroup-local u32 scan for bounded capable wide rows; sixth lazy program, original scratch/passes/receipts and fallback shaders | 188 native executions; parent/u64/raw/RGBA/guards/capability/cache/resources/concurrency/ownership, CPU/WASM; all 144 final warm/10 fresh-cache and 72 exploratory comparisons in `notes/2026-10-19_gpu_box_blur_row_prefixes.md` |
| Checked CPU voxel digit-scan prefixes | Complete | Merge exact omitted-nibble/sign representatives of checked blocks with fallback suffix; repeat at most one block per worker; identical masks and GPU contracts | 282 native executions; complete parent/oracle/raw outputs, block/worker/probe edges, profiles/resources/admission/concurrency, CPU/WASM; all 144 warm/12 fresh-runtime comparisons and controls in `notes/2026-10-19_reuse_cpu_voxel_scan_prefixes.md` |
| Saturated CPU voxel digit probes | Complete | Matching probe-class and complete global masks prove the original class mask; retain one-sign/late fallback and exact GPU contracts | 462 native executions; parent/oracle/raw order, workers/probes, owned outputs/resources/errors/concurrency, CPU/WASM; saturated builds: 12 favorable / 10 inconclusive / 0 adverse / 0 unresolved; full calls: 7 favorable / 15 inconclusive / 0 adverse / 0 unresolved; all 144 warm/12 fresh-runtime comparisons and late costs in `notes/2026-10-18_saturated_cpu_voxel_digit_probe.md` |
| Sign-class CPU voxel radix digits | Complete | Exact per-sign AND/OR summaries, retained sign separation, bounded probe/four-worker merge and even owned-output parity; GPU sort/transfer contracts unchanged | 460 native tests, independent signed order/raw words, parent APIs, profiles/resources/errors/concurrency/CPU/WASM; reduced-digit builds: 35 favorable / 5 inconclusive / 0 adverse / 0 unresolved; full calls: 33 favorable / 7 inconclusive / 0 adverse / 0 unresolved; all 224 delivered/224 rejected warm and 12 fresh-runtime comparisons and CPU/control costs in `notes/2026-10-18_sign_class_gpu_voxel_radix_digits.md` |
| Constant CPU voxel radix digits | Complete | Original clamped entries, bounded varying-nibble summaries and stable selected-digit sorting; optional constant parity pass keeps owned output without a copy; GPU-key path unchanged | 456 native tests, complete parent/BTreeMap/raw-word and exact profile/resource/worker/error/concurrency/CPU/WASM checks; build targets: 30 favorable / 0 inconclusive / 0 adverse / 0 unresolved; full targets: 30 favorable / 0 inconclusive / 0 adverse / 0 unresolved; all 152 warm/8 fresh-runtime comparisons including host costs in `notes/2026-10-18_constant_cpu_voxel_radix_digits.md` |
| Identity GPU Vision chain resize | Complete | Matching dimensions borrow the source for grayscale after original resize admission; omit one copy submission/texture request, preserve source/history/saturation and report actual 5-7 passes | 185 native tests; 576 parent/CPU-exact cases, 13,076,640 tensor-word comparisons, 4096-square five-stage regression, profiler/resources/errors/concurrency/CPU/WASM; identity GPU: 18 favorable / 0 inconclusive / 0 adverse / 0 unresolved; resident: 15 favorable / 3 inconclusive / 0 adverse / 0 unresolved; full: 8 favorable / 10 inconclusive / 0 adverse / 0 unresolved; all 90 warm/8 fresh-cache comparisons in `notes/2026-10-18_identity_gpu_vision_chain_resize.md` |
| Four wide GPU box blur horizontal windows | Complete | At least 65,536 pixels and kernel widths >=4,099 use up to four disjoint adjacent outputs sharing exact full-u32 sums; row-crossing/partial-tail safety, original scalar/pair schedules, host/five-variant/resources/transfers | 181 native tests, parent/u64/CPU/guards/owners/cache/concurrency/CPU/WASM; large pass: 14 favorable / 0 inconclusive / 0 adverse / 0 unresolved; resident: 13 favorable / 1 inconclusive / 0 adverse / 0 unresolved; full: 8 favorable / 6 inconclusive / 0 adverse / 0 unresolved; all 162 warm/10 fresh-cache comparisons, 262 rejected-alternative comparisons and 52 exploratory rows in `notes/2026-10-18_four_wide_gpu_box_blur_horizontal_windows.md` |
| Short GPU 3x3 median selection | Complete | Reduce generated compare-exchange pairs from 63 to 19 with exact lane-wise integer min/max; retain identical 5x5 shader and all API/resource boundaries | All 9! rank permutations, exhaustive binary and independent sorted-byte references, frozen GPU/CPU bytes, ownership/receipts, feature/WASM checks and paired completed resident/full-call surveys; `notes/2026-10-15_gpu_median_nine_selection.md` |
| Short GPU 5x5 median selection | Complete | Odd-even merge with maximal-padding identity removal and median dependency pruning: 191 to 113 generated pairs; preserve 3x3 shader and all API/resource/transfer boundaries | All 33,554,432 binary windows, 384 parent GPU comparisons / 50,439,680 bytes, existing CPU/ROI/ownership regressions, 346 native tests and Clippy/CPU/WASM; two shared-host paired surveys favor all 18 GPU-pass rows and 15/18 resident rows, with mixed full-call results retained. `notes/2026-10-17_gpu_median_five_selection.md` |
| CPU median filtering | Complete | Bounded native output bands with per-band reusable samples; exact total-order selection for larger apertures and tiny-sort fallback | Frozen u8/u16/f32/f64 bit parity, 18,000 small conditions, threshold/pool/ROI/metadata/error checks, before/current-serial/current-parallel comparisons and GPU measurements against the improved CPU; `notes/2026-10-14_parallel_cpu_median.md` |
| GPU/CPU bilateral filtering | Complete | Explicit reusable GPU coefficients, circular neighborhoods and exact joint-color weighted byte output; bounded native generic CPU bands | Frozen CPU bits/ROI/metadata/limits/ownership checks, 3,000 GPU byte-exact conditions, 79,822 division boundaries and 14,960 raw GPU accumulator words; current-parallel-CPU/new-plan/reused-plan/resident measurements; `notes/2026-10-14_gpu_bilateral_filter.md` |
| GPU bilateral byte candidates | Complete | Normalized-word floating candidate with exact midpoint verification and bounded integer search; same weighted sums, pass, transfers and ownership | 655,023 arbitrary-candidate rows / 1,965,069 exact outputs, frozen GPU/CPU image parity, 105 native regressions and two paired new-plan/reused-plan/resident surveys with adverse cases retained; `notes/2026-10-14_gpu_bilateral_byte_candidates.md` |
| GPU Canny edge extraction | Complete | Resident integer 3/5/7 gradients/NMS and complete eight-connected atomic union/root-strong hysteresis, with owned binary output and five/six optional pass timestamps | 864 byte-exact CPU conditions, 326,401 exact signed rounding values, independent f64/graph oracles, long weak chains/ROI/metadata/receipts/concurrency/capacity and completed-call timing; `notes/2026-10-13_gpu_canny.md` |
| GPU Canny compact gradients | Complete | Pack exact signed derivatives into two 16-bit halves while retaining u32 magnitude; halve the gradient binding and remove 8*N private scratch bytes | Frozen PR 180 shader/CPU parity, exhaustive signed packing and rounding, extrema/capacity/ROI/profiler/concurrency regressions and paired completed-call measurements; `notes/2026-10-13_gpu_canny_compact_gradients.md` |
| GPU Canny compact horizontal rows | Complete | Pack unrounded derivative/smoothing pairs exactly into one u32 for 5/7 apertures; retain final rounding and remove another 4*N scratch bytes | Frozen PR 180/181 and independent horizontal/full-gradient/graph/CPU references, extrema/limits/ROI/metadata/concurrency/profiling checks and paired completed-call measurements; `notes/2026-10-13_gpu_canny_horizontal_packing.md` |
| GPU Canny shared state/root flags | Complete | Preserve low classification bits while atomically marking a weak root in bit 2; remove one 4*N buffer for all apertures, retain full parent indices and pass count | Three frozen GPU generations/CPU parity, 433,320 exact shared words with poisoned scratch, independent BFS and contended roots, four-buffer/capacity/ownership regressions and paired completed-call surveys; `notes/2026-10-13_gpu_canny_shared_state_flags.md` |
| GPU Canny row/state lifetime reuse | Complete | Store exact horizontal pairs in the later atomic state/flag binding until vertical gradients finish; remove separate rows/dummy and use 16*N private bytes with three buffers | Four frozen GPU generations/CPU parity, 4,018,560 state words and pre-overwrite row references, actual lease sizes/counts, capability/ownership regressions and paired completed-call evidence with adverse cases; `notes/2026-10-13_gpu_canny_reuse_horizontal_state.md` |
| GPU CLAHE tile scan | Complete | Workgroup-wide clipping/CDF scan with analytic residual redistribution, retaining exact CPU rounding and existing dispatch/storage limits | 1,048,576 independent bin comparisons including every residual and maximum u32 mass, full GPU regression and paired old/new clip/grid/distribution timestamps; `notes/2026-10-11_parallel_gpu_clahe_tile_scan.md` |
| GPU voxel radix prefix | Complete | Histogram-generated earlier-digit subtotals in existing offset scratch and work-efficient chunk-total tree scan | Maximum u32 prefix mass and chunk boundaries, direct million-point stable order/live outputs/pool reuse, image/point regressions and paired full-sort stage/completion measurements; `notes/2026-10-11_parallel_gpu_voxel_radix_prefix.md` |
| GPU radix word ranks | Complete | Reuse per-digit mask counts as shared transposed exclusive word prefixes after the existing cached barrier | 8,235,972 frozen/CPU intermediate words, complete signed/tied boundary sorts, dirty scratch/concurrency, image/point/downstream/facade/WASM checks and paired completed-sort measurements; `notes/2026-10-12_cached_gpu_radix_word_ranks.md` |
| GPU CLAHE padded traversal | Complete | Power-of-two row strips with short/narrow tile lane assignment and reflection restricted to padded coordinates | Independent raw-bin counts across stride/area boundaries and repeated reflection, full byte parity/regressions and paired current/previous LUT timestamps; `notes/2026-10-11_gpu_clahe_row_traversal.md` |
| GPU RANSAC plane scoring | Complete | Bounded point-direction integer count reduction, lazy per-runtime pipelines and one compute/readback-copy submission | Frozen GPU score-word parity through 1,000,003 points, CPU threshold-boundary counts, degenerate models, input/limit rejection, cold-cache concurrency/reuse, segmentation/image regressions and paired stage/completion measurements; `notes/2026-10-11_parallel_gpu_ransac_plane.md` |
| Four-way GPU RANSAC counts | Complete | Exact 256-lane integer reduction in four stages instead of eight; unchanged plane/distance math, serial controls, planner and API/resource/transfer boundaries | 366 native tests; 881 raw u64 totals per shader, 218,700 raw and 80,040 complete-parent score words, max-u32/threshold/stride/pool/ownership and segmentation checks; Clippy/CPU/WASM; two paired surveys, pass 6 favorable / 10 inconclusive / 0 adverse, full API 2 favorable / 14 inconclusive / 0 adverse; every condition retained. `notes/2026-10-17_four_way_gpu_ransac_counts.md` |
| Four-way GPU voxel scan | Complete | Exact four-way u32 scan tree, sixteen to eight tree barriers; preserve hierarchy, block offsets, passes/scratch/transfers and complete compactor | 369 native tests; 12,058,630 raw and 12,011,620 hierarchy parent/u64 words, 21,890,424 complete compactor segment values; max-u32/tails/zero/ownership, Clippy/CPU/WASM; three paired surveys, scan 24 favorable / 0 inconclusive / 0 adverse, resident 1 favorable / 22 inconclusive / 1 adverse, full 4 favorable / 19 inconclusive / 1 adverse; every result retained. `notes/2026-10-17_four_way_gpu_voxel_scan.md` |
| Four-way GPU radix prefix | Deferred after measurement | Exact four-way cached vec2<u32> candidate retained only in native fixtures; production binary scan/threshold/stability/resources/transfers unchanged | 371 final native tests; 51,565,932 raw parent/production/candidate/u64 words and 225,286,624 complete stable sort/retained words; Clippy/CPU/WASM; all 144 rows from two initial and one final fixture survey retained, including repeated 2,000,003-point GPU regressions and no established upload/sort/readback benefit. `notes/2026-10-17_four_way_gpu_radix_prefix.md` |
| GPU separable box blur | Complete | Exact unrounded u32 horizontal/vertical sums for large work, bounded direct traversal for small work, validated arithmetic/device limits and six/seven-stage Vision profiling | Frozen full-RGBA/u64-oracle parity, rounding/ROI/large-border cases, maximum pixel and limit checks, concurrency/reuse, 122 native regressions and paired resident blur measurements; `notes/2026-10-11_separable_gpu_box_blur.md` |
| Packed GPU box-blur scratch | Complete | Pair exact u16 horizontal sums through width 257, halving even-pixel logical requests with unchanged direct/wide policy and final rounding | 218 image/voxel/facade regressions; 7,414,036 raw parent/CPU words and 14,471,193 complete bytes; width/limits/guards/odd tails/profile/cache/ownership, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_box_blur_scratch.md` |
| GPU separable rectangular morphology | Complete | Exact horizontal/vertical extrema with replicate borders, submitted pooled textures, validated device/coordinate limits and six-to-eight-stage Vision profiling | 560 frozen/full-RGBA/oracle cases, ROI/ownership/cache and device/selection checks, 128 native regressions and 38 paired resident measurements; `notes/2026-10-11_separable_gpu_morphology.md` |
| Packed GPU morphology intermediate | Complete | Four horizontal extrema per RGBA texel with exact row tails, actual scratch/pool extents and unchanged direct/single-column controls | 228 native image/voxel/facade regressions; 2,171 frozen planner, 19,203,084 raw horizontal and 118,835,784 public/physical byte checks; guards/receipts/concurrency/ownership, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_morphology_scratch.md` |
| Standalone GPU point estimator reuse | Complete | Lazy per-runtime normal/covariance pipelines, bounded storage leases, one compute/readback-copy submission and pre-allocation input/device validation | 416 frozen cases / 464,896 exact f32 words, ownership/reuse/limit checks, 132 native regressions and twelve paired full-API measurements; `notes/2026-10-11_reused_gpu_point_estimators.md` |
| GPU grid row traversal | Complete | Merge adjacent dense-grid x cells into ordered contiguous intervals for radius normal/covariance gathers | 486 frozen cases / 1,702,944 f32 words, 295,520 independent ordered-neighbor fingerprint words, native/facade/feature/WASM checks and paired full-API measurements; `notes/2026-10-12_gpu_grid_row_traversal.md` |
| GPU component union/find | Complete | Actual device minimum-root connection/compression on resident sparse cells, root-only payload return and accurate GPU phase receipts | 335,276 frozen/independent/analytic GPU root words, long chains/dirty scratch/concurrency, six attribute types/size-filter/receipt/facade/WASM checks and 24 cyclic-triple condition records; `notes/2026-10-12_gpu_cluster_union.md` |
| Complete GPU covariance outputs | Complete | Explicit zero cross terms/padding on the insufficient-neighbor return, preserving the epsilon diagonal and supported solver | Failing raw/public dirty-output reproductions, 266,112 frozen/current raw and canary words, forced pool reuse, four concurrent callers, natural plane-to-sparse facade reuse and native/feature/WASM checks; `notes/2026-10-12_complete_gpu_covariance_outputs.md` |
| CPU dense-grid row traversal | Complete | Stable adjacent-x CSR intervals and clipped valid y/z rows shared by serial and existing parallel component labeling | 5,976 independent/frozen ordered queries, 56,994 neighbor indices, brute-force roots, dispatch boundaries/concurrency, minimal/facade/WASM checks and two paired completed-API surveys; `notes/2026-10-12_cpu_grid_row_traversal.md` |
| Reduced CPU grid scratch | Complete | Stable reverse scattering reuses cumulative ends as final CSR starts without a cell-sized cursor clone | 2,316,018 independent/frozen CSR words, 520,344 exact GPU output words, concurrency/nonfinite mapping, CPU/image/point/facade/WASM regressions and paired grid/public-GPU-API measurements; `notes/2026-10-12_reduced_cpu_grid_scratch.md` |
| Checked dense-grid bounds | Complete | Valid axes/scales and checked span/product planning before u32 narrowing and downstream grid/GPU work | Failing wrapped-span/panic/GPU-call reproductions, 300 exact frozen bounds, cap/reciprocal/empty/infinite-radius boundaries, 92 cold/warm GPU rejections and debug/release/image/point/facade/WASM checks; `notes/2026-10-12_validated_grid_bounds.md` |
| Bounded CPU grid prefixes | Complete | Scoped native inclusive scans for large sparse cell tables, worker-sized metadata and serial small/dense/WASM paths shared by GPU preparation | 672,414,960 independent prefix comparisons, 2,237,056 exact GPU output words, concurrency/feature/facade/WASM checks and 42 paired completed-call conditions; `notes/2026-10-12_parallel_grid_prefix.md` |
| GPU split-tile CLAHE histograms | Complete | Bounded independent partial histograms for large low-count tile grids, lazy per-runtime pipelines, exact merge and adaptive three/four-pass profiling | 4,030,464 raw partial bins / 73,728 count-scan-LUT words, ownership/receipt/capacity checks, 136 image/point regressions and 48 paired completed resident conditions; `notes/2026-10-11_split_gpu_clahe_histograms.md` |
| Bounded CPU neighborhood graphs | Complete | Explicit parallel k-NN/radius graph APIs with one shared tree, ordered edge joins and existing bounded chunk dispatch | Frozen edge-order parity across duplicate/tie and small/threshold/partial cases with 1/2/6 workers, independent membership/oversized k/index tests, feature/facade/WASM checks and 24 completed-API comparisons; `notes/2026-10-12_parallel_neighbor_graphs.md` |
| GPU exact geometry ratios | Complete | Exact exponent scaling for power-of-two tile sizes, preserving all general ratio/interpolation math | 332,917 ratio/axis cases against independent CPU and frozen GPU words, 137 image/point regressions, feature/WASM checks and 24 paired completed resident conditions; `notes/2026-10-12_exact_gpu_clahe_geometry.md` |
| Bounded CPU MLS smoothing | Complete | Independent point fits on shared KD-tree with existing bounded staging and disjoint ordered output bands | Frozen exact-coordinate parity with 1/2/6 workers, curved/duplicate/degenerate/support cases, all attribute types and metadata, feature/facade/WASM checks and twelve completed-position-API comparisons; `notes/2026-10-12_parallel_mls_smoothing.md` |
| Bounded CPU FPFH descriptors | Complete | Shared KD-tree, scoped SPFH/weighting phases and direct disjoint descriptor writes using existing point staging | 18,402,648 frozen exact descriptor words with 1/2/6 workers and public staging, independent planar/degenerate cases, validation/concurrency/seeded alignment, feature/facade/WASM checks and twelve completed-descriptor-API comparisons; `notes/2026-10-12_parallel_fpfh_descriptors.md` |
| Bounded CPU boundary detection | Complete | Shared KD-tree, reusable worker angle scratch and direct disjoint mask writes using existing point staging | 1,081,992 frozen exact decisions with 1/2/6 workers and public staging, independent complete rim/degenerate/normal cases, all attribute types and metadata, validation/concurrency, feature/facade/WASM checks and twelve completed-mask-API comparisons; `notes/2026-10-12_parallel_boundary_detection.md` |
| Bounded CPU ISS keypoints | Complete | Shared KD-tree, scoped covariance/selection phases and ordered selection with existing point staging | 1,081,992 frozen exact selection decisions and selected clouds with 1/2/6 workers and public staging, independent covariance/tied maxima/degenerate support, all attribute types and metadata, validation/concurrency, feature/facade/WASM checks and twelve completed-detection-API comparisons; `notes/2026-10-12_parallel_iss_keypoints.md` |
| Bounded CPU cloud distances | Complete | Shared target trees, reusable worker-bounded score windows and original-order f64 accumulation in both directions | 660 frozen exact metric words with 1/2/6 workers and public staging, independent asymmetric metrics/ties, multi-window order-sensitive sums, NaN/None/infinity and empty behavior, renamed schema/concurrency, feature/facade/WASM checks and twelve completed-distance-API comparisons; `notes/2026-10-12_parallel_cloud_distances.md` |
| Exact KD-tree minimum pivots | Complete | Reuse axis minima and retained-range lower bounds to skip no-match partition scans with original pivot swaps | 191,916 frozen selection permutations, 117 exact tree layouts and 11,700 ordered query results, independent 65,537 coincident points, concurrency, 115 downstream regressions, feature/facade/WASM checks and sixteen paired construction measurements; `notes/2026-10-12_exact_kdtree_minimum_pivots.md` |
| Allocation-free scalar KD-tree search | Complete | Stack-resident optional neighbor with original k=1 traversal, rejection and pruning semantics | 410,151 exhaustive nonfinite queries, 18,486 frozen/buffered comparisons, independent tie/NaN/overflow expectations, shared-tree concurrency, downstream/feature/facade/WASM checks and twelve paired scalar-query measurements; `notes/2026-10-12_allocation_free_nearest_one.md` |
| GPU spatial execution guards | Complete | Pre-allocation runtime/dimension/device/profiler checks for nearest resize and Sobel, including checked resize coordinate products | CPU/frozen full-RGBA parity, invalid-limit/capacity and ROI/concurrent/live-output/pool checks, image/point regressions and feature/WASM builds; `notes/2026-10-12_gpu_spatial_dispatch_validation.md` |
| Bounded GPU gray dispatch | Complete | Balanced multi-row flat workgroups beyond the single-axis limit, independently lazy kernels and pre-allocation arithmetic/device/profiler validation | All 16,777,216 RGB values, 120 forced grid mappings, maximum-counter device probes, ownership/capacity/ROI/concurrency/reuse checks, native regressions and resident timing; `notes/2026-10-12_bounded_gpu_gray_dispatch.md` |
| GPU AI execution guards | Complete | Pre-allocation image/index/buffer/dispatch and profiler validation while preserving the original CHW shader | 23,857,560 frozen output words across 192 cases, ROI/concurrency/live tensor reuse, negative-limit/capacity checks, 4096x4096 six-stage chain and feature/native/WASM validation; `notes/2026-10-12_validated_gpu_ai_packing.md` |
| GPU image transfer guards | Complete | Pre-allocation texture/row/host-capacity and padded staging-buffer validation with explicit transfer accounting | 12,300 independent layouts, exact arithmetic/device/buffer boundaries, 120 full-RGBA packed/strided/ROI cases, rejection scopes/receipts/pools, concurrency/reuse/live outputs and native/feature/facade/WASM checks; `notes/2026-10-12_validated_gpu_image_transfers.md` |
| Bounded host image transfer rows | Complete | Native disjoint row bands for large logical-channel transforms and scalar RGBA row copies within explicit GPU uploads/readbacks | 838,120,764 frozen/oracle CPU byte comparisons, 69,574,296 native GPU bytes, concurrency/metadata/receipt/facade/WASM checks and 96 paired condition records across two isolated surveys; `notes/2026-10-12_parallel_image_transfer_rows.md` |
| Borrowed packed RGBA uploads | Complete | Exact logical packed CPU slice capability and borrowed four-channel payload, removing the redundant 4*width*height-byte framework host vector/copy | 13 model / 110 native image / 4 facade tests, 117,638,192 physical/logical byte comparisons, frozen complete uploader and lifetime/concurrency checks; isolated latency unavailable due competing jobs, no speed claim; `notes/2026-10-15_borrow_packed_rgba_upload.md` |
| Direct mapped image readback | Complete | Unpack channels 1..=4 from borrowed mapped staging into owned final images; remove the padded host intermediate and release mappings after workers/unwind | 352,479,568 bytes across 48 fixtures, panic/remap/concurrency/ownership checks, 113 native and 4 facade tests, feature/WASM checks; two isolated completed-readback surveys retain all 16 rows (12 favorable, 4 inconclusive); `notes/2026-10-15_direct_mapped_image_readback.md` |
| Direct mapped AI tensor readback | Complete | Decode native-endian CHW floats from borrowed mapped staging into the owned vector, removing its byte intermediate with the existing map lifetime guard | 10,570,960 raw float-word and 663,060 analytic CHW comparisons, 116 native and 4 facade tests, receipt/ownership/concurrency and feature/WASM checks; one isolated eight-condition survey (7 favorable, 1 inconclusive), repeat unavailable due competing work; `notes/2026-10-15_direct_mapped_ai_readback.md` |
| Direct mapped histogram/CDF readback | Complete | Decode mapped staging directly into owned 256-count arrays, removing a 1 KiB intermediate vector per native readback; keep LE integers and receipts | 335 native tests; 81,920 raw original/CPU, 153,600 public histogram/CDF and 6,144 concurrent retained words; full u32 range, views/receipts/rejection/ownership/map-lifetime checks, Clippy/CPU/WASM; latency unmeasured. `notes/2026-10-16_direct_mapped_histogram_readback.md` |
| Direct mapped voxel XYZ readback | Complete | Direct mapped XYZ outputs for resident-input gather/centroid calls; remove 8*N host copy bytes and excess returned X capacity | 54 native GPU/filter tests, 2,951,604 raw and 3,553,848 public float-word comparisons, panic/remap/ownership/concurrency, Clippy/CPU/WASM checks; 2 isolated survey(s), 32 rows: 16 favorable, 2 adverse, 14 inconclusive; `notes/2026-10-15_direct_mapped_voxel_xyz.md` |
| Direct mapped voxel channel readback | Complete | Direct owned f32 outputs for standalone multi-channel gather/average, retaining two/four-channel batching | 58 GPU/filter tests; 27,526,800 raw and 13,425,648 public float-word comparisons; validation/ownership/concurrency/panic-remap, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim; `notes/2026-10-15_direct_mapped_voxel_channels.md` |
| Direct mapped combined voxel readback | Complete | Direct owned XYZ/f32/u8 outputs for all four combined gather/average modes; remove excess X capacity and intermediate byte arrays | 63 GPU/filter regressions; 48,171,900 raw and 77,394,912 public element comparisons, million-cell capacities, validation/ownership/panic-remap/concurrency; Clippy/CPU/WASM; deterministic copy/allocation evidence, no latency claim; `notes/2026-10-15_direct_mapped_voxel_mixed.md` |
| Packed voxel u8 output | Complete | Four-cell GPU output words for all combined XYZ/u8 gather/average modes; 75% less byte download for multiples of four, with unchanged per-cell math | 67 GPU/filter regressions; 417,504 kernel cell results including 59,640 half ties, all byte codes, 24,964,632 mapped and 124,868,464 parent/CPU elements; Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim; `notes/2026-10-15_packed_voxel_u8_output.md` |
| Fused voxel float average | Complete | Capability/size-gated XYZ plus one-to-four float averages in one GPU pass; preserve scalar division, XYZ inverse and separate fallback | 71 voxel/filter regressions; 1,481,722 kernel outputs, 238,534,500 complete parent/CPU elements, planner boundaries, non-power-of-two math, guards/NaNs/concurrency; Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim; `notes/2026-10-15_fused_voxel_float_average.md` |
| Fused voxel first-point average | Complete | One capability/size-gated pass for first-point XYZ and one-to-four float averages, preserving point-zero empty-cell coordinates and separate fallback | 74 voxel/filter regressions; 1,481,722 kernel outputs, 238,534,500 complete parent/CPU elements, coordinate bits/NaNs/guards, scalar division, fallback/concurrency; Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-15_fused_voxel_first_average.md` |
| Fused voxel centroid and first attributes | Complete | One capability/size-gated pass preserving centroid math, scalar/multi empty rules, pooled uploads and fallback; remove dummy outputs | 78 voxel/filter regressions; 1,481,722 kernel outputs, 238,534,500 complete parent/CPU elements, bit/NaN/guard/pool/concurrency boundaries; Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-15_fused_voxel_centroid_first.md` |
| Broader fused first-point voxel gathering | Complete | Word-exact XYZ/attribute gathering for one-to-three floats and mixed byte calls; preserve original four-float/no-byte path and split fallback | 83 voxel/filter regressions; 1,481,722 kernel words, 238,698,596 parent/CPU elements, sparse/empty/NaN/guard/pool/concurrency checks; Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-15_fused_voxel_first_gather.md` |
| Batched GPU voxel byte attributes | Complete | Two-to-four packed first/average byte channels share one pass/copy with bounded lazy kernels and original fallback/rounding/payload | 88 voxel/filter regressions; 1,514,664 direct cell results, 731,120,768 parent/CPU elements; limits/ties/tails/guards/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-15_batched_voxel_u8_attributes.md` |
| Coalesced voxel gather submissions | Complete | At most two existing two/four-channel batches per submit/map/wait, with checked staging and original fallback | 93 voxel/filter regressions; 72,623,494 public, 116,850 sparse and 76,586 forced-width parent/CPU words; limits/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-15_coalesced_voxel_gather_submit.md` |
| Batched standalone voxel float averages | Complete | Share index traversal and planar f32 output in groups of four, with original sequential math, scalar tails and fallback | 98 voxel/filter regressions; 1,818,481 kernel and 52,927,918 public parent/CPU words; limits/nonfinite/order/guards/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-15_batched_voxel_float_average.md` |
| Batched combined voxel attribute averages | Complete | Reuse four-channel averages in separate-XYZ branches, with original fused priority and checked staging regions | 103 voxel/filter regressions; 4,343,790 original GPU words and 525,631,692 public parent/CPU elements; offsets/guards/nonfinite/limits/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_combined_voxel_attribute_averages.md` |
| Packed combined voxel first attributes | Complete | Preserve original two/four-wide batches with one output/copy per batch, scalar-tail empty rules, pooled uploads and checked fallback | 108 voxel/filter regressions; 17,375,160 parent/CPU words and 525,631,692 complete elements; limits/guards/nonfinite/pools/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_combined_voxel_gather.md` |
| Packed standalone GPU voxel gathering | Complete | One planar output/copy per original two/four-wide batch; preserve bounded windows, single-tail point-zero bits, unpooled uploads and checked fallback | 241 native GPU/filter/facade regressions; 72,623,494 public, 137,350 sparse and 9,080,124 forced-width parent/CPU words; limits/guards/nonfinite/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_standalone_voxel_gather.md` |
| Packed resident GPU voxel XYZ outputs | Complete | One planar output/copy for standalone first XYZ and centroid calls, with original sequential math, owned capacities and full fallback | 251 native GPU/filter/facade regressions; 23,649,858 public parent/CPU/repeated and 789,576 raw words; limits/guards/nonfinite/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_resident_voxel_xyz.md` |
| Packed combined GPU voxel XYZ regions | Complete | Reuse packed XYZ outputs inside all four combined attribute APIs, with original fused priority and append-only target guards | 256 native GPU/filter/facade regressions; 178,334,384 public parent/CPU and 789,576 recorder words; limits/guards/nonfinite/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_packed_combined_voxel_xyz.md` |
| Wide GPU voxel average fusion | Complete | Reuse fused XYZ/first-four attributes with original remainder batches and full checked fallback | 261 native GPU/filter/facade tests; 188,946,314 public parent/CPU elements, 7,237,780 recorder words, limits/nonfinite/byte-tail/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_wide_voxel_average_fusion.md` |
| Wide GPU voxel first-attribute fusion | Complete | Reuse fused XYZ/first-four attributes with unchanged original-width tail gathering and pooled uploads | 266 native tests; 188,946,314 public parent/CPU elements, 14,475,560 direct recorder words; raw bits/scalar tails/limits/recycling/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_wide_voxel_first_fusion.md` |
| Bounded CPU voxel attribute readback | Complete | Scoped native copies of large mapped float planes, at most four workers with original small/single/WASM serial path | 271 native tests; 104,964,370 full current-parent/CPU elements, 75,497,946 mapped and 116,392,193 host words; worker/host/cutoff bounds, raw bits/capacity/unmap/ownership/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_parallel_voxel_attribute_readback.md` |
| Bounded CPU standalone voxel channel readback | Complete | Reuse scoped native plane copies per large mapped window for first-value/average APIs; at most four workers with original small/single/WASM serial path | 276 native tests; 159,391,128 full current-parent/CPU, 25,166,074 mapped and 58,720,312 forced-width elements; cutoff, sparse nonfinite tails, exact capacities, panic/remap, concurrent owned lifetimes, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_parallel_standalone_voxel_readback.md` |
| Borrowed voxel u8 upload payloads | Complete | Borrow host channels with lengths divisible by four through all scalar/batched upload sites; preserve original padded tails and complete GPU work | 281 native tests; 198,220,200 full current-parent/CPU elements, 64,512,096 host and 4,458,040 GPU upload byte checks; all remainders/odd addresses, snapshot/source mutation, sparse rounding, capacity/validation/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_borrowed_voxel_u8_uploads.md` |
| Bounded CPU standalone voxel XYZ readback | Complete | At most three scoped native copy workers including caller, shared large-input policy and original serial small/WASM path | 286 native tests; 63,704,304 complete current-parent/CPU, 94,372,164 host, 18,874,470 mapped and 12,582,924 packed/split GPU words; forced workers/cutoff/raw bits/capacity/panic-remap/source mutation/concurrency, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_parallel_voxel_xyz_readback.md` |
| Bounded CPU combined voxel XYZ readback | Complete | Reuse existing XYZ copier in all four combined APIs after original attribute/byte phases; at most three XYZ workers including caller | 290 native tests; 447,244,968 complete current-parent/CPU and 113,246,480 mapped elements; cutoff/zero/single/wide attributes, byte tails, exact capacities, panic-remap/error order/concurrent host and GPU mutation, Clippy/CPU/WASM; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_parallel_combined_voxel_xyz_readback.md` |
| Bounded CPU packed voxel byte readback | Complete | At least four MiB per u8 channel; reuse checked plane planner with at most four scoped children after float copies and before XYZ | 296 native tests; 1,512,598,448 complete current-parent/CPU, 377,487,702 mapped and 369,099,054 host elements; forced workers/all tails/exact capacities/overflow/remap/error order/concurrency and CPU/WASM checks; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_parallel_voxel_u8_readback.md` |
| Reserved voxel byte upload padding | Complete | Reserve exact padded capacity before copy/zero extension; original aligned borrow and GPU work remain | 301 native tests; 340,826,888 complete parent/CPU, 181,952,768 host and 4,458,040 captured GPU upload elements; all remainders/offsets/capacities/source mutation/validation/concurrency and CPU/WASM; seven million-point channel preparations avoid 7,340,018 summed capacity bytes; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_reserved_voxel_u8_padding.md` |
| Coalesced host voxel XYZ centroids | Complete | Share segment uploads and one scratch output across three original scalar passes; one submit/map/wait with checked original fallback | 306 native tests; 19,272,306 complete parent/f64 and 11,565 sparse/nonfinite words; capability/count/staging bounds, error/ownership/concurrency and CPU/WASM; source ledger removes 10 MiB H2D for 1,048,576 points / 262,144 cells; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_coalesced_host_voxel_xyz.md` |
| Reused voxel average outputs | Complete | One private output per standalone or combined attribute recorder; active batch copied before overwrite, original fused wide prefix retained | 310 native tests; 496,273,824 complete parent/CPU and 15,005,136 raw parent words; all partial widths, staging guards/sparse/nonfinite/error/capacity/concurrency and CPU/WASM; 17-channel / 262,144-cell standalone output requests 5 to 1 and 13 MiB fewer summed descriptor bytes; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_reused_voxel_average_output.md` |
| Reused voxel first-attribute outputs | Complete | Private output reuse per bounded standalone window or combined packed tail; original widths, scalar empty semantics and upload lifetimes retained | 314 native tests; 496,273,824 complete public parent/CPU, 47,647,888 raw and 30,010,272 forced-width words; partial widths/guards/nonfinite/errors/capacities/concurrency and CPU/WASM; 17-channel / 262,144-cell four-wide standalone output requests 5 to 3 with 8 MiB fewer summed descriptor bytes; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_reused_voxel_first_output.md` |
| Wide GPU voxel byte batches | Complete | Group more than four byte attributes with existing four-channel shaders; one private output, scalar single tails and checked original fallback | 319 native tests; 672,578,280 complete parent/CPU elements, 27,596,192 raw and 655,369 rounding/long-cell bytes; 146 capability/host/index/offset plans, all packed tails, ordered byte-pair ties, owned concurrency and CPU/WASM; 17 channels reduce byte passes 17 to 5 and output requests 17 to 1; Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_wide_voxel_u8_batches.md` |
| Initialized image compute uniforms | Complete | Initialize private resize/Sobel and CHW parameter buffers during creation; preserve dispatch and payload accounting | 320 native tests, including 96 queued image chains / 4,005,210 CPU-exact f32 words; metadata/receipts/owned lifetimes, existing profiler/error/ROI cases, Clippy/CPU/WASM; three explicit queue writes per resize-Sobel-CHW chain become zero; latency unmeasured. `notes/2026-10-16_initialized_image_uniforms.md` |
| Unwind-safe scalar GPU readback | Complete | Reuse existing mapped-view cleanup guard so scalar prefix panics permit later maps/uploads; preserve successful copy/error behavior | 323 native tests; 6,685,002 exact word comparisons, frozen-parent map retention proof, three panic/reuse cycles and 524,500 words after four concurrent buffers/runtime drop; Clippy/CPU/WASM, no new GPU work or performance claim. `notes/2026-10-16_guarded_scalar_gpu_readback.md` |
| Reused legacy voxel gather outputs | Complete | Reuse a private output array per original two-batch fallback window; keep packed priority and original tail kernels/copies | 326 native tests; 6,108,264 guarded raw, 102,322,584 complete fallback and 688,238 concurrent owned words; forced widths, packed rejection, CPU/WASM and 8 additional conditions; 8 to 4 output requests for two full four-channel batches. Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_reused_legacy_voxel_outputs.md` |
| Trimmed combined voxel gather dummies | Complete | Allocate only absent output slots in existing three/four-channel fallback blocks; retain packed/fused priority and copies | 329 native tests; 5,180,820 guarded raw, 122,988,675 batch and 557,278 concurrent owned words; 8 additional fixture conditions, Clippy/CPU/WASM; three-channel dummy requests 4 to 1, four-channel 4 to 0. Isolated latency unavailable due competing work; no speed claim. `notes/2026-10-16_trimmed_voxel_gather_dummies.md` |
| Reused combined voxel gather outputs | Complete | Reuse the active output prefix across original fallback batches; preserve packed priority, dummy bindings, tails and copy order | 332 native tests; 19,406,168 guarded raw, 285,334,110 recorder/CPU and 803,088 concurrent owned words; Clippy/CPU/WASM; four-wide 19-attribute output requests 20 to 5; latency unmeasured. `notes/2026-10-16_reused_combined_voxel_outputs.md` |
| Reused packed voxel gather controls | Complete | Per-call/window immutable controls keyed by all original words; preserve uniform data, batch widths, tails and copies | 336 native tests; 3,055,494 guarded queued original/CPU words across 64 contexts / 128 passes, existing forced-width/public/concurrency regressions, Clippy/CPU/WASM; four-wide combined 19-attribute uniform requests 5 to 2; latency unmeasured. `notes/2026-10-17_reused_packed_gather_controls.md` |
| Reused voxel average controls | Complete | Per-invocation immutable controls keyed by all original words; preserve f32 math, scalar tails and copies | 337 native tests; 3,055,494 guarded queued words across 64 contexts / 128 passes against frozen original GPU and CPU, complete public/concurrency regressions, Clippy/CPU/WASM; four-wide recorder receiving 19 attributes uses 2 rather than 5 uniform requests; outer fused controls retained, latency unmeasured. `notes/2026-10-17_reused_voxel_average_controls.md` |
| Reused voxel u8 controls | Complete | Per-wide-recorder immutable controls keyed by all original words; preserve first/average pipelines, rounding, packed/scalar tails and copies | 338 native tests; 1,532,792 guarded original/CPU words across 128 contexts / 256 queued passes, existing all-pairs/long-sum/public regressions, Clippy/CPU/WASM; 19-byte-channel uniform requests 5 to 2, 20-channel requests 5 to 1; latency unmeasured. `notes/2026-10-17_reused_voxel_u8_controls.md` |
| Reused legacy voxel gather controls | Complete | Per-window immutable controls keyed by original words and descriptor label; retain preferred packed priority, original widths/tails and copies | 339 native tests; 6,379,488 guarded original/CPU words across 96 contexts / 192 queued passes, forced fallback/public/concurrency regressions, Clippy/CPU/WASM; two full batches request 1 rather than 2 controls, four-wide 19-channel fallback 5 to 3; latency unmeasured. `notes/2026-10-17_reused_legacy_gather_controls.md` |
| Contiguous RGBA readback copies | Complete | One owned prefix copy for unpadded four-channel mapped rows; preserve other layouts, lifetime and transfer receipts | 341 native tests; 64,564,840 original/CPU host bytes and 50,499,904 complete RGBA bytes, owned capacities/ROI/metadata/recycle, Clippy/CPU/WASM; paired host/completed-call diagnostics across five shapes, shared-host conditions retained. `notes/2026-10-17_contiguous_rgba_readback.md` |
| Owned RGBA transfer rows | Deferred after measurement | Safe reserved-row append candidate retained only in native fixtures; production packing, padded readback and earlier borrowing/contiguous paths remain | 343 native tests; 328,948,048 original/CPU host bytes and 100,999,808 candidate/production RGBA bytes; Clippy/CPU/WASM; three complete paired diagnostic runs including slower cases do not establish a latency benefit. `notes/2026-10-17_owned_rgba_row_copies.md` |
| Replicated GPU histogram counters | Complete | Eight padded workgroup-local bin replicas for large inputs with supported local storage, retaining original small/limited-device path | 307,200 frozen/CPU/public bin comparisons, maximum u32 mass, exact selection boundaries, large RGBA/concurrent lazy cache/two-pass receipts, native/feature/facade/WASM checks and paired completed-call measurements; `notes/2026-10-12_replicated_gpu_histograms.md` |
| GPU histogram run compression | Deferred after measurement | Keep per-lane equal-value accumulation only in native fixtures | Exact frozen/CPU counts; gains depend on geometry/distribution and do not provide consistent completion improvement; `notes/2026-10-12_replicated_gpu_histograms.md` |
| GPU AI channel kernels | Deferred after measurement | Retain channel-lane and fixed-channel prototypes only in native fixtures | Lane kernels regressed across all 12 selected GPU conditions; fixed-channel micro gains did not establish completed-call improvement, with adverse cases retained; `notes/2026-10-12_validated_gpu_ai_packing.md` |
| GPU shared Sobel tiles | Deferred after repeat measurement | Preserve the direct production shader while retaining an exact shared-halo prototype in native tests | 54 CPU/frozen/direct/experimental fixtures; initial gains did not survive repeat measurement across 36 conditions, with adverse wide/odd strips; `notes/2026-10-12_gpu_spatial_dispatch_validation.md` |
| GPU contrast geometry reuse | Deferred after measurement | Keep per-call device geometry and existing bounded scratch while shortening exact ratios | A 24-condition survey found geometry below 1% of GPU pass time for 3840x2160 / 8x8 mixed and hot fixtures; revisit retention if repeated geometry becomes a material completed-call cost; `notes/2026-10-12_exact_gpu_clahe_geometry.md` |

## Remaining major GPU algorithm phases

These phases extend actual device computation rather than CPU parallelism.
Dependencies and explicit transfers come before broader algorithm coverage.

| Phase | Status | Dependency and deliverable | Acceptance |
| --- | --- | --- | --- |
| ICP correspondences | Complete | Reusable runtime-owned GPU KD-tree and explicit batched nearest uploads/readback; opt-in ICP update/fitness integration with CPU solver | CPU/brute nearest ties and gates, ownership/reuse/capacity, pose/trace/receipt checks and completed query/registration comparisons; `notes/2026-10-12_gpu_icp_correspondences.md` |
| GICP correspondences | Complete | Reuse GPU target for unbounded matching followed by host gating; explicit radius covariance runtime and full receipt with CPU Hessian/pose solver | Pose/stopping/gating/receipt/unequal-cloud checks, error propagation, completed comparisons for CPU/radius-GPU covariance; `notes/2026-10-12_gpu_gicp_correspondences.md` |
| Feature neighborhoods | Resident API complete; FPFH/ISS/MLS consumers added | Reused target/query buffers, GPU k-NN and bounded radius rows with exact counts/overflow, explicit readback and raw storage for device consumers | CPU/brute membership/order, degenerate/tied/partial/ownership/capacity cases, device consumer test and completed/resident measurements; `notes/2026-10-12_gpu_bounded_neighborhoods.md` |
| FPFH descriptors | Complete | Input-bound reusable radius rows, GPU SPFH and weighted FPFH, explicit upload/readback with overflow status; CPU matching/RANSAC retained | CPU/analytic descriptor comparisons, input/ownership/overflow/error guards, native/facade/WASM builds and completed comparisons against current parallel CPU; `notes/2026-10-12_gpu_fpfh_descriptors.md` |
| ISS keypoints | Complete | Shared original queries/positions, input-bound radius graphs, GPU covariance/eigenvalue saliency and local maxima; explicit flag/status download, CPU compaction/attributes | Analytic eigen/tie checks, f64 math/error-band and independent NMS references, schema/attribute bits, ownership/overflow/underflow guards and completed parallel-CPU comparisons; `notes/2026-10-13_gpu_iss_keypoints.md` |
| MLS compute | Complete | Input-bound reusable radius rows, GPU weighted frames and order-0/1/2 fits, explicit position/mode/status download; CPU tree and reconstruction | Analytic quadratic/noise checks, per-mode CPU references with raw requested-order errors retained, schema/attribute bits, ownership/overflow/underflow guards and completed parallel-CPU comparisons; `notes/2026-10-13_gpu_mls_smoothing.md` |
| ICP resident transforms/reductions | Complete | Reusable source/target/tree, GPU absolute transforms and bounded matching, centered centroid/covariance/fitness reductions; fixed moments readback with CPU Horn solve | Independent f64 moments, 54 pose/scale/unequal-cloud conditions, receipt/overflow/ownership/concurrency checks and completed CPU/previous-GPU comparisons; `notes/2026-10-13_gpu_icp_reduction.md` |
| GICP resident normal equations | Complete | Reusable positions/covariances, GPU inverse weighting and Hessian/gradient/Euclidean reductions; fixed 256-byte equations with CPU normalized solve/world conversion | Independent f64 inverse/Jacobian/full equations, 108 pose/scale/covariance/stopping cases, strict numeric/ownership/receipt guards and completed CPU/previous-GPU comparisons; `notes/2026-10-13_gpu_gicp_reduction.md` |
| GICP working-set and CPU preparation | Complete | Reuse transformed rows for centered positions/counts, 88 to 72 private bytes per source and single-store pairs; bounded direct packed CPU k-NN preparation for the GPU entry point | 747 exact frozen equation conditions, 90 GPU covariance/status cases, 1,108,059 exact CPU rows, 594 registration accuracy conditions, paired completed/public-call measurements and prior full-call phase diagnostics; `notes/2026-10-13_gpu_gicp_reuse_transformed_queries.md` |
| GICP explicit covariance upload staging | Complete | Fuse exact f64 SPD validation with original-order padded f32 packing, using bounded native workers and serial small/WASM staging | Frozen row/scale/error parity, independent numeric and chunk boundaries, uploaded equation/receipt/ownership parity, registration accuracy, feature/facade/WASM checks and isolated completed-call comparisons; `notes/2026-10-14_parallel_gicp_upload_preparation.md` |
| Shared GPU registration geometry staging | Complete | Bounded native finite-XYZ/bounds summaries with original f64 nonzero anchors/extents and canonical private zero anchors; small/WASM serial scans | Frozen nonzero geometry/extent bits, explicit zero signs and worker/error boundaries, uploaded moment/receipt/ownership parity, ICP/GICP/plane accuracy and isolated completed-call measurements; `notes/2026-10-14_parallel_gpu_registration_geometry.md` |
| Direct SoA registration target staging | Complete | Prepare target bounds and padded original-order GPU queries from borrowed KD-tree SoA coordinates, removing the intermediate 12*T host-byte vector | Exact AoS/SoA geometry and query packing, shape/finite/worker boundaries, complete moment-word/receipt/ownership parity, registration accuracy and isolated completed-call comparisons; `notes/2026-10-14_direct_soa_registration_upload.md` |
| Direct SoA feature input staging | Complete | Pack FPFH/ISS/MLS queries and FPFH samples from borrowed original-order CPU tree coordinates; remove one 12*N-byte host position vector per explicit upload | 26,236,560 exact uploaded words, 60 owned-result comparisons / 51,432 output words, 35 frozen errors, native/public/feature checks and two paired completed-upload surveys with adverse cases retained; `notes/2026-10-14_direct_soa_feature_upload.md` |
| Bounded SoA GPU query staging | Complete | Native finite validation and padded query packing in disjoint output bands, capped by CPU count, useful 262,144-point grains and four workers; small/one-worker/WASM serial path | 257,951,616 CPU and 41,943,344 GPU input words, invalid worker boundaries/concurrency/receipts/ownership, related native/public/feature checks and paired completed host/query-upload surveys; `notes/2026-10-14_parallel_soa_query_staging.md` |
| Bounded GPU tree point staging | Complete | Native validation of original axes and disjoint traversal-order XYZ/index gathering at least 524,288 points, capped by CPU count, useful 131,072-point grains and four workers; serial nodes and small/one-worker/WASM path | 346,032,096 CPU and 68,157,840 GPU point/node words, 252 frozen errors, 108 large neighborhood rows, worker/capacity/concurrency/receipt/ownership checks and paired completed host/tree-upload surveys; `notes/2026-10-14_parallel_gpu_tree_staging.md` |
| Explicit GPU tree/query gathering | Complete | Integer GPU gather into existing tree records; optional original-order queries share uploaded SoA axes and indices, halving point/query host payload with explicit device output receipts | Bitwise point/node/query probes, finite errors, cold concurrent pipelines, runtime/ownership/search checks, native/public/feature checks and isolated completed-upload surveys; `notes/2026-10-14_gpu_tree_point_gather.md` |
| Explicit GPU feature input gathering | Complete | Shared tree/query gathering for ISS/MLS/FPFH, plus integer GPU assembly of FPFH position/normal samples; point host payload 16*N for ISS/MLS and 28*N for FPFH with explicit device receipts | Exact coordinate/normal/padding probes, input/error/runtime/ownership/status checks, public and feature builds, unchanged compute/default bodies and isolated completed-upload comparisons; `notes/2026-10-14_gpu_feature_input_gather.md` |
| Fused FPFH GPU input gathering | Complete | Below 1,048,576 points, one integer GPU pass generates tree points, original-order queries and position/normal samples from two temporary inputs; larger calls retain two passes after measured regressions, preserving receipts and default placement | Frozen two-pass/independent bit probes, cutoff/cache/descriptor/status/error/concurrency/ownership checks, size/binding planners, shared native/public/feature regression checks and isolated completed-upload comparisons; `notes/2026-10-14_fused_fpfh_input_gather.md` |
| Bounded GPU feature input validation | Complete | Native CPU finite/bounds/unit-normal scans for FPFH/ISS/MLS, capped at four workers with serial small/one-worker/WASM behavior | 104,858,320 uploaded words, frozen extent/error parity, worker/numeric boundaries, 66 native and 11 public tests, feature/WASM checks and 80 isolated validation/completed-upload comparisons with adverse cases retained; `notes/2026-10-14_parallel_feature_validation.md` |
| Explicit GPU registration input gathering | Complete | Shared target tree/query gather for ICP/GICP/point-to-plane inputs; reduce target host payload by 16*T with 32*T explicit device output bytes | 174,016,032 input words, 6,664 result/status words, frozen errors/metadata, trim/ownership/concurrency, 91 native and 8 public tests, feature/WASM checks and 60 isolated completed-upload comparisons with adverse cases retained; `notes/2026-10-15_gpu_registration_input_gather.md` |
| Bounded AoS GPU query staging | Complete | Native finite checks and exact original-order padded query packing, capped at four workers with serial small/one-worker/WASM paths | Frozen host/GPU words, invalid worker edges, neighborhood/ICP ownership and receipts, feature/WASM checks and isolated paired completed uploads; `notes/2026-10-15_parallel_aos_query_staging.md` |
| Bounded point-to-plane normal staging | Complete | Two native CPU phases for finite common-scale bounds and exact scaled-row packing, at most four workers with original serial small/one-worker/WASM loops | 339,479,420 host words, 25,167,928 GPU normal words, 2,304 equation/status words, 30 errors, worker/numeric/ownership boundaries, 97 native and 8 public tests, feature/WASM checks and 120 candidate/final comparisons with adverse/inconclusive cases retained; `notes/2026-10-15_parallel_plane_normal_staging.md` |
| Point-to-plane resident normal equations | Complete | Original positions/tree/raw normals, GPU transforms/matching/plane Hessian/gradient/fitness; 256-byte equations with CPU normalized solve/world conversion | Independent raw-normal f64 equations, 108 pose/scale/weight/stopping cases, gating/ownership/schema/receipt guards and completed CPU comparisons; `notes/2026-10-13_gpu_point_to_plane_reduction.md` |
| Point-to-plane query/row lifetime reuse | Complete | Reuse transformed query rows after nearest matching for rotation Jacobian/residual; reduce private per-source scratch from 56 to 40 bytes with unchanged five-pass math/transfers | Frozen GPU parity in 324 conditions with poisoned scratch, exact leases/limits, queued/concurrent ownership, CPU registration precision and paired completed-call surveys; `notes/2026-10-13_gpu_plane_reuse_transformed_queries.md` |
| GICP resident k-NN covariances | Complete | Reused query/sample positions and target tree, bounded GPU k-NN/sample covariance/plane eigenvectors with status carried into equations; no covariance readback/re-upload | Independent f64/spectral/Rayleigh checks, 162 pose/scale/k/stopping cases, malformed/status/ownership/receipt guards and completed CPU/previous-GPU comparisons; `notes/2026-10-13_gpu_gicp_resident_covariances.md` |
| GICP resident radius covariances | Complete | Complete inclusive GPU radius neighborhoods with explicit capacity, compensated central sums/eigenvectors retained for equations; sparse epsilon*I and overflow rejection | Independent brute-force/f64/spectral references, 162 pose/scale/radius/stopping cases, sparse extent beyond grid capacity, receipts/ownership/status checks and completed API comparisons; `notes/2026-10-13_gpu_gicp_resident_radius_covariances.md` |
| GICP resident grid covariances | Complete | Reused original positions and canonical GPU grid gather/covariance sweeps; no source KD-tree/neighborhood rows, private grid scratch and complete-support rejection | Exact legacy covariance-word parity, 162 registration conditions, boundary/error/ownership/concurrency and transfer checks with completed-call comparisons; `notes/2026-10-13_gpu_gicp_resident_grid_covariances.md` |
| ICP resident trimming | Complete | Stable GPU distance/source-index selection and exact binary64 retained-count floor; selected moments plus all-gated fitness in one submission, fixed 176-byte readback | Independent comparison-sort/f64 moment references, rounding/boundary/ownership/status checks, 486 registration conditions, completed comparisons and an explicit adverse large-gate precision case; `notes/2026-10-13_gpu_icp_resident_trimming.md` |
| ICP trimmed moment precision | Complete | Default two-word coordinate/quotient/product arithmetic with integer product residuals and preserved rounding; explicit previous f32 choice, same GPU matching/readback/CPU solve | 2,076 independent arithmetic comparisons, fixed-pose f64 moments, 27 large trajectories, mixed precision/concurrency and completed timing; `notes/2026-10-13_gpu_icp_trimmed_precision.md` |
| GPU ICP trim histogram lanes | Complete | Optional 256-lane integer bin reduction with existing 64-lane small/device-limited path, no extra pass or scratch | Frozen selection words, independent maximum-mass digit/rank references, capability/lazy/concurrent reuse checks, ICP numerical regressions and paired completed selector/moment calls; `notes/2026-10-13_gpu_icp_trim_selector.md` |
| GPU ICP trim selection tree | Complete | Exact 255-node shared integer sum tree and eight rank decisions in the capable-device wide pipeline; existing small/device-limited linear path, global scratch and passes | Two frozen GPU references, all-bin/max-mass rank endpoints, stable-sort and f64/registration regressions, paired completed surveys with limited synthetic improvement and retained adverse cases; `notes/2026-10-13_gpu_icp_trim_selection_tree.md` |
| Remaining registration updates | Planned | Small pose-update boundaries and broader numerical conditioning | Pose/convergence/fitness references, numerical conditioning, source/target residency and end-to-end registration timing |
