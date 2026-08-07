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
| 148C | Planned | Unify PCL/PDAL/Open3D comparison receipts and aggregate runner with fail-closed checks | aggregate command |
| 148D | Planned | Dated honest comparison receipt, docs, and README updates | note + `docs/` |

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
| 146 | Planned | 141–145 | API stability, Python/docs/CI updates, security review, and v1.3 release receipt |

### Epic 141 acceptance slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 141A | Complete | Canonical external input snapshot, topic counts, output hashes, storage policy, and fail-closed downstream gates | [`docs/REAL_DATA_ACCEPTANCE.md`](REAL_DATA_ACCEPTANCE.md) |
| 141B | Complete | Automated absolute-root/free-space preflight and local checksum manifest validation | `storage-preflight`, `DatasetManifest::validate_local_files`, and rosbag2 CLI smoke |

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
