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
boundary. Storage is packed interleaved `u8` expanded to one `u32` value per
component for WGSL clarity (texture path is deferred).

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
| 91A | Complete | `SchemaId`/`SchemaVersion`/`SchemaDescriptor`, compatibility reports, `SpatialRecord` | `records` |
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
| 99 | Complete | `spatialrust-distribute` partitions, backpressure, named transfers |
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

Epic 92 completes when stamped multimodal records can be indexed deterministically,
bundled within a sync window, and transformed across a calibrated frame graph.
Optional `sync-mcap` write/read path covers XYZ-only stamped clouds today.

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
