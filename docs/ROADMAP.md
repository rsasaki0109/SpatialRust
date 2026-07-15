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
| 112 | Planned | 111 | Attribute native kernel, allocation, Python conversion, and transfer costs with reproducible throughput and memory receipts |
| 113 | Planned | 112 | Caller-owned outputs and reusable workspaces for multi-stage CPU vision without hidden copies |
| 114 | Planned | 112–113 | Safe size-aware CPU dispatch for packed fast paths, strided fallbacks, and bounded row/tile parallelism |
| 115 | Planned | 113–114 | Accelerated resize and color conversion with precomputed sampling plans and fused preprocessing experiments |
| 116 | Planned | 113–115 | Accelerated separable Gaussian and Sobel engine with cached kernels and shared gradient passes |
| 117 | In progress | 113–116 | Sliding-window morphology engine with exact OpenCV comparison and generic-mask fallback |
| 118 | Planned | 113–117 | Fused Canny fast path that avoids public intermediates unless explicitly requested |
| 119 | Planned | 104, 115–118 | Explicit upload-once GPU-resident vision chain with no intermediate readback |
| 120 | Planned | 112–119 | Vision 2 cross-platform correctness, speed, memory, allocation, and transfer release gate |

Each Epic lands as one reviewable PR using implement → test → commit → PR →
merge. Stable Vision 1 ownership and error contracts remain compatible;
reusable workspace surfaces are additive. CPU APIs do not choose a GPU or copy
to one implicitly, and GPU receipts must retain named upload/readback stages.

### Epic 112 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 112A | Planned | Separate Python conversion, allocation, native kernel, upload, execution, and readback time | versioned component timing receipt |
| 112B | Planned | Report MPix/s, ns/pixel, bytes allocated, peak workspace, and batch policy | strict finite JSON contract tests |
| 112C | Planned | Add native Criterion counterparts for every OpenCV vision workload | VGA/1080p/4K matched workload manifest |
| 112D | Planned | Record single-thread and default-thread CPU modes | host and thread-policy receipt |
| 112E | Planned | Publish bottleneck attribution without changing kernels | dated baseline note and Pages update |

### Epic 113 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 113A | Planned | `*_into` entry points for Gaussian, Sobel, morphology, and Canny | packed/strided identity and padding tests |
| 113B | Planned | Explicit reusable scratch storage for multi-pass algorithms | steady-state allocation receipt |
| 113C | Planned | Validate dimensions, metadata, overlap, and channel contracts | negative and property tests |
| 113D | Planned | Reuse outputs through Python `out=` where supported | object-identity and numerical tests |
| 113E | Complete | Exact EDT caller-owned output and explicit reusable scratch | Rust/Python identity, capacity, and brute-force tests |

### Epic 114 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 114A | Planned | Shared small-image scalar and large-image row/tile dispatch policy | deterministic threshold tests |
| 114B | Planned | Packed `u8` one/three-channel and `f32` internal fast-path selection | dispatch receipt and fallback parity |
| 114C | Planned | Preserve generic components, channels, strides, and borders as safe fallbacks | full property suite |
| 114D | Planned | Bound worker creation and temporary memory | thread-count and peak-memory receipt |
| 114E | Complete | Exact EDT binary-row fast path, tiled transpose, and bounded pool dispatch | VGA/1080p/4K Criterion and OpenCV receipt |
| 114F | Complete | Cache EDT parabola heights and balance column tasks for dense masks | exact OpenCV parity and 4K Python reuse win |
| 114G | Complete | Cache NMS box geometry and avoid packed Python score copies | exact OpenCV index parity and 100/1,000/8,400-candidate wins |
| 114H | Complete | Bucket class-aware NMS keeps and expose one-call Python batched NMS | exact OpenCV parity and 26.38×/97.25× wins |
| 114I | Complete | One-pass active-set Soft-NMS selection, disjoint IoU exit, and borrowed Python scores | exact indices, bounded scores, and 3.42×–7.40× wins |
| 114J | Complete | Run-length union-find connected components with borrowed/non-zero Python masks | exact SAUF labels/stats and 2.17×–3.61× structured-mask wins |

### Epic 115 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 115A | Planned | Precompute resize source coordinates and interpolation coefficients | plan reuse tests |
| 115B | Planned | Packed bilinear/nearest/area and RGB-to-gray fast paths | OpenCV max-error contract |
| 115C | Planned | Evaluate resize+gray and resize+CHW fusion without changing standalone APIs | fused/unfused parity and timing |
| 115D | Planned | Improve current SpatialRust throughput by at least 5x on one canonical large profile | native and Python receipts |

### Epic 116 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 116A | Planned | Reuse separable-filter intermediates and cache validated Gaussian kernels | allocation and cache tests |
| 116B | Planned | Split border handling from contiguous interior loops | all border-mode property tests |
| 116C | Planned | Specialized 3x3/5x5/7x7 Gaussian and paired Sobel X/Y passes | OpenCV error receipt |
| 116D | Planned | Improve Gaussian by at least 10x and Sobel by at least 5x on one canonical large profile | native timing receipt |

### Epic 117 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 117A | Complete | Separable sliding min/max for rectangular elements | generic-reference property tests and OpenCV receipt |
| 117B | Complete | Packed rectangular `u8` dispatch and generic Cross/Ellipse/Diamond/custom-mask fallback | shape, border, stride, iteration, and anchor parity |
| 117C | Planned | Ping-pong workspace for iterations and composite operations | allocation and alias tests |
| 117D | Complete | Improve morphology by at least 20x on one canonical large profile | 43.8× 4K 5×5 baseline improvement; bit-exact 511×511 OpenCV wins |

### Epic 118 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 118A | Planned | Compute paired gradients, magnitude, and direction with shared traversal | intermediate parity tests |
| 118B | Planned | Ring-buffer suppression and reusable hysteresis queue | peak-memory receipt |
| 118C | Planned | Keep inspectable intermediates opt-in while making `canny()` allocation-light | API behavior tests |
| 118D | Planned | Improve Canny by at least 5x on one canonical large profile | F1/IoU and timing receipt |

### Epic 119 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 119A | Planned | Chain resize, gray, blur, edge, morphology, and AI packing on explicit GPU images | named stage receipt |
| 119B | Planned | One caller-requested upload, no intermediate readback, optional final readback | transfer-ledger denial tests |
| 119C | Planned | Reuse textures and pipelines in steady state | pool/cache receipt |
| 119D | Planned | Compare CPU, GPU round-trip, and GPU-resident modes separately | synchronized VGA/1080p/4K receipt |

### Epic 120 delivery slices

| Slice | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 120A | Planned | Cross-platform correctness and API compatibility matrix | Linux/Windows/macOS CI |
| 120B | Planned | Native and Python allocate/reuse performance budgets | fail-closed performance evidence |
| 120C | Planned | Peak memory, allocation count, thread policy, and GPU transfer budgets | typed release measurements |
| 120D | Planned | Generated algorithm/performance documentation and migration guidance | GitHub Pages and README links |
| 120E | Planned | Vision 2 release gate and runnable receipt example | `Vision2ReleaseGate` denial tests |

The improvement thresholds above compare against the checked Epic 112
SpatialRust baseline on the same host; they are not claims against every OpenCV
build. Accuracy gates remain workload-specific: resize/gray/Gaussian retain
their documented bounded error, Sobel and morphology retain exact comparison,
and Canny retains binary precision, recall, F1, and IoU requirements.
