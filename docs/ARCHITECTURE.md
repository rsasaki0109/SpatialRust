# SpatialRust Master Architecture (v0.1)

Design version: **v0.1 Master Architecture Draft**

North star: **Rust-native spatial intelligence: capture, understand, reconstruct, and act**

## Core decisions

| Area | Decision |
| --- | --- |
| Core model | `SpatialTensor` + `SpatialIndex` + `SpatialAlgorithm` + `SpatialRuntime` |
| Storage | Hybrid Schema-SoA + chunked AoSoA views |
| GPU | wgpu/WebGPU-first + CUDA specialized backend |
| Robotics | ROS2 first-class, zero-copy oriented |
| AI | DLPack / ONNX / embedding-native point cloud |
| Repository | Mono repo / Cargo workspace |

## Initial workspace (Epic 0)

- `spatialrust` — meta crate
- `spatialrust-core` — schema, metadata, traits
- `spatialrust-math` — Vec/Mat/Pose math
- `spatialrust-io` — readers/writers (feature-gated formats)
- `spatialrust-gpu` — device buffers and GPU runtime
- `spatialrust-image` — typed CPU image buffers and strided zero-copy views
- `spatialrust-image-io` — bounded, feature-gated image codecs and source metadata
- `spatialrust-tensor` — runtime-independent tensor metadata, CPU ownership, and DLPack boundary
- `spatialrust-camera` — camera models, distortion, and RGB-D/point-cloud bridge
- `spatialrust-vision` — feature-gated CPU preprocessing, Feature2D, geometry/multiview, warps, detection, masks, and dense maps

## MVP scope

1. Load PCD/PLY/LAS/LAZ
2. Voxel downsample
3. Normal estimation
4. RANSAC plane segmentation
5. Euclidean clustering
6. ICP registration
7. Save output
8. Feature-gated wgpu execution with explicit host/device receipts

Post-MVP additions:

- Unified file IO via `read_point_cloud_file` / `write_point_cloud_file`
- Explicit `spatialrust_io::StorageRoots` for external input/output storage;
  `io-manifest` adds checksummed size receipts without adding storage policy to
  `spatialrust-core`
- `MvpPipelineConfig::*_policy` for feature-gated GPU MVP stages
- `MvpPipelineResult::receipt` for per-stage backend and transfer accounting

## Dependency direction

```
math -> core -> search/geometry/io/gpu -> algorithms -> integration
```

Forbidden: `core -> io`, `core -> gpu impl`, `core -> ros2`, `core -> ai`.

Image and camera dependency direction:

```
math -> image
math -> image -> image-io
math -> image -> vision
math -> image -> tensor -> ai
math + image + core -> camera -> vision::spatial/rgbd/odometry
```

`spatialrust-image` remains independent of `spatialrust-core`. GPU image storage
must use a dedicated backend and explicit upload/readback APIs; Epic 89 provides
`GpuImage` and image compute kernels in `spatialrust-gpu` behind `gpu-image`.
The v2 representation is a pooled `rgba8uint` texture with a retained logical
channel count. Kernels return device-resident images and append named receipt
stages; only `upload_u8` and `readback_u8` cross the host/device boundary.
Runtime adapter identity and synchronization are explicit (`adapter_info`,
`wait_idle`), and `recycle` returns textures to the runtime pool.
`spatialrust-image-io` depends on storage, never the reverse; standard codecs
are additive, while TIFF and OpenEXR remain independently gated.
Calibration datasets, robust solver controls, and residual receipts live in
`spatialrust-camera`; they depend only on `spatialrust-math` small dense solves.
External nonlinear optimizers may be added only behind dedicated features.
`spatialrust-vision` keeps preprocessing, Feature2D, geometry/multiview (H/F/E,
PnP, sparse LK, stereo BM), warp, detection, dense-map, and spatial bridges in
separate additive features. Geometry depends on `spatialrust-camera` only and does
not pull Feature2D or dense-map types. CPU APIs never perform implicit device
copies; future GPU/CUDA implementations belong behind explicit backend features.
Video algorithms depend on dense/detection contracts, while timestamped pull
sources are isolated behind `video-adapters`; native codec/camera runtimes stay
in future dedicated adapter crates/features.
Visual and RGB-D odometry kernels remain in the additive `odometry` vision
feature. Their conversion into stamped trajectory motion is a one-way optional
bridge in `spatialrust-mapping`; monocular scale and invalid depth remain
explicit at that boundary.
Computational photography composes image/warp/geometry primitives behind the
additive `photography` feature. Panorama APIs expose canvas bounds and enforce a
pre-allocation pixel budget; codec and device execution remain separate.
The optional runtime execution graph owns operator scheduling only. Placement,
watermarks, and named transfers reuse `spatialrust-distribute`; fusion is
limited to linear fusable nodes on one device and cannot erase transfer edges.
Its `imgproc-*` features share one border extrapolation contract; `filter2d`
means correlation, while true convolution is an explicitly named operation.
`spatialrust-tensor` is distinct from the point-cloud chunk iterator named
`spatialrust-core::SpatialTensor`; it owns generic dtype/shape/stride/device
contracts and never performs implicit host/device transfers.
`spatialrust-ai` depends on `spatialrust-tensor`, never the reverse. Its default
build defines only backend/session and explicit-copy contracts, plus a
deterministic `MockInferenceBackend` for demos and tests. ONNX Runtime and
each hardware execution provider are additive features. Runtime-owned CPU
outputs cross back through the runtime-independent `HostTensorStorage` trait,
so their allocator lifetime can be retained without copying or adding an ONNX
dependency to the tensor crate.
`spatialrust-vision` `ai-adapters` bridge CPU images to contiguous host tensors
and decode depth/mask/detection tensors back into dense vision types without
depending on `spatialrust-ai`.

ROS 2 bag storage follows the same boundary. `spatialrust-runtime` owns only
runtime-independent PointCloud2 CDR and type-negotiation contracts;
`spatialrust-ros2` owns the optional read-only rosbag2 SQLite source behind
`rosbag2-sqlite`. Native `rclrs` executors, custom message bindings, and
compressed bag storage remain separate adapter features. The source emits
bounded XYZ/XYZI `SpatialRecordChunk` leases with frame/timestamp metadata and
protocol-independent `spatialrust-records::RecordProvenance`; its bounded
TF inventory preserves source frame edges without composing or applying them.
It never places ROS 2 or SQLite types in `spatialrust-core`.

Versioned record envelopes remain in `spatialrust-records`. Their concrete
point fields are described by `SchemaDescriptor`, capture time and coordinate
frame remain in `SpatialMetadata`, and source lineage is carried separately by
`RecordProvenance`. Schema migration and bounded record transforms must retain
source lineage; aggregations clear a single source sequence when it no longer
identifies one input record.

## Roadmap epics

| Year | Focus |
| --- | --- |
| 1 | Foundation / MVP |
| 2 | v1.0 stable geometry runtime |
| 3 | Robotics adoption (ROS2/Autoware/Nav2) |
| 4 | AI integration |
| 5 | Spatial computing platform |

The canonical post-foundation horizon is Epics 91–100 in `docs/ROADMAP.md`.
Epic 91 adds `spatialrust-records` and `spatialrust-arrow` while keeping Arrow
out of `spatialrust-core`. Epic 92 adds `spatialrust-sync` for clocked stamps,
frame graphs, and deterministic multimodal episode replay. Epic 93 adds
`spatialrust-mapping` for trajectories, relative motion traits, and pose graphs.
Epics 94–100 extend into scene reconstruction (`spatialrust-scene`), semantic
spatial data (`spatialrust-semantic`), embodied-AI episodes
(`spatialrust-episode`), robotics runtime contracts (`spatialrust-runtime`),
glTF/OpenUSD interchange (`spatialrust-interchange`), explicit distributed
execution (`spatialrust-distribute`), and platform stability
(`spatialrust-platform`). Heavy native bindings stay optional.

See the full master architecture document in project planning materials for trait-level design, ADRs, and Codex execution tasks (Epics 0–13).

## SpatialRust 1.2 bounded-streaming boundary

The canonical 1.2 program is Epics 121–126 in `docs/ROADMAP.md`. It adds
bounded point-cloud execution without expanding `spatialrust-core`:

```text
core PointCloud/SpatialTensor
        -> records stream contracts
        -> feature-gated io adapters
        -> chunk-safe algorithms/pipeline
        -> CLI/Python
```

`SpatialTensor` continues to describe borrowed chunks of one materialized
`PointCloud`; it is not an out-of-core source. Streaming memory limits account
for explicitly owned buffers and fail before exceeding their hard ceiling.
Format-specific caches and unavoidable temporary spool storage are named in
versioned receipts. GPU upload/readback remains caller-selected and measurable.
Leased record chunks keep their memory reservation alive for exactly the record
borrow lifetime. The additive `PointCloud::into_parts` inverse and mutable
column iteration are the only core ownership primitives required for
allocation-free source-owned buffer recycling; stream traits, threads, queues,
receipts, and policies remain outside core.

The 1.2 release boundary is machine-checked by
`Streaming12ReleaseGate`. Its dedicated Linux/Windows/macOS matrix exercises
records, all bounded formats, deterministic operations, the real CLI path, and
the canonical receipt example. The gate rejects missing or skipped evidence,
tracked-memory or spool overruns, retained reservations after finish,
unrecorded host copies, any CPU-workflow device transfer, deterministic
mismatches, and excessive run/file-handle fan-out. Concrete IO/pipeline/Python
adapters remain provisional; the bounded record traits, chunk lease, memory
budget, cancellation, options, and versioned receipt are the stable 1.2
foundation.

## Visual architecture boundary

The Visual program is Epics 133–140 in `docs/ROADMAP.md`. Visualization remains
an additive consumer of spatial data and never expands `spatialrust-core`:

```text
math + optional core capabilities -> viz contracts
viz contracts + gpu-wgpu          -> render-wgpu
render-wgpu + io/scene/mapping    -> native viewer
render-wgpu + bounded records     -> explicit streaming LOD
render-wgpu                        -> Web/WASM and Python/Jupyter adapters
```

`spatialrust-viz` owns backend-independent borrowed geometry, camera, style,
layer, residency, and transfer-receipt contracts. Borrowed host views do not
interleave or copy structure-of-arrays point data. GPU allocation, upload,
render passes, picking, and readback belong to `spatialrust-render-wgpu`; every
host/device crossing is caller-requested and recorded. Windowing and UI
dependencies stay in the dedicated `spatialrust-viewer` application crate.
That crate's default build contains only portable viewer state, input reduction,
attribute inspection, and owned debug-overlay adapters. Its `native` feature
adds winit window/event handling; it does not upload geometry. Applications
retain control of explicit renderer upload and presentation integration.
Scene, mapping, camera, and semantic adapters are separate additive viewer
features. Mesh storage is borrowed directly; adapters that must transpose AoS
data or generate lines return a byte-exact `AdapterReceipt`. Synchronized RGB-D
inspection borrows image/cloud payloads and validates dimensions, ordering, and
sensor skew before exposing a timeline frame.

Native windowing, Web/WASM bindings, and Python/Jupyter adapters are independently
feature-gated. Large-cloud display consumes the bounded records and persisted
index contracts from Epics 127–132, with explicit point, memory, upload, and
in-flight chunk budgets. Camera motion may cancel obsolete requests but must not
leak leases or GPU allocations. Headless rendering and deterministic image
comparison provide the portable correctness boundary; interactive frame-rate
numbers remain dated, adapter-specific receipts.

`spatialrust-lod` is the renderer-independent enforcement layer. It accepts a
validated hierarchy rather than opening data itself, uses separate enter/exit
screen-error thresholds, and selects resident ancestors while finer children
are unavailable. Optional records integration owns drop-scoped decoded-memory
leases; GPU admission records caller-declared upload/allocation bytes and
evicts only unprotected LRU nodes. The COPC adapter produces a bounded query but
never performs range IO implicitly.

`spatialrust-web` owns only the browser/WASM boundary. Its portable JSON
envelope embeds the same `ViewerState`, and WebGPU rendering delegates to
`spatialrust-render-wgpu` after caller-requested upload. Native process-global
wgpu sharing is unavailable on wasm32; browser callers construct a main-thread
runtime asynchronously. Remote byte access requires a bounded range plan before
fetch, an abort signal, `206` plus exact response-length evidence, and explicit
admission into the bounded WASM cache.

The standalone `spatialrust-py` wheel wraps that same Web viewer envelope
instead of defining another state model. NumPy viewer geometry either retains
three contiguous SoA owners without copying or enters owned Rust storage through
an explicitly named, byte-receipted copy. Native launch creates only the winit
state/input shell; renderer uploads remain separate. The independently packaged
`spatialrust-jupyter` AnyWidget validates state with the Rust binding and sends
it to `spatialrust-web` through a versioned iframe protocol with exact
source/origin checks. Neither adapter chooses a device, range source, upload, or
readback.

The aggregate boundary is machine-checked by `VisualReleaseGate`. Stable
backend-independent contracts and provisional adapters are registered
explicitly. The gate requires strict Linux/Windows/macOS headless image
evidence, native/browser/Python/Jupyter smoke evidence, transfer and LOD
receipts, documentation and unsafe audit, and acknowledgement of the
`visual-1` migration policy. It fails closed on missing, skipped, duplicate,
future-dated, older-than-30-day, over-budget, or experimentally unstable
evidence. The canonical decision and measurements are committed in
`docs/VISUAL_RELEASE_RECEIPT.md`.
