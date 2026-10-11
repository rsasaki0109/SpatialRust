# Feature matrix

This is the authoritative map of optional capabilities in SpatialRust. The
workspace keeps heavy dependencies behind crate-local features; applications
should normally depend on `spatialrust` and select only the profiles they need.

## User-facing profiles

| Profile | Enables | Intended use |
| --- | --- | --- |
| `mvp` | PCD, PLY, LAS/LAZ, COPC; voxel, normals, plane, Euclidean clustering, ICP, MVP pipeline | CPU-first end-to-end pipeline |
| `mvp-http` | `mvp` + HTTP byte-range COPC | Remote COPC input |
| `gpu-wgpu` | wgpu runtime and GPU kernels | Explicit GPU operation APIs |
| `pipeline-mvp-gpu` | `mvp` GPU stages plus `gpu-wgpu` | GPU-enabled MVP pipeline |
| `gpu-aoso-staging` | AoSoA core packing plus GPU-resident frame APIs | Chained GPU execution with explicit readback |
| `serde` | serde derives for core/math metadata and schemas | Configuration and metadata serialization |
| `io-manifest` | explicit storage roots plus file manifests | SHA-256/size receipts for local IO |
| `rosbag2-sqlite` | read-only rosbag2 SQLite PointCloud2 CDR source | Bounded XYZ/XYZI records in `spatialrust-ros2`; native ROS 2 executors remain separate |

The Python extension selects its supported meta-crate features in
`crates/spatialrust-py/Cargo.toml`. It is intentionally outside the Rust
workspace because its build requires a Python toolchain.

## Crate capability map

| Crate | CPU/default capability | Optional capability | Heavy dependency boundary |
| --- | --- | --- | --- |
| `spatialrust-core` | schema, metadata, `PointCloud`, tensors, execution contracts | serde, AoSoA packing | none |
| `spatialrust-math` | vector/matrix/pose math | serde | none |
| `spatialrust-io` | no format enabled by default | PCD, PLY, LAS/LAZ, E57, COPC, HTTP COPC, explicit roots/manifests, per-node `CopcNodeReader` hierarchy walk | format and checksum crates are optional |
| `spatialrust-ros2` | ROS 2 type contracts through `spatialrust-runtime` | read-only rosbag2 SQLite PointCloud2 CDR streaming with optional float32 intensity, plus source-bound TFMessage inventory | `rusqlite` is isolated behind `rosbag2-sqlite` |
| `spatialrust-search` | KD-tree | graph, bounded parallel radius/k-NN queries | none; Criterion is dev-only |
| `spatialrust-semantic` | entities, embeddings, fusion, search | `semantic-model`: `OnnxEntityEmbedder` through an existing model session | `spatialrust-ai`/`spatialrust-tensor` optional; ONNX stays in the AI crate |
| `spatialrust-filtering` | voxel | GPU voxel, outlier, crop, FPS, MLS | wgpu/search optional |
| `spatialrust-features` | normals | ISS, orientation, boundary, GPU normals | wgpu/search optional |
| `spatialrust-segmentation` | plane and Euclidean clustering | GPU stages, DBSCAN, ground, primitives, region growing | wgpu/search optional |
| `spatialrust-registration` | ICP | point-to-plane, GICP, GPU covariance, NDT, FPFH | wgpu/search optional |
| `spatialrust-gpu` | device markers only | wgpu runtime, AoSoA staging | wgpu/bytemuck/pollster optional |
| `spatialrust-pipeline` | MVP pipeline | GPU MVP stages | algorithm crates only |
| `spatialrust-interchange` | `interchange-gltf`, `interchange-openusd` | `tiles3d`: deterministic OGC 3D Tiles 1.1 `tileset.json` + `pnts` octree export; `tiles3d-copc`: bounded COPC hierarchy → tileset | `tiles3d-copc` pulls `spatialrust-io` + `spatialrust-core` for COPC node reads |
| `spatialrust-arrow` | no capability by default | C Data `__arrow_c_array__`, C Stream `__arrow_c_stream__`, C Device `__arrow_c_device_array__` exports/imports for point records | `arrow-c-data`, `arrow-c-stream`, `arrow-c-device`; no Arrow runtime dependency |
| `spatialrust-py` | Python binding surface | selected meta-crate features, including `export_tiles3d` / `export_copc_tiles3d`, Arrow C Data `__arrow_c_array__` / `__arrow_c_stream__`, `OnnxEntityEmbedder` (onnxruntime) | PyO3/NumPy |

## Execution contract

- `gpu-image` also adds `equalize_histogram_gpu` and its five-pass profiled
  variant for resident one-channel u8 images. Histogram/CDF/lookup/application
  have no intermediate payload readback; the owned output preserves metadata
  and constant-image bytes. GPU integer word arithmetic reproduces CPU binary64
  half-boundary rounding without shader f64. Inputs accept at most u32::MAX
  pixels and require supported texture/compute limits. Existing partial/CDF
  storage is reused; additional lookup scratch is 1 KiB. Native output readback
  is explicit and copies the physical RGBA8 image; five timestamp stages collect
  80 query bytes separately. APIs remain provisional; dependencies are unchanged.
- `gpu-image` also adds `histogram_cdf_gpu`, its one-pass profiled variant and
  resident `GpuImageCdf`. Each of the 256 u32 entries counts pixels at or below
  that byte value; the final entry equals source pixel count. A single 256-lane
  tree scan uses 1 KiB local storage and a separate pooled 1 KiB output, with no
  implicit download or normalization. Native `readback_u32` transfers 1 KiB per
  call, while explicit timestamp collection transfers 16 bytes per CDF pass.
  Source bins remain immutable and can be recycled after the submitted reads;
  live results retain original runtime ownership. These APIs are provisional.
- `gpu-image` adds `histogram_u8_gpu`, its two-pass profiled variant and resident
  `GpuImageHistogram` for any logical channel of a 1..=4-channel u8 texture.
  Exact 256-bin u32 counts accept at most u32::MAX pixels; private partials are
  capped at 256 KiB and the live result at 1 KiB. Construction stays resident;
  native `readback_u32` explicitly reads 1 KiB per call, and `recycle` returns
  storage only to its original runtime. Device limits, invalid channels and
  foreign sources are rejected before allocation. Grayscale conversion is an
  explicit preceding kernel when needed. Benchmark CPU/Rayon dependencies are
  dev-only; no production/default/core dependency is added.
- `spatialrust-search/parallel` bounds each radius/k-NN tensor query by available
  CPU parallelism and original chunk count, without a production dependency.
  Contiguous worker bands preserve backend chunk ranges and append result order.
  Inputs below 4,096 points and single-worker/single-chunk jobs run on the caller;
  empty inputs and parallel k-NN with `k=0` leave output untouched. Metadata is
  worker-sized, while output and local neighbor buffers remain result-sized.
  `spatialrust/search-parallel` exposes the same functions through the facade.
- GPU storage retention is explicit and bounded under `gpu-wgpu`: upload/CHW
  and private point scratch pools each default to 64 MiB, with at most eight
  resources per size. `gpu-image` adds an independent 64 MiB RGBA8 texture pool.
  `GpuBufferPool::{with_budget,set_budget,stats,clear}` and runtime scratch/image
  budget, stats and clear methods expose control. Zero disables retention;
  reducing below retained bytes clears that pool. Budgets cover cached payloads
  rather than live outputs, uniforms, readback staging, driver overhead or peak
  memory. Scratch is recycled after submission; images/tensors retain runtime
  ownership across moves and reject recycling to another device's pools.
- `spatialrust-vision/imgproc-analysis` enables optional Rayon for bounded CPU
  threshold/histogram row bands. Large thresholds dispatch at 100,000 components
  and u8 histograms at 1,000,000 pixels when at least two rows and pool workers
  are available. Single-worker, small, empty and single-row inputs use scalar
  execution. Histograms use 2 KiB per local band, with exact integer merging.
  The default vision feature set and `spatialrust-core` gain no dependencies.
- Contrast enhancement uses the same `imgproc-analysis` Rayon dependency.
  Equalization lookup mapping dispatches at 100,000 pixels with independent
  rows. CLAHE tile lookup and output interpolation dispatch at 1,000,000 pixels,
  each bounded by its independent tile/row count and the current pool. Only the
  parallel interpolation path retains width-sized shared horizontal geometry;
  scalar interpolation, clipping, reflected image padding and rounding remain
  numerically identical. Invalid grid/lookup-size products fail before
  allocation. No default/core dependencies or automatic GPU placement change.
- `ExecutionPolicy::CpuSingle` and `CpuParallel` request CPU execution.
- `ExecutionPolicy::Gpu(DeviceKind)` explicitly names the requested GPU
  backend. A backend that cannot satisfy the request must return an error; it
  must not silently turn an explicit request into CPU work.
- `ExecutionPolicy::Auto` is the only policy that permits algorithm-specific
  heuristics and CPU fallback, including fallback when no compatible wgpu
  adapter is available.
- `ExecutionReceipt` and `ExecutionOutput<T>` preserve the requested policy,
  resolved backend, logical stages, and explicit transfer byte accounting.
- `MvpPipelineResult.receipt` exposes the receipt for every MVP stage and an
  aggregate transfer view; a GPU stage may still report a documented host-side
  refinement stage.
- `GpuSpatialFrame` is the explicit GPU-resident path. Host positions are
  uploaded at the frame boundary; position, normal, and segment arrays remain
  resident until explicitly read back. Frame construction partitions source
  points without computing unused centroids. Nonempty construction synchronizes
  two four-byte cell counts (voxel partition and radius grid), reported in the
  frame receipt. Rebuilding a nonempty radius grid adds four bytes; reusing a
  matching grid adds none. Empty construction reports zero transfer bytes.
  Receipts count payload copies and scalar metadata moved by frame methods,
  including attribute concatenation, but exclude kernel parameter uniforms,
  shader storage accesses, and transfers made outside frame methods. See the
  [frame transfer receipt](../notes/2026-10-11_gpu_frame_transfer_accounting.md).
- GPU voxel compaction scans 256-element blocks and recursively scans their
  totals, then propagates offsets in the same submission. Segment arrays stay
  resident; the four-byte cell-count readback is unchanged. Profiled frames
  reserve the actual hierarchy's pass count, including partial blocks. See
  [parallel execution measurements](../notes/2026-10-11_parallel_image_point_execution.md).
- `GpuStageProfiler` (under `gpu-wgpu`) records named pass timestamps through
  `run_gpu_vision_chain_profiled` (`gpu-image`) and
  `run_aoso_voxel_normal_frame_profiled` (`gpu-aoso-staging`). Create an optional
  timestamp runtime, check `timestamp_queries_enabled`, then explicitly call
  native `collect`. The report counts 16 query readback bytes per pass; image
  and frame payload receipts remain separate. Default constructors request no
  timestamp features. GPU pass sums exclude copies, CPU work and inter-pass
  gaps; use `wait_idle` when measuring completed wall time. See
  [shared GPU timing](../notes/2026-10-11_gpu_stage_timing.md).
- `SpatialRuntime` contains only backend identity and policy compatibility;
  backend-specific queues, buffers, and transfers stay in the implementing
  crate.
- `spatialrust-core` contains no wgpu, CUDA, ROS2, ONNX, or IO implementation.

Host-staged GPU algorithms are explicit hybrid implementations: voxel, normal,
and plane APIs expose their GPU kernel boundary, while Euclidean clustering
builds, sorts, and compacts the sparse grid on wgpu and performs deterministic
component union on the host. This split is visible in receipts and is not an
implicit CPU fallback.

Voxel key sorting uses stable four-bit GPU radix passes over signed i32 XYZ
coordinates, preserving increasing source indices inside each cell. Sorting
allocates exact-sized entries, stays on the device, and no longer needs padded
entry filtering or its scalar readback. Segment compaction still synchronizes
the existing four-byte cell count; GPU segment metadata remains resident until
an explicit metadata readback. See the
[radix validation receipt](../notes/2026-10-11_gpu_voxel_radix_sort.md).

## Verification commands

```bash
cargo test --workspace
cargo test --workspace --all-features --no-run
cargo test -p spatialrust --features mvp
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

GPU runtime tests require a compatible adapter. On GPU-less CI, compile the
GPU test targets and execute them on a runner with a wgpu adapter.

## Deliberately separate future crates

The following are not core features and must remain separate integration
crates: `spatialrust-ros2`, `spatialrust-ai`/DLPack/ONNX, and the CUDA backend.
