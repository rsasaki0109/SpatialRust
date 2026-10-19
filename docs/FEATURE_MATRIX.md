# Feature matrix

This is the authoritative map of optional capabilities in SpatialRust. The
workspace keeps heavy dependencies behind crate-local features; applications
should normally depend on `spatialrust` and select only the profiles they need.

## User-facing profiles

`gpu-wgpu` exposes `WgpuBackendSelection`, `WgpuRuntimeOptions` and owned
`WgpuRuntime::new_headless_with_options` / `new_headless_with_options_async`.
Existing constructors and the shared runtime retain primary/high-performance
defaults and their timestamp contracts. Explicit GL selection requires platform
support and does not select a different backend on failure. The public RGB-D
example records requested/actual backends; its auditor keeps operation checks
and distinct raw hardware-class coverage separate. See the
[fixed GLES validation protocol](EXPLICIT_GLES_GPU_VALIDATION.md).

`vision-odometry-dense` exposes checked borrowed `DenseRgbdFrame` views and
`estimate_dense_rgbd_odometry` through `spatialrust-vision/odometry-dense`.
Both grayscale/metric-depth frames and the initial proper pose are explicit.
Bounded CPU pyramids and ordered row tiles use at most four private workers;
WASM is serial. Python owns all four inputs before releasing the GIL and returns
pose/error/support/working-buffer receipts. Original sparse APIs and device
transfers retain their contracts. See the
[fixed full-sequence protocol](DENSE_RGBD_ODOMETRY_VALIDATION.md).
The frozen final-overlap variant passes the original accuracy/coverage gates on
TUM fr1/xyz (798/798 poses); fr1/rpy remains incomplete (40/722). This is an
experimental CPU capability with bounded resources, not full motion acceptance.
The separate [causal initialization study](CAUSAL_DENSE_RGBD_VALIDATION.md)
reuses the same numerical binary with an explicit previous-increment seed and
independent seed-lineage checks; all final acceptance gates remain fixed.
It completes both 798/722-frame development timelines. XYZ passes; rpy RPE is
0.048145m but ATE 0.114793m still exceeds the unchanged 0.10m limit.

`filter-mls-gpu` exposes `MlsSmoothing::{smoothed_positions_with_gpu,filter_with_gpu}`
with a supplied runtime and self-inclusive neighborhood capacity. Radius search,
weighted local frames and constant/linear/quadratic fits run on GPU; CPU tree
construction and schema/attribute reconstruction remain. `GpuMlsInput` under
`gpu-wgpu` retains input-bound neighborhoods and owned position/mode results for
device consumers. H2D payload is 32 bytes per point plus 32 per tree node;
D2H is 16 bytes per point plus 32 status bytes, both zero when empty. No graph
readback or automatic CPU fallback occurs. Finite XYZ/extents and positive finite
nonzero radius squares are required; evaluated nonzero local extents with
subnormal/zero squares reject. Invalid rows/numerics zero all positions/modes.
Downloaded outcomes distinguish unchanged points, plane fallback and requested
polynomial fits. GPU f32 relative Cholesky rank tests differ from CPU f64
absolute Gaussian pivots; nearly singular fits can have large coordinate
differences. See [MLS precision and measurements](../notes/2026-10-13_gpu_mls_smoothing.md).

`feature-iss-gpu` exposes `IssKeypointDetector::detect_with_gpu` with a supplied
runtime and salient/non-max capacities. Both graphs, compensated local
covariance, eigenvalue saliency and local maxima run on GPU; CPU tree
construction, ordered flag compaction and schema/attribute gathering remain.
Capacities include self and must fit all rows, even rejected candidates.
Positions/extents must be finite, radii need positive finite nonzero squares and
gamma values must be finite in [0,1], stricter than ordinary CPU ISS. Evaluated
nonzero local extents with subnormal/zero f32 squares return numerical errors.
GPU f32 decisions can differ near thresholds/NMS ties from CPU f64. `GpuIssInput`
under `gpu-wgpu` exposes resident graphs/flags and eigen diagnostics. Default
readback is 4 bytes per input point plus 32 status bytes; optional score readback
costs 16 bytes per point plus 32 status bytes. No intermediate graph readback or
automatic CPU fallback occurs. See [ISS evidence](../notes/2026-10-13_gpu_iss_keypoints.md).

`register-fpfh-gpu` exposes
`fpfh_descriptors_gpu(runtime, cloud, radius, neighbor_capacity)` alongside the
ordinary CPU descriptor API. CPU validation and KD-tree construction precede
explicit GPU input upload; radius search, SPFH and weighted FPFH run on device.
Capacity includes self and must fit every radius match; incomplete/malformed
rows set GPU status, zero all descriptors and reject at readback. No automatic
CPU fallback or placement change occurs. `GpuFpfhInput` under `gpu-wgpu` also
exposes input-bound reusable `GpuFpfhNeighborhoods` and resident descriptors
for later device consumers, which must check the GPU status buffer. The two
feature passes share a submission without neighborhood/SPFH readback. Output
transfer is 132 bytes per point plus 16 status bytes; GPU angular rounding can
change bins near boundaries. Descriptor matching/RANSAC remain CPU. See
[FPFH measurements](../notes/2026-10-12_gpu_fpfh_descriptors.md).

`gpu-wgpu` exposes `GpuQueryBatch::upload` and
`GpuKdTree::{nearest_k_resident,radius_neighbors_resident}` for shared GPU
neighborhoods. Target/query uploads are explicit and reusable; search submits
without payload copies or a host wait and returns owned `GpuNeighborhoods`.
Each row stores two u32 counts plus bounded index/distance pairs. k=0..=64 is
stably sorted by distance with CPU selection ties. Radius membership retains
CPU traversal order, counts all matches, and supports capacity=0 count-only
search. `download` accounts for padded readback bytes; complete row access
rejects overflow while `truncated_rows` explicitly returns a traversal prefix.
Self/duplicate points are included. GPU feature consumers must check counts
and preserve query/target semantics. GPU FPFH/ISS/MLS consume radius rows as described
above; other feature algorithms retain their separate CPU APIs. Native correctness and completed/
resident measurements are in
[the neighborhood note](../notes/2026-10-12_gpu_bounded_neighborhoods.md).

`register-icp-gpu` enables the explicit
`IcpRegistration::align_with_gpu_correspondences(runtime, source, target)` API.
The target KD-tree is built on CPU and uploaded once; each update and fitness
query traverses it on GPU. Transform application, trimming, Kabsch, ordered f64
fitness accumulation and convergence remain on CPU. The returned
`GpuIcpDiagnostics` includes the ordinary trace, completed query count and
target/query upload plus result download bytes. Ordinary ICP remains CPU.
`GpuKdTree::upload` / `nearest_one_within_download` under `gpu-wgpu` also expose
this reusable search directly, with runtime ownership, finite-target validation,
inclusive gates and near-first tie order. Native synchronous execution is
verified; browser execution is not claimed. See
[correspondence measurements](../notes/2026-10-12_gpu_icp_correspondences.md).

The same `register-icp-gpu` feature adds
`IcpRegistration::align_with_gpu_reduction(runtime, source, target)`.
Original source and original-order target stay resident alongside the tree.
GPU transforms, bounded nearest matching, compensated centroid/fitness and
centered cross-covariance reductions share one six-pass submission per
evaluation; only 176 bytes of owned moments/status are downloaded. CPU Horn
pose solving and convergence remain. Post-update moments feed the next update
and final fitness. Input payload is 16*S + 32*T + 32*M bytes for source/target/
tree-node counts; no repeated point upload or match readback occurs. The API
requires `trim_fraction=1`, finite extents and a normal positive combined
normalization extent square (coincident inputs use scale 1). Device f32 math
can change matching/pose/stopping near gates and degeneracy. `GpuIcpInput`/
`GpuIcpMoments` under `gpu-wgpu` expose this reusable boundary directly. See
[resident ICP evidence](../notes/2026-10-13_gpu_icp_reduction.md).

`register-gicp-gpu` also exposes
`GicpRegistration::align_with_gpu_correspondences(runtime, source, target)`.
The device reuses a target for unbounded nearest matching, followed by GICP's
original host rejection gate. CPU Hessian/gradient sums, pose solving and
fitness accumulation are retained. CPU k-NN covariance is used when
`covariance_radius` is None; configured radius covariance uses the provided
runtime and adds its complete uploads and padded readbacks to the receipt.
This explicit path requires finite input/transformed XYZ and propagates GPU
errors. Legacy native alignment retains its shared-runtime covariance fallback;
the legacy shared-runtime route uses CPU covariance on WASM. Browser execution
of the explicit GPU route is not verified. See
[GICP measurements](../notes/2026-10-12_gpu_gicp_correspondences.md).

The same `register-gicp-gpu` feature adds `GicpRegistration::align_with_gpu_reduction`.
Original source/target/tree and packed covariances stay resident during GPU
transform/search, covariance rotation/inversion and Hessian/gradient/fitness
passes. A centered/scaled parameter basis and compensated sums produce an owned
256-byte result per evaluation. CPU solves 6x6 equations and converts the
linearized step to the existing world convention; post-update equations are
reused. `GpuGicpInput` and owned normal equations under `gpu-wgpu` expose this
boundary directly. Initial upload is 48*S + 64*T + 32*M bytes. CPU k-NN or
explicit radius-GPU covariance readback/re-upload remains preparation and is
included in receipts. Finite SPD f32 covariance rows, supported geometry/weight
scales, eight storage bindings and finite valid configuration are required.
Relative covariance Cholesky pivots <=1e-6 reject all result payload; no CPU
fallback occurs. f32 weights/transforms and normalized solving can change
matching/pose/stopping. See [GICP reduction evidence](../notes/2026-10-13_gpu_gicp_reduction.md).

The same `register-gicp-gpu` feature adds
`GicpRegistration::align_with_gpu_resident_covariances` and
`GpuGicpInput::upload_with_gpu_covariances`. Both bounded k-NN searches, centered
sample covariances and plane eigenvectors run on GPU and remain resident for
registration. Original positions and target tree are reused; source-tree upload
is counted. Payload H2D is 32*(S+T+source_nodes+target_nodes); only 256-byte
equations per evaluation are read back. CPU tree construction/6x6 pose solve
remain. Requires nonempty clouds, k=3..=64, epsilon as f32 in 1e-4..=1 and
covariance_radius=None, plus existing reduction limits/valid configuration.
Malformed neighborhoods or unsupported arithmetic zero all covariance rows;
equation status bit 16 rejects their use without fallback. f32 eigenvectors,
small/repeated spectra and CPU raw-product rounding can change results.
The existing reduction method retains its radius readback/re-upload route.
See [resident covariance evidence](../notes/2026-10-13_gpu_gicp_resident_covariances.md).

`GicpRegistration::align_with_gpu_resident_radius_covariances` and
`GpuGicpInput::upload_with_gpu_radius_covariances` add complete inclusive radius
search, centered/scaled covariance and eigenvectors retained for registration.
Positive explicit capacity includes self/duplicates; overflow rejects all
equations without truncation/fallback. Requires configured positive finite
radius with normal f32 square and epsilon=1e-4..=1. Fewer than three samples
use epsilon*I; other rows use compensated central sums and 32 max-pivot Jacobi
steps at physical 1e-20 stopping. Arithmetic/traversal/repeated eigenvalues can
differ from grid raw means/sweeps. Upload is 32*(S+T+source_nodes+target_nodes),
download is only 256 bytes per evaluation; CPU trees and 6x6 solve remain.
See [resident radius evidence](../notes/2026-10-13_gpu_gicp_resident_radius_covariances.md).

`GicpRegistration::align_with_gpu_resident_grid_covariances` and
`GpuGicpInput::upload_with_gpu_grid_covariances` prepare radius covariances
directly from resident original positions and CPU-built counting-sort grids.
The canonical grid f32 math is retained, with complete support bounded by the
caller's capacity including self/duplicates. No source KD-tree, neighborhood
row storage or covariance readback/re-upload is needed. Dense grids must fit
existing cell/device limits; sparse selection remains explicit. Grid indices
are leased scratch, covariance/equation outputs owned. Invalid support/data/SPD
zero all covariances and reject equations. CPU grids/target tree/6x6 solves
remain; download is 256 bytes per evaluation with complete upload receipts.
See [resident grid evidence](../notes/2026-10-13_gpu_gicp_resident_grid_covariances.md).

`register-icp-point-to-plane-gpu` adds
`PointToPlaneIcp::align_with_gpu_reduction` and diagnostic iteration history.
`GpuPointToPlaneInput` under `gpu-wgpu` explicitly retains originals/tree and
raw target normals; five device passes transform, match, prepare plane rows and
reduce centered/scaled Hessian/gradient/raw-normal plane squared error. Normals
use one common scale, preserving relative lengths and zero weights. CPU solves
6x6 equations and converts the existing world step; post-update equations are
reused. Upload is 16*S + 48*T + 32*M bytes, explicit owned equations readback is
256 bytes per evaluation (updates plus one). Finite normals, supported scales,
a finite unit pose, positive finite nonzero distance square, at least six
configured correspondences and one iteration, eight storage bindings and valid
stopping thresholds are required. No fallback occurs; ordinary CPU behavior is
unchanged. f32 arithmetic/normalized solving can change matching and stopping.
See [point-to-plane evidence](../notes/2026-10-13_gpu_point_to_plane_reduction.md).

Standalone `gpu-wgpu` k-NN/radius normals and plane covariances reuse lazy
per-runtime pipelines and bounded storage scratch. Compute and the existing
host-result copy share one submission. These APIs still upload their input
slices and synchronously return CPU results; no resident-frame contract or
automatic placement changes. Kernel math stays unchanged, and invalid indices
or unsupported storage/uniform/dispatch sizes return errors before allocation.

GPU gray box blur under `gpu-image` keeps the previous integer mean and border
semantics. Positive odd boxes with area at least 81 and at least 65,536 total
pixel/sample visits use two device passes and a four-byte-per-pixel private
intermediate; smaller workloads use one pass.
Host image transfers stay explicit. Profiled resident Vision chains reserve six
stages plus one per separable blur/morphology path (up to eight). Unsupported
pixel/sum/coordinate or device limits return an error before blur allocation.

GPU rectangular erosion/dilation under `gpu-image` retains replicate borders
and odd kernels in `1..=31`. Boxes with area at least 81 and at least 1,048,576
pixel/sample visits use exact horizontal/vertical extrema in one submission,
with four scratch bytes per pixel in the bounded image texture pool. Smaller
workloads keep the original one-pass shader. Coordinate and device limits fail
before allocation; no dependency, public signature or image readback is added.

| Profile | Enables | Intended use |
| --- | --- | --- |
| `mvp` | PCD, PLY, LAS/LAZ, COPC; voxel, normals, plane, Euclidean clustering, ICP, MVP pipeline | CPU-first end-to-end pipeline |
| `mvp-http` | `mvp` + HTTP byte-range COPC | Remote COPC input |
| `gpu-wgpu` | wgpu runtime and GPU kernels | Explicit GPU operation APIs |
| `register-icp-gpu` | ICP GPU correspondence search and resident transforms/moment reductions | Explicit correspondence or untrimmed resident reduction API; CPU pose solver |
| `register-icp-point-to-plane-gpu` | Resident transforms, nearest matching, plane Hessian/gradient/fitness reductions | Raw target normal weights; explicit receipts and CPU 6x6 pose solver |
| `register-gicp-gpu` | GICP correspondence/reductions, resident k-NN/radius or legacy covariance preparation | Caller-owned runtime and complete receipts; CPU 6x6 pose solver |
| `register-fpfh-gpu` | GPU radius search, SPFH and weighted FPFH | Explicit descriptor API and payload receipts; CPU matching/RANSAC |
| `feature-iss-gpu` | GPU radius graphs, covariance/eigenvalue saliency and local maxima | Explicit detection/receipts; CPU tree, compaction and attribute gathering |
| `filter-mls-gpu` | GPU radius search, weighted frames and order-0/1/2 surface fits | Explicit smoothing/receipts and per-point outcomes; CPU tree and attribute reconstruction |
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
| `spatialrust-search` | KD-tree | graph, bounded parallel radius/k-NN queries and graph construction | none; Criterion is dev-only |
| `spatialrust-semantic` | entities, embeddings, fusion, search | `semantic-model`: `OnnxEntityEmbedder` through an existing model session | `spatialrust-ai`/`spatialrust-tensor` optional; ONNX stays in the AI crate |
| `spatialrust-filtering` | voxel | GPU voxel, outlier, crop, FPS, MLS | wgpu/search optional |
| `spatialrust-features` | normals | ISS, orientation, boundary, GPU normals | wgpu/search optional |
| `spatialrust-segmentation` | plane and Euclidean clustering | GPU stages, DBSCAN, ground, primitives, region growing | wgpu/search optional |
| `spatialrust-registration` | ICP | GPU ICP correspondences, point-to-plane, GICP, GPU covariance, NDT, FPFH | wgpu/search optional |
| `spatialrust-gpu` | device markers only | wgpu runtime, AoSoA staging | wgpu/bytemuck/pollster optional |
| `spatialrust-pipeline` | MVP pipeline | GPU MVP stages | algorithm crates only |
| `spatialrust-interchange` | `interchange-gltf`, `interchange-openusd` | `tiles3d`: deterministic OGC 3D Tiles 1.1 `tileset.json` + `pnts` octree export; `tiles3d-copc`: bounded COPC hierarchy → tileset | `tiles3d-copc` pulls `spatialrust-io` + `spatialrust-core` for COPC node reads |
| `spatialrust-arrow` | no capability by default | C Data `__arrow_c_array__`, C Stream `__arrow_c_stream__`, C Device `__arrow_c_device_array__` exports/imports for point records | `arrow-c-data`, `arrow-c-stream`, `arrow-c-device`; no Arrow runtime dependency |
| `spatialrust-py` | Python binding surface | selected meta-crate features, including `export_tiles3d` / `export_copc_tiles3d`, Arrow C Data `__arrow_c_array__` / `__arrow_c_stream__`, `OnnxEntityEmbedder` (onnxruntime) | PyO3/NumPy |

## Execution contract

- GPU image transfer host transforms use complete RGBA row copies. Native
  1..=3-channel packing/unpacking from 1,048,576 pixels uses scoped row bands,
  capped by CPUs, 262,144-pixel chunks and height. Small/single-row/WASM paths
  stay serial. Budgets are per call; ROI/padding, metadata, owned results and
  explicit physical-byte receipts are preserved. See
  [transfer row measurements](../notes/2026-10-12_parallel_image_transfer_rows.md).
- `gpu-image` adds resident grayscale `clahe_gpu` and its three/four-pass profiled
  variant. Reflect101 tile histograms, clip redistribution, lookup rounding and
  bilinear interpolation match CPU CLAHE using u32-word binary64 arithmetic.
  Tile clipping/CDF use a workgroup-wide integer tree scan with analytic residual
  prefixes, replacing serial bin loops within the existing 4 KiB local limit.
  Larger tile histograms use row strips with shape-dependent lane assignment and
  reflected edge handling on the direct three-pass path. For up to 16 tiles with
  at least 65,536 padded pixels, a separate histogram pass splits each tile into
  8–32 chunks, then merges exact integer counts before clipping. Up to 512 KiB
  of additional submitted scratch is bounded by device binding/buffer limits;
  insufficient capacity selects direct execution. Split pipelines initialize
  lazily per runtime, and receipts include partial histogram write bytes.
  Geometry is computed on device, with no image/bin/geometry payload transfer.
  Temporary storage is 1 KiB per tile plus 16 bytes per source row/column,
  checked against binding limits and leased through final submission. Metadata
  is preserved; constant bytes follow CPU CLAHE rather than global equalization.
  Output download is explicit/full RGBA8; query collection reads 48 direct or
  64 split bytes, requiring three or four unused profile stages respectively.
  GPU limits and invalid grids/clips/channels reject before work allocation.
  Browser shader execution is not claimed. APIs remain provisional.
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
- GPU histogram partials use eight padded local bin replicas for
  4,194,304..=8,388,608 pixels when 8,224 bytes of local storage are supported.
  Calls outside that range or storage limit use the original kernel. Counts, two passes,
  global scratch, resident outputs and receipts are preserved. See
  [histogram accumulation measurements](../notes/2026-10-12_replicated_gpu_histograms.md).
- `spatialrust-search/parallel` bounds each radius/k-NN tensor query by available
  CPU parallelism and original chunk count, without a production dependency.
  Contiguous worker bands preserve backend chunk ranges and append result order.
  Inputs below 4,096 points and single-worker/single-chunk jobs run on the caller;
  empty inputs and parallel k-NN with `k=0` leave output untouched. Metadata is
  worker-sized, while output and local neighbor buffers remain result-sized.
  `spatialrust/search-parallel` exposes the same functions through the facade.
  Combining `search-graph` and `parallel` adds `knn_graph_parallel` and
  `radius_graph_parallel`, with one shared KD-tree, 256-point query chunks and
  identical serial edge order. The facade exposes these with `search-graph` plus
  `search-parallel`. Serial graph APIs retain caller execution. Graph storage
  scales with edges, potentially quadratically for radius graphs; u32 index
  limits and oversized k arithmetic are checked. See the
  [completed graph measurements](../notes/2026-10-12_parallel_neighbor_graphs.md).
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
- GPU CLAHE geometry uses exact exponent scaling for power-of-two tile sizes.
  General ratios retain binary64 long division; all ratio and axis-weight words
  are unchanged. Geometry remains regenerated per call, with existing scratch
  bounds, stages and receipt bytes.
  See [geometry ratio measurements](../notes/2026-10-12_exact_gpu_clahe_geometry.md).
- GPU nearest resize and Sobel validate runtime, dimensions, compute/binding
  limits and profiler capacity before allocation; resize checks coordinate-product
  overflow. Valid shader math, metadata, receipts and transfers remain unchanged.
  Shared-input Sobel tiling remains test-only after inconsistent paired results.
  See [spatial dispatch validation and measurements](../notes/2026-10-12_gpu_spatial_dispatch_validation.md).
- GPU RGB-to-gray distributes large flat workloads across bounded dispatch rows,
  while single-row workloads retain their original shader. Both pipelines are
  independently lazy; checked u32 pixel/index math, pre-allocation limits and
  profiler checks preserve exact BT.601 bytes, receipts and explicit transfers.
  See [bounded gray dispatch validation](../notes/2026-10-12_bounded_gpu_gray_dispatch.md).
- GPU CHW packing checks dimension/index/buffer/dispatch limits and profiler
  ownership/capacity before allocation. Normalization math, output order,
  one-pass receipts and explicit transfers are unchanged. Channel-lane and
  fixed-channel optimizations remain test-only after completed-call comparisons.
  See [AI packing validation and measurements](../notes/2026-10-12_validated_gpu_ai_packing.md).
- `gpu-image` upload/readback validate texture dimensions, checked RGBA row
  lengths, host capacity and padded readback-buffer limits before allocation.
  Pixel values, metadata and physical transfer receipts are preserved; texture
  transfers require no compute/storage-binding capability. See
  [image transfer validation](../notes/2026-10-12_validated_gpu_image_transfers.md).
- `filter-mls` uses bounded scoped CPU workers for independent point fits,
  sharing one KD-tree and preserving coordinate words, field order and metadata.
  Existing small/one-worker point staging remains serial; each concurrent call
  has its own worker budget. No GPU transfer or extra runtime dependency is added.
  See [MLS execution measurements](../notes/2026-10-12_parallel_mls_smoothing.md).
- `register-fpfh` computes SPFH and weighted FPFH rows with bounded scoped CPU
  workers sharing one KD-tree. The first phase joins before weighting begins;
  neighbor order, descriptor words and seeded registration results are preserved.
  Small/one-worker clouds retain serial staging, with no additional dependency.
  See [FPFH execution measurements](../notes/2026-10-12_parallel_fpfh_descriptors.md).
- `feature-boundary` computes tangent-plane boundary masks with bounded scoped
  CPU workers sharing one KD-tree and writing disjoint mask bands directly.
  Each worker reuses angular scratch; neighborhood and point order, threshold
  decisions, selected attributes and metadata are preserved. Small/one-worker
  calls stay serial, with no additional dependency or device transfer.
  See [boundary execution measurements](../notes/2026-10-12_parallel_boundary_detection.md).
- `feature-iss` shares one KD-tree across bounded covariance and local-maximum
  phases. Saliency joins before selection reads it; parallel selection uses one
  temporary byte per point. Small/one-worker calls keep direct serial index
  collection. Equal maxima, selected point order, schema, all attribute types
  and metadata are preserved, without another dependency or device transfer.
  See [ISS detection measurements](../notes/2026-10-12_parallel_iss_keypoints.md).
- `metrics-distance` shares each target KD-tree across bounded CPU query workers
  for Chamfer/Hausdorff statistics. A reusable optional-distance window holds at
  most one 16,384-point chunk per worker; ordered f64 accumulation preserves
  exact statistics without worker-local sums. Each band reuses its one-neighbor
  query buffer. Small/one-worker directions keep
  the original serial loop. Neither dependencies nor device transfers change.
  See [cloud distance measurements](../notes/2026-10-12_parallel_cloud_distances.md).
- `search-kdtree` carries an axis minimum through median selection to bypass
  scans for minimum/NaN pivots. Exact point permutations, node/split words and
  query order remain unchanged, including ties. Construction stays serial and
  retains existing storage and dependencies. See
  [KD-tree construction measurements](../notes/2026-10-12_exact_kdtree_minimum_pivots.md).
- `search-kdtree` single-neighbor queries use one stack-resident optional result
  instead of a temporary Vec. Exact indices/distance words, traversal ties and
  nonfinite behavior remain compatible with k=1 queries; no dependency or
  device transfer changes. See
  [scalar search measurements](../notes/2026-10-12_allocation_free_nearest_one.md).
- Shared dense-grid bounds validate axes/coordinates/cell scales and perform
  checked dimension/product arithmetic before u32 narrowing. Overflowing spans
  return errors and `uniform_grid_fits` returns false before CPU grid or GPU
  work. Accepted f32 bounds, the exact cell cap and positive-infinite-radius
  one-cell behavior remain compatible. See
  [grid validation regression receipt](../notes/2026-10-12_validated_grid_bounds.md).
- `spatialrust-search/parallel` adds explicit `KdTree::from_slices_parallel`
  and `from_point_cloud_parallel`, also exposed under facade `search-parallel`.
  Native subtrees use disjoint point/node ranges in one preallocated preorder
  array, reusing canonical axis/partition code. Layout, split bits and query
  ties remain exact. From 32,768 points, workers are bounded by available CPU
  parallelism and one per 16,384 points; small/one-worker/WASM calls stay serial.
  Ordinary constructors retain serial placement. Resident GICP GPU methods
  opt into this host preparation; GPU math and receipts are unchanged. This
  is a per-call worker bound, not a global thread/stack budget. See
  [parallel tree measurements](../notes/2026-10-13_parallel_kdtree_build.md).
- Shared CPU `build_grid` uses stable reverse scattering and one cell-offset
  table, eliminating the auxiliary cursor clone. Logical live heap payload
  drops by `4 * (cells + 1)` bytes; returned CSR order/sentinels and GPU uploads
  remain identical, with O(points + cells) work. Native `parallel` builds use
  bounded scoped workers from 8,388,608 offset entries with at least 1,024 entries
  per point; mapping and scattering remain serial. Small/denser/one-worker/WASM
  calls keep scalar prefixes.
  `gpu-wgpu` enables this search feature for host preparation and existing native
  dense-root dispatch; WASM dense roots stay serial. Worker/handle metadata is
  worker-sized, not a process-wide CPU/stack budget. See
  [parallel grid prefix measurements](../notes/2026-10-12_parallel_grid_prefix.md).
  Standalone GPU radius normals/covariances reuse this preparation. See
  [grid construction measurements](../notes/2026-10-12_reduced_cpu_grid_scratch.md).
- Dense-grid `spatialrust-search::euclidean_cluster_roots` reads adjacent x cells
  as stable CSR row intervals, with clipped y/z traversal. Ordered neighborhoods,
  radius arithmetic and minimum roots remain identical under serial execution
  and the existing `parallel` feature. No heap scratch or dependency is added;
  GPU sparse-grid clustering and the segmentation KD-tree CPU path are separate.
  See [CPU grid measurements](../notes/2026-10-12_cpu_grid_row_traversal.md).
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

Host-returning GPU algorithms expose their transfer boundary explicitly.
Euclidean clustering now builds/sorts/compacts sparse cells and labels
minimum-index components on wgpu. Initialization, atomic root hooking and final
compression use three device passes in one submission after cell construction;
no host union/find or iterative convergence readback remains. Finite calls upload
12 bytes per point and download 4 root bytes per point plus the existing 4-byte
cell-count synchronization. Positive-infinite tolerance uses a device zero-root
initialization without XYZ upload or cell construction. Host validation/origin
reduction and high-level cluster-size filtering/labeled-cloud assembly remain.
Receipts report `gpu-component-labeling`; placement thresholds stay unchanged.
See [GPU component labeling](../notes/2026-10-12_gpu_cluster_union.md).

GPU RANSAC plane scoring additionally parallelizes the point direction from
1,024 points, reducing exact integer inlier counts within each 256-lane group
while retaining per-point f32 distance arithmetic. Smaller inputs retain scalar
point traversal. Both pipelines are lazy/runtime-cached; bounded hypothesis
strides, checked indices/limits and one compute-plus-score-copy submission keep
uploads and score readback explicit. See the
[RANSAC validation receipt](../notes/2026-10-11_parallel_gpu_ransac_plane.md).

Voxel key sorting uses stable four-bit GPU radix passes over signed i32 XYZ
coordinates, preserving increasing source indices inside each cell. Sorting
uses the legacy histogram/prefix through 524,288 points. Larger sorts reuse
offset scratch for histogram-generated earlier-digit subtotals; a
work-efficient tree scan avoids repeated global reads of earlier histograms.
The 24 digits/72 passes, buffer sizes, stage timing and transfers are retained.
The cached large radix histogram also reuses each digit's eight mask-word counts
as transposed exclusive rank prefixes. One prefix lookup and one partial-word
popcount replace repeated preceding-word counts after the existing barrier.
The 524,288-point cutoff, stable ordering, global scratch and passes stay
unchanged. See [word-rank measurements](../notes/2026-10-12_cached_gpu_radix_word_ranks.md).
Exact-sized entries stay on the device and no longer need padded
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

Dense-grid radius normals and plane covariances gather neighboring x cells in
contiguous rows with preserved point/accumulation order. Host grid construction,
uploads and synchronous result mapping stay explicit; see
[row traversal evidence](../notes/2026-10-12_gpu_grid_row_traversal.md).
Plane covariance's insufficient-neighbor fallback overwrites both output rows,
including cross terms and padding, rather than relying on fresh storage zeros.
See [dirty-output regression evidence](../notes/2026-10-12_complete_gpu_covariance_outputs.md).

GPU runtime tests require a compatible adapter. On GPU-less CI, compile the
GPU test targets and execute them on a runner with a wgpu adapter.

## Deliberately separate future crates

The following are not core features and must remain separate integration
crates: `spatialrust-ros2`, `spatialrust-ai`/DLPack/ONNX, and the CUDA backend.
