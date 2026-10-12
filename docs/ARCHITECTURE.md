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
Image uploads validate positive u32 dimensions against the runtime's 2D texture
limit, checked RGBA row/storage bytes and host Vec capacity before texture or
packing allocation. Readback validates runtime ownership and logical channels,
then checks row alignment, padded staging bytes against `max_buffer_size`, and
host staging/output capacities before buffer allocation or copy submission.
Transfer planning uses no compute or storage-binding limit: uploads write a
texture and readback stages through an unmapped COPY_DST/MAP_READ buffer.
Receipts retain four physical bytes per pixel rather than staging-row padding;
rejections leave them unchanged. Internal image construction also checks u64
physical storage arithmetic. Plans do not impose a new retention/peak-memory
budget or change successful transfers, metadata, kernels or public signatures.
RGB-to-gray retains the original 256-lane linear shader when its group count
fits one dispatch axis. Larger inputs balance those flat groups across bounded
dispatch rows. Padding groups return before pixel-index multiplication to prevent
u32 wrap; source dimensions, pixel count, compute/binding limits and one free
profiler stage are validated before work allocation or pipeline initialization.
Linear and multi-row pipelines are independently lazy within one runtime-owned
family and share the existing 16-byte uniform/layout. Integer BT.601 output,
gray metadata, one pass, four physical write bytes per pixel and 16 query bytes
on explicit collection remain unchanged. There is no new scratch or payload copy.
GPU AI CHW packing validates positive dimensions, logical channels, signed texture
coordinates, u32 pixel/element indices, storage-binding/output-buffer sizes,
16x16 dispatch limits and the 64-byte uniform before allocation or cache setup.
Profile ownership and one free stage are checked before work begins. Finite
scale/mean and finite nonzero standard deviations retain their existing contract;
the original f32 normalization and CHW shader, single pass, storage retention
and explicit payload/query collection remain unchanged. Experimental channel-lane
and fixed-channel kernels remain native test fixtures after completed-call measurements.
Gray GPU box blur retains the integer full-box mean with one final half-up
rounding. Boxes with area at least 81 and at least 65,536 total pixel/sample
visits use horizontal unrounded u32 sums and a vertical reduction in one
submission; smaller workloads retain one-pass traversal.
The exclusive intermediate buffer is four bytes per pixel and uses submitted
scratch leases. Oversized separable kernels scan only valid coordinates and
weight replicated edge pixels analytically; constant-zero padding retains the
full kernel divisor. Pixel, coordinate, sum and device limits reject before
allocation. Both paths use bounded grid strides and independently lazy runtime
pipelines. GPU rectangular morphology retains odd kernels in `1..=31` and
replicate borders. With box area at least 81 and at least 1,048,576 pixel/sample
visits it reduces horizontal then vertical minima/maxima. Clipping valid
coordinates is exact because repeated edge values do not alter an extremum.
Both passes share one submission; the exclusive RGBA8 intermediate is four
bytes per pixel and returns to the bounded image pool only after submission.
Small workloads keep the original full-box pass. Coordinate, texture, dispatch,
uniform and workgroup limits reject before output/scratch/cache allocation.
The separable pair is lazy within the existing spatial pipeline family.
The resident Vision chain records six passes for small workloads and one extra
for each separable blur/morphology path (up to eight), reserving all timestamp
capacity before recording.
GPU nearest resize and Sobel validate their runtime identity, positive source/
output dimensions, signed coordinates, texture/compute/binding limits and one
free profiler stage before output allocation or spatial-pipeline initialization.
Resize checks that each maximum output-coordinate/source-dimension product fits
u32 before division. Valid work retains the original shader and one output pass,
with four physical write bytes per pixel and 16 query bytes on explicit collection.
No global scratch, intermediate texture, pipeline or payload transfer is added;
Vision chains keep their six-to-eight-pass contract.
Optional `GpuStageProfiler` instrumentation is shared by image and point-cloud
chains in `spatialrust-gpu`. Timestamp-enabled runtime construction requests
only adapter-supported pass queries; default runtimes enable none. Profiled
entry points reserve bounded capacity before execution. Query collection is an
explicit native synchronization/readback boundary with a separate byte receipt.
GPU pass times exclude CPU setup, transfers and inter-pass gaps; completed wall
time remains a separate measurement. Failed recordings cannot be collected.
Standalone host-returning k-NN normals, grid radius normals and plane covariance
estimators share bounded execution. Each kind lazily initializes its own pipeline
on its owning runtime; shader math and neighborhood order stay unchanged.
Storage inputs/output use exclusive submitted scratch leases, with full input
uploads on every call and no content cache. Compute and result staging copy share
one submission before scratch is returned. Mapping remains the explicit
synchronous boundary of these host-returning APIs. Uniform and staging buffers
are still created per call; the existing scratch pool bounds retained storage.
Point, index, buffer, uniform and dispatch limits reject before device allocation.
Grid cell storage is checked before CPU counting-sort allocation. Resident AoSoA
frame execution and its timing/transfer receipts remain separate.
Dense-grid radius normals and plane covariances merge the valid neighboring
x cells into one counting-sort interval per y/z row, for at most nine row
intervals per gather instead of 27 cell intervals. Both mean and covariance
gathers retain the original z/y/x/point order, distance test and floating
operations. Clamped x endpoints are reused across both gathers. The existing
positive i32 dimensions and bounded cell/sentinel storage also bound these
endpoints; no buffer, workgroup memory, barrier or transfer is added.
The sparse AoSoA radius kernel has a dedicated shader matching the previous
generated source. Its signed-key binary searches and per-cell traversal remain
independent of dense CSR row gathering.
Voxel boundary compaction uses a 256-lane shared-memory scan with recursively
scanned block totals and offset propagation. All levels stay device-resident
in one submission; the existing four-byte cell-count synchronization remains
explicit. Temporary hierarchy storage shrinks by a factor of 256 per level.
Radix, boundary flags, prefix levels and atomic count scratch use a private
runtime-owned buffer pool. A scratch batch owns every lease until its encoder,
including the scalar count copy, is submitted; abandoned batches are not
recycled. Submitted buffers can be reused through queue ordering without a new
wait or device copy. Caller-owned segment outputs stay outside this pool.
Large voxel radix sorts reuse offset scratch for per-workgroup earlier-digit
subtotals written by the histogram pass. The prefix pass reads only its own
digit slice, scans chunk totals with a work-efficient tree and overwrites offsets
after all lanes finish reading. This removes repeated global earlier-digit loads
without another scratch allocation or pass. Stable ranks, 24 digits/72 passes, bounded
dispatch, queue ordering and the explicit compaction count readback are retained.
Up to 2,048 input workgroups (524,288 points), legacy histogram/prefix kernels
remain specialized without the extra subtotal barrier. Larger inputs lazily cache
two optimized pipelines using a compile-time override; scatter is shared. This
internal GPU selection retains small-input work and adds no CPU/GPU placement.
The cached histogram's existing per-digit count loop also writes transposed
exclusive prefixes for eight 32-bit mask words. After the existing subtotal
barrier, each point reads its prefix and popcounts only preceding bits in its
own word. Local ranks stay stable and exact, including partial input groups.
All 128 prefix words are overwritten by sixteen digit lanes. This adds 512
bytes of histogram workgroup storage, keeping its 1,088-byte footprint below
the already-required 2,052-byte prefix kernel. No barrier, global allocation or
pass is added; legacy specialization, pipeline selection and scatter are retained.
RANSAC plane scoring retains its explicit XYZ/hypothesis uploads and score
readback. Below 1,024 points the scalar per-hypothesis loop is retained; larger
inputs assign a 256-lane workgroup to a hypothesis and tree-reduce integer
inlier counts. Plane construction and each point's f32 distance expression
retain their operation order. At most 256 groups stride over hypotheses, with
subtraction guards preventing u32 wrap and a barrier before model reuse.
Serial/parallel pipelines are independently lazy and cached per runtime. Input
indices and buffer/compute limits are checked before allocation. The score copy
and compute pass share one submission; no global partial-count scratch is added.
Upload/CHW buffers, point scratch and RGBA8 textures each retain at most 64 MiB
by default, independently, and at most eight resources per size. Full pools
reject incoming resources; a budget decrease below retention clears the cache,
and zero disables retention. Budgets cover recycled payload bytes, not active
outputs, uniforms, mapped staging, allocator/driver overhead or peak process
memory. `GpuResourcePoolStats` counts requested allocations, cache hits and
retained/discarded resources; clearing preserves these lifetime counters.
Image/tensor ownership uses a retained runtime identity token rather than a
runtime address, so moving a runtime preserves valid resources.
GPU channel histograms under `gpu-image` use exact integer RGBA8 loads and two
compute passes in one submission. At most 256 workgroups build private 256-bin
partials (at most 256 KiB), then independently reduce each bin into a live
1 KiB `GpuImageHistogram`. Every partial and output bin is overwritten, including
zero counts in reused storage. Scratch follows the submitted-lease protocol;
live histogram outputs stay outside the scratch pool until explicit recycling.
Channels are logical source components, with no implicit luma conversion.
Inputs above u32::MAX pixels and insufficient compute/buffer limits are rejected
before allocation. Counts and their sum therefore remain exactly representable.
Optional profiling records exactly two passes; native `readback_u32` explicitly
transfers 1,024 bytes and timestamp collection separately transfers 32 bytes.
Histogram construction has no payload readback or implicit host synchronization.
The public APIs are provisional; CPU references/Rayon are dev-only dependencies.
At 4,194,304..=8,388,608 pixels and at least 8,224 bytes of workgroup storage,
histogram partials use eight independent local bin replicas. One padding word
per replica separates equal-bin addresses; each lane accumulates into its
`lane & 7` replica and sums the eight counters after a workgroup barrier. All
used local bins are explicitly initialized. Integer mass stays bounded by the
original u32 pixel count. The extra partial pipeline is lazy per runtime;
inputs outside that range or lower shared-memory limits retain the original pipeline.
Group count, global partial/output storage, final reduction, two-pass profiling
and transfer receipts are unchanged. Native per-lane run compression remains
an experimental fixture after distribution-dependent completion measurements.
Resident histogram consumers include `histogram_cdf_gpu`, a one-workgroup
inclusive tree scan of the 256 bins. It uses 1 KiB local memory, a separate 1 KiB
output, and no global temporary buffer or uniform. Integer counts remain exact
through u32::MAX; the last cumulative count equals source pixels. `GpuImageCdf`
inherits source transfer history, adds one device-write stage, and requires
explicit native readback/recycling. The histogram may be recycled immediately
after the CDF call submits its reads; queue order protects later pool reuse.
CDF recording adds one profiler stage (16 query bytes upon explicit collection).
Grayscale `equalize_histogram_gpu` consumes the histogram/CDF inside a five-pass
resident operation: two histogram passes, one CDF pass, a 256-entry lookup pass
and bounded grid-stride image application. Lookup scratch is 1 KiB, leased until
both lookup/application passes are submitted; internal histogram/CDF outputs are
recycled after their dependent reads. The final texture remains caller-owned.
A u32-word arithmetic model reproduces the CPU's binary64 reciprocal/multiply
rounding at exact half-byte boundaries, without requiring float64 shaders or
changing CPU output. Non-half rational values are rounded exactly; constant
images retain their byte and all images retain metadata. Construction performs
no payload readback or host wait. Five profiled passes require five free stages;
explicit timestamp collection returns 80 query bytes separately from payloads.
Other channel counts and unsupported counters/device limits fail before work
allocation. Empty GPU images remain unsupported. Normalized public CDFs and
adaptive tile equalization remain separate future contracts.
Resident grayscale `clahe_gpu` runs three or four passes in one submission:
axis geometry, histogram/lookup, then interpolation/application. Lookup storage
is 1 KiB per tile; axis geometry is 16 bytes per source row/column, both private
scratch leased through application submission. Reflect101 padding and exact
clipped-count redistribution match the CPU. Per-tile cumulative counts use a
workgroup-wide tree scan of bounded counts and clipped excess. The residual
redistribution prefix is computed analytically from the CPU's bin positions,
avoiding a second scan and serial 256-bin clipping/CDF loops. Histogram bins and
scan storage use about 3 KiB of workgroup memory within the existing 4 KiB limit;
the minimum workgroup storage requirement remains 4 KiB. Direct histograms use
three passes. Tiles with at most 256 padded pixels assign one pixel per lane.
Larger direct tiles partition columns/rows with power-of-two strides, avoiding per-pixel flat-index
division. Short tiles use more columns to keep lanes occupied; narrow tiles use
more rows. Reflect101 computes a remainder only outside the source bounds, and
subtraction guards keep stride increments within the validated u32 dimensions.
For at most 16 tiles with at least 65,536 padded pixels each, a separate
histogram pass partitions each tile into `ceil(area/8192)` contiguous chunks,
bounded to 8–32 chunks. Up to 512 independent workgroups write 256 integer bins
per chunk; fewer available groups stride through all chunks. The lookup pass
merges these exact counts before the same clipping/scan/rounding. This adds at
most 512 KiB of private scratch and one pass; insufficient buffer/binding limits
select the original direct path. Both split pipelines initialize lazily in the
owning runtime. Partial buffers stay exclusively leased through application
submission and follow the existing byte-bounded scratch retention controls.
Receipts include `clahe_histogram_gpu` and the partial histogram write bytes.
Shared integer-word lookup rounding and a finite non-negative binary64 arithmetic
subset preserve the CPU's division, subtraction, multiplication/addition
operation order without shader float64.
Geometry ratios with positive power-of-two divisors scale the exponent of an
exact u32 value, returning identical binary64 words without fractional long
division. Other ratios retain the original division and rounding algorithm.
All axis entries are still regenerated on device per call using the same bounded
scratch; no geometry-retention cache or additional transfer is introduced.
Equal integer corners bypass interpolation only when the final byte is invariant
under CPU rounding. No payload readback, geometry upload or host wait is hidden.
The clip limit is a scalar host parameter; per-tile counts, padded coordinates,
buffer bindings and compute limits are checked before allocations. Caller-owned
output preserves metadata; constant bytes may change as in CPU CLAHE. Direct
work requires three free profile stages and collects 48 query bytes; split work
requires four stages and collects 64 bytes. Capacity is checked before work
allocation. Browser shader execution is still unverified; adaptive contrast APIs remain provisional.
CPU image analysis under `imgproc-analysis` uses optional Rayon row bands for
large thresholds, u8 histograms and histogram-equalization lookup application.
Tasks are bounded by image rows and the current pool; histograms merge exact
integer counts from 256-bin local arrays. CLAHE independently groups lookup
tiles and output rows into at most the current pool's worker count at one
million pixels. Tiles retain their original clipping and reflect-101 sampling;
parallel interpolation shares one horizontal geometry table proportional to
image width, without changing f64 operation order or ties-to-even rounding.
Strided/ROI views exclude storage padding. Small inputs and one-worker pools
keep scalar interpolation; a single output row cannot dispatch row bands, but
large CLAHE lookup grids can still dispatch independent tiles. CLAHE grid size
and lookup byte size are checked before allocation. The existing lookup matrix
and output buffer remain image/grid-sized; per-active-tile scratch is 256 usize
bins, and horizontal geometry is shared rather than duplicated per worker.
CPU point queries under `spatialrust-search/parallel` group contiguous original
`SpatialTensor` chunks into scoped workers, capped per call by available CPU
parallelism and chunk count. Worker bands differ by at most one chunk; callbacks
keep their original ranges and result order. Fewer than 4,096 points, one chunk
or one available worker run on the caller. Dispatch range/handle metadata scales
with workers instead of chunks; result buffers still scale with neighbor count.
This uses standard-library threads without an additional runtime dependency or
implicit device transfer; concurrent callers each have their own worker budget.
`search-graph` together with `parallel` adds explicit `knn_graph_parallel` and
`radius_graph_parallel` APIs. They build one shared KD-tree, reuse that bounded
chunk dispatcher with 256-point chunks, and produce worker-local edge lists in
source-point order. Ordered joins preserve the serial neighbor/edge sequence,
including KD-tree ties and duplicate-coordinate selection. Single-worker and
below-threshold graph calls keep one caller range. No all-query neighbor table
or distance matrix is staged; per-query neighbors and live local/final edge
vectors still consume result-dependent memory (radius graphs may be quadratic).
Existing serial APIs keep serial execution. Graph point counts must fit u32;
large k requests are capped by cloud size with checked reservation arithmetic.
MLS smoothing under `spatialrust-filtering/filter-mls` shares one caller-built
KD-tree and writes independent contiguous output bands with scoped CPU workers.
It reuses the existing point-staging policy: below 4,096 points, one available
worker, or fewer than two useful 16,384-point chunks, execution stays serial.
Worker count is capped by available CPU parallelism and useful chunk count.
Each point retains unsorted KD-tree radius-neighbor order, leave-one-out
selection and the original weighted covariance/polynomial arithmetic. The final position
vector is allocated once; fitting scratch remains local to each active point.
No all-query neighborhood table, new dependency or implicit device transfer is
introduced. Filter reconstruction preserves field names, schema, all non-position
buffers and metadata. Concurrent calls have independent per-call worker budgets.
FPFH descriptors under `spatialrust-registration/register-fpfh` reuse the same
point-staging policy for independent SPFH rows and weighted FPFH rows. Each
phase fills disjoint slices directly in its final descriptor array, sharing one
immutable KD-tree. The SPFH scope joins completely before weighting reads its
rows. Per-point neighbor order, Darboux/bin/normalization arithmetic and output
order are preserved; descriptor matching and seeded RANSAC retain their existing
execution. Two 132-byte-per-point descriptor arrays are retained during weighting,
with one radius-neighbor vector per active worker and no all-query neighborhood
table. Worker caps are per call, not a process-wide CPU or scratch-byte budget.
Boundary detection under `spatialrust-features/feature-boundary` uses the same
bounded staging policy, sharing one immutable KD-tree while filling disjoint
contiguous slices directly in the final boundary mask. Each worker reuses one
angle vector across its points and holds only the current radius-query neighbors.
Neighbor order, tangent-plane math, angle sorting and threshold comparisons stay
unchanged; no floating-point reduction spans workers. The detector gathers
selected points in original index order and preserves all field types and
metadata. Small/one-worker calls remain serial. Neighbor/angle scratch scales
with radius support per active worker; the worker cap is not a scratch-byte or
process-wide CPU budget. No device copy or new dependency is introduced.
ISS keypoint detection under `spatialrust-features/feature-iss` uses the same
bounded staging policy in two scoped phases sharing one KD-tree. Workers fill
disjoint slices of the existing four-byte-per-point saliency array; the first
scope joins before local-maximum selection can read any row. Parallel selection
writes a temporary one-byte-per-point mask and scans it in original index order,
dropping it before schema/attribute gathering. Serial calls retain direct index
collection and allocate no selection mask. Unsorted neighborhood order, f64
centroid/covariance arithmetic, eigenvalue ratios and the existing >= tie rule
are preserved. Per-query neighbor vectors scale with radius support; worker
caps remain per call rather than a scratch-byte or process-wide CPU budget.
No floating-point reduction spans workers and no device transfer is introduced.
Cloud distances under `spatialrust-metrics/metrics-distance` build one KD-tree
per cloud and independently bound each direction's query workers with the same
point-staging policy. Parallel queries fill disjoint slices of a reusable window
of at most `workers * DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE` optional f32 distances.
Each window joins before its values are converted to f64, added and square-rooted
in original point order; sums are never regrouped across workers or windows.
Missing neighbors remain distinct from returned NaNs. One-worker directions use
the unchanged serial loop and allocate no score window. The two directions run
sequentially, so their windows are not live together. Scratch scales with worker
budget rather than total query count; KD-tree storage remains input-sized, and
worker caps are per call rather than a process-wide CPU limit. Each active band
reuses one one-neighbor result vector with the same KD-tree traversal and fresh
per-query accumulator, avoiding a heap allocation for every parallel query.
KD-tree construction carries the selected axis's already-computed minimum into
median selection. A minimum or NaN pivot cannot have a strictly smaller element;
selection reproduces the original pivot swap directly instead of scanning the
range. Moving right past a finite pivot raises the active range's lower bound;
NaN pivots leave it unchanged. General partitions, axis choice, leaf size and
recursive node construction keep their original behavior. Point permutations
and split words remain identical, preserving query ties and unsorted neighbor
order for downstream floating-point algorithms. This removes repeated scans of
minimum buckets while retaining serial execution and existing storage. General
quickselect worst-case complexity and parallel tree construction are unchanged.
Unbounded KD-tree single-neighbor queries hold one optional neighbor on the
stack, without heap scratch. Traversal and distance arithmetic match the k=1
accumulator: a candidate is rejected with `>=`, including the original NaN
replacement behavior; far branches use strict distance pruning once a neighbor
exists. Empty trees return no neighbor. Query validation, nonfinite results,
ties and bounded/k-NN APIs retain their existing semantics. This path introduces
no device copy, allocation, dependency or worker scheduling.
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
