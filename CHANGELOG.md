# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning policy (from 1.0.0)

From **1.0.0** onward SpatialRust follows SemVer strictly:

- **MAJOR** — breaking changes to APIs marked **Stable** in
  [`docs/API_STABILITY.md`](docs/API_STABILITY.md)
- **MINOR** — backward-compatible features; deprecations may land here
- **PATCH** — bug fixes and documentation corrections only

Breaking changes to **Stable** symbols require a major release, a CHANGELOG
entry, and migration notes. Deprecations are announced in a minor release and
removed no sooner than the next major (see `docs/API_STABILITY.md`).

## [Unreleased]

### Changed

- Share resident GPU Canny classification and root-neighbor flags in one atomic
  u32 per pixel, preserving classification bits during concurrent root marking.
  Remove the separate flag buffer for every aperture: private scratch falls to
  16*N+8 bytes (3) or 20*N (5/7), saving 33,177,600 bytes at 4K. Four storage
  buffers suffice; stage destination footprints fall to 32*N/36*N. Five/six
  passes, exact edge bytes, metadata and explicit image transfers retain behavior.

- Replace the capable-device resident Trimmed ICP selector's linear bin walk
  with an exact 256-bin integer sum tree and eight rank decisions. The wide
  pipeline uses 2 KiB shared storage under the existing limit and adds eight
  workgroup barriers; global scratch, digit passes, retained pairs, moment math
  and explicit transfers retain behavior. Paired completed-call evidence records
  the limited synthetic moment improvement, inconclusive selectors and adverse
  full-call cases.

- Halve resident GPU Canny's unrounded horizontal derivative/smoothing storage
  from 8 to 4 bytes per pixel for apertures 5/7, preserving signed integer values
  and final rounding. Private scratch falls from 28*N to 24*N, saving another
  33,177,600 bytes at 4K; stage destination footprints fall from 44*N to 40*N.
  Aperture 3, public options, edge bytes and explicit transfers retain behavior.

- Parallelize resident Trimmed ICP histogram-bin gathering with a lazy 256-lane
  selector on capable devices from 64 partial groups. The existing 64-lane path
  handles smaller inputs and lower workgroup limits. Stable distance/source-index
  selection, exact retained counts, moment precision, scratch, pass counts and
  explicit transfers remain; native paired measurements retain adverse cases.

- Halve resident GPU Canny gradient storage from 16 to 8 bytes per pixel by
  packing bounded signed derivatives exactly and retaining full u32 magnitude.
  Private scratch falls to 20*N+8 bytes for aperture 3 or 28*N for 5/7, saving
  66,355,200 bytes per 4K call. Buffer capacity checks and destination receipts
  use the smaller layout; public edge bytes and explicit transfer behavior remain.

- Add explicit resident Canny edge extraction under `gpu-image`: integer 3/5/7
  Sobel derivatives, exact L1/L2 threshold and directional tie rules, and complete
  GPU eight-connected hysteresis via weak unions and root strong-neighbor flags.
  Five/six passes share one submission; gradients, states and labels remain
  private on device. Binary output preserves metadata and explicit transfer
  receipts, with optional pass profiling and pre-allocation capacity checks.

- Add explicit resident Gaussian blur under `gpu-image`, with 1..=4 logical
  channels, 3/5/7 kernels, five border modes and optional two-pass timestamps.
  Q8/Q15 weights and wide integer final sums preserve specialized CPU u8
  quantization/rounding. Private scratch stays on device through one submission;
  output metadata and explicit transfer receipts are retained. Image payload
  readbacks and CPU/GPU placement remain caller-controlled.

- Improve resident trimmed ICP moment precision with a default DoubleSingle
  route: integer product residuals and preserved f32 rounding carry two words
  through coordinate normalization, centroid division and covariance products.
  Explicit `GpuIcpTrimPrecision::CompensatedF32` retains the earlier arithmetic.
  Fraction one keeps canonical untrimmed results; matching, all-gated fitness,
  CPU Horn solving and fixed 176-byte readbacks remain. Large gated registration
  regressions now enforce the unchanged CPU pose-agreement tolerance.

- Add explicit resident GPU ICP trimming under `register-icp-gpu`. Stable
  distance/source-index selection retains floor(trim_fraction*gated_count)
  pairs for moments while all gated pairs still define fitness/stopping.
  Integer device arithmetic preserves binary64 fraction rounding and floor;
  no point/match readback, repeated payload upload or CPU sorting is required.
  Each evaluation downloads 176 bytes, and fraction one uses canonical
  untrimmed kernels. CPU target-tree/Horn solving and convergence remain.

- Add explicit bounded parallel KD-tree construction under search `parallel`
  and facade `search-parallel`. Disjoint preorder node/point ranges retain exact
  canonical partition order and search ties, without unsafe, per-worker trees
  or a node merge. Native large inputs use available-CPU-bounded workers;
  small/one-worker/WASM builds remain serial. Resident GPU GICP opts into the
  same preparation with unchanged equations/transfer receipts. Ordinary tree
  constructors and CPU GICP retain serial placement.

- Add explicit resident grid GICP covariance preparation under
  `register-gicp-gpu`. Reuse original GPU positions and canonical radius-grid
  arithmetic, eliminating source KD-tree preparation, per-point neighborhood
  buffers and covariance readback/re-upload. CPU counting-sort grids and target
  tree remain; support capacity includes self/duplicates and overflow rejects
  all equations. Grid indices use private leased scratch with full receipts.
  Existing sparse/resident/legacy routes retain their explicit contracts.

- Add explicit resident radius covariance preparation for GICP under
  `register-gicp-gpu`. Inclusive bounded GPU neighborhoods, centered/scaled
  compensated central sums and plane eigenvectors feed resident equations
  without covariance readback/re-upload. Caller capacity includes self and
  duplicates; overflow rejects all output without truncation. Sparse rows
  preserve epsilon*I. Radius must be positive finite with normal f32 square,
  epsilon=1e-4..=1. CPU trees/6x6 solves remain; legacy routes retain behavior.
  Device f32 accumulation and Jacobi iteration can differ from grid results.

- Add explicit resident k-NN covariance preparation for GICP under
  `register-gicp-gpu`. GPU bounded neighborhoods, centered/scaled sample
  covariance and plane eigenvectors feed the existing resident equation loop
  without neighborhood/covariance readback or re-upload. Both original position
  buffers and the target tree are reused. CPU tree construction and 6x6 pose
  solving remain; receipts count both trees and fixed equation downloads.
  Supports k=3..=64 and epsilon=1e-4..=1 with covariance_radius=None. Invalid
  device preparation rejects all equations; existing CPU/radius routes retain
  their behavior. f32 eigenvectors can differ at small/repeated spectra.

- Add opt-in `register-icp-point-to-plane-gpu` resident transforms, nearest
  matching, plane Hessian/gradient and raw-normal plane fitness reductions.
  Reusable input and owned 256-byte equations retain explicit transfer receipts;
  CPU solves normalized 6x6 equations and converts the linearized world step.
  Target normals retain relative length weights (including zero rows); ordinary
  CPU point-to-plane ICP stays unchanged. Invalid configuration, unsupported
  numeric scales and device limits return errors without CPU fallback.

- Extend `register-gicp-gpu` with explicit resident transform and normal-equation
  reduction. GPU covariance rotation/inversion, Hessian/gradient and Euclidean
  fitness phases share one submission; each evaluation downloads 256 bytes.
  Centered/scaled equations use a CPU 6x6 solve and linearized world-step
  conversion. CPU k-NN or explicit radius-GPU covariance preparation retains
  full transfer receipts. Ordinary GICP remains unchanged; new finite/SPD and
  conditioning guards reject unsupported numerics without CPU fallback.

- Extend `register-icp-gpu` with explicit `align_with_gpu_reduction`: retain
  original source/target on GPU, transform, match and reduce centered moments
  on device, then download a fixed 176-byte result per evaluation. CPU Horn
  pose solving and convergence remain; post-update moments are reused for the
  next update and final fitness. Ordinary CPU/correspondence paths retain their
  math and trimming support. The new route requires untrimmed ICP and documents
  f32 matching/reduction/conditioning differences and full-call measurements.

- Add opt-in `filter-mls-gpu` smoothing for constant, linear and quadratic
  surfaces. Reusable input-bound radius rows feed GPU weighted frames and
  polynomial fits in three passes; explicit upload/download receipts and
  per-point unchanged/plane/polynomial outcomes accompany the result. CPU tree
  construction and schema/attribute reconstruction remain. Normalized f32
  Cholesky uses a different rank test from CPU f64; nearly singular fits can
  produce large coordinate differences, documented with adverse fixtures.
  Incomplete/malformed rows or numerical errors invalidate all device output.

- Add opt-in `feature-iss-gpu` keypoint detection: shared uploaded positions feed
  GPU radius graphs, compensated covariance, eigenvalue saliency and local
  maxima. Explicit payload receipts cover upload and flag/status download;
  incomplete/malformed graphs invalidate all output. CPU defaults, ordered
  host compaction and attribute gathering are retained. GPU selection can
  differ at numerical thresholds/ties; diagnostic eigen readback is explicit.

- Add opt-in `register-fpfh-gpu` descriptor computation: reusable radius
  neighborhoods feed GPU SPFH and distance-weighted FPFH passes, with explicit
  sample/tree/query upload and descriptor/status readback receipts. Input and
  neighborhood ownership are checked; incomplete/malformed rows invalidate
  the entire device result and return errors at download. CPU FPFH/RANSAC
  defaults remain unchanged; GPU angular/bin rounding is documented.

- Add explicitly uploaded reusable GPU query batches and resident KD-tree
  nearest-k/radius neighborhoods under `gpu-wgpu`. k=0..=64 retains CPU
  selection and stable distance order; radius rows retain traversal order and
  count all matches even when capacity is exceeded. Complete readback access
  rejects overflow; named truncated access and count-only search are explicit.
  Owned result buffers can feed subsequent GPU kernels without readback.

- Extend explicit GPU correspondence reuse to GICP under `register-gicp-gpu`,
  retaining unbounded matching followed by the original host distance gate,
  ordered Hessian/gradient reduction and CPU pose solving. Optional radius
  covariances use the caller's runtime and propagate errors on this explicit
  path. Covariance receipts count XYZ, sorted indices, grid offsets and the full
  padded readback; legacy alignment keeps its covariance fallback behavior.

- Add explicit reusable `GpuKdTree` upload and batched nearest-point readback,
  using GPU KD-tree traversal with CPU near-first tie order and inclusive gates.
  The optional `register-icp-gpu` API reuses the target for update and fitness
  matching and reports transfer bytes and query counts; CPU pose solving and
  ordinary ICP defaults are retained. Search exposes a borrowed backend-neutral
  layout without adding GPU dependencies to the core or search crates.

- Euclidean GPU clustering now performs minimum-index component union/find on
  wgpu after sparse-cell construction, replacing host labeling and cell-metadata
  readbacks with root readback. Receipts name `gpu-component-labeling`; host
  cluster-size filtering and placement thresholds are retained. Nonfinite
  coordinates, invalid reciprocals or unsupported limits reject before work;
  infinite tolerance initializes zero roots on device.
- GPU image transfers use complete RGBA row copies and bounded native CPU row
  bands for large 1..=3-channel packing/unpacking. ROI/padding, metadata, explicit
  transfer receipts and owned results are preserved; small/single-row/WASM
  transforms retain serial execution.
- Native dense-grid prefixes under `parallel` use bounded scoped workers for
  large sparse cell tables, retaining stable CSR indices/offsets and worker-sized
  metadata. `gpu-wgpu` enables this dependency-free search feature for host
  preparation. Small/denser/one-worker calls and WASM keep scalar grid prefixes;
  WASM dense-root dispatch also retains serial execution.
- CPU dense-grid construction reuses cumulative cell ends as final CSR starts
  through stable reverse scattering. This removes a cell-sized cursor copy
  from CPU clustering and standalone GPU normal/covariance preparation while
  preserving all sorted point indices and offsets.
- CPU dense-grid component labeling scans adjacent x cells as one stable CSR
  interval per valid y/z row, preserving ordered neighbors and minimum roots
  for both serial and existing parallel execution without new heap scratch.
- GPU grid radius normals and plane covariances scan adjacent x cells as one
  contiguous row interval, preserving neighborhood order and floating operations
  while reducing repeated grid-offset traversal without new transfers or scratch.
- Large GPU voxel radix sorts reuse per-digit mask counts as shared exclusive
  word prefixes for stable local ranks. The existing cached-path barrier and
  524,288-point cutoff are preserved, with unchanged ordering, 24 digits/72
  passes, global scratch, profiling and explicit transfers.
- Large GPU channel histograms use eight padded workgroup-local counter replicas
  to reduce concentrated atomic updates. Inputs outside 4,194,304..=8,388,608
  pixels or devices with less than 8,224 local bytes keep the original kernel. Exact
  counts, two passes, scratch bounds, resident ownership and receipts are retained.
- GPU image upload/readback validate texture dimensions, RGBA row arithmetic,
  host byte capacities and padded staging-buffer limits before work allocation.
  Unsupported transfers return errors without changing receipts or texture
  pools; valid pixels, metadata and explicit transfer accounting are preserved.
- KD-tree single-neighbor queries keep one stack-resident optional result instead
  of allocating a one-element Vec. The existing traversal, equal-distance ties,
  NaN replacement/pruning and overflow behavior remain unchanged, with no new
  dependency, public signature or device transfer.
- KD-tree construction skips partition scans when a pivot equals the known
  minimum or is NaN. The original pivot swaps, point permutation, split words
  and tree layout are preserved, retaining nearest/radius query order and ties.
  Duplicate-heavy inputs avoid repeated full scans without another allocation,
  dependency, public signature or device transfer; construction stays serial.
- Chamfer/Hausdorff cloud metrics share each target KD-tree across bounded CPU
  query workers under `metrics-distance`. A reusable score window holds at
  most one default point chunk per worker, with one reused query buffer per
  active band; f64 sums and maxima retain original
  query order and exact output words. Small/one-worker sides keep the original
  serial loop, with no new dependency, public signature or device transfer.
- ISS keypoint detection shares one KD-tree across bounded scoped CPU workers
  for covariance saliency and local-maximum selection under `feature-iss`.
  Saliency completes before selection; small/one-worker clouds stay serial.
  Equal maxima, selected point order, schema, attributes and metadata remain
  unchanged. Parallel selection uses one temporary byte per point, without a
  new dependency or public signature.
- Point-cloud boundary detection shares one KD-tree across bounded scoped CPU
  workers under `feature-boundary`. Workers reuse angular scratch and write
  disjoint mask bands directly; small/one-worker clouds stay serial. Boundary
  decisions, selected point order, schema, attributes and metadata are preserved
  without a new dependency or public signature.
- GPU CHW image packing validates output indices, buffer/binding/dispatch limits
  and profiler ownership/capacity before work allocation or pipeline setup.
  The original f32 normalization shader, CHW order, one-pass receipts and
  explicit transfers are preserved.
- GPU RGB-to-gray supports images beyond one dispatch axis by distributing
  flat 256-lane workgroups across bounded rows. Single-row work retains its
  original shader; dimensions, counters, device limits and profiler capacity
  reject before allocation, with unchanged luma bytes and explicit transfers.
- FPFH point-cloud descriptors share one KD-tree across bounded scoped CPU
  workers for both SPFH construction and neighbor weighting. Small/one-worker
  clouds stay serial; output order and exact descriptor words are preserved
  under `register-fpfh`, without a new dependency or public signature.
- GPU nearest resize and Sobel reject foreign runtimes, unsupported dimensions,
  device limits and insufficient profiler capacity before work allocation or
  pipeline initialization. Resize coordinate products are checked for u32
  overflow; valid output bytes, one-pass receipts and explicit transfers stay unchanged.
- MLS point-cloud smoothing shares one KD-tree across bounded scoped CPU
  workers using the existing point-staging policy. Small/one-worker clouds stay
  serial; neighborhood order, exact coordinate words, schema and attributes
  remain unchanged under `filter-mls`, without an additional dependency.
- GPU CLAHE axis geometry bypasses fractional long division for power-of-two
  tile sizes, retaining identical binary64 ratio and interpolation-weight words.
  General ratio arithmetic, scratch retention, pass count, explicit transfers
  and public signatures remain unchanged.
- Point-cloud k-NN/radius graph construction gains explicit bounded CPU parallel
  APIs under `search-graph` plus `parallel`, also exposed by the facade with
  `search-graph` plus `search-parallel`. One shared KD-tree and ordered worker
  joins retain the serial edge sequence, including ties and duplicates. Small
  inputs and one-worker calls execute on the caller. Graph indices are checked
  against u32 limits, and oversized k requests avoid addition/reservation-product
  overflow by limiting the search to the actual cloud size. Existing serial
  graph APIs keep serial execution; no production dependency is added.
- Resident GPU CLAHE splits histograms for up to 16 tiles of at least 65,536
  padded pixels, merging bounded integer partials before unchanged clipping and
  lookup rounding. Partial storage uses submitted scratch leases and falls back
  to the direct path when device binding limits are insufficient. Selected work
  records four stages and 64 explicit query bytes; direct work retains three
  stages and 48 bytes. Transfer receipts include the private histogram writes.
- Standalone GPU k-NN/radius normals and plane covariance estimators lazily reuse
  per-runtime pipelines and pooled storage buffers, and submit compute plus
  readback copy together. Shader arithmetic and returned float words remain
  unchanged. Index, buffer, uniform and dispatch bounds reject before device
  allocation; grid storage is checked before CPU counting-sort allocation.
  Invalid neighbor indices now return an error, and z-length errors report the
  actual z length. Empty inputs preserve the existing early-return behavior.

- GPU rectangular erosion and dilation use exact horizontal/vertical extrema
  for boxes with area at least 81 and at least 1,048,576 pixel/sample visits.
  Replicate borders and odd kernels in `1..=31` stay unchanged. Two passes share
  one submission and a four-byte-per-pixel pooled texture; small workloads keep
  the original single pass. Unsupported coordinate/device limits reject before
  allocation. Profiled Vision chains reserve six to eight stages for the selected
  blur and morphology paths, without image payload readback.

- GPU gray box blur uses unrounded horizontal u32 sums and a vertical pass for
  boxes with area at least 81 and at least 65,536 total pixel/sample visits,
  preserving the integer mean and border
  semantics with four scratch bytes per pixel. Both passes share one submission;
  host image transfers stay explicit. Small workloads retain one pass. Profiled
  Vision chains reserve six or seven stages, and unsupported pixel/sum/coordinate
  or device limits reject before blur allocation.

- Local GPU measurements omit the adapter model name in public documentation;
  backend, software versions, measurement conditions, and results are retained.

- GPU RANSAC plane scoring parallelizes points within each hypothesis from
  1,024 points and tree-reduces integer inlier counts, retaining f32 plane and
  distance arithmetic. Smaller inputs retain scalar traversal. Lazy per-runtime
  serial/parallel pipelines replace per-call compilation, at most 256 groups
  stride over hypotheses, and the score copy shares the compute submission.
  Invalid indices, empty point arrays and unsupported sizes/limits reject before
  GPU allocation; explicit XYZ/hypothesis uploads and score readback stay the same.

- GPU CLAHE builds larger tile histograms with power-of-two row strips, avoiding
  per-pixel flat-index division and reflecting only padded coordinates. Short and
  narrow tiles adjust lane assignment; tiles of at most 256 pixels keep direct
  one-pixel-per-lane traversal. CPU byte parity, scratch bounds, transfers and
  three-pass profiling remain unchanged. Native raw-count proofs and paired
  previous/current lookup diagnostics cover padding and degenerate shapes.

### Fixed

- Shared dense-grid bounds reject malformed axes, nonfinite coordinates,
  unrepresentable cell reciprocals/spans and overflowing dimensions before
  counting-sort or GPU work. Large dimensions can no longer wrap into small
  grids or panic; valid f32 bounds, the cell cap and positive-infinite-radius
  one-cell behavior remain compatible.
- GPU plane covariance estimation writes both output rows when fewer than three
  neighbors are available. Reused storage no longer leaks previous cross terms
  or NaNs into the isotropic fallback; supported-point math and transfers remain
  unchanged, with no buffer-clear pass or additional allocation.
- The local COPC HTTP test server waits for complete request headers within its
  existing 4 KiB bound before responding. Fragmented HEAD and Range requests no
  longer cause premature responses or connection resets; a read deadline bounds
  incomplete requests. Accepted sockets explicitly use blocking mode, including
  Darwin's inherited nonblocking sockets. Real TCP regressions force that mode
  on every platform and retain strict response assertions.

- Large GPU voxel radix sorts compute earlier-digit subtotals once per input
  workgroup and reuses existing offset scratch for the prefix pass. A parallel
  tree scan replaces redundant shared additions and global histogram reads,
  preserving stable signed XYZ/source-index order, all 72 sort passes, storage
  sizes and explicit transfers. Maximum-counter prefix, million-point live
  output/pool reuse tests and paired stage/completion measurements cover the path.
  Through 524,288 points the legacy specialization avoids the extra histogram
  barrier; larger inputs lazily cache the optimized histogram/prefix pair.

- GPU CLAHE parallelizes tile clipping and cumulative counts across all 256
  workgroup lanes. A tree scan of bounded counts/excess plus analytic residual
  prefixes preserves exact CPU redistribution and byte rounding, including
  maximum u32 tile counts. Global storage, dispatch limits, transfers and the
  three-pass API are unchanged. Independent million-bin parity and alternating
  old/new GPU pass measurements cover clip/grid/distribution variations.

- Parallel point-cloud radius/k-NN queries cap scoped workers by available CPU
  parallelism instead of spawning one OS thread per tensor chunk. Balanced
  contiguous chunk groups preserve callback boundaries, neighbor order and
  existing output prefixes; single-worker jobs run on the caller. Dispatch
  metadata scales with workers, while neighbor buffers remain result-sized.
  Minimal `parallel` feature tests no longer require KD-tree. Completed-query
  benchmarks and facade/minimal parallel CI profiles cover the bounded path.

- GPU AoSoA frame construction no longer computes and reads back unused voxel
  centroids; all source points and per-point normals remain device-resident.
  Frame receipts include scalar cell-count readbacks on construction/grid
  rebuild and device copies during attribute reduction. Empty frames report no
  transfers, and the frame benchmark waits for completed normal computation.

- Outlier filters reject nonfinite XYZ before neighborhood search, including
  impossible radius-neighbor requests, instead of processing invalid geometry.
- Neighborhood outlier filters reject invalid floating settings, bound SOR
  neighbor requests by available points and handle impossible radius neighbor
  counts without overflow. Rust and Python regressions cover extreme settings.
- ASCII PCD integer fields reject fractional, nonfinite and out-of-range values
  instead of silently rounding or saturating labels and IDs. Tests exercise
  U8/U16/U32/I32 validation in full and bounded streaming readers.
- Packed RGB PCD ASCII output now writes the float bit representation declared
  by TYPE F instead of an integer token, preserving colors on reload. ASCII
  TYPE U RGB is decoded as integer bits; malformed RGB layouts are rejected.
  Primary-color regressions cover ASCII, binary and bounded streaming reads.
- PCD rejects explicit POINTS values inconsistent with declared WIDTH * HEIGHT,
  including explicit zero counts. Missing POINTS is still inferred from the
  dimensions, and point-count-only headers remain accepted.
- PCD header parsing rejects overflowing dimension, field and payload sizes,
  zero-sized fields and payloads exceeding addressable allocation size before
  whole-cloud or streaming readers allocate point buffers.
- Compressed PCD validates declared decoded size against its header and rejects
  impossible LZF encoded lengths before allocating payload buffers. Tests cover
  truncated data, invalid references and valid overlapping LZF references.
- PCD scalar decoding and ASCII writing preserve float64 attributes and the
  full precision of signed/unsigned 32-bit integer fields instead of rounding
  them through float32. Regression tests cover ASCII/binary round trips and
  Python file alignment with intensity, labels, timestamps and rotated normals.

### Added

- Explicit bounded native keyframe study adapters retain a fixed keyframe or
  promote an already accepted pose using input-only geometric support. A
  separate coarse-SVD/fine-plane variant records each objective. Four official
  TUM sequence comparisons preserve all failed/missing poses, typed input
  attributes, frozen hashes and independent audits; none meets all fixed
  accuracy/coverage gates. The default estimator remains unchanged.

- `gpu-image` adds resident grayscale `clahe_gpu` and a three-pass profiled
  variant. Reflect101 tiles, clip redistribution and binary64 lookup/interpolation
  rounding match CPU CLAHE without payload readback or shader float64. GPU axis
  geometry and tile lookups use checked private scratch until application is
  submitted. Odd-grid/ROI/metadata/constant parity, 65,536 arithmetic/geometry
  bit comparisons, 66,049 interpolation boundary pixels, ownership/concurrency
  and completed-work placement benchmarks cover the provisional API; CPU output and dependencies are unchanged.

- `gpu-image` adds resident grayscale `equalize_histogram_gpu` and a five-pass
  profiled variant. Histogram/CDF/lookup/application avoid intermediate readback;
  constant bytes and metadata are preserved. Integer word arithmetic reproduces
  CPU binary64 half-boundary rounding without shader float64. Device/counter
  limits, 2,309,120 lookup entries, ROI/byte parity, concurrent live outputs and
  completed-work/transfer benchmarks cover the provisional API.

- `gpu-image` adds exact inclusive `histogram_cdf_gpu` and its one-pass profiled
  variant, returning independently owned resident `GpuImageCdf`. The 256-bin
  integer scan uses 1 KiB local memory and writes 1 KiB without bin readback,
  normalization or rounding; the final count equals the source pixel count.
  Source recycling, concurrent results, maximum u32 counts, explicit native
  readback receipts and completed-work comparisons are covered.

- `gpu-image` adds exact two-pass `histogram_u8_gpu` channel counts and a profiled
  variant, returning resident `GpuImageHistogram`. Private partials are capped
  at 256 KiB; the result is 1 KiB with explicit native readback/recycling and
  retained runtime ownership. Invalid channels, overflowing pixel counts and
  insufficient device limits fail before allocation. GPU/CPU parity, pool reuse,
  concurrent live outputs and completed-work/transfer benchmarks cover the API.

- CPU contrast enhancement now parallelizes CLAHE lookup tiles and output row
  bands at one million pixels, sharing horizontal interpolation geometry while
  preserving clipping, reflect-101 padding and exact rounding. Histogram
  equalization applies its lookup in bounded row bands at 100,000 pixels.
  Scalar interpolation remains available for small inputs and one-worker pools;
  overflowing CLAHE lookup grids fail before allocation. This reuses optional
  Rayon under `imgproc-analysis` without adding default/core dependencies.

- Bounded GPU resource retention: point-cloud radix/compaction/scan scratch now
  returns to a private runtime pool only after its commands and scalar copy are
  submitted. Upload buffers, scratch buffers, and image textures each default
  to 64 MiB and eight retained resources per size, with configurable budgets,
  explicit clearing, and allocation/reuse/retained-byte snapshots. Zero disables
  retention; active outputs remain caller-owned. GPU image/tensor ownership
  uses a runtime token that survives moves and rejects foreign recycle targets.
  `gpu_pool_reuse` and cached/uncached Criterion cases measure completed work.

- Parallel image and point-cloud execution: GPU voxel compaction now uses a
  work-efficient hierarchical prefix scan in one submission, preserving device
  index order and four-byte cell-count readback. CPU image thresholding and u8
  histograms use bounded Rayon row bands under `imgproc-analysis`, with exact
  strided/ROI results and scalar paths for small or single-row inputs.

- Optional shared GPU compute-pass timestamps for resident image chains and
  AoSoA voxel/radius-normal frames. Timestamp-enabled runtimes check adapter
  support, bounded profilers validate capacity/runtime ownership before work,
  and explicit native collection reports named timings and query readback bytes
  separately from payload receipts. Default execution enables no timestamp
  features. A runnable example compares GPU pass time with completed wall time.

- GPU voxel segmentation uses stable four-bit radix sorting of signed XYZ keys,
  preserving input order within a cell. Exact-sized entries remove power-of-two
  padding, post-sort filtering and its scalar readback. GPU buffer/count limits
  are validated before dispatch; voxel, AoSoA frame and sparse-grid callers share
  this path. The voxel benchmark now records adapter/backend identity.

- Alignment example exposes `align_clouds` for loaded read-only inputs; bounded
  candidate evaluation reads each file once rather than once per candidate.
- Alignment HTML displays candidate scores, both support directions, convergence
  and escaped failure messages; validates selection order and selected summaries.
- Real-file Python multistart example evaluates 1–16 supplied rigid poses,
  records failures and support scores, exports the selected full source, and
  optionally renders HTML with overwrite protection and failed-write cleanup.
- Bounded multistart alignment study compares candidate search with SOR, selects
  by original-source support without ground-truth labels, and records compute
  cost, failures and known-pose assessment in JSON/HTML.
- Filter study HTML compares genuine and nonmatching point retention with
  mean/range bars and denominators, alongside registration success counts.
- Python outlier-filter attribute regressions verify complete row selection and
  dtype preservation through Arrow and PCD round trips, including empty results.
- Python SOR and radius outlier filtering release the GIL during native work;
  concurrent same-cloud calls and Python thread progress have regression coverage.
- Reproducible SOR/radius-filter plus ICP study tracks retained genuine/outlier
  points and evaluates final poses on the original source, with JSON/HTML output.
- Gate study adds a deletion-only control preserving identical noisy retained
  points, to distinguish reduced overlap from added nonmatching source points.
- Gate sensitivity study adds reproducible Gaussian noise and nearby source
  point replacements using independent random streams; HTML records conditions.
- Distance-gate study supports fixed evaluation distance and standalone HTML
  summaries. Paired synthetic runs separate optimizer recovery from relaxed
  support evaluation while verifying unchanged estimated transforms.
- Python alignment accepts an independent `evaluation_distance` and CLI
  `--evaluation-distance`, separating support diagnostics from ICP correspondence
  search. HTML reports display both distances and accept older report files.
- Alignment reports include full-resolution support immediately after the
  initial pose, separating initialization effects from subsequent ICP refinement.
- Alignment diagnostics record whether the initial transform was supplied;
  HTML reports show caller prior, default identity or unrecorded provenance.
- Alignment HTML reports show excluded point counts and fractions alongside
  gated RMSE, making residuals based on only a subset of points explicit.
- Python alignment CLI accepts `--html-report` to save its support diagnostic
  visualization alongside the aligned cloud and JSON, with failed-write cleanup.
- Standalone Python alignment HTML/SVG diagnostic report compares forward and
  reverse distance-gated support, counts and RMSE to reveal overlap asymmetry.
  It validates the report contract, escapes filenames and refuses overwrites.
- **Epic 150C Python entity embedder**: `OnnxEntityEmbedder` in the Python
  extension embeds point-entity feature vectors through an existing
  `OnnxRuntimeSession` and returns a NumPy embedding vector with its dimension,
  gated by the `onnxruntime` wheel feature. The `double_dynamic.onnx` fixture
  proves the real ONNX path doubles `[1,2,3]` → `[2,4,6]`.
- **Epic 150 real AI semantic meaning** (`semantic-model`): `OnnxEntityEmbedder`
  runs point-entity feature vectors through an already-open ONNX model session
  with explicit copy policy and produces an `Embedding` ready for the existing
  `SemanticSearchIndex`. The embedder never loads a model or picks a backend,
  keeping device placement and transfer semantics explicit. A committed
  `double_dynamic.onnx` fixture drives the facade integration test through the
  real ONNX Runtime path (`feature → session → embedding → semantic search`),
  and the `onnx_semantic` test requires `ai-onnxruntime`.
- **Epic 149 Arrow canonical interchange in Python**: `PyPointCloud` exposes
  `__arrow_c_array__`, returning `(schema, array)` Arrow C Data capsules so
  PyArrow/pandas consume SpatialRust clouds zero-copy, and `PyPointCloudStream`
  exposes `__arrow_c_stream__` for bounded record streaming into
  `pa.RecordBatchReader`. Capsule destructors call the Arrow release callbacks
  exactly once; verified against PyArrow round trips in the wheel test gate.
- **Epic 148C/148D point-cloud conformance aggregator**: `aggregate.py` unifies
  PCL/PDAL/Open3D receipts into one `pointcloud-conformance-aggregate` report
  with fail-closed checks (unsupported suites, missing ids, duplicate
  workloads). The report contract tests grew to eight cases, and the dated
  PCL receipt is committed.
- **Epic 148A/148B point-cloud conformance harness**: a versioned point-cloud
  benchmark manifest (`spatialrust.pointcloud-benchmark-manifest.v1`) and a
  stdlib-only report contract (`spatialrust.pointcloud-comparison.v1`) with
  seven contract tests, plus a PDAL comparison runner
  (`bench/pdal_comparison/`). `bench_ops` now also times `translate_xyz`
  behind `transform-ops` to match PDAL's `filters.transformation`. A dated,
  honest SpatialRust-vs-PCL receipt (`bench/pcl_comparison/receipt-2026-08-07.json`)
  records voxel 1.70×, normals 4.56×, SOR 4.91×, and radius-outlier 6.62× on
  the identical public 460,400-point cloud.
- **Epic 147F Python 3D Tiles bindings**: `export_tiles3d` converts a
  `PointCloud` into a tileset directory and `export_copc_tiles3d` converts a
  `.copc.laz` file into a bounded tileset, both returning a
  `tileset_json_bytes`/`tile_count`/`point_count`/`pnts_bytes` dict. Typed
  `.pyi` stubs and smoke tests gate the wheel build.
- **Epic 147E bounded COPC → 3D Tiles exporter** (`interchange-tiles3d-copc`):
  `spatialrust-io` gains `CopcNodeReader`, which opens a COPC file once, loads
  only hierarchy metadata, and yields one `PointCloud` per octree node in
  deterministic `(level, x, y, z)` order. `spatialrust-interchange` adds
  `export_copc_tileset`, which mirrors the COPC octree into a `tileset.json`
  with one `pnts` tile per node and a per-tile `RTC_CENTER`, without ever
  materializing the whole cloud. `max_level` bounds the exported hierarchy,
  and the `tiles3d_copc_export` example converts a `.copc.laz` file directly.
  LAS `u16` color channels are preserved as 8-bit `pnts` RGB when the source
  carries color fields.
- **Epic 147 OGC 3D Tiles 1.1 point-cloud tileset export** (`interchange-tiles3d`):
  a dependency-light `pnts` binary codec, a validated `tileset.json` model with
  box bounding volumes and geometric error, and a deterministic octree tileset
  builder. Positions are stored relative to a per-tile `RTC_CENTER` so `f32`
  precision is retained far from the origin; internal geometric error halves
  each level and leaves are zero. `build_point_tileset` takes plain interleaved
  positions plus optional RGB, and `write_point_tileset` emits `tileset.json`
  plus one `.pnts` file per tile with a byte/point receipt. The `tiles3d_export`
  example converts a PCD through the voxel filter into a tileset consumable by
  any 3D Tiles 1.1 runtime. The codec never depends on `spatialrust-core`,
  serde, or a GPU backend.
- **145K-A bounded full-bag mapping gate**: `FullBagMappingState` and the
  feature-gated `rosbag2_full_bag_mapping` example consume complete selected
  PointCloud2 streams within explicit record/point/byte bounds, reassemble
  source chunks, apply only registered clock/frame evidence, and emit
  odometry/pose-graph/TSDF/glTF receipts. Mission Cockpit can consume the
  mapping state; the canonical run remains fail-closed until matching
  calibration artifacts exist.
- **145J-B calibration evidence gate**: `CalibrationEvidenceState` and the
  feature-gated `rosbag2_calibration_evidence` example validate explicit,
  source-bound clock quality values and a root-to-front/rear extrinsic graph.
  `rosbag2_mission_cockpit` can consume the resulting receipt and expose its
  registration blockers without applying an unverified clock or transform.
- **145J-A interactive Mission Cockpit**: `MissionCockpitState` and the
  feature-gated `rosbag2_mission_cockpit` example join source-bound 145H/145I
  receipts with a bounded source-indexed XYZ sample. The self-contained
  dashboard adds 3D orbit/zoom, timeline playback, layer toggles, point
  selection, distance measurement, and an edge-to-host graph while keeping
  calibration/mapping admission fail-closed.
- **v1.3 real-data acceptance contract**: `docs/REAL_DATA_ACCEPTANCE.md` fixes
  the external rosbag2 snapshot, topic/count/hash baseline, SSD result-root
  policy, bounded synchronization evidence, and explicit calibration and
  downstream E2E gates. Sensor and derived artifacts remain outside the repo.
- **External storage preflight and manifest verification**: the opt-in
  `storage-preflight` feature rejects relative/missing output roots and free
  space below the caller's floor before source admission. `DatasetManifest`
  can read and re-hash existing JSON receipts, while `rosbag2_ingest` exposes
  `--min-output-free-bytes` and `--verify-manifest` for bounded runs.
- **Bounded rosbag2 E2E smoke**: the feature-gated `rosbag2_e2e` example now
  connects external PointCloud2 records through deterministic synchronization,
  bounded ICP odometry, TSDF extraction, record semantic entities, borrowed
  Viewer layers, embedded glTF export, and a re-hashed run manifest. It keeps
  calibration and front/rear fusion explicitly out of the acceptance claim.
- **Run-scoped checkpointing**: `rosbag2_e2e` atomically records each completed
  stage, refuses to overwrite partial runs, verifies complete runs with
  `--resume`, and removes only stale checkpoint temp files. Ingested
  PointXYZ/PointXYZI episodes now have an atomic, provenance-preserving binary
  artifact and JSON summary so `--stop-after ingest` can resume downstream
  work without reopening the rosbag2 database; those survivors are tracked as
  auxiliary manifest entries.
- **E2E performance receipt**: receipt version 2 records fresh-vs-resume mode,
  per-stage wall time, configured/observed memory, and explicit host/device
  transfer counters. The canonical bounded smoke has a 68.7-second observed
  pipeline, with ICP odometry as the dominant measured stage.
- **E2E repeated-run comparison**: `rosbag2_e2e_compare` rejects mixed input or
  resume receipts, compares canonical input/episode/glTF hashes, computes
  median/p95/CV stage statistics, and evaluates bounded-smoke budgets in an
  atomic external comparison receipt.
- **Calibration/frame readiness inventory (143B)**:
  `rosbag2_calibration_readiness` records the canonical bag's front/rear and
  relevant ROS topic presence, optionally registers external clock/frame files
  by size and SHA-256, and exits non-zero with an atomic blocker receipt when
  either artifact is absent. It does not interpret artifact contents or apply
  transforms. The canonical receipt is external at
  `/media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-readiness/rosbag2.calibration.readiness.json`.
- **Source-bound rosbag2 TF inventory (143B)**: `spatialrust-runtime` now
  decodes little-endian CDR `tf2_msgs/msg/TFMessage` payloads, while
  `spatialrust-ros2` exposes bounded `list_tf_messages` access and the
  `rosbag2_tf_inventory` receipt example. The CLI requires an exact input
  SHA-256, records transform edges without composing or applying them, and
  fails closed on source mismatch. A separate external fixture receipt under
  `/media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/` demonstrates
  the parser; it is not canonical front/rear calibration evidence.
- **Canonical calibration artifact survey (143B)**: the external SSD was
  searched read-only against the canonical bag identity, its migration
  manifest, the companion archive, and canonical topic/time identifiers. No
  matching clock or front/rear frame artifact was found. The refreshed
  fail-closed readiness receipt is external at
  `/media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-survey/canonical.calibration.readiness.json`;
  the canonical mapping gate remains blocked until real source-matched
  evidence is supplied.
- **Spatial Studio source-bound dashboard (145A)**: `spatialrust-viewer`
  now exposes a portable `StudioState` for layer inventory, timeline,
  calibration/TF admission, and explicit pipeline metrics. The feature-gated
  `rosbag2_studio` example emits JSON, a static HTML dashboard, and an output
  manifest under the external SSD. It requires exact source identity and
  refuses to admit mismatched TF or unmanifested E2E geometry; the canonical
  dashboard remains visibly blocked for mapping until real calibration exists.
- **TF / Calibration Observatory (145B)**: `spatialrust-viewer` now exposes
  artifact, clock, frame-edge, graph-topology, and composition admission state.
  The feature-gated `rosbag2_calibration_observatory` example emits a portable
  JSON state, HTML dashboard, and checksummed manifest. Opaque artifacts are
  never treated as applied calibration, and source-mismatched TF edges remain
  outside the accepted graph.
- **One-command Replay Demo (145C)**: `spatialrust-viewer` now exposes a
  portable deterministic replay trace state. The feature-gated
  `rosbag2_replay_demo` example performs a bounded, exact-source rosbag2 replay
  and emits JSON, a static HTML dashboard, and a checksummed manifest on the
  external SSD. Replay readiness is separate from mapping admission, so absent
  clock/TF calibration remains visibly fail-closed.
- **Source-bound Map Diff (145D)**: `spatialrust-interchange` can decode the
  embedded buffers emitted by its minimal glTF exporter, while
  `spatialrust-viewer` exposes `MapDiffState` for source/frame-bound geometry
  comparison. The feature-gated `rosbag2_map_diff` example emits stable-index
  displacement metrics, topology/hash checks, a 16×16 HTML heatmap, and a
  checksummed external manifest; uncalibrated maps remain inspection-only.
- **AI Semantic Overlay (145E)**: `spatialrust-ai` adds a deterministic
  `SemanticClasses` mock profile with explicit host tensor copy policies, and
  `spatialrust-viewer` exposes quantized, source-indexed
  `SemanticOverlayState` plus an RGB/confidence adapter. The feature-gated
  `rosbag2_semantic_overlay` example reuses a receipt-backed E2E glTF map to
  emit a portable JSON state, canvas dashboard, and checksummed manifest.
  Source/frame identity mismatches withhold inference, while absent clock/TF
  calibration and a production model receipt keep mapping admission blocked.
- **glTF/USD Digital Twin (145F)**: `spatialrust-viewer` now exposes a
  portable `DigitalTwinState` with independent twin and mapping gates. The
  feature-gated `rosbag2_digital_twin` example copies a receipt-backed glTF
  byte-for-byte and emits an explicit ASCII USDA companion/reference layer,
  HTML dashboard, and checksummed manifest. Optional semantic attachment is
  source/frame checked, and wrong canonical identity withholds both geometry
  assets fail-closed.
- **Dataset Health Dashboard (145G)**: `spatialrust-viewer` now exposes a
  source-bound `DatasetHealthState` that aggregates topic, canonical artifact,
  frame, calibration, and 145A–145F stage health. The feature-gated
  `rosbag2_dataset_health` example emits a JSON dashboard, static HTML, and
  checksummed manifest on the external SSD; exact source/frame mismatches
  withhold stage aggregation and mapping admission fail-closed.
- **ROS 2 Live Publish Bridge (145H)**: `spatialrust-viewer` now exposes a
  topic/frame-bound `LivePublishState` with explicit CDR payload, queue, and
  round-trip receipts. The feature-gated `rosbag2_live_publish` example
  publishes a bounded canonical episode through the CPU loopback adapter,
  emits a static dashboard and checksummed manifest, and withholds packets on
  source or topic-frame identity mismatch; uncalibrated packets remain
  inspection-only.
- **Edge Partition Execution Receipt (145I)**: `spatialrust-viewer` now
  exposes source-bound edge/host partition, named-transfer, and queue receipts.
  The `rosbag2_edge_partition` example consumes a verified live-publish state,
  exercises `spatialrust-distribute` graph/topology/backpressure contracts,
  emits a portable dashboard and manifest, and withholds transfers on source or
  upstream admission failure; calibrated mapping remains a separate gate.
- **Versioned record lineage**: `spatialrust-records::SpatialRecord` now carries
  a validated, protocol-independent `RecordProvenance` envelope for source
  identity, source URI, logical stream, and deterministic sequence. Schema
  migration, chunk slicing, bounded crop/transform, and external voxel
  aggregation preserve lineage; the rosbag2 source populates it without adding
  ROS or SQLite types to `spatialrust-core`.
- **rosbag2 SQLite PointCloud2 source**: the new feature-gated
  `spatialrust-ros2` crate opens rosbag2 SQLite files read-only, lists topics,
  decodes CDR `sensor_msgs/msg/PointCloud2` XYZ fields with optional float32
  intensity, splits oversized messages into bounded `SpatialRecordChunk`
  leases, and provides bounded SQLite-to-LAS examples for single-topic and
  inventory-driven batch ingest with per-topic receipts and checksummed
  input/output manifests. Native ROS 2 executors, custom message bindings,
  compressed bag storage, and non-float32-intensity attributes remain separate
  follow-up boundaries.
- **Bounded sensor-time episode assembly**: `spatialrust-sync` now provides
  `MemoryEpisodeBuilder` admission by record count, point count, and allocated
  column bytes. `FrameGraph::transform_record_to` applies calibrated rigid
  transforms to positions, normals, and sensor origins while preserving all
  other fields, metadata timestamps, schema, and `RecordProvenance`. The ROS 2
  adapter includes a bounded front/rear synchronization preview with an
  explicit no-calibration time-domain assumption and count-only receipt.
- **Bounded scan odometry bridge**: `spatialrust-mapping` now selects a bounded
  same-topic prefix from `MemoryEpisode`, validates one frame/clock domain,
  converts matcher deltas into a deterministic `Trajectory` and `PoseGraph`,
  and reports truncation explicitly. The generic `ScanMatcher` keeps algorithm
  choice separate; `mapping-scan-icp` adds the optional point-to-point ICP
  adapter and facade feature.
- **PointCloud-to-TSDF bridge**: `TsdfVolume::integrate_cloud` now consumes
  validated XYZ capability columns directly, returns the visited sample count,
  and preserves the existing explicit sensor-origin/invalid-sample semantics
  without allocating an interleaved temporary buffer. `integrate_cloud_with_pose`
  extends the same path with an explicit sensor-to-volume isometry, transforming
  positions and sensor origin in the integration loop.
- **SpatialRecord semantic adapter**: `spatialrust-semantic` now provides
  `SpatialRecordEntity` and deterministic `record_entity_id`, validating label
  confidence, deriving a finite XYZ centroid from capability columns, and
  retaining `RecordProvenance`, frame, and timestamp. Embeddings remain caller-
  supplied, so no AI runtime is pulled into the semantic default build.
- **GPU Euclidean clustering** (`segment-euclidean-gpu`): GPU sparse-grid key
  generation/sort/compaction with deterministic host component labeling,
  `GpuEuclideanClusterExtractor`, `EuclideanClusterExtractor::extract_with_policy`,
  MVP `cluster_policy`, and CLI `--cluster-policy auto|cpu|gpu`.
- **Execution receipts**: core `ExecutionReceipt`/`ExecutionOutput<T>`, per-stage
  `MvpPipelineResult::receipt`, CLI backend/transfer summaries, and Python
  `PipelineResult` receipt properties.
- **Visual wgpu renderer (Epics 133–134)**: backend-independent borrowed
  visualization contracts now feed an explicit, feature-gated wgpu backend.
  Point, line, and indexed-triangle geometry stays device-resident across
  headless color/depth passes; point styles support uniform, RGB, and scalar
  color maps with pixel-sized quads. Caller-requested RGBA screenshots and point
  picking emit exact readback receipts, while camera bounds fitting, runtime
  identity, pipeline caching, and bounded buffer recycling remain explicit.

- **Native viewer and debug overlays (Epic 135)**: the opt-in
  `spatialrust-viewer` crate adds portable viewer state, orbit/pan/zoom and
  bounds fitting, drag/drop admission, stable layer visibility/style state,
  point-attribute inspection, and a winit native shell. Owned normal, voxel,
  plane, cluster, correspondence, bounds, and search-radius overlays expose
  deterministic borrowed visual layers without hidden GPU transfers.

- **Scene, mapping, semantic, and RGB-D inspection (Epic 136)**: explicit
  viewer adapters cover reconstructed meshes, surfels, Gaussian primitives,
  trajectories, pose graphs, calibrated camera frusta, and semantic centroids
  with source/count/copy receipts. A borrowed RGB-D/cloud timeline rejects
  dimension, timestamp-skew, ordering, and payload/timestamp mismatches and
  provides deterministic nearest-frame and calibrated pixel inspection.
- **Record-backed semantic viewer bridge**: `spatialrust-viewer` accepts
  lineage-preserving `SpatialRecordEntity` slices directly, retaining source
  identity while materializing only renderer-facing XYZ/confidence columns and
  reporting their generated byte count.

- **Bounded streaming LOD (Epic 137)**: `spatialrust-lod` adds validated index
  hierarchies, camera-frustum selection, screen-error hysteresis, deterministic
  point/upload/in-flight admission, obsolete-request cancellation, and
  progressive resident-ancestor display. Record-memory leases, per-frame upload
  limits, protected LRU GPU eviction, cleanup receipts, and an optional bounded
  COPC query adapter make every materialization and transfer caller-visible.

- **WebAssembly/WebGPU viewer (Epic 138)**: `spatialrust-web` round-trips the
  versioned portable viewer state, maps strict browser input JSON through the
  shared controller, and renders uploaded geometry through the same
  device-resident wgpu backend. Its WASM surface provides AbortController-backed
  exact HTTP Range fetches plus deterministic request/cache budgets, response
  length validation, explicit JS/WASM copies, and LRU eviction receipts.

- **Python and Jupyter viewer adapters (Epic 139)**: the abi3 Python wheel
  exposes the canonical viewer state, shared input reducer, validated native
  launch, retained zero-copy NumPy SoA columns, and byte-exact explicit-copy
  receipts. `spatialrust-jupyter` adds a versioned AnyWidget transport with
  strict origin/source checks, canonical frontend state validation, and an
  executable notebook smoke fixture for the Web viewer embed.

- **Visual conformance and release gate (Epic 140)**: a fail-closed whole-image
  renderer fixture and Linux/Windows/macOS matrix enforce canonical pixels and
  exact upload/readback ledgers. A typed aggregate gate covers native,
  Web/WASM/browser, Python 3.8/current, Jupyter, bounded LOD, docs, security,
  API stability, and unsafe audit; it rejects missing, skipped, duplicate,
  stale, future-dated, over-budget, or wrong-migration evidence. The Visual
  guide, `visual-1` migration policy, and allowed release receipt are committed.

## [1.2.0] — 2026-07-27

### Added

- **SpatialRust 1.2 bounded-streaming contract (Epic 121)**:
  `spatialrust-records` now provides positive stream options, concurrent
  fail-closed memory reservations, cooperative cancellation, a strict
  feature-gated JSON execution receipt, and a canonical
  1M/10M/100M-by-chunk-size workload matrix. The runnable
  `streaming_receipt` example records phases, bytes, peak tracked memory,
  spill, and explicit transfers without changing `spatialrust-core`.

- **Leased bounded record streams (Epic 122)**: additive bounded source/sink
  traits preserve the existing record APIs while attaching deterministic chunk
  identity, global point offsets, finite bounds, and drop-scoped memory
  reservations. A single-worker prefetch adapter enforces queue and memory
  admission before starting, and `RecyclingMemoryChunkSource` returns owned
  column allocations to its pool in steady state. Core only gains the safe
  ownership-preserving `PointCloud::into_parts` inverse and mutable column
  iteration needed to clear recycled storage without allocating field names.

- **Bounded point-cloud IO and spool contracts (Epic 123)**: feature-gated
  PCD, PLY, LAS/LAZ, and deterministic local/HTTP COPC sources emit leased
  chunks without whole-cloud materialization. Exact-count and open-ended sinks,
  source-driven COPC output, bounded ASCII records, and extent-limited
  temporary storage fail before exceeding their declared limits.

- **Chunk-safe streaming operations (Epic 124)**: shared-budget crop and affine
  transform adapters, compensated global position reductions, and
  deterministic external voxel centroids compose outside
  `spatialrust-core`. Voxel output is invariant to source chunk and sort-run
  size, with explicit spill/run/file-handle limits.

- **Rust, CLI, and Python streaming workflows (Epic 125)**:
  `StreamingPipeline` provides a type-erased metered iterator and sink drain;
  `spatialrust-stream` converts local/HTTP inputs to open-ended LAS/LAZ with
  Ctrl-C cancellation and JSON receipts; Python `PointCloudStream` wraps the
  same Rust iterator with cancellation and live receipts.

- **SpatialRust 1.2 release gate (Epic 126)**: a dedicated Linux/Windows/macOS
  matrix and `Streaming12ReleaseGate` enforce memory, spool, cleanup, copy,
  transfer, determinism, and file-handle budgets. The runnable release receipt,
  migration guide, stability registry, and feature-isolation gates complete
  the bounded-streaming program.

### Changed

- Python ONNX session inference releases the GIL during both bound and copied
  native CPU runs; Python-thread progress and output correctness are tested.

- Add ONNX-enabled Python CPU CI with pinned official runtime libraries,
  real-model/reference checks and a gate rejecting skipped tests.

- Python voxel downsampling validates finite XYZ/positive leaf size and releases
  the GIL during native filtering; CPU concurrency and boundary tests are added.

- Python wheel CI retains per-version JUnit binding-test results for 14 days,
  including results produced before a later CI failure.

- Python point-to-point ICP releases the GIL during native registration, with
  concurrent shared-input and Python-thread progress integration checks.

- Python distance-gated support releases the GIL during native computation,
  allowing other Python threads and concurrent read-only diagnostic calls.

- Python exposes KD-tree distance-gated support and records final full-resolution
  bidirectional support separately from kernel fitness in the alignment workflow.

- Add a validated Python two-file alignment workflow with optional initial pose,
  voxel/full-resolution ICP stages and full-resolution PCD/JSON export.
  File-based integration tests cover composition, input rejection and output cleanup.

- **Prefetch admission under load**: the bounded prefetch preflight now
  accounts for the producer lease held while a full queue blocks in addition
  to queued and consumer leases. This removes a scheduler-dependent
  end-of-stream error while preserving fail-closed memory behavior.

- **GPU voxel dispatch and truthful Auto policy**: headless wgpu compute now
  prefers a high-performance adapter while retaining an explicit low-power
  option. Batched bitonic-sort/prefix-scan command recording and device-side
  buffer clearing reduce measured centroid GPU latency by 16%–31% at
  500k–2M points. A recycled-buffer usage mismatch that could panic at 2M is
  fixed. Because the current CPU fast path remains faster through 2M points,
  centroid `ExecutionPolicy::Auto` now stays on CPU until a new crossover is
  measured; explicit GPU and GPU-resident paths remain available.

## [1.1.0] — 2026-07-16

### Added

- **Centered 5×5 morphology (Epic 117E)**: exact grayscale rectangular
  morphology now bypasses the large-window prefix/suffix and transpose engine
  for the canonical centered Replicate case. Fixed extrema, direct row-major
  vertical passes, safe SIMD dispatch, and bounded row blocks reduce the old
  OpenCV gaps by 6.6×–31.8× and measure 1.22× faster for 1080p caller-output
  opening; all 980 randomized operation comparisons remain bit-exact.

- **Direct and fused 3×3 Sobel (Epic 116E)**: grayscale `u8` first derivatives
  now use bounded parallel three-row `i16` rings instead of the generic
  full-image `f64` intermediate. Added Rust/Python caller-output APIs for exact
  `f32` derivatives, saturated absolute `u8` responses, and fused binary edge
  masks. Packed NumPy inputs are borrowed without copying. Standalone Sobel
  beats OpenCV 1.88× at 1080p and 2.03× at 4K; fused masks win 3.81×–6.64×
  allocated and 2.95×–8.68× with caller-owned output across 300 bit-exact
  randomized cases.

- **Allocation-light Canny (Epic 118A/118C)**: `canny()` no longer materializes
  public gradient, magnitude, and suppression images only to discard them.
  Added safe strided `canny_into`, reusable `CannyWorkspace`, large-image
  parallel stages, Python `out=`/workspace support, and a focused bit-exact
  OpenCV comparison harness. Epic 118B/118D replace the full comparison-
  magnitude image with parallel three-row rings and skip hysteresis when no
  weak edges exist; 4K document-line reuse is 11.92× faster than the inspectable
  path and measured 1.42× faster than OpenCV.
- **Exact Euclidean distance transform**: `spatialrust-vision` now computes
  foreground-to-nearest-background L2 distances in linear time, supports
  anisotropic pixel spacing, exposes a NumPy binding, and includes native
  Criterion plus OpenCV `DIST_MASK_PRECISE` comparison coverage.

- **Vision 2 performance roadmap and documentation site**: Epics 112–120 now
  define cost attribution, reusable workspaces, safe CPU dispatch, resize/color,
  Gaussian/Sobel, morphology, Canny, explicit GPU-chain, and release-gate work.
  GitHub Pages adds a searchable cross-workspace algorithm catalog beside the
  generated Rust API.

- **OpenCV comparison methodology v2 (Epic 111)**: seeded interleaved paired
  timing, adaptive batching for sub-5-ms calls, robust dispersion and throughput
  metrics, strict finite JSON, and workload-specific numerical/edge accuracy
  across VGA, 1080p, and 4K. Results name the faster implementation per workload
  instead of asserting blanket SpatialRust superiority.

- **SpatialRust Vision 1.0 release gate (Epic 110)**: mandatory cross-platform,
  Python, OpenCV, GPU-transfer, unsafe-audit, fixed performance-budget, example,
  and migration-policy evidence with an executable release decision.

- **Bounded spatial execution graph (Epic 109)**: deterministic DAG compilation,
  same-device linear fusion groups, bounded source admission, cross-device
  transfer validation, and per-run named copy receipts behind `runtime-graph`.

- **Computational photography and panorama (Epic 108)**: gray-world white
  balance, aligned well-exposedness fusion, bounded homography panorama canvas,
  feather blending, Python entry points, and OpenCV warp parity receipts behind
  an additive vision feature.

- **Robust visual/RGB-D odometry integration (Epic 107)**: spatially balanced
  keypoint selection, forward/backward LK validation, scale-explicit monocular
  motion, metric depth-backed PnP, mapping `DeltaMotion` bridges, Python RGB-D
  odometry, and an OpenCV pose receipt.

- **Video recognition substrate (Epic 106)**: dense integer block flow,
  adaptive Gaussian foreground segmentation, deterministic same-class IoU
  tracking, and timestamped pull-based video source contracts. Codec/camera
  adapters remain feature-gated; Python dense flow, OpenCV Farneback comparison,
  and QQVGA/QVGA Criterion workloads cover the portable core.

- **Camera calibration contracts (Epic 105)**: robust pinhole intrinsics,
  Kannala–Brandt4 fisheye fitting, supplied-rotation stereo and hand-eye
  translation solves, and fixed-camera sparse point bundle adjustment. All
  solvers return common RMS/max/iteration receipts, validate rotations and
  indices, and use deterministic small dense math without a native optimizer.

- **Texture-backed GPU image chains (Epic 104)**: `GpuImage` now uses pooled
  `rgba8uint` textures instead of component-expanded storage buffers. Explicit
  upload/readback receipts compose through copy, RGB-to-gray, box blur, nearest
  resize, Sobel, erosion, and dilation with no hidden host transfer. Runtime
  adapter/backend identity and explicit `wait_idle` profiling boundaries are
  exposed; synchronized VGA/1080p/4K Criterion groups cover steady-state chains.

- **Reusable CPU vision kernels (Epic 103)**: stable caller-owned
  `resize_into`, `rgb_to_gray_into`, `normalize_into`, and `pack_chw_into`
  paths, NumPy `out=` bindings, size-aware safe parallel CHW packing, and a
  VGA/1080p/4K allocation/reuse comparison receipt. The reference run records
  both OpenCV's resize/color-conversion lead and SpatialRust's 8.54x–16.11x
  reusable RGB-to-CHW advantage.

- **Vision API conformance (Epic 102)**: a machine-readable stable/provisional
  image-camera-vision registry, a compile-and-behavior API contract, and a
  dedicated Linux/Windows/macOS CI matrix for the image, camera, and full vision
  feature surface. Geometry, stereo, flow, AI adapters, and GPU images remain
  explicitly provisional while ownership and common entry points are frozen.

- **OpenCV comparison contract (Epic 101)**: versioned correctness/performance
  JSON reports with environment receipts, raw timing samples, median/p95,
  allocation/reuse modes, a canonical VGA/1080p/4K workload manifest, and an
  aggregate runner. The schema tests are standard-library only; OpenCV remains
  comparison tooling rather than a production dependency.

- **RGB-D dense XYZ vs OpenCV**: `depth_to_xyz_dense` / `_into` with an x86_64 AVX2
  fill path; Python `depth_to_xyz(..., out=)` for streaming reuse. Fast identity
  packing for `rgbd_to_point_cloud`. The `bench/opencv_rgbd_comparison` harness
  gates faster-or-equal medians against `cv.rgbd.depthTo3d` (alloc/into) and
  against OpenCV depth+mask+color gather for colored clouds.

- **Distributed execution deepen (Epic 99)**: `spatialrust-distribute` adds
  cycle-aware partition topological order, validated `TransferPlan` /
  `TransferLedger` with measurable copy bytes, and `BoundedTransferQueue`
  watermark admissions; wired into `north-star-e2e`.

- **Platform release-gate deepen (Epic 100)**: `spatialrust-platform` adds
  performance budgets, a seeded north-star stability surface, baseline security
  checklist helpers, conformance summaries, and an aggregated `ReleaseGate`
  consumed by `north-star-e2e`.

- **North-star E2E full portable path**: `north-star-e2e` now enables
  `sync-mcap` and `runtime-ros2`, and the integration test/demo exercise MCAP
  XYZ round-trip, ROS 2 CDR PointCloud2 loopback, USDA ASCII mesh export, and
  Gaussian CPU soft-splat alongside the existing RGB→TSDF→glTF flow.

- **ROS 2 / OpenUSD / Gaussian deepen**: `runtime-ros2` adds CDR LE
  `PointCloud2` XYZ codecs and an in-process loopback node (no `rclrs` link);
  `interchange-openusd` exports/imports USDA ASCII mesh stages; `scene-gaussian`
  adds a CPU soft-splat renderer over `GaussianScene`.

- **MCAP episode IO + TSDF meshing deepen**: `spatialrust-sync` `mcap` /
  facade `sync-mcap` writes and reads XYZ stamped records to MCAP files;
  `TsdfVolume::extract_mesh` uses truncation-band integration and marching
  tetrahedra instead of the occupied-voxel triangle proxy.

- **Canonical 2D → AI → 3D roadmap and image IO (Epic 83)**: `docs/ROADMAP.md`
  reserves Epics 83–90; new `spatialrust-image-io` provides bounded path,
  reader/writer, and memory PNG/JPEG/PNM codecs, independently gated TIFF and
  OpenEXR, typed pixels, Exif orientation handling, Python/NumPy bindings,
  property tests, and 640p/1080p/4K decode benchmarks.

- **Shared CPU filters (Epic 84A–84B)**: `spatialrust-vision` now exposes a
  common `BorderMode`, validated 1D/2D kernels, OpenCV-style filter2D
  correlation, explicit convolution, f32-output and separable filters, and
  normalized box/Gaussian blur, median and bilateral filters, signed
  Sobel/Scharr/Laplacian derivatives, and Gaussian pyramids. The feature
  includes strided-view property coverage, Python bindings/stubs, OpenCV
  comparison, and 640p/1080p/4K benchmarks.

- **CPU morphology (Epic 84C)**: validated rectangular, cross, elliptical,
  diamond, and custom structuring elements; explicit-anchor erode/dilate;
  open/close/gradient/top-hat/black-hat operations; additive feature and meta
  feature; u8/u16/f32 and strided-view tests; Python bindings; exact OpenCV
  comparisons; and 640p/1080p/4K benchmarks.

- **CPU image analysis (Epic 84D)**: fixed and adaptive thresholds, u8/u16
  Otsu selection, masked configurable histograms, exact u8 equalization,
  contrast-limited adaptive equalization, and checked summed-area tables;
  additive Rust/meta features, Python bindings/stubs, strided properties,
  OpenCV comparisons, and representative-resolution benchmarks.

- **Canny edge detection (Epic 84E)**: configurable 3/5/7 Sobel apertures,
  L1/L2 gradient magnitude, directional non-maximum suppression, 8-neighbor
  hysteresis, and inspectable intermediate stages; additive Rust/meta features,
  strided and property tests, Python binding/stub, 640p/1080p/4K benchmark, and
  exact OpenCV comparison across all six aperture/magnitude combinations.

- **Tensor foundation (Epic 85A)**: new dependency-light `spatialrust-tensor`
  crate with byte-addressable dtype, arbitrary-rank shape, signed element
  strides, checked byte offsets/spans, explicit device identity, safe borrowed
  CPU views, and named owned copies. Non-host device memory cannot be exposed as
  a Rust byte slice, and the meta-crate integration is opt-in through `tensor`.

- **Image/spatial tensor bridges (Epic 85B)**: packed interleaved images expose
  zero-copy HWC views, packed planar images expose zero-copy CHW views, and
  Schema-SoA `f32` point fields expose zero-copy one-dimensional views. Explicit
  `pack_*` operations handle padded/ROI images, with feature-alone tests and
  640p/1080p/4K packing benchmarks.

- **DLPack and Python tensor interoperability (Epic 85C–85D)**: audited
  `DLManagedTensorVersioned` major-version 1 CPU import/export, explicit deleter
  transfer, read-only/copy flags, signed strides and byte offsets, malformed ABI
  rejection, and zero-copy Python `__dlpack__`/`__dlpack_device__`. NumPy and
  PyTorch round trips preserve allocations and producer lifetimes; device or
  host copy requests remain explicit.

- **Inference contracts and ONNX Runtime CPU (Epic 86)**: new optional
  `spatialrust-ai` crate with named dynamic model metadata, stable backend and
  session traits, explicit input/output copy permissions, CPU ONNX Runtime,
  separately gated CUDA/TensorRT/DirectML providers, typed zero-copy I/O
  Binding, caller-preallocated u8/u16/f32 outputs, runtime-allocation retention,
  output-to-input chaining, Python bindings/stubs, reference-runtime comparison,
  and 640p/1080p/4K Criterion coverage. Multi-byte raw storage is rejected at
  zero-copy boundaries instead of being cast from an under-aligned byte buffer.

- **Feature2D and ORB matching (Epic 87)**: checked keypoint, binary/float
  descriptor, feature-set, and match contracts; Harris and Shi–Tomasi corners;
  OpenCV-exact FAST-9/16 detection and scores; deterministic multi-scale ORB
  with 256-bit rotated BRIEF; Hamming/L2 brute-force matching with ratio,
  cross-check, and distance filters; Python/NumPy bindings and stubs; property
  tests, OpenCV comparison, and 640p/1080p/4K Criterion baselines.

- **Camera geometry, motion, and stereo (Epic 88)**: checked correspondence and
  projective contracts; normalized DLT and deterministic RANSAC for homography,
  fundamental, and essential matrices; triangulation and essential pose
  disambiguation; EPnP-class PnP with iterative refine and RANSAC; sparse
  pyramidal Lucas–Kanade tracking; stereo rig, rectify remap grids, SAD block
  matching, and disparity-to-depth/XYZ reproject; Python bindings; OpenCV
  comparison with documented tolerances; property tests; and Criterion coverage.

- **Explicit GpuImage and chainable wgpu vision (Epic 89)**: `spatialrust-gpu`
  `gpu-image` feature adds `GpuImage` ownership with packed/`ImageView` upload,
  named stride packing, explicit readback, transfer receipts, and buffer
  recycle; device-resident `copy_gpu_image`; BT.601 `rgb_to_gray_gpu`; gray
  `box_blur_gpu` with replicate/constant-zero borders; facade `gpu-image` flag;
  headless chain tests that keep mid-pipeline `device_to_host_bytes == 0`; and
  640p/1080p/4K Criterion coverage. Texture-backed storage remains deferred.

- **Model adapters and image → AI → point-cloud pipelines (Epic 90)**:
  `MockInferenceBackend` / `ModelSource::Mock` for deterministic host inference
  without ONNX; vision `ai-adapters` for letterbox NCHW prep and
  tensor→`DepthMap`/`BinaryMask`/`Detection` decode; facade
  `ai-vision-pipeline` E2E covering RGB → mock depth → unproject → MVP.

- **Spatial records and Arrow bridges (Epic 91)**: new `spatialrust-records`
  with versioned `SpatialRecord`, schema compatibility/migration, and in-memory
  chunked sources/sinks; new `spatialrust-arrow` with Arrow C Data export/import,
  C Stream over record sources, and CPU C Device arrays; facade flags
  `records`, `arrow-c-data`, `arrow-c-stream`, `arrow-c-device`; ROADMAP 92–100
  activated with planned delivery slices.

- **Sensor time and frame graphs (Epic 92)**: new `spatialrust-sync` with clock
  domains/`SyncQuality`, stamped records, calibrated `FrameGraph` lookups,
  in-memory `MemoryEpisode` index, and `DeterministicReplayer` topic bundling;
  `mcap` feature reserved for file codecs; facade `sync` / `sync-mcap`.

- **Localization and mapping substrate (Epic 93)**: new `spatialrust-mapping`
  with `Trajectory`/`StampedPose`, `RelativeMotionEstimator`/`SyntheticOdometry`,
  `PoseGraph` root localization, and distance-based loop-closure candidates;
  facade `mapping`.

- **North-star substrate (Epics 94–100)**: new crates for scene reconstruction
  (`spatialrust-scene` TSDF/surfel/mesh + gated Gaussians), semantic intelligence
  (`spatialrust-semantic`), embodied episodes (`spatialrust-episode`), bounded
  robotics runtime (`spatialrust-runtime` + `ros2` gate), glTF/OpenUSD
  interchange (`spatialrust-interchange`), distributed execution
  (`spatialrust-distribute`), and platform stability (`spatialrust-platform`);
  facade `north-star` enables the stack without native heavy codecs.

- **North-star E2E demo**: facade feature `north-star-e2e` adds an integration
  test and example covering RGB → mock depth → stamped episode → pose graph →
  TSDF mesh → glTF JSON, plus semantic search, bounded runtime, partition graph,
  and platform conformance markers.

- **AI-ready image and vision foundation (Epics 75–79)**: mutable ROI views,
  planar/interleaved layouts and color metadata in `spatialrust-image`; new
  feature-gated `spatialrust-vision` preprocessing, warp, detection, mask/RLE,
  and dense spatial-map APIs; camera/point-cloud/pipeline bridges; Python/NumPy
  bindings and stubs; property tests, OpenCV comparison, Criterion benchmark,
  and an end-to-end vision-to-point-cloud demo.

- **OpenCV-oriented image/camera foundation**: new `spatialrust-image` typed
  packed buffers and strided zero-copy views; new `spatialrust-camera` pinhole
  projection/unprojection, Brown–Conrady distortion, aligned depth/RGB-D to
  XYZ/XYZRGB conversion, Python binding, MVP integration test, Criterion bench,
  synthetic demo, and OpenCV `rgbd.depthTo3d` comparison harness.

- **Euclidean cluster benchmark** (`bench/euclidean_cluster/`): CPU vs GPU timing
  harness with optional `--mvp-leaf` preprocess path
  (`notes/2026-07-03_euclidean_cluster_bench.md`).
- **MVP normal backend selection**: `MvpPipelineConfig::normal_policy`,
  `NormalEstimator::estimate_with_policy`, and CLI `--normal-policy auto|cpu|gpu`
  (included in `pipeline-mvp-gpu` / `feature-normal-gpu`).
- **MVP GPU normal radius**: `MvpPipelineConfig::normal_gpu_radius_scale` derives
  `search_radius` from voxel leaf when the wgpu backend is selected, enabling the
  GPU uniform-grid path (`notes/2026-07-03_mvp_normal_gpu_grid_radius.md`).
- **Provisional `SpatialTensor` chunked views** (`spatialrust-core`): zero-copy
  chunk iteration over `PointCloud` columns (`notes/2026-07-03_spatial_tensor_chunked_views.md`).

### Fixed

- Python exception documentation matches ValueError from native input failures;
  ONNX tests cover invalid inputs, reuse, dynamic batches and output lifetimes.

- Python workspace caches use const thread-local initialization, and Python
  wheel CI checks the extension crate with warning-strict Clippy.

- Streaming memory reservations use an overflow-checked compare/exchange loop,
  removing a Rust 1.99 deprecation while preserving budget and release semantics.

- Python Arrow documentation now describes owned buffer copies rather than
  zero-copy point-cloud exports; PyArrow lifetime tests cover arrays and streams.

- Direct Python ICP rejects invalid gates, zero iteration budgets and insufficient
  or nonfinite input clouds with named ValueErrors before native alignment.

- GPU uniform-grid stages (Euclidean cluster, MVP normal radius) fall back to CPU when
  the spatial extent would exceed the wgpu cell cap.
- MVP integration tests: `EuclideanClusterConfig` partial initializers include
  `..Default::default()` for `gpu_min_points`.
- **GPU Euclidean cluster contract**: removed the misleading CPU-only GPU entry
  point; GPU policy now dispatches sparse-grid construction and reports the
  explicit host component-labeling stage.
- **Shared `uniform_grid` module** (Epic 68): `spatialrust-search` hosts grid
  bounds/build/cluster roots; GPU normals and cluster paths reuse it
  (`notes/2026-07-03_cpu_grid_euclidean_cluster.md`).
- **Chunked spatial index queries** (Epic 70): `ChunkedRadiusSearchIndex`,
  `ChunkedNearestNeighborIndex`, and `SpatialTensor` staging helpers in
  `spatialrust-search` (`notes/2026-07-03_spatial_index_chunked_queries.md`).
- **Parallel chunked queries** (Epic 71): threaded `SpatialTensor` chunk dispatch
  via `spatialrust-search/parallel` (`notes/2026-07-03_parallel_chunked_queries.md`).
- **AoSoA chunk packing** (Epic 72): interleaved XYZ buffers per chunk behind
  `tensor-aoso` (`notes/2026-07-03_aoso_chunk_packing.md`).
- **Parallel staging hot paths** (Epic 73): shared index-range staging in
  `spatialrust-search` wired into CPU normals and outlier filters
  (`notes/2026-07-03_parallel_staging_hot_paths.md`).
- **AoSoA GPU upload** (Epic 74): `GpuAoSoXyzChunk` and
  `upload_spatial_tensor_xyz_chunks` behind `gpu-aoso-staging`
  (`notes/2026-07-03_aoso_gpu_upload.md`).
- **AoSoA GPU voxel dispatch** (Epic 75): voxel keys are computed directly from
  uploaded interleaved XYZ chunks without re-uploading separate coordinate
  columns (`notes/2026-07-13_aoso_gpu_voxel_dispatch.md`).
- **AoSoA GPU voxel pipeline** (Epic 76): uploaded chunks are combined with
  GPU-to-GPU copies and flow through global key, sort/segment, and interleaved
  centroid reduction stages without intermediate readbacks
  (`notes/2026-07-13_aoso_gpu_voxel_pipeline.md`).
- **GPU-resident AoSoA result** (Epic 77): centroid results retain the combined
  interleaved position buffer for downstream kernels, with safe buffer access
  and explicit recycling (`notes/2026-07-13_gpu_resident_aoso_result.md`).
- **AoSoA attribute layouts** (Epic 78): explicit XYZ/intensity/normal
  stride-offset metadata, capability-based chunk packers, GPU upload wrappers,
  and global voxel `Average` / `First` aggregation
  (`notes/2026-07-13_aoso_attribute_layouts.md`).
- **AoSoA GPU normal chaining** (Epic 79): retained interleaved XYZ buffers feed
  normal estimation without position re-upload, and normal/curvature output can
  remain GPU-resident until explicit readback
  (`notes/2026-07-13_aoso_gpu_normals.md`).
- **GPU sparse radius grid** (Epic 80): radius cell keys, sorting, segment
  compaction, sparse neighbor lookup, and normal estimation run directly from
  retained AoSoA positions without CPU neighbor upload or dense cell allocation
  (`notes/2026-07-13_gpu_sparse_radius_grid.md`).
- **GPU-resident spatial frame** (Epic 81): schema-aware ownership of positions,
  segments, radius grids, normals, and attributes with device validation,
  explicit readback/recycling, transfer receipts, and a chained voxel-to-normal
  pipeline (`notes/2026-07-13_gpu_spatial_frame.md`).
- **Frame-native GPU operations** (Epic 82): radius-grid rebuilding, normal
  replacement, and attribute reduction execute through `GpuSpatialFrame`, with
  device/length validation, old-buffer recycling, and receipt updates
  (`notes/2026-07-13_frame_native_gpu_operations.md`).

## [1.0.0] — 2026-07-03

First stable release: **Stable** APIs in `docs/API_STABILITY.md`, MVP pipeline +
COPC IO, wgpu acceleration paths behind feature flags, and Python bindings with
stubtest CI.

### Added

- **Public COPC validation** (`bench/public_copc/`): reproducible harness and
  integration test `mvp_public_copc` exercising bounds/resolution partial reads
  on the public PCL `table_scene_lms400` sample through the MVP pipeline.
- **GPU RANSAC plane scoring** (`segment-ransac-plane-gpu`): WGSL inlier-count
  kernel, `GpuRansacPlaneSegmenter`, and `RansacPlaneSegmenter::segment_with_policy`
  with `ExecutionPolicy` selection (~11× on 460k points, ~2.7× after MVP
  downsampling in local release measurements).
- **MVP plane backend selection**: `MvpPipelineConfig::plane_policy`,
  `pipeline-mvp-gpu` feature, and CLI `--plane-policy auto|cpu|gpu`.
- **RANSAC plane benchmark** (`bench/ransac_plane/`): CPU vs GPU timing harness
  with optional `--mvp-leaf` MVP-preprocess path.
- **Public GPU buffer pool** (`GpuBufferPool` on `WgpuRuntime`): explicit
  upload/recycle for wgpu storage buffers (`upload_pod_storage`,
  `upload_u32_storage`, `clear_buffer_pool`).
- **PyG PointNet++ demo** (`crates/spatialrust-py/examples/pyg_pointnet_demo.py`):
  SpatialRust preprocessing → PyTorch Geometric tensors.
- **API stability guide** (`docs/API_STABILITY.md`): stability tiers and v1.0
  release checklist.
- **PCL comparison benchmark** (`bench/pcl_comparison/`): a reproducible,
  apples-to-apples timing harness against PCL 1.14 on voxel downsampling, normal
  estimation, and outlier removal, plus rotating cluster/voxel README GIFs
  generated through the Python bindings (`examples/make_gifs.py`).
- **Python type stubs** (`spatialrust.pyi` + `py.typed`, PEP 561): full type
  information for the compiled extension so editors and type checkers (mypy,
  pyright) get autocomplete and signature checking. CI runs `mypy.stubtest` on
  every push to keep the stubs in sync with the runtime API.
- **Python bindings** (`spatialrust-py`, PyO3 + maturin): NumPy interop for point
  clouds, `read`/`write` (PCD/PLY/LAS/COPC), `voxel_downsample`, `run_pipeline`,
  `region_growing`, and registration (`register_icp`, `register_point_to_plane`,
  `register_gicp`, `register_ndt`). Built as `abi3` wheels (CPython 3.8+).
- **Voxel occupancy grids** crate (`spatialrust-voxelize`, `voxelize-occupancy`):
  voxelizes a cloud into a dense 3D occupancy or count grid (the tensor learned
  models consume), exposed in the Python bindings as `voxelize` returning an
  `(nz, ny, nx)` NumPy array.
- **ML preprocessing example** (`examples/ml_preprocess.py`): the end-to-end
  "point cloud → model-ready tensors" pipeline (clean → unit-sphere normalize →
  FPS → voxel occupancy grid / LiDAR range image / k-NN `edge_index`), with a
  four-panel figure.
- **Neighborhood graph construction** (`spatialrust-search`, `search-graph`):
  directed k-NN and radius graphs over a cloud, exposed in the Python bindings
  as `knn_graph` / `radius_graph` returning a PyG-style `(2, E)` `edge_index`
  for graph neural networks.
- **Spherical range-image projection** (`spatialrust-voxelize`,
  `voxelize-range-image`): projects a rotating-LiDAR scan into the dense 2D
  range image used by LiDAR segmentation models, exposed in the Python bindings
  as `range_image` returning a `(height, width)` NumPy array.
- **Cloud transform utilities** crate (`spatialrust-transform`, `transform-ops`):
  4×4 affine transforms (positions + normals), recentering, scale and unit-sphere
  normalization, cloud merging, and AABB / PCA-based OBB computation, exposed in
  the Python bindings (`apply_transform`, `recenter`, `scale`,
  `normalize_unit_sphere`, `merge`, `centroid`, `bounding_box`,
  `oriented_bounding_box`).
- **Point cloud metrics** crate (`spatialrust-metrics`, `metrics-distance`):
  symmetric Chamfer and Hausdorff distances (plus directed statistics) for
  scoring registration / reconstruction against a reference, exposed in the
  Python bindings (`chamfer_distance`, `hausdorff_distance`).
- **Farthest Point Sampling** in `spatialrust-filtering` (`filter-fps`): even
  spatial downsampling to a target point count (the standard front-end for
  learned point-cloud models), exposed in the Python bindings
  (`farthest_point_sampling`).
- **MLS surface smoothing** in `spatialrust-filtering` (`filter-mls`): Moving
  Least Squares projection onto a local polynomial surface (order 1 or 2) that
  removes noise while preserving curvature, exposed in the Python bindings
  (`mls_smooth`).
- **Crop and range filters** in `spatialrust-filtering` (`filter-crop`):
  axis-aligned `CropBox` and field-range `PassThrough` (both invertible),
  exposed in the Python bindings (`crop_box`, `pass_through`).
- **Outlier removal filters** in `spatialrust-filtering` (`filter-outlier`):
  Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR), both
  exposed in the Python bindings (`statistical_outlier_removal`,
  `radius_outlier_removal`).
- **Registration suite** in `spatialrust-registration`:
  - Point-to-plane ICP (`register-icp-point-to-plane`).
  - Generalized ICP / GICP (`register-gicp`), with an optional GPU covariance
    path (`register-gicp-gpu`, ~1.7× faster covariance estimation).
  - NDT — Normal Distributions Transform (`register-ndt`), point-to-distribution
    with Levenberg–Marquardt.
  - FPFH + RANSAC global registration (`register-fpfh`): coarse alignment with
    no initial guess via Fast Point Feature Histograms and a RANSAC pose search,
    exposed in the Python bindings (`register_fpfh_ransac`).
  - Public FPFH descriptor API (`fpfh_descriptors`, `FpfhDescriptor`) and a
    keypoint-based registration path (ISS keypoints → FPFH → RANSAC) exposed in
    the Python bindings as `register_fpfh_keypoints`.
  - Registration backend selection in the MVP pipeline (`MvpRegistrationMethod`).
- **Boundary / edge point detection** (`spatialrust-features`,
  `feature-boundary`): flags hole-rim and scan-edge points via tangent-plane
  neighbor angle gaps, exposed in the Python bindings as `detect_boundary`.
- **Consistent normal orientation** (`spatialrust-features`,
  `feature-normal-orient`): MST/Prim propagation over a k-NN graph that flips
  estimated normals to agree in sign, exposed in the Python bindings as
  `orient_normals` (estimate + orient).
- **ISS keypoint detection** (`spatialrust-features`, `feature-iss`): Intrinsic
  Shape Signatures saliency with non-maximum suppression, returning a sparse
  keypoint sub-cloud; exposed in the Python bindings (`iss_keypoints`).
- **GPU normal estimation** (`spatialrust-features`, `feature-normal-gpu`): a wgpu
  path with a fully-GPU uniform-grid radius neighbor search (covariance + Jacobi
  eigensolver), up to ~50× faster than the CPU KD-tree estimator at 500k points.
- **Multi-plane segmentation** (`spatialrust-segmentation`,
  `segment-multi-plane`): sequential RANSAC that extracts the N dominant planes
  (floor, walls, ceiling) and labels each point by plane index, exposed in the
  Python bindings as `segment_multi_plane`.
- **DBSCAN segmentation** (`spatialrust-segmentation`, `segment-dbscan`):
  density-based clustering with explicit noise labeling, exposed in the Python
  bindings (`dbscan`).
- **Ground segmentation** (`spatialrust-segmentation`, `segment-ground`):
  grid-based ground/non-ground split using per-cell minimum heights eroded
  against neighbors (robust to slopes and rooftops), exposed in the Python
  bindings (`ground_segmentation`).
- **RANSAC sphere & cylinder fitting** (`spatialrust-segmentation`,
  `segment-ransac-primitives`): detect spheres (positions only) and cylinders
  (axis recovered from surface normals), partitioning inliers/outliers, exposed
  in the Python bindings (`ransac_sphere`, `ransac_cylinder`).
- **Region growing segmentation** (`spatialrust-segmentation`,
  `segment-region-growing`): normal-smoothness region growing with curvature
  seeding.
- Benchmarks comparing CPU vs GPU normal estimation and the four registration
  backends (speed + accuracy), with writeups under `notes/`.
- End-to-end Python example (`end_to_end.py`): outlier removal → downsample →
  DBSCAN → FPFH global registration → ICP refinement, rendered as a four-panel
  figure; accepts a real scan via `--input`.
- CI: Python wheel build/publish workflow; benchmark-compile and per-feature
  test matrix entries for all new features.

### Changed

- Split the AoSoA GPU staging implementation into focused `types`, `upload`,
  `attributes`, `normals`, `voxel`, and `tests` modules without changing its
  public API or measured performance
  (`notes/2026-07-13_aoso_module_cleanup.md`).
- Restrict internal normal-shader re-exports to `gpu-aoso-staging`, fixing
  `-D warnings` failures in narrower GPU/segmentation CI feature jobs.

- **Plane Auto threshold**: `DEFAULT_GPU_MIN_POINTS_PLANE` lowered from 100,000
  to 2,000 so MVP downsampled clouds (~2k points on the reference sample) select
  GPU under `Auto`; override with `RansacPlaneConfig::gpu_min_points`.
- Performance: radius outlier removal is **~16× faster** (1.70 s → 0.10 s on
  210k points) via a new `KdTree::radius_reaches` early-exit density test that
  stops once enough neighbors are found and allocates nothing. `radius_search`
  also no longer sorts results (no caller relied on the order). CPU voxel
  downsampling is **~2× faster** (0.047 s → 0.022 s on 210k points) via a fast
  integer hasher and a single-pass centroid accumulator for the default
  Centroid + Average case. Measured against PCL 1.14, SpatialRust now wins 3 of
  4 compared operations, with PCL's voxel grid keeping a ~2× edge
  (`bench/pcl_comparison/`).

### Fixed

- Workspace `[workspace.dependencies]` version requirements aligned to `1.0.0`
  so `cargo build` resolves after the release version bump.
- **Python binding tests**: a pytest suite (`crates/spatialrust-py/tests/`)
  covering the NumPy ⇄ Rust boundary of every exported function (shapes, dtypes,
  keyword signatures, and sane results on synthetic clouds), wired into CI as a
  `python-bindings` job that builds the extension with maturin and runs the
  tests on Python 3.8 and current — closing a gap where the bindings were only
  compile-checked, never imported or called.
- CI now runs clippy under `--all-features` (library targets) and a full
  `--all-features` test pass, closing a gap where clippy and combined-feature
  builds were only checked at default features; resolved the pre-existing
  clippy warnings this surfaced in the GPU and IO crates.
- Boundary detection panicked on isolated points when `min_neighbors` was 0
  (empty angle list indexed out of bounds); it now returns non-boundary.
- RANSAC samplers (plane and sphere / cylinder) drew indices from the low
  (short-period) bits of their LCG; they now use the well-mixed high bits via
  multiply-shift, making fits more reliable (the plane fix is what let
  multi-plane extraction find every plane).
- Gated the `io-copc-http` integration test behind its feature so
  `cargo test --workspace` builds with default features.
- Resolved pre-existing rustfmt and clippy drift surfaced by current stable
  toolchains so the CI fmt/clippy gates pass.

## [0.1.0] — MVP foundation

### Added

- MVP pipeline end-to-end: PCD/PLY/LAS/COPC IO, voxel downsampling (CPU + optional
  wgpu), normal estimation, RANSAC plane segmentation, Euclidean clustering, and
  point-to-point ICP.
- COPC partial reads (bounds + LOD) in the library and `spatialrust-mvp` CLI.
- wgpu voxel downsampling with automatic CPU/GPU policy selection.

[Unreleased]: https://github.com/rsasaki0109/SpatialRust/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/rsasaki0109/SpatialRust/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/rsasaki0109/SpatialRust/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/rsasaki0109/SpatialRust/releases/tag/v1.0.0
[0.1.0]: https://github.com/rsasaki0109/SpatialRust/releases/tag/v0.1.0
