# Resident k-NN GICP covariance preparation — 2026-10-13

`GicpRegistration::align_with_gpu_resident_covariances` under `register-gicp-gpu`
adds GPU k-NN neighborhoods, sample covariance and plane eigenvectors before
the existing resident transform/normal-equation loop. The default k=20 path
is supported. CPU builds both trees, solves normalized 6x6 equations, converts
the world step and composes/stops poses. Original CPU GICP, explicit
correspondence APIs and `align_with_gpu_reduction` retain their placement and
covariance preparation. Radius covariance remains explicit readback/re-upload;
the new API requires covariance_radius=None.

## Data and ownership boundary

`GpuGicpInput::upload_with_gpu_covariances(runtime, source_tree, target_tree,
k, epsilon)` explicitly uploads original-order source/target positions and
both trees. The original query buffers also supply covariance samples, and
the target tree serves both preparation and iterative matching. The source
tree is needed only for preparation. Two existing bounded nearest-k passes
submit owned original-index/count rows. A third submission computes covariance
and then zeros all covariance payload on any invalid preparation. No host wait,
neighborhood readback, covariance readback or covariance re-upload occurs.
Resource handles referenced by submissions remain alive until completion.

Payload upload is **32*(S+T) + 32*(Ms+Mt) bytes**, with S/T original point
counts and Ms/Mt source/target tree node counts. The previous GPU-reduction
path with CPU k-NN preparation uploads 48*S + 64*T + 32*Mt bytes. Receipts
include the temporary source-tree payload. Control uniforms and device writes
are excluded consistently with existing APIs.

Resident covariance storage is **32*(S+T+1) bytes**: a 32-byte status prefix
followed by original-order source then target rows packed as
`[xx,yy,zz,xy,xz,yz,0,0]`. Prefix status bits 1/2/4 indicate malformed
neighborhood, unsupported numerical arithmetic or invalid SPD/weight bound;
all other prefix words are zero. Any nonzero prefix status zeros BOTH covariance
sets. The existing equation shader reads the prefix directly and sets equation
status bit **16**, zeroing all equation payload. Explicit equation download
rejects it. The existing uploaded-covariance path has offset zero and retains
its original packed row layout, transfers and arithmetic.

Registration downloads only its **256-byte equations per evaluation**, with
I+1 evaluations for I completed updates. Both preparation batches are recorded
as covariance_batches=2. No automatic CPU fallback or placement selection is
added. Prepared equations own their output and survive input drop/pool reuse.
The synchronous alignment/download boundary targets native execution; WASM is
compile checked here, not browser execution tested.

## Math, validation and precision

Neighbor rows include self/duplicates and must contain exactly min(k,N)
neighbors, stored=total, valid original indices and finite nonnegative
distances. Each point's coordinates are centered at its original query and
normalized by its actual local maximum absolute extent. Compensated two-f32
means and second-pass covariance sums form unbiased sample covariance with
denominator count-1. The maximum-pivot Jacobi eigensolver tracks eigenvectors
for at most 32 iterations and uses the CPU physical absolute 1e-12 stopping
threshold divided by local extent squared. A scaled hypot avoids tangent
overflow. The smallest eigenvector is normalized, then covariance is rebuilt
as `I + (epsilon-1)*n*n^T`. Diagonals use `1-n_i^2 + epsilon*n_i^2` to retain
epsilon exactly for axial normals. One sample falls back to regularized
identity (normal X); coincident samples select the same axis convention.

Exact packed-f32 Cholesky pivots and inverse-trace bounds are checked on GPU.
Weight normalization uses 6/epsilon, with slack beyond the ideal 2+1/epsilon
trace. Both trees must be nonempty. k=3..=64 and epsilon as f32 in
1e-4..=1 are supported, along with existing valid pose/configuration/storage/
dispatch requirements. Counts, combined bytes, neighborhood strides, shader
word offsets and combined dispatch sizes are checked before allocation.
Nonzero local extent squared must be normal finite f32. Invalid preparation
propagates through equations instead of silently dropping pairs.

This is an explicit precision path: CPU covariance accumulates raw f32 products
in f64, while this device path uses centered/scaled f32 products. Nearest ties,
rounding/cancellation, small eigengaps and repeated spectra can choose different
normals, weights, poses, fitness or stopping. Repeated smallest eigenvalues
have no unique normal. Such cases are checked through SPD spectral invariants
and Rayleigh energy, without claiming one arbitrary matrix matches CPU.

## Correctness evidence

- Independent centered f64 sample covariance/Jacobi references cover
  1/2/3/63/64/65/257/4097 points, coordinate scales 1e-4/1/1e4 and k=3/20/64.
  **40,968** source rows and their identical target rows are checked for
  regularized SPD eigenvalues and normal Rayleigh energy. Well-separated
  smallest spectra additionally compare all six matrix components; maximum
  error was **1.2342815890e-4**, against a 3e-4 absolute tolerance. Spectral
  eigenvalue error tolerance is 2e-6; Rayleigh tolerance is
  1e-11 + 3e-5*abs(max eigenvalue).
- Device-generated covariances are explicitly downloaded only in tests and
  fed to the existing upload API. Physically rescaled full Hessian/gradient
  and identical Euclidean error verify the offset/status consumer boundary.
- Single/duplicate/two-point fallback, epsilon boundaries, byte/dispatch
  limits, runtime mismatch, owned output, concurrent callers and numerical
  underflow are exercised. GPU-injected malformed counts, indices and infinite
  distances on either source/target invalidate all covariance/equation rows.
- High-level native integration compares **148,392 points in 162 conditions**:
  six sizes, coordinate scales 1e-4/1/1e4, k=6/20/64 and three stopping modes.
  CPU/previous-GPU/new-GPU poses agree to maximum scale-normalized coordinate
  difference **9.536743e-7**. Fitness is independently recomputed; exact
  transfer/evaluation/preparation receipts and unchanged inputs are checked.
  Unequal clouds, outliers, unsupported settings and four concurrent calls
  are also covered.

## Completed-call protocol

Run this exact ignored test alone in release mode:

```text
cargo test -p spatialrust-registration --no-default-features --features register-gicp-gpu --release --test gpu_gicp_resident_covariances_benchmark completed_resident_covariance_gicp_calls -- --include-ignored --exact --nocapture --test-threads=1
```

Two isolated surveys cover 257/4097/16384/65537 source points and k=6/20/64.
k=6 and k=20 use a regular grid; k=64 duplicates each grid point and has 17
additional targets. Input translation is exactly representable and both clouds
keep the same layout. Each full call performs three updates with zero stopping
thresholds. CPU ordinary, previous GPU reductions (CPU k-NN preparation), new
GPU resident preparation and one prepared equation evaluation are sampled in
cyclic order, with three warmups and 20 samples per stage. A supplied runtime
exists before timing; no other local build/test/statistical analysis runs during
measurement.

All full timers include CPU tree construction, preparation, upload as
applicable, GPU completion/copies/waits, CPU solve/composition and final fitness.
The prepared resident timer measures one equation evaluation and completion
wait, excluding all preparation/uploads/download/CPU solve. Its output is
downloaded and checked after timing. Every sampled completed result is checked
against its warmed reference. Logs report backend/device type only.

Measurements used Windows, Rust 1.98.1 release optimization, Vulkan and a
discrete compute adapter, with identical compiled search/parallel features
for all compared paths. Full medians below are milliseconds. Ratios use all
20 paired samples and show the geometric mean; above one favors the new GPU
path. Brackets are within-survey 95% paired log-bootstrap intervals from
10,000 resamples, seed `20261013 + 24*(survey-1) + 2*condition_index + baseline`
(baseline 0=CPU, 1=previous GPU), sorted endpoints 249/9749.

| Survey | Sources | k | CPU full median | Previous GPU full median | New GPU full median | Prepared resident median | CPU/new [95% CI] | Previous/new [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 257 | 6 | 0.80 | 6.33 | 5.99 | 0.33 | 0.14 [0.13, 0.15] | 1.06 [1.01, 1.12] |
| 1 | 257 | 20 | 1.98 | 7.85 | 6.29 | 0.33 | 0.29 [0.27, 0.31] | 1.19 [1.16, 1.23] |
| 1 | 257 | 64 | 3.30 | 9.51 | 7.25 | 0.34 | 0.45 [0.41, 0.48] | 1.36 [1.27, 1.46] |
| 1 | 4,097 | 6 | 21.26 | 14.32 | 7.85 | 0.44 | 2.68 [2.42, 2.94] | 2.06 [1.91, 2.23] |
| 1 | 4,097 | 20 | 36.18 | 31.01 | 13.77 | 1.27 | 2.43 [2.24, 2.62] | 2.24 [2.08, 2.40] |
| 1 | 4,097 | 64 | 77.92 | 66.64 | 35.90 | 1.30 | 2.06 [1.93, 2.20] | 1.85 [1.75, 1.97] |
| 1 | 16,384 | 6 | 107.29 | 78.00 | 35.24 | 4.41 | 3.30 [3.01, 3.64] | 2.27 [2.08, 2.50] |
| 1 | 16,384 | 20 | 168.24 | 133.38 | 55.29 | 2.93 | 3.19 [2.63, 3.97] | 2.38 [1.99, 2.87] |
| 1 | 16,384 | 64 | 352.59 | 340.85 | 39.67 | 1.01 | 4.93 [3.52, 6.89] | 4.77 [3.52, 6.37] |
| 1 | 65,537 | 6 | 453.98 | 250.79 | 76.14 | 7.67 | 5.11 [4.21, 6.21] | 3.05 [2.61, 3.56] |
| 1 | 65,537 | 20 | 520.48 | 395.04 | 81.32 | 1.92 | 5.35 [4.02, 7.05] | 4.12 [3.17, 5.30] |
| 1 | 65,537 | 64 | 1317.60 | 1230.34 | 216.62 | 1.97 | 6.64 [5.59, 7.82] | 6.19 [5.25, 7.32] |
| 2 | 257 | 6 | 0.58 | 4.39 | 4.33 | 0.22 | 0.14 [0.13, 0.14] | 1.01 [0.99, 1.03] |
| 2 | 257 | 20 | 0.95 | 5.13 | 4.87 | 0.27 | 0.21 [0.20, 0.22] | 1.09 [1.07, 1.12] |
| 2 | 257 | 64 | 1.79 | 5.89 | 4.36 | 0.23 | 0.41 [0.39, 0.43] | 1.32 [1.26, 1.38] |
| 2 | 4,097 | 6 | 13.34 | 11.83 | 5.59 | 0.42 | 2.57 [2.39, 2.80] | 2.23 [2.14, 2.34] |
| 2 | 4,097 | 20 | 26.29 | 25.39 | 11.15 | 0.99 | 2.46 [2.18, 2.79] | 2.45 [2.21, 2.74] |
| 2 | 4,097 | 64 | 52.70 | 53.75 | 33.84 | 1.22 | 1.61 [1.53, 1.72] | 1.64 [1.55, 1.74] |
| 2 | 16,384 | 6 | 71.65 | 57.28 | 31.91 | 4.04 | 2.34 [2.16, 2.52] | 1.80 [1.71, 1.90] |
| 2 | 16,384 | 20 | 113.39 | 94.94 | 39.82 | 2.61 | 2.63 [2.05, 3.44] | 2.18 [1.68, 2.85] |
| 2 | 16,384 | 64 | 287.27 | 269.85 | 37.98 | 1.04 | 4.58 [3.42, 6.08] | 4.32 [3.22, 5.72] |
| 2 | 65,537 | 6 | 283.86 | 180.58 | 70.67 | 7.92 | 4.29 [3.59, 5.16] | 2.74 [2.40, 3.13] |
| 2 | 65,537 | 20 | 499.30 | 367.57 | 54.66 | 1.94 | 5.79 [4.41, 7.48] | 4.36 [3.39, 5.56] |
| 2 | 65,537 | 64 | 1270.84 | 1217.04 | 225.03 | 2.02 | 6.19 [5.14, 7.45] | 5.87 [4.95, 6.93] |

All 18 conditions at 4,097+ points had within-survey confidence intervals
above one against both CPU and previous GPU: CPU/new gains **1.61–6.64x**,
previous/new **1.64–6.19x**. At 65,537 points, gains were **4.29–6.64x** against
CPU and **2.74–6.19x** against previous GPU; default k=20 gained 4.12x/4.36x
against previous GPU in the two surveys. Small 257-point calls remained slower
than CPU (CPU/new 0.136–0.446). One small k=6 previous/new interval included
one; no reliable small-call improvement is claimed for that condition.

Timing varies substantially within and between surveys, especially large
k=20/64 calls. Prepared equation costs also depend on the final pose/matches
and covariance weights. Intervals describe paired samples within a survey;
these results do not establish portable gains, crossover sizes or cold-runtime
latency. Every observed sample, including adverse cases, is retained.

All timed outputs matched their warmed references. All final prepared
correspondences retained every source point. Maximum measured CPU/new GPU
pose-coordinate error was **1.4901161e-8**. CPU Euclidean fitness was exactly
zero on these translated fixtures; new fitness was at most 6e-18. Relative
fitness error at this near-zero scale is not a useful accuracy measure.

| Sources | Previous H2D k=6/20 | New H2D k=6/20 | Previous H2D duplicated k=64 | New H2D duplicated k=64 | D2H both GPU paths |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 257 | 29,840 | 18,560 | 31,888 | 20,064 | 1,024 |
| 4,097 | 475,280 | 295,040 | 477,456 | 296,672 | 1,024 |
| 16,384 | 1,900,512 | 1,179,584 | 1,902,688 | 1,181,216 | 1,024 |
| 65,537 | 7,602,320 | 4,718,720 | 7,604,496 | 4,720,352 | 1,024 |

For 65,537 unique points, H2D falls **37.9%**, from 7,602,320 to 4,718,720
bytes, including the new source-tree upload. D2H is unchanged: four 256-byte
equation evaluations for three updates. Previous CPU k-NN preparation had no
covariance D2H; this slice migrates that CPU computation to device and removes
its covariance H2D, without suggesting CPU-only calls had transfers.

## Final verification and delivery

- GPU all-feature library: **197 passed, 20 ignored**; integration targets:
  **27 passed, 3 ignored** (224 passing total). The five new native covariance
  tests and all four prior GICP reduction tests passed together.
- Registration all-feature library: **35 passed, 1 ignored**. Two new native
  integration tests passed explicitly, as did both previous GPU-reduction tests
  including radius preparation. New facade smoke: one passed.
- Both exact release surveys passed: 24 conditions, 20 samples for each of
  four stages, with no broad ignored-benchmark selection.
- All-feature/all-target GPU and registration Clippy, CPU-only GICP Clippy,
  WASM optional registration, minimal facade and final workspace formatting
  passed. No core/math/search code, dependencies, unsafe blocks or automatic
  placement rules were added.
- Source inspection against parent `d0e23f6` confirms CPU
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-registration\src\gicp.rs`
  and the shared pose/fitness/update loop retain their code. Existing uploaded input
  receipts and independent f64 equation error (1.0000217021e-6) remain unchanged.
- Local raw samples and bootstrap results are preserved in
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-covariances-measurements.json`.
  Survey logs are
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-covariances-bench-1.log`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-covariances-bench-2.log`.
  Logs/notes omit the local adapter model; publication scan reported zero
  matches. Existing 124 named CI checks remain required before merge.
