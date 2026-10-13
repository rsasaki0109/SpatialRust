# Resident radius GICP covariance preparation — 2026-10-13

`GicpRegistration::align_with_gpu_resident_radius_covariances(runtime, source,
target, neighbor_capacity)` under `register-gicp-gpu` removes the explicit
radius covariance readback and re-upload from the previous GPU reduction path.
Both inclusive radius searches, covariance sums and plane eigenvectors remain
on the caller's GPU runtime and feed the existing resident equation loop.
CPU builds both KD-trees, solves normalized 6x6 equations, converts the world
step and composes/stops poses. Ordinary GICP and previous GPU methods retain
their covariance placement and fallback/error contracts.

## Explicit preparation and transfer boundary

`GpuGicpInput::upload_with_gpu_radius_covariances(runtime, source_tree,
target_tree, radius, capacity, epsilon)` reuses original source/target query
buffers as covariance samples and shares the target tree with registration.
The temporarily uploaded source tree serves preparation. Two existing radius
passes submit owned rows; a third submission builds/validates covariances then
zeros BOTH sets on any failure. No host wait or neighborhood/covariance
readback/re-upload is introduced. Submitted resources remain alive until GPU
completion; owned equations survive input drop and scratch reuse.

Capacity is a positive per-point bound, including self and duplicate indices.
Radius membership uses the existing inclusive f32 squared-distance gate and
near-first tree traversal. Rows must have stored=total<=capacity and total<=N;
zero support, invalid indices, nonfinite/negative distances or distances beyond
the radius are rejected. Overflow never becomes a truncated covariance.
Capacity need not be <=64 (unlike k-NN); checked device storage/word/dispatch
limits constrain it. Neighborhood storage per dataset is N*(8+8*capacity)
bytes. This is a fixed-memory explicit contract, with no automatic host-driven
retry or placement choice.

Covariance storage is 32*(S+T+1) bytes: a zero-reserved 32-byte status prefix
then source/target `[xx,yy,zz,xy,xz,yz,0,0]` rows in original order. Preparation
status 1/2/4/8 means malformed graph, unsupported arithmetic, invalid SPD/weight
bound or radius overflow. Every failure zeros all covariance rows and carries
equation status 16, zeroing all 256-byte equation payload before download rejects
it. Radius overflow details do not require a separate production readback.

Upload is **32*(S+T+source_nodes+target_nodes)** payload bytes, including source
tree preparation. Only **256*(I+1)** bytes are downloaded for I updates;
`covariance_batches=2`. Device writes/control uniforms are excluded consistently
with existing receipts. The previous reduction method uploads
48*S+64*T+32*target_nodes plus each radius grid's 16*N+4*(cells+1), and downloads
32*(S+T) covariance bytes plus 256*(I+1) equation bytes. The new method removes
that covariance download and upload but adds source-tree staging; receipts
account for the complete tradeoff.

## Math and limits

With fewer than three neighbors, the device emits **epsilon*I**, preserving the
existing radius-grid sparse fallback. Coincident support of at least three
selects normal X, rebuilding diagonal `(epsilon,1,1)`. Other support uses the
query origin, actual maximum absolute local extent, compensated two-f32 means
and second-pass central products. Radius covariance uses unnormalized central
sums with physical absolute **1e-20** Jacobi stopping divided by extent squared.
The shared solver tracks eigenvectors for at most 32 maximum-pivot steps and
uses scaled hypot to avoid tangent overflow. The smallest normalized normal
rebuilds `I+(epsilon-1)*n*n^T`; packed Cholesky/inverse trace checks protect the
existing common weight normalization 6/epsilon. Sparse epsilon*I has inverse
trace 3/epsilon, inside that bound.

Both trees must be nonempty/finite, capacity positive, epsilon as f32 in
1e-4..=1, radius finite/positive with normal finite f32 square. Nonzero local
extent squared must also be normal finite f32. Combined dispatches, node bytes,
neighborhood strides and u32 offsets are checked before allocation. Existing
GICP input/configuration/pose/conditioning restrictions still apply.

The existing grid uses raw absolute f32 means, plain sums and 16 sweeps of
three rotations. This new explicit precision path uses centered/scaled sums,
a different traversal order and maximum-pivot iteration. It does not promise
bitwise grid equivalence. Eigenvalues with small/repeated gaps have no stable
unique normal; gates, poses, fitness and stopping can differ. The k-NN path
retains count-1 sample covariance and physical 1e-12 stopping, and its previous
f64 component error is unchanged.

## Correctness evidence

- Brute-force inclusive memberships, centered f64 covariance, SPD eigenweights
  and normal Rayleigh energy cover 1/2/3/63/64/65/257/1025 points, scales
  1e-4/1/1e4 and radii 0.01/0.8/20 times scale: **13,320** source rows and
  identical target rows. Unique-spectrum/sparse component error is at most
  **1.5026412008e-5**; repeated spectra are checked without forcing an arbitrary
  normal. Native legacy grid planes with unique normals match within 3e-5.
- Exact-radius/next-representable membership, sparse epsilon boundaries,
  65 duplicate samples at capacity 64 versus 65, source/target overflow and
  tiny local extents exercise validity propagation and fully zero output.
  Malformed source/target counts/indices/distances, including out-of-radius
  finite distances, invalidate both covariance sets and all equations.
- Radius-derived covariances feed the existing uploaded-covariance equations
  with matching Euclidean error and physically rescaled Hessian/gradient
  tolerances. Runtime mismatches reject; outputs survive input drop.
- **162** registration conditions cover six cloud sizes, three scales, three
  radii and three stopping configurations. Host-update/previous GPU pose, stopping,
  fitness, immutable inputs and exact receipts agree within declared tolerance;
  maximum scale-normalized host-update pose-coordinate error is **9.536743e-7** across
  **148,392** compared points. Unequal clouds/outliers, four concurrent complete
  alignments and invalid capacities/configuration are also checked. The host
  update reference uses legacy shared-runtime GPU radius covariances; this
  comparison does not imply pure CPU radius preparation.
- A 4,098-point sparse cloud with radius 0.01 exceeds the existing dense grid
  cell cap but completes the resident-radius route at capacity two with zero
  fitness and fixed-size readback. The new sparse representation avoids a
  dense grid allocation. Minimal facade exposes the method.

## Completed-call measurements

The exact native release test is
`completed_resident_radius_covariance_gicp_calls` in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-registration\tests\gpu_gicp_resident_radius_covariances_benchmark.rs`.
Runtime construction is outside timing. Each condition has three warmups and
20 samples per stage, cyclically rotating host-update, previous GPU reduction,
new resident-radius alignment and prepared resident equations. Complete APIs
include validation, CPU trees/grid preparation, uploads, covariance computation,
all waits/readback, CPU solves/pose composition and final fitness. Prepared
equations include submission/completion for one evaluation on pre-uploaded,
precomputed input, excluding preparation/upload, readback and CPU solve. That
stage is recorded separately and is not an alignment latency claim.

Four sizes (257/4097/16384/65537) use fixed 0.5 spacing, 32x32 grid coordinates.
Profiles 0/1 use unique points, radii 0.51/0.8 and capacities 32/64; profile 2
duplicates each grid position, radius 1.05, capacity 128 and 17 extra targets.
Every source starts at binary translation `(0.03125,-0.015625,0.0078125)`.
Three updates, distance gate 0.25, epsilon 0.001 and zero stopping thresholds
are fixed. Every warm/timed output is checked against its warmed reference;
final correspondences must retain all source points.

The **host-update** baseline uses ordinary CPU matching, Hessian/gradient,
fitness and pose solve, with legacy shared-runtime GPU radius covariances.
The **previous GPU** baseline uses explicit grid GPU covariance readback and
re-upload on the caller runtime. Neither baseline is labeled as pure CPU radius
preparation. Surveys run alone without simultaneous local builds, tests or
statistical processing. Backend, device type and relevant limits are recorded
without the local adapter model.

### Survey 1

| N | Radius / capacity | Host updates ms | Previous GPU ms | Resident radius ms | Prepared equations ms | Host/new ratio [95%] | Previous/new ratio [95%] |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 257 | 0.51 / 32 | 3.195 | 9.315 | 6.597 | 0.478 | 0.49 [0.48–0.51] | 1.43 [1.39–1.47] |
| 257 | 0.8 / 64 | 3.265 | 9.489 | 6.669 | 0.692 | 0.50 [0.48–0.52] | 1.41 [1.34–1.47] |
| 257 | 1.05 / 128 | 4.146 | 12.016 | 8.560 | 0.616 | 0.49 [0.45–0.53] | 1.36 [1.25–1.45] |
| 4,097 | 0.51 / 32 | 11.646 | 11.338 | 8.067 | 0.819 | 1.54 [1.45–1.65] | 1.44 [1.37–1.51] |
| 4,097 | 0.8 / 64 | 13.040 | 11.757 | 9.262 | 0.835 | 1.39 [1.33–1.47] | 1.25 [1.21–1.28] |
| 4,097 | 1.05 / 128 | 12.774 | 12.282 | 10.604 | 0.777 | 1.18 [1.12–1.24] | 1.14 [1.11–1.17] |
| 16,384 | 0.51 / 32 | 64.579 | 21.636 | 18.509 | 1.729 | 3.48 [3.29–3.68] | 1.14 [1.09–1.19] |
| 16,384 | 0.8 / 64 | 56.786 | 18.855 | 18.636 | 1.792 | 3.02 [2.85–3.18] | 1.00 [0.96–1.04] |
| 16,384 | 1.05 / 128 | 41.507 | 18.955 | 21.746 | 1.730 | 1.96 [1.83–2.11] | 0.87 [0.84–0.90] |
| 65,537 | 0.51 / 32 | 211.292 | 49.043 | 51.044 | 3.529 | 4.12 [3.94–4.31] | 0.97 [0.93–1.01] |
| 65,537 | 0.8 / 64 | 214.940 | 50.190 | 58.759 | 3.645 | 3.70 [3.52–3.88] | 0.88 [0.84–0.93] |
| 65,537 | 1.05 / 128 | 200.584 | 52.286 | 75.540 | 3.552 | 2.57 [2.43–2.73] | 0.68 [0.65–0.72] |

### Survey 2

| N | Radius / capacity | Host updates ms | Previous GPU ms | Resident radius ms | Prepared equations ms | Host/new ratio [95%] | Previous/new ratio [95%] |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 257 | 0.51 / 32 | 2.805 | 7.482 | 5.113 | 0.246 | 0.55 [0.53–0.56] | 1.45 [1.42–1.48] |
| 257 | 0.8 / 64 | 2.816 | 7.415 | 5.126 | 0.246 | 0.54 [0.52–0.55] | 1.42 [1.37–1.46] |
| 257 | 1.05 / 128 | 2.819 | 7.521 | 5.213 | 0.248 | 0.54 [0.52–0.55] | 1.45 [1.40–1.49] |
| 4,097 | 0.51 / 32 | 10.401 | 8.828 | 6.452 | 0.425 | 1.66 [1.58–1.74] | 1.39 [1.33–1.45] |
| 4,097 | 0.8 / 64 | 12.142 | 9.814 | 8.040 | 0.475 | 1.50 [1.41–1.59] | 1.24 [1.19–1.29] |
| 4,097 | 1.05 / 128 | 11.971 | 12.157 | 13.633 | 0.889 | 0.95 [0.88–1.02] | 0.90 [0.85–0.94] |
| 16,384 | 0.51 / 32 | 69.468 | 30.023 | 32.468 | 3.079 | 2.10 [1.98–2.21] | 0.91 [0.88–0.94] |
| 16,384 | 0.8 / 64 | 57.557 | 21.499 | 26.536 | 2.042 | 2.01 [1.88–2.16] | 0.77 [0.73–0.81] |
| 16,384 | 1.05 / 128 | 46.626 | 17.234 | 20.600 | 1.194 | 2.12 [1.91–2.33] | 0.80 [0.73–0.85] |
| 65,537 | 0.51 / 32 | 221.118 | 67.137 | 73.491 | 7.073 | 3.22 [2.94–3.58] | 0.93 [0.89–0.96] |
| 65,537 | 0.8 / 64 | 216.502 | 49.141 | 56.165 | 2.227 | 3.37 [2.90–3.89] | 0.83 [0.78–0.89] |
| 65,537 | 1.05 / 128 | 227.242 | 49.142 | 66.858 | 1.744 | 3.00 [2.62–3.37] | 0.68 [0.59–0.75] |

Times are medians in milliseconds. Ratios are geometric means of paired
baseline/new latency; >1 is faster for the new path. The 95% intervals are
10,000 paired log-ratio bootstrap samples, seeded by
20261013+24*(survey-1)+2*condition_index+baseline_index (host=0, previous=1),
using sorted endpoints 249/9749. All raw samples, including adverse results,
are preserved. Intervals describe each survey, not backend/device portability,
cold-runtime latency or a universal crossover. Large within/between-survey
variance also appears in the separate prepared-equation stage.

For 257 points, the new route is 1.36–1.45x faster than previous GPU reduction
but remains slower than host updates (host/new 0.49–0.55). At 4,097 points,
smaller radii improve on previous GPU by 1.24–1.44x; duplicated/larger support
changes from 1.14x in survey 1 to 0.90x in survey 2. At 16,384 points, gains do
not repeat across surveys and duplicated support is consistently slower.
At 65,537 points, host updates/new ratios are 2.57–4.12x, but previous GPU/new
ratios are only 0.68–0.97: **this route does not establish a large-cloud latency
improvement over the existing GPU implementation**. Source-tree staging and
fixed-capacity neighborhood allocation/computation are additional costs;
these completed-call measurements do not isolate their individual shares.
This is an opt-in residency/sparse-grid-capacity improvement, with measured
latency tradeoffs, and does not change default placement.

All warmed and timed outputs matched their warmed references. Every final
prepared evaluation retained N correspondences. Maximum measured host/new
pose-coordinate error is **3.7252903e-9**; host Euclidean fitness is zero and
new fitness is at most **5.4620291e-19**. Relative error around zero is not a
useful precision measure. Both native surveys passed (37.43/39.98 seconds,
excluding compilation), using Vulkan/DiscreteGpu and reported storage limit
2,147,483,647 bytes, maximum dispatch dimension 65,535; required storage
bindings are eight. No local adapter model is recorded.

| N | Radius | Previous H2D bytes | New H2D bytes | Previous D2H bytes | New D2H bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 257 | 0.51 | 40,056 | 18,560 | 17,472 | 1,024 |
| 257 | 0.8 | 39,032 | 18,560 | 17,472 | 1,024 |
| 257 | 1.05 | 40,632 | 20,064 | 18,016 | 1,024 |
| 4,097 | 0.51 | 637,144 | 295,040 | 263,232 | 1,024 |
| 4,097 | 0.8 | 615,992 | 295,040 | 263,232 | 1,024 |
| 4,097 | 1.05 | 610,640 | 296,672 | 263,776 | 1,024 |
| 16,384 | 0.51 | 2,540,128 | 1,179,584 | 1,049,600 | 1,024 |
| 16,384 | 0.8 | 2,456,808 | 1,179,584 | 1,049,600 | 1,024 |
| 16,384 | 1.05 | 2,434,456 | 1,181,216 | 1,050,144 | 1,024 |
| 65,537 | 0.51 | 10,183,856 | 4,718,720 | 4,195,392 | 1,024 |
| 65,537 | 0.8 | 9,830,712 | 4,718,720 | 4,195,392 | 1,024 |
| 65,537 | 1.05 | 9,730,760 | 4,720,352 | 4,195,936 | 1,024 |

For 65,537 source points the previous radius-covariance D2H is
4,195,392–4,195,936 bytes; the new completed three-update call downloads
**1,024 bytes** (99.9756% less). H2D is 4,718,720–4,720,352 versus
9,730,760–10,183,856 bytes (51.49–53.67% less), including the new source tree.
The resident radius/neighborhood buffers still consume device memory; fewer
transfers do not imply faster complete calls or a smaller total memory footprint.

## Final verification and delivery

- GPU all-feature library: **200 passed, 20 ignored**; integration targets:
  **27 passed, 3 ignored**, 227 passing total. Radius covariance tests and
  expanded source/target malformed-row tests pass, as do previous k-NN and
  independent f64 GICP equations.
- Registration all-feature library: **35 passed, 1 ignored**. Three new native
  radius integration tests pass, plus previous k-NN's 162 conditions and both
  previous reduction tests' 108 conditions. New minimal facade smoke passes.
- All-feature/all-target GPU/registration Clippy, CPU-only GICP Clippy, WASM
  optional registration, minimal facade, benchmark-target Clippy and final
  workspace formatting pass. WASM is compile checked, not browser executed.
- No core/math/search code, dependencies, unsafe or automatic placement rules
  were added. CPU GICP and the common post-preparation pose/fitness/update loop
  are unchanged against parent 5df2a62.
- Local measurement JSON:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-radius-covariances-measurements.json`.
  Raw surveys:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-radius-covariances-bench-1.log`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-radius-covariances-bench-2.log`.
  Publication files/logs omit the local adapter model. Existing 124 named CI
  checks remain required before merge, together with the pinned parent slice.
