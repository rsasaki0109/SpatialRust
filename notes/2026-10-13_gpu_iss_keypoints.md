# Resident GPU ISS keypoints

Date: 2026-10-13. Named slice: ISS radius graphs, covariance/eigenvalue saliency
and local maxima on GPU, following the resident neighborhood and FPFH APIs.

## Execution and transfer contract

`feature-iss-gpu` exposes
`IssKeypointDetector::detect_with_gpu(runtime, cloud, salient_capacity, non_max_capacity)`
in features and the facade. Ordinary detection retains its existing CPU path.
The explicit GPU wrapper builds a CPU KD-tree, uploads it and original queries,
runs both radius searches and the two ISS passes on device, then downloads
flags/status. Ordered flag compaction and schema/attribute gathering run on CPU.
No neighborhood/eigenvalue readback or automatic CPU fallback occurs here.

`GpuIssInput::upload` under `gpu-wgpu` is reusable on one runtime. The query
position buffer is shared with covariance computation; no duplicate sample
buffer is uploaded. For N original points and M tree nodes, H2D payload is
`32*N + 32*M` bytes: 16*N tree positions, 16*N queries and 32*M nodes. Empty
input transfers zero. CPU tree construction is external to this resident API.

`neighborhoods_resident` retains separate salient and non-max radius graphs tied
to that exact uploaded input. Creating them submits search without readback or
a completion wait. Capacities include self and must fit ALL matches in every
row, even rows that will not qualify as candidates. A truncated prefix is not
accepted as a complete ISS neighborhood.

`keypoints_resident` uses two ordered device passes in one submission. The first
validates BOTH graphs' counts, target indices and stored distance gates before
any downstream selection. It computes centroid covariance in coordinates
relative to the query, scaled by actual local extent with compensated scalar
accumulation. Max-pivot Jacobi computes ascending eigenvalues without unused
eigenvectors, using 32 iterations and a physical 1e-12 stopping rule. A scaled
hypot avoids tangent overflow. Gamma ratio tests produce eligibility; the second
pass retains local maxima of the smallest eigenvalue. Equal COMPUTED GPU maxima
all survive. Original point ordering is retained in flags and downloaded indices.

Positions and global extent squares must be finite. Radii must be positive with
finite nonzero f32 squares; gamma values must be finite in [0,1]. These checks
are stricter than ordinary CPU ISS and do not change its configuration behavior.
Minimum support includes self and covariance always needs at least three points.
Huge usize support requirements map to u32::MAX and reject all supported point
counts. Zero local extent produces an ineligible zero row; an evaluated nonzero
extent with a subnormal/zero f32 square sets numerical-error status. A huge search
radius around a small cloud does not dictate covariance scaling.

Returned `GpuIssKeypoints` owns its flags, scores and status independently of
inputs and other results; none are pooled. Flags are original-order u32 0/1.
Scores are original-order vec4<f32>: ascending physical eigenvalues and
eligibility 0/1. Rejected ratios still expose eigenvalues; insufficient support
has a zero score row. Empty physical buffers have unused 4/16 bytes respectively.
The shader requires six storage bindings, one 32-byte uniform and 64 invocations
per workgroup. Buffer sizes, shader indices, dispatch, host bounds and bindings
are checked before allocation. No dependency/core change or unsafe code is added.

Eight u32 status words contain bad salient rows, bad NMS rows, required salient
capacity, required NMS capacity, malformed rows, last bad row plus one, numerical
errors and a reserved word. Any nonzero status invalidates the complete result.
The second pass zeros ALL flags/scores after graph/numerical errors; downloads
reject status before decoding. Custom GPU consumers must check status and treat
exposed storage as immutable application data.

Resident search/ISS transfers zero payload bytes. `download` waits and explicitly
copies `4*N + 32` D2H bytes, then compacts flags on CPU. Optional
`download_scores` separately waits and copies `16*N + 32` D2H bytes for diagnostic
eigen/eligibility rows; the wrapper never calls it. Both empty transfers are
zero. Uniform controls are excluded from payload receipts. WASM evidence is
compilation only; browser execution of synchronous downloads was not tested.

## Precision and correctness evidence

GPU covariance/eigenvalue arithmetic is f32; CPU ISS uses f64. Thresholds,
nearly zero eigenvalues and nearly tied NMS scores can produce different indices.
This API does not guarantee identical CPU indices across inputs/backends or
equal computed saliencies for all mathematically identical neighborhoods. The
existing CPU implementation remains available when its exact decisions matter.

Four explicitly run adapter integration tests cover planar, curved/noisy and
coincident-pair fixtures at 0/1/2/17/63/64/65/257/4097 points and radii 0.075/0.16.
They audit 27,396 decisions and 82,188 eigenvalues. Every eigenvalue is compared
with f64 covariance/Jacobi under the fixture bound
`1e-12 + 5e-6 * abs(reference covariance trace)`. Maximum observed absolute
eigenvalue error is 0.000000007803552025051341. This is selected-fixture evidence,
not a universal numerical error guarantee.

All device flags match independent CPU NMS using the downloaded GPU scores and
CPU radius graph. Compared with ordinary f64 detection, 202 decisions differ
(0.74% of audited input decisions). Every difference is checked against a
threshold uncertainty band or a competing saliency gap within the sum of the
two eigenvalue bounds. Radius-0.16, 4,097-point curved/coincident cases match CPU
indices exactly; radius-0.075 cases and partial narrow clouds include differences.
The accuracy receipts retain every case, including adverse cases.

Independent analytic box tests verify eigenvalues 0.0004/0.0016/0.0036 within
2e-9 and all eight equal maxima surviving; isolated/linear/coincident support
has no keypoints. Tests exercise both graph overflows and poisoned indices,
exact status and complete output zeroing, empty transfers, numeric underflow,
huge radii, support limits, retained results, four concurrent callers and
runtime/frame ownership. Scalar F32/F64/U8/U16/U32/I32 attributes (including
NaN payloads and signed zero), custom position semantics, schema and metadata
are preserved bit for bit in the selected cloud.

All-feature GPU regression (203 passed, 23 ignored), feature library regression
(23 passed, two ignored), facade integration, all-feature and CPU-only
Clippy with warnings denied, minimal GPU compilation, WASM feature compilation
and workspace formatting passed. Source comparison verifies that existing CPU
ISS math, selection and gathering are unchanged. Two isolated feature CI cases
are added; hosted CI compiles adapter tests rather than proving GPU execution.

## Completed-call measurements

Release builds on a native Vulkan discrete-device backend; adapter model names
are omitted. Three warmups per stage precede 20 paired rounds with rotating
CPU/full-GPU/resident stage order. Two surveys run separately with no other
local build, test or statistical analysis during timing. All samples are retained.

CPU timings use the current public detector, including its bounded worker policy
(four workers at 65,537 points, one for smaller sizes). Full GPU timings include
validation, CPU tree construction, input uploads, BOTH radius graphs, graph
validation, covariance/eigensolves, NMS, flags/status download, completion wait,
ordered compaction and selected-cloud gathering. Runtime/pipeline creation is
warmed outside timers. Result verification/drop is excluded; internal staging
and tree destruction is included in complete calls.

Resident timings include owned result allocation, graph validation, covariance/
eigenvalue saliency, NMS, submission and `wait_idle`. They exclude CPU preparation,
input uploads, BOTH neighborhood searches and download/compaction/gathering.
These columns measure a prepared device phase and cannot establish whole-call
or global-registration performance on arbitrary inputs.

P is a planar 128-column grid with spacing 0.05; C adds curved height and
deterministic z noise; D duplicates consecutive C points. Salient radius is
0.11 for P and 0.16 for C/D; NMS radius is 0.75 times that. Salient capacity is
64 for P/C and 96 for D; NMS capacity is 64 throughout. Numerical audits run
before timing; repeated CPU calls match their CPU reference, and repeated GPU
calls match their separately audited GPU reference. Selected counts and
CPU/GPU decision differences accompany performance: the two precision modes
can return different indices, so ratios do not imply accuracy-equivalent output.

Times below are medians in milliseconds. Ratios are geometric means of 20
paired CPU/full-GPU timings. Percentile 95% bootstrap intervals use 10,000
resamples of paired log ratios with seeds 20261013 plus zero-based record index.
They do not establish a deployment crossover or remove run-to-run variability.

| Survey | Points | Fixture | Workers | CPU ms | Full GPU ms | Resident ms | CPU/GPU selected | Decision differences | Paired full-call ratio (95% interval) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 257 | P | 1 | 0.250 | 1.465 | 0.201 | 256/256 | 0 | 0.17x (0.16-0.18) |
| 1 | 257 | C | 1 | 0.389 | 1.492 | 0.242 | 44/30 | 14 | 0.28x (0.26-0.30) |
| 1 | 257 | D | 1 | 0.322 | 1.436 | 0.200 | 248/248 | 0 | 0.23x (0.22-0.24) |
| 1 | 4,097 | P | 1 | 5.344 | 3.111 | 0.241 | 624/624 | 0 | 1.72x (1.66-1.80) |
| 1 | 4,097 | C | 1 | 12.167 | 3.006 | 0.298 | 421/421 | 0 | 3.99x (3.81-4.18) |
| 1 | 4,097 | D | 1 | 14.596 | 2.969 | 0.395 | 444/444 | 0 | 4.96x (4.77-5.16) |
| 1 | 16,384 | P | 1 | 23.620 | 7.394 | 0.548 | 1008/1008 | 0 | 3.13x (2.71-3.56) |
| 1 | 16,384 | C | 1 | 57.514 | 24.491 | 10.328 | 2046/2046 | 0 | 2.74x (2.50-3.01) |
| 1 | 16,384 | D | 1 | 87.431 | 23.404 | 10.637 | 1762/1762 | 0 | 3.06x (2.67-3.48) |
| 1 | 65,537 | P | 4 | 39.465 | 36.799 | 9.382 | 2544/2544 | 0 | 1.06x (1.00-1.12) |
| 1 | 65,537 | C | 4 | 97.191 | 32.245 | 3.352 | 5412/5412 | 0 | 2.69x (2.37-3.05) |
| 1 | 65,537 | D | 4 | 125.311 | 35.082 | 8.125 | 6483/6483 | 0 | 3.55x (3.32-3.76) |
| 2 | 257 | P | 1 | 0.221 | 1.296 | 0.165 | 256/256 | 0 | 0.17x (0.16-0.18) |
| 2 | 257 | C | 1 | 0.341 | 1.193 | 0.167 | 44/30 | 14 | 0.28x (0.27-0.29) |
| 2 | 257 | D | 1 | 0.273 | 1.213 | 0.176 | 248/248 | 0 | 0.22x (0.21-0.23) |
| 2 | 4,097 | P | 1 | 4.002 | 2.484 | 0.241 | 624/624 | 0 | 1.68x (1.64-1.72) |
| 2 | 4,097 | C | 1 | 13.223 | 3.265 | 0.323 | 421/421 | 0 | 3.82x (3.66-3.98) |
| 2 | 4,097 | D | 1 | 12.090 | 3.077 | 0.530 | 444/444 | 0 | 2.87x (2.06-3.97) |
| 2 | 16,384 | P | 1 | 19.347 | 6.616 | 0.525 | 1008/1008 | 0 | 2.91x (2.79-3.03) |
| 2 | 16,384 | C | 1 | 68.500 | 14.967 | 4.334 | 2046/2046 | 0 | 5.65x (4.82-6.66) |
| 2 | 16,384 | D | 1 | 82.495 | 22.265 | 9.680 | 1762/1762 | 0 | 3.12x (2.75-3.51) |
| 2 | 65,537 | P | 4 | 43.909 | 40.607 | 9.406 | 2544/2544 | 0 | 1.06x (0.97-1.14) |
| 2 | 65,537 | C | 4 | 95.802 | 32.239 | 3.344 | 5412/5412 | 0 | 2.68x (2.33-3.07) |
| 2 | 65,537 | D | 4 | 130.787 | 37.264 | 8.104 | 6483/6483 | 0 | 3.59x (3.40-3.79) |

At 65,537 points, both surveys have identical CPU/GPU indices on C/D, with
paired complete-call ratios 2.68-3.59x against four-worker CPU detection. P is
about 1.06x and its bootstrap intervals reach/include 1, giving no clear gain.
All 257-point cases favor CPU, and C differs by 14 of 257 decisions. Timing
variation remains visible between surveys and stages; these synthetic fixtures
establish neither a universal gain nor an accuracy-equivalent deployment claim.
Benchmark audits have maximum absolute eigenvalue error 0.00000000935703506517449
under the same trace-based fixture bound; diagnostics run outside timed calls.

## Reproduction

From `C:/Users/rsasa/Workspace/SpatialRust`, run adapter checks explicitly:

```powershell
cargo test -p spatialrust-features --no-default-features --features feature-iss-gpu --test gpu_iss -- --include-ignored --nocapture
cargo test -p spatialrust --no-default-features --features feature-iss-gpu --test gpu_iss_smoke -- --include-ignored
```

Run this exact benchmark filter alone twice to exclude the included reference
module's adapter tests from timing runs:

```powershell
cargo test -p spatialrust-features --no-default-features --features feature-iss-gpu --test gpu_iss_benchmark --release completed_iss_calls -- --include-ignored --nocapture
```

Local untracked evidence:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-integration-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-native-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-feature-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-facade-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-iss-measurements.json`
