# GICP working-set reuse and bounded CPU preparation

Date: 2026-10-13. Named slice: reduce resident GICP working buffers and the
existing GPU-entry-point CPU k-NN preparation, with completed public registration
measurements against the original implementation.

## Execution, memory and precision

Nearest matching completes before preparation consumes each transformed query.
Preparation builds a complete local logical pair using the same covariance
rotation, Cholesky inverse, normalization and finite/conditioning checks. Its
centered position/count row overwrites that invocation's query; the two inverse
weight rows and residual/count row occupy a separate 48-byte record. Every valid
or rejected component is stored once. Reduction reconstructs the original
logical pair. No invocation consumes another source's query during preparation.

The five ordered passes and single submission remain; pass boundaries order
the changed storage accesses. No workgroup barrier, production copy, extra pass,
submission or host wait is introduced. Eight storage bindings, nine bindings
total, 96 uniform bytes, 1024 workgroup bytes, compensated accumulation order,
immutable inputs and independently owned 256-byte equations retain their
contract. Any nonzero status still zeros all result payload; invalid generated
covariances propagate bit 16. Zero/rejected pairs never consume stale rows.

For nonempty source count S and G=min(ceil(S/64),256):

| Private binding | Previous bytes | Reused bytes |
| --- | ---: | ---: |
| Transformed queries / later centered positions-counts | 16*S | 16*S |
| Nearest matches | 8*S | 8*S |
| Prepared pairs / now inverse weights-residuals | 64*S | 48*S |
| Compensated partials | 240*G | 240*G |
| Total | 88*S + 240*G | 72*S + 240*G |

Savings are **16*S bytes**, or **18.18% of source-sized private scratch**.
At 1,048,577 sources this saves **16,777,232 bytes**. These requested working
buffer bytes exclude resident positions/tree/covariances, output, uniforms,
staging, retained caches and driver overhead. The largest source-sized pair
binding drops from 64*S to 48*S. The separate combined covariance binding
32*(S+T+offset) and existing source/target/dispatch/index/result limits remain;
equal-sized source/target covariances can still determine the supported count.

Default scratch retention stays 64 MiB and does not limit live allocations.
The cold large-buffer inspection test uses a private 128-MiB test budget so all
four inspected leases remain cached; production/measurements retain the default.
At 1,048,577 sources, recycling queries then matches already retains 24*S, so
neither the old 64*S pair nor new 48*S record fits the remaining default budget.
Requested working savings therefore do not imply identical retained-cache
savings at every size. Measured counters are retained with the samples.

Explicit uploaded-covariance payload remains **48*S + 64*T + 32*M bytes**,
where M is target tree-node count. Resident k-NN/radius/grid covariance routes
keep their existing receipts and have no added covariance download/re-upload.
Each nonempty evaluation explicitly downloads 256 bytes. Three updates use four
evaluations and 1024 equation-readback bytes. Controls/scratch writes remain
outside the existing payload receipt; empty primitive input still reads back zero.

For `align_with_gpu_reduction` with CPU k-NN covariance preparation, each
independent point uses the original nearest-k order, f64 accumulator and
eigenvector regularization through a shared private `covariance_at` function.
It directly packs six f32 components into a 24*S output vector, avoiding the
intermediate 72*S vector of f64 3x3 matrices. A bounded scoped worker handles
each disjoint output chunk: at most one worker per 16,384 points and available
CPU parallelism. Below 32,768 points, one available worker and WASM are serial.
Source and target preparation are sequential; 65,537-point clouds use four
workers on the measured host. Ordinary CPU GICP retains serial execution and
the same shared per-point math. GPU-generated covariance routes retain their
existing preparation. No dependency, automatic placement/fallback, public
unsafe API or core/math/search implementation change is added.

## Correctness references

The actual old GPU WGSL, host planning/allocation and equation submission are
frozen from `e7cabc273e2f983524b6517c4767a73d8b879995` (PR #187). Host reference
changes only rename methods and inject a frozen pipeline cache. A separate
serial CPU covariance oracle preserves the parent loop and neighbor/accumulator
order. Neither reference substitutes approximate math for the original route.

- **747 conditions / 47,808 exact output u32 words** cross source counts
  0/1/3/63/64/65/257/4097/65537; single-target, lattice and duplicate/unequal
  geometry; isotropic, non-diagonal SPD and anisotropic covariances;
  covariance scales 1e-8/1/1e8; inclusive distance gates 0.001/0.25/100 and
  rotated/identity poses. Each nonempty regular call takes four pre-poisoned
  all-one scratch leases with four confirmed cache hits and no allocation.
  Eighteen identity-pose conditioning failures preserve status 8 and zero
  payload. Rotating an anisotropic source can improve the combined covariance;
  the rotated cases preserve that supported behavior too.
- **90 conditions / 5,760 exact words** cover resident k-NN, radius and grid
  covariances, ordinary/duplicate geometry, varied gates, poisoned working
  buffers and deliberate radius/grid capacity overflow. Invalid preparation
  preserves status 16 and zero payload in old/new outputs.
- Exact four distinct lease sizes and cold allocation counters at
  1/65/65537/1048577 sources. Synthetic 48-KiB limits accept the new 1024-source
  pair binding while rejecting the old binding; extra sources/covariances,
  combined equal-cloud covariance bytes, dispatch and u32 bounds still reject.
- Twelve varied queued calls and eight concurrent callers keep owned output
  words/decoded equations equal after input drop and repeated pool reuse.
- **1,108,059 packed CPU covariance rows** match the frozen f64 oracle bit for
  bit, with 1/2/6 requested workers, worker/chunk boundary counts, duplicate and
  constant geometry, k=3/6/20, coordinate scales 1e-4/1/1e4 and epsilon choices.
  The ordinary serial CPU routine also matches all nine f64 matrix components.
  Worker bounds and four concurrent preparations preserve output order.
- Original CPU registration comparisons retain **594 distinct conditions /
  544,104 points**: 108 legacy covariance and 162 each resident k-NN/radius/grid
  conditions. Maximum scale-normalized CPU pose-coordinate error remains
  **9.536743e-7**, below the unchanged 1e-4 tolerance. Configuration, singular,
  outlier/unequal, schema, capacity and concurrent-call cases also pass.

## Completed registration measurements

The identical ignored integration test is compiled once against the original
GPU/registration sources before production edits, then against the final sources.
The original executable and SHA-256 are saved locally. Its benchmark-source
hash must match the final test source. Both long-lived processes use real public
GICP APIs, independent runtimes and default 64-MiB pools. The controller sends
one command at a time, so the other process is idle while a call computes.
Runtime/process initialization and caller-owned synthetic cloud construction
are outside timing; input validation, CPU trees, covariance preparation,
packing/transfers, all GPU passes/copies/waits, CPU solves, pose updates and
owner cleanup inside the public call are timed.

Eight conditions use a 0.5-spaced lattice and a small source translation.
Duplicate profiles repeat positions and add 17 targets. Each call has three
updates with zero stopping thresholds, k=6, match distance 0.25 and radius 0.8
for radius/grid routes; their capacity is 64, including self/duplicates.
Three warmups precede twenty paired rounds per condition. Old/new order rotates
each round and begins in the opposite order in the second independent survey.
Every completed pose, fitness, stopping field, iteration history and receipt
matches its local reference and the other executable's complete diagnostic
string. Each call downloads exactly 1024 equation bytes, with no added payload.

Other Cargo/compiler/linker/test/benchmark processes are monitored. Early whole
survey attempts overlapped unrelated benchmarks and were discarded in full.
The final controller waits for three quiet observations two seconds apart
before each condition and monitors its preparation, warmups and all twenty
paired rounds through completion. An overlapping condition is discarded in
full and retried; accepted records never drop individual samples. All eight
conditions in both final surveys passed on their first per-condition attempt
without detected overlap. The two processes remain alive between conditions;
the scratch pools clear at each condition and shader caches remain warmed.
No build, test or statistical analysis runs during accepted measurement.

Windows, Rust 1.98.1 release optimization, Vulkan and a discrete compute adapter
were used. Native primitive/diagnostic logs report twelve Rayon workers without
a local adapter model; CPU covariance staging uses four workers for 65,537 points.
Ratios are paired geometric means, above one favoring the final implementation.
Within-survey 95% log-bootstrap intervals use 10,000 resamples, seed
20261013+8*(survey-1)+condition_index and endpoints 249/9749. Intervals are not
adjusted for multiple comparisons or between-survey system variation.

| Survey | Sources | Profile | Covariance preparation | Old full ms | Reused full ms | Old/reused [95%] |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | 257 | lattice | CPU k-NN | 7.455 | 7.599 | 0.978x (0.931-1.024) |
| 1 | 4,097 | lattice | CPU k-NN | 19.548 | 19.350 | 0.998x (0.957-1.038) |
| 1 | 4,097 | duplicate/unequal | GPU k-NN | 16.369 | 16.126 | 1.030x (1.003-1.060) |
| 1 | 16,384 | lattice | GPU grid | 38.540 | 36.915 | 1.001x (0.958-1.040) |
| 1 | 65,537 | lattice | CPU k-NN | 191.804 | 83.890 | 2.242x (2.129-2.353) |
| 1 | 65,537 | lattice | GPU k-NN | 26.639 | 27.154 | 0.959x (0.896-1.025) |
| 1 | 65,537 | lattice | GPU radius | 38.909 | 37.248 | 1.059x (1.001-1.120) |
| 1 | 65,537 | duplicate/unequal | GPU grid | 28.738 | 30.933 | 0.964x (0.858-1.085) |
| 2 | 257 | lattice | CPU k-NN | 7.298 | 6.818 | 1.041x (0.958-1.130) |
| 2 | 4,097 | lattice | CPU k-NN | 14.854 | 15.101 | 0.986x (0.929-1.046) |
| 2 | 4,097 | duplicate/unequal | GPU k-NN | 7.250 | 7.346 | 0.972x (0.917-1.030) |
| 2 | 16,384 | lattice | GPU grid | 12.818 | 13.075 | 0.983x (0.899-1.076) |
| 2 | 65,537 | lattice | CPU k-NN | 187.426 | 92.602 | 2.090x (1.961-2.234) |
| 2 | 65,537 | lattice | GPU k-NN | 29.405 | 31.573 | 0.938x (0.864-1.019) |
| 2 | 65,537 | lattice | GPU radius | 41.188 | 43.038 | 0.926x (0.821-1.014) |
| 2 | 65,537 | duplicate/unequal | GPU grid | 29.195 | 28.347 | 1.102x (0.966-1.282) |

The **65,537-source CPU k-NN covariance path improves 2.242x/2.090x** in the two
surveys, with both intervals above one. Median completed calls fall
191.804→83.890 ms and 187.426→92.602 ms. This is actual three-update registration,
including preparation and transfers, rather than a shader-only timing claim.
Its output/history/receipts remain exact. It demonstrates a gain for this large
synthetic lattice on this host; other geometry/hardware/worker availability
remain unmeasured for this specific optimization.

Small CPU-preparation paths are inconclusive. GPU k-NN/radius/grid paths show
mixed directions and no repeated interval-above-one gain across both surveys.
Positive intervals for 4097 duplicate GPU k-NN and 65537 GPU radius in the first
survey do not repeat. Adverse central estimates are retained, including large
GPU k-NN/radius and first-survey duplicate/grid results. System variation is
substantial; no universal all-route registration gain is claimed.

## Resident and one-evaluation boundaries

The native primitive test compares the frozen old equations submission with
production reuse on independent runtimes. Ten conditions per survey use three
warmups and twenty rounds rotating old full/new full/old resident/new resident.
One-evaluation full calls include serial CPU target-tree construction, immutable
input/covariance validation/packing/upload, controls/scratch/output, completion
download/decode and owner drops. They do not call high-level CPU covariance
estimation, so their timings intentionally exclude the added preparation gain.
Prepared resident calls include submission and an explicit completion wait;
download/validation follows the timer. Every result/receipt is checked. Both
surveys passed alone with no detected overlap, in 16.13/21.51 seconds.

| Survey | Sources/targets | Profile/distance/count | Old full ms | Reused full ms | Old resident ms | Reused resident ms | Old/reused full (95%) | Old/reused resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 257/274 | lattice/0.25/257 | 1.235 | 1.259 | 0.365 | 0.381 | 0.991x (0.965-1.016) | 1.000x (0.890-1.126) |
| 1 | 4,097/4,114 | lattice/0.25/4,097 | 2.049 | 2.084 | 0.538 | 0.516 | 1.022x (0.982-1.073) | 1.029x (0.955-1.108) |
| 1 | 4,097/4,114 | duplicates/0.25/4,097 | 2.573 | 2.422 | 0.560 | 0.521 | 1.049x (0.980-1.130) | 1.060x (0.955-1.199) |
| 1 | 65,537/65,554 | lattice/0.25/65,537 | 23.029 | 23.271 | 2.341 | 2.263 | 1.005x (0.980-1.033) | 1.067x (1.015-1.160) |
| 1 | 65,537/65,554 | duplicates/0.25/65,537 | 25.169 | 25.580 | 2.564 | 2.497 | 1.003x (0.967-1.044) | 1.017x (1.003-1.032) |
| 1 | 65,537/1 | single/100/65,537 | 5.627 | 5.659 | 1.625 | 1.550 | 0.987x (0.964-1.013) | 1.027x (0.992-1.060) |
| 1 | 262,145/1 | single/0.001/0 | 18.044 | 18.420 | 2.965 | 2.909 | 0.974x (0.932-1.017) | 1.017x (0.999-1.035) |
| 1 | 262,145/1 | single/100/262,145 | 20.089 | 20.830 | 3.616 | 3.433 | 0.966x (0.924-1.005) | 1.051x (1.036-1.067) |
| 1 | 1,048,577/1 | single/0.001/0 | 74.518 | 75.376 | 14.154 | 13.349 | 0.995x (0.969-1.018) | 1.057x (1.051-1.063) |
| 1 | 1,048,577/1 | single/100/1,048,577 | 72.794 | 75.244 | 14.761 | 13.627 | 0.969x (0.949-0.987) | 1.090x (1.082-1.097) |
| 2 | 257/274 | lattice/0.25/257 | 1.915 | 1.883 | 0.481 | 0.509 | 1.058x (0.992-1.133) | 0.893x (0.785-1.004) |
| 2 | 4,097/4,114 | lattice/0.25/4,097 | 3.078 | 3.211 | 0.629 | 0.575 | 1.003x (0.937-1.075) | 1.014x (0.894-1.150) |
| 2 | 4,097/4,114 | duplicates/0.25/4,097 | 3.012 | 2.970 | 0.547 | 0.549 | 1.014x (0.945-1.089) | 0.954x (0.862-1.056) |
| 2 | 65,537/65,554 | lattice/0.25/65,537 | 35.094 | 36.832 | 2.328 | 2.210 | 0.984x (0.940-1.031) | 1.025x (0.934-1.123) |
| 2 | 65,537/65,554 | duplicates/0.25/65,537 | 34.641 | 34.999 | 2.128 | 2.114 | 1.018x (0.973-1.068) | 0.990x (0.881-1.116) |
| 2 | 65,537/1 | single/100/65,537 | 8.544 | 8.175 | 1.580 | 1.444 | 1.024x (0.954-1.096) | 1.099x (0.954-1.262) |
| 2 | 262,145/1 | single/0.001/0 | 27.657 | 28.648 | 5.465 | 4.950 | 0.984x (0.927-1.041) | 1.142x (1.032-1.280) |
| 2 | 262,145/1 | single/100/262,145 | 28.920 | 29.789 | 6.497 | 5.610 | 0.952x (0.876-1.031) | 1.070x (0.950-1.186) |
| 2 | 1,048,577/1 | single/0.001/0 | 105.550 | 108.342 | 18.215 | 17.397 | 0.978x (0.937-1.022) | 1.019x (0.911-1.132) |
| 2 | 1,048,577/1 | single/100/1,048,577 | 90.716 | 94.764 | 15.535 | 13.947 | 0.970x (0.949-0.993) | 1.069x (0.981-1.157) |

Large prepared calls have promising central ratios (up to 1.090x in the first
survey), but no resident condition has intervals above one in both surveys.
All full-evaluation intervals are inconclusive except the fully gated
1,048,577-source single-target case, which worsens 0.969x/0.970x in both surveys.
Its serial upload/packing/validation and drop overhead remain measured; the
working memory reduction is not a claim of universal primitive-call speedup.
Synthetic single-target reductions do not predict equally large target-tree
registration or CPU k-NN preparation.

## Earlier point-to-plane full-call diagnostic

The previous point-to-plane slice retained adverse full-call measurements.
A test-only diagnostic now instruments tree, upload/packing, submission,
download/wait/decode and owner-drop phases against its frozen old submission.
It preserves the original four-way full/resident rotation and live prepared
inputs, checking output and receipt after every completed call. Three source
sizes with a single target and a fully inclusive gate each retain twenty pairs
after three warmups. Both exact isolated diagnostic surveys pass in 3.46/3.10 s.

| Survey | Sources | Variant | Tree ms | Upload/packing ms | Submit ms | Download/wait ms | Drop ms | Full ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65,537 | old | 0.002 | 1.236 | 0.222 | 1.590 | 0.007 | 3.220 |
| 1 | 65,537 | reuse | 0.001 | 1.207 | 0.227 | 1.832 | 0.008 | 3.292 |
| 1 | 262,145 | old | 0.001 | 3.426 | 0.213 | 2.354 | 0.005 | 6.161 |
| 1 | 262,145 | reuse | 0.001 | 3.332 | 0.172 | 2.419 | 0.005 | 6.175 |
| 1 | 1,048,577 | old | 0.001 | 16.216 | 0.209 | 6.258 | 0.007 | 23.126 |
| 1 | 1,048,577 | reuse | 0.001 | 15.441 | 0.241 | 6.258 | 0.007 | 22.917 |
| 2 | 65,537 | old | 0.001 | 0.767 | 0.135 | 1.366 | 0.004 | 2.289 |
| 2 | 65,537 | reuse | 0.000 | 0.762 | 0.136 | 1.373 | 0.004 | 2.280 |
| 2 | 262,145 | old | 0.001 | 2.964 | 0.196 | 2.609 | 0.009 | 5.826 |
| 2 | 262,145 | reuse | 0.000 | 2.973 | 0.187 | 2.656 | 0.009 | 5.842 |
| 2 | 1,048,577 | old | 0.001 | 13.790 | 0.257 | 6.366 | 0.008 | 20.169 |
| 2 | 1,048,577 | reuse | 0.001 | 13.975 | 0.263 | 6.514 | 0.008 | 20.697 |

At roughly one million sources, packing/upload takes about 14–16 ms and
download/completion about 6 ms; tree/submission/drop costs are much smaller.
These CPU-visible phases account for each full sample (component medians need
not sum to the full median). The original large 65537-source slowdown does
not repeat at the same magnitude in these instrumented surveys: paired mean
new-minus-old differences are +0.071 ms and -0.011 ms. One-million-source
paired mean differences are -0.008/-0.410 ms despite mixed medians/outliers.
This does not invalidate earlier adverse results or prove a driver/allocator
cause. Device transfer and shader work overlap inside completion; the probe
does not separate those GPU costs. Initial full-only probes are retained
locally, but the published table uses the original mixed call schedule.
Production point-to-plane code is unchanged.

## Verification and source/evidence paths

- Native affected GICP kernels/covariance/status tests: **21 passed, one ignored**
  (the corrected conditioning oracle was rerun exactly after the initial fixture
  expectation was found to improve under rotation). Registration library GICP
  subset: **five passed, two ignored**. The four explicit release integration
  targets passed **16 test executions / ten distinct tests**, with repeated
  imported legacy references counted only once in the accuracy totals.
- GPU all-feature/all-target and optional/CPU-only registration Clippy,
  CPU-only GPU, GPU WASM, minimal facade and WASM facade builds pass.
  Final workspace formatting is checked before delivery. Tests/math tolerances,
  core/math/search crates and dependencies retain their contracts.
- Production GPU host/shader:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\gicp_reduction.rs`
  and `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\shaders\gicp_reduction.wgsl`.
- GPU references/measurement:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\gicp_reuse_query_tests.rs`
  and `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\gicp_reuse_query_benchmark.rs`.
- CPU preparation/reference:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-registration\src\gicp_covariance_preparation.rs`
  and `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-registration\src\gicp_covariance_preparation_tests.rs`.
- Public API paired test:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-registration\tests\gpu_gicp_reuse_query_benchmark.rs`.
- Point-to-plane diagnostic:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\point_to_plane_reuse_query_phase_benchmark.rs`.
- Ignored original binary/provenance:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\spatialrust_gicp_reuse_previous.exe`
  and `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-reuse-transformed-queries-baseline.json`.
- Accepted raw records, bootstrap results and exact logs are in
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-reuse-registration-measurements.json`,
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-reuse-transformed-queries-measurements.json`
  and `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-plane-reuse-full-phase-diagnostic-measurements.json`.
  Their family-prefixed benchmark-1/2 files retain the original sample arrays;
  early memory-only/overlapping controller attempts remain separate and are
  excluded from the final public-call table. Required repository CI gates merge.
