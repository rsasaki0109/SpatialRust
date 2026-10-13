# Resident GPU ICP trimming

Date: 2026-10-13. Named slice: stable device distance/index selection for
point-to-point ICP, retaining the explicit host pose-solve boundary.

## API and numerical contract

`IcpRegistration::align_with_gpu_trimmed_reduction(runtime, source, target)`
under `register-icp-gpu` honors the existing `with_trim_fraction(f64)` setting.
`GpuIcpInput::trimmed_moments_resident(runtime, absolute_pose, max_distance,
fraction)` under `gpu-wgpu` provides the low-level asynchronous owned result.
Both APIs explicitly retain original source/target/tree buffers. GPU computes
absolute transforms, bounded nearest matching, stable trimming, centroids,
centered cross-covariance and all-gated squared error. CPU builds the target
tree, solves the existing Horn quaternion eigensystem, composes the pose and
applies convergence criteria. There is no CPU selection/fallback or intermediate
point/match readback. Ordinary CPU and correspondence APIs retain their math.

For N gated matches, K is `floor(f64(N)*trim_fraction)` with the binary64
product rounded to nearest-even before floor, exactly as the CPU configuration
contract specifies. The fraction must be finite and in (0,1]. GPU integer limbs
multiply the u32 count by the exact 53-bit significand, round the resulting
96-bit representation to 53 significant bits and shift/floor it. Subnormal
fractions retain zero points. Casting the fraction to f32 or using a quantized
histogram cutoff would change K and is avoided.

Selection order is ascending GPU squared-distance word, then original source
index; signed zero distances share key zero. Retaining the first K matches
preserves source input order on exact device-distance ties. A most-significant
digit histogram narrows the Kth pair's prefix, without permuting matches or
performing CPU sorting. Four distance bytes and only the significant index
bytes are processed. Each digit has a histogram pass and a prefix-selection
pass. Up to 256 groups of 64 threads use private 256-bin histograms; the final
64-thread group combines them. The exact threshold selects the retained set
for the canonical compensated moment arithmetic.

Retained matches contribute to centroids/covariance and the next pose solve.
ALL gated matches contribute to post-update fitness and its denominator;
fitness stopping does not become easier by dropping residuals. The trace's
`correspondences` reports retained update support, while
`evaluated_correspondences` reports all post-update gated matches. Insufficient
gated or retained support reports the corresponding existing error. A fraction
of one delegates to ordinary resident kernels and preserves their complete
diagnostics and transfer receipts. `align_with_gpu_reduction` retains its
existing fraction-one requirement and error contract.

Selection and reduction share ONE queue submission. Input upload remains
`16*S+32*T+32*M` bytes for source/target counts S/T and M target tree nodes;
uniform controls are excluded. Each nonempty evaluation explicitly downloads
176 bytes, with I+1 evaluations for I updates. There are no intermediate host
waits or repeated payload uploads. Empty source low-level calls transfer zero
readback bytes. The wrapper validates nonempty clouds as before.

`GpuIcpTrimmedMoments` owns its output outside the scratch pool and exposes
`buffer`, `normalization_scale`, `squared_distance_scale`, `readback_bytes` and
explicit `download`. Its layout is the canonical 176 bytes: status, retained
count, ALL-gated count, reserved zero; two vec4 anchors and sixteen compensated
vec2 terms. Terms 0..14 use retained support, and term 15 uses all gated support.
Fraction one uses canonical reserved zeros, with the decoder supplying the
ordinary count as the all-gated denominator. Zero retained support zeros
anchors/centroids/covariance while preserving all-gated residual error. Invalid
status or zero gated support zeros all payload. Status bits 1/2/4 retain their
transform-overflow/malformed-match/nonfinite-moment meanings.

`GpuIcpTrimmedStatistics` contains the retained-pair `moments` and separate
`evaluated_correspondences`. Its nested squared-error sum must be divided by
the separate all-gated count. Download validates runtime ownership, status,
retained<=gated<=source count, reserved words, zero-support invariants and finite
payload before decoding physical f64 values. Result buffers outlive inputs;
callers must inspect status before using device storage and keep it immutable.

The trim route requires seven storage bindings, eight total bindings, and
2048 shared bytes, in addition to canonical dispatch/uniform/geometry limits.
Histogram scratch is `1024*groups` bytes (maximum 262,144), plus 32-byte
selection state; transformed/match/partial scratch remains canonical.
Selection state is explicitly cleared and every consumed histogram word is
overwritten. Scratch stays exclusively leased through submission before
recycling, including concurrent shared-runtime calls. Unsupported limits or
nonfinite arithmetic return errors without choosing another backend.

f32 device transforms, products and distances still have their existing
precision limits. CPU/GPU distance rounding can alter ordering near ties;
different retained subsets can alter centroids, pose and stopping. Matching K
does not promise the same subset or bit-exact CPU moments. Independent f64
moment oracles use verified GPU distance words for ordering on nonbinary
coordinate scales; full registration also compares against ordinary CPU ICP.

## Validation

Seven new native primitive tests cover:

- 443,827 exact retained-count comparisons across 31 fraction encodings and
  14,317 counts, including binary64 neighbors of common fractions, subnormal
  fractions, integer/power-of-two boundaries and u32 maximum values.
- 135 independent comparison-sort selection cases: sizes 1/3/63/64/65/257/
  4097/16385/65537, five fractions, and dispersed/all-tied/signed-zero/boundary/
  subnormal distance keys with missing matches. Every selected threshold/set
  and gated/retained count matches the reference.
- Independent f64 selected means/cross-covariance and all-gated error at six
  sizes, three scales and three fractions (54 cases). Known scalar target
  membership is verified before comparison sorting device distance words.
- Exact all-gated error retention while K varies, including zero retained
  support and fraction-one equality; inclusive distance boundaries; plan
  limits; malformed readbacks; transform overflow zeroing; empty gates/inputs;
  result ownership after input drop; runtime mismatches; four concurrent callers.

The native full-call suite checks 486 conditions (369,603 source points):
six sizes 17/63/64/65/257/4097, three unique/partially gated/duplicate profiles,
three scales 1e-4/1/1e4, three fractions 0.5/0.8/1, and three stopping modes.
CPU/GPU update/evaluation counts, stop reason and convergence agree. Independent
all-gated final residual fitness agrees within the stated f32 tolerance.
Maximum scale-normalized CPU pose-coordinate difference was 0.0000019073486,
below the 3e-5 test limit. Three additional rotated initial-guess cases exercise
absolute original-position reuse. Inputs, complete receipts and fraction-one
diagnostics are preserved. Invalid configuration, insufficient gated/retained
support and concurrent full calls are checked separately.

Existing untrimmed native tests also passed: 54 pose/scale/unequal-cloud cases,
exact concurrent diagnostics, validation/error receipts and convergence.
The public facade checks retained/all-gated counts and fixed readback size.
All-feature registration library tests passed (35 passed, three intentional
ignored tests); Clippy all targets/features, CPU-only ICP, WASM optional ICP
and formatting checks passed. No dependency, unsafe boundary or core model is
added. The canonical untrimmed WGSL file, nearest WGSL file, ordinary ICP loop
and Horn solver remain unchanged.

Source references:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/icp_trim.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/src/icp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_icp_trimmed_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_icp_trimmed_reduction_benchmark.rs`

## Completed-call measurements and precision limit

Two sequential isolated release surveys completed on the same native Vulkan /
DiscreteGpu runtime. No local build, test, statistical analysis or other local
compute work overlapped a survey. Compilation was completed separately. Each
condition has three warmups per stage and 20 measured rounds. The stage order
rotates CPU ICP, previous GPU correspondences with CPU trimming/moments, new
resident trimmed ICP, and a completed prepared resident evaluation. Every warm
and timed result equals its own precision path's complete control result.
Source sizes are 257/4097/65537/262145, with N+17 target points and three updates
with stopping thresholds disabled. U is unique geometry with trim 0.5; G adds
partially gated/next-layer matches and trim 0.8; D includes duplicates and trim
0.9. All profiles include repeated inlier/outlier residual levels.

Full calls include XYZ extraction, source packing, serial target-tree
construction, upload, all evaluations, explicit waits/readback, CPU Horn
solving, composition, history and final fitness. The prepared evaluation
includes host command/buffer preparation, transformation/search/selection/
reduction submission and GPU completion wait; it EXCLUDES input packing/tree
construction/upload, fixed readback, pose solve and registration looping.
It is a completed prepared-operation wall time, not a GPU timestamp or a
full-registration speedup. Uniform control bytes remain excluded from receipts.

The table reports stage median milliseconds. Ratios are geometric means of
paired baseline/new times, with 10,000 paired-log bootstrap draws, sorted
entries 249/9749, seed `20261013+2*record_index+baseline_index` (CPU 0, previous
GPU 1). Intervals describe paired rounds within one run, not other devices,
drivers or environmental drift; the repeated surveys retain their separate
ranges. The tables do not discard any measured condition.

| Survey | Source/target | Fixture | CPU ms | Previous GPU ms | New GPU ms | Prepared evaluation ms | Max new / legacy pose-coordinate error | Pose within 3e-5 | CPU/new ratio (95% interval) | Previous/new ratio (95% interval) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- |
| 1 | 257/274 | U/0.5 | 0.270 | 6.491 | 4.459 | 0.459 | 5.96e-08/0 | yes | 0.06x (0.06-0.06) | 1.47x (1.43-1.50) |
| 1 | 257/274 | G/0.8 | 0.270 | 6.547 | 4.439 | 0.499 | 2.38e-07/0 | yes | 0.06x (0.06-0.06) | 1.49x (1.46-1.52) |
| 1 | 257/274 | D/0.9 | 0.326 | 6.532 | 4.421 | 0.494 | 1.19e-07/0 | yes | 0.07x (0.07-0.07) | 1.47x (1.41-1.53) |
| 1 | 4,097/4,114 | U/0.5 | 5.647 | 9.037 | 5.348 | 0.682 | 0/0 | yes | 1.05x (1.02-1.08) | 1.69x (1.65-1.73) |
| 1 | 4,097/4,114 | G/0.8 | 6.716 | 9.192 | 5.359 | 0.757 | 9.54e-07/0 | yes | 1.27x (1.22-1.32) | 1.69x (1.64-1.75) |
| 1 | 4,097/4,114 | D/0.9 | 6.960 | 11.034 | 6.530 | 1.298 | 9.54e-07/0 | yes | 1.06x (1.02-1.10) | 1.64x (1.59-1.69) |
| 1 | 65,537/65,554 | U/0.5 | 130.350 | 72.595 | 50.208 | 8.578 | 1.98e-06/0 | yes | 2.61x (2.54-2.69) | 1.42x (1.40-1.44) |
| 1 | 65,537/65,554 | G/0.8 | 147.968 | 75.154 | 57.030 | 10.529 | 5.25e-06/0 | yes | 2.63x (2.57-2.70) | 1.36x (1.32-1.42) |
| 1 | 65,537/65,554 | D/0.9 | 163.664 | 73.972 | 55.959 | 9.804 | 9.54e-07/0 | yes | 2.99x (2.93-3.06) | 1.34x (1.32-1.35) |
| 1 | 262,145/262,162 | U/0.5 | 570.988 | 236.647 | 143.786 | 19.344 | 7.15e-07/0 | yes | 4.19x (3.87-4.56) | 1.67x (1.60-1.75) |
| 1 | 262,145/262,162 | G/0.8 | 623.784 | 211.499 | 121.726 | 14.520 | 0.000815/7.63e-06 | NO | 5.01x (4.52-5.58) | 1.77x (1.63-1.93) |
| 1 | 262,145/262,162 | D/0.9 | 677.595 | 243.593 | 150.065 | 14.175 | 5.72e-06/0 | yes | 4.92x (4.50-5.38) | 1.73x (1.62-1.86) |
| 2 | 257/274 | U/0.5 | 0.273 | 6.613 | 4.472 | 0.470 | 5.96e-08/0 | yes | 0.06x (0.06-0.06) | 1.47x (1.44-1.50) |
| 2 | 257/274 | G/0.8 | 0.278 | 6.702 | 4.589 | 0.504 | 2.38e-07/0 | yes | 0.06x (0.06-0.06) | 1.45x (1.43-1.48) |
| 2 | 257/274 | D/0.9 | 0.335 | 6.666 | 4.598 | 0.499 | 1.19e-07/0 | yes | 0.07x (0.07-0.07) | 1.43x (1.37-1.47) |
| 2 | 4,097/4,114 | U/0.5 | 5.759 | 9.106 | 5.377 | 0.679 | 0/0 | yes | 1.06x (1.01-1.10) | 1.69x (1.64-1.73) |
| 2 | 4,097/4,114 | G/0.8 | 6.826 | 8.571 | 5.106 | 0.709 | 9.54e-07/0 | yes | 1.30x (1.25-1.35) | 1.66x (1.60-1.72) |
| 2 | 4,097/4,114 | D/0.9 | 7.030 | 8.921 | 5.413 | 0.728 | 9.54e-07/0 | yes | 1.30x (1.25-1.34) | 1.64x (1.60-1.69) |
| 2 | 65,537/65,554 | U/0.5 | 132.530 | 68.646 | 47.349 | 7.986 | 1.98e-06/0 | yes | 2.99x (2.82-3.19) | 1.53x (1.45-1.62) |
| 2 | 65,537/65,554 | G/0.8 | 166.110 | 72.527 | 52.598 | 9.054 | 5.25e-06/0 | yes | 3.32x (3.10-3.61) | 1.44x (1.38-1.50) |
| 2 | 65,537/65,554 | D/0.9 | 180.553 | 78.617 | 57.076 | 10.136 | 9.54e-07/0 | yes | 3.12x (2.97-3.26) | 1.34x (1.30-1.38) |
| 2 | 262,145/262,162 | U/0.5 | 618.907 | 250.262 | 151.087 | 19.347 | 7.15e-07/0 | yes | 4.50x (4.17-4.86) | 1.77x (1.67-1.88) |
| 2 | 262,145/262,162 | G/0.8 | 650.866 | 235.857 | 134.637 | 14.365 | 0.000815/7.63e-06 | NO | 5.06x (4.62-5.56) | 1.77x (1.65-1.90) |
| 2 | 262,145/262,162 | D/0.9 | 804.375 | 249.540 | 162.823 | 22.086 | 5.72e-06/0 | yes | 5.36x (4.99-5.75) | 1.70x (1.60-1.81) |

65,537-point full calls improve 2.61-3.32x over CPU and 1.34-1.53x over previous
GPU, with all within-run intervals above one. The pose-qualified unique/
duplicate 262,145-point calls improve 4.19-5.36x over CPU and 1.67-1.77x over
previous GPU; their maximum CPU pose-coordinate difference is 0.000005722046.
4,097-point CPU/new ratios are only 1.05-1.30x (lower intervals near one), while
257-point CPU/new ratios are 0.06-0.07x: CPU is substantially faster there.
The route remains explicitly selected and does not infer placement from size.

The large G/0.8 case does NOT meet the 3e-5 CPU pose-agreement tolerance in
either survey. Its maximum coordinate difference is 0.0008154977 versus CPU;
previous GPU/CPU difference is 0.0000076293945. CPU / previous GPU / new GPU
final fitness is 0.004841253771790894 / 0.004841253990255459 /
0.0048409760054502335. Both runs retain 209,642 of 262,053 gated pairs and agree
on counts, iteration limit and nonconvergence; independent all-gated residual
fitness checks pass. GPU f32 transforms/distances/moments can change subsets
near the repeated residual cutoff and propagate into pose. These measurements
do not isolate which stage accounts for the difference. The case is preserved
with `within_pose_tolerance=false` and excluded from pose-qualified speedup
ranges above; its timing is NOT evidence of CPU-equivalent accuracy. Further
selected-set and precision investigation is needed for this geometry/regime.
The tolerance is not increased to make that condition pass.

An initial probe stopped on CPU/previous-GPU complete-diagnostics equality;
a second stopped on the large G case's pose tolerance. Both diagnostic logs
are retained separately and are not inputs to the successful-survey bootstrap.
The final benchmark retains the original tolerance and records the explicit
adverse-case flag. Production kernels did not change between those probes
and the final surveys.

Payload receipts for three updates (four resident evaluations vs seven
previous correspondence batches):

| Source/target | Fixture | Previous H2D | New H2D | Previous D2H | New D2H |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 257/274 | U/0.5 | 35,184 | 14,896 | 14,392 | 704 |
| 257/274 | G/0.8 | 35,184 | 14,896 | 14,392 | 704 |
| 257/274 | D/0.9 | 35,184 | 14,896 | 14,392 | 704 |
| 4,097/4,114 | U/0.5 | 542,192 | 214,704 | 229,432 | 704 |
| 4,097/4,114 | G/0.8 | 542,192 | 214,704 | 229,432 | 704 |
| 4,097/4,114 | D/0.9 | 542,192 | 214,704 | 229,432 | 704 |
| 65,537/65,554 | U/0.5 | 8,652,272 | 3,409,584 | 3,670,072 | 704 |
| 65,537/65,554 | G/0.8 | 8,652,272 | 3,409,584 | 3,670,072 | 704 |
| 65,537/65,554 | D/0.9 | 8,652,272 | 3,409,584 | 3,670,072 | 704 |
| 262,145/262,162 | U/0.5 | 34,604,528 | 13,633,200 | 14,680,120 | 704 |
| 262,145/262,162 | G/0.8 | 34,604,528 | 13,633,200 | 14,680,120 | 704 |
| 262,145/262,162 | D/0.9 | 34,604,528 | 13,633,200 | 14,680,120 | 704 |

At 262,145 points the D2H payload falls from 14,680,120 to 704 bytes; H2D falls
from 34,604,528 to 13,633,200 bytes. These savings include the whole call, not
only a selected pass. Host target-tree construction, pose solving and stopping
remain, and the adverse precision case remains visible.

Reproduction (PowerShell, isolated exact benchmark; compile separately first):

```powershell
Set-Location C:/Users/rsasa/Workspace/SpatialRust
$env:PATH='C:/Users/rsasa/.cargo/bin;'+$env:PATH
cargo test -p spatialrust-registration --no-default-features --features register-icp-gpu --release --test gpu_icp_trimmed_reduction_benchmark --no-run
cargo test -p spatialrust-registration --no-default-features --features register-icp-gpu --release --test gpu_icp_trimmed_reduction_benchmark completed_resident_trimmed_icp_calls -- --include-ignored --exact --nocapture --test-threads=1
```

Raw successful surveys:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-resident-trimming-benchmark-1.log`
and `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-resident-trimming-benchmark-2.log`.
Parsed samples/medians/intervals:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-resident-trimming-measurements.json`.
Analysis:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-icp-resident-trimming.py`.
These local artifacts contain backend/device class and measurement conditions,
without an adapter model name.
