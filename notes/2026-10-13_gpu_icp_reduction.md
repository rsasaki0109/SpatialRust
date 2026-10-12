# Resident ICP transforms and moment reduction

Date: 2026-10-13. Named slice: device source transforms, bounded correspondence
search and centered ICP moments, with an explicit CPU pose-solve boundary.

## Execution and transfer contract

`register-icp-gpu` adds
`IcpRegistration::align_with_gpu_reduction(runtime, source, target)` alongside
the existing CPU and GPU-correspondence APIs. Original source and target points
stay on GPU throughout the call. Coordinate transforms, bounded nearest search,
centroid, centered cross-covariance and squared-error reduction run on device.
CPU retains the small Horn quaternion eigensolve, transform composition, trace
construction and convergence checks. CPU target-tree construction remains.
No point/query/match readback, repeated point upload or CPU fallback occurs.

The new entry point requires `trim_fraction=1`. The existing
`align_with_gpu_correspondences` still supports trimming and retains its
ordered CPU sums. Shared validation preserves existing configuration/input
error order. The ordinary CPU loop and both existing numerical solvers keep
their arithmetic. This slice does not move GICP/point-to-plane Hessians,
trimming or the small pose solve to GPU.

`GpuIcpInput::upload` under `gpu-wgpu` explicitly retains immutable original
source XYZ, original-order target XYZ and the uploaded target KD-tree. For S
source points, T target points and M tree nodes, H2D payload is
`16*S + 32*T + 32*M` bytes: source 16*S, original target 16*T, tree-ordered
target 16*T and nodes 32*M. Uniform controls are excluded. A nonempty target is
required. Empty source evaluations submit no work/readback, but target upload
still has its ordinary payload.

`moments_resident(runtime, absolute_pose, max_distance)` applies the absolute
pose to ORIGINAL source points on each evaluation. It validates runtime,
finite unit-quaternion pose, positive finite nonzero f32 distance square,
counts, host/storage/uniform/shared-memory and dispatch limits. Both point sets
need finite XYZ/global extent squares. The maximum of their extent squares
defines normalization: a positive value must be normal in f32, while coincident
sets use scale 1. This is stricter than ordinary CPU ICP.

Six ordered device passes share ONE submission:

1. Transform original source into private query scratch; detect overflow.
2. Run the unchanged bounded nearest shader/traversal/tie rules into private
   match scratch. The distance gate is inclusive.
3. Accumulate centroid offsets and normalized residual squared lengths in
   at most 256 partial groups with 64 threads and compensated scalar sums.
4. Combine partial centroids/fitness and exact u32 counts.
5. Accumulate cross-covariance about those device centroids.
6. Combine covariance and finalize validity, zeroing ALL payload on errors.

Bounding-box midpoint anchors and a common diagonal scale avoid summing large
absolute coordinates. Actual GPU-transformed source anchors are retained in
the result, so explicit decoding does not substitute a CPU transform's rounding.
Centered covariance is a second pass over the matched points, rather than
subtracting two large uncentered moments. Normalized residuals use the maximum
correspondence distance as their scale. Compensated hi/lo f32 pairs survive
partial combination and are explicitly decoded to f64 physical units on CPU.
Products, division and transforms remain f32; compensation does not supply
f64 arithmetic or guarantee accuracy on all conditioning regimes.

Private scratch is 16*S transformed bytes, 8*S match bytes and 144 bytes per
partial group (at most 36,864). Consumed scratch words are initialized before
use, remain exclusively leased through submission and then return to the
existing bounded pool. Pipelines are lazy per runtime. Reduction uses six
storage bindings, a 112-byte uniform, 64 threads and 768 shared bytes per
workgroup. The existing nearest pipeline is reused without shader changes.

`GpuIcpMoments` owns a 176-byte storage buffer independently of the input and
pool. Its first four u32 words are status, correspondence count and two reserved
zeros. Status bits 1/2/4 indicate transformed-coordinate overflow, malformed
match rows and nonfinite moments; any nonzero status invalidates the result.
The following two vec4<f32> records contain source/target anchors, then sixteen
vec2<f32> records contain compensated normalized moments. Terms 0..5 are source/
target centroid offsets, 6..14 centered cross-covariance in row-major order,
and 15 residual squared-error SUM. `normalization_scale` and
`squared_distance_scale` expose the required physical scales to GPU consumers.
Invalid or zero-match results have all-zero payload with status retained.
Consumers must inspect status and treat exposed storage as immutable.

`download` copies only 176 D2H bytes for a nonempty source, waits, checks status,
count/reserved words, finite payload and zero-match invariants, then returns
`GpuIcpStatistics` with f64 physical centroids/cross-covariance/error sum and a
transfer receipt. Empty source readback transfers zero bytes. There is no
intermediate transfer or host completion wait in `moments_resident`.

The registration wrapper evaluates the initial pose once. Every update uses
the current moments for its CPU solve and evaluates the updated absolute pose
once. Post-update moments serve fitness, the next update and final fitness.
I updates therefore need I+1 evaluations and `176*(I+1)` D2H bytes. The existing
correspondence path uses 2*I+1 query/readback batches, including its final fitness
pass. Reuse reduces duplicate matching as well as transfer volume; timing gains
are not attributed solely to the reduction kernel.

## Precision and correctness evidence

This is an opt-in f32 device precision path, not a CPU pose/fitness/trace parity
promise. Near match gates, ties, degenerate geometry or stopping thresholds,
transforms and reductions can change matches, updates, fitness and stopping.
Collinear/coincident points do not certify an observable pose. Existing CPU
and GPU-correspondence routes retain their f64 sums and remain available.

Four primitive tests cover counts 1/3/63/64/65/257/4097/16385. Independent f64
centroid, all nine centered covariance entries and squared-error sums are
checked. The selected covariance bound is
`1e-7 + 1e-6*sum(abs(centered_source_component*centered_target_component))`.
This handles zero/near-zero cross entries and is not a universal error guarantee.
Other checks cover exact inclusive gate boundaries, zero matches, empty source,
invalid limits/coordinates/extent/radius/quaternion, transform overflow status
and complete payload zeroing. Translation fixtures at origins 0 and 1e6,
four concurrent callers with scratch reuse, retained outputs after input drop
and wrong-runtime rejection pass.

Three explicitly run adapter integration tests compare 54 combinations of
17/63/64/65/257/4097 points, volume/plane/unequal-cloud geometry and scales
1e-4/1/1e4. Poses are compared by transforming all 41,067 input points, with
maximum scale-normalized CPU coordinate difference 0.0000014305115 under the
selected bound 2e-5*scale. All selected trace correspondence counts and four
updates agree. Fitness is checked separately against CPU nearest/residual math
at the returned pose; receipts verify five fixed-size readbacks and one input
upload. Inputs remain unchanged. Tests also check unsupported trimming, shared
validation errors, insufficient matches, one-update convergence and concurrent
complete calls.

GPU all-features regression: 211 passed / 23 ignored. Registration library:
35 passed / 1 ignored. Three existing GPU-correspondence adapter tests and one
new facade adapter test pass. All-feature GPU/registration and CPU-only ICP
Clippy, minimal GPU build, WASM feature compilation and formatting pass.
Source comparison against `57afc8958172fa5e1c3059d5cd355a90cc586397` verifies
unchanged CPU centroid/covariance/Horn/eigensolver arithmetic, validation order
and update/fitness/trimming loops after extracting shared helpers. Native
Vulkan execution is verified; WASM evidence is compilation only. Hosted CI
does not replace native adapter evidence. No core/math/search dependency or
unsafe code is added.

## Completed-call measurements

Two isolated release surveys on native Vulkan / discrete-device backend use
three warmups and 20 paired rounds, rotating CPU/previous-GPU/new-GPU/resident
stage order. No other local builds, tests or statistical analysis run during
timing. Every sample and condition is retained. Runtime/pipeline setup is warmed
outside timers; repeated outputs are checked against their separate references.

All complete APIs use FOUR updates with stopping thresholds disabled, the
same initial guess and distance gate 0.25. Full CPU and previous-GPU timings
include their ordinary tree construction, updates, matching, fitness and
trace. New GPU timings include CPU tree/input preparation, one explicit input
upload, five complete device evaluations/readbacks, CPU pose solves and trace.
Verification and returned-output destruction are outside timers; internal
preparation/staging destruction is included. Different schedules (nine versus
five evaluations) and precision paths are part of the complete-call comparison.

V is a 32x32-column volume grid with spacing 0.5; P is a 128-column plane
without coordinate repetition; D duplicates consecutive V points and has 17
extra target rows. Source is translated by (0.03125,-0.015625,0.0078125), with
initial guess (-0.001,0.002,0). Before timing, pose coordinates are compared
with CPU; raw coordinate errors, fitness, counts and transfer bytes accompany
measurements. These synthetic fixtures do not establish a universal crossover.

Resident timings measure one prepared final-pose evaluation, including owned
output allocation, scratch leasing, transform/search/reductions, submission
and `wait_idle`. They exclude upload, tree construction, moment readback,
CPU solving and the iterative registration loop. These are not GPU timestamps
or whole-registration timings.

Times are medians in milliseconds. Ratios are geometric means of 20 paired
baseline/new-GPU timings. Percentile 95% intervals use 10,000 paired-log
bootstrap resamples, sorted entries 249/9749. Seeds are 20261013 + 2*zero-based
record index + baseline index (CPU=0, previous GPU=1).

| Survey | Source/target | Fixture | CPU ms | Previous GPU ms | New GPU ms | Resident ms | Max pose-coordinate error | CPU/new ratio (95% interval) | Previous/new ratio (95% interval) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 257/257 | V | 0.365 | 10.134 | 6.105 | 0.255 | 2.38e-07 | 0.06x (0.06-0.06) | 1.68x (1.65-1.71) |
| 1 | 257/257 | P | 0.355 | 11.819 | 6.946 | 0.296 | 9.54e-07 | 0.05x (0.05-0.06) | 1.67x (1.60-1.74) |
| 1 | 257/274 | D | 0.638 | 13.039 | 7.886 | 0.373 | 2.38e-07 | 0.08x (0.07-0.08) | 1.66x (1.61-1.71) |
| 1 | 4,097/4,097 | V | 13.019 | 15.215 | 8.525 | 0.448 | 2.38e-07 | 1.52x (1.39-1.67) | 1.86x (1.79-1.93) |
| 1 | 4,097/4,097 | P | 8.299 | 17.004 | 8.316 | 0.778 | 9.54e-07 | 1.00x (0.90-1.11) | 2.02x (1.93-2.11) |
| 1 | 4,097/4,114 | D | 10.053 | 16.615 | 8.134 | 0.947 | 2.38e-07 | 1.17x (1.07-1.26) | 2.00x (1.88-2.12) |
| 1 | 16,384/16,384 | V | 54.937 | 34.945 | 18.671 | 2.336 | 2.38e-07 | 3.04x (2.82-3.27) | 1.90x (1.85-1.96) |
| 1 | 16,384/16,384 | P | 39.513 | 32.346 | 16.764 | 2.052 | 9.54e-07 | 2.29x (2.08-2.51) | 1.91x (1.84-1.98) |
| 1 | 16,384/16,401 | D | 49.409 | 34.491 | 18.183 | 2.380 | 2.38e-07 | 2.79x (2.57-3.04) | 1.91x (1.87-1.95) |
| 1 | 65,537/65,537 | V | 234.455 | 87.491 | 52.197 | 6.960 | 2.38e-07 | 5.01x (4.37-5.88) | 1.84x (1.71-2.00) |
| 1 | 65,537/65,537 | P | 222.101 | 84.039 | 29.773 | 1.280 | 1.91e-06 | 6.40x (5.65-7.20) | 2.49x (2.27-2.73) |
| 1 | 65,537/65,554 | D | 570.734 | 171.998 | 78.240 | 4.182 | 2.38e-07 | 7.14x (6.55-7.80) | 2.31x (2.12-2.55) |
| 2 | 257/257 | V | 0.709 | 18.968 | 11.550 | 0.429 | 2.38e-07 | 0.06x (0.05-0.06) | 1.61x (1.55-1.68) |
| 2 | 257/257 | P | 0.609 | 13.687 | 8.251 | 0.341 | 9.54e-07 | 0.07x (0.06-0.07) | 1.65x (1.59-1.69) |
| 2 | 257/274 | D | 0.605 | 12.946 | 7.665 | 0.324 | 2.38e-07 | 0.05x (0.03-0.06) | 1.82x (1.67-2.00) |
| 2 | 4,097/4,097 | V | 23.983 | 55.101 | 29.400 | 2.544 | 2.38e-07 | 0.77x (0.69-0.84) | 1.81x (1.63-1.96) |
| 2 | 4,097/4,097 | P | 23.594 | 63.538 | 34.077 | 2.932 | 9.54e-07 | 0.70x (0.65-0.75) | 1.89x (1.73-2.05) |
| 2 | 4,097/4,114 | D | 25.240 | 60.154 | 31.701 | 2.424 | 2.38e-07 | 0.76x (0.67-0.85) | 1.82x (1.68-1.98) |
| 2 | 16,384/16,384 | V | 134.104 | 71.462 | 34.411 | 1.958 | 2.38e-07 | 3.68x (3.36-4.01) | 2.00x (1.81-2.21) |
| 2 | 16,384/16,384 | P | 95.460 | 83.798 | 38.794 | 3.688 | 9.54e-07 | 2.60x (2.40-2.83) | 2.12x (1.99-2.29) |
| 2 | 16,384/16,401 | D | 111.749 | 60.845 | 29.861 | 1.097 | 2.38e-07 | 3.71x (3.28-4.15) | 1.94x (1.71-2.20) |
| 2 | 65,537/65,537 | V | 274.733 | 86.622 | 37.928 | 2.060 | 2.38e-07 | 6.76x (5.87-7.68) | 2.22x (2.08-2.37) |
| 2 | 65,537/65,537 | P | 206.214 | 81.448 | 31.785 | 1.226 | 1.91e-06 | 5.59x (4.52-6.64) | 2.21x (1.87-2.54) |
| 2 | 65,537/65,554 | D | 295.489 | 87.686 | 45.119 | 3.082 | 2.38e-07 | 6.45x (5.09-7.95) | 1.99x (1.64-2.36) |

The new completed route improves over the previous GPU-correspondence route
in all 24 selected conditions: paired ratios 1.61-2.49x, with all reported
intervals above 1. Against CPU, 257-point calls are slower (0.05-0.08x);
4,097-point conditions vary between 0.70-1.52x across surveys, so no stable
CPU crossover is claimed there. At 16,384 points ratios are 2.29-3.71x;
at 65,537 points 5.01-7.14x. The largest selected benchmark CPU pose-coordinate
difference is 0.0000019073486, all final match counts equal S, and all APIs
perform four updates. Raw CPU/GPU fitness values remain in the local JSON;
small positive residuals can differ despite close poses. Substantial timing
variation is retained, including small-call regressions and the second survey's
4,097-point CPU advantage. This supports an explicit option, without changing
ordinary placement/defaults or promising arbitrary-input accuracy/performance.

Transfer payloads below are bytes for complete four-update calls, identical
across surveys. Uniform controls are excluded. At 65,537 source points D2H
falls from 4,718,664 to 880 bytes (about 99.98% lower).

| Source/target | Fixture | Previous H2D | New H2D | Previous D2H | New D2H |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 257/257 | V | 42,176 | 13,392 | 18,504 | 880 |
| 257/257 | P | 42,176 | 13,392 | 18,504 | 880 |
| 257/274 | D | 43,408 | 14,896 | 18,504 | 880 |
| 4,097/4,097 | V | 671,936 | 213,072 | 294,984 | 880 |
| 4,097/4,097 | P | 671,936 | 213,072 | 294,984 | 880 |
| 4,097/4,114 | D | 673,296 | 214,704 | 294,984 | 880 |
| 16,384/16,384 | V | 2,686,944 | 851,936 | 1,179,648 | 880 |
| 16,384/16,384 | P | 2,686,944 | 851,936 | 1,179,648 | 880 |
| 16,384/16,401 | D | 2,688,304 | 853,568 | 1,179,648 | 880 |
| 65,537/65,537 | V | 10,748,096 | 3,407,952 | 4,718,664 | 880 |
| 65,537/65,537 | P | 10,748,096 | 3,407,952 | 4,718,664 | 880 |
| 65,537/65,554 | D | 10,749,456 | 3,409,584 | 4,718,664 | 880 |


Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --all-features
cargo test -p spatialrust-registration --all-features --test gpu_icp_reduction -- --include-ignored --nocapture
cargo test -p spatialrust-registration --all-features --test gpu_icp -- --include-ignored
cargo test -p spatialrust --no-default-features --features register-icp-gpu --test gpu_icp_reduction_smoke -- --include-ignored
cargo test -p spatialrust-registration --all-features --release --test gpu_icp_reduction_benchmark completed_resident_icp_calls -- --include-ignored --nocapture --exact --test-threads=1
```

The benchmark exact filter excludes its imported adapter tests. Benchmarks
and integration adapter tests are ignored by default; explicit runs fail
without an adapter rather than silently skipping. Raw surveys, validation and
JSON are at `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-reduction-*.log`
and `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-reduction-measurements.json`.
