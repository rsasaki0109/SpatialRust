# Resident GICP normal-equation reduction

The subsequent [working-set/preparation slice](2026-10-13_gpu_gicp_reuse_transformed_queries.md)
reuses query storage and reduces nonempty private scratch from 88*S + 240*G
to 72*S + 240*G. Its GPU-entry-point CPU k-NN preparation directly packs rows
with bounded workers, retaining the original arithmetic and receipts. The
execution layout and measurements below describe the initial implementation.

Date: 2026-10-13. Named slice: device transforms, covariance inverse weighting,
Hessian/gradient and Euclidean error, with an explicit CPU 6x6 solve boundary.

## Execution and transfer contract

`register-gicp-gpu` adds
`GicpRegistration::align_with_gpu_reduction(runtime, source, target)`.
Original source/target, target tree and per-point covariances remain on GPU
across iterations. Five device passes per evaluation transform originals, run
the existing unbounded nearest traversal followed by the inclusive distance
gate, prepare rotated covariance inverses, reduce partial equations/error and
finalize validity. CPU solves normalized 6x6 equations and applies the existing
world update/stopping convention. There is no repeated point upload, neighbor/
pair readback or automatic CPU fallback.

Covariance preparation remains CPU k-NN when radius is absent. Configured
radius covariance uses the supplied GPU runtime, explicitly downloads its
32-byte rows, then re-uploads packed covariances with full receipts. This
preparation is NOT device resident. Source k-NN builds its own CPU tree; radius
preparation avoids that unused tree. The target CPU tree is shared with k-NN
preparation and uploaded search. Ordinary CPU GICP and the existing explicit
correspondence route retain their code and fallback/placement contracts.

`GpuGicpInput::upload` under `gpu-wgpu` retains the ICP position representation
and original-order covariance rows `[xx,yy,zz,xy,xz,yz]`. H2D payload for S
source points, T target points and M tree nodes is `48*S + 64*T + 32*M` bytes:
16*S source positions, 32*T original/tree target positions, 32*M nodes and
32*(S+T) padded covariance rows in one storage buffer. Covariance preparation
receipts are additional. Uniform controls are excluded. Target must be nonempty;
empty source evaluation submits/readbacks nothing, while target upload remains.

Packed f32 covariances must be finite SPD, validated with f64 Cholesky.
The maximum target inverse trace bounds combined inverse weights, since
`C_target + R*C_source*R^T >= C_target`. That weight scale must be normal finite
f32. Positions and global extent squares must be finite; the combined positive
normalization extent square must be normal in f32 (coincident sets use scale 1).
Every evaluation requires a finite unit pose, positive finite nonzero f32
distance square and supported counts/buffers/dispatch/bindings.

The device symmetrizes rotated combined covariance and uses scaled 3x3 Cholesky
inversion. Any pivot <=1e-6 of the largest combined diagonal sets unsupported
conditioning status. Overflow/nonfinite weights or subnormal inverse diagonals
also reject. The whole result becomes invalid; pairs are not silently dropped.
These restrictions differ from CPU's f64 cofactor inverse/absolute determinant
test and are documented opt-in behavior.

The normalized linearized equation basis is explicit. For transformed source
point p, actual GPU-transformed bounding-box midpoint a, coordinate scale s
and maximum distance d:

```
q = (p-a)/s
r = (p-target)/d
W = inverse(C_target + R*C_source*R^T) / weight_scale
J = [-skew(q) | I]
H = sum(J^T W J), g = sum(J^T W r)
solve H*u = -g
omega = u_rotation * d/s
local_t = u_translation * d
world_t = local_t - omega cross a
```

`GpuGicpEquations::world_step` performs that LINEARIZED conversion. The existing
CPU finite rotation construction, transform composition and solution norm are
then used; this is not a new rotation-about-anchor finite update policy.
Centering/scaling and a common weight multiplier preserve the mathematical
linearized least-squares problem but change numerical conditioning/rounding.
The normalized CPU solve still uses existing Gaussian elimination. It does
not provide a CPU coordinate, observability or stopping-equivalence guarantee.

GPU reductions use compensated hi/lo f32 pairs, with at most 256 groups of
64 threads and a scalar term per workgroup. Twenty-one upper-triangle Hessian
terms, six gradients and one Euclidean residual sum form the fixed output.
Products, covariance inversion, transforms and division remain f32. Physical
Euclidean error is the normalized residual squared sum times the f32 distance
square; it is not the Mahalanobis objective.

Private scratch is 16*S query bytes, 8*S match bytes, 64*S prepared-pair bytes
and 240 bytes per partial group (at most 61,440). Consumed words are initialized,
leased exclusively through one submission and recycled to the bounded pool.
Owned results never enter the pool and survive input drop/concurrent calls.
Lazy per-runtime pipelines require eight storage bindings, one 96-byte uniform,
64-thread groups and 1,024 shared bytes. No nearest shader or CPU GICP math
changes are made; no core/math/search dependency or unsafe code is added.

`GpuGicpNormalEquations` owns 256 bytes: u32 status/equation count/fitness count/
reserved zero, one vec4<f32> actual anchor and 28 vec2<f32> compensated terms.
Status bits 1/2/4/8 denote transform overflow, malformed match, nonfinite
arithmetic and unsupported covariance conditioning. Any nonzero status
invalidates ALL payload, which is zeroed in finalization. Zero-match payload
is zero too. Consumers must inspect status and preserve immutable data.
`parameter_scales` exposes rotation/translation/weight scales for decoding.

Explicit `download` copies 256 D2H bytes, waits, validates status/counts/reserved
word and finite/zero-match invariants, then decodes normalized H/g and physical
fitness in f64. Empty source readback is zero. No payload transfer/wait occurs
inside resident evaluation. The wrapper evaluates the initial pose once and
each updated pose once; post-update equations serve the next solve and final
fitness. I updates need I+1 evaluations and `256*(I+1)` iterative D2H bytes,
plus any covariance-preparation readbacks. Last post-update equations are fully
validated even when only final fitness is needed.

New wrapper configuration requires at least one iteration, minimum six pairs,
finite positive representable covariance epsilon/distance square, nonnegative
finite stopping thresholds and finite unit initial pose. Legacy GICP accepts
additional gate/configuration edge cases; its behavior remains unchanged.

## Precision and correctness evidence

Four primitive tests compare every H/g component with independent f64 matrix
inversion by Gaussian solves, rotated covariances and Jacobian/residual math at
1/3/63/64/65/257/4097 points. Selected bounds are
`2e-5*(1e-4 + sum(abs(reference scalar products)))`. Maximum observed normalized
equation error is 0.0000010000217020915932. This is selected-fixture evidence,
not a universal error bound. Linearized basis conversion is checked independently.
Tests cover SPD/count/limit validation, empty source, inclusive gate, zero
matches, strict covariance conditioning with status 8/all-zero payload, transform
overflow, ownership/runtime mismatches and concurrent scratch reuse.

Two explicitly run adapter integrations cover 108 size/scale/covariance/stopping
conditions and 98,928 transformed points. Six corner sizes span workgroup tails
and 4,098 points; scales 1e-4/1/1e4, CPU k-NN/GPU radius preparation and iteration/
transform/fitness stopping are compared. Maximum scale-normalized CPU final
pose-coordinate error is 0.0000009536743 under the selected 1e-4*scale bound.
Iteration/convergence results, history, fixed readbacks and complete covariance
transfer receipts agree. Fitness is independently recomputed at the returned
pose. Unequal clouds with seven gated outliers, concurrency, invalid configuration,
bad radius propagation and insufficient matches are checked; inputs are unchanged.

GPU all-features regression: 215 passed / 23 ignored. Registration library:
35 passed / 1 ignored. Three existing GPU GICP adapter tests and one facade
adapter test pass. All-feature GPU/registration and CPU-only GICP Clippy,
minimal GPU, WASM feature compilation and Rustfmt pass. Source comparison
against `24efc33c29f4c12fa5ecece0a5ad7f3e45eb9475` verifies that removing the
new child-module wiring leaves existing GICP source exactly unchanged. Removing
the internal position-field visibility changes leaves the ICP device source
exactly unchanged too.

Native Vulkan execution is verified; WASM evidence is compilation only.
Hosted CI does not replace native adapter execution. f32 inverse weights and
normalized solving can change matching, pose and stopping near gates, covariance
conditioning or geometric degeneracy. Existing f64 CPU/correspondence APIs remain
available; neither convergence nor small fixture error certifies a correct pose.

## Completed-call measurements

Two isolated release surveys use a native Vulkan discrete-device backend,
three warmups and 20 paired rounds, cyclic CPU/previous-GPU/new-GPU/resident
stage order. No other local builds/tests/statistics run during timing. All
samples are retained. Runtime/pipeline setup is warmed outside timers; returned
output verification/drop is outside, internal preparation/staging destruction
is inside full-call timers.

All APIs perform THREE updates with disabled stopping thresholds and gate 0.25.
K uses CPU k-NN (k=6) on a 32x32-column volume grid, spacing 0.5; R uses GPU
radius 0.625 on that grid; D duplicates consecutive grid points, has 17 extra
target rows and uses the radius path. Source is translated by
(0.03125,-0.015625,0.0078125). Full-call timings include covariance preparation,
tree construction, all uploads/readbacks, iteration search/updates, fitness
and CPU solving. Radius preparation uses a shared runtime in the ordinary CPU
route and the supplied runtime in both explicit GPU routes. Their actual
preparation placement is part of the comparison.

Prepared resident timings use explicit radius covariances in ALL profiles,
including K. They include one evaluation's owned allocation, scratch leases,
all five passes, submission and `wait_idle`, excluding preparation/upload,
readback, CPU solving and the iterative loop. They cannot isolate full-call
k-NN covariance behavior and are not GPU timestamps or whole-registration times.
Normal-equation/precision changes and evaluation reuse (seven versus four
batches at the fixed iteration budget) also contribute to full-call ratios.

Times are medians in milliseconds. Ratios are geometric means of 20 paired
baseline/new timings. Percentile 95% intervals use 10,000 paired-log bootstrap
resamples, sorted entries 249/9749 and seeds 20261013 + 2*zero-based record
index + baseline index (ordinary API=0, previous GPU=1). The "CPU" column is
the ordinary API; R/D already use GPU radius covariance preparation.

| Survey | Source/target | Profile | CPU ms | Previous GPU ms | New GPU ms | Prepared resident ms | Max pose-coordinate error | CPU/new ratio (95% interval) | Previous/new ratio (95% interval) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 257/257 | K | 0.681 | 8.988 | 5.574 | 0.275 | 0 | 0.12x (0.12-0.13) | 1.61x (1.56-1.67) |
| 1 | 257/257 | R | 2.897 | 10.951 | 7.901 | 0.318 | 0 | 0.37x (0.36-0.39) | 1.37x (1.33-1.41) |
| 1 | 257/274 | D | 3.442 | 12.758 | 9.505 | 0.354 | 0 | 0.36x (0.34-0.38) | 1.34x (1.30-1.40) |
| 1 | 4,097/4,097 | K | 22.802 | 25.316 | 17.865 | 0.793 | 9.31e-09 | 1.23x (1.12-1.35) | 1.37x (1.29-1.46) |
| 1 | 4,097/4,097 | R | 12.712 | 15.502 | 9.632 | 0.466 | 3.73e-09 | 1.25x (1.13-1.38) | 1.60x (1.51-1.69) |
| 1 | 4,097/4,114 | D | 10.585 | 14.808 | 8.837 | 0.399 | 5.59e-09 | 1.16x (1.06-1.28) | 1.62x (1.51-1.73) |
| 1 | 16,384/16,384 | K | 100.238 | 77.435 | 63.414 | 1.417 | 4.66e-09 | 1.56x (1.38-1.75) | 1.20x (1.08-1.31) |
| 1 | 16,384/16,384 | R | 54.814 | 26.317 | 15.589 | 0.947 | 2.98e-08 | 3.46x (3.13-3.83) | 1.66x (1.52-1.80) |
| 1 | 16,384/16,401 | D | 59.652 | 35.230 | 23.071 | 1.855 | 5.96e-08 | 2.43x (2.18-2.69) | 1.50x (1.44-1.55) |
| 1 | 65,537/65,537 | K | 378.290 | 246.528 | 238.713 | 16.484 | 7.45e-09 | 1.54x (1.46-1.63) | 1.05x (0.99-1.13) |
| 1 | 65,537/65,537 | R | 213.786 | 84.080 | 68.756 | 8.077 | 1.3e-08 | 2.94x (2.67-3.21) | 1.18x (1.10-1.26) |
| 1 | 65,537/65,554 | D | 191.618 | 87.784 | 87.861 | 7.899 | 1.19e-07 | 2.40x (2.15-2.68) | 1.07x (1.00-1.15) |
| 2 | 257/257 | K | 0.718 | 9.077 | 5.768 | 0.278 | 0 | 0.13x (0.13-0.14) | 1.65x (1.58-1.74) |
| 2 | 257/257 | R | 2.714 | 10.186 | 7.107 | 0.266 | 0 | 0.39x (0.38-0.40) | 1.47x (1.44-1.51) |
| 2 | 257/274 | D | 2.489 | 9.331 | 6.715 | 0.246 | 0 | 0.37x (0.36-0.38) | 1.41x (1.37-1.44) |
| 2 | 4,097/4,097 | K | 15.652 | 19.527 | 12.649 | 0.423 | 9.31e-09 | 1.22x (1.13-1.32) | 1.45x (1.36-1.55) |
| 2 | 4,097/4,097 | R | 13.419 | 18.952 | 10.771 | 1.121 | 3.73e-09 | 1.23x (1.15-1.32) | 1.72x (1.63-1.81) |
| 2 | 4,097/4,114 | D | 10.516 | 17.796 | 10.876 | 1.264 | 5.59e-09 | 1.00x (0.93-1.07) | 1.63x (1.54-1.71) |
| 2 | 16,384/16,384 | K | 86.749 | 72.536 | 69.176 | 3.900 | 4.66e-09 | 1.28x (1.18-1.41) | 1.10x (1.03-1.18) |
| 2 | 16,384/16,384 | R | 53.532 | 36.968 | 29.341 | 3.899 | 2.98e-08 | 1.80x (1.72-1.89) | 1.25x (1.22-1.27) |
| 2 | 16,384/16,401 | D | 40.267 | 25.968 | 15.155 | 0.928 | 5.96e-08 | 2.77x (2.58-2.98) | 1.72x (1.62-1.82) |
| 2 | 65,537/65,537 | K | 383.586 | 252.885 | 203.429 | 1.959 | 7.45e-09 | 1.79x (1.66-1.93) | 1.19x (1.10-1.27) |
| 2 | 65,537/65,537 | R | 224.298 | 79.496 | 47.166 | 1.855 | 1.3e-08 | 4.71x (4.29-5.15) | 1.73x (1.62-1.85) |
| 2 | 65,537/65,554 | D | 197.096 | 73.971 | 41.336 | 1.623 | 1.19e-07 | 4.45x (4.05-4.85) | 1.68x (1.55-1.80) |

Previous-GPU/new point estimates are 1.05-1.73x across the 24 selected
conditions. Two first-survey 65,537-point conditions (K and D) have intervals
including 1, so their small gains are inconclusive. At that size, ordinary-API/
new ratios are 1.54-4.71x across preparation profiles. 257-point new GPU calls
remain slower than the ordinary API (0.12-0.39x), and the second 4,097-point D
condition has an ordinary/new interval spanning 1. These adverse/variable cases
are retained. All final match counts equal S; maximum selected benchmark
CPU pose-coordinate difference is 0.00000011920929. Raw fitness values remain
in JSON, including tiny nonzero GPU residuals when CPU reports exact zero.

This supports an explicit device phase without changing defaults or establishing
universal accuracy/performance or a stable deployment crossover. CPU k-NN
covariance preparation and radius readback/re-upload remain potential bottlenecks.
Resident times vary substantially between surveys; no samples are excluded.

Complete transfer payloads below include covariance preparation, and agree
between surveys. Three updates need only 1,024 bytes of iterative equations
readback. At 65,537 points K drops from 3,670,072 to 1,024 total D2H bytes;
R still includes 4,194,368 covariance bytes, dropping from 7,864,440 to
4,195,392 TOTAL D2H bytes. Uniform controls are excluded.

| Source/target | Profile | Previous H2D | New H2D | Previous D2H | New D2H (including covariance) |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 257/257 | K | 33,952 | 29,840 | 14,392 | 1,024 |
| 257/257 | R | 43,584 | 39,472 | 30,840 | 17,472 |
| 257/274 | D | 44,488 | 41,192 | 31,384 | 18,016 |
| 4,097/4,097 | K | 540,832 | 475,280 | 229,432 | 1,024 |
| 4,097/4,097 | R | 691,944 | 626,392 | 491,640 | 263,232 |
| 4,097/4,114 | D | 683,576 | 618,840 | 492,184 | 263,776 |
| 16,384/16,384 | K | 2,162,656 | 1,900,512 | 917,504 | 1,024 |
| 16,384/16,384 | R | 2,751,952 | 2,489,808 | 1,966,080 | 1,049,600 |
| 16,384/16,401 | D | 2,721,084 | 2,459,756 | 1,966,624 | 1,050,144 |
| 65,537/65,537 | K | 8,650,912 | 7,602,320 | 3,670,072 | 1,024 |
| 65,537/65,537 | R | 11,008,104 | 9,959,512 | 7,864,440 | 4,195,392 |
| 65,537/65,554 | D | 10,879,736 | 9,831,960 | 7,864,984 | 4,195,936 |


Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --all-features
cargo test -p spatialrust-registration --all-features --test gpu_gicp_reduction -- --include-ignored --nocapture
cargo test -p spatialrust-registration --all-features --test gpu_gicp -- --include-ignored
cargo test -p spatialrust --no-default-features --features register-gicp-gpu --test gpu_gicp_reduction_smoke -- --include-ignored
cargo test -p spatialrust-registration --all-features --release --test gpu_gicp_reduction_benchmark completed_resident_gicp_calls -- --include-ignored --nocapture --exact --test-threads=1
```

The exact benchmark filter excludes imported adapter tests. Benchmarks and
adapter integrations are ignored by default; explicit runs fail without a native
adapter rather than skipping. Logs and raw timing/JSON artifacts are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-gicp-reduction-*.log`
and `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-gicp-reduction-measurements.json`.
