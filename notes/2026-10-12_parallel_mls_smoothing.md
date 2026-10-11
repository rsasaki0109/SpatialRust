# Bounded CPU MLS smoothing

Date: 2026-10-12. Named slice: Bounded CPU MLS smoothing.

## Result and execution boundary

`MlsSmoothing::smoothed_positions` and the existing `filter` implementation under
`filter-mls` now share one caller-built KD-tree across scoped CPU workers. Each
worker fits a contiguous disjoint band directly into the final position vector.
Point indices, the KD-tree's unsorted radius-neighbor order, leave-one-out
selection and the entire weighted covariance/polynomial projection calculation
remain unchanged. No floating reductions cross worker boundaries.

The implementation is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-filtering/src/mls.rs`.
It reuses `spatialrust-search::parallel_worker_count`, already used by
neighborhood outlier filters. Its current policy returns one worker below
4,096 points; otherwise it caps workers by available CPU parallelism and
`max(1, floor(points / 16,384))`. Thus fewer than 32,768 points still use one
worker with the default chunk size. A one-worker call retains the original
capacity-plus-push output allocation. Parallel calls initialize one output
vector and fill its disjoint slices without an intermediate result merge.
Concurrent callers have separate worker budgets, not a process-wide CPU limit.

The shared tree and final positions scale with point count. Each active worker
temporarily holds one point's neighbors, fitted positions and weights; radius
neighborhoods may contain the entire cloud. The worker cap is not a byte bound
on fitting scratch. No all-query neighbor table, distance matrix, GPU copy,
Rayon/runtime dependency, feature flag or public signature is added.

Projection and output reconstruction helpers were byte-checked against parent
main 144e0b3 after trailing newline normalization. Filter reconstruction retains
renamed position fields, original schema, all non-position columns and metadata.
Validation and sparse/singular-support fallbacks retain their previous behavior,
including the existing order-zero fit. This slice does not expand numeric input
validation or alter surface-fitting math.

## Correctness and feature validation

The frozen pre-change production module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-filtering/tests/fixtures/mls_serial.rs`,
checked against 144e0b3 after line-ending/trailing-newline normalization with
only the original test module omitted. It is used solely in test builds.
The independent current/frozen executor comparison is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-filtering/src/mls_parallel_tests.rs`.

Its primary fixture test checks 444,528 exact f32 coordinate words across
0/1/17/4,095/4,096/4,103-point inputs, noisy planes, curved surfaces and duplicate
coordinates. Orders 0/1/2, forced 1/2/6-worker bands and the public staging path
retain frozen results. Worker injection is private to tests and does not change
the public staging policy. Separate tests cover:

- A 32,771-point planar cloud through the public multi-worker path, frozen
  coordinate words and an independent planar geometry check.
- Duplicate/collinear neighborhoods, insufficient support, tiny radii,
  maximal minimum-neighbor counts and signed-zero inputs.
- Concurrent calls with independent outputs and invalid radius/order settings.
- Renamed XYZ fields, schema/metadata, every stored attribute type
  (f32/f64/u8/u16/u32/i32), integer row ordering and NaN payload bits.
- Existing noisy-plane RMS improvement and invalid-setting regressions.

Minimal `filter-mls` native tests pass (seven tests, one ignored diagnostic).
Combined MLS/outlier/crop/FPS/default-voxel tests pass (29 tests, one ignored
diagnostic). The facade's minimal-feature API smoke passes and uses the existing
`spatialrust::filtering::PointCloudFilter` trait path. Minimal and all-feature/
all-target strict Clippy, facade smoke Clippy, formatting and diff checks pass.
WASM `filter-mls` crate and facade compilation pass; no browser/thread execution
is claimed.
The existing 114 named CI checks gate merge; no workflow/job count changes.

Private logs under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:
`parallel-mls-regressions.log`, `parallel-mls-combined-tests.log`,
`parallel-mls-facade.log`, `parallel-mls-facade-smoke.log`,
`parallel-mls-clippy.log`, `parallel-mls-all-clippy.log`,
`parallel-mls-facade-clippy.log`, `parallel-mls-wasm.log`,
`parallel-mls-facade-wasm.log`, and `parallel-mls-release-build.log`.

## Completed position-API measurement

Release native Windows, Rust/Cargo 1.98.1, 12 available CPU workers. There are
twelve point-count/surface/order conditions: 512/16,384/32,768/65,536 points,
noisy plane/order 1, curved/order 2 and duplicate curved/order 2. Radius is 0.16
and minimum support is six. Default policy uses 1/1/2/4 workers respectively.
The ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-filtering/src/mls_parallel_benchmark.rs`:

```powershell
cargo test -p spatialrust-filtering --no-default-features --features filter-mls --release --lib paired_previous_serial_parallel_mls -- --ignored --nocapture --test-threads=1
```

Each condition runs two warmups, then twenty rotating rounds of the frozen
previous position API, the current forced-serial executor, and the current
public automatic-worker API. Every output, including warmups and measured
calls, is checked against frozen f32 words outside its timer. Timers include
validation, KD-tree build, output allocation, neighborhood searches, all fits,
worker spawn/join and completion. They exclude fixture construction, correctness
checks, result destruction and Cargo compilation. Output-cloud reconstruction
and non-position field copies are outside these position-API measurements.
Native regression runs and the release build completed before the measurement;
no agent-owned test, build or other benchmark ran concurrently. CPU clock/state
and desktop load are uncontrolled.

The tree is still built serially, so worker count is not a speedup guarantee.
The forced-serial control distinguishes staging gains from the extracted
per-point helper. All small/one-worker and adverse conditions are reported.
Ratios are medians of corresponding per-round previous/current ratios, not
quotients of separate medians. Ratio above one means the current path was
faster. Intervals use seeded 10,000-resample paired-median bootstrap estimates
(seed 20261012); these are local samples, not portable performance guarantees.

All six multi-worker conditions have paired intervals above one. At 32,768
points with two workers, previous/current-public ratios are 1.503–1.845; at
65,536 points with four workers they are 2.366–2.661. The six one-worker public
conditions range from 0.993 to 1.043. The forced-serial control and all conditions
are included below; small differences do not establish portable speedups.

The 65,536-point noisy-plane condition had substantial wall-time variance:
its separate previous/parallel medians suggest a larger improvement than its
paired ratio. The reported 2.661 [2.496, 3.041] estimate uses paired samples,
and no quotient-of-medians speedup or ideal fourfold scaling is claimed.
Other desktop load and CPU frequency changes were not controlled.

| Points | Surface / order | Workers | Previous / serial / parallel ms | Previous / parallel ratio [95% CI] | Previous / serial ratio [95% CI] |
| ---: | --- | ---: | --- | --- | --- |
| 512 | noisy plane / 1 | 1 | 1.598 / 1.605 / 1.606 | 0.997 [0.988, 1.017] | 0.996 [0.981, 1.019] |
| 512 | curved / 2 | 1 | 2.162 / 2.115 / 2.107 | 1.033 [1.021, 1.043] | 1.031 [1.028, 1.052] |
| 512 | duplicate curved / 2 | 1 | 2.594 / 2.496 / 2.497 | 1.033 [1.022, 1.038] | 1.037 [1.031, 1.048] |
| 16,384 | noisy plane / 1 | 1 | 78.165 / 79.666 / 78.995 | 0.993 [0.969, 1.017] | 0.984 [0.947, 1.004] |
| 16,384 | curved / 2 | 1 | 108.290 / 102.914 / 106.714 | 1.019 [0.990, 1.097] | 1.031 [1.015, 1.064] |
| 16,384 | duplicate curved / 2 | 1 | 170.853 / 166.034 / 164.739 | 1.043 [1.002, 1.051] | 1.040 [1.001, 1.069] |
| 32,768 | noisy plane / 1 | 2 | 169.614 / 162.263 / 118.788 | 1.503 [1.373, 1.630] | 1.031 [1.007, 1.058] |
| 32,768 | curved / 2 | 2 | 237.378 / 222.275 / 148.268 | 1.597 [1.456, 1.780] | 1.044 [0.971, 1.086] |
| 32,768 | duplicate curved / 2 | 2 | 472.464 / 459.428 / 259.484 | 1.845 [1.646, 2.021] | 1.065 [0.960, 1.193] |
| 65,536 | noisy plane / 1 | 4 | 663.181 / 459.305 / 167.261 | 2.661 [2.496, 3.041] | 1.130 [0.883, 1.338] |
| 65,536 | curved / 2 | 4 | 395.970 / 380.823 / 174.554 | 2.366 [2.141, 2.572] | 1.040 [1.020, 1.064] |
| 65,536 | duplicate curved / 2 | 4 | 686.403 / 665.211 / 279.782 | 2.607 [2.344, 2.669] | 1.036 [1.014, 1.060] |

Raw samples, ranges and analysis are retained privately at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-mls-paired.log`,
`parallel-mls-estimates.json`, `parallel-mls-table.md` and
`summarize-parallel-mls.py` in the same directory.

Parent GPU geometry PR142 is merged at 144e0b3 with all 114 checks successful.
GPU shader/ownership/transfer contracts and `spatialrust-core` are unchanged.
