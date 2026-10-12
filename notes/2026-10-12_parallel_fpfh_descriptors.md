# Bounded CPU FPFH descriptors

Date: 2026-10-12. This named slice follows merged
[PR #144](https://github.com/rsasaki0109/SpatialRust/pull/144), main
`f83ed256fdcce6b6927374eadf71195d96d6eb63`.

## Result and execution boundary

FPFH previously computed every SPFH row and weighted descriptor on the caller,
although rows within each phase depend only on immutable input. Under
`register-fpfh`, the public `fpfh_descriptors` API and descriptor construction
inside `FpfhRansacRegistration` now share one caller-built KD-tree across bounded
scoped CPU workers. Workers fill disjoint contiguous slices directly in each
phase's final array. The SPFH scope joins completely before the weighting phase
can read any SPFH row. There is no concurrent read of partially written rows.

The implementation is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/src/fpfh.rs`.
It reuses `spatialrust-search::parallel_worker_count`: one worker below 4,096
points; otherwise at most the available CPU parallelism and
`max(1, floor(points / 16,384))` useful chunks. With default chunking, fewer than
32,768 points retain serial execution. A one-worker phase performs its original
indexed loop without a thread. Parallel phases allocate no per-worker result
vector or result-merge buffer. Worker injection is private and used by tests;
there is no new public worker configuration, dependency or feature flag.

Point order and unsorted KD-tree radius-neighbor order are preserved. Each
point retains the original Darboux-frame math, self/degenerate-neighbor
exclusions, angular bins, per-sub-histogram normalization, f32 distance weights
and the exact accumulation order of all 33 bins. No floating reduction spans
workers. Descriptor matching, tie handling, seeded RNG, RANSAC sampling/scoring
and pose estimation retain their existing execution and arithmetic. Existing
radius, finite-position/unit-normal and coordinate-extent validation is unchanged.

SPFH and FPFH arrays each contain 132 payload bytes per point, with both present
during weighting as before. Lifted positions/normals and the KD-tree also scale
with input count. Each active worker holds one query's radius-neighbor vector;
an unbounded radius may include the entire cloud. The worker cap is not a byte
limit on active scratch. Concurrent callers each have their own worker budget,
not a process-wide CPU limit. There is no all-query neighbor table, distance
matrix, GPU copy or global mutable state.

## Correctness and feature validation

The frozen parent module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/fixtures/fpfh_serial.rs`.
It matches parent main f83ed25 after line-ending/trailing-newline normalization,
with only the original test module omitted. It is compiled solely in native
test builds and retains the complete independent serial public API and seeded
registration implementation.

Five new tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/src/fpfh_parallel_tests.rs`
cover:

- **18,402,648 exact f32 descriptor words** against the frozen public API.
  Inputs have 0/1/17/4,095/4,096/4,103 points, planar/curved/duplicate curved
  geometry and radii 0.075/0.16. Additional 32,771-point curved and duplicate
  cases cross the real multi-worker staging boundary with uneven final bands.
  Forced 1/2/6 workers and public staging retain identical bin words and row order.
- An independent two-point planar expectation: only the center of each
  11-bin angular histogram has value 200. Coincident points, excluded-radius
  neighbors, singleton support and displacement parallel to the normal retain
  zero descriptors. Empty/zero/oversized private worker requests remain safe.
- Invalid/underflowing/overflowing radii, nonfinite positions and nonunit or
  nonfinite normals retain the frozen error strings. Existing extreme-extent
  and configuration rejection tests continue to pass.
- Concurrent forced-worker calls preserve independent results and the borrowed
  input, including renamed XYZ/normal fields, metadata and an integer ID column.
- Seeded registration on a transformed, spatially varied 193-point cloud gives
  the same transform, fitness bits, iteration count and convergence diagnostics
  as frozen registration; convergence and finite fitness are required.

Minimal `register-fpfh` tests pass: 19 native tests and one ignored benchmark.
All-feature registration tests pass: 35 native tests and one ignored benchmark,
including ICP, point-to-plane, GICP, NDT, rigid transforms and FPFH.
The minimal facade API smoke passes. Minimal/all-feature all-target strict
Clippy and facade smoke Clippy pass. Crate and facade WASM compilation passes;
this does not establish browser execution or threading performance. Formatting
and diff checks pass. No workflow or job names changed; all 114 named CI checks
gate merge.

Checks used:

```text
cargo test -p spatialrust-registration --no-default-features --features register-fpfh --lib -- --test-threads=1
cargo test -p spatialrust-registration --all-features --tests -- --test-threads=1 --nocapture
cargo test -p spatialrust --no-default-features --features register-fpfh --test fpfh_parallel_smoke
cargo clippy -p spatialrust-registration --no-default-features --features register-fpfh --all-targets -- -D warnings
cargo clippy -p spatialrust-registration --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features register-fpfh --test fpfh_parallel_smoke -- -D warnings
cargo check -p spatialrust-registration --target wasm32-unknown-unknown --no-default-features --features register-fpfh
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features register-fpfh
cargo fmt --all --check
git diff --check
```

## Completed descriptor-API measurement

Release native Windows, Rust/Cargo 1.98.1. Twelve conditions combine
512/16,384/32,768/65,536 points with planar, curved and duplicate curved inputs.
Radius is 0.11 for the plane and 0.16 for the other two fixtures. This machine
reports 12 available CPU workers; default policy uses 1/1/2/4 workers respectively.
No other tests or builds launched by this task overlap the performance run.
Desktop activity and other callers' CPU usage are not controlled.

The ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/src/fpfh_parallel_benchmark.rs`:

```text
cargo test -p spatialrust-registration --no-default-features --features register-fpfh --release --lib paired_previous_serial_parallel_fpfh -- --ignored --nocapture --test-threads=1
```

Every condition runs two warmups and twenty rotating rounds of the frozen
serial public API, the new forced-one-worker path and the current public API.
Every warmup and measured output is checked against frozen descriptor bits.
Timers include input validation, point/normal lifting, KD-tree construction,
both descriptor allocations and phases, worker spawn/join and destruction of
the SPFH/tree/input staging. Fixture construction, output verification and
destruction of the returned descriptor vector are outside the timer. These are
complete host-returning API measurements, not isolated query timings.

Each table time is a per-variant median. Ratios are medians of twenty paired
old/variant times, with 10,000 paired median bootstrap resamples (seed 20261012,
sorted indices 249 and 9749). Above one favors the new path. The intervals
describe this run's samples and are not independent-run confidence or a
portable/ideal scaling guarantee. All conditions and serial controls are shown.

At 32,768 points/two workers, paired old/public median ratios are
1.421–1.458. At 65,536 points/four workers, they are 2.375–2.595. All six
multi-worker intervals are above one. The six one-worker public controls range
0.984–1.000: some show small regressions, including the 16,384-point plane
(0.984 [0.976, 0.993]) and curved input (0.987 [0.979, 0.997]). Forced-serial
controls also show small regressions in several conditions; every control and
sample range is retained. These observed gains apply to this fixture/host/run,
and do not establish ideal scaling or behavior under concurrent CPU load.

| Points | Surface / radius | Workers | Previous / serial / public ms | Previous/public ratio [95% CI] | Previous/serial ratio [95% CI] |
| ---: | --- | ---: | --- | --- | --- |
| 512 | plane / 0.11 | 1 | 0.791 / 0.791 / 0.790 | 0.996 [0.980, 1.009] | 1.000 [0.986, 1.034] |
| 512 | curved / 0.16 | 1 | 1.221 / 1.232 / 1.229 | 0.993 [0.988, 1.000] | 0.992 [0.989, 0.996] |
| 512 | duplicate curved / 0.16 | 1 | 1.193 / 1.198 / 1.196 | 0.996 [0.991, 1.001] | 0.995 [0.991, 1.000] |
| 16,384 | plane / 0.11 | 1 | 34.960 / 35.795 / 35.450 | 0.984 [0.976, 0.993] | 0.980 [0.961, 0.992] |
| 16,384 | curved / 0.16 | 1 | 67.874 / 68.044 / 68.492 | 0.987 [0.979, 0.997] | 0.988 [0.975, 1.004] |
| 16,384 | duplicate curved / 0.16 | 1 | 106.467 / 107.221 / 107.744 | 1.000 [0.979, 1.010] | 0.996 [0.983, 0.999] |
| 32,768 | plane / 0.11 | 2 | 70.709 / 71.854 / 50.449 | 1.421 [1.240, 1.616] | 0.990 [0.980, 1.008] |
| 32,768 | curved / 0.16 | 2 | 130.051 / 134.410 / 91.899 | 1.444 [1.400, 1.542] | 0.977 [0.939, 0.991] |
| 32,768 | duplicate curved / 0.16 | 2 | 213.817 / 216.531 / 149.335 | 1.458 [1.371, 1.570] | 0.989 [0.971, 1.001] |
| 65,536 | plane / 0.11 | 4 | 158.563 / 156.152 / 67.186 | 2.377 [2.229, 2.534] | 1.022 [0.957, 1.059] |
| 65,536 | curved / 0.16 | 4 | 248.476 / 253.536 / 107.257 | 2.375 [2.272, 2.448] | 0.981 [0.950, 1.003] |
| 65,536 | duplicate curved / 0.16 | 4 | 457.102 / 458.787 / 172.411 | 2.595 [2.387, 2.811] | 0.983 [0.950, 1.030] |

Private reproduction logs are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:
`parallel-fpfh-tests.log`, `parallel-fpfh-all-tests.log`,
`parallel-fpfh-facade.log`, `parallel-fpfh-seeded.log`,
`parallel-fpfh-clippy.log`, `parallel-fpfh-all-clippy.log`,
`parallel-fpfh-facade-clippy.log`, `parallel-fpfh-wasm.log`,
`parallel-fpfh-facade-wasm.log`, `parallel-fpfh-release-build.log`,
`parallel-fpfh-paired.log`, `parallel-fpfh-estimates.json`,
`parallel-fpfh-table.md` and `summarize-parallel-fpfh.py`.
The estimates retain all samples, minimum/maximum times and paired intervals.
