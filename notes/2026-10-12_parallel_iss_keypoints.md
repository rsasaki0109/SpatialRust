# Bounded CPU ISS keypoint detection

Date: 2026-10-12. This named slice follows
[PR #148](https://github.com/rsasaki0109/SpatialRust/pull/148), source
`e5b0c6a6299350679fc40d521844561c41510b26`.

## Result and execution boundary

ISS previously evaluated every neighborhood covariance and local maximum on
the caller. Under `feature-iss`, `IssKeypointDetector::detect` now shares one
caller-built KD-tree across bounded scoped CPU workers. The first phase fills
disjoint contiguous slices of the existing saliency array. Its scope joins
completely before local-maximum selection reads any saliency value. Parallel
selection fills a temporary mask, then gathers indices in ascending original
order. The mask is dropped before the selected cloud's attributes are copied.

The implementation is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/src/iss.rs`.
It reuses `spatialrust-search::parallel_worker_count`: one worker below 4,096
points; otherwise at most available CPU parallelism and
`max(1, floor(points / 16,384))` useful chunks. With default chunking, fewer than
32,768 points retain serial execution. One-worker calls run on the caller and
collect indices directly, without a selection mask. Worker injection is private
and used by tests; no public worker setting, dependency or feature flag is added.

Unsorted radius-neighbor order, f64 centroid/covariance arithmetic, eigenvalue
sorting, ratio comparisons and f32 saliency conversion retain their original
operation order. There is no floating-point reduction across workers. Non-salient
NaN rows are skipped; equal local maxima all survive the existing `>=` rule.
The old comment describing strict maxima is corrected to match that behavior.
Radius validation, unusual/NaN configurations and empty-cloud behavior remain
unchanged. Selected indices, schema, metadata and all six buffer types are
preserved. The existing covariance/eigenvalue and attribute-gather helpers are
verified unchanged against the parent source.

Saliency retains four payload bytes per input point. Parallel selection adds
one temporary byte per point; it does not allocate per-worker index vectors
or a second result-sized index merge buffer. KD-tree, selected indices and the
returned cloud remain input/result-sized. Each worker holds only its current
radius-neighbor vector, which may include the entire cloud for a large radius.
Worker limits are per call rather than a process-wide CPU or scratch-byte budget.
Concurrent callers have independent budgets. No all-query neighbor table,
distance matrix, global mutable state or device transfer is introduced.

## Correctness and feature validation

The frozen parent module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/tests/fixtures/iss_serial.rs`.
It matches source e5b0c6a after line-ending/trailing-newline normalization, with
only the original test module omitted. It is compiled solely in native test
builds and retains the independent serial public detector.

Four tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/src/iss_parallel_tests.rs`
cover:

- **1,081,992 exact selection decisions and selected clouds** against the frozen
  serial public API: 0/1/17/4,095/4,096/4,103/32,771 points, planar/noisy curved/
  duplicate noisy curved geometry, radii 0.075/0.16, forced 1/2/6 workers and
  public staging. Large cases cross real multi-worker staging with uneven bands.
- Independent covariance eigenvalues for eight anisotropic box corners. All
  eight equal maxima survive suppression and an isolated ninth point is
  excluded. Coincident, collinear and sparse support yield no keypoints.
- Exact selected indices and gathered F32/F64/U8/U16/U32/I32 fields on a
  32,771-point cloud, with renamed XYZ semantics, metadata, signed-zero words
  and NaN payload words. Both forced-six-worker and public paths are checked.
- Frozen results/errors for invalid, NaN and infinite radii, unusual/NaN gamma
  thresholds, zero/oversized minimum support and empty/singleton input. Zero/
  oversized private worker requests stay safe. Concurrent calls preserve
  independent results on shared borrowed input.

Minimal `feature-iss` tests pass: seven native tests and one ignored benchmark.
All-feature tests pass: twenty-three native tests and two ignored benchmarks,
including boundary detection, CPU/GPU normals and orientation. One facade
smoke passes. Minimal/all-feature all-target strict Clippy and facade smoke
Clippy pass. Crate and facade WASM compilation passes; browser execution or
threading performance is not established. Formatting and diff checks pass.
No workflow or check names change; all 114 named CI checks gate merge.

```text
cargo test -p spatialrust-features --no-default-features --features feature-iss --lib -- --test-threads=1 --nocapture
cargo test -p spatialrust-features --all-features --tests -- --test-threads=1
cargo test -p spatialrust --no-default-features --features feature-iss --test iss_parallel_smoke
cargo clippy -p spatialrust-features --no-default-features --features feature-iss --all-targets -- -D warnings
cargo clippy -p spatialrust-features --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features feature-iss --test iss_parallel_smoke -- -D warnings
cargo check -p spatialrust-features --target wasm32-unknown-unknown --no-default-features --features feature-iss
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features feature-iss
cargo fmt --all --check
git diff --check
```

## Completed detection-API measurement

Release native Windows, Rust/Cargo 1.98.1. Twelve conditions combine
512/16,384/32,768/65,536 points with planar, noisy curved and duplicate noisy
curved inputs. Radius is 0.11 for the plane and 0.16 for the other fixtures;
suppression radius is 0.75 times saliency radius. Gamma thresholds and minimum
support retain their defaults. This machine reports 12 available CPU workers;
default policy uses 1/1/2/4 workers respectively. No other tests or builds
launched by this task overlap the run. Desktop activity and other callers'
CPU usage are not controlled.

The ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/src/iss_parallel_benchmark.rs`:

```text
cargo test -p spatialrust-features --no-default-features --features feature-iss --release --lib paired_previous_serial_parallel_iss -- --ignored --nocapture --test-threads=1
```

Each condition runs two warmups and twenty rotating rounds of the frozen serial
public API, the new forced-one-worker path and the current public API. All 792
completed warmup/measured results are checked against frozen selected indices
and clouds. Timers include validation, KD-tree construction, saliency allocation,
covariance/eigenvalue computation, local-maximum selection, worker spawn/join,
ordered index collection, schema/attribute gathering and scratch/tree destruction.
Fixture construction, output verification and returned result destruction are
outside the timer. These are complete host-returning `detect` measurements,
including the three XYZ attribute columns in these fixtures.

Table times are per-variant medians. Ratios are medians of twenty paired old/new
times, with 10,000 paired median bootstrap resamples (seed 20261012, sorted
indices 249 and 9749). Above one favors the new path. Intervals describe this
run's samples, rather than portability or repeat-run guarantees.

| Points | Geometry | Workers | Old / forced one / public ms | Old / forced one (95% CI) | Old / public (95% CI) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | plane | 1 | 0.896 / 0.921 / 0.869 | 1.039 (0.973–1.168) | 0.962 (0.881–1.065) |
| 512 | noisy curved | 1 | 1.497 / 1.560 / 1.459 | 0.987 (0.939–1.094) | 1.006 (0.951–1.058) |
| 512 | duplicate noisy curved | 1 | 1.004 / 0.928 / 0.951 | 1.037 (0.985–1.149) | 0.999 (0.982–1.030) |
| 16,384 | plane | 1 | 36.406 / 35.885 / 36.119 | 0.989 (0.961–1.050) | 1.000 (0.955–1.034) |
| 16,384 | noisy curved | 1 | 52.939 / 53.267 / 59.313 | 0.998 (0.918–1.114) | 0.957 (0.887–0.994) |
| 16,384 | duplicate noisy curved | 1 | 65.307 / 61.678 / 61.220 | 1.031 (0.991–1.129) | 1.013 (0.911–1.070) |
| 32,768 | plane | 2 | 42.016 / 41.405 / 29.342 | 0.953 (0.901–1.088) | 1.411 (1.262–1.577) |
| 32,768 | noisy curved | 2 | 99.390 / 104.419 / 69.727 | 0.902 (0.866–0.956) | 1.450 (1.373–1.492) |
| 32,768 | duplicate noisy curved | 2 | 165.468 / 144.637 / 90.523 | 1.049 (0.949–1.207) | 1.837 (1.431–2.005) |
| 65,536 | plane | 4 | 70.653 / 73.114 / 34.562 | 1.041 (0.918–1.077) | 2.096 (1.946–2.266) |
| 65,536 | noisy curved | 4 | 201.126 / 212.835 / 77.801 | 0.985 (0.865–1.074) | 2.541 (2.382–2.729) |
| 65,536 | duplicate noisy curved | 4 | 299.961 / 312.491 / 117.359 | 0.918 (0.827–1.023) | 2.321 (2.113–2.670) |

All six two/four-worker conditions favor public staging, with paired median
speedups 1.411–2.541 and interval lower bounds above one. The 16,384-point noisy
curved public control has an adverse ratio of 0.957 (0.887–0.994); the other
five one-worker public intervals include one. The forced-one-worker noisy
curved case at 32,768 points also regresses, at 0.902 (0.866–0.956). Both adverse
controls are retained; serial neutrality and fixed cross-machine speedups are
not claimed. Timing variation is visible throughout this uncontrolled desktop
run, and bootstrap intervals do not capture all sources of systematic variation.
The existing conservative worker policy is retained. All measured public
parallel paths complete the same detection API, including selected XYZ copying;
there is no claim about GPU ISS or a whole registration pipeline.

Raw samples and analysis remain private ignored files under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-iss-*`.
Published measurements describe execution conditions without a local adapter
model name. GPU ISS, parallel attribute gathering and process-wide worker
budgets are outside this slice.
