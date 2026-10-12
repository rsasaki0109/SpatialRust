# Bounded CPU cloud distance metrics

Date: 2026-10-12. This named slice follows
[PR #149](https://github.com/rsasaki0109/SpatialRust/pull/149), source
`e86cef4f3114dbd213f04e6243af4a898f931e79`.

## Result and execution boundary

Chamfer/Hausdorff statistics previously queried and accumulated every nearest
distance on the caller. Under `metrics-distance`, `cloud_distances` now shares
each target KD-tree across bounded scoped CPU query workers. It still builds
exactly one tree per side and computes a-to-b before b-to-a. Public Chamfer and
Hausdorff convenience functions use the same path; signatures and metric
definitions are unchanged.

The implementation is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-metrics/src/distance.rs`.
Each direction independently reuses `spatialrust-search::parallel_worker_count`:
one worker below 4,096 points; otherwise at most available CPU parallelism and
`max(1, floor(query_points / 16,384))` useful chunks. With default chunking,
fewer than 32,768 query points retain serial execution. Worker injection is
private and used by tests. The one-worker directed loop matches the parent
verbatim, including its original query allocation behavior.

Parallel workers fill disjoint slices of an optional f32 distance window. Its
capacity is at most `min(query_points, workers * 16,384)` entries; the product
uses saturating arithmetic before limiting it to the query length. Each window
joins completely, then its values are converted to f64 and accumulated in
original point order. The sum is not reset between windows and no worker-local
sums are combined. Division, square roots, directed maximum comparisons and
symmetric statistics preserve their original arithmetic. Missing neighbors stay
distinct from a returned NaN. Empty inputs retain the original error.

Each active band reuses a one-neighbor vector through `KdTree::nearest_k_into`
with k=1. The existing `nearest_one` uses that same traversal via `nearest_k`;
the reusable API clears its result and initializes a fresh query accumulator.
This avoids a heap allocation per parallel query without changing backend
traversal, tie handling or distance math. KD-tree construction is unchanged.

An optional distance occupies eight bytes on the measurement host. Parallel
score scratch is therefore at most 128 KiB per worker there, or 1.5 MiB at the
host's maximum twelve-worker budget, independent of total query count above
that window size. The two directions' windows are not live together. Each
worker also holds its one-neighbor vector and existing search state. The two
KD-trees remain input-sized. These per-call budgets do not establish a
process-wide CPU limit; concurrent callers have independent budgets. No
all-query neighborhood table, new dependency, global mutable state or device
transfer is introduced.

## Correctness and feature validation

The frozen parent module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-metrics/tests/fixtures/distance_serial.rs`.
It matches source e86cef4 after line-ending/trailing-newline normalization, with
only the original test module omitted. It is compiled solely in native test
builds and retains the complete independent serial public metric APIs.

Five tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-metrics/src/distance_parallel_tests.rs`
cover:

- **660 exact f64 output words** across the four directed statistics, symmetric
  Chamfer/Hausdorff methods and public wrappers. Inputs have
  1/17/4,095/4,096/4,103/32,767/32,768/32,771/65,537 points, planar/noisy curved/
  duplicate geometry and asymmetric target counts, with forced 1/2/6 workers
  and public staging. This includes multi-window and partial-band boundaries.
- Independent asymmetric means/maxima for two points versus three, plus tied
  nearest distances and duplicate points. Different worker counts per side,
  zero requests and oversized requests preserve those exact expectations.
- A 65,537-query order-sensitive case with a first squared distance around
  1e16 and distinct later distances below one. Each later addition rounds away
  in the original sum, while regrouping them changes the result. Forced two
  workers cross two full windows and a one-point tail; public staging crosses
  a full four-worker window and a one-point tail. Both retain frozen words.
- NaN payloads, infinities, squared-distance overflow and signed zero retain
  frozen outputs in both directions. Returned NaNs remain distinct from
  missing-neighbor rows on a private empty-tree query. Empty public inputs
  retain frozen error strings through all three APIs.
- Renamed XYZ semantics, an ID attribute and metadata remain untouched during
  concurrent calls with different per-side worker budgets.

The ordered-sum fixture initially used tens of thousands of identical small
coordinates. Its pre-existing KD-tree equal-coordinate partitioning dominated
the test. Distinct small coordinates retain the same numerical stress and
window boundaries while keeping this correctness test focused on accumulation;
KD-tree partition optimization is outside this slice.

Distance-feature tests pass: eight native tests and one ignored benchmark. One
facade smoke passes. All-target strict crate Clippy and facade smoke Clippy
pass. Disabling the sole metrics feature compiles without its optional search
dependency. Crate and facade WASM compilation passes; browser execution or
threading performance is not established. Formatting and diff checks pass.
No workflow or check names change; all 114 named CI checks gate merge.

```text
cargo test -p spatialrust-metrics --no-default-features --features metrics-distance --lib -- --test-threads=1 --nocapture
cargo clippy -p spatialrust-metrics --no-default-features --features metrics-distance --all-targets -- -D warnings
cargo check -p spatialrust-metrics --no-default-features
cargo test -p spatialrust --no-default-features --features metrics-distance --test distance_parallel_smoke
cargo clippy -p spatialrust --no-default-features --features metrics-distance --test distance_parallel_smoke -- -D warnings
cargo check -p spatialrust-metrics --target wasm32-unknown-unknown --no-default-features --features metrics-distance
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features metrics-distance
cargo fmt --all --check
git diff --check
```

## Completed distance-API measurements

Release native Windows, Rust/Cargo 1.98.1. Twelve conditions combine
512/16,384/32,768/65,536 source points with planar, noisy curved and duplicate
curved geometry. The first two geometries have equal target counts; duplicate
sources use a unique noisy curved target with half as many points. The target
is translated by (0.017, -0.011, 0.023). This host reports twelve available CPU
workers. Default budgets are 1/1 below 32,768 points, 2/2 then 4/4 for equal
sides, and 2/1 then 4/2 for the large asymmetric conditions.

The ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-metrics/src/distance_parallel_benchmark.rs`:

```text
cargo test -p spatialrust-metrics --no-default-features --features metrics-distance --release --lib paired_previous_serial_parallel_distances -- --ignored --nocapture --test-threads=1
```

Each condition runs two warmups and twenty rotating rounds of frozen serial
`cloud_distances`, new forced-one-worker execution and the current public API.
Each run verifies all six statistic words for all 792 completed warmup/measured
results. Timers include validation, both tree builds, both query directions,
window allocation/reuse, worker spawn/join, ordered f64 accumulation and
tree/scratch destruction. Fixture construction and exact-word verification are
outside the timer. Returned statistics have no heap allocation to drop.
No other tests or builds launched by this task overlap either run. Desktop
activity and other callers' CPU usage are not controlled.

Table times are per-variant medians. Ratios are medians of twenty paired old/new
times, with 10,000 paired median bootstrap resamples (seed 20261012, sorted
indices 249 and 9749). Above one favors the new path. Intervals describe each
run's samples rather than portability or repeat-run guarantees. Cross-run raw
times are not a controlled comparison of the two staging implementations.

### Initial staging with per-query result allocation

The initial parallel implementation called `nearest_one` for every row. All
six large-condition paired medians exceeded one, but the 32,768 versus
16,384-point asymmetric case did not establish improvement. This survey is
retained in full. It motivated reusing the existing one-neighbor query buffer
inside each band; the final implementation was then retested and remeasured.

| Points A | Geometry | Points B | Workers A/B | Old / forced one / public ms | Old / forced one (95% CI) | Old / public (95% CI) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 512 | plane | 512 | 1/1 | 0.247 / 0.246 / 0.246 | 1.001 (0.999–1.004) | 1.004 (0.992–1.007) |
| 512 | noisy curved | 512 | 1/1 | 0.305 / 0.301 / 0.299 | 1.033 (0.995–1.089) | 1.002 (0.986–1.058) |
| 512 | duplicate / unique curved | 256 | 1/1 | 0.255 / 0.294 / 0.257 | 0.983 (0.940–0.998) | 1.005 (0.994–1.016) |
| 16,384 | plane | 16,384 | 1/1 | 14.274 / 13.800 / 14.125 | 1.016 (0.991–1.038) | 1.008 (0.991–1.034) |
| 16,384 | noisy curved | 16,384 | 1/1 | 12.463 / 12.375 / 12.367 | 1.016 (0.992–1.024) | 1.008 (0.992–1.025) |
| 16,384 | duplicate / unique curved | 8,192 | 1/1 | 9.219 / 8.908 / 8.850 | 1.028 (1.006–1.043) | 1.021 (0.999–1.036) |
| 32,768 | plane | 32,768 | 2/2 | 25.596 / 25.075 / 20.888 | 1.009 (0.989–1.030) | 1.218 (1.129–1.368) |
| 32,768 | noisy curved | 32,768 | 2/2 | 25.434 / 25.479 / 23.395 | 1.006 (0.978–1.020) | 1.099 (1.065–1.205) |
| 32,768 | duplicate / unique curved | 16,384 | 2/1 | 18.400 / 18.457 / 18.205 | 0.996 (0.989–1.013) | 1.014 (0.990–1.095) |
| 65,536 | plane | 65,536 | 4/4 | 53.005 / 54.227 / 33.735 | 0.995 (0.974–1.002) | 1.567 (1.537–1.630) |
| 65,536 | noisy curved | 65,536 | 4/4 | 53.464 / 53.495 / 32.933 | 0.994 (0.962–1.036) | 1.641 (1.593–1.699) |
| 65,536 | duplicate / unique curved | 32,768 | 4/2 | 38.205 / 38.278 / 28.593 | 1.001 (0.989–1.018) | 1.373 (1.329–1.433) |

### Final staging with per-band query-buffer reuse

| Points A | Geometry | Points B | Workers A/B | Old / forced one / public ms | Old / forced one (95% CI) | Old / public (95% CI) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 512 | plane | 512 | 1/1 | 0.265 / 0.261 / 0.262 | 1.011 (1.004–1.105) | 1.011 (0.979–1.105) |
| 512 | noisy curved | 512 | 1/1 | 0.285 / 0.281 / 0.287 | 1.007 (0.928–1.221) | 1.008 (0.955–1.107) |
| 512 | duplicate / unique curved | 256 | 1/1 | 0.216 / 0.217 / 0.202 | 1.003 (0.938–1.082) | 1.000 (0.991–1.079) |
| 16,384 | plane | 16,384 | 1/1 | 12.376 / 12.382 / 12.528 | 0.996 (0.987–1.011) | 0.996 (0.979–1.004) |
| 16,384 | noisy curved | 16,384 | 1/1 | 12.335 / 12.457 / 12.376 | 0.995 (0.980–1.009) | 0.994 (0.987–1.003) |
| 16,384 | duplicate / unique curved | 8,192 | 1/1 | 9.020 / 8.901 / 9.009 | 1.006 (0.997–1.020) | 0.995 (0.976–1.011) |
| 32,768 | plane | 32,768 | 2/2 | 25.387 / 25.299 / 18.322 | 1.006 (0.996–1.021) | 1.390 (1.302–1.607) |
| 32,768 | noisy curved | 32,768 | 2/2 | 25.420 / 25.158 / 19.657 | 1.009 (0.988–1.032) | 1.305 (1.256–1.386) |
| 32,768 | duplicate / unique curved | 16,384 | 2/1 | 18.472 / 18.253 / 16.306 | 1.013 (1.003–1.023) | 1.138 (1.118–1.163) |
| 65,536 | plane | 65,536 | 4/4 | 55.003 / 57.278 / 31.419 | 1.004 (0.984–1.032) | 1.804 (1.763–1.978) |
| 65,536 | noisy curved | 65,536 | 4/4 | 53.961 / 53.467 / 29.345 | 1.006 (0.995–1.038) | 1.849 (1.817–1.897) |
| 65,536 | duplicate / unique curved | 32,768 | 4/2 | 37.726 / 37.602 / 24.518 | 1.001 (0.994–1.020) | 1.548 (1.523–1.683) |

In the final run all six parallel public conditions favor staging, with paired
median speedups 1.138–1.849 and interval lower bounds above one. All six final
one-worker public intervals include one. The initial 512/256-point forced-one
control's adverse ratio 0.983 (0.940–0.998) is retained above. No small-input
neutrality, fixed speedup or controlled cross-run improvement is claimed.
Serial KD-tree construction and ordered accumulation still limit scaling; the
existing conservative worker policy is retained. The measurements cover the
complete host-returning metric API, not GPU execution or a registration pipeline.

Raw samples, summaries, analysis and the initial implementation snapshot remain
private ignored files under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-distances-*`.
Published measurements describe execution conditions without a local adapter
model name. GPU metrics, parallel tree construction and parallel reassociation
of distance sums are outside this slice.
