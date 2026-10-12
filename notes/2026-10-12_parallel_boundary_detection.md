# Bounded CPU boundary detection

Date: 2026-10-12. This named slice follows
[PR #147](https://github.com/rsasaki0109/SpatialRust/pull/147), source
`149e9ed028ce0de9ac8aaf8e276022f6227ff747`.

## Result and execution boundary

Tangent-plane boundary detection previously queried and evaluated every point
on the caller. Under `feature-boundary`, `BoundaryDetector::boundary_mask` now
shares one caller-built KD-tree across bounded scoped CPU workers. Workers write
disjoint contiguous slices directly in the final mask. Each worker reuses one
angle vector across its points; radius-neighbor vectors are limited to the
current query. The scope joins before the mask is returned or `detect` gathers
selected points. There is no worker result vector or mask-merge buffer.

The implementation is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/src/boundary.rs`.
It reuses `spatialrust-search::parallel_worker_count`: one worker below 4,096
points; otherwise at most available CPU parallelism and
`max(1, floor(points / 16,384))` useful chunks. With default chunking, fewer than
32,768 points retain serial execution. One-worker calls run directly on the
caller. Worker injection is private and test-only; no public worker setting,
dependency, feature flag, device transfer or global mutable state is added.

Unsorted radius-neighbor order, self exclusion, tangent basis, projected-distance
cutoff, `atan2`, total angle sorting, wrap-around gap and threshold comparison
retain their original arithmetic. No floating-point reduction spans workers.
Radius validation and the empty-cloud shortcut retain their order. This slice
does not add normalization or rejection of nonunit/nonfinite normals or change
the existing behavior for unusual thresholds or zero minimum support.
`detect` still gathers boundary points in ascending original index order, with
the complete schema, metadata and all six buffer types unchanged.

Mask and KD-tree storage scale with point count. Each active worker holds a
radius-query neighbor vector and retains the capacity of its largest angle
vector until its band ends. A large radius may include the whole cloud; worker
limits are not a scratch-byte budget. Concurrent callers each have their own
worker budget, rather than a process-wide CPU cap. There is no all-query
neighbor table or pairwise distance matrix.

## Correctness and feature validation

The frozen parent module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/tests/fixtures/boundary_serial.rs`.
It matches source 149e9ed after line-ending/trailing-newline normalization, with
only the original test module omitted. It is compiled solely in native test
builds. Production tangent-basis, angle-gap and attribute-gather helpers are
also verified unchanged against that source.

Four tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/src/boundary_parallel_tests.rs`
cover:

- **1,081,992 exact boundary decisions** against the frozen serial public API:
  0/1/17/4,095/4,096/4,103/32,771 points, planar/curved/duplicate curved geometry,
  radii 0.075/0.16, forced 1/2/6 workers and public staging. The large cases cross
  real multi-worker staging and include uneven final bands.
- An independent complete 17-by-17 planar rim expectation, isolated/coincident/
  collinear support, signed zero, zero/nonunit/nonfinite normals, tangent-basis
  branch cases, zero/oversized minimum support and unusual/NaN angle thresholds.
- The public detector's exact selected indices and every gathered F32/F64/U8/
  U16/U32/I32 field, renamed position/normal semantics, metadata, signed-zero
  words and NaN payload words on a 32,771-point cloud.
- Frozen error strings for invalid radii and missing normals; empty clouds
  without normals retain the shortcut. Zero/oversized private worker requests
  on empty/singleton clouds remain safe. Concurrent calls preserve independent
  results on shared borrowed input.

Minimal `feature-boundary` tests pass: seven native tests and one ignored
benchmark. All-feature tests pass: nineteen native tests and one ignored
benchmark, including ISS, CPU/GPU normal estimation and orientation. One facade
API smoke passes. Minimal/all-feature all-target strict Clippy and facade smoke
Clippy pass. Crate and facade WASM compilation passes; browser execution or
threading performance is not established. Formatting and diff checks pass.
No workflow or check names change; all 114 named CI checks gate merge.

```text
cargo test -p spatialrust-features --no-default-features --features feature-boundary --lib -- --test-threads=1 --nocapture
cargo test -p spatialrust-features --all-features --tests -- --test-threads=1
cargo test -p spatialrust --no-default-features --features feature-boundary --test boundary_parallel_smoke
cargo clippy -p spatialrust-features --no-default-features --features feature-boundary --all-targets -- -D warnings
cargo clippy -p spatialrust-features --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features feature-boundary --test boundary_parallel_smoke -- -D warnings
cargo check -p spatialrust-features --target wasm32-unknown-unknown --no-default-features --features feature-boundary
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features feature-boundary
cargo fmt --all --check
git diff --check
```

## Completed mask-API measurement

Release native Windows, Rust/Cargo 1.98.1. Twelve conditions combine
512/16,384/32,768/65,536 points with planar, curved and duplicate curved inputs.
Radius is 0.11 for the plane and 0.16 for the other two fixtures, with minimum
support three and the default 90-degree angle threshold. This machine reports
12 available CPU workers; default policy uses 1/1/2/4 workers respectively.
No other tests or builds launched by this task overlap the performance run.
Desktop activity and other callers' CPU usage are not controlled.

The ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-features/src/boundary_parallel_benchmark.rs`:

```text
cargo test -p spatialrust-features --no-default-features --features feature-boundary --release --lib paired_previous_serial_parallel_boundary -- --ignored --nocapture --test-threads=1
```

Each condition runs two warmups and twenty rotating rounds of the frozen serial
public API, the new forced-one-worker path and the current public API. All 792
completed warmup/measured masks are checked against frozen decisions. Timers
include validation, KD-tree construction, mask allocation, angle-gap evaluation,
worker spawn/join and scratch/tree destruction. Fixture construction, mask
verification and returned mask destruction are outside the timer. These timings
cover `boundary_mask`; gathering the selected cloud in `detect` is not timed.

Table times are per-variant medians. Ratios are medians of twenty paired old/new
times, with 10,000 paired median bootstrap resamples (seed 20261012, sorted
indices 249 and 9749). Above one favors the new path. Intervals describe this
run's samples, rather than portability or repeat-run guarantees.

| Points | Geometry | Workers | Old / forced one / public ms | Old / forced one (95% CI) | Old / public (95% CI) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | plane | 1 | 0.613 / 0.605 / 0.601 | 1.013 (0.991–1.039) | 1.013 (0.991–1.040) |
| 512 | curved | 1 | 1.088 / 1.069 / 1.173 | 1.006 (0.989–1.048) | 0.997 (0.962–1.031) |
| 512 | duplicate curved | 1 | 0.867 / 0.864 / 0.860 | 0.999 (0.981–1.011) | 1.001 (0.968–1.027) |
| 16,384 | plane | 1 | 36.586 / 35.827 / 33.581 | 1.041 (0.890–1.130) | 1.005 (0.933–1.108) |
| 16,384 | curved | 1 | 90.785 / 87.350 / 90.829 | 1.071 (0.970–1.132) | 0.930 (0.899–1.000) |
| 16,384 | duplicate curved | 1 | 133.775 / 132.460 / 131.445 | 1.010 (0.967–1.122) | 1.028 (0.943–1.124) |
| 32,768 | plane | 2 | 76.877 / 74.822 / 45.343 | 1.029 (0.900–1.172) | 1.695 (1.481–1.830) |
| 32,768 | curved | 2 | 108.790 / 117.129 / 69.236 | 0.986 (0.859–1.100) | 1.642 (1.401–1.730) |
| 32,768 | duplicate curved | 2 | 139.640 / 137.206 / 96.002 | 0.994 (0.966–1.029) | 1.454 (1.346–1.643) |
| 65,536 | plane | 4 | 114.198 / 118.965 / 45.188 | 0.983 (0.881–1.032) | 2.483 (2.206–2.692) |
| 65,536 | curved | 4 | 215.800 / 215.576 / 79.596 | 1.000 (0.929–1.091) | 2.663 (2.516–3.071) |
| 65,536 | duplicate curved | 4 | 266.975 / 282.040 / 120.414 | 0.993 (0.957–1.019) | 2.469 (2.197–2.757) |

All six two/four-worker conditions favor public staging, with paired median
speedups 1.454–2.663 and interval lower bounds above one. Forced-one-worker
intervals include one in every condition. The 16,384-point curved public
control has an adverse paired ratio of 0.930 (0.899–1.000), about 7% slower;
this result is retained and small-input neutrality is not claimed. The other
five one-worker public intervals include one. Timing variation is visible
throughout this uncontrolled desktop run; the bootstrap intervals do not
capture all sources of systematic variation. The existing conservative worker
policy is retained, with no promise of a fixed speedup across machines,
neighborhood densities or concurrent workloads.

Raw samples and analysis are private ignored files under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-boundary-*`.
Published measurements describe execution conditions without a local adapter
model name. GPU boundary detection, ISS scheduling and parallel attribute
gathering are outside this slice.
