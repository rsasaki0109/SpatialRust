# Exact KD-tree minimum pivots

Date: 2026-10-12. This named slice follows merged
[PR #150](https://github.com/rsasaki0109/SpatialRust/pull/150), source
`2ba7b6575f8234450dbbc54ed325e32158aa65e1`, merge
`48df4757f9fd5e05d6412b0a2c9beea688beee45`.

## Result and correctness argument

The median selector's original partition scanned the complete active range even
when no element could compare below its pivot. A large equal-coordinate bucket
then advanced its left boundary one point at a time, repeating quadratic work.
This was visible while validating the preceding distance-metric slice.

The implementation is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/kdtree.rs`.
Axis selection already computes three minima; it now returns the selected
axis's minimum alongside that axis. Median selection carries this lower bound
without another coordinate scan or allocation.

If a pivot equals the lower bound, no remaining ordered coordinate can be
strictly smaller. No coordinate compares below a NaN pivot either. The parent
partition therefore leaves its store index unchanged, performs its midpoint-to-
left pivot swap, and finishes with a self-swap. The new path reproduces that
pivot swap directly and returns the same left index. It retains every original
swap across repeated pivots, rather than treating equal point identities as
interchangeable.

When selection moves right past a finite pivot, all retained finite values did
not compare below it, so that pivot becomes a valid lower bound. A NaN pivot
provides no stronger numeric bound and leaves the prior bound unchanged. Left
narrowing retains the existing lower bound. General partitions are unchanged.
Axis comparisons/arithmetic, median positions, leaf size, node construction,
coordinate copies and query methods also keep their original behavior.

Point permutations, root/node indices, split float words and ranges are
preserved. Nearest ties and unsorted radius-neighbor order remain stable for
downstream floating-point calculations. Construction remains serial; storage,
dependencies, features, public signatures and transfers are unchanged. Equal/
minimum-bucket selection removes repeated full scans, but this is not a general
worst-case complexity guarantee for quickselect on arbitrary inputs.

## Validation

The frozen parent module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/tests/fixtures/kdtree_serial.rs`.
Its implementation matches source 2ba7b65 after line-ending/trailing-newline
normalization, with only the original test module omitted. Two small native
test-only observers append read-only tree snapshots and a selection wrapper;
they do not alter the frozen algorithm or expose production APIs. The snapshot
contains the root, complete point permutation and six exact u32 words per node,
including `split.to_bits()`. Query code and general partition code are verified
unchanged against the parent.

Four tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/kdtree_partition_tests.rs`
cover:

- **191,916 exact selection permutations** against frozen selection, exhausting
  three ternary alphabets through seven elements: ordered values/NaNs, signed
  zeros/+infinity, and negative/positive infinity/NaNs. Every nth position and
  axis is checked with reversed input permutations and outside-range guards.
- **117 exact complete tree layouts and 11,700 ordered query results** across
  0/1/15/16/17/31/32/33/127/128/129/257/1,025 points and coincident, outlier,
  discrete, curved, sorted, NaN-payload, signed-zero, infinity and mixed-axis
  inputs. Sorted/unsorted k-NN, radius and bounded-nearest outputs retain both
  point IDs and distance words, including nonfinite queries and unusual bounds.
- An independent 65,537-point coincident cloud retains each point ID exactly
  once; squared distance, radius membership, k-NN count and gate rejection meet
  known expectations without building the slow parent at that size.
- Concurrent duplicate-heavy tree builds retain frozen snapshots.

Minimal search tests pass: 22 native tests and one ignored diagnostic. All-feature
search tests pass: 35 native tests and one ignored diagnostic, including graph
order and bounded parallel queries. The facade smoke passes. Minimal/all-feature
all-target strict Clippy and facade smoke Clippy pass. Search without features
and search/facade WASM compilation pass; browser execution is not established.
Downstream regressions pass: features 23, filtering 40, metrics 8, registration
35 and Euclidean/DBSCAN/region-growing segmentation 9 tests, totaling **115**.
Formatting/diff checks pass. No workflow or check names change; all 114 named
CI checks gate merge.

```text
cargo test -p spatialrust-search --no-default-features --features search-kdtree --lib -- --test-threads=1 --nocapture
cargo test -p spatialrust-search --all-features --lib -- --test-threads=1 --nocapture
cargo clippy -p spatialrust-search --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-search --no-default-features --features search-kdtree --all-targets -- -D warnings
cargo test -p spatialrust --no-default-features --features search-kdtree --test kdtree_minimum_pivots_smoke
cargo clippy -p spatialrust --no-default-features --features search-kdtree --test kdtree_minimum_pivots_smoke -- -D warnings
cargo check -p spatialrust-search --no-default-features
cargo check -p spatialrust-search --target wasm32-unknown-unknown --all-features
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features search-kdtree
cargo test -p spatialrust-features -p spatialrust-filtering -p spatialrust-registration -p spatialrust-metrics --all-features --lib -- --test-threads=1
cargo test -p spatialrust-segmentation --no-default-features --features segment-euclidean,segment-dbscan,segment-region-growing --lib -- --test-threads=1
cargo fmt --all --check
git diff --check
```

## Complete constructor measurement

Release native Windows, Rust/Cargo 1.98.1. Sixteen conditions combine
512/4,096/16,384/32,768 points with coincident points, coincident points plus one
large outlier, noisy curved grids and mixed-axis hash coordinates. Both old and
new constructors remain serial. All coordinates are deterministic synthetic
fixtures. No tests/builds launched by this task overlap either survey; desktop
activity and other callers' CPU usage are not controlled.

The ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/kdtree_partition_benchmark.rs`:

```text
cargo test -p spatialrust-search --no-default-features --features search-kdtree --release --lib paired_previous_minimum_pivot_kdtree -- --ignored --nocapture --test-threads=1
```

Each condition runs two warmups per variant and twenty alternating paired
rounds. All 704 completed warmup/measured trees are checked against the full
frozen tree snapshot. Timers include point ordering, node allocation,
axis/partition work and coordinate copies. Fixture creation, snapshot checks,
query work and returned tree destruction are outside the timer. These timings
cover complete `KdTree::from_slices` construction, not query or pipeline speed.

Times are per-variant medians. Ratios are medians of twenty paired old/current
times, with 10,000 paired median bootstrap resamples (seed 20261012, sorted
indices 249 and 9749). Above one favors the new path. Intervals describe this
run's samples rather than portability or repeat-run guarantees.

| Points | Geometry | Old ms | Current ms | Old/current paired ratio (95% CI) |
| ---: | --- | ---: | ---: | ---: |
| 512 | coincident | 0.200 | 0.016 | 9.816 (7.343–10.634) |
| 512 | coincident plus outlier | 0.160 | 0.017 | 9.077 (8.793–9.786) |
| 512 | noisy curved | 0.027 | 0.024 | 1.109 (1.077–1.215) |
| 512 | mixed axes | 0.035 | 0.036 | 1.026 (0.961–1.212) |
| 4,096 | coincident | 9.041 | 0.235 | 38.508 (37.391–39.122) |
| 4,096 | coincident plus outlier | 8.561 | 0.229 | 37.191 (35.513–38.050) |
| 4,096 | noisy curved | 0.391 | 0.388 | 1.016 (0.949–1.077) |
| 4,096 | mixed axes | 0.518 | 0.513 | 1.008 (0.954–1.040) |
| 16,384 | coincident | 127.758 | 1.110 | 116.367 (109.102–125.727) |
| 16,384 | coincident plus outlier | 135.686 | 1.215 | 113.756 (92.746–125.197) |
| 16,384 | noisy curved | 1.788 | 1.762 | 1.045 (0.980–1.070) |
| 16,384 | mixed axes | 2.551 | 2.574 | 1.013 (0.978–1.022) |
| 32,768 | coincident | 514.647 | 2.236 | 229.939 (218.760–237.847) |
| 32,768 | coincident plus outlier | 512.461 | 2.211 | 231.153 (225.144–235.964) |
| 32,768 | noisy curved | 3.986 | 3.847 | 1.012 (0.986–1.052) |
| 32,768 | mixed axes | 5.632 | 5.505 | 1.017 (0.975–1.039) |

All eight duplicate-heavy conditions favor the new constructor, with paired
median speedups 9.077–231.153 and interval lower bounds above one. At 32,768
points the two duplicate-heavy fixtures drop from about 512–515 ms to 2.2 ms,
with paired ratios around 230. All four mixed-axis intervals include one;
three curved-grid intervals include one and the 512-point curve favors the
new path. No universal neutral-overhead or fixed-speedup guarantee is claimed.
The gains concern removal of a degenerate construction bottleneck, not search
latency, GPU execution or end-to-end registration. General partitions and the
original point ordering remain important compatibility constraints.

An initial twelve-condition survey omitted mixed-axis controls; its complete
samples/summary are retained alongside the final sixteen-condition survey.
The constructor implementation is identical in both surveys. Cross-run raw times
are not a controlled comparison. Raw samples and analysis remain private ignored
files under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/kdtree-minimum-pivots-*`.
Published measurements describe conditions without a local adapter model name.
