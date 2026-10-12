# Allocation-free scalar KD-tree search

## Scope and compatibility

`KdTree::nearest_one` previously called `nearest_k(..., 1)` and discarded its
one-element result vector. The new path holds one `Option<Neighbor>` on the
stack and traverses the existing tree directly. This removes per-query heap
scratch in scalar callers, including the existing serial cloud-distance loop.
It adds no dependency, public signature, unsafe code, worker or device transfer.
Tree construction, bounded queries and general k-NN APIs are unchanged.

The implementation is in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/kdtree.rs`.
Leaf visitation, f32 subtraction/multiplication/addition order, near-side choice
and strict far-side pruning preserve the old k=1 algorithm. Rejecting candidates
with `>=` preserves first-visited finite ties and the old replacement behavior
when either distance is NaN. Replacing this with a positive `<` test would change
results. Nonfinite queries retain their previous returned-neighbor behavior;
the bounded-nearest API's finite-query validation is not introduced here.
Missing neighbors remain distinct from a returned NaN or infinite distance.
An absent incumbent always permits the far branch, as before.

## Correctness and builds

Four new native tests are in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/kdtree_nearest_tests.rs`:

- 410,151 exhaustive scalar queries over five symbols through six-point clouds,
  all coordinate axes and seven query values. Signed zero, NaN payloads and both
  infinities compare exact point IDs and f32 distance words with unchanged k=1
  queries, including empty and overflowing cases.
- 18,486 frozen/buffered comparisons over 117 trees, nine geometries, leaf/split
  boundaries through 1,025 points, deterministic finite queries, subnormals,
  extrema and nonfinite values on every axis.
- Independent known expectations for finite ties, all-infinite distances,
  NaN queries, finite-to-NaN-to-finite replacement, final NaN candidates, distance
  overflow and empty trees.
- Four concurrent readers perform 8,192 scalar queries against one shared tree
  and compare exact words with k=1 results.

The existing frozen module is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/tests/fixtures/kdtree_serial.rs`.
Its production implementation was verified against source
`2ba7b6575f8234450dbbc54ed325e32158aa65e1`, omitting the original tests and
appending only the two existing read-only test observers. Its scalar/k-NN
methods, traversal and accumulator were also verified unchanged in direct
parent `2fbc43e10cfc7664a5aab43da7b47b09ab41da26`. The older constructor is
outside query timers; every measured tree layout matches the current tree
before timing. No constructor improvement is credited to this slice.

Minimal search passes 26 tests; all-feature search passes 39. Both configurations
have two ignored native diagnostics. The new facade smoke passes. Strict
all-target Clippy passes for minimal/all-feature search and the facade smoke.
Feature-off search and all-feature search/minimal facade WASM compilation pass.
WASM browser execution or performance is not established. Downstream tests pass:
features 23, filtering 40, metrics 8, registration 35, and Euclidean/DBSCAN/
region-growing segmentation 9, totaling 115. Existing tree-layout, partition,
graph and bounded-query regressions also pass. Formatting/diff checks pass.

```text
cargo test -p spatialrust-search --no-default-features --features search-kdtree --lib -- --nocapture
cargo test -p spatialrust-search --all-features --lib -- --nocapture
cargo test -p spatialrust --no-default-features --features search-kdtree --test kdtree_scalar_nearest_smoke
cargo clippy -p spatialrust-search --no-default-features --features search-kdtree --all-targets -- -D warnings
cargo clippy -p spatialrust-search --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features search-kdtree --test kdtree_scalar_nearest_smoke -- -D warnings
cargo check -p spatialrust-search --no-default-features
cargo check -p spatialrust-search --all-features --target wasm32-unknown-unknown
cargo check -p spatialrust --no-default-features --features search-kdtree --target wasm32-unknown-unknown
cargo test -p spatialrust-features -p spatialrust-filtering -p spatialrust-registration -p spatialrust-metrics --all-features --lib -- --test-threads=1
cargo test -p spatialrust-segmentation --no-default-features --features segment-euclidean,segment-dbscan,segment-region-growing --lib -- --test-threads=1
cargo fmt --all --check
git diff --check
```

## Completed scalar-query measurement

Release native Windows, Rust/Cargo 1.98.1, serial CPU queries. Twelve conditions
combine 512/4,096/32,768 target points with coincident points, noisy curved grids,
sorted lines and mixed-axis hash coordinates. Each batch has 8,192 deterministic
queries. Coincident queries lie exactly on their common point; other geometries
alternate exact points with offsets `[0.002, -0.001, 0.004]`. Query indices step
by 37 modulo the target length. These conditions do not cover every search
distribution, particularly distant queries into coincident clouds.

The native ignored diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/kdtree_nearest_benchmark.rs`:

```text
cargo test -p spatialrust-search --no-default-features --features search-kdtree --release --lib paired_previous_scalar_nearest -- --ignored --nocapture --test-threads=1
```

Each condition has two warmups per variant followed by twenty alternating
paired rounds. Timers include all scalar queries and writing their returned
optional neighbors to a preallocated result array. Tree construction, fixtures,
output allocation, result conversion/checks and destruction are outside timers.
All 528 completed timed/warmup batches per run are checked for exact indices and
distance words. These are completed query measurements, not registration,
cloud-metric pipeline, GPU, construction or concurrent-throughput measurements.

Two surveys use the identical implementation and conditions. No tests/builds
launched by this task overlap either survey; desktop activity and other callers'
CPU usage are uncontrolled. The repeat survey below reports per-variant median
batch milliseconds and the median of twenty paired old/current ratios. A ratio
above one favors the new path. Intervals use 10,000 paired median bootstrap
resamples, seed 20261012 and sorted indices 249/9749; they characterize this run's
samples rather than portability. Cross-run raw times are not a controlled
comparison.

| Points | Geometry | Old ms | Current ms | Old/current paired ratio (95% CI) |
| ---: | --- | ---: | ---: | ---: |
| 512 | coincident | 1.100 | 0.487 | 2.259 (2.258–2.274) |
| 512 | noisy curved | 1.725 | 0.949 | 1.850 (1.822–1.912) |
| 512 | sorted line | 1.352 | 0.567 | 2.385 (2.378–2.400) |
| 512 | mixed axes | 1.518 | 0.830 | 1.834 (1.803–1.861) |
| 4,096 | coincident | 1.250 | 0.638 | 2.013 (1.883–2.122) |
| 4,096 | noisy curved | 2.237 | 1.470 | 1.523 (1.518–1.530) |
| 4,096 | sorted line | 1.478 | 0.684 | 2.162 (2.159–2.167) |
| 4,096 | mixed axes | 1.940 | 1.313 | 1.513 (1.492–1.519) |
| 32,768 | coincident | 1.591 | 0.854 | 1.866 (1.858–1.895) |
| 32,768 | noisy curved | 2.803 | 1.874 | 1.485 (1.465–1.495) |
| 32,768 | sorted line | 1.604 | 0.820 | 1.963 (1.953–1.969) |
| 32,768 | mixed axes | 2.678 | 2.017 | 1.335 (1.320–1.339) |

All twelve conditions favor the new scalar path in both surveys. Repeat paired
median speedups range 1.335–2.385; initial ratios range 1.349–2.397. Every interval
lower bound is above one in both surveys. No fixed speedup or universal latency
guarantee is claimed. General k-NN worker buffers remain unchanged.

Raw logs, samples and analysis remain private ignored files under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/scalar-nearest-*`.
Published conditions omit the local adapter model. Workflows/check names remain
unchanged; all 114 named CI checks gate merge of the exact source commit.
