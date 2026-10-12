# CPU dense-grid row traversal

## Scope and invariant

The public `spatialrust-search::euclidean_cluster_roots` uses a stable dense
counting-sort grid and an iterator over nearby cells. The old iterator visited
up to 27 individual cells per query. The revised iterator combines consecutive
x cells into one CSR interval for each valid y/z row, at most nine intervals.
Query-local clipped y/z bounds also eliminate invalid offset visits on faces,
edges, corners and single-axis grids. Empty intervals advance without yielding.

The cell permutation is unchanged: z, then y, then x, then original point index
within each cell. Combining contiguous CSR ranges therefore preserves every
candidate and accepted neighbor in order, including self exclusion and the
original f32 squared-distance expression and inclusive radius comparison.
Both sequential and existing parallel atomic union-find use this iterator;
minimum-index roots, union math, the 4,096-point threshold and thread scheduling
remain unchanged. There is no new heap allocation, worker, feature, dependency,
unsafe code or device transfer. Iterator state is constant-sized on the stack.

Production source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid.rs`.

For positive-dimension grids within the existing 64,000,000-cell cap, each
dimension fits i32. Clipping keeps x endpoints within `[0, dim_x]`, y within `[0, dim_y)` and
z within `[0, dim_z)`. Thus `row + after_x <= cell_count`, and the final row may
read the existing CSR sentinel. Exhausted iterators remain exhausted.

Grid bounds, stable grid construction, sparse segment labeling and serial /
parallel union code were verified byte-identical to the parent before the
iterator definition. This optimization applies to the dense-grid search API
and its facade. GPU sparse-grid clustering uses a separate segment walker;
the segmentation extractor's CPU path uses KD-tree BFS. Standalone GPU normal /
covariance APIs still build their grids on the CPU. No throughput improvement
for these other APIs is claimed.

## Correctness and source replay

The frozen production prefix at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/tests/fixtures/uniform_grid_previous.rs`
matches the parent uniform-grid module before its existing test module,
with only a terminal blank line trimmed:
`fef58225ff2b292b165396c79ca87f8975ee9d6d`. No baseline algorithm was edited.
It is compiled only into test builds, allowing completed-API comparison with
the original validation, bounds, allocation, counting-sort and union math.

Regression source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid_rows_tests.rs`.

- 162 prescribed grids cover all 1/2/4-cell dimension combinations, three
  binary-exact radii, populated/sparse cells, reversed cell input order,
  duplicate points, exact cell/radius boundaries, and out-of-bounds clamping.
  All **5,976 ordered queries / 56,994 returned neighbor indices** agree with
  the frozen iterator and an independent brute-force distance/key-order oracle.
  Repeated calls after exhaustion return None.
- 200 small cloud/radius combinations cover empty, singleton, duplicate,
  volume, tilted-sheet, separated eight-point-cluster and chain inputs.
  Component roots match an independent relabeling oracle and the frozen API.
- Prescribed valid grids compare NaN, infinity, extreme finite coordinates,
  signed zero and infinite radius behavior against the frozen iterator.
  These probes do not assert validation of nonfinite grid bounds; that existing
  behavior is outside this iterator change. Public unequal-axis-length and
  nonpositive/NaN radius rejection are checked.
- 4,095 / 4,096 / 4,097 / 10,003-point fixtures verify minimum-index roots at
  dispatch and partial-group boundaries. Four concurrent callers each check
  complete independent results. Both minimal serial and all-feature parallel
  builds run these checks.
- A facade regression interleaves 4,097 points across 257 separated groups,
  checking the independently expected minimum root for every original index
  under minimal features and `search-parallel`.

Validation completed on Windows with rustc 1.98.1:

| Check | Result |
| --- | --- |
| Search all-feature tests | 43 passed, 0 failed, 3 ignored timing diagnostics |
| Search minimal tests | 11 passed, 0 failed, 1 ignored timing diagnostic |
| Segmentation `segment-euclidean` tests | 3 passed |
| Facade minimal and `search-parallel` smoke | 1 passed each |
| Search all-feature/all-target and facade Clippy | `-D warnings` passed |
| Minimal search and `search-parallel` facade WASM | Compile checks passed |
| Formatting and diff whitespace | Passed |

WASM checks establish compilation only; no browser thread execution or
performance was measured. Native CPU tests need no GPU adapter.

## Completed public API measurements

The ignored release diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid_rows_benchmark.rs`.
Run it alone with:

```powershell
cargo test -p spatialrust-search --release --no-default-features paired_previous_current_grid_rows -- --ignored --nocapture --test-threads=1
cargo test -p spatialrust-search --release --no-default-features --features parallel paired_previous_current_grid_rows -- --ignored --nocapture --test-threads=1
```

All own correctness/check jobs completed before the timing surveys. Each
condition warms both versions three times, then records 30 alternating-order
pairs. Input generation is outside the timer. Each timed public call includes
validation, grid bounds, grid allocations/counting-sort, neighbor traversal,
union-find, compression and returned root construction. Root equality checking
and result destruction occur after timing. There are no GPU pass timestamps.
The host reports 12 available CPU workers; serial builds use the caller,
parallel builds retain the original caller path below 4,096 points and the
existing worker dispatcher above it. Per-call budgets remain unchanged.

Two final surveys cover 24 conditions each, without changing production source
between them. Tables show median completed milliseconds and geometric mean of
paired `previous/current` ratios. Values greater than one favor the revision.
95% intervals use 10,000 paired log-ratio bootstrap resamples, seed 20261012;
these describe this host/survey and do not remove systematic measurement bias.

At 65,536 points, volume/sheet/cluster ratios reproduce at **1.34–1.51** for
serial builds and **1.16–1.27** with the existing parallel feature, corresponding
to approximately **25–34%** and **14–21%** shorter completed calls respectively.
Serial chains also improve (1.12–1.14). Parallel chains' intervals include one
in both surveys, and several 4,096-point parallel conditions remain uncertain;
no general improvement is established for these cases. Small parallel-feature
calls actually remain serial and must not be described as thread speedups.

A first traversal prototype retained invalid y/z offset visits and regressed
serial chains; it was replaced with the clipped valid-row traversal before
final measurements. The prototype is not shipped. Its private diagnostics
remain available for audit; all published final tables below use the revision.

### Final survey A

| Mode | Points | Distribution | Previous ms | Current ms | Paired ratio [95% interval] |
| --- | ---: | --- | ---: | ---: | --- |
| serial | 257 | volume | 0.1160 | 0.0835 | 1.366 [1.329, 1.399] |
| serial | 257 | tilted_sheet | 0.0713 | 0.0448 | 1.603 [1.566, 1.643] |
| serial | 257 | eight_point_clusters | 0.0513 | 0.0387 | 1.371 [1.320, 1.438] |
| serial | 257 | chain | 0.0254 | 0.0218 | 1.137 [1.097, 1.164] |
| serial | 4,096 | volume | 1.6759 | 1.2295 | 1.344 [1.284, 1.385] |
| serial | 4,096 | tilted_sheet | 1.1382 | 0.6874 | 1.668 [1.605, 1.726] |
| serial | 4,096 | eight_point_clusters | 0.9715 | 0.6991 | 1.421 [1.377, 1.482] |
| serial | 4,096 | chain | 0.4244 | 0.3661 | 1.154 [1.111, 1.217] |
| serial | 65,536 | volume | 26.9991 | 19.7578 | 1.343 [1.322, 1.363] |
| serial | 65,536 | tilted_sheet | 22.6614 | 15.4702 | 1.476 [1.462, 1.491] |
| serial | 65,536 | eight_point_clusters | 14.9846 | 10.0427 | 1.480 [1.449, 1.508] |
| serial | 65,536 | chain | 6.6109 | 5.8383 | 1.124 [1.095, 1.152] |
| parallel | 257 | volume | 0.1154 | 0.0832 | 1.373 [1.319, 1.427] |
| parallel | 257 | tilted_sheet | 0.0713 | 0.0447 | 1.585 [1.551, 1.620] |
| parallel | 257 | eight_point_clusters | 0.0524 | 0.0380 | 1.439 [1.382, 1.521] |
| parallel | 257 | chain | 0.0279 | 0.0217 | 1.278 [1.258, 1.291] |
| parallel | 4,096 | volume | 1.3310 | 1.3060 | 1.006 [0.968, 1.043] |
| parallel | 4,096 | tilted_sheet | 1.1809 | 1.1056 | 1.063 [1.038, 1.091] |
| parallel | 4,096 | eight_point_clusters | 1.1113 | 1.0884 | 1.006 [0.979, 1.033] |
| parallel | 4,096 | chain | 0.9360 | 0.9551 | 0.984 [0.963, 1.005] |
| parallel | 65,536 | volume | 9.7585 | 8.1099 | 1.223 [1.198, 1.249] |
| parallel | 65,536 | tilted_sheet | 11.6473 | 10.1390 | 1.162 [1.145, 1.180] |
| parallel | 65,536 | eight_point_clusters | 6.4807 | 5.1913 | 1.256 [1.228, 1.284] |
| parallel | 65,536 | chain | 3.9720 | 4.0816 | 1.000 [0.978, 1.023] |

### Final survey B

| Mode | Points | Distribution | Previous ms | Current ms | Paired ratio [95% interval] |
| --- | ---: | --- | ---: | ---: | --- |
| serial | 257 | volume | 0.1137 | 0.0815 | 1.346 [1.276, 1.403] |
| serial | 257 | tilted_sheet | 0.0694 | 0.0430 | 1.587 [1.537, 1.633] |
| serial | 257 | eight_point_clusters | 0.0500 | 0.0377 | 1.334 [1.318, 1.354] |
| serial | 257 | chain | 0.0249 | 0.0213 | 1.161 [1.123, 1.191] |
| serial | 4,096 | volume | 1.7164 | 1.2685 | 1.367 [1.343, 1.391] |
| serial | 4,096 | tilted_sheet | 1.1776 | 0.7066 | 1.703 [1.655, 1.766] |
| serial | 4,096 | eight_point_clusters | 0.8767 | 0.6255 | 1.427 [1.366, 1.498] |
| serial | 4,096 | chain | 0.4079 | 0.3538 | 1.145 [1.123, 1.163] |
| serial | 65,536 | volume | 28.4452 | 21.1602 | 1.341 [1.319, 1.364] |
| serial | 65,536 | tilted_sheet | 22.6853 | 15.2728 | 1.470 [1.451, 1.490] |
| serial | 65,536 | eight_point_clusters | 15.5175 | 10.8819 | 1.507 [1.464, 1.562] |
| serial | 65,536 | chain | 6.5291 | 5.8947 | 1.142 [1.095, 1.193] |
| parallel | 257 | volume | 0.1153 | 0.0828 | 1.369 [1.314, 1.408] |
| parallel | 257 | tilted_sheet | 0.0710 | 0.0451 | 1.568 [1.517, 1.611] |
| parallel | 257 | eight_point_clusters | 0.0522 | 0.0381 | 1.357 [1.328, 1.376] |
| parallel | 257 | chain | 0.0286 | 0.0222 | 1.330 [1.249, 1.464] |
| parallel | 4,096 | volume | 1.3590 | 1.3675 | 1.047 [1.004, 1.092] |
| parallel | 4,096 | tilted_sheet | 1.1270 | 1.0831 | 1.038 [1.018, 1.059] |
| parallel | 4,096 | eight_point_clusters | 1.1456 | 1.1050 | 1.043 [1.012, 1.079] |
| parallel | 4,096 | chain | 0.9273 | 0.9413 | 1.009 [0.973, 1.048] |
| parallel | 65,536 | volume | 9.6678 | 8.0216 | 1.235 [1.193, 1.279] |
| parallel | 65,536 | tilted_sheet | 14.9112 | 12.2406 | 1.178 [1.137, 1.219] |
| parallel | 65,536 | eight_point_clusters | 7.5763 | 5.8883 | 1.269 [1.231, 1.307] |
| parallel | 65,536 | chain | 4.3566 | 4.3586 | 0.993 [0.932, 1.055] |

Raw local diagnostics and computed measurements remain ignored under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with the
`cpu-grid-rows-` prefix. Public summaries contain no local GPU model identity.

## Remaining work

This removes repeated CSR-offset traversal; it does not parallelize grid
construction, change union contention or address sparse segment complexity.
CPU grid preparation remains a separate candidate for standalone GPU point
estimator throughput, requiring its own stable-order and memory measurements.
