# Bounded parallel CPU neighborhood graphs

Date: 2026-10-12. Named slice: bounded CPU neighborhood graphs.

## Result and contract

`knn_graph_parallel` and `radius_graph_parallel` are additive under
`spatialrust-search` features `search-graph,parallel`. The facade exposes them
with `search-graph,search-parallel`. Both build one KD-tree on the caller and
share its immutable storage across bounded scoped workers. Their complete
`NeighborGraph` (node count and ordered edges) matches the serial API, including
KD-tree tie selection, duplicate coordinates and radius traversal order.
Existing `knn_graph` and `radius_graph` retain their dedicated serial loops.

The implementation reuses the previously tested chunk-query dispatcher, now
private-generic over its output element type. Graph callbacks emit `[u32; 2]`
edges directly from each per-point neighbor list. There is no all-query tagged
neighbor table, copied tree per worker or distance matrix. Graph query chunks
contain 256 points; calls with fewer than 4,096 points or one available worker
keep one caller range. Otherwise workers are capped by available CPU parallelism
and chunk count, with contiguous balanced chunk bands and ordered joins. These
are per-call limits; concurrent calls have independent budgets.

Handle/range metadata remains worker-sized. Per-query neighbor vectors and
live local/final edge lists remain result-sized. Ordered merging can hold local
and final edge storage concurrently; this change does not impose a graph byte
budget. Dense radius graphs can still contain quadratically many edges.

Both serial and parallel APIs reject point counts exceeding u32::MAX before
KD-tree construction or graph allocation. Large k searches saturate/cap k+1 at
cloud size and cap the initial reservation by cloud size; its multiplication is
checked. A failed reservation product starts with an empty vector instead of
wrapping the product. This addresses arithmetic overflow, not allocator failure
or the potentially quadratic size of a complete graph. k=0 and nonpositive/NaN
radii retain errors; positive infinite radii remain accepted. The existing
positions requirement, including errors for a schema-only empty cloud without
column buffers, is preserved. A valid empty XYZ cloud returns an empty graph.

No new dependency, production runtime, feature name, unsafe boundary or GPU
transfer is introduced. Criterion remains the existing dev-only dependency.
The two existing graph CI jobs now enable the parallel features as well; the
114 named successful-check inventory remains unchanged.

## Correctness and compilation

Final native search tests: 31 passed with all features; 25 passed with
`--no-default-features --features search-graph`; 11 passed with only `parallel`
and no default backend. The facade's focused parallel-search test target passes
both its existing ordered query test and new ordered graph test.

New graph coverage includes:

- Frozen previous traversal equality for 0/1/17/4,095/4,096/4,103 points, three k
  values and three radii, forced 1/2/6 worker budgets, duplicate and negative
  coordinates, distance ties and a partial last chunk. Public serial APIs also
  match each frozen result exactly. Existing dispatcher tests continue checking
  thread counts, caller fallbacks and exact chunk callback boundaries.
- Independent all-pairs radius membership (including infinity), nearest-distance
  multisets for k=1 and k=8, complete-graph membership at k>=len and usize::MAX,
  exclusion of self edges, valid empty graphs, invalid query arguments,
  missing position columns and pure u32 index limits without enormous allocation.
- Four concurrent public parallel graph callers with exact ordered parity.

Strict all-feature/all-target Clippy, serial-graph-only Clippy and minimal
parallel Clippy pass. `wasm32-unknown-unknown` graph plus parallel compilation
passes; no browser/thread execution was measured or claimed.

Private validation logs are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:
`parallel-graphs-tests.log`, `parallel-graphs-serial-tests.log`,
`parallel-graphs-minimal-tests.log`, `parallel-graphs-facade-tests.log`,
`parallel-graphs-clippy.log`, `parallel-graphs-serial-clippy.log`,
`parallel-graphs-minimal-clippy.log`, `parallel-graphs-wasm.log` and
`parallel-graphs-release-build.log`.

## Completed-API measurement

Release native Windows execution, Rust/Cargo 1.98.1, 12 available CPU workers.
Each condition has two warmups then 20 rotating-order rounds across frozen
previous serial, current serial and public parallel API. Fixtures cover a 3D
integer grid and groups of eight duplicate coordinates, 512/4,096/16,384/65,536
points, k=8/k=32 and radius=1.01: 24 conditions and 480 measured rounds.
Every measured output is compared to the frozen previous graph, including edge
order, after its timer closes.

The timer includes validation, tree construction, graph/query allocation,
searches, worker creation/join and ordered merge through completed API return.
It excludes fixture construction, verification, result destruction and Cargo
compilation. Own tests/builds completed before the final measurement; no other
agent test or benchmark was run concurrently. Timings are local observations,
not cross-machine performance guarantees. The 512-point parallel API uses the
caller, so no parallel speedup is expected there.

The frozen production loops in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/tests/fixtures/graph_serial.rs`
come from a804c23 (the parent source). Imports and the shared output type are
adapted so the standalone old functions can compile; their original k+1 query,
reservation and radius traversal loops remain intact. New pure tests separately
cover oversized k inputs that those old loops cannot handle.

The paired example is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/examples/parallel_graphs.rs`.
Reproduce it with:

```powershell
cargo run -p spatialrust-search --release --features search-graph,parallel --example parallel_graphs
```

A Criterion target for continued comparisons is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/benches/neighbor_graphs.rs`:

```powershell
cargo bench -p spatialrust-search --features search-graph,parallel --bench neighbor_graphs
```

It is compiled by all-target Clippy; the results below come from the rotating
paired example, not from an executed Criterion run. Private raw CSV rows are
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-graphs-paired.log`;
`parallel-graphs-estimates.json`, `parallel-graphs-table.md` and
`summarize-parallel-graphs.py` in the same directory retain absolute ranges and
paired estimates. The table shows separate timing medians and the median of
per-round previous/parallel ratios; ratios need not equal the quotient of the
reported timing medians. Intervals are seeded 10,000-resample paired-median
bootstrap 95% intervals (seed 20261012).

Across the 18 threaded conditions, paired median speedups relative to the
frozen previous loops range from 1.228 to 3.852; their bootstrap lower
bounds exceed one. These are completed-API ratios with tree build included.
The six 512-point conditions show caller-path timings and do not establish
parallel gains. Current serial timings and paired serial ratios are reported
alongside them; no serial speedup is claimed. Variation and observed ranges
remain available in the private JSON. The frozen reference is compiled into the
benchmark example alongside calls to the current library; this compares the
frozen algorithm/API loops, not two separately built historical executables.

| Points | Distribution | Query | Edges | Previous / serial ms | Parallel ms | Previous / parallel, median [95% CI] | Previous / serial ratio |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: |
| 512 | grid | Knn8 | 4,096 | 0.567 / 0.555 | 0.575 | 0.985 [0.971, 0.993] | 1.005 |
| 512 | grid | Knn32 | 16,384 | 1.309 / 1.184 | 1.174 | 0.997 [0.972, 1.043] | 1.013 |
| 512 | grid | Radius | 1,904 | 0.246 / 0.249 | 0.241 | 0.984 [0.976, 1.001] | 1.001 |
| 512 | 8 duplicates | Knn8 | 4,096 | 0.237 / 0.220 | 0.227 | 1.030 [0.951, 1.058] | 0.997 |
| 512 | 8 duplicates | Knn32 | 16,384 | 0.780 / 0.784 | 0.804 | 0.982 [0.929, 1.048] | 1.040 |
| 512 | 8 duplicates | Radius | 3,584 | 0.333 / 0.330 | 0.338 | 0.977 [0.910, 1.084] | 1.042 |
| 4,096 | grid | Knn8 | 32,768 | 4.660 / 4.664 | 2.599 | 1.743 [1.557, 1.860] | 0.957 |
| 4,096 | grid | Knn32 | 131,072 | 17.411 / 18.827 | 7.231 | 2.433 [2.227, 2.595] | 0.953 |
| 4,096 | grid | Radius | 16,128 | 4.234 / 4.346 | 2.669 | 1.597 [1.245, 1.782] | 1.026 |
| 4,096 | 8 duplicates | Knn8 | 32,768 | 4.156 / 4.218 | 2.438 | 1.752 [1.535, 1.957] | 1.035 |
| 4,096 | 8 duplicates | Knn32 | 131,072 | 13.207 / 13.480 | 5.165 | 2.640 [2.511, 2.971] | 0.991 |
| 4,096 | 8 duplicates | Radius | 28,672 | 2.460 / 2.481 | 2.035 | 1.228 [1.138, 1.272] | 0.979 |
| 16,384 | grid | Knn8 | 131,072 | 27.155 / 26.773 | 10.194 | 2.483 [2.203, 2.675] | 0.969 |
| 16,384 | grid | Knn32 | 524,288 | 86.812 / 83.568 | 24.464 | 3.434 [3.306, 3.578] | 1.024 |
| 16,384 | grid | Radius | 89,088 | 29.975 / 30.657 | 12.930 | 2.340 [1.915, 2.523] | 0.994 |
| 16,384 | 8 duplicates | Knn8 | 131,072 | 20.170 / 19.181 | 8.223 | 2.391 [2.275, 2.675] | 1.032 |
| 16,384 | 8 duplicates | Knn32 | 524,288 | 53.218 / 56.570 | 16.296 | 3.407 [3.181, 3.465] | 0.956 |
| 16,384 | 8 duplicates | Radius | 114,688 | 10.856 / 11.136 | 6.060 | 1.809 [1.740, 2.028] | 0.963 |
| 65,536 | grid | Knn8 | 524,288 | 171.614 / 161.386 | 54.279 | 2.842 [2.615, 3.199] | 0.974 |
| 65,536 | grid | Knn32 | 2,097,152 | 439.662 / 431.053 | 106.762 | 3.852 [3.615, 4.297] | 1.039 |
| 65,536 | grid | Radius | 380,928 | 108.583 / 111.501 | 39.737 | 2.670 [2.546, 2.875] | 1.071 |
| 65,536 | 8 duplicates | Knn8 | 524,288 | 131.026 / 134.971 | 47.804 | 2.581 [2.335, 2.990] | 0.959 |
| 65,536 | 8 duplicates | Knn32 | 2,097,152 | 405.996 / 386.614 | 120.535 | 3.421 [2.833, 3.819] | 1.013 |
| 65,536 | 8 duplicates | Radius | 458,752 | 131.909 / 134.968 | 61.379 | 1.973 [1.767, 2.509] | 0.925 |

The prior CLAHE PR140 is merged at a6c2819 with all 114 checks successful.
This CPU slice changes no image/GPU source and builds on its frozen source
commit a804c23. Remaining GPU contrast geometry work still requires profiling
before choosing any retention policy.
