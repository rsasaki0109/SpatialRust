# Bounded parallel CPU point queries — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Parent: `58d9670d832323b8bd2cc1ed313312317007ac8f`, PR 125,
`codex/parallel-contrast-enhancement`. This is one follow-up slice family;
parent GPU storage and image contrast implementations are unchanged.

## Implementation and contract

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-search\src\parallel_chunked.rs`
previously collected every tensor chunk range and spawned one scoped OS thread
per range once the tensor had at least 4,096 points. A 65,536-point tensor with
chunk size 1 therefore attempted 65,536 thread spawns. The implementation now
limits each invocation to `min(available_parallelism, ceil(points/chunk_size))`
workers, falling back to one when parallelism discovery fails. Inputs below
4,096 points also use one worker. One-worker jobs run directly on the caller.

Original chunk ordinals are divided into contiguous worker bands whose chunk
counts differ by at most one. Ranges are generated lazily, with endpoints
computed from the remaining point count to avoid overflowing oversized chunk
sizes. Each backend callback receives its original tensor chunk boundary;
callbacks within a band run in ascending chunk order. Private result vectors
are joined/appended in band order, preserving query and within-chunk neighbor
order without sorting. All four existing public radius/k-NN functions retain
their signatures. `_into` preserves the caller's existing prefix; empty input
and parallel k-NN with `k=0` remain no-ops.

Range/handle/vector metadata is O(workers), rather than O(chunks). This is **not
a fixed output-memory cap**: output and worker-local vectors remain proportional
to returned neighbors, and radius/k are caller controlled. Threads are still
created and joined per call. Multiple concurrent callers have independent
budgets; there is no process-wide pool or global concurrency cap. No production
runtime dependency, core change, unsafe code or CPU/GPU transfer is introduced.
Criterion is a dev-only dependency behind a required-feature benchmark target.

## Correctness and build validation

Five parallel-query unit tests cover:

- Explicit 1/2/6-worker budgets, zero reported availability, threshold edges,
  partial chunks and usize-sized policy arithmetic.
- A recording generic backend with exact callback boundaries, ordered callbacks
  within each worker, balanced chunk counts, actual worker thread identities,
  descending neighbor order and retained output prefixes. Chunk sizes are
  1/7/512/full-length/usize::MAX over 4,103 points.
- Both allocating and `_into` public wrappers with a generic backend and tiny
  chunks, plus empty, below-threshold, single-chunk and zero-k behavior.
- KD-tree radius and k-NN parity **without sorting** for 1/2/6-worker dispatch,
  negative coordinates, duplicates, equal-distance neighbors and odd tails.

Existing serial tensor tests use the always-available brute-force backend;
KD-tree-specific tests/imports are feature gated. The staging range-coverage
fixture now requests six workers explicitly instead of requiring a machine
with more than one available CPU. Production staging policy is unchanged.
A facade integration test checks public radius/k-NN functions and append order
on 4,103 one-point chunks.

Validation passed:

- `cargo test -p spatialrust-search --all-features`: 27 unit tests.
- `cargo test -p spatialrust-search --no-default-features --features parallel`:
  11 unit tests.
- `cargo test -p spatialrust-search`: 18 unit tests.
- `cargo test -p spatialrust --features search-parallel`: three existing smoke
  tests; the subsequently added `--test parallel_search_smoke` adds one passing
  public-query test.
- Full and minimal search Clippy on all targets, and facade Clippy on tests,
  each with `-D warnings`.
- Minimal parallel and all-feature WASM library compilation with `-D warnings`;
  this checks compilation, not browser thread execution/performance.
- Minimal parallel rustdoc with `RUSTDOCFLAGS='-D warnings'`, rustfmt and diff
  checks. CI YAML parsed and all three new profile names are unique.

CI now executes search parallel, minimal parallel and facade parallel profiles;
the existing all-feature job only compiles tests. The benchmark compile job
explicitly includes the required-feature `chunked_queries` target. This adds
three checks to the previous 110-success gate.

## Completed-work measurements

Machine: Windows, i7-9750H, six physical/twelve reported available logical CPUs,
Rust 1.98.1 release profile with thin LTO. Other user Python computations were
active throughout; none were stopped or reprioritized. Numbers describe this
shared desktop session, not isolated portable performance guarantees.

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-search\examples\parallel_queries.rs`
prebuilds the point cloud, KD-tree, coordinate views and tensor. It compares
serial queries, the new bounded implementation, and a faithful copy of the
previous thread-per-chunk body retained only in the example/benchmark helper.
The legacy reference has an explicit 256-chunk ceiling and is skipped for
one-point chunks. Each measured call includes output allocation, all completed
queries, scoped thread spawning/joining and ordered result merging; fixture
setup, exact output verification and result destruction occur outside timing.

Grid coordinates are `(i % 256, i / 256, 0)`, radius is 1.01, and k is 8. Two
warmups per mode precede 12 rounds rotating mode order in the same process.
Every output equals the unsorted serial reference. Ranges below are observed
min/max of 12 calls, not confidence intervals. All bounded cases spawn exactly
12 threads; legacy ordinary-chunk cases spawn 64.

| Points / chunk size | Query | Serial median [range], ms | Bounded median [range], ms | Legacy median [range], ms |
| --- | --- | --- | --- | --- |
| 16,384 / 256 | radius | 7.135 [5.837, 11.668] | 3.907 [3.149, 4.979] | 7.978 [6.857, 9.143] |
| 16,384 / 256 | k-NN | 12.370 [10.796, 15.124] | 5.997 [5.318, 6.917] | 8.395 [7.176, 9.647] |
| 65,536 / 1,024 | radius | 27.251 [24.846, 33.318] | 11.695 [9.931, 14.592] | 14.714 [12.317, 16.860] |
| 65,536 / 1,024 | k-NN | 42.412 [37.244, 48.095] | 18.740 [15.549, 20.485] | 17.729 [15.941, 21.455] |
| 65,536 / 1 | radius | 41.006 [33.023, 50.195] | 13.843 [11.283, 15.776] | skipped |
| 65,536 / 1 | k-NN | 51.429 [45.790, 57.522] | 19.789 [18.571, 22.719] | skipped |

Tiny chunks complete with 12 workers instead of attempting 65,536 spawns; their
observed median ratios versus serial are about 3.0x radius and 2.6x k-NN. The
16,384-point cases have legacy/new median ratios of about 2.0x/1.4x.
The 65,536/1,024 k-NN bounded median is **5.7% slower** than legacy in this
rotation. This unfavorable case is retained; there is no universal speedup or
no-regression claim. Bounding thread demand is the primary correctness/resource
improvement. Both strategies retain the same per-chunk KD-tree scratch behavior.

### Criterion evidence

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-search\benches\chunked_queries.rs`
uses the same fixtures/reference and excludes result destruction with
`iter_custom`. It checks exact output once before each case, then gathers 20
samples with two-second warmups and four-second target measurement windows
(Criterion extends windows where needed). Cases run sequentially; changing
desktop load can bias comparisons even within one process. The following are
Criterion arithmetic-mean estimates and their 95% confidence intervals in ms,
not the slope estimates printed for linearly sampled cases.

| Points / chunk size | Query | Serial mean [95% CI], ms | Bounded mean [95% CI], ms | Legacy mean [95% CI], ms |
| --- | --- | --- | --- | --- |
| 16,384 / 256 | radius | 7.263 [6.909, 7.615] | 3.227 [3.202, 3.250] | 8.254 [8.067, 8.456] |
| 16,384 / 256 | knn | 12.780 [12.272, 13.385] | 5.771 [5.575, 5.972] | 8.804 [8.677, 8.929] |
| 65,536 / 1,024 | radius | 33.005 [32.119, 33.891] | 16.588 [14.792, 19.480] | 16.921 [16.515, 17.435] |
| 65,536 / 1,024 | knn | 48.316 [46.821, 49.830] | 20.541 [20.191, 20.926] | 32.590 [30.018, 35.839] |
| 65,536 / 1 | radius | 87.748 [75.852, 101.061] | 24.667 [23.194, 26.357] | skipped |
| 65,536 / 1 | knn | 76.476 [71.044, 82.252] | 29.967 [26.426, 34.616] | skipped |

The rotating experiment is the main comparison under changing desktop load;
Criterion is supporting evidence. Its intervals do not quantify between-mode
load changes. No legacy measurement is made for 65,536 one-point chunks.

## Reproduction and artifacts

```powershell
cargo run -p spatialrust-search --release --features parallel --example parallel_queries
cargo bench -p spatialrust-search --features parallel --bench chunked_queries -- --noplot
```

Ignored local receipts:

- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\bounded-queries-interleaved.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\bounded-queries-criterion.log`
- `C:\Users\rsasa\Workspace\SpatialRust\target\automation\bounded-queries-estimates.json`

Delivery uses a frozen source commit stacked on PR 125. Merge automation waits
for the parent to merge, retargets to main, requires all 113 successes with no
pending/failed checks and a clean merge state, and stops on source/dependency
changes or conflicts. A pending CI result is not reported as a completed merge.
