# Batched explicit GPU voxel metadata readback

Date: 2026-10-18. Named slice: batched GPU voxel metadata readback. Immediate
parent: `0c599657702364a9d710b38538681809da3a5022` (merged PR #270).

## Change and transfer contract

The explicit GPU segment-to-CPU conversion previously created, submitted,
mapped and waited for three staging buffers in sequence: unique keys, cell
starts and point indices. Accepted private layouts now record the same three
copies into one staging allocation, submit once, map once and wait once. The
mapped key records convert directly to signed i64 tuples, avoiding the original
intermediate host Vec of 16-byte key records. Starts and indices copy to their
original owned vectors. Existing count derivation and per-cell index sorting
retain their exact behavior. Mapping errors keep the original messages and
result type; a drop guard unmaps after the mapped view on return or unwind.

For C occupied cells and N points, the exact ranges are keys `[0,16*C)`, starts
`[16*C,20*C)` and indices `[20*C,20*C+4*N)`. They are adjacent, nonoverlapping
and all copy-aligned. The one mapped range starts at zero, satisfying map
alignment even when an internal u32 plane begins at 4 modulo 8. Checked u64
products/sums, host slice bounds, the runtime maximum buffer size and a private
64MiB cap bound the layout. Zero cells and cells exceeding points do not batch.
Unsupported layouts take the original serial reader. The public empty route
still returns before any metadata readback.

Logical metadata transfer bytes remain `20*C+4*N`; the GPU construction scalar
still reads four bytes for nonempty inputs. This metadata transfer remains an
explicit public conversion rather than a new hidden copy in resident kernels.
GPU key generation, stable radix sorting, compaction, shaders, pipeline caches,
passes, profiling, owned outputs, scratch, public validation/error precedence,
public signatures, features and dependencies retain their parent contracts.

## Allocation and memory tradeoff

Source creation/submit/map/poll requests fall from three to one for accepted
layouts; the number of copies stays three. These are source-level request
counts, not driver transaction or allocation measurements. The source staging
peak changes from `max(16*C,4*N)` to `20*C+4*N`, up to 2.25 times the original
at C=N/4. Simultaneous staging requests grow and are bounded to 64MiB.
Physical residency, allocator granularity, bandwidth and peak device/host memory
were not measured. The direct key conversion removes one source intermediate
allocation of 16*C logical bytes; this does not prove a physical memory saving.

For 1,000,003 unique cells, staging changes from a largest original plane of
16,000,048 bytes to 24,000,072 bytes. For one cell and the same N, it changes
from 4,000,012 to 4,000,032 bytes. For 2,796,203 unique cells the combined
payload is 67,108,872 bytes, above the cap; the original independent
44,739,248/11,184,812/11,184,812-byte staging requests are used instead.
No pipeline is added. Timings use an initialized runtime and do not establish
cold-runtime latency or a speed benefit for the large fallback.

## Correctness and provenance

Five new native tests exercise checked layouts, device/cap boundaries,
arithmetic failures and copy-versus-map alignment; padded device buffers with
signed endpoints, nonzero key padding, partial cell lengths and independently
constructed counts; exact raw parent/CPU outputs; public CPU-key and device-key
builds with clamping, duplicates and dispatch boundaries; actual cap fallback;
and concurrent readers retaining live source buffers and owned outputs through
pool clear and zero scratch budgets. The fallback fixture crosses the actual
64MiB cap using metadata buffers, avoiding a large sorting fixture.

A native reference freezes the immediate-parent serial reader, mapping helpers,
count derivation and final index ordering. Complete parent calls share only the
original, independently identity-checked GPU builder and runtime. No separate
cache is injected. Production sources and the benchmark executable are frozen
before timing; no Rust/WGSL source changes occur during builds, tests or timing.

VOXEL_METADATA_LAYOUT: verified 24 exact nonoverlapping ranges, copy/map alignments, device/cap boundaries, arithmetic failures and source staging peak up to 2.25x parent.

VOXEL_METADATA_RAW: verified 30 padded GPU fixtures, three exact planes, independent counts, ignored key padding, signed endpoints, parent and final per-cell ordering.

VOXEL_METADATA_PUBLIC: verified 21 complete CPU-key and GPU-key builds, empty/singleton, clamp endpoints, duplicates, dispatch boundaries and unchanged construction scalar receipts.

VOXEL_METADATA_FALLBACK: verified actual serial readback at 2796203 cells: combined payload exceeds 64MiB while each original plane fits, all output bytes match parent.

VOXEL_METADATA_OWNERSHIP: verified twelve concurrent reads on shared live source buffers, repeated maps, owned retained results and zero-budget/pool-clear independence.

There are 581 unchanged original GPU/image source files and 592 source/manifest/executable SHA-256 values. All original shaders, runtime, public conversion and GPU builder are byte-identical. The only edited original Rust source adds private metadata dispatch with the original fallback. The finalized release executable, sources, manifests and resolved local ignored lockfile are identical before/after both surveys and broad validation. Repository and PR-body privacy scans find zero forbidden adapter-name matches; raw process receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 159 |
| voxel | 223 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 424 |

All-features/all-targets Clippy with warnings denied, CPU-only check, minimal/CI-feature WASM checks with warnings denied, workspace formatting and diff checks pass. Native suites include ignored correctness tests and exclude explicit benchmarks.

## Paired measurement method

Two consecutive surveys use the same finalized release unit executable and
headless Vulkan/DiscreteGpu runtime, with 12 available CPU workers reported.
The runtime reports max_buffer_size=18446744073709551615; this is an API limit,
not a physical memory capacity. The private staging cap is 67,108,864 bytes.
Each survey has ten cases, three warmup pairs and 30 measured pairs per route.
Pair order alternates evenly and route order rotates. CPU oracle creation,
input fixture preparation, runtime initialization, checks and output destruction
are outside timing. No build/test started by this workflow overlaps timing.
Other shared-host work is not controlled; no case is retried or filtered.

- `readback`: complete explicit public conversion on already resident segments,
  including staging, copies, submit/map/wait, key conversion, owned host vectors,
  count derivation and final per-cell index sorting.
- `full`: complete CPU-key preparation/upload, unchanged GPU sort/compaction,
  construction scalar sync and explicit metadata readback/finalization. Both
  paths use the same unchanged GPU builder. Transfers are included.

Pattern 0 has mixed duplicate keys (217 occupied cells when N>=257), pattern 1
has one clamped constant key, and pattern 2 has descending unique keys. Every
returned array is checked against independent CPU segments after each timed
call. The empty case is an unchanged-path control. All nonempty timed cases
fit the batched layout; cap fallback correctness is tested separately.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times with seed
`2026101806 + 1000*survey + 10*case_index + route_index`, zero-based indices,
and sorted draws 249/9749. Entirely above/below one is favorable/adverse,
otherwise inconclusive. Zero samples produce no ratio or interval; they are
retained without adding an epsilon. There is no multiplicity correction.

| Route | Nonempty: 18 comparisons | Empty control: 2 comparisons |
| --- | --- | --- |
| readback | 18 favorable, 0 inconclusive, 0 adverse, 0 unresolved (zero samples) | 0 favorable, 0 inconclusive, 0 adverse, 2 unresolved (zero samples) |
| full | 18 favorable, 0 inconclusive, 0 adverse, 0 unresolved (zero samples) | 1 favorable, 1 inconclusive, 0 adverse, 0 unresolved (zero samples) |

Across nonempty cases, all 18 readback comparisons are favorable (1.623–3.042 ratio), and all 18 complete-call comparisons are favorable (1.067–1.930). Million-point mixed/constant complete calls range 1.067–1.103; million-point unique calls are 1.332/1.340. These initialized shared-host diagnostics do not establish portable/application guarantees.

Both empty readback controls have zero-resolution samples, so their ratios are undefined. The unchanged empty complete-call control has one favorable 1.019 interval and one inconclusive interval at about 0.003ms. That favorable control difference is not attributable to this metadata batching; wrapper/code layout, timer resolution and host noise are unresolved. No performance exclusion or causal speed claim is made for it.

## Pre/post host observations

Samples span two seconds before/after each survey. Process names/PIDs remain only in ignored receipts. GPU pairs are utilization/memory-controller utilization percentages, not allocated-memory percentages. No sample observes every timed pair.

| Survey | Boundary | Busy process count | CPU cores per busy process | GPU/controller samples | Observed quiet |
| --- | --- | ---: | --- | --- | --- |
| 1 | pre | 1 | 1.000 | [[[0,0]],[[0,0]]] | False |
| 1 | post | 1 | 1.000 | [[[0,0]],[[0,0]]] | False |
| 2 | pre | 1 | 1.000 | [[[0,0]],[[0,0]]] | False |
| 2 | post | 1 | 1.000 | [[[0,0]],[[0,0]]] | False |

## All final comparisons

Medians include every measured sample. Intervals and ratios use paired log statistics rather than ratios of medians. Zero counts are parent/current; undefined rows retain their zero samples.

| Survey | N | C | Pattern | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 0 | 0 | 0 | readback | 0.000000 | 0.000100 | undefined | undefined | 19/12 | unresolved (zero samples) |
| 1 | 0 | 0 | 0 | full | 0.003000 | 0.003000 | 1.008438 | 0.993249–1.025786 | 0/0 | inconclusive |
| 1 | 1 | 1 | 0 | readback | 3.179500 | 1.014100 | 3.019049 | 2.957540–3.085659 | 0/0 | favorable |
| 1 | 1 | 1 | 0 | full | 4.190250 | 2.163050 | 1.925895 | 1.885981–1.963307 | 0/0 | favorable |
| 1 | 2 | 2 | 2 | readback | 2.996300 | 0.997500 | 3.038957 | 2.947519–3.136680 | 0/0 | favorable |
| 1 | 2 | 2 | 2 | full | 4.925900 | 2.945850 | 1.656412 | 1.595056–1.711836 | 0/0 | favorable |
| 1 | 257 | 217 | 0 | readback | 3.088500 | 1.016850 | 3.041517 | 2.948814–3.152409 | 0/0 | favorable |
| 1 | 257 | 217 | 0 | full | 5.036100 | 2.969850 | 1.689443 | 1.662864–1.716904 | 0/0 | favorable |
| 1 | 65537 | 217 | 0 | readback | 2.929950 | 1.036300 | 2.820865 | 2.761524–2.881671 | 0/0 | favorable |
| 1 | 65537 | 217 | 0 | full | 5.658600 | 3.736400 | 1.551250 | 1.515273–1.594397 | 0/0 | favorable |
| 1 | 65537 | 1 | 1 | readback | 3.308800 | 1.149850 | 2.820354 | 2.756465–2.884058 | 0/0 | favorable |
| 1 | 65537 | 1 | 1 | full | 6.063950 | 3.991050 | 1.525755 | 1.497593–1.553899 | 0/0 | favorable |
| 1 | 65537 | 65537 | 2 | readback | 4.508750 | 1.908600 | 2.409505 | 2.340615–2.480024 | 0/0 | favorable |
| 1 | 65537 | 65537 | 2 | full | 7.462350 | 5.089600 | 1.480949 | 1.413526–1.554567 | 0/0 | favorable |
| 1 | 1000003 | 217 | 0 | readback | 5.855600 | 3.486550 | 1.644551 | 1.601073–1.690197 | 0/0 | favorable |
| 1 | 1000003 | 217 | 0 | full | 23.814350 | 22.082700 | 1.094638 | 1.075054–1.115196 | 0/0 | favorable |
| 1 | 1000003 | 1 | 1 | readback | 5.408600 | 3.361550 | 1.623217 | 1.577983–1.669393 | 0/0 | favorable |
| 1 | 1000003 | 1 | 1 | full | 23.087100 | 20.785650 | 1.103092 | 1.086138–1.119823 | 0/0 | favorable |
| 1 | 1000003 | 1000003 | 2 | readback | 31.281050 | 17.949050 | 1.748013 | 1.703957–1.794308 | 0/0 | favorable |
| 1 | 1000003 | 1000003 | 2 | full | 57.137250 | 43.165750 | 1.331650 | 1.307765–1.357034 | 0/0 | favorable |
| 2 | 0 | 0 | 0 | readback | 0.000100 | 0.000100 | undefined | undefined | 11/10 | unresolved (zero samples) |
| 2 | 0 | 0 | 0 | full | 0.002900 | 0.002800 | 1.018814 | 1.008139–1.028617 | 0/0 | favorable |
| 2 | 1 | 1 | 0 | readback | 2.935850 | 0.993400 | 2.943171 | 2.867958–3.015804 | 0/0 | favorable |
| 2 | 1 | 1 | 0 | full | 3.973700 | 2.009150 | 1.930232 | 1.845358–1.994774 | 0/0 | favorable |
| 2 | 2 | 2 | 2 | readback | 3.044500 | 1.005250 | 3.004181 | 2.920009–3.085777 | 0/0 | favorable |
| 2 | 2 | 2 | 2 | full | 4.918100 | 2.980900 | 1.651246 | 1.608996–1.688863 | 0/0 | favorable |
| 2 | 257 | 217 | 0 | readback | 2.722200 | 0.905200 | 2.995799 | 2.876683–3.138837 | 0/0 | favorable |
| 2 | 257 | 217 | 0 | full | 4.691350 | 2.845400 | 1.639063 | 1.588334–1.687581 | 0/0 | favorable |
| 2 | 65537 | 217 | 0 | readback | 3.196300 | 1.147900 | 2.888140 | 2.769009–3.068658 | 0/0 | favorable |
| 2 | 65537 | 217 | 0 | full | 6.026850 | 4.068300 | 1.481911 | 1.439237–1.519143 | 0/0 | favorable |
| 2 | 65537 | 1 | 1 | readback | 3.201400 | 1.096650 | 2.864635 | 2.784455–2.943552 | 0/0 | favorable |
| 2 | 65537 | 1 | 1 | full | 6.012350 | 3.956000 | 1.516767 | 1.492142–1.541796 | 0/0 | favorable |
| 2 | 65537 | 65537 | 2 | readback | 4.336550 | 1.886600 | 2.375466 | 2.323210–2.428777 | 0/0 | favorable |
| 2 | 65537 | 65537 | 2 | full | 7.854200 | 5.148000 | 1.508496 | 1.451961–1.570689 | 0/0 | favorable |
| 2 | 1000003 | 217 | 0 | readback | 5.635000 | 3.371950 | 1.700232 | 1.658096–1.741449 | 0/0 | favorable |
| 2 | 1000003 | 217 | 0 | full | 22.076650 | 20.440100 | 1.092623 | 1.075165–1.110320 | 0/0 | favorable |
| 2 | 1000003 | 1 | 1 | readback | 6.139400 | 3.765400 | 1.628742 | 1.573672–1.680183 | 0/0 | favorable |
| 2 | 1000003 | 1 | 1 | full | 23.355950 | 21.429450 | 1.067291 | 1.007270–1.114310 | 0/0 | favorable |
| 2 | 1000003 | 1000003 | 2 | readback | 33.557900 | 18.562100 | 1.779847 | 1.704260–1.848190 | 0/0 | favorable |
| 2 | 1000003 | 1000003 | 2 | full | 60.419050 | 45.602550 | 1.339901 | 1.289382–1.384290 | 0/0 | favorable |

## Source locations

- [voxel_compact.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_compact.rs)
- [voxel_metadata_readback.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_readback.rs)
- [voxel_metadata_readback_tests.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_readback_tests.rs)
- [voxel_metadata_readback_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_readback_reference.rs)
- [voxel_metadata_readback_benchmark.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_readback_benchmark.rs)
