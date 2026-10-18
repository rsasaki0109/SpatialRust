# Parallel conversion of large GPU voxel metadata keys

Date: 2026-10-18. Named slice: parallel GPU voxel metadata key conversion.
Immediate parent: `2152ed537b76c8a303d170c2ebe3e878eb650b0d` (merged PR #271).

## Change and boundaries

The existing explicit batched metadata reader converts mapped 16-byte signed
key records into owned i64 coordinate tuples. Native inputs with at least
524,288 occupied cells now split this conversion into disjoint original-order
bands, using at most four workers including the caller. Worker availability is
queried only above that threshold. At most three scoped threads are spawned;
all complete before the mapped view or its unmap guard drops. Dirty key padding
is ignored. Each band writes only its own final output slice, preserving exact
sign extension and global key order.

Small inputs, one available worker and WASM retain the original exact-capacity
collector. Selection uses occupied-cell count rather than source point count:
a million points with few occupied cells keep the original conversion. The
existing >64MiB/device-limit serial metadata reader also retains its original
key conversion. No parallel policy is added to that fallback.

The scoped path safely initializes one final tuple vector before filling its
bands. Source output allocation size/capacity remains 24*C logical bytes; there
are no per-worker point vectors or new point-sized intermediates. This adds
24*C logical bytes of host zero-initialization writes before the original key
values are written. Threads and scheduling add overhead. Source request counts
and these logical writes do not establish physical memory usage, transactions,
bandwidth or peak residency, which were not measured.

The batched staging allocation, 64MiB cap, exact offsets, three copies, one
submission/map/wait, error messages and unmap guard retain their parent code.
Starts and indices remain on their original copy paths; count derivation and
final per-cell index sorting are unchanged. Logical metadata transfer remains
20*C+4*N; nonempty GPU construction still synchronizes four scalar bytes.
GPU key generation, sorting, compaction, shaders, caches, profiling, pools,
resident ownership, explicit transfer boundaries, public APIs, validation,
features and dependencies retain their parent contracts.

## Exploratory mapped-buffer comparison

An ignored standalone optimized probe compares four-way key bands (variant 0)
and three independent plane conversions (variant 1) to the serial decoder.
Every complete output equals the serial output. It uses real GPU-mapped key/
start/index bytes, three warmup pairs and 30 alternating measured pairs, rotating
variant order. Thread startup, owned output allocation and key-band zero
initialization are included. Fixture/copy/map preparation and checks/destruction
are excluded. This is a standalone conversion probe, not the complete library
implementation; it has no confidence intervals or controlled host isolation.
All 14 observations, including slower small inputs, are retained below.

| C | N | Variant (0=key bands, 1=planes) | Serial median ms | Candidate median ms | Ratio of medians |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65537 | 65537 | 0 | 0.490450 | 0.758850 | 0.646307 |
| 65537 | 65537 | 1 | 0.481100 | 0.619850 | 0.776156 |
| 262143 | 262143 | 0 | 2.130200 | 1.804350 | 1.180591 |
| 262143 | 262143 | 1 | 2.089650 | 1.688800 | 1.237358 |
| 524287 | 524287 | 0 | 3.761850 | 2.744600 | 1.370637 |
| 524287 | 524287 | 1 | 3.928950 | 3.260950 | 1.204848 |
| 524288 | 524288 | 0 | 3.850300 | 2.723750 | 1.413603 |
| 524288 | 524288 | 1 | 3.951200 | 3.373200 | 1.171351 |
| 1000003 | 1000003 | 0 | 6.950200 | 4.806700 | 1.445940 |
| 1000003 | 1000003 | 1 | 7.336000 | 5.996400 | 1.223401 |
| 2000003 | 2000003 | 0 | 13.720950 | 8.621250 | 1.591527 |
| 2000003 | 2000003 | 1 | 13.922750 | 10.996000 | 1.266165 |
| 65537 | 1000003 | 0 | 1.279450 | 1.553200 | 0.823751 |
| 65537 | 1000003 | 1 | 1.299000 | 1.381500 | 0.940282 |

The key-band candidate is chosen only at C>=524,288, where the probe supports absorbing startup/initialization overhead. Small mapped probes become slower in both variants. This exploratory threshold choice is local evidence, not a universal crossover; the finalized policy is independently tested and measured below.

## Correctness and provenance

Five native tests check worker/threshold/maximum-size policies, nonempty bands,
explicit 0/1/2/3/4/12 worker budgets, independent signed coordinates, exact output
capacity/order, dirty source guards/padding and unchanged borrowed words. Real
GPU fixtures cover threshold tails, both scalar/cell geometry extremes, full
parent/CPU outputs and actual >64MiB serial fallback. Complete CPU-key/device-key
builds cover empty/singleton, clamping, duplicates and occupied-cell thresholds.
Four simultaneous readers use parallel conversion while sharing live GPU
sources; owned results remain valid after pool clear and zero scratch budgets.

The complete immediate-parent batched reader and compaction metadata dispatch,
legacy fallback, mapping helpers, count derivation and final ordering are frozen
in native references. Parent complete calls share only independently identity-
checked original GPU builders/runtime. The CPU baseline freezes the original
key collector. No alternate cache is injected. All source, manifest and release
executable hashes are frozen before timing and remain identical through broad
validation. Rust/WGSL files are not edited while builds, tests or timings run.

VOXEL_DECODE_POLICY: verified 63 threshold/availability/maximum extents, bounded nonempty bands, caller participation and small/one-worker fallbacks.

VOXEL_DECODE_WORDS: verified 83491398 independent signed coordinates across explicit 0/1/2/3/4/12 worker budgets, dirty padding/guards, exact capacity and original order.

VOXEL_DECODE_MAPPED: verified 8 padded real GPU fixtures, threshold tails, parent/independent keys/counts/indices/final ordering and actual >64MiB serial fallback.

VOXEL_DECODE_PUBLIC: verified 21 complete CPU/device-key builds, actual occupied-cell policy, empty/singleton/clamping/duplicates/thresholds and unchanged scalar receipts.

VOXEL_DECODE_OWNERSHIP: verified eight concurrent parallel-key reads of live shared source buffers, scope-complete maps, retained host results and pool-clear/zero-budget independence.

There are 585 unchanged original GPU/image source files and 597 source/manifest/executable SHA-256 values. All original shaders, runtime, GPU construction, public conversion, compaction and fallback are byte-identical. The one edited original source adds private key conversion dispatch; its other statements retain the original body. Privacy scans find zero forbidden adapter-name matches; raw local process receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 159 |
| voxel | 228 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 429 |

All-features/all-targets Clippy with warnings denied, CPU-only check, minimal and CI-feature WASM checks with warnings denied, formatting and diff checks pass. Native suites include ignored correctness tests and exclude explicit benchmarks.

## Final paired method

Two consecutive surveys use the same finalized optimized release executable,
three warmup pairs and 30 measured pairs per route/case. Pair order alternates
evenly and three route positions rotate. A headless initialized runtime is used.
Input fixtures, oracle construction, runtime initialization, checks and output
destruction are excluded. No build/test initiated by this workflow overlaps the
timed surveys. Other shared-host work is not controlled; no case is retried or
filtered. All output coordinates and arrays are checked after every call.

- `decode`: production key converter versus the frozen original collector on
  actual mapped key bytes; owned allocation, CPU availability query, startup,
  zero initialization, conversion and joins are included when selected.
- `readback`: complete public conversion of resident segments, including original
  staging/copies/submit/map/wait, all owned vectors, counts and final index order.
- `full`: CPU key preparation/upload, unchanged GPU sort/compaction, scalar sync
  and explicit metadata readback/finalization; all payload transfers are included.

A separate 16*max(C,1)-byte mapped key fixture stays alive throughout all three
routes for both variants. It is benchmark-only, prepared outside timing, and
increases the benchmark memory context equally. Production does not add it.
Pattern 0 uses mixed duplicates, pattern 1 a clamped constant, pattern 2 descending
unique keys. Twelve cases include empty, small, threshold-1/threshold/threshold+1,
million/two-million cells, million-point few-cell controls and actual >64MiB
fallback. The direct decode benchmark still exercises its worker policy for
the cap-fallback case, while both public routes retain their serial fallback.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times with seed
2026101807 + 1000*survey + 10*case_index + route_index (zero-based indices) and
sorted draws 249/9749. An interval entirely above/below one is favorable/adverse;
otherwise inconclusive. Zero samples retain undefined ratios/intervals, with no
epsilon substitution. There is no multiplicity correction. Pre/post samples do
not prove isolation during every pair. No portable, application or cold-runtime
speed guarantee is inferred.

| Route | Selected parallel comparisons | Original-path controls |
| --- | --- | --- |
| decode | 12 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 9 inconclusive / 1 adverse / 2 unresolved |
| readback | 10 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 9 inconclusive / 3 adverse / 2 unresolved |
| full | 7 favorable / 3 inconclusive / 0 adverse / 0 unresolved | 2 favorable / 9 inconclusive / 3 adverse / 0 unresolved |

Selected CPU conversion includes the direct converter on the public cap-fallback case; selected readback/full requires both batching and more than one worker. All controls, adverse and inconclusive results remain below. Seven original-path control comparisons have adverse intervals; their causes remain unresolved, including the small serial dispatch, code layout, timer resolution or shared-host load. The full 65,537-cell control reaches ratio 0.949 in survey 2; the unchanged public cap fallback has adverse readback/full ratios 0.965/0.970 in survey 1. All are retained, and the selected parallel public paths have no adverse interval.

For selected decode, favorable paired ratios range 1.501-2.152.

For selected readback, favorable paired ratios range 1.111-1.217.

For selected full, favorable paired ratios range 1.020-1.086.

## Pre/post observations

Two-second boundary samples report only busy process counts/CPU cores. Names and PIDs stay in ignored local receipts. GPU pairs are utilization/memory-controller utilization percentages, not allocated-memory percentages. These samples do not observe every timed pair.

| Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Observed quiet |
| --- | --- | ---: | --- | --- | --- |
| 1 | pre | 0 | none | [[[0,0]],[[0,0]]] | True |
| 1 | post | 0 | none | [[[0,0]],[[0,0]]] | True |
| 2 | pre | 0 | none | [[[0,0]],[[0,0]]] | True |
| 2 | post | 1 | 0.302 | [[[0,0]],[[0,0]]] | False |

Environment 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 threshold=524288 max_workers=4 warmups=3 pairs=30 pair_order=alternating route_order=rotating checks_drop_excluded=true parent_build_shared=true mapped_key_fixture_retained=true`.

Environment 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 threshold=524288 max_workers=4 warmups=3 pairs=30 pair_order=alternating route_order=rotating checks_drop_excluded=true parent_build_shared=true mapped_key_fixture_retained=true`.

## All final comparisons

Medians retain every measured sample; ratios/intervals use paired log statistics. Zero counts are parent/current.

| Survey | N | C | Pattern | Route | Workers | Batched | Parallel route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 0 | 0 | 0 | decode | 1 | False | False | 0.000050 | 0.000000 | undefined | undefined | 15/19 | unresolved |
| 1 | 0 | 0 | 0 | readback | 1 | False | False | 0.000050 | 0.000100 | undefined | undefined | 15/12 | unresolved |
| 1 | 0 | 0 | 0 | full | 1 | False | False | 0.002900 | 0.002900 | 1.008064 | 0.996734-1.019526 | 0/0 | inconclusive |
| 1 | 1 | 1 | 0 | decode | 1 | True | False | 0.000100 | 0.000200 | 0.812252 | 0.690956-0.977160 | 0/0 | adverse |
| 1 | 1 | 1 | 0 | readback | 1 | True | False | 0.089600 | 0.091100 | 0.995621 | 0.968674-1.026438 | 0/0 | inconclusive |
| 1 | 1 | 1 | 0 | full | 1 | True | False | 0.256950 | 0.261150 | 0.978023 | 0.960778-0.993198 | 0/0 | adverse |
| 1 | 65537 | 65537 | 2 | decode | 1 | True | False | 0.380850 | 0.378900 | 1.001141 | 0.943977-1.059473 | 0/0 | inconclusive |
| 1 | 65537 | 65537 | 2 | readback | 1 | True | False | 0.839050 | 0.879950 | 0.973441 | 0.936790-1.015084 | 0/0 | inconclusive |
| 1 | 65537 | 65537 | 2 | full | 1 | True | False | 3.300350 | 3.286400 | 0.998981 | 0.961740-1.046310 | 0/0 | inconclusive |
| 1 | 524287 | 524287 | 2 | decode | 1 | True | False | 2.411600 | 2.428050 | 1.015741 | 0.980778-1.056083 | 0/0 | inconclusive |
| 1 | 524287 | 524287 | 2 | readback | 1 | True | False | 8.673250 | 8.802950 | 0.989177 | 0.930331-1.043582 | 0/0 | inconclusive |
| 1 | 524287 | 524287 | 2 | full | 1 | True | False | 23.210950 | 23.575850 | 0.981364 | 0.913616-1.054429 | 0/0 | inconclusive |
| 1 | 524288 | 524288 | 2 | decode | 4 | True | True | 2.439350 | 1.586350 | 1.623768 | 1.565526-1.684990 | 0/0 | favorable |
| 1 | 524288 | 524288 | 2 | readback | 4 | True | True | 6.295750 | 5.177700 | 1.191394 | 1.152542-1.231072 | 0/0 | favorable |
| 1 | 524288 | 524288 | 2 | full | 4 | True | True | 19.533200 | 18.236550 | 1.061285 | 1.046300-1.076690 | 0/0 | favorable |
| 1 | 524289 | 524289 | 2 | decode | 4 | True | True | 2.637850 | 1.513800 | 1.742510 | 1.688628-1.797255 | 0/0 | favorable |
| 1 | 524289 | 524289 | 2 | readback | 4 | True | True | 6.584650 | 5.505450 | 1.187180 | 1.155235-1.217838 | 0/0 | favorable |
| 1 | 524289 | 524289 | 2 | full | 4 | True | True | 22.033100 | 20.482550 | 1.066480 | 1.051685-1.081265 | 0/0 | favorable |
| 1 | 1000003 | 1000003 | 2 | decode | 4 | True | True | 4.682750 | 2.648000 | 1.792280 | 1.743045-1.846614 | 0/0 | favorable |
| 1 | 1000003 | 1000003 | 2 | readback | 4 | True | True | 16.024150 | 14.582100 | 1.110734 | 1.084200-1.134836 | 0/0 | favorable |
| 1 | 1000003 | 1000003 | 2 | full | 4 | True | True | 41.936200 | 39.787750 | 1.052769 | 1.037914-1.068451 | 0/0 | favorable |
| 1 | 1048576 | 1048576 | 2 | decode | 4 | True | True | 5.372250 | 2.911800 | 1.875529 | 1.803141-1.952256 | 0/0 | favorable |
| 1 | 1048576 | 1048576 | 2 | readback | 4 | True | True | 12.597150 | 10.557750 | 1.216540 | 1.180836-1.257477 | 0/0 | favorable |
| 1 | 1048576 | 1048576 | 2 | full | 4 | True | True | 38.992750 | 37.026500 | 1.054543 | 1.038554-1.071611 | 0/0 | favorable |
| 1 | 2000003 | 2000003 | 2 | decode | 4 | True | True | 10.216900 | 5.114550 | 2.046987 | 1.973818-2.126024 | 0/0 | favorable |
| 1 | 2000003 | 2000003 | 2 | readback | 4 | True | True | 29.189800 | 24.370900 | 1.209458 | 1.162172-1.262389 | 0/0 | favorable |
| 1 | 2000003 | 2000003 | 2 | full | 4 | True | True | 76.530400 | 72.421150 | 1.045486 | 0.997277-1.086109 | 0/0 | inconclusive |
| 1 | 1000003 | 217 | 0 | decode | 1 | True | False | 0.000950 | 0.001100 | 0.946486 | 0.624691-1.420094 | 0/0 | inconclusive |
| 1 | 1000003 | 217 | 0 | readback | 1 | True | False | 2.125850 | 2.121250 | 0.995029 | 0.962436-1.030485 | 0/0 | inconclusive |
| 1 | 1000003 | 217 | 0 | full | 1 | True | False | 17.633150 | 17.302600 | 0.995693 | 0.983810-1.008009 | 0/0 | inconclusive |
| 1 | 1000003 | 1 | 1 | decode | 1 | True | False | 0.000400 | 0.000350 | 1.021940 | 0.591991-1.770286 | 0/0 | inconclusive |
| 1 | 1000003 | 1 | 1 | readback | 1 | True | False | 2.054550 | 2.089100 | 0.965322 | 0.929232-0.992630 | 0/0 | adverse |
| 1 | 1000003 | 1 | 1 | full | 1 | True | False | 16.575850 | 16.554600 | 0.999916 | 0.990317-1.010014 | 0/0 | inconclusive |
| 1 | 2796203 | 2796203 | 2 | decode | 4 | False | True | 13.192300 | 6.591150 | 2.003909 | 1.948791-2.058561 | 0/0 | favorable |
| 1 | 2796203 | 2796203 | 2 | readback | 4 | False | False | 54.339350 | 56.113600 | 0.965473 | 0.938107-0.990507 | 0/0 | adverse |
| 1 | 2796203 | 2796203 | 2 | full | 4 | False | False | 113.507450 | 113.876250 | 0.969517 | 0.939646-0.996139 | 0/0 | adverse |
| 2 | 0 | 0 | 0 | decode | 1 | False | False | 0.000000 | 0.000000 | undefined | undefined | 19/17 | unresolved |
| 2 | 0 | 0 | 0 | readback | 1 | False | False | 0.000000 | 0.000100 | undefined | undefined | 16/14 | unresolved |
| 2 | 0 | 0 | 0 | full | 1 | False | False | 0.002500 | 0.002500 | 0.993219 | 0.980214-1.005247 | 0/0 | inconclusive |
| 2 | 1 | 1 | 0 | decode | 1 | True | False | 0.000100 | 0.000100 | 0.954842 | 0.850667-1.071773 | 0/0 | inconclusive |
| 2 | 1 | 1 | 0 | readback | 1 | True | False | 0.080250 | 0.080450 | 0.976775 | 0.932066-1.015065 | 0/0 | inconclusive |
| 2 | 1 | 1 | 0 | full | 1 | True | False | 0.230350 | 0.228800 | 1.036836 | 1.006072-1.076494 | 0/0 | favorable |
| 2 | 65537 | 65537 | 2 | decode | 1 | True | False | 0.329050 | 0.366050 | 0.990422 | 0.911260-1.073553 | 0/0 | inconclusive |
| 2 | 65537 | 65537 | 2 | readback | 1 | True | False | 0.808550 | 0.817400 | 0.961909 | 0.925381-0.997652 | 0/0 | adverse |
| 2 | 65537 | 65537 | 2 | full | 1 | True | False | 3.212600 | 3.343300 | 0.948943 | 0.904490-0.988625 | 0/0 | adverse |
| 2 | 524287 | 524287 | 2 | decode | 1 | True | False | 2.422100 | 2.417000 | 1.012877 | 0.985562-1.044910 | 0/0 | inconclusive |
| 2 | 524287 | 524287 | 2 | readback | 1 | True | False | 8.568200 | 8.816550 | 1.003199 | 0.945306-1.078591 | 0/0 | inconclusive |
| 2 | 524287 | 524287 | 2 | full | 1 | True | False | 20.746800 | 21.286400 | 0.982716 | 0.923727-1.046295 | 0/0 | inconclusive |
| 2 | 524288 | 524288 | 2 | decode | 4 | True | True | 2.364500 | 1.500700 | 1.616058 | 1.572561-1.662705 | 0/0 | favorable |
| 2 | 524288 | 524288 | 2 | readback | 4 | True | True | 5.928350 | 5.368500 | 1.125568 | 1.098907-1.156877 | 0/0 | favorable |
| 2 | 524288 | 524288 | 2 | full | 4 | True | True | 18.245350 | 18.206600 | 1.011952 | 0.994081-1.030682 | 0/0 | inconclusive |
| 2 | 524289 | 524289 | 2 | decode | 4 | True | True | 2.364700 | 1.534700 | 1.500749 | 1.448564-1.547274 | 0/0 | favorable |
| 2 | 524289 | 524289 | 2 | readback | 4 | True | True | 6.118700 | 5.596450 | 1.112425 | 1.091669-1.134889 | 0/0 | favorable |
| 2 | 524289 | 524289 | 2 | full | 4 | True | True | 20.671450 | 20.287150 | 1.020476 | 1.007721-1.034170 | 0/0 | favorable |
| 2 | 1000003 | 1000003 | 2 | decode | 4 | True | True | 4.798550 | 2.743750 | 1.786769 | 1.729110-1.847445 | 0/0 | favorable |
| 2 | 1000003 | 1000003 | 2 | readback | 4 | True | True | 16.403200 | 14.695850 | 1.148483 | 1.103042-1.221826 | 0/0 | favorable |
| 2 | 1000003 | 1000003 | 2 | full | 4 | True | True | 41.365450 | 39.037450 | 1.018123 | 0.973311-1.060268 | 0/0 | inconclusive |
| 2 | 1048576 | 1048576 | 2 | decode | 4 | True | True | 7.002200 | 3.169000 | 2.151830 | 2.060303-2.238783 | 0/0 | favorable |
| 2 | 1048576 | 1048576 | 2 | readback | 4 | True | True | 15.078100 | 12.311900 | 1.211505 | 1.170137-1.248795 | 0/0 | favorable |
| 2 | 1048576 | 1048576 | 2 | full | 4 | True | True | 47.903800 | 44.654250 | 1.068240 | 1.050314-1.085466 | 0/0 | favorable |
| 2 | 2000003 | 2000003 | 2 | decode | 4 | True | True | 9.553100 | 4.801950 | 2.025310 | 1.955765-2.099740 | 0/0 | favorable |
| 2 | 2000003 | 2000003 | 2 | readback | 4 | True | True | 28.519900 | 23.409050 | 1.211525 | 1.174685-1.255902 | 0/0 | favorable |
| 2 | 2000003 | 2000003 | 2 | full | 4 | True | True | 74.931000 | 69.952250 | 1.085556 | 1.058507-1.119779 | 0/0 | favorable |
| 2 | 1000003 | 217 | 0 | decode | 1 | True | False | 0.001400 | 0.001600 | 1.008828 | 0.665551-1.558268 | 0/0 | inconclusive |
| 2 | 1000003 | 217 | 0 | readback | 1 | True | False | 2.325550 | 2.359550 | 0.986787 | 0.965224-1.009774 | 0/0 | inconclusive |
| 2 | 1000003 | 217 | 0 | full | 1 | True | False | 18.105950 | 18.423050 | 0.989410 | 0.978032-1.000713 | 0/0 | inconclusive |
| 2 | 1000003 | 1 | 1 | decode | 1 | True | False | 0.000550 | 0.000600 | 0.917685 | 0.542276-1.559188 | 0/0 | inconclusive |
| 2 | 1000003 | 1 | 1 | readback | 1 | True | False | 2.313100 | 2.225950 | 1.008216 | 0.976963-1.042754 | 0/0 | inconclusive |
| 2 | 1000003 | 1 | 1 | full | 1 | True | False | 17.255200 | 16.897200 | 1.020119 | 1.004088-1.035934 | 0/0 | favorable |
| 2 | 2796203 | 2796203 | 2 | decode | 4 | False | True | 13.501650 | 6.707600 | 2.030518 | 1.989782-2.072346 | 0/0 | favorable |
| 2 | 2796203 | 2796203 | 2 | readback | 4 | False | False | 55.090000 | 56.237550 | 0.965686 | 0.929857-1.000871 | 0/0 | inconclusive |
| 2 | 2796203 | 2796203 | 2 | full | 4 | False | False | 115.428400 | 113.434300 | 1.016365 | 0.988380-1.045894 | 0/0 | inconclusive |

## Source locations

- [voxel_metadata_readback.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_readback.rs)
- [voxel_metadata_keys.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_keys.rs)
- [voxel_metadata_keys_tests.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_keys_tests.rs)
- [voxel_metadata_keys_readback_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_keys_readback_reference.rs)
- [voxel_metadata_keys_compact_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_keys_compact_reference.rs)
- [voxel_metadata_keys_benchmark.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_metadata_keys_benchmark.rs)
