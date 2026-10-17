# Bounded CPU preparation before voxel GPU upload

Date: 2026-10-18. Named slice: bounded CPU voxel entry preparation. Parent:
`680ad12ad6e091d0fba38dab0380bc0b721413cd` (merged PR #266).

## Change and contracts

CPU-key voxel construction prepares exact 16-byte sort entries before the
existing explicit upload. For native inputs of at least 524,288 points, the
host now fills disjoint original-order output bands with at most four workers,
including the caller. The available CPU count and a 131,072-point useful-worker
bound cap dispatch; at most three additional scoped threads are spawned per
call. Small inputs avoid CPU-count queries and worker startup. One-worker and
WASM paths retain the complete original exact-capacity serial collector.

One exact-sized output Vec is safely allocated/zeroed through the existing
bytemuck dependency's `extern_crate_alloc` feature. Workers borrow disjoint
mutable bands and immutable source slices, adding each band's global base to
the point index. There are no point-sized per-worker vectors, payload joins,
new unsafe blocks, public API changes or new dependencies. Thread/runtime
metadata is worker bounded; zero initialization is included in measured work.
The 16*N output length/capacity and uploaded bytes remain identical. This
does not claim that a parallel zeroed allocation has identical page-fault or
physical-memory behavior to the original collector.

Both public CPU-key APIs still validate empty/count/device limits before host
preparation. Every GPU shader, sort/compaction pass, scratch request, query,
submission, pipeline cache, GPU-key buffer entry point and frame preflight is
unchanged. Nonempty construction still synchronizes its original four-byte
scalar, and the full API explicitly downloads its original metadata payload.
Image processing, core, automatic device placement and admission gates remain
unchanged. Concurrent calls have independent outputs; the four-worker cap is
per call rather than a global scheduler.

## Correctness and provenance

Five new native tests compare full frozen parent entry points, independent
CPU segments and exact host/GPU words. Fixtures cover all signed saturation
combinations, random/extreme coordinates, keys clamping into one cell, repeated
reverse keys, empty/singleton, useful-worker and dispatch thresholds, partial
bands, prefix boundaries and 2,000,003 points. Forced available counts
0/1/2/3/4/12/usize::MAX check bitwise record/point-index parity, original
capacity, dirty output canaries and high-u32 global indices. Complete GPU
indices/keys/starts retain CPU and parent order; GPU inputs stay immutable.
Results survive host mutation and scratch clear/reuse. Four concurrent callers
exercise mixed large inputs and forced 1/2/3/4 worker shapes. Fresh/warm/zero
pool budgets and instrumented primitive compositions keep the same resource,
pass/query and explicit scalar-transfer counts. No new public profiler API
is introduced.

The frozen native reference copies complete immediate-parent CPU-key functions
and the serial preparation, changing names and the compactor import path only.
The unchanged GPU dependencies are shared. An independent source audit checks
the exact host delegation, original serial fallback and frozen functions,
the one dependency-feature edit and every other original GPU/image source.
The host test checks 510,778,884 record words; direct-band/guard tests check
30,688,608 values, and complete parent/CPU/retained comparisons check
104,119,552 raw index/key/start values across 24 API fixture pairs. These are
checked values across repeated fixtures and worker shapes, not unique points.

There are 562 unchanged original source files and 572 source/manifest/executable SHA-256 values. The same finalized release binary, GPU/image source tree, Cargo manifests and local resolved Cargo.lock are fixed before/after both surveys and broad validation. Cargo.lock is intentionally ignored by this library workspace; its resolved bytes are captured locally, not invented as a tracked parent file. Repository/PR text privacy scans find zero forbidden adapter-name matches. Raw evidence stays ignored under `target/automation/`.

## Validation

| Check | Passed |
| --- | ---: |
| image | 151 |
| voxel | 213 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 406 |

Clippy with all features/targets and warnings denied, CPU-only check, minimal/CI-feature WASM checks with warnings denied, workspace formatting and diff checks pass. Hardware tests include ignored cases and exclude explicit benchmarks only.

## Exploratory host allocation sketches

Before changing production, two standalone host sketches checked 1/2/4 workers,
three warmups and 30 rotating measured rounds, validating every result and
capacity outside timing. Both used one 16-byte-record output Vec and counted
the caller as a worker. Ordinary default-value initialization failed to show
a meaningful million-point gain; safe allocator-level zero initialization
was selected for finalized-library testing. These separate sketch binaries
(optimized rustc, then an optimized standalone Cargo package) are exploratory
host diagnostics. They do not establish a complete-call speedup, and cross-
binary timing differences are not treated as controlled comparisons. Every
sketch size is retained below; values are median milliseconds.

| Sketch | N | 1 worker | 2 workers | 4 workers |
| --- | ---: | ---: | ---: | ---: |
| default-value initialization | 65537 | 0.332600 | 0.447350 | 0.627150 |
| default-value initialization | 262143 | 1.410550 | 1.469550 | 1.451400 |
| default-value initialization | 262144 | 1.444650 | 1.445600 | 1.442500 |
| default-value initialization | 524287 | 2.648300 | 2.646900 | 2.603250 |
| default-value initialization | 524288 | 2.713350 | 2.778750 | 2.705000 |
| default-value initialization | 524289 | 2.654750 | 2.649100 | 2.649100 |
| default-value initialization | 1000003 | 5.177150 | 5.271450 | 5.204350 |
| default-value initialization | 2000003 | 10.797850 | 9.939850 | 9.715500 |
| safe zeroed allocation | 65537 | 0.344150 | 0.385200 | 0.417400 |
| safe zeroed allocation | 262143 | 1.202000 | 0.878200 | 0.757700 |
| safe zeroed allocation | 262144 | 1.226200 | 0.874800 | 0.763450 |
| safe zeroed allocation | 524287 | 2.573150 | 1.674500 | 1.369550 |
| safe zeroed allocation | 524288 | 2.343400 | 1.589850 | 1.316950 |
| safe zeroed allocation | 524289 | 2.280050 | 1.537350 | 1.233200 |
| safe zeroed allocation | 1000003 | 4.517650 | 2.925800 | 2.279400 |
| safe zeroed allocation | 2000003 | 9.679100 | 5.543850 | 4.383350 |

## Final paired measurement method

Two consecutive surveys use the same finalized release binary and one
headless runtime per survey. There are 13 cases and four routes, three warmup
pairs followed by 30 measured pairs each. Each route has 15 parent-first and
15 current-first pairs; route order rotates. Inputs and CPU segment oracles
are generated outside timing. Output checks and destruction are excluded,
and every timed result is verified. No tests or builds overlap timed surveys,
no case is removed or retried, and cold compilation is not measured.

- `prep`: original serial preparation versus current default policy, including
  allocation, initialization, CPU-count query when applicable, worker startup,
  filling and joins. The selected worker count is a policy snapshot from the
  runtime's available CPU count, not per-sample thread telemetry.
- `one`: original serial preparation versus the current forced-one-worker
  fallback, a control without a CPU-count query. Both retain the original
  collector; a timing difference is not an algorithmic improvement.
- `resident`: completed public CPU-key call, including preparation, upload,
  sorting and scalar compaction synchronization. Final payload download used
  for checking is excluded from timing.
- `full`: completed public CPU-key-to-CPU-metadata API, including preparation,
  upload, sort/compaction and final payload download.

For host-only routes at <=1,024 points, a sample batches 64 preparations and
divides elapsed time by 64. The common outer output Vec is preallocated outside
timing; inner output allocations and pushes are included. Every host result
is checked against the frozen serial record bytes and exact length/capacity.
All larger host samples use one preparation per sample.

Pattern 0 contains deterministic full-width signed coordinates with every
seventh point at saturation extremes; pattern 1 clamps every input to one
cell; pattern 2 repeats reverse x/modulo y/z coordinates. The unchanged small
inputs, just-below-threshold case and forced-one-worker controls are retained.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples the 30 paired log ratios 10,000 times using seed
`2026101802 + 1000*survey + 10*case_index + route_index` (zero-based case/route
indices) and sorted draws 249/9749. Entirely above/below one is favorable/adverse;
otherwise inconclusive. These shared-host intervals have no multiplicity
correction and do not establish portable speed or workload isolation.

| Route | All 26 comparisons | Large-input 16 comparisons (N>=524,288) |
| --- | --- | --- |
| prep | 16 favorable / 10 inconclusive / 0 adverse | 16 favorable / 0 inconclusive / 0 adverse |
| one | 0 favorable / 20 inconclusive / 6 adverse | 0 favorable / 15 inconclusive / 1 adverse |
| resident | 15 favorable / 11 inconclusive / 0 adverse | 15 favorable / 1 inconclusive / 0 adverse |
| full | 16 favorable / 10 inconclusive / 0 adverse | 16 favorable / 0 inconclusive / 0 adverse |

In the measured default policy, all 16 large-input host and full-call intervals
favor this change (paired ratios 1.719–2.271 and 1.035–1.149 respectively).
Large resident calls have 15 favorable intervals and one inconclusive interval.
The ten small/default-policy controls in each of those routes are inconclusive.
The forced-one-worker route has six adverse intervals: N=1 and N=257 in both
surveys, 1,000,003 one-cell keys in survey 1, and the just-below-threshold case
in survey 2. Source-identical serial arithmetic does not guarantee identical
compiled wrapper/allocator latency. The public small-input route returns
directly to the serial collector, while the explicit one-worker control goes
through the worker planner; single-core complete-call performance is not
established by the large-input parallel measurements. No claim of universally
unchanged serial latency is made, and all adverse control rows remain below.

Host/process identities are excluded from published observations:

- Survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 max_workers=4 min_points=524288 points_per_worker=131072 warmups=3 pairs=30 order=alternating route_order=rotating checks_drop_excluded=true small_preparation_batch=64`.
  pre: 1 busy processes above 0.25 CPU cores, estimated cores [0.97]; GPU/memory-controller utilization samples [[[3, 0]], [[4, 0]]]; observed quiet=False.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [1.02]; GPU/memory-controller utilization samples [[[19, 5]], [[22, 4]]]; observed quiet=False.
- Survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 max_workers=4 min_points=524288 points_per_worker=131072 warmups=3 pairs=30 order=alternating route_order=rotating checks_drop_excluded=true small_preparation_batch=64`.
  pre: 1 busy processes above 0.25 CPU cores, estimated cores [0.96]; GPU/memory-controller utilization samples [[[22, 4]], [[33, 8]]]; observed quiet=False.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [0.99]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.

Pre/post observations cannot establish isolation during every pair. All favorable, adverse and inconclusive rows follow (times in milliseconds):

| Survey | N | Pattern | Selected workers | Route | Parent median | Current median | Paired ratio | 95% interval | Result |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 0 | 0 | 1 | prep | 0.000008 | 0.000008 | 0.918732 | 0.803251–1.000635 | inconclusive |
| 1 | 0 | 0 | 1 | one | 0.000008 | 0.000008 | 1.039185 | 0.950200–1.198737 | inconclusive |
| 1 | 0 | 0 | 1 | resident | 0.001600 | 0.001600 | 1.007321 | 0.888858–1.134104 | inconclusive |
| 1 | 0 | 0 | 1 | full | 0.002500 | 0.002500 | 0.988351 | 0.961045–1.008814 | inconclusive |
| 1 | 1 | 2 | 1 | prep | 0.000047 | 0.000044 | 1.022646 | 0.960697–1.084006 | inconclusive |
| 1 | 1 | 2 | 1 | one | 0.000044 | 0.000047 | 0.954188 | 0.926952–0.981331 | adverse |
| 1 | 1 | 2 | 1 | resident | 1.189800 | 1.213850 | 0.994943 | 0.970838–1.020984 | inconclusive |
| 1 | 1 | 2 | 1 | full | 4.472550 | 4.498250 | 1.000761 | 0.978684–1.025373 | inconclusive |
| 1 | 257 | 2 | 1 | prep | 0.000552 | 0.000535 | 1.031871 | 0.984221–1.083039 | inconclusive |
| 1 | 257 | 2 | 1 | one | 0.000502 | 0.000503 | 0.981199 | 0.963845–0.997916 | adverse |
| 1 | 257 | 2 | 1 | resident | 2.059800 | 2.031050 | 1.025537 | 0.996065–1.058989 | inconclusive |
| 1 | 257 | 2 | 1 | full | 5.123300 | 5.091100 | 1.016452 | 0.987927–1.049994 | inconclusive |
| 1 | 65537 | 0 | 1 | prep | 0.414500 | 0.404850 | 1.016488 | 0.965452–1.070524 | inconclusive |
| 1 | 65537 | 0 | 1 | one | 0.396800 | 0.414350 | 0.963698 | 0.920521–1.006997 | inconclusive |
| 1 | 65537 | 0 | 1 | resident | 3.951250 | 3.875800 | 1.029084 | 0.994945–1.064395 | inconclusive |
| 1 | 65537 | 0 | 1 | full | 8.800150 | 8.958750 | 0.999854 | 0.978428–1.023060 | inconclusive |
| 1 | 524287 | 0 | 1 | prep | 3.138700 | 3.135000 | 1.004535 | 0.952213–1.059150 | inconclusive |
| 1 | 524287 | 0 | 1 | one | 2.945400 | 3.101400 | 0.970677 | 0.936083–1.004464 | inconclusive |
| 1 | 524287 | 0 | 1 | resident | 18.638250 | 18.479200 | 0.992624 | 0.974276–1.012049 | inconclusive |
| 1 | 524287 | 0 | 1 | full | 33.049000 | 33.582300 | 0.990132 | 0.978568–1.001474 | inconclusive |
| 1 | 524288 | 0 | 4 | prep | 3.118150 | 1.516600 | 1.874843 | 1.784568–1.964700 | favorable |
| 1 | 524288 | 0 | 1 | one | 3.124050 | 3.148800 | 1.021171 | 0.984861–1.060557 | inconclusive |
| 1 | 524288 | 0 | 4 | resident | 21.128300 | 20.101250 | 1.059067 | 1.047007–1.071340 | favorable |
| 1 | 524288 | 0 | 4 | full | 36.708750 | 35.724000 | 1.039649 | 1.029763–1.049476 | favorable |
| 1 | 524289 | 0 | 4 | prep | 3.103650 | 1.533800 | 1.935713 | 1.859438–2.008663 | favorable |
| 1 | 524289 | 0 | 1 | one | 3.122400 | 3.170600 | 0.987518 | 0.947240–1.032709 | inconclusive |
| 1 | 524289 | 0 | 4 | resident | 22.940600 | 21.708500 | 1.064662 | 1.046963–1.082304 | favorable |
| 1 | 524289 | 0 | 4 | full | 43.205500 | 41.707200 | 1.039632 | 1.022871–1.057761 | favorable |
| 1 | 1000003 | 0 | 4 | prep | 4.837150 | 2.478250 | 1.929622 | 1.864741–1.997741 | favorable |
| 1 | 1000003 | 0 | 1 | one | 4.462500 | 4.520350 | 0.986368 | 0.956010–1.015818 | inconclusive |
| 1 | 1000003 | 0 | 4 | resident | 32.314750 | 29.856500 | 1.093133 | 1.075860–1.115554 | favorable |
| 1 | 1000003 | 0 | 4 | full | 57.261350 | 54.519300 | 1.058376 | 1.031971–1.089388 | favorable |
| 1 | 1000003 | 1 | 4 | prep | 5.018550 | 2.521500 | 1.936371 | 1.844976–2.030700 | favorable |
| 1 | 1000003 | 1 | 1 | one | 4.437050 | 4.803350 | 0.943841 | 0.910077–0.975766 | adverse |
| 1 | 1000003 | 1 | 4 | resident | 24.652700 | 22.342800 | 1.109992 | 1.089039–1.133303 | favorable |
| 1 | 1000003 | 1 | 4 | full | 35.635000 | 32.682550 | 1.105590 | 1.075210–1.148059 | favorable |
| 1 | 1000003 | 2 | 4 | prep | 5.034200 | 2.520950 | 1.929182 | 1.860377–2.000226 | favorable |
| 1 | 1000003 | 2 | 1 | one | 4.976700 | 4.887050 | 1.001508 | 0.961715–1.045440 | inconclusive |
| 1 | 1000003 | 2 | 4 | resident | 24.886950 | 22.740300 | 1.093684 | 1.081853–1.105876 | favorable |
| 1 | 1000003 | 2 | 4 | full | 36.482400 | 34.501450 | 1.075123 | 1.060746–1.090618 | favorable |
| 1 | 2000003 | 0 | 4 | prep | 10.528550 | 5.234200 | 2.017846 | 1.930229–2.113382 | favorable |
| 1 | 2000003 | 0 | 1 | one | 10.468500 | 10.839300 | 1.027601 | 0.985999–1.071597 | inconclusive |
| 1 | 2000003 | 0 | 4 | resident | 58.685850 | 50.103700 | 1.147691 | 1.120314–1.180468 | favorable |
| 1 | 2000003 | 0 | 4 | full | 109.086650 | 102.824350 | 1.034881 | 1.011281–1.058570 | favorable |
| 1 | 2000003 | 1 | 4 | prep | 10.927600 | 4.856450 | 2.235020 | 2.088252–2.392823 | favorable |
| 1 | 2000003 | 1 | 1 | one | 10.802900 | 11.257850 | 0.978783 | 0.927899–1.035726 | inconclusive |
| 1 | 2000003 | 1 | 4 | resident | 41.614300 | 37.250350 | 1.140391 | 1.115035–1.167047 | favorable |
| 1 | 2000003 | 1 | 4 | full | 64.529550 | 59.065900 | 1.148942 | 1.099703–1.216159 | favorable |
| 1 | 2000003 | 2 | 4 | prep | 11.110400 | 4.953150 | 2.271381 | 2.141271–2.430611 | favorable |
| 1 | 2000003 | 2 | 1 | one | 10.212150 | 9.939550 | 1.008226 | 0.955972–1.059493 | inconclusive |
| 1 | 2000003 | 2 | 4 | resident | 42.092100 | 34.973900 | 1.159347 | 1.079141–1.244545 | favorable |
| 1 | 2000003 | 2 | 4 | full | 63.960250 | 56.159700 | 1.122774 | 1.078949–1.165579 | favorable |
| 2 | 0 | 0 | 1 | prep | 0.000008 | 0.000008 | 0.963492 | 0.921438–1.007466 | inconclusive |
| 2 | 0 | 0 | 1 | one | 0.000008 | 0.000008 | 0.956352 | 0.901105–1.014987 | inconclusive |
| 2 | 0 | 0 | 1 | resident | 0.001600 | 0.001600 | 0.997981 | 0.978845–1.017359 | inconclusive |
| 2 | 0 | 0 | 1 | full | 0.002400 | 0.002400 | 1.001420 | 0.993219–1.011062 | inconclusive |
| 2 | 1 | 2 | 1 | prep | 0.000074 | 0.000075 | 0.981981 | 0.874128–1.095691 | inconclusive |
| 2 | 1 | 2 | 1 | one | 0.000059 | 0.000059 | 0.906701 | 0.861594–0.950339 | adverse |
| 2 | 1 | 2 | 1 | resident | 1.287100 | 1.269250 | 1.017963 | 0.951544–1.086175 | inconclusive |
| 2 | 1 | 2 | 1 | full | 4.666150 | 4.635250 | 1.016377 | 0.982688–1.054862 | inconclusive |
| 2 | 257 | 2 | 1 | prep | 0.000534 | 0.000545 | 0.959565 | 0.864000–1.033890 | inconclusive |
| 2 | 257 | 2 | 1 | one | 0.000497 | 0.000506 | 0.971099 | 0.945164–0.995621 | adverse |
| 2 | 257 | 2 | 1 | resident | 2.252650 | 2.217100 | 1.044770 | 0.982840–1.112784 | inconclusive |
| 2 | 257 | 2 | 1 | full | 5.614200 | 5.461500 | 1.010776 | 0.964723–1.055924 | inconclusive |
| 2 | 65537 | 0 | 1 | prep | 0.407150 | 0.404900 | 1.003165 | 0.937117–1.072949 | inconclusive |
| 2 | 65537 | 0 | 1 | one | 0.373900 | 0.380000 | 0.972624 | 0.909094–1.037146 | inconclusive |
| 2 | 65537 | 0 | 1 | resident | 3.940550 | 3.946950 | 0.979723 | 0.894409–1.074749 | inconclusive |
| 2 | 65537 | 0 | 1 | full | 9.234600 | 9.261950 | 0.995653 | 0.944921–1.048801 | inconclusive |
| 2 | 524287 | 0 | 1 | prep | 3.056250 | 2.946600 | 1.021269 | 0.967265–1.080268 | inconclusive |
| 2 | 524287 | 0 | 1 | one | 2.813500 | 2.912100 | 0.944205 | 0.906169–0.982177 | adverse |
| 2 | 524287 | 0 | 1 | resident | 13.700200 | 13.521350 | 0.998051 | 0.908095–1.098663 | inconclusive |
| 2 | 524287 | 0 | 1 | full | 30.176100 | 30.150550 | 1.016708 | 0.969470–1.064689 | inconclusive |
| 2 | 524288 | 0 | 4 | prep | 3.010700 | 1.767000 | 1.754139 | 1.659584–1.856876 | favorable |
| 2 | 524288 | 0 | 1 | one | 3.072500 | 2.981700 | 1.021046 | 0.974362–1.069467 | inconclusive |
| 2 | 524288 | 0 | 4 | resident | 13.048000 | 12.307000 | 1.061156 | 0.980895–1.148014 | inconclusive |
| 2 | 524288 | 0 | 4 | full | 31.476950 | 29.204750 | 1.069408 | 1.018950–1.121049 | favorable |
| 2 | 524289 | 0 | 4 | prep | 3.024250 | 1.794150 | 1.719264 | 1.630594–1.817367 | favorable |
| 2 | 524289 | 0 | 1 | one | 2.936400 | 2.970000 | 1.008607 | 0.964600–1.059015 | inconclusive |
| 2 | 524289 | 0 | 4 | resident | 13.621000 | 11.579300 | 1.182769 | 1.086620–1.283443 | favorable |
| 2 | 524289 | 0 | 4 | full | 34.945550 | 34.319050 | 1.049566 | 1.014133–1.084150 | favorable |
| 2 | 1000003 | 0 | 4 | prep | 5.781050 | 2.818550 | 2.041325 | 1.950638–2.134849 | favorable |
| 2 | 1000003 | 0 | 1 | one | 5.532700 | 5.643300 | 1.008624 | 0.954932–1.066251 | inconclusive |
| 2 | 1000003 | 0 | 4 | resident | 23.096050 | 19.579000 | 1.131829 | 1.056388–1.208085 | favorable |
| 2 | 1000003 | 0 | 4 | full | 55.568450 | 51.289850 | 1.096718 | 1.057968–1.135848 | favorable |
| 2 | 1000003 | 1 | 4 | prep | 5.839500 | 2.831850 | 2.019305 | 1.926466–2.118955 | favorable |
| 2 | 1000003 | 1 | 1 | one | 5.968800 | 6.346900 | 0.953187 | 0.904019–1.003320 | inconclusive |
| 2 | 1000003 | 1 | 4 | resident | 25.877500 | 22.953600 | 1.113962 | 1.070771–1.156853 | favorable |
| 2 | 1000003 | 1 | 4 | full | 38.505400 | 35.705500 | 1.080587 | 1.044753–1.117333 | favorable |
| 2 | 1000003 | 2 | 4 | prep | 5.875500 | 2.837500 | 2.053888 | 1.916769–2.173265 | favorable |
| 2 | 1000003 | 2 | 1 | one | 6.101500 | 5.820500 | 1.040294 | 0.974450–1.111014 | inconclusive |
| 2 | 1000003 | 2 | 4 | resident | 29.931000 | 27.377550 | 1.101324 | 1.085764–1.116023 | favorable |
| 2 | 1000003 | 2 | 4 | full | 44.865100 | 40.097800 | 1.114811 | 1.080671–1.158875 | favorable |
| 2 | 2000003 | 0 | 4 | prep | 10.497800 | 4.915350 | 2.171941 | 2.080826–2.267127 | favorable |
| 2 | 2000003 | 0 | 1 | one | 10.240150 | 10.236050 | 1.006963 | 0.964821–1.052675 | inconclusive |
| 2 | 2000003 | 0 | 4 | resident | 55.235050 | 50.075700 | 1.101989 | 1.089379–1.114795 | favorable |
| 2 | 2000003 | 0 | 4 | full | 107.307000 | 101.281850 | 1.059315 | 1.041937–1.076283 | favorable |
| 2 | 2000003 | 1 | 4 | prep | 11.353450 | 5.304450 | 2.117899 | 1.957507–2.239061 | favorable |
| 2 | 2000003 | 1 | 1 | one | 11.962750 | 11.689800 | 1.013163 | 0.965078–1.070245 | inconclusive |
| 2 | 2000003 | 1 | 4 | resident | 45.744150 | 38.660600 | 1.174148 | 1.147219–1.204076 | favorable |
| 2 | 2000003 | 1 | 4 | full | 68.632250 | 62.612050 | 1.091884 | 1.069707–1.115379 | favorable |
| 2 | 2000003 | 2 | 4 | prep | 11.599050 | 5.080500 | 2.261787 | 2.161673–2.366119 | favorable |
| 2 | 2000003 | 2 | 1 | one | 10.373500 | 10.483950 | 1.019058 | 0.962358–1.075246 | inconclusive |
| 2 | 2000003 | 2 | 4 | resident | 45.932100 | 39.988850 | 1.158692 | 1.130896–1.191173 | favorable |
| 2 | 2000003 | 2 | 4 | full | 68.454400 | 61.586500 | 1.096500 | 1.077308–1.115756 | favorable |

Claims are limited to these individual comparisons. The preserved small-input and one-worker controls can fluctuate even with identical computation. Extra threads are bounded per call and can compete with other callers; host-only gains alone do not prove a transfer-inclusive improvement.
