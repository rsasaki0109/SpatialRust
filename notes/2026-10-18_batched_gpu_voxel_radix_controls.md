# Batched immutable GPU voxel radix controls

Date: 2026-10-18. Named slice: batched GPU voxel radix controls. Immediate parent:
`e6b39dd6dcd2ab18b2375670d8f403a8556d2c31` (merged PR #268).

## Change and boundaries

Every non-singleton radix invocation previously created 24 separate 16-byte
uniform buffers, one for each four-bit digit on z, y and x. The private recorder
now optionally initializes one immutable aligned uniform buffer containing all
24 original control records. Each original bind group selects a static 16-byte
range. No dynamic offsets, shared mutable controls, queue writes or new shader
features are required. Pending groups/commands retain their original ranges
after the host owner is dropped; concurrent invocations own distinct buffers.

The private layout computes `stride = ceil(16/alignment)*alignment` and
`bytes = 23*stride + 16`, validating the device's nonzero power-of-two alignment,
16-byte uniform binding support, maximum buffer size, checked arithmetic and
host conversion. A 64KiB cap bounds host padding and the control allocation
independently of the point count. Unsupported layouts retain all original
individual buffer creations. Empty and singleton sorts return before the new
helper and preserve their original paths and validation/error precedence.

The 24 bind groups, 72 passes, pipeline selections/cache, fused first scatter,
signed digits, stable local ranks, ping-pong ownership, scratch leases,
submissions and profiler stage/query counts are unchanged. Both prebuilt-entry
and borrowed-device-key routes use the same helper. Public APIs, CPU key
preparation, compaction, frame preflight and logical point-payload/readback
receipts retain their contracts. All shaders and pipeline layouts are unchanged.

## Allocation and memory tradeoff

For accepted layouts, source-level uniform creation requests change from 24
to one. Bind-group creation requests remain 24; no pipeline is added. These
counts are established by the recorder source and the all-slice GPU probe,
not an allocator/driver/VRAM measurement. This is fewer creation requests, not
a uniform-memory saving. The 384 logical control bytes acquire zero padding:
at alignment 256, the buffer/initializer and temporary host vector have 5,904
bytes, 5,520 more than the old 24 logical initializers combined. The measured
runtime uses stride 64: uniform initialization has 1,488 bytes, 1,104 more than
the original logical initializer bytes. Packing also uses a temporary 1,488-byte
host vector. Device physical
allocation granularity and peak VRAM were not measured. Logical point-payload
transfer counters exclude both old and current internal control initializers;
their unchanged values do not imply unchanged internal control upload bytes.

No new shader compilation is introduced, but final timings are warmed and do
not establish cold-runtime latency. The controls remain per invocation rather
than being cached across changing point counts or concurrent callers.

## Correctness and provenance

Five new native tests verify independent 24-record axis/shift/count words,
zero padding, alignment and exact buffer/binding/cap limits, and all fallback
boundaries. A GPU probe reads every old/current control slice into a guarded
output, including maximum-u32 control words and owners dropped before submit.
Complete raw sorts compare an immediate-parent recorder and independent signed
comparison sort. Complete CPU-key/device-key APIs compare the immediate parent
and CPU segments, including empty/singleton/partial groups, signed extrema,
clamping, ties, poisoned key padding, cached-prefix boundaries and 1,000,003
points. Borrowed source words and retained outputs remain independent.

The resource test compares exact profiler stage/query counts, scratch descriptor
totals/allocation deltas, warmed reuse, zero/tiny budgets and invalid preflight
errors. Four synchronized callers with different counts/patterns retain their
own live outputs across pool clear and changed-input reuse. The complete native
references freeze the immediate-parent radix recorder, public sorter and CPU
preparation; only reference module paths, import ordering and test registration
are changed. Runtime/shader dependencies are shared after independent identity
verification, and both references use the same runtime pipeline cache lookup.

RADIX_CONTROLS_LAYOUT: verified 4608 independent control words, zero padding, exact buffer/binding/alignment boundaries and bounded fallbacks.

RADIX_CONTROLS_BINDINGS: verified 800 parent/current GPU control/guard words with all owners dropped before submission.

RADIX_CONTROLS_FULL: verified 114999780 raw parent/current/oracle stable words across 30 complete CPU/device-key cases, both prefix families, untouched borrowed inputs and retained outputs.

RADIX_CONTROLS_RESOURCES: verified identical pass/query counts, scratch descriptors/allocation deltas, warm reuse, zero budgets and pre-allocation rejection.

RADIX_CONTROLS_CONCURRENT: verified four concurrent counts/patterns and independent live outputs after clear and changed-input reuse.

There are 571 unchanged original GPU/image source files and 584 source/manifest/executable SHA-256 values. The finalized release unit executable, source files, manifests and resolved ignored Cargo.lock are identical before/after both timed surveys and broad validation. All original WGSL files and pipeline layouts are byte-identical. Repository/PR privacy scans find zero forbidden adapter-name matches; raw local receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 155 |
| voxel | 218 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 415 |

All-features/all-targets Clippy with warnings denied, CPU-only check, minimal/CI-feature WASM checks with warnings denied, workspace formatting and diff checks pass. Native hardware checks include ignored correctness tests and exclude explicit benchmarks.

## Paired measurement method

Two consecutive surveys use the same finalized release executable. Each uses
one headless runtime, ten cases, three warmup and 30 measured pairs. Route
order rotates across three positions; each parent/current pair is recorded
parent-first/current-first 15 times. Inputs, CPU expectations and unchanged
pipelines are prepared outside timing. Result checks/destruction are excluded.
No builds/tests initiated by this workflow overlap either timed survey, and no
case is retried or filtered. Other shared-host work is not controlled.

- `setup`: synthetic host control/bind-group preparation for all 24 digits,
  averaged over four invocations while retaining their groups. Both variants
  use the same borrowed buffers, original static layout and common descriptor
  vector construction. The production helper supplies current controls, and
  the original individual initializer supplies parent controls. This excludes
  pass encoding, submission and GPU execution; it is not a complete API time.
- `resident`: completed public device-key sort/compaction, including the
  original four-byte scalar synchronization. Payload source upload and metadata
  readback for checking occur outside timing.
- `full`: completed public CPU-key preparation, upload, sort/compaction and
  metadata/indices readback. This is the transfer-inclusive diagnostic.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times with seed
`2026101804 + 1000*survey + 10*case_index + route_index` (zero-based case/route)
and sorted draws 249/9749. Entirely above/below one is favorable/adverse,
otherwise inconclusive. No multiplicity correction is applied. Shared-host
pre/post samples cannot establish isolation during every pair or portable speed.

| Route | All 20 comparisons | Paired ratio range |
| --- | --- | --- |
| setup | 20 favorable / 0 inconclusive / 0 adverse | 1.652988–1.907880 |
| resident | 15 favorable / 5 inconclusive / 0 adverse | 1.002221–1.118631 |
| full | 5 favorable / 15 inconclusive / 0 adverse | 0.987878–1.068307 |

All adverse/inconclusive rows remain in the table below; setup-only results do not establish completed-call gains. Cold-runtime, driver allocation and physical memory costs remain outside this survey.

All 20 synthetic setup comparisons favor batching, at 1.653–1.908. Completed resident calls have 15 favorable and five inconclusive comparisons; their favorable ratios span 1.012–1.119. Transfer-inclusive calls have five favorable and 15 inconclusive comparisons, with no interval wholly adverse. All six 1,000,003-point full-call comparisons are inconclusive, including point estimates below one. The bounded host-setup change is adopted for its reduced creation work and measured resident results; this survey does not establish a general full-call or GPU arithmetic throughput improvement.

Published host observations exclude process identities:

- Survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 uniform_stride=64 uniform_bytes=1488 warmups=3 pairs=30 pair_order=alternating route_order=rotating setup_batch=4 checks_drop_excluded=true`.
  pre: 2 busy processes above 0.25 CPU cores, estimated cores [1.78, 0.99]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 4 busy processes above 0.25 CPU cores, estimated cores [2.1, 1.16, 0.3, 0.26]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
- Survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 uniform_stride=64 uniform_bytes=1488 warmups=3 pairs=30 pair_order=alternating route_order=rotating setup_batch=4 checks_drop_excluded=true`.
  pre: 3 busy processes above 0.25 CPU cores, estimated cores [2.33, 1.21, 0.51]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [0.99]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.

All 60 comparisons follow (times in milliseconds):

| Survey | Points | Pattern (0=random, 1=equal, 2=ties) | Route | Parent median | Current median | Paired ratio | 95% interval | Result |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 2 | 0 | setup | 0.168900 | 0.089650 | 1.907880 | 1.733665–2.115740 | favorable |
| 1 | 2 | 0 | resident | 2.417050 | 2.265800 | 1.073299 | 1.008978–1.145157 | favorable |
| 1 | 2 | 0 | full | 5.953800 | 5.647700 | 1.065916 | 1.024573–1.107963 | favorable |
| 1 | 256 | 2 | setup | 0.165313 | 0.090812 | 1.833981 | 1.694642–1.967712 | favorable |
| 1 | 256 | 2 | resident | 2.500700 | 2.409250 | 1.076125 | 0.993078–1.170771 | inconclusive |
| 1 | 256 | 2 | full | 6.217350 | 6.003800 | 1.068307 | 0.979866–1.173181 | inconclusive |
| 1 | 257 | 2 | setup | 0.252725 | 0.150388 | 1.734014 | 1.616408–1.843636 | favorable |
| 1 | 257 | 2 | resident | 3.143200 | 2.943550 | 1.084512 | 1.029852–1.142632 | favorable |
| 1 | 257 | 2 | full | 6.884250 | 6.685150 | 1.045916 | 1.003574–1.092690 | favorable |
| 1 | 65537 | 0 | setup | 0.270900 | 0.147338 | 1.823342 | 1.716402–1.937368 | favorable |
| 1 | 65537 | 0 | resident | 5.885650 | 5.634850 | 1.049123 | 1.020962–1.079856 | favorable |
| 1 | 65537 | 0 | full | 12.141300 | 11.893200 | 1.013922 | 0.997692–1.029669 | inconclusive |
| 1 | 65537 | 1 | setup | 0.252312 | 0.149850 | 1.709679 | 1.573814–1.851926 | favorable |
| 1 | 65537 | 1 | resident | 5.174050 | 5.062350 | 1.031840 | 1.000455–1.064335 | favorable |
| 1 | 65537 | 1 | full | 10.372200 | 9.915300 | 1.053870 | 1.020783–1.093763 | favorable |
| 1 | 524288 | 0 | setup | 0.215000 | 0.108662 | 1.775244 | 1.611055–1.960820 | favorable |
| 1 | 524288 | 0 | resident | 9.894200 | 9.795600 | 1.029517 | 1.011085–1.049473 | favorable |
| 1 | 524288 | 0 | full | 30.206050 | 29.394300 | 0.998092 | 0.968444–1.023501 | inconclusive |
| 1 | 524289 | 0 | setup | 0.233338 | 0.143313 | 1.652988 | 1.499975–1.813924 | favorable |
| 1 | 524289 | 0 | resident | 13.843450 | 13.667100 | 1.018562 | 1.002307–1.034666 | favorable |
| 1 | 524289 | 0 | full | 36.262050 | 36.461100 | 0.996506 | 0.979268–1.016142 | inconclusive |
| 1 | 1000003 | 0 | setup | 0.266200 | 0.156213 | 1.725006 | 1.555274–1.911823 | favorable |
| 1 | 1000003 | 0 | resident | 22.204550 | 21.846200 | 1.011888 | 1.004528–1.019309 | favorable |
| 1 | 1000003 | 0 | full | 57.345250 | 57.409650 | 1.001279 | 0.986412–1.015034 | inconclusive |
| 1 | 1000003 | 1 | setup | 0.263375 | 0.151587 | 1.829347 | 1.652501–2.019399 | favorable |
| 1 | 1000003 | 1 | resident | 12.677000 | 12.499900 | 1.017663 | 1.008985–1.026677 | favorable |
| 1 | 1000003 | 1 | full | 32.594250 | 32.668500 | 0.998937 | 0.979504–1.018931 | inconclusive |
| 1 | 1000003 | 2 | setup | 0.279500 | 0.168412 | 1.732115 | 1.634882–1.844746 | favorable |
| 1 | 1000003 | 2 | resident | 12.505600 | 12.359450 | 1.016803 | 1.004409–1.030030 | favorable |
| 1 | 1000003 | 2 | full | 31.951650 | 31.803650 | 1.001960 | 0.987682–1.016664 | inconclusive |
| 2 | 2 | 0 | setup | 0.169463 | 0.092162 | 1.796968 | 1.627821–1.965264 | favorable |
| 2 | 2 | 0 | resident | 2.589650 | 2.372650 | 1.118631 | 1.040752–1.211110 | favorable |
| 2 | 2 | 0 | full | 6.239750 | 6.316850 | 1.024617 | 0.971129–1.075574 | inconclusive |
| 2 | 256 | 2 | setup | 0.168413 | 0.099362 | 1.745861 | 1.605585–1.893776 | favorable |
| 2 | 256 | 2 | resident | 2.555850 | 2.435050 | 1.061373 | 0.989364–1.139066 | inconclusive |
| 2 | 256 | 2 | full | 6.265050 | 6.306900 | 1.005789 | 0.947425–1.072430 | inconclusive |
| 2 | 257 | 2 | setup | 0.264125 | 0.153150 | 1.797734 | 1.697276–1.924182 | favorable |
| 2 | 257 | 2 | resident | 3.238300 | 3.067300 | 1.043965 | 1.005103–1.083941 | favorable |
| 2 | 257 | 2 | full | 6.646600 | 6.568000 | 1.022934 | 0.986102–1.060178 | inconclusive |
| 2 | 65537 | 0 | setup | 0.223450 | 0.130638 | 1.733715 | 1.589621–1.886193 | favorable |
| 2 | 65537 | 0 | resident | 5.833200 | 5.745100 | 1.017806 | 0.991217–1.043574 | inconclusive |
| 2 | 65537 | 0 | full | 11.929700 | 11.783650 | 1.037563 | 1.006047–1.076332 | favorable |
| 2 | 65537 | 1 | setup | 0.243512 | 0.139763 | 1.771288 | 1.679918–1.858993 | favorable |
| 2 | 65537 | 1 | resident | 5.240700 | 4.957400 | 1.028792 | 0.973951–1.076258 | inconclusive |
| 2 | 65537 | 1 | full | 9.903750 | 9.582800 | 1.037391 | 1.012547–1.065365 | favorable |
| 2 | 524288 | 0 | setup | 0.254575 | 0.150150 | 1.664026 | 1.551584–1.785875 | favorable |
| 2 | 524288 | 0 | resident | 12.060950 | 11.460100 | 1.028906 | 1.016481–1.042356 | favorable |
| 2 | 524288 | 0 | full | 32.217550 | 31.516400 | 1.004961 | 0.987693–1.022172 | inconclusive |
| 2 | 524289 | 0 | setup | 0.270600 | 0.145350 | 1.742099 | 1.638107–1.851270 | favorable |
| 2 | 524289 | 0 | resident | 15.950750 | 15.657500 | 1.018880 | 1.006586–1.031386 | favorable |
| 2 | 524289 | 0 | full | 41.114650 | 40.614850 | 1.004512 | 0.984040–1.021498 | inconclusive |
| 2 | 1000003 | 0 | setup | 0.257462 | 0.156600 | 1.738261 | 1.636559–1.842225 | favorable |
| 2 | 1000003 | 0 | resident | 19.616700 | 19.367400 | 1.017962 | 1.010103–1.026202 | favorable |
| 2 | 1000003 | 0 | full | 51.753850 | 52.137900 | 0.987878 | 0.971101–1.004083 | inconclusive |
| 2 | 1000003 | 1 | setup | 0.219900 | 0.126200 | 1.773401 | 1.685967–1.868295 | favorable |
| 2 | 1000003 | 1 | resident | 11.515950 | 11.146350 | 1.002221 | 0.956164–1.045459 | inconclusive |
| 2 | 1000003 | 1 | full | 29.087100 | 28.352050 | 1.023433 | 0.969833–1.087813 | inconclusive |
| 2 | 1000003 | 2 | setup | 0.150487 | 0.090150 | 1.749952 | 1.582999–1.935385 | favorable |
| 2 | 1000003 | 2 | resident | 10.627850 | 10.408750 | 1.012763 | 1.001707–1.023531 | favorable |
| 2 | 1000003 | 2 | full | 28.543700 | 28.433050 | 1.001442 | 0.983271–1.018799 | inconclusive |
