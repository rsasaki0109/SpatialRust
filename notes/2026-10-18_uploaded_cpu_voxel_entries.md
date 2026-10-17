# Upload CPU voxel keys as owned sort entries

Date: 2026-10-18. Named slice: CPU-key voxel entry upload. Parent:
`1034f7ee9979ad642bb7e8fa3f3c04299533b992` (merged PR #265).

## Change and contract

`build_voxel_segments_gpu_from_keys` already clamps each CPU coordinate to i32
and uploads a 16-byte record per point. That same pass now puts the increasing
source index in the fourth word instead of zero padding. The resulting owned
STORAGE | COPY_SRC buffer is passed directly to the existing prebuilt-entry
radix sort and unchanged segment compactor. No second device entry buffer is
created. Both public CPU-key entry points retain their signatures; the full
wrapper still explicitly downloads metadata. Empty handling and u32/device
limits are checked before host preparation, as in the parent.

For N>0, one device buffer creation with descriptor size 16*N is removed. At
N=1 the build uniform (16 bytes), bind group, build pass and submission are
also removed. For N>1, radix passes (72), submissions (one sort submission),
scratch descriptors, uniform/group counts and payload transfer size stay the
same. The host preparation vector remains 16*N bytes with identical length
and capacity in the checked fixtures. CPU-to-GPU upload remains 16*N; nonempty
compaction still synchronizes one four-byte scalar. This is a source-level
allocation/lifetime improvement, not a measured physical/peak VRAM claim.

The GPU-key buffer API, its private first scatter, singleton path, pipeline
caches, every WGSL shader, frame profiling preflight, scan/compactor and image
processing are unchanged. No feature, dependency, public API, hidden transfer,
CPU worker pool, automatic placement or operational admission gate is added.

## Correctness and provenance

Four new native tests compare all signed saturation combinations, exact host
record bytes/index sequence, 16-byte layout/capacity, empty/singleton, workgroup
tails, the 524,288/524,289 radix prefix threshold and 1,000,003-point inputs.
Mixed signed/extreme, all-clamped-to-one-cell and repeated reverse-key fixtures
cover complete immediate-parent/public CPU APIs, independently CPU-built
segments, raw GPU indices/keys/starts and raw prebuilt sorted entries. Results
remain owned after host mutation, pool clear/reuse and four concurrent calls.
Fresh/warm/zero-budget pool requests and bytes remain identical; unchanged GPU
input buffers remain immutable. Instrumented primitive composition verifies
the singleton build stage/query pair disappears; there is no new public CPU
profiler API. Invalid maximum count validation remains before preparation.

The frozen reference copies the complete immediate-parent CPU-key entry
points, changing names only. It delegates to the original GPU-key builder,
whose entire source and dependencies are independently checked unchanged.
An independent audit verifies the exact allowed production transformations,
the original host clamp expression, frozen functions and all other sources.
The host test checks 38,180,232 parent/clamp/index words. Complete parent/CPU
and retained-output comparisons check 68,740,576 raw index/key/start values
across 30 parent/current API fixture pairs, in addition to uploaded bytes and
the independently sorted prebuilt-entry fixture.

There are 558 unchanged original GPU/image source files and 565 source/executable SHA-256 values. The finalized release executable and all source hashes are identical before/after both surveys and broad validation. Repository/PR text privacy scans report zero forbidden adapter-name matches. Evidence receipts and raw local host diagnostics stay ignored under `target/automation/`.

## Validation

| Check | Passed |
| --- | ---: |
| image | 151 |
| voxel | 208 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 401 |

Clippy with all features/targets and warnings denied, CPU-only check, minimal and CI-feature WASM checks with warnings denied, workspace formatting and diff checks pass. Native GPU tests include ignored hardware checks while excluding only explicit benchmarks.

## Paired measurement method

Two consecutive surveys use the same finalized release binary, one headless
runtime per survey, three warmup pairs followed by 30 measured pairs per case
and route. Each route has 15 parent-first and 15 current-first pairs; route
order rotates. Inputs and CPU expectations are prepared outside timing.
Output checks and destruction are excluded, and every timed result is checked.
The complete parent/current functions share unchanged shader/runtime machinery.
No builds or tests overlap timed surveys, no case is selected out and there
is no retry survey. Pipelines/pools are warm; cold compilation is not measured.

Routes:

- `prep`: host clamp/record allocation alone. For <=1,024 points a sample batches
  64 preparations, dividing elapsed time by 64. Outer Vec capacity is allocated
  before timing for both variants; record allocation, clamp/index writes and
  common outer pushes are included, and all results are checked afterward.
- `resident`: completed CPU-key public call, including host preparation, upload,
  sorting and compaction's scalar synchronization; payload download for checking
  is excluded. The output remains GPU resident at the timer endpoint.
- `full`: completed public CPU-key-to-CPU-metadata call, including upload and
  final metadata download. This is the relevant transfer-inclusive diagnostic.

Patterns: 0 is deterministic full-width signed coordinates with every seventh
point at saturation extremes; 1 clamps every input to the same cell; 2 repeats
reverse x/modulo y/z coordinates. N=0 and N=1 controls are retained. Every
survey covers the same 12 cases shown below. Values are milliseconds.

Ratios are paired geometric means of parent/current, so >1 favors this change.
Each 95% interval resamples 30 paired log ratios 10,000 times, with deterministic
seed `2026101801 + 1000*survey + 10*case_index + route_index` and sorted draws
249/9749. Intervals entirely above/below one are favorable/adverse; otherwise
inconclusive. These are shared-host diagnostics without multiplicity correction,
not portable speed guarantees or evidence of complete workload isolation.

| Route | Favorable / inconclusive / adverse comparisons |
| --- | --- |
| prep | 0 favorable / 21 inconclusive / 3 adverse |
| resident | 6 favorable / 17 inconclusive / 1 adverse |
| full | 6 favorable / 18 inconclusive / 0 adverse |

These totals include the unchanged empty-input controls. The favorable full
N=0 result in survey 2 cannot be attributed to the entry change. At 524,288
and 524,289 points the full-call ratios are favorable in both surveys
(1.032–1.069); all six million-point full-call intervals cross one. One
257-point resident result is adverse in survey 2 (ratio 0.948), while its
full-call interval crosses one. Host preparation has three adverse results
(256 mixed keys, 65,537 clamped one-cell keys, and 257 mixed keys); constructing
indices is additional CPU work, so no host-preparation speedup is claimed.

Pre/post host observations (process identities are excluded from publication):

- Survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 order=alternating route_order=rotating checks_drop_excluded=true small_preparation_batch=64`.
  pre: 2 busy processes above 0.25 CPU cores, estimated cores [0.35, 0.92]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 4 busy processes above 0.25 CPU cores, estimated cores [0.3, 0.99, 0.99, 0.97]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
- Survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 order=alternating route_order=rotating checks_drop_excluded=true small_preparation_batch=64`.
  pre: 2 busy processes above 0.25 CPU cores, estimated cores [0.92, 1.02]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 2 busy processes above 0.25 CPU cores, estimated cores [0.44, 1.02]; GPU/memory-controller utilization samples [[[1, 0]], [[0, 0]]]; observed quiet=False.

Pre/post sampling cannot establish isolation during every pair. All rows, including adverse and inconclusive results, are retained:

| Survey | N | Pattern | Route | Parent median ms | Current median ms | Paired ratio | 95% interval | Result |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 0 | 0 | prep | 0.000007 | 0.000007 | 1.000000 | 0.935248–1.069235 | inconclusive |
| 1 | 0 | 0 | resident | 0.002000 | 0.002000 | 0.996828 | 0.985544–1.008249 | inconclusive |
| 1 | 0 | 0 | full | 0.003100 | 0.003100 | 1.001198 | 0.991430–1.011238 | inconclusive |
| 1 | 1 | 2 | prep | 0.000052 | 0.000051 | 0.988834 | 0.920091–1.059373 | inconclusive |
| 1 | 1 | 2 | resident | 1.365100 | 1.286650 | 1.026864 | 0.975533–1.072205 | inconclusive |
| 1 | 1 | 2 | full | 4.939500 | 4.826750 | 1.040509 | 1.005264–1.086555 | favorable |
| 1 | 2 | 0 | prep | 0.000053 | 0.000052 | 0.986148 | 0.927911–1.047608 | inconclusive |
| 1 | 2 | 0 | resident | 2.213450 | 2.251200 | 0.967952 | 0.903869–1.032328 | inconclusive |
| 1 | 2 | 0 | full | 5.566050 | 5.548800 | 0.995217 | 0.962306–1.029132 | inconclusive |
| 1 | 256 | 2 | prep | 0.000542 | 0.000573 | 0.944779 | 0.902804–0.994841 | adverse |
| 1 | 256 | 2 | resident | 2.279750 | 2.275300 | 1.002948 | 0.952448–1.049444 | inconclusive |
| 1 | 256 | 2 | full | 5.485300 | 5.625800 | 0.999616 | 0.962430–1.039982 | inconclusive |
| 1 | 257 | 2 | prep | 0.000901 | 0.000911 | 0.976546 | 0.927276–1.035113 | inconclusive |
| 1 | 257 | 2 | resident | 3.638500 | 3.984600 | 0.982766 | 0.901009–1.058399 | inconclusive |
| 1 | 257 | 2 | full | 8.092800 | 7.603900 | 1.011934 | 0.967979–1.058266 | inconclusive |
| 1 | 65537 | 0 | prep | 0.453850 | 0.468450 | 1.021968 | 0.942884–1.108056 | inconclusive |
| 1 | 65537 | 0 | resident | 7.196650 | 7.119000 | 1.025845 | 0.991964–1.061466 | inconclusive |
| 1 | 65537 | 0 | full | 13.876950 | 13.670750 | 1.013523 | 0.989161–1.037418 | inconclusive |
| 1 | 65537 | 1 | prep | 0.462900 | 0.490200 | 0.934333 | 0.894810–0.973829 | adverse |
| 1 | 65537 | 1 | resident | 6.682850 | 6.457600 | 1.013680 | 0.970887–1.051719 | inconclusive |
| 1 | 65537 | 1 | full | 11.507250 | 11.609500 | 0.998214 | 0.975070–1.022025 | inconclusive |
| 1 | 524288 | 0 | prep | 3.502050 | 3.311700 | 1.012549 | 0.958731–1.072516 | inconclusive |
| 1 | 524288 | 0 | resident | 21.782050 | 21.247250 | 1.048496 | 1.026319–1.070446 | favorable |
| 1 | 524288 | 0 | full | 42.732100 | 39.817600 | 1.068652 | 1.043800–1.096336 | favorable |
| 1 | 524289 | 0 | prep | 3.082900 | 2.898100 | 1.017546 | 0.950897–1.087083 | inconclusive |
| 1 | 524289 | 0 | resident | 26.939800 | 24.930700 | 1.064086 | 1.046301–1.081778 | favorable |
| 1 | 524289 | 0 | full | 49.391200 | 47.003100 | 1.068460 | 1.048321–1.090038 | favorable |
| 1 | 1000003 | 0 | prep | 5.632300 | 5.656950 | 0.966347 | 0.922998–1.013863 | inconclusive |
| 1 | 1000003 | 0 | resident | 33.476100 | 33.155950 | 1.008800 | 0.991677–1.025899 | inconclusive |
| 1 | 1000003 | 0 | full | 60.011550 | 59.454600 | 1.000866 | 0.982394–1.019885 | inconclusive |
| 1 | 1000003 | 1 | prep | 5.602200 | 5.438100 | 0.986398 | 0.935288–1.038982 | inconclusive |
| 1 | 1000003 | 1 | resident | 24.255000 | 23.683750 | 1.015327 | 0.997475–1.033851 | inconclusive |
| 1 | 1000003 | 1 | full | 35.173000 | 35.289050 | 1.006289 | 0.990401–1.023101 | inconclusive |
| 1 | 1000003 | 2 | prep | 5.860100 | 5.919600 | 0.988884 | 0.934146–1.046889 | inconclusive |
| 1 | 1000003 | 2 | resident | 28.050100 | 27.209400 | 1.026223 | 1.002792–1.048541 | favorable |
| 1 | 1000003 | 2 | full | 41.329400 | 41.296700 | 1.002423 | 0.986899–1.020038 | inconclusive |
| 2 | 0 | 0 | prep | 0.000008 | 0.000006 | 1.045640 | 0.977933–1.118034 | inconclusive |
| 2 | 0 | 0 | resident | 0.001900 | 0.001900 | 1.005319 | 0.990390–1.019080 | inconclusive |
| 2 | 0 | 0 | full | 0.003000 | 0.003000 | 1.013503 | 1.003638–1.022631 | favorable |
| 2 | 1 | 2 | prep | 0.000053 | 0.000052 | 0.992207 | 0.920730–1.061457 | inconclusive |
| 2 | 1 | 2 | resident | 1.391000 | 1.313550 | 1.056393 | 1.003090–1.111503 | favorable |
| 2 | 1 | 2 | full | 4.846150 | 4.731950 | 1.000557 | 0.960754–1.037117 | inconclusive |
| 2 | 2 | 0 | prep | 0.000055 | 0.000055 | 0.960897 | 0.906063–1.014457 | inconclusive |
| 2 | 2 | 0 | resident | 2.447400 | 2.287700 | 1.038726 | 0.970188–1.114893 | inconclusive |
| 2 | 2 | 0 | full | 5.667850 | 5.607650 | 1.021136 | 0.990734–1.053678 | inconclusive |
| 2 | 256 | 2 | prep | 0.000576 | 0.000590 | 0.982019 | 0.933447–1.032977 | inconclusive |
| 2 | 256 | 2 | resident | 2.484200 | 2.531600 | 1.024256 | 0.972344–1.083551 | inconclusive |
| 2 | 256 | 2 | full | 5.735000 | 6.101250 | 0.997315 | 0.967231–1.026710 | inconclusive |
| 2 | 257 | 2 | prep | 0.000731 | 0.000814 | 0.929114 | 0.874565–0.979079 | adverse |
| 2 | 257 | 2 | resident | 4.382200 | 4.738700 | 0.947666 | 0.901691–0.996957 | adverse |
| 2 | 257 | 2 | full | 8.427500 | 8.230950 | 1.016551 | 0.981529–1.051226 | inconclusive |
| 2 | 65537 | 0 | prep | 0.370500 | 0.380050 | 0.965376 | 0.925823–1.004522 | inconclusive |
| 2 | 65537 | 0 | resident | 6.693100 | 6.604500 | 1.015276 | 0.986782–1.042961 | inconclusive |
| 2 | 65537 | 0 | full | 12.240300 | 11.966150 | 1.018060 | 0.985839–1.058922 | inconclusive |
| 2 | 65537 | 1 | prep | 0.374950 | 0.395750 | 0.972611 | 0.923743–1.021937 | inconclusive |
| 2 | 65537 | 1 | resident | 5.973700 | 5.865150 | 1.027045 | 0.998924–1.056745 | inconclusive |
| 2 | 65537 | 1 | full | 9.728150 | 9.443750 | 1.007232 | 0.979271–1.036056 | inconclusive |
| 2 | 524288 | 0 | prep | 2.575200 | 2.650450 | 0.989265 | 0.929572–1.053840 | inconclusive |
| 2 | 524288 | 0 | resident | 17.989400 | 17.309850 | 1.041834 | 1.013415–1.072397 | favorable |
| 2 | 524288 | 0 | full | 33.536400 | 32.698850 | 1.031714 | 1.007768–1.056731 | favorable |
| 2 | 524289 | 0 | prep | 3.224700 | 3.155250 | 1.006439 | 0.950485–1.072814 | inconclusive |
| 2 | 524289 | 0 | resident | 25.809100 | 24.281650 | 1.085539 | 1.053060–1.136905 | favorable |
| 2 | 524289 | 0 | full | 49.124350 | 45.851250 | 1.055788 | 1.002517–1.114154 | favorable |
| 2 | 1000003 | 0 | prep | 5.405400 | 5.211600 | 1.022016 | 0.976578–1.069563 | inconclusive |
| 2 | 1000003 | 0 | resident | 33.488650 | 32.907750 | 1.002677 | 0.983065–1.021846 | inconclusive |
| 2 | 1000003 | 0 | full | 61.209650 | 60.712300 | 1.009893 | 0.991534–1.030169 | inconclusive |
| 2 | 1000003 | 1 | prep | 5.361650 | 5.425400 | 0.976511 | 0.921804–1.034610 | inconclusive |
| 2 | 1000003 | 1 | resident | 24.604100 | 25.257350 | 0.990063 | 0.966871–1.013813 | inconclusive |
| 2 | 1000003 | 1 | full | 36.548600 | 36.226350 | 1.006676 | 0.980646–1.032592 | inconclusive |
| 2 | 1000003 | 2 | prep | 5.583950 | 5.150250 | 1.017945 | 0.965651–1.075366 | inconclusive |
| 2 | 1000003 | 2 | resident | 25.792550 | 25.965800 | 0.985566 | 0.967555–1.004885 | inconclusive |
| 2 | 1000003 | 2 | full | 38.828650 | 38.446500 | 1.018034 | 0.994209–1.044023 | inconclusive |

The guaranteed change is the removed device entry allocation (and singleton setup). Timing claims are limited to the individual comparisons above; host index preparation has a cost and total latency can remain unresolved or regress. CPU preparation measurements alone do not imply a complete-call speedup.
