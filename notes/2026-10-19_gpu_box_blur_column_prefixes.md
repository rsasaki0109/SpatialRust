# GPU box blur with workgroup-local column prefixes

Date: 2026-10-19. Named slice: GPU box-blur column prefixes.
Immediate parent: `632c6001f14c349691e88ea3f570e254b05287e1` (merged PR #282).

## Change and integer bounds

The original unpacked vertical pass shares each window with the following row,
but repeatedly adds the remaining horizontal sums. The new private alternative
constructs exclusive column prefixes and obtains each valid window by a prefix
difference. It combines independently with the existing horizontal row-prefix
variant. Both operations remain GPU resident within the original two passes.

After unchanged admission, the new vertical variant requires width >=128,
height 512..4096, both kernel dimensions >=259, at least 16,384 bytes of
workgroup storage, and `height*kernel_width*255 <= u32::MAX`. This final bound
is calculated in u64 after the bounded-height check. A valid final box sum does
not by itself bound a whole-column prefix: exceeding this additional bound
selects the unchanged original GPU vertical pass without a new admission error.
Packed kernels, smaller vertical kernels, thin images, unsupported storage and
taller textures retain the original selection. Cutoffs are private heuristics,
not portable crossover guarantees.

A workgroup has 256 lanes and 4096 u32 shared words. Let n be the next power of
two of height and c=min(8,4096/n). Adjacent columns are interleaved as y*c+column.
The group initializes all n*c entries, including padded rows and inactive tail
columns. A work-efficient Blelloch upsweep and downsweep scan each column;
disjoint nodes and barriers separate every tree level. Every subtree, total and
exclusive prefix is <=height*kernel_width*255, so padding to n does not weaken
the exact u32 bound. Independent u64 tests approach the limit at every height.

For inclusive lo/hi, the valid sum is
`prefix[hi]-prefix[lo]+horizontal[hi]`. Monotone prefixes avoid subtraction
underflow. Replicate borders add the first/last horizontal sums times their
original edge masses; zero borders omit those masses. Original area and half-up
rounding admission bound the final sum plus area/2. No float operations, atomics,
approximation, CPU scan or extra global prefix buffer are introduced.

Groups own disjoint c-column tiles. Every lane follows uniform barriers even
for inactive tail columns. A final barrier precedes reuse by the next tile;
subtraction-based tile advancement avoids overflow. The original group cap and
validated pixel count bound storage indices. Each logical pixel is stored once,
with physical RGBA `[value,0,0,0]`.

## Contracts and regression coverage

The seventh program is lazy, with the original uniform/read-only horizontal
storage/output-texture bindings and final-stage name. All six previous shaders
remain byte-identical, as do planning/admission, public API bodies, explicit
transfers, horizontal scratch leases, receipts, metadata, texture extents,
submission, one/two pass counts and timestamp query counts. The additional
16KiB workgroup memory and compiled variant can cost occupancy and first-use
time; physical memory, driver traffic, bandwidth and energy are unmeasured.
There are no new dependencies, features, unsafe or core changes.

Three native tests cover exact shape/capability/overflow boundaries, every CPU
scan height 1..4096, near-u32 prefixes and tiled ownership with capped dispatch.
Raw GPU tests compare the unchanged parent final shader, new shader and an
independent u64 oracle using patterned/full-range horizontal sums. They cover
both borders, selected-range power-of-two edges, maximum admitted kernels, incomplete
column tiles, forced repeated tiles, idle groups, untouched input/canary words
and output canary pixels beyond both dimensions. Full calls also compare the
complete immediate-parent API with u64 rectangular means and physical RGBA,
checking receipts, metadata, profiler queries, all seven lazy programs, stable
cache identity, pool budgets, admission before initialization, concurrent first
use and live output ownership after pool clears/changed-input reuse.

The earlier row-only cache assertions are limited to their original six
programs; their fixtures/oracles are unchanged. All other previous native tests
remain unchanged except registering the new native module. The frozen parent
copies its complete six-program host path and uses its unchanged WGSL sources;
only native-test visibility/imports and explicit cache ownership differ.

- BOX_COLUMN_PREFIX_PUBLIC checked 141934344 complete parent/CPU/RGBA bytes in 92 fixtures; unchanged receipts/pass/query counts, capability/shape fallback, lazy seven-program cache, budgets, errors, concurrent first use and live output ownership.

- BOX_COLUMN_PREFIX_PLAN checked 4480 selection bounds, 16078168 near-u32-limit interleaved prefixes, all heights 1..4096, tiled column ownership and capped dispatches.

- BOX_COLUMN_PREFIX_RAW checked 32812272 parent/u64/physical RGBA/input-canary bytes in 209 fixtures; padding, interleaved tails, near-u32 prefixes, both borders and repeated column tiles.

Final validation: 191 native executions, excluding repeated targeted prechecks.

| Check | Passed tests |
| --- | ---: |
| clippy | 0 |
| image | 188 |
| facade | 2 |
| cpu-only | 0 |
| wasm-minimal | 0 |
| wasm-ci | 0 |
| fmt | 0 |
| diff | 0 |
| facade-gaussian | 1 |

All-feature/all-target warning-denying Clippy, CPU-only, minimal GPU WASM,
CI-feature image/staging WASM with warnings denied, formatting and diff checks
pass. 629 original GPU/image source files are byte-identical.
642 source/manifest/executable SHA-256 values agree before measurements
and after final validation. Executable SHA-256: `b2db23d7a49a7a73e0babc7d8d28195395da8089f30957557978992d8d95a556`.

## Measurement method and limits

Two complete fixed-order surveys contain 18 cases each: 12 selected cases and
6 unchanged controls, for 108 route summaries. Each comparison has three warmups
and 30 retained pairs. The wall-clock routes balance all four positions using
an XOR schedule: complete public blur plus device completion on an uploaded
source, and explicit upload/blur/readback. Separate alternating profiler calls
measure the actual final GPU stage in the complete API, after the horizontal
stage; they assert both stages and 32 query-readback bytes. GPU-stage timestamps
are not complete-call latency. Checks, output drops and explicit recycling occur
after each timed interval. CPU fixture/oracle generation is excluded.

A further two fresh-application-cache surveys have six scenarios and 20
alternating pairs without warmups (12 summaries). The device is retained;
application pipeline caches and image/scratch pools are cleared before each
variant. Timing includes complete job sequences and completion. Runtime creation,
source upload and pool/cache reset are excluded. The dedicated parent cache
object is constructed outside timing; the current runtime's lazy cache object
is initialized inside its first call. Neither driver caches, process state nor
hardware are reset. Pipeline counts count realized application programs, not
physical compiled memory or general process-cold startup.

No source/heuristic changes, sample filters, selected reruns or omitted results
followed these measurements. All collected pairs are retained. The paired
geometric ratio is exp(mean(log(old_ms/new_ms))). Values >1 favor the new path.
Ten thousand paired log-bootstrap draws yield individual 95% percentile bounds
(sorted indices 249 and 9749), without multiplicity adjustment. Warm seeds are
2026101961 + 1000*survey + 10*case + route index (pass/resident/full); cold seeds
are 2026101962 + 1000*survey + case. Zero timestamp pairs would be unresolved;
no replacement, epsilon or pair removal is used. Ratios/intervals, both medians,
all controls and adverse/inconclusive outcomes are reported below.

Runs are serialized against our own builds/tests. Pre/post anonymous utilization
samples cannot prove isolation throughout a run on this shared host. Results
are synthetic local parent comparisons, not competitor comparisons, portable
GPU crossovers or application-wide speed guarantees.

- warm, survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=final_pass_resident_full wall_order=balanced final_order=alternating checks_drop_recycle_excluded=true`; pre/post observed quiet=False; GPU utilization samples=[[[0, 0]], [[0, 0]]]/[[[0, 0]], [[0, 0]]]; busy-process counts=1/1.

- warm, survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=final_pass_resident_full wall_order=balanced final_order=alternating checks_drop_recycle_excluded=true`; pre/post observed quiet=False; GPU utilization samples=[[[0, 0]], [[0, 0]]]/[[[0, 0]], [[0, 0]]]; busy-process counts=1/2.

- fresh application cache, survey 1: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`; pre/post observed quiet=False; GPU utilization samples=[[[0, 0]], [[0, 0]]]/[[[2, 0]], [[0, 0]]]; busy-process counts=2/1.

- fresh application cache, survey 2: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`; pre/post observed quiet=False; GPU utilization samples=[[[0, 0]], [[0, 0]]]/[[[2, 0]], [[0, 0]]]; busy-process counts=1/1.

## Complete results

| Selection | Route | Favorable / inconclusive / adverse / unresolved | Paired ratio range |
| --- | --- | --- | --- |
| selected | pass | 24 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1.374404..12.467589 |
| selected | resident | 20 favorable / 4 inconclusive / 0 adverse / 0 unresolved | 1.000716..4.912074 |
| selected | full | 13 favorable / 11 inconclusive / 0 adverse / 0 unresolved | 0.975527..2.066012 |
| control | pass | 2 favorable / 10 inconclusive / 0 adverse / 0 unresolved | 0.994541..1.018061 |
| control | resident | 1 favorable / 11 inconclusive / 0 adverse / 0 unresolved | 0.919951..1.087705 |
| control | full | 1 favorable / 9 inconclusive / 2 adverse / 0 unresolved | 0.974036..1.042548 |

Fresh application caches: 1 favorable / 4 inconclusive / 7 adverse / 0 unresolved.

Selected final-stage comparisons are favorable in all 24 rows (ratios 1.374404..12.467589). Selected resident calls have 20 favorable and 4 inconclusive rows (ratios 1.000716..4.912074); selected transfer-inclusive calls have 13 favorable and 11 inconclusive rows (ratios 0.975527..2.066012). Neither selected wall-clock route has an adverse interval. Controls include two adverse full-call rows (roughly 1.8% and 2.7% more time); they are retained without attributing causality.

First-use tradeoff: all four mixed-schedule comparisons are adverse, with about 32.5%..36.6% more sequence time while realizing 2 versus 3 programs. Both all-program comparisons are adverse (about 9.0%..12.9% more, 6 versus 7 programs). One column-only cold comparison is adverse (about 3.9% more); its other survey is inconclusive. The second dual-prefix survey is favorable and its first is inconclusive. These startup costs limit the warm speedup claim.

### Warm comparisons

| Survey | Width x height | kw x kh | Zero border | Selected | Route | Old median ms | New median ms | Old/new ratio | 95% interval | Outcome | Seed |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 127 x 512 | 259 x 259 | False | False | pass | 0.032144 | 0.031872 | 1.018061 | 1.000150..1.040435 | favorable | 2026102961 |
| 1 | 127 x 512 | 259 x 259 | False | False | resident | 0.380700 | 0.373850 | 0.983519 | 0.920454..1.038697 | inconclusive | 2026102962 |
| 1 | 127 x 512 | 259 x 259 | False | False | full | 1.315500 | 1.294500 | 1.042548 | 1.005703..1.081954 | favorable | 2026102963 |
| 1 | 128 x 511 | 259 x 259 | False | False | pass | 0.030208 | 0.029904 | 1.012472 | 1.002260..1.022228 | favorable | 2026102971 |
| 1 | 128 x 511 | 259 x 259 | False | False | resident | 0.305400 | 0.316100 | 1.020018 | 0.982899..1.060994 | inconclusive | 2026102972 |
| 1 | 128 x 511 | 259 x 259 | False | False | full | 1.232250 | 1.235250 | 0.984348 | 0.967561..1.001150 | inconclusive | 2026102973 |
| 1 | 128 x 512 | 257 x 259 | False | False | pass | 0.035040 | 0.035072 | 0.997213 | 0.989913..1.004221 | inconclusive | 2026102981 |
| 1 | 128 x 512 | 257 x 259 | False | False | resident | 0.304400 | 0.326850 | 0.940642 | 0.832196..1.023311 | inconclusive | 2026102982 |
| 1 | 128 x 512 | 257 x 259 | False | False | full | 1.282450 | 1.280800 | 0.995445 | 0.968681..1.020870 | inconclusive | 2026102983 |
| 1 | 128 x 512 | 259 x 257 | False | False | pass | 0.028592 | 0.028624 | 0.997889 | 0.987426..1.008368 | inconclusive | 2026102991 |
| 1 | 128 x 512 | 259 x 257 | False | False | resident | 0.290900 | 0.292650 | 0.985825 | 0.966838..1.005405 | inconclusive | 2026102992 |
| 1 | 128 x 512 | 259 x 257 | False | False | full | 1.284800 | 1.296500 | 0.982655 | 0.964793..0.999985 | adverse | 2026102993 |
| 1 | 128 x 4097 | 259 x 259 | False | False | pass | 0.220192 | 0.219776 | 1.002868 | 0.998226..1.007363 | inconclusive | 2026103001 |
| 1 | 128 x 4097 | 259 x 259 | False | False | resident | 0.784100 | 0.776600 | 1.033781 | 0.997842..1.095378 | inconclusive | 2026103002 |
| 1 | 128 x 4097 | 259 x 259 | False | False | full | 2.568400 | 2.595150 | 0.999834 | 0.978529..1.020045 | inconclusive | 2026103003 |
| 1 | 128 x 1024 | 16449 x 259 | False | False | pass | 0.058112 | 0.058016 | 1.000623 | 0.991866..1.008868 | inconclusive | 2026103011 |
| 1 | 128 x 1024 | 16449 x 259 | False | False | resident | 0.361000 | 0.365350 | 1.079638 | 0.974690..1.256872 | inconclusive | 2026103012 |
| 1 | 128 x 1024 | 16449 x 259 | False | False | full | 1.457800 | 1.434400 | 1.034911 | 0.992390..1.085760 | inconclusive | 2026103013 |
| 1 | 128 x 512 | 259 x 259 | False | True | pass | 0.040816 | 0.028240 | 1.440873 | 1.432389..1.449742 | favorable | 2026103021 |
| 1 | 128 x 512 | 259 x 259 | False | True | resident | 0.392200 | 0.388250 | 1.000716 | 0.963215..1.040224 | inconclusive | 2026103022 |
| 1 | 128 x 512 | 259 x 259 | False | True | full | 1.425350 | 1.432050 | 0.984895 | 0.944550..1.021376 | inconclusive | 2026103023 |
| 1 | 129 x 513 | 259 x 259 | True | True | pass | 0.050512 | 0.034448 | 1.475137 | 1.452521..1.497876 | favorable | 2026103031 |
| 1 | 129 x 513 | 259 x 259 | True | True | resident | 0.423400 | 0.421450 | 1.028114 | 0.998932..1.059884 | inconclusive | 2026103032 |
| 1 | 129 x 513 | 259 x 259 | True | True | full | 1.451900 | 1.464850 | 0.986855 | 0.934629..1.032525 | inconclusive | 2026103033 |
| 1 | 512 x 512 | 259 x 259 | False | True | pass | 0.341744 | 0.068704 | 4.886440 | 4.728575..5.043899 | favorable | 2026103041 |
| 1 | 512 x 512 | 259 x 259 | False | True | resident | 0.743050 | 0.493600 | 1.486380 | 1.449906..1.523675 | favorable | 2026103042 |
| 1 | 512 x 512 | 259 x 259 | False | True | full | 2.152150 | 2.148250 | 1.006441 | 0.979776..1.034078 | inconclusive | 2026103043 |
| 1 | 513 x 513 | 511 x 511 | True | True | pass | 0.669120 | 0.102800 | 6.495219 | 6.422256..6.566277 | favorable | 2026103051 |
| 1 | 513 x 513 | 511 x 511 | True | True | resident | 1.181750 | 0.607500 | 1.928467 | 1.891229..1.971381 | favorable | 2026103052 |
| 1 | 513 x 513 | 511 x 511 | True | True | full | 2.558200 | 2.231500 | 1.144454 | 1.111173..1.179537 | favorable | 2026103053 |
| 1 | 1920 x 1080 | 259 x 259 | False | True | pass | 2.440768 | 0.515600 | 4.734440 | 4.731561..4.737587 | favorable | 2026103061 |
| 1 | 1920 x 1080 | 259 x 259 | False | True | resident | 3.192750 | 1.247950 | 2.572513 | 2.503992..2.624227 | favorable | 2026103062 |
| 1 | 1920 x 1080 | 259 x 259 | False | True | full | 9.194900 | 7.199250 | 1.290515 | 1.265377..1.314172 | favorable | 2026103063 |
| 1 | 1920 x 1080 | 259 x 259 | True | True | pass | 2.426416 | 0.507632 | 4.781890 | 4.775406..4.787744 | favorable | 2026103071 |
| 1 | 1920 x 1080 | 259 x 259 | True | True | resident | 3.120800 | 1.194900 | 2.589407 | 2.555665..2.622144 | favorable | 2026103072 |
| 1 | 1920 x 1080 | 259 x 259 | True | True | full | 9.003900 | 7.043550 | 1.271170 | 1.255803..1.286656 | favorable | 2026103073 |
| 1 | 1920 x 1080 | 511 x 511 | False | True | pass | 3.122784 | 0.357120 | 8.744498 | 8.737670..8.751431 | favorable | 2026103081 |
| 1 | 1920 x 1080 | 511 x 511 | False | True | resident | 4.014850 | 1.028450 | 4.074985 | 3.985414..4.169519 | favorable | 2026103082 |
| 1 | 1920 x 1080 | 511 x 511 | False | True | full | 10.050350 | 6.890950 | 1.476833 | 1.440577..1.514451 | favorable | 2026103083 |
| 1 | 3840 x 2160 | 259 x 259 | False | True | pass | 10.722640 | 2.023888 | 5.298481 | 5.283143..5.313914 | favorable | 2026103091 |
| 1 | 3840 x 2160 | 259 x 259 | False | True | resident | 14.767500 | 5.989900 | 2.457409 | 2.442349..2.471477 | favorable | 2026103092 |
| 1 | 3840 x 2160 | 259 x 259 | False | True | full | 30.489450 | 21.789000 | 1.400121 | 1.373054..1.423845 | favorable | 2026103093 |
| 1 | 3840 x 2160 | 511 x 511 | False | True | pass | 25.058384 | 2.011072 | 12.467589 | 12.433261..12.502009 | favorable | 2026103101 |
| 1 | 3840 x 2160 | 511 x 511 | False | True | resident | 28.916900 | 5.990050 | 4.912074 | 4.796810..5.052867 | favorable | 2026103102 |
| 1 | 3840 x 2160 | 511 x 511 | False | True | full | 44.683850 | 21.435050 | 2.066012 | 2.041770..2.088292 | favorable | 2026103103 |
| 1 | 1920 x 1080 | 4099 x 259 | False | True | pass | 2.617264 | 0.558752 | 4.609823 | 4.465076..4.692537 | favorable | 2026103111 |
| 1 | 1920 x 1080 | 4099 x 259 | False | True | resident | 2.311450 | 0.986650 | 2.282502 | 2.116661..2.391997 | favorable | 2026103112 |
| 1 | 1920 x 1080 | 4099 x 259 | False | True | full | 10.561900 | 10.488400 | 1.009313 | 0.984168..1.037869 | inconclusive | 2026103113 |
| 1 | 1920 x 1080 | 4099 x 259 | True | True | pass | 3.646336 | 0.791808 | 4.600848 | 4.583611..4.617195 | favorable | 2026103121 |
| 1 | 1920 x 1080 | 4099 x 259 | True | True | resident | 3.976300 | 1.454700 | 2.706950 | 2.673166..2.742429 | favorable | 2026103122 |
| 1 | 1920 x 1080 | 4099 x 259 | True | True | full | 10.829850 | 10.482050 | 1.009390 | 0.981452..1.036527 | inconclusive | 2026103123 |
| 1 | 128 x 4096 | 4111 x 259 | False | True | pass | 0.523232 | 0.346592 | 1.468285 | 1.439053..1.493568 | favorable | 2026103131 |
| 1 | 128 x 4096 | 4111 x 259 | False | True | resident | 1.476450 | 1.370550 | 1.095623 | 1.040054..1.152403 | favorable | 2026103132 |
| 1 | 128 x 4096 | 4111 x 259 | False | True | full | 7.446600 | 7.428050 | 1.015263 | 0.978345..1.054048 | inconclusive | 2026103133 |
| 2 | 127 x 512 | 259 x 259 | False | False | pass | 0.031968 | 0.031840 | 0.999797 | 0.988507..1.011628 | inconclusive | 2026103961 |
| 2 | 127 x 512 | 259 x 259 | False | False | resident | 0.386950 | 0.389450 | 0.967844 | 0.910262..1.021193 | inconclusive | 2026103962 |
| 2 | 127 x 512 | 259 x 259 | False | False | full | 1.360100 | 1.343050 | 1.020063 | 0.992653..1.052340 | inconclusive | 2026103963 |
| 2 | 128 x 511 | 259 x 259 | False | False | pass | 0.029968 | 0.029952 | 0.995838 | 0.987208..1.004227 | inconclusive | 2026103971 |
| 2 | 128 x 511 | 259 x 259 | False | False | resident | 0.366700 | 0.363000 | 1.014098 | 0.975994..1.056746 | inconclusive | 2026103972 |
| 2 | 128 x 511 | 259 x 259 | False | False | full | 1.369100 | 1.388400 | 1.000061 | 0.965730..1.037192 | inconclusive | 2026103973 |
| 2 | 128 x 512 | 257 x 259 | False | False | pass | 0.034864 | 0.034960 | 0.994541 | 0.987590..1.001552 | inconclusive | 2026103981 |
| 2 | 128 x 512 | 257 x 259 | False | False | resident | 0.324050 | 0.327100 | 0.919951 | 0.804504..1.002600 | inconclusive | 2026103982 |
| 2 | 128 x 512 | 257 x 259 | False | False | full | 1.298000 | 1.353050 | 0.974036 | 0.952106..0.998271 | adverse | 2026103983 |
| 2 | 128 x 512 | 259 x 257 | False | False | pass | 0.028672 | 0.028672 | 1.000441 | 0.989990..1.011095 | inconclusive | 2026103991 |
| 2 | 128 x 512 | 259 x 257 | False | False | resident | 0.291350 | 0.291800 | 1.009979 | 0.985326..1.034768 | inconclusive | 2026103992 |
| 2 | 128 x 512 | 259 x 257 | False | False | full | 1.335950 | 1.290400 | 1.009605 | 0.983926..1.034406 | inconclusive | 2026103993 |
| 2 | 128 x 4097 | 259 x 259 | False | False | pass | 0.219664 | 0.220784 | 0.997262 | 0.993899..1.000491 | inconclusive | 2026104001 |
| 2 | 128 x 4097 | 259 x 259 | False | False | resident | 0.802600 | 0.801750 | 1.018773 | 0.979960..1.078633 | inconclusive | 2026104002 |
| 2 | 128 x 4097 | 259 x 259 | False | False | full | 2.722400 | 2.753900 | 0.991102 | 0.972847..1.009218 | inconclusive | 2026104003 |
| 2 | 128 x 1024 | 16449 x 259 | False | False | pass | 0.075904 | 0.076112 | 1.004645 | 0.997434..1.011845 | inconclusive | 2026104011 |
| 2 | 128 x 1024 | 16449 x 259 | False | False | resident | 0.413250 | 0.371500 | 1.087705 | 1.002384..1.240672 | favorable | 2026104012 |
| 2 | 128 x 1024 | 16449 x 259 | False | False | full | 1.490750 | 1.450800 | 0.998992 | 0.967770..1.032104 | inconclusive | 2026104013 |
| 2 | 128 x 512 | 259 x 259 | False | True | pass | 0.043392 | 0.030080 | 1.438136 | 1.418602..1.458290 | favorable | 2026104021 |
| 2 | 128 x 512 | 259 x 259 | False | True | resident | 0.365950 | 0.356350 | 1.029925 | 0.989623..1.066482 | inconclusive | 2026104022 |
| 2 | 128 x 512 | 259 x 259 | False | True | full | 1.412650 | 1.423050 | 0.993079 | 0.968239..1.019800 | inconclusive | 2026104023 |
| 2 | 129 x 513 | 259 x 259 | True | True | pass | 0.054528 | 0.037808 | 1.449481 | 1.435190..1.463913 | favorable | 2026104031 |
| 2 | 129 x 513 | 259 x 259 | True | True | resident | 0.457250 | 0.425450 | 1.024554 | 0.987519..1.061184 | inconclusive | 2026104032 |
| 2 | 129 x 513 | 259 x 259 | True | True | full | 1.496250 | 1.513200 | 1.005739 | 0.974654..1.040141 | inconclusive | 2026104033 |
| 2 | 512 x 512 | 259 x 259 | False | True | pass | 0.352736 | 0.073104 | 4.799234 | 4.691357..4.905075 | favorable | 2026104041 |
| 2 | 512 x 512 | 259 x 259 | False | True | resident | 0.808300 | 0.528750 | 1.484963 | 1.442639..1.525781 | favorable | 2026104042 |
| 2 | 512 x 512 | 259 x 259 | False | True | full | 2.173300 | 2.156450 | 1.014747 | 0.984824..1.044756 | inconclusive | 2026104043 |
| 2 | 513 x 513 | 511 x 511 | True | True | pass | 0.726864 | 0.110640 | 6.577893 | 6.522143..6.636149 | favorable | 2026104051 |
| 2 | 513 x 513 | 511 x 511 | True | True | resident | 1.243450 | 0.643150 | 1.928604 | 1.882158..1.975859 | favorable | 2026104052 |
| 2 | 513 x 513 | 511 x 511 | True | True | full | 2.630200 | 2.270050 | 1.179041 | 1.141014..1.217737 | favorable | 2026104053 |
| 2 | 1920 x 1080 | 259 x 259 | False | True | pass | 2.601472 | 0.553568 | 4.702717 | 4.696057..4.713812 | favorable | 2026104061 |
| 2 | 1920 x 1080 | 259 x 259 | False | True | resident | 3.669700 | 1.384650 | 2.655207 | 2.596329..2.705079 | favorable | 2026104062 |
| 2 | 1920 x 1080 | 259 x 259 | False | True | full | 9.808100 | 7.548250 | 1.305769 | 1.279574..1.329724 | favorable | 2026104063 |
| 2 | 1920 x 1080 | 259 x 259 | True | True | pass | 2.552256 | 0.534400 | 4.778281 | 4.774399..4.782680 | favorable | 2026104071 |
| 2 | 1920 x 1080 | 259 x 259 | True | True | resident | 3.268750 | 1.249400 | 2.624153 | 2.590722..2.657471 | favorable | 2026104072 |
| 2 | 1920 x 1080 | 259 x 259 | True | True | full | 9.480800 | 7.467200 | 1.268343 | 1.245700..1.291298 | favorable | 2026104073 |
| 2 | 1920 x 1080 | 511 x 511 | False | True | pass | 3.216160 | 0.368576 | 8.735206 | 8.727759..8.742879 | favorable | 2026104081 |
| 2 | 1920 x 1080 | 511 x 511 | False | True | resident | 4.220100 | 1.057200 | 4.081318 | 4.004597..4.163970 | favorable | 2026104082 |
| 2 | 1920 x 1080 | 511 x 511 | False | True | full | 10.464000 | 7.190750 | 1.465778 | 1.426004..1.507104 | favorable | 2026104083 |
| 2 | 3840 x 2160 | 259 x 259 | False | True | pass | 10.705344 | 2.024336 | 5.285244 | 5.268715..5.302113 | favorable | 2026104091 |
| 2 | 3840 x 2160 | 259 x 259 | False | True | resident | 14.774450 | 6.056650 | 2.374935 | 2.241760..2.451286 | favorable | 2026104092 |
| 2 | 3840 x 2160 | 259 x 259 | False | True | full | 31.075700 | 22.288600 | 1.402154 | 1.362194..1.456710 | favorable | 2026104093 |
| 2 | 3840 x 2160 | 511 x 511 | False | True | pass | 24.952464 | 2.009968 | 12.432046 | 12.402393..12.463150 | favorable | 2026104101 |
| 2 | 3840 x 2160 | 511 x 511 | False | True | resident | 29.094950 | 6.103850 | 4.701196 | 4.553754..4.795194 | favorable | 2026104102 |
| 2 | 3840 x 2160 | 511 x 511 | False | True | full | 46.089500 | 23.458900 | 2.004167 | 1.942538..2.094842 | favorable | 2026104103 |
| 2 | 1920 x 1080 | 4099 x 259 | False | True | pass | 3.098832 | 0.662608 | 4.703076 | 4.680818..4.733804 | favorable | 2026104111 |
| 2 | 1920 x 1080 | 4099 x 259 | False | True | resident | 2.316500 | 1.021650 | 2.195413 | 2.037056..2.310231 | favorable | 2026104112 |
| 2 | 1920 x 1080 | 4099 x 259 | False | True | full | 12.718700 | 12.833700 | 0.979867 | 0.950153..1.008663 | inconclusive | 2026104113 |
| 2 | 1920 x 1080 | 4099 x 259 | True | True | pass | 3.648000 | 0.792144 | 4.605315 | 4.601385..4.608895 | favorable | 2026104121 |
| 2 | 1920 x 1080 | 4099 x 259 | True | True | resident | 4.957600 | 2.076100 | 2.499620 | 2.429883..2.572084 | favorable | 2026104122 |
| 2 | 1920 x 1080 | 4099 x 259 | True | True | full | 13.093550 | 11.990500 | 1.070416 | 1.033877..1.107707 | favorable | 2026104123 |
| 2 | 128 x 4096 | 4111 x 259 | False | True | pass | 0.668032 | 0.491456 | 1.374404 | 1.356234..1.394963 | favorable | 2026104131 |
| 2 | 128 x 4096 | 4111 x 259 | False | True | resident | 1.518300 | 1.391850 | 1.123496 | 1.073781..1.177173 | favorable | 2026104132 |
| 2 | 128 x 4096 | 4111 x 259 | False | True | full | 7.769200 | 7.937400 | 0.975527 | 0.942473..1.010192 | inconclusive | 2026104133 |

### Fresh application-cache comparisons

| Survey | Scenario | Programs old/new | Old median ms | New median ms | Old/new ratio | 95% interval | Outcome | Seed |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | small | [2, 2] | 2.321100 | 2.226300 | 1.163262 | 0.978823..1.465820 | inconclusive | 2026102962 |
| 1 | large | [2, 2] | 2.319800 | 2.414550 | 0.965387 | 0.928988..1.000952 | inconclusive | 2026102963 |
| 1 | dual_prefix | [2, 2] | 2.212400 | 2.183600 | 1.015753 | 0.970042..1.066791 | inconclusive | 2026102964 |
| 1 | small_then_large | [2, 3] | 2.596550 | 3.322400 | 0.754831 | 0.720825..0.790674 | adverse | 2026102965 |
| 1 | large_then_small | [2, 3] | 2.392850 | 3.318450 | 0.750928 | 0.728031..0.779417 | adverse | 2026102966 |
| 1 | all_programs | [6, 7] | 6.827800 | 7.479050 | 0.917294 | 0.882279..0.952009 | adverse | 2026102967 |
| 2 | small | [2, 2] | 1.990850 | 2.059800 | 1.091536 | 0.963198..1.338201 | inconclusive | 2026103962 |
| 2 | large | [2, 2] | 2.197250 | 2.276900 | 0.962238 | 0.938364..0.986429 | adverse | 2026103963 |
| 2 | dual_prefix | [2, 2] | 2.171050 | 2.028200 | 1.105606 | 1.061605..1.165735 | favorable | 2026103964 |
| 2 | small_then_large | [2, 3] | 2.534550 | 3.515200 | 0.732108 | 0.712340..0.750451 | adverse | 2026103965 |
| 2 | large_then_small | [2, 3] | 2.493950 | 3.439100 | 0.741944 | 0.719563..0.762896 | adverse | 2026103966 |
| 2 | all_programs | [6, 7] | 6.539700 | 7.393800 | 0.885982 | 0.849939..0.924228 | adverse | 2026103967 |

Small and column-only first use each realize two programs in both versions. Mixing the original vertical schedule with the new one grows two programs to three; the all-program sequence grows six to seven. The dual-prefix case still uses two programs, with both scans sharing the original global intermediate. Fresh-cache costs and unchanged-control variation must be considered alongside warm GPU gains.
