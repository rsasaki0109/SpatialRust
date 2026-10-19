# Capacity-specialized resident GPU neighborhoods

Date: 2026-10-19. Named slice: capacity-specialized GPU neighborhoods.
Immediate parent: `0c6361256c082e77fc2ab0b7a7ec532d2e00818c` (merged PR #283).

## Change and exact selection contract

Nearest-k search previously declared 64 function-local candidate records per
invocation even when k was 1 or 8. The private pipeline selector now chooses
compile-time candidate array extents 1, 8, 16 or 64: k=1, k=2..8, k=9..16 and
k=17..64 respectively. Calls with fewer than 1024 queries retain the original
64-entry candidate array. This private conservative guard follows adverse
measurements on small query batches in the initial experiment. A module-level
WGSL const supplies only the array extent. No override feature or optional GPU capability is required. All admitted k are
bounded by their selected extent; k>64 retains the original admission error.
Radius search retains the original 64-candidate shader extent and arbitrary
admitted output capacity; its candidate array is outside the radius path.

Only the array type changes in the shader body. The query cutoff is a host
choice. Target/query/node layouts,
bindings, 64-lane workgroups, dispatch sizes, traversal stack, branch/plane tests,
distance arithmetic, accumulator insertion/replacement and stable final sort
remain byte-identical. In particular, worst candidates change only on the same
strict comparisons, replacement retains the original accumulator slot and equal
distances retain the original stable order. Small arrays never hold more than
capacity entries. Radius count-only, traversal-prefix truncation, exact totals
and overflow rejection are preserved. There is no heap or altered tie rule.
Backend-specific f32 search-boundary limits in the public docs still apply.

The original shared layout is retained. Each requested nearest extent and the
radius pipeline now has a separate lazy OnceLock. Small query batches share the
64-entry cache slot. A single first-use search
realizes one compute program instead of eagerly compiling both original entry
points. Mixed nearest sizes may realize four programs; including radius may
realize five instead of the original two. Concurrent initialization is protected
by those locks. Empty queries/targets, zero-k and invalid radius gates still
return original zero-initialized owned buffers before cache initialization.

Public signatures, planning/admission/error order, CPU staging, output row
schema, zero unused slots, logical bytes, independent ownership and all explicit
transfers are unchanged. Search still submits one compute pass with no payload
upload/readback/host wait; target/query construction and download remain explicit.
Outputs never use the scratch pool. Core, dependencies, features and unsafe
boundaries do not change. Declared candidate slots are smaller; physical register
allocation, spills, occupancy, compiled memory, bandwidth and energy are unmeasured.

## Regression coverage and provenance

Four new native tests cover every admitted k and each extent boundary, all
nearest-k result words against both the complete parent GPU path and CPU KD-tree,
stable ties/duplicate/line/grid inputs, target sizes smaller than k, radius
count-only and overflowing capacities, unused zero slots, poisoned output guards
and extra idle workgroups. Raw tests force every extent below the cutoff too;
public tests check actual 1023/1024-query selection. Complete public calls check
exact buffers, readbacks,
receipts, empty/error-before-cache behavior, all five lazy identities, extreme
finite coordinates, underflow and signed zeros. Concurrent first use across
all variants and radius verifies retained output ownership after input drops,
pool clears, changed input and different budgets.

The frozen parent contains its exact original WGSL plus complete pipeline/cache
constructor and submitted search path. Only native-test visibility, receiver
syntax and dedicated cache ownership differ. Both parent/current cold timings
include their full actual lazy cache/pipeline constructors. Existing tests and
all downstream algorithm bodies remain unchanged. Native regressions cover
nearest/query/tree staging plus GICP covariances, ISS, MLS and FPFH; five public
smoke targets exercise those consumers.

- NEIGHBOR_CAPACITY_PUBLIC checked 20500 complete parent/CPU rows, exact raw buffers, receipts, empty/error-before-cache behavior, all five lazy programs and overflow/subnormal/signed-zero inputs.

- NEIGHBOR_CAPACITY_RAW checked 16540352 complete parent/current/CPU u32 words in 1312 cases; every k, all array boundaries, stable ties, radius overflow/count-only, zero padding, guards and idle workgroups.

Final validation: 75 native test executions, excluding repeated targeted prechecks.

| Check | Passed tests |
| --- | ---: |
| clippy | 0 |
| nearest | 49 |
| gicp | 8 |
| iss | 4 |
| mls | 4 |
| fpfh | 3 |
| facade | 7 |
| cpu-only | 0 |
| wasm-minimal | 0 |
| wasm-ci | 0 |
| fmt | 0 |
| diff | 0 |

All-feature/all-target warning-denying Clippy, CPU-only, minimal GPU WASM,
CI image/staging WASM with warnings denied, formatting and diff checks pass.
669 original GPU/search/image sources are byte-identical.
686 source/manifest/executable SHA-256 values agree before measurement
and after final validation. Executable SHA-256: `ec681a986e7ee60cf1638d2ee0abd01e426b18f7af6d3bda9d0d77c06b90c205`.

## Measurement method and limits

Two fixed-order surveys each contain 20 cases, for 120 route summaries. Each
comparison retains 30 paired samples after three warmups. Complete resident
wall timings include the public search call and device completion on retained
target/query buffers. Complete full timings include tree upload, query upload,
public search and complete explicit download; CPU tree construction and query
fixture generation are excluded. Tree/query uploads use the same unchanged
production helpers in both paths. Timed wall variants balance all four positions
with an XOR schedule. Correctness checks, result/source drops and CPU oracle
construction are excluded. Search itself reports no payload transfers.

A separate alternating raw-kernel route profiles the actual selected pipeline,
original dispatch and one compute pass with the same uploaded target/queries.
It uses an output with eight additional guard words, compares every parent/current
word after timing and explicitly collects one stage / 16 timestamp bytes. Kernel
timestamps exclude setup, uploads, copies, completion and readback, and are not
public-call latency. The correctness harness's extra idle workgroup is disabled
for profiling. The constant host buffer guard extent is not a production change.

Two further fresh-application-cache surveys have six scenarios, 20 alternating
pairs and no warmups (12 summaries). The device and uploaded target/queries are
retained. Application neighborhood caches are dropped outside timing after device
completion. Both timed paths include complete constructor/layout/shader/pipeline
initialization through a fresh empty OnceLock and all sequence calls/completion.
Output drops/checks are excluded. No driver cache, process or hardware reset is
performed; these are not general process-cold startup measurements. Program
counts are application cache entries, not physical pipeline memory.

All pairs, controls and adverse/inconclusive results are retained. There were no
source/heuristic changes, selected reruns or filters after measuring the final build.
Ratios are exp(mean(log(old_ms/new_ms))); >1 favors the new path. Ten thousand
paired log-bootstrap draws yield individual 95% percentile bounds (sorted indices
249 and 9749), without multiplicity adjustment. Warm seeds are 2026101981 +
1000*survey + 10*case + route index (pass/resident/full); cold seeds are 2026101982
+ 1000*survey + case. Zero timestamp pairs would be unresolved: no epsilon,
replacement or pair removal. Both medians and every interval are reported below.

Measurements are serialized against our own builds/tests. Anonymous pre/post
utilization cannot prove isolation throughout a shared-host run. All results are
local synthetic parent comparisons, not cross-GPU, competitor or application-wide
speed guarantees. Controls retain the original 64-entry GPU computation but share
the new lazy cache structure; host cache lookup/setup differs.

- warm, survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=kernel_resident_full wall_order=balanced kernel_order=alternating checks_drops_excluded=true`; pre/post observed quiet=False; GPU samples=[[[0, 0]], [[0, 0]]]/[[[0, 0]], [[0, 0]]]; busy-process counts=3/2.

- warm, survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=kernel_resident_full wall_order=balanced kernel_order=alternating checks_drops_excluded=true`; pre/post observed quiet=False; GPU samples=[[[0, 0]], [[0, 0]]]/[[[0, 0]], [[0, 0]]]; busy-process counts=4/4.

- fresh application cache, survey 1: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true cache_initialization_included_both=true checks_drops_excluded=true`; pre/post observed quiet=False; GPU samples=[[[0, 0]], [[0, 0]]]/[[[2, 0]], [[0, 0]]]; busy-process counts=4/5.

- fresh application cache, survey 2: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true cache_initialization_included_both=true checks_drops_excluded=true`; pre/post observed quiet=False; GPU samples=[[[0, 0]], [[0, 0]]]/[[[30, 1]], [[35, 1]]]; busy-process counts=5/4.

## Complete results

| Selection | Route | Favorable / inconclusive / adverse / unresolved | Paired ratio range |
| --- | --- | --- | --- |
| smaller array | pass | 20 favorable / 0 inconclusive / 2 adverse / 0 unresolved | 0.990790..1.580626 |
| smaller array | resident | 14 favorable / 7 inconclusive / 1 adverse / 0 unresolved | 0.989732..1.361564 |
| smaller array | full | 7 favorable / 15 inconclusive / 0 adverse / 0 unresolved | 0.991438..1.141052 |
| 64-entry control | pass | 4 favorable / 14 inconclusive / 0 adverse / 0 unresolved | 0.992979..1.004946 |
| 64-entry control | resident | 0 favorable / 17 inconclusive / 1 adverse / 0 unresolved | 0.862351..1.076279 |
| 64-entry control | full | 0 favorable / 17 inconclusive / 1 adverse / 0 unresolved | 0.940982..1.037832 |

Fresh application caches: 8 favorable / 0 inconclusive / 4 adverse / 0 unresolved.

Final smaller-array kernel comparisons: 20 favorable / 0 inconclusive / 2 adverse; ratios 0.990790..1.580626. Resident calls: 14 favorable / 7 inconclusive / 1 adverse; ratios 0.989732..1.361564. Transfer-inclusive calls: 7 favorable / 15 inconclusive / 0 adverse; ratios 0.991438..1.141052. The two adverse kernel intervals and one adverse resident interval involve repeated planar ties at k=8 (roughly 0.9%..1.0% more time); no content-dependent scan is added to detect these inputs. The smaller arrays are not a universal speedup.

First-use tradeoff: all eight single-program comparisons favor the lazy cache (old/new ratios 1.057446..1.499230). All four mixed-nearest/all-program comparisons are adverse: roughly 2.04..2.23 times the old sequence duration while realizing 4 or 5 programs instead of 2. These retained-device application-cache results do not establish general startup latency. Unchanged-array controls include one adverse resident and one adverse full comparison; they are retained without causal attribution.

### Warm comparisons

| Survey | Points / queries | k | Pattern | Radius | Smaller array | Route | Old median ms | New median ms | Old/new ratio | 95% interval | Outcome | Seed |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 / 1 | 1 | 0 | False | False | pass | 0.011488 | 0.011456 | 1.004018 | 0.992395..1.016504 | inconclusive | 2026102981 |
| 1 | 257 / 1 | 1 | 0 | False | False | resident | 0.181700 | 0.183350 | 0.996058 | 0.965265..1.026443 | inconclusive | 2026102982 |
| 1 | 257 / 1 | 1 | 0 | False | False | full | 1.444650 | 1.470300 | 0.975546 | 0.917004..1.023590 | inconclusive | 2026102983 |
| 1 | 4097 / 65 | 1 | 0 | False | False | pass | 0.050112 | 0.049984 | 1.001732 | 0.999371..1.004114 | inconclusive | 2026102991 |
| 1 | 4097 / 65 | 1 | 0 | False | False | resident | 0.247150 | 0.259200 | 0.862351 | 0.656212..1.073568 | inconclusive | 2026102992 |
| 1 | 4097 / 65 | 1 | 0 | False | False | full | 1.606100 | 1.743250 | 0.940982 | 0.850178..1.026383 | inconclusive | 2026102993 |
| 1 | 4097 / 65 | 8 | 0 | False | False | pass | 0.117232 | 0.116800 | 1.003562 | 1.001846..1.005271 | favorable | 2026103001 |
| 1 | 4097 / 65 | 8 | 0 | False | False | resident | 0.287250 | 0.285600 | 0.941126 | 0.818612..1.020413 | inconclusive | 2026103002 |
| 1 | 4097 / 65 | 8 | 0 | False | False | full | 1.377200 | 1.366750 | 1.037832 | 0.994506..1.098234 | inconclusive | 2026103003 |
| 1 | 4097 / 65 | 16 | 0 | False | False | pass | 0.193424 | 0.192736 | 1.003431 | 1.001045..1.005674 | favorable | 2026103011 |
| 1 | 4097 / 65 | 16 | 0 | False | False | resident | 0.332450 | 0.332750 | 1.004135 | 0.991038..1.016291 | inconclusive | 2026103012 |
| 1 | 4097 / 65 | 16 | 0 | False | False | full | 1.195250 | 1.195150 | 0.994462 | 0.969763..1.020295 | inconclusive | 2026103013 |
| 1 | 4097 / 65 | 17 | 0 | False | False | pass | 0.199520 | 0.199856 | 1.000782 | 0.997976..1.003645 | inconclusive | 2026103021 |
| 1 | 4097 / 65 | 17 | 0 | False | False | resident | 0.363250 | 0.362300 | 1.052576 | 0.983182..1.184832 | inconclusive | 2026103022 |
| 1 | 4097 / 65 | 17 | 0 | False | False | full | 1.274800 | 1.292150 | 0.999467 | 0.981667..1.018095 | inconclusive | 2026103023 |
| 1 | 65537 / 1023 | 8 | 0 | False | False | pass | 0.341552 | 0.341936 | 0.997636 | 0.992014..1.003116 | inconclusive | 2026103031 |
| 1 | 65537 / 1023 | 8 | 0 | False | False | resident | 0.525700 | 0.529600 | 0.998500 | 0.981619..1.016220 | inconclusive | 2026103032 |
| 1 | 65537 / 1023 | 8 | 0 | False | False | full | 2.374650 | 2.580150 | 0.955035 | 0.905358..1.009295 | inconclusive | 2026103033 |
| 1 | 65537 / 1024 | 8 | 0 | False | True | pass | 0.690352 | 0.582032 | 1.184132 | 1.168631..1.200440 | favorable | 2026103041 |
| 1 | 65537 / 1024 | 8 | 0 | False | True | resident | 0.539900 | 0.536150 | 1.012809 | 0.995400..1.032154 | inconclusive | 2026103042 |
| 1 | 65537 / 1024 | 8 | 0 | False | True | full | 2.704750 | 2.612400 | 1.000647 | 0.962502..1.041508 | inconclusive | 2026103043 |
| 1 | 65537 / 4097 | 1 | 0 | False | True | pass | 0.420320 | 0.322512 | 1.307942 | 1.300073..1.316037 | favorable | 2026103051 |
| 1 | 65537 / 4097 | 1 | 0 | False | True | resident | 0.618350 | 0.541150 | 1.133973 | 1.113539..1.154537 | favorable | 2026103052 |
| 1 | 65537 / 4097 | 1 | 0 | False | True | full | 2.882200 | 2.890350 | 1.007948 | 0.958460..1.066797 | inconclusive | 2026103053 |
| 1 | 65537 / 4097 | 2 | 0 | False | True | pass | 0.521984 | 0.430512 | 1.215212 | 1.207230..1.222644 | favorable | 2026103061 |
| 1 | 65537 / 4097 | 2 | 0 | False | True | resident | 0.719700 | 0.655500 | 1.101329 | 1.087884..1.117395 | favorable | 2026103062 |
| 1 | 65537 / 4097 | 2 | 0 | False | True | full | 2.773100 | 2.609600 | 1.018823 | 0.983140..1.054732 | inconclusive | 2026103063 |
| 1 | 65537 / 4097 | 8 | 0 | False | True | pass | 1.098032 | 0.844816 | 1.297505 | 1.288396..1.306946 | favorable | 2026103071 |
| 1 | 65537 / 4097 | 8 | 0 | False | True | resident | 1.352400 | 1.105750 | 1.216605 | 1.200165..1.233193 | favorable | 2026103072 |
| 1 | 65537 / 4097 | 8 | 0 | False | True | full | 3.576400 | 3.347350 | 1.059462 | 1.026350..1.089960 | favorable | 2026103073 |
| 1 | 65537 / 4097 | 16 | 0 | False | True | pass | 1.455104 | 1.388544 | 1.049928 | 1.044715..1.055011 | favorable | 2026103081 |
| 1 | 65537 / 4097 | 16 | 0 | False | True | resident | 1.844500 | 1.793500 | 1.023580 | 1.011001..1.035518 | favorable | 2026103082 |
| 1 | 65537 / 4097 | 16 | 0 | False | True | full | 4.026650 | 3.936900 | 1.019625 | 0.996970..1.042258 | inconclusive | 2026103083 |
| 1 | 65537 / 4097 | 64 | 0 | False | False | pass | 4.694864 | 4.723328 | 0.996730 | 0.987897..1.005871 | inconclusive | 2026103091 |
| 1 | 65537 / 4097 | 64 | 0 | False | False | resident | 5.057950 | 5.108050 | 0.987577 | 0.973224..1.000162 | inconclusive | 2026103092 |
| 1 | 65537 / 4097 | 64 | 0 | False | False | full | 10.283150 | 10.428800 | 0.994461 | 0.973736..1.015784 | inconclusive | 2026103093 |
| 1 | 262144 / 16384 | 1 | 0 | False | True | pass | 0.361648 | 0.358592 | 1.007941 | 1.001384..1.014349 | favorable | 2026103101 |
| 1 | 262144 / 16384 | 1 | 0 | False | True | resident | 0.606400 | 0.609250 | 1.012083 | 0.985563..1.039551 | inconclusive | 2026103102 |
| 1 | 262144 / 16384 | 1 | 0 | False | True | full | 9.095950 | 8.299550 | 1.045115 | 1.009639..1.083508 | favorable | 2026103103 |
| 1 | 262144 / 16384 | 8 | 0 | False | True | pass | 1.682976 | 1.437648 | 1.169233 | 1.165345..1.173187 | favorable | 2026103111 |
| 1 | 262144 / 16384 | 8 | 0 | False | True | resident | 1.991650 | 1.745650 | 1.257549 | 1.176132..1.358223 | favorable | 2026103112 |
| 1 | 262144 / 16384 | 8 | 0 | False | True | full | 10.787800 | 10.043100 | 1.082770 | 1.022473..1.144029 | favorable | 2026103113 |
| 1 | 262144 / 16384 | 16 | 0 | False | True | pass | 3.376048 | 3.365264 | 1.004114 | 1.002820..1.005488 | favorable | 2026103121 |
| 1 | 262144 / 16384 | 16 | 0 | False | True | resident | 4.483250 | 4.490150 | 0.995735 | 0.984122..1.005476 | inconclusive | 2026103122 |
| 1 | 262144 / 16384 | 16 | 0 | False | True | full | 12.798050 | 12.639700 | 1.014421 | 0.983087..1.047308 | inconclusive | 2026103123 |
| 1 | 65537 / 4097 | 8 | 1 | False | True | pass | 0.142704 | 0.136608 | 1.052039 | 1.043102..1.062033 | favorable | 2026103131 |
| 1 | 65537 / 4097 | 8 | 1 | False | True | resident | 0.346050 | 0.354500 | 1.020551 | 0.995840..1.044549 | inconclusive | 2026103132 |
| 1 | 65537 / 4097 | 8 | 1 | False | True | full | 2.908250 | 2.853400 | 0.996853 | 0.965366..1.028293 | inconclusive | 2026103133 |
| 1 | 65537 / 4097 | 8 | 2 | False | True | pass | 14.179168 | 14.072960 | 1.007972 | 1.006160..1.010086 | favorable | 2026103141 |
| 1 | 65537 / 4097 | 8 | 2 | False | True | resident | 14.471700 | 14.415350 | 1.004626 | 1.002508..1.006790 | favorable | 2026103142 |
| 1 | 65537 / 4097 | 8 | 2 | False | True | full | 16.670800 | 15.669650 | 1.063539 | 1.053669..1.071909 | favorable | 2026103143 |
| 1 | 65537 / 4097 | 8 | 3 | False | True | pass | 1.853440 | 1.877392 | 0.990790 | 0.985692..0.995991 | adverse | 2026103151 |
| 1 | 65537 / 4097 | 8 | 3 | False | True | resident | 2.086200 | 2.115050 | 0.989732 | 0.980234..1.000039 | inconclusive | 2026103152 |
| 1 | 65537 / 4097 | 8 | 3 | False | True | full | 4.349750 | 4.272800 | 1.007112 | 0.984001..1.032201 | inconclusive | 2026103153 |
| 1 | 4097 / 65 | 8 | 0 | True | False | pass | 0.056528 | 0.056288 | 1.002388 | 0.998899..1.005585 | inconclusive | 2026103161 |
| 1 | 4097 / 65 | 8 | 0 | True | False | resident | 0.222700 | 0.218000 | 1.005908 | 0.973739..1.038504 | inconclusive | 2026103162 |
| 1 | 4097 / 65 | 8 | 0 | True | False | full | 1.295000 | 1.243650 | 1.003112 | 0.976420..1.028695 | inconclusive | 2026103163 |
| 1 | 65537 / 4097 | 65 | 0 | True | False | pass | 0.212080 | 0.212480 | 0.997762 | 0.990758..1.004718 | inconclusive | 2026103171 |
| 1 | 65537 / 4097 | 65 | 0 | True | False | resident | 0.428250 | 0.439300 | 0.979730 | 0.951468..1.007763 | inconclusive | 2026103172 |
| 1 | 65537 / 4097 | 65 | 0 | True | False | full | 3.746900 | 3.773000 | 0.969363 | 0.935463..1.002146 | inconclusive | 2026103173 |
| 2 | 257 / 1 | 1 | 0 | False | False | pass | 0.011424 | 0.011456 | 0.992979 | 0.979197..1.005514 | inconclusive | 2026103981 |
| 2 | 257 / 1 | 1 | 0 | False | False | resident | 0.183400 | 0.187700 | 0.936966 | 0.876073..0.991448 | adverse | 2026103982 |
| 2 | 257 / 1 | 1 | 0 | False | False | full | 1.349000 | 1.372050 | 0.962602 | 0.928260..0.995636 | adverse | 2026103983 |
| 2 | 4097 / 65 | 1 | 0 | False | False | pass | 0.050064 | 0.049952 | 1.002900 | 1.000814..1.005129 | favorable | 2026103991 |
| 2 | 4097 / 65 | 1 | 0 | False | False | resident | 0.203450 | 0.207600 | 0.996100 | 0.952293..1.037076 | inconclusive | 2026103992 |
| 2 | 4097 / 65 | 1 | 0 | False | False | full | 1.345250 | 1.358650 | 0.980669 | 0.946407..1.014125 | inconclusive | 2026103993 |
| 2 | 4097 / 65 | 8 | 0 | False | False | pass | 0.117408 | 0.116800 | 1.004946 | 1.002651..1.007342 | favorable | 2026104001 |
| 2 | 4097 / 65 | 8 | 0 | False | False | resident | 0.267000 | 0.274750 | 0.934041 | 0.823516..1.001000 | inconclusive | 2026104002 |
| 2 | 4097 / 65 | 8 | 0 | False | False | full | 1.255300 | 1.274550 | 0.992623 | 0.980743..1.004728 | inconclusive | 2026104003 |
| 2 | 4097 / 65 | 16 | 0 | False | False | pass | 0.193008 | 0.192144 | 1.002000 | 0.999723..1.004284 | inconclusive | 2026104011 |
| 2 | 4097 / 65 | 16 | 0 | False | False | resident | 0.386650 | 0.378650 | 1.008762 | 0.981245..1.038371 | inconclusive | 2026104012 |
| 2 | 4097 / 65 | 16 | 0 | False | False | full | 1.464250 | 1.482850 | 0.978563 | 0.941633..1.015063 | inconclusive | 2026104013 |
| 2 | 4097 / 65 | 17 | 0 | False | False | pass | 0.199840 | 0.199600 | 1.000763 | 0.998491..1.003124 | inconclusive | 2026104021 |
| 2 | 4097 / 65 | 17 | 0 | False | False | resident | 0.402900 | 0.401350 | 1.076279 | 0.988190..1.222984 | inconclusive | 2026104022 |
| 2 | 4097 / 65 | 17 | 0 | False | False | full | 1.606700 | 1.696300 | 0.972828 | 0.926308..1.018558 | inconclusive | 2026104023 |
| 2 | 65537 / 1023 | 8 | 0 | False | False | pass | 0.679680 | 0.681120 | 0.996741 | 0.991199..1.001578 | inconclusive | 2026104031 |
| 2 | 65537 / 1023 | 8 | 0 | False | False | resident | 0.527350 | 0.526200 | 1.004927 | 0.991635..1.019135 | inconclusive | 2026104032 |
| 2 | 65537 / 1023 | 8 | 0 | False | False | full | 2.520900 | 2.467700 | 1.036165 | 0.987380..1.087178 | inconclusive | 2026104033 |
| 2 | 65537 / 1024 | 8 | 0 | False | True | pass | 0.705136 | 0.614864 | 1.145850 | 1.139126..1.152890 | favorable | 2026104041 |
| 2 | 65537 / 1024 | 8 | 0 | False | True | resident | 0.908100 | 0.880000 | 1.026740 | 1.008327..1.046285 | favorable | 2026104042 |
| 2 | 65537 / 1024 | 8 | 0 | False | True | full | 2.634750 | 2.693900 | 0.991438 | 0.957650..1.026616 | inconclusive | 2026104043 |
| 2 | 65537 / 4097 | 1 | 0 | False | True | pass | 0.441296 | 0.341936 | 1.296194 | 1.288705..1.303400 | favorable | 2026104051 |
| 2 | 65537 / 4097 | 1 | 0 | False | True | resident | 0.647550 | 0.591700 | 1.122812 | 1.093816..1.151794 | favorable | 2026104052 |
| 2 | 65537 / 4097 | 1 | 0 | False | True | full | 3.251500 | 3.073600 | 1.013382 | 0.971050..1.059221 | inconclusive | 2026104053 |
| 2 | 65537 / 4097 | 2 | 0 | False | True | pass | 0.552912 | 0.458752 | 1.241271 | 1.209933..1.280960 | favorable | 2026104061 |
| 2 | 65537 / 4097 | 2 | 0 | False | True | resident | 0.776500 | 0.717400 | 1.075467 | 1.058403..1.092474 | favorable | 2026104062 |
| 2 | 65537 / 4097 | 2 | 0 | False | True | full | 3.319050 | 3.275450 | 1.010244 | 0.970392..1.053697 | inconclusive | 2026104063 |
| 2 | 65537 / 4097 | 8 | 0 | False | True | pass | 1.514896 | 0.960464 | 1.580626 | 1.565100..1.596478 | favorable | 2026104071 |
| 2 | 65537 / 4097 | 8 | 0 | False | True | resident | 1.976000 | 1.446100 | 1.361564 | 1.341662..1.381547 | favorable | 2026104072 |
| 2 | 65537 / 4097 | 8 | 0 | False | True | full | 4.792900 | 4.293000 | 1.141052 | 1.121055..1.162855 | favorable | 2026104073 |
| 2 | 65537 / 4097 | 16 | 0 | False | True | pass | 1.504096 | 1.434208 | 1.050738 | 1.046858..1.055404 | favorable | 2026104081 |
| 2 | 65537 / 4097 | 16 | 0 | False | True | resident | 1.849250 | 1.805800 | 1.046153 | 1.022722..1.084792 | favorable | 2026104082 |
| 2 | 65537 / 4097 | 16 | 0 | False | True | full | 4.519050 | 4.419000 | 1.006875 | 0.985883..1.028699 | inconclusive | 2026104083 |
| 2 | 65537 / 4097 | 64 | 0 | False | False | pass | 4.692128 | 4.717456 | 0.997015 | 0.991416..1.002590 | inconclusive | 2026104091 |
| 2 | 65537 / 4097 | 64 | 0 | False | False | resident | 5.195500 | 5.176500 | 1.003861 | 0.996151..1.011094 | inconclusive | 2026104092 |
| 2 | 65537 / 4097 | 64 | 0 | False | False | full | 10.562050 | 10.474000 | 0.999674 | 0.983334..1.015924 | inconclusive | 2026104093 |
| 2 | 262144 / 16384 | 1 | 0 | False | True | pass | 0.363648 | 0.357600 | 1.011790 | 1.005706..1.017639 | favorable | 2026104101 |
| 2 | 262144 / 16384 | 1 | 0 | False | True | resident | 0.627650 | 0.636050 | 0.991342 | 0.973247..1.010258 | inconclusive | 2026104102 |
| 2 | 262144 / 16384 | 1 | 0 | False | True | full | 8.252850 | 8.346900 | 0.993859 | 0.963085..1.024392 | inconclusive | 2026104103 |
| 2 | 262144 / 16384 | 8 | 0 | False | True | pass | 1.754624 | 1.558656 | 1.125282 | 1.122732..1.127826 | favorable | 2026104111 |
| 2 | 262144 / 16384 | 8 | 0 | False | True | resident | 2.081600 | 1.876150 | 1.218813 | 1.146895..1.306878 | favorable | 2026104112 |
| 2 | 262144 / 16384 | 8 | 0 | False | True | full | 10.998250 | 10.443050 | 1.088688 | 1.043386..1.139910 | favorable | 2026104113 |
| 2 | 262144 / 16384 | 16 | 0 | False | True | pass | 3.562384 | 3.547680 | 1.004835 | 1.002998..1.006597 | favorable | 2026104121 |
| 2 | 262144 / 16384 | 16 | 0 | False | True | resident | 4.701400 | 4.676500 | 1.007404 | 1.000294..1.014444 | favorable | 2026104122 |
| 2 | 262144 / 16384 | 16 | 0 | False | True | full | 14.371350 | 14.239300 | 1.003159 | 0.976303..1.030520 | inconclusive | 2026104123 |
| 2 | 65537 / 4097 | 8 | 1 | False | True | pass | 0.157600 | 0.151728 | 1.040350 | 1.031619..1.049954 | favorable | 2026104131 |
| 2 | 65537 / 4097 | 8 | 1 | False | True | resident | 0.398650 | 0.389950 | 1.021995 | 0.993894..1.051332 | inconclusive | 2026104132 |
| 2 | 65537 / 4097 | 8 | 1 | False | True | full | 3.176300 | 3.065850 | 1.014348 | 0.977451..1.051142 | inconclusive | 2026104133 |
| 2 | 65537 / 4097 | 8 | 2 | False | True | pass | 14.158704 | 14.111376 | 1.004962 | 1.004015..1.005930 | favorable | 2026104141 |
| 2 | 65537 / 4097 | 8 | 2 | False | True | resident | 14.484700 | 14.414350 | 1.006274 | 1.003977..1.008770 | favorable | 2026104142 |
| 2 | 65537 / 4097 | 8 | 2 | False | True | full | 16.541150 | 15.606450 | 1.065787 | 1.059473..1.071947 | favorable | 2026104143 |
| 2 | 65537 / 4097 | 8 | 3 | False | True | pass | 1.852480 | 1.875824 | 0.991171 | 0.987178..0.995037 | adverse | 2026104151 |
| 2 | 65537 / 4097 | 8 | 3 | False | True | resident | 2.126850 | 2.133650 | 0.990208 | 0.980736..0.999281 | adverse | 2026104152 |
| 2 | 65537 / 4097 | 8 | 3 | False | True | full | 4.577850 | 4.535900 | 0.994752 | 0.978754..1.011423 | inconclusive | 2026104153 |
| 2 | 4097 / 65 | 8 | 0 | True | False | pass | 0.056672 | 0.056656 | 1.001282 | 0.997957..1.004543 | inconclusive | 2026104161 |
| 2 | 4097 / 65 | 8 | 0 | True | False | resident | 0.269950 | 0.272350 | 1.004074 | 0.973346..1.036512 | inconclusive | 2026104162 |
| 2 | 4097 / 65 | 8 | 0 | True | False | full | 1.544850 | 1.529600 | 1.009313 | 0.982017..1.036606 | inconclusive | 2026104163 |
| 2 | 65537 / 4097 | 65 | 0 | True | False | pass | 0.211936 | 0.212656 | 1.001179 | 0.996084..1.006439 | inconclusive | 2026104171 |
| 2 | 65537 / 4097 | 65 | 0 | True | False | resident | 0.473350 | 0.468650 | 1.003345 | 0.965103..1.043792 | inconclusive | 2026104172 |
| 2 | 65537 / 4097 | 65 | 0 | True | False | full | 4.079700 | 4.302700 | 0.960520 | 0.915077..1.004403 | inconclusive | 2026104173 |

### Fresh application-cache comparisons

| Survey | Scenario | Programs old/new | Old median ms | New median ms | Old/new ratio | 95% interval | Outcome | Seed |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | nearest_one | [2, 1] | 1.918500 | 1.542950 | 1.499230 | 1.255880..1.919828 | favorable | 2026102982 |
| 1 | nearest_eight | [2, 1] | 2.027050 | 1.752150 | 1.187445 | 1.124644..1.248132 | favorable | 2026102983 |
| 1 | nearest_wide | [2, 1] | 3.236900 | 2.912000 | 1.082784 | 1.016715..1.154588 | favorable | 2026102984 |
| 1 | radius | [2, 1] | 1.791950 | 1.557900 | 1.172585 | 1.040324..1.338867 | favorable | 2026102985 |
| 1 | mixed_nearest | [2, 4] | 3.280700 | 7.042400 | 0.454742 | 0.415297..0.491816 | adverse | 2026102986 |
| 1 | all_programs | [2, 5] | 4.834850 | 10.173100 | 0.483457 | 0.453481..0.520652 | adverse | 2026102987 |
| 2 | nearest_one | [2, 1] | 1.939150 | 1.565250 | 1.358068 | 1.158266..1.709113 | favorable | 2026103982 |
| 2 | nearest_eight | [2, 1] | 1.901100 | 1.554650 | 1.162396 | 1.072415..1.255997 | favorable | 2026103983 |
| 2 | nearest_wide | [2, 1] | 2.951650 | 2.708550 | 1.057446 | 1.006055..1.101677 | favorable | 2026103984 |
| 2 | radius | [2, 1] | 1.822150 | 1.490000 | 1.278524 | 1.175565..1.377366 | favorable | 2026103985 |
| 2 | mixed_nearest | [2, 4] | 3.431700 | 7.040650 | 0.490404 | 0.446518..0.545595 | adverse | 2026103986 |
| 2 | all_programs | [2, 5] | 4.779400 | 11.057600 | 0.449281 | 0.415410..0.486665 | adverse | 2026103987 |

### Initial unguarded warm comparisons

| Survey | Points / queries | k | Pattern | Radius | Smaller array | Route | Old median ms | New median ms | Old/new ratio | 95% interval | Outcome | Seed |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 / 1 | 1 | 0 | False | True | pass | 0.011456 | 0.010160 | 1.128132 | 1.109603..1.147805 | favorable | 2026102971 |
| 1 | 257 / 1 | 1 | 0 | False | True | resident | 0.154750 | 0.162250 | 0.980458 | 0.954431..1.004978 | inconclusive | 2026102972 |
| 1 | 257 / 1 | 1 | 0 | False | True | full | 1.232300 | 1.242200 | 0.995295 | 0.973221..1.015549 | inconclusive | 2026102973 |
| 1 | 4097 / 65 | 1 | 0 | False | True | pass | 0.050448 | 0.049648 | 1.016170 | 1.013565..1.018863 | favorable | 2026102981 |
| 1 | 4097 / 65 | 1 | 0 | False | True | resident | 0.196350 | 0.198000 | 1.023189 | 0.995203..1.057321 | inconclusive | 2026102982 |
| 1 | 4097 / 65 | 1 | 0 | False | True | full | 1.306100 | 1.309150 | 0.992361 | 0.949846..1.031032 | inconclusive | 2026102983 |
| 1 | 4097 / 65 | 8 | 0 | False | True | pass | 0.118000 | 0.121920 | 0.966982 | 0.964672..0.969136 | adverse | 2026102991 |
| 1 | 4097 / 65 | 8 | 0 | False | True | resident | 0.278200 | 0.273250 | 0.987633 | 0.955316..1.015441 | inconclusive | 2026102992 |
| 1 | 4097 / 65 | 8 | 0 | False | True | full | 1.266150 | 1.299200 | 1.000344 | 0.960252..1.056060 | inconclusive | 2026102993 |
| 1 | 4097 / 65 | 16 | 0 | False | True | pass | 0.193584 | 0.195056 | 0.990156 | 0.988401..0.991990 | adverse | 2026103001 |
| 1 | 4097 / 65 | 16 | 0 | False | True | resident | 0.349500 | 0.347650 | 1.009613 | 0.990520..1.029499 | inconclusive | 2026103002 |
| 1 | 4097 / 65 | 16 | 0 | False | True | full | 1.302850 | 1.276750 | 1.001752 | 0.962915..1.036858 | inconclusive | 2026103003 |
| 1 | 4097 / 65 | 17 | 0 | False | False | pass | 0.199440 | 0.202544 | 0.984462 | 0.981602..0.987448 | adverse | 2026103011 |
| 1 | 4097 / 65 | 17 | 0 | False | False | resident | 0.350050 | 0.352350 | 0.986458 | 0.972761..0.999790 | adverse | 2026103012 |
| 1 | 4097 / 65 | 17 | 0 | False | False | full | 1.225950 | 1.228200 | 1.011231 | 0.982550..1.042639 | inconclusive | 2026103013 |
| 1 | 65537 / 4097 | 1 | 0 | False | True | pass | 0.169264 | 0.155056 | 1.092026 | 1.089009..1.095002 | favorable | 2026103021 |
| 1 | 65537 / 4097 | 1 | 0 | False | True | resident | 0.335500 | 0.325450 | 1.018653 | 0.996088..1.040803 | inconclusive | 2026103022 |
| 1 | 65537 / 4097 | 1 | 0 | False | True | full | 2.446550 | 2.460000 | 1.000949 | 0.973529..1.027930 | inconclusive | 2026103023 |
| 1 | 65537 / 4097 | 2 | 0 | False | True | pass | 0.511936 | 0.417232 | 1.230347 | 1.220795..1.239841 | favorable | 2026103031 |
| 1 | 65537 / 4097 | 2 | 0 | False | True | resident | 0.706800 | 0.625550 | 1.134712 | 1.107406..1.162340 | favorable | 2026103032 |
| 1 | 65537 / 4097 | 2 | 0 | False | True | full | 2.512850 | 2.738550 | 0.975967 | 0.937425..1.016869 | inconclusive | 2026103033 |
| 1 | 65537 / 4097 | 8 | 0 | False | True | pass | 1.065600 | 0.814176 | 1.309228 | 1.298917..1.319553 | favorable | 2026103041 |
| 1 | 65537 / 4097 | 8 | 0 | False | True | resident | 1.304850 | 1.070650 | 1.217924 | 1.199443..1.236951 | favorable | 2026103042 |
| 1 | 65537 / 4097 | 8 | 0 | False | True | full | 3.222100 | 3.099300 | 1.053782 | 1.028683..1.078428 | favorable | 2026103043 |
| 1 | 65537 / 4097 | 16 | 0 | False | True | pass | 1.463840 | 1.391600 | 1.049291 | 1.043905..1.055410 | favorable | 2026103051 |
| 1 | 65537 / 4097 | 16 | 0 | False | True | resident | 1.904850 | 1.842150 | 1.031635 | 1.019084..1.044444 | favorable | 2026103052 |
| 1 | 65537 / 4097 | 16 | 0 | False | True | full | 4.242950 | 4.222200 | 0.988863 | 0.968447..1.009767 | inconclusive | 2026103053 |
| 1 | 65537 / 4097 | 64 | 0 | False | False | pass | 4.695376 | 4.718496 | 0.996932 | 0.990575..1.003578 | inconclusive | 2026103061 |
| 1 | 65537 / 4097 | 64 | 0 | False | False | resident | 4.909100 | 4.980450 | 0.990356 | 0.983884..0.997170 | adverse | 2026103062 |
| 1 | 65537 / 4097 | 64 | 0 | False | False | full | 9.297700 | 9.273600 | 1.008161 | 0.991092..1.025528 | inconclusive | 2026103063 |
| 1 | 262144 / 16384 | 1 | 0 | False | True | pass | 0.362640 | 0.359392 | 1.006930 | 1.000611..1.013075 | favorable | 2026103071 |
| 1 | 262144 / 16384 | 1 | 0 | False | True | resident | 0.577550 | 0.603300 | 0.956589 | 0.927652..0.987356 | adverse | 2026103072 |
| 1 | 262144 / 16384 | 1 | 0 | False | True | full | 7.379250 | 7.341500 | 1.007838 | 0.975054..1.043236 | inconclusive | 2026103073 |
| 1 | 262144 / 16384 | 8 | 0 | False | True | pass | 1.758672 | 1.557168 | 1.127769 | 1.124023..1.131314 | favorable | 2026103081 |
| 1 | 262144 / 16384 | 8 | 0 | False | True | resident | 1.903600 | 1.718550 | 1.185337 | 1.147651..1.225313 | favorable | 2026103082 |
| 1 | 262144 / 16384 | 8 | 0 | False | True | full | 9.082650 | 8.743350 | 1.039291 | 1.010119..1.070494 | favorable | 2026103083 |
| 1 | 262144 / 16384 | 16 | 0 | False | True | pass | 3.276048 | 3.254528 | 1.005607 | 1.001919..1.011476 | favorable | 2026103091 |
| 1 | 262144 / 16384 | 16 | 0 | False | True | resident | 4.360300 | 4.366550 | 1.002070 | 0.995742..1.008292 | inconclusive | 2026103092 |
| 1 | 262144 / 16384 | 16 | 0 | False | True | full | 11.723600 | 11.752550 | 1.005286 | 0.983885..1.026881 | inconclusive | 2026103093 |
| 1 | 65537 / 4097 | 8 | 1 | False | True | pass | 0.107136 | 0.101472 | 1.055638 | 1.047127..1.064358 | favorable | 2026103101 |
| 1 | 65537 / 4097 | 8 | 1 | False | True | resident | 0.302950 | 0.295850 | 1.028058 | 1.001740..1.054582 | favorable | 2026103102 |
| 1 | 65537 / 4097 | 8 | 1 | False | True | full | 2.384400 | 2.435250 | 1.009483 | 0.967844..1.050660 | inconclusive | 2026103103 |
| 1 | 65537 / 4097 | 8 | 2 | False | True | pass | 14.153440 | 14.121216 | 1.003485 | 1.002603..1.004562 | favorable | 2026103111 |
| 1 | 65537 / 4097 | 8 | 2 | False | True | resident | 14.412450 | 14.334800 | 1.003039 | 0.998192..1.006579 | inconclusive | 2026103112 |
| 1 | 65537 / 4097 | 8 | 2 | False | True | full | 16.269900 | 15.187300 | 1.072499 | 1.066498..1.078443 | favorable | 2026103113 |
| 1 | 65537 / 4097 | 8 | 3 | False | True | pass | 1.853808 | 1.865152 | 0.994854 | 0.990640..0.999223 | adverse | 2026103121 |
| 1 | 65537 / 4097 | 8 | 3 | False | True | resident | 2.050900 | 2.068650 | 0.991575 | 0.984618..0.998561 | adverse | 2026103122 |
| 1 | 65537 / 4097 | 8 | 3 | False | True | full | 3.841300 | 3.927350 | 0.998767 | 0.979542..1.019670 | inconclusive | 2026103123 |
| 1 | 4097 / 65 | 8 | 0 | True | False | pass | 0.057120 | 0.057024 | 1.003066 | 1.000621..1.005519 | favorable | 2026103131 |
| 1 | 4097 / 65 | 8 | 0 | True | False | resident | 0.220600 | 0.220100 | 1.002242 | 0.960136..1.043527 | inconclusive | 2026103132 |
| 1 | 4097 / 65 | 8 | 0 | True | False | full | 1.331500 | 1.301950 | 0.993287 | 0.966564..1.019668 | inconclusive | 2026103133 |
| 1 | 65537 / 4097 | 65 | 0 | True | False | pass | 0.211696 | 0.213488 | 0.994001 | 0.986620..1.000804 | inconclusive | 2026103141 |
| 1 | 65537 / 4097 | 65 | 0 | True | False | resident | 0.415000 | 0.431200 | 0.981765 | 0.955711..1.008814 | inconclusive | 2026103142 |
| 1 | 65537 / 4097 | 65 | 0 | True | False | full | 3.268550 | 3.294600 | 1.016988 | 0.981609..1.053189 | inconclusive | 2026103143 |
| 2 | 257 / 1 | 1 | 0 | False | True | pass | 0.011424 | 0.010112 | 1.128164 | 1.115630..1.140548 | favorable | 2026103971 |
| 2 | 257 / 1 | 1 | 0 | False | True | resident | 0.163850 | 0.162550 | 1.005736 | 0.974120..1.037904 | inconclusive | 2026103972 |
| 2 | 257 / 1 | 1 | 0 | False | True | full | 1.278750 | 1.247250 | 1.017381 | 0.991021..1.046618 | inconclusive | 2026103973 |
| 2 | 4097 / 65 | 1 | 0 | False | True | pass | 0.050080 | 0.049552 | 1.010805 | 1.008650..1.013179 | favorable | 2026103981 |
| 2 | 4097 / 65 | 1 | 0 | False | True | resident | 0.199600 | 0.197350 | 1.006519 | 0.991202..1.021268 | inconclusive | 2026103982 |
| 2 | 4097 / 65 | 1 | 0 | False | True | full | 1.330700 | 1.324500 | 0.998845 | 0.964514..1.030064 | inconclusive | 2026103983 |
| 2 | 4097 / 65 | 8 | 0 | False | True | pass | 0.117040 | 0.122272 | 0.957181 | 0.955119..0.959298 | adverse | 2026103991 |
| 2 | 4097 / 65 | 8 | 0 | False | True | resident | 0.257750 | 0.263050 | 0.986386 | 0.975373..0.997775 | adverse | 2026103992 |
| 2 | 4097 / 65 | 8 | 0 | False | True | full | 1.230750 | 1.224350 | 1.031874 | 0.984809..1.103688 | inconclusive | 2026103993 |
| 2 | 4097 / 65 | 16 | 0 | False | True | pass | 0.191840 | 0.193936 | 0.990232 | 0.988426..0.991956 | adverse | 2026104001 |
| 2 | 4097 / 65 | 16 | 0 | False | True | resident | 0.334100 | 0.331100 | 0.996843 | 0.977826..1.016392 | inconclusive | 2026104002 |
| 2 | 4097 / 65 | 16 | 0 | False | True | full | 1.219700 | 1.206400 | 1.007532 | 0.987671..1.026387 | inconclusive | 2026104003 |
| 2 | 4097 / 65 | 17 | 0 | False | False | pass | 0.198464 | 0.201088 | 0.986441 | 0.982960..0.989723 | adverse | 2026104011 |
| 2 | 4097 / 65 | 17 | 0 | False | False | resident | 0.335200 | 0.339150 | 0.996568 | 0.988760..1.004218 | inconclusive | 2026104012 |
| 2 | 4097 / 65 | 17 | 0 | False | False | full | 1.192650 | 1.192550 | 1.024374 | 0.985685..1.086714 | inconclusive | 2026104013 |
| 2 | 65537 / 4097 | 1 | 0 | False | True | pass | 0.168400 | 0.153824 | 1.095385 | 1.092441..1.098421 | favorable | 2026104021 |
| 2 | 65537 / 4097 | 1 | 0 | False | True | resident | 0.312750 | 0.303300 | 1.034637 | 1.017855..1.052405 | favorable | 2026104022 |
| 2 | 65537 / 4097 | 1 | 0 | False | True | full | 2.090350 | 2.057900 | 1.014249 | 0.968866..1.068005 | inconclusive | 2026104023 |
| 2 | 65537 / 4097 | 2 | 0 | False | True | pass | 0.202976 | 0.191536 | 1.061740 | 1.057119..1.066237 | favorable | 2026104031 |
| 2 | 65537 / 4097 | 2 | 0 | False | True | resident | 0.377450 | 0.363450 | 1.033390 | 1.016731..1.050519 | favorable | 2026104032 |
| 2 | 65537 / 4097 | 2 | 0 | False | True | full | 2.243150 | 2.286850 | 0.980040 | 0.943297..1.017577 | inconclusive | 2026104033 |
| 2 | 65537 / 4097 | 8 | 0 | False | True | pass | 0.960848 | 0.707472 | 1.356920 | 1.346613..1.367457 | favorable | 2026104041 |
| 2 | 65537 / 4097 | 8 | 0 | False | True | resident | 1.179000 | 0.949950 | 1.231000 | 1.200151..1.269297 | favorable | 2026104042 |
| 2 | 65537 / 4097 | 8 | 0 | False | True | full | 3.003450 | 2.833350 | 1.052741 | 1.019641..1.085140 | favorable | 2026104043 |
| 2 | 65537 / 4097 | 16 | 0 | False | True | pass | 0.869904 | 0.864512 | 1.007760 | 1.003807..1.011821 | favorable | 2026104051 |
| 2 | 65537 / 4097 | 16 | 0 | False | True | resident | 1.639800 | 1.622150 | 1.009062 | 0.975031..1.031017 | inconclusive | 2026104052 |
| 2 | 65537 / 4097 | 16 | 0 | False | True | full | 3.567650 | 3.489250 | 1.007508 | 0.986613..1.029968 | inconclusive | 2026104053 |
| 2 | 65537 / 4097 | 64 | 0 | False | False | pass | 4.744192 | 4.729712 | 1.002219 | 0.994224..1.010126 | inconclusive | 2026104061 |
| 2 | 65537 / 4097 | 64 | 0 | False | False | resident | 5.004000 | 4.978600 | 0.998106 | 0.989902..1.006800 | inconclusive | 2026104062 |
| 2 | 65537 / 4097 | 64 | 0 | False | False | full | 9.187550 | 9.209050 | 1.009207 | 0.991202..1.025688 | inconclusive | 2026104063 |
| 2 | 262144 / 16384 | 1 | 0 | False | True | pass | 0.363120 | 0.356864 | 1.012765 | 1.005952..1.018985 | favorable | 2026104071 |
| 2 | 262144 / 16384 | 1 | 0 | False | True | resident | 0.604700 | 0.582200 | 1.017216 | 0.984212..1.051211 | inconclusive | 2026104072 |
| 2 | 262144 / 16384 | 1 | 0 | False | True | full | 7.098200 | 6.835450 | 1.002495 | 0.968985..1.037166 | inconclusive | 2026104073 |
| 2 | 262144 / 16384 | 8 | 0 | False | True | pass | 1.743312 | 1.543088 | 1.128270 | 1.124108..1.132467 | favorable | 2026104081 |
| 2 | 262144 / 16384 | 8 | 0 | False | True | resident | 1.789000 | 1.349600 | 1.244402 | 1.205917..1.285409 | favorable | 2026104082 |
| 2 | 262144 / 16384 | 8 | 0 | False | True | full | 9.200000 | 9.260850 | 1.000775 | 0.964790..1.039722 | inconclusive | 2026104083 |
| 2 | 262144 / 16384 | 16 | 0 | False | True | pass | 3.294416 | 3.287696 | 1.005206 | 1.001837..1.010421 | favorable | 2026104091 |
| 2 | 262144 / 16384 | 16 | 0 | False | True | resident | 4.526650 | 4.547850 | 1.002861 | 0.995174..1.011149 | inconclusive | 2026104092 |
| 2 | 262144 / 16384 | 16 | 0 | False | True | full | 12.389000 | 12.788850 | 0.990683 | 0.965346..1.017910 | inconclusive | 2026104093 |
| 2 | 65537 / 4097 | 8 | 1 | False | True | pass | 0.122528 | 0.116496 | 1.057872 | 1.047747..1.068269 | favorable | 2026104101 |
| 2 | 65537 / 4097 | 8 | 1 | False | True | resident | 0.343250 | 0.339150 | 1.035206 | 1.002087..1.067827 | favorable | 2026104102 |
| 2 | 65537 / 4097 | 8 | 1 | False | True | full | 2.551850 | 2.684350 | 0.968671 | 0.932041..1.006445 | inconclusive | 2026104103 |
| 2 | 65537 / 4097 | 8 | 2 | False | True | pass | 14.174448 | 14.122400 | 1.004386 | 1.003377..1.005507 | favorable | 2026104111 |
| 2 | 65537 / 4097 | 8 | 2 | False | True | resident | 14.395650 | 14.351650 | 1.004227 | 1.002051..1.006545 | favorable | 2026104112 |
| 2 | 65537 / 4097 | 8 | 2 | False | True | full | 16.201750 | 15.131650 | 1.072045 | 1.066968..1.077074 | favorable | 2026104113 |
| 2 | 65537 / 4097 | 8 | 3 | False | True | pass | 1.853040 | 1.866400 | 0.996087 | 0.991388..1.000958 | inconclusive | 2026104121 |
| 2 | 65537 / 4097 | 8 | 3 | False | True | resident | 2.086200 | 2.109700 | 0.990124 | 0.983077..0.997440 | adverse | 2026104122 |
| 2 | 65537 / 4097 | 8 | 3 | False | True | full | 4.088150 | 3.982900 | 1.009234 | 0.985744..1.034589 | inconclusive | 2026104123 |
| 2 | 4097 / 65 | 8 | 0 | True | False | pass | 0.057888 | 0.057856 | 1.001395 | 0.998439..1.004369 | inconclusive | 2026104131 |
| 2 | 4097 / 65 | 8 | 0 | True | False | resident | 0.237550 | 0.229600 | 1.023852 | 0.983976..1.065520 | inconclusive | 2026104132 |
| 2 | 4097 / 65 | 8 | 0 | True | False | full | 1.284350 | 1.306800 | 1.004585 | 0.977427..1.032671 | inconclusive | 2026104133 |
| 2 | 65537 / 4097 | 65 | 0 | True | False | pass | 0.213152 | 0.212624 | 0.998576 | 0.991623..1.005633 | inconclusive | 2026104141 |
| 2 | 65537 / 4097 | 65 | 0 | True | False | resident | 0.416450 | 0.428450 | 0.984298 | 0.955478..1.010311 | inconclusive | 2026104142 |
| 2 | 65537 / 4097 | 65 | 0 | True | False | full | 3.315300 | 3.264750 | 1.004401 | 0.974467..1.032827 | inconclusive | 2026104143 |

### Initial unguarded fresh-cache comparisons

| Survey | Scenario | Programs old/new | Old median ms | New median ms | Old/new ratio | 95% interval | Outcome | Seed |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | nearest_one | [2, 1] | 1.773650 | 1.441200 | 1.360895 | 1.177126..1.705534 | favorable | 2026102972 |
| 1 | nearest_eight | [2, 1] | 1.699250 | 1.432700 | 1.207329 | 1.163657..1.256891 | favorable | 2026102973 |
| 1 | nearest_wide | [2, 1] | 2.906500 | 2.600400 | 1.134758 | 1.091939..1.188362 | favorable | 2026102974 |
| 1 | radius | [2, 1] | 1.735250 | 1.386550 | 1.189804 | 1.088729..1.306436 | favorable | 2026102975 |
| 1 | mixed_nearest | [2, 4] | 2.647950 | 5.682600 | 0.479000 | 0.458782..0.500777 | adverse | 2026102976 |
| 1 | all_programs | [2, 5] | 5.001900 | 10.264950 | 0.469760 | 0.432712..0.512466 | adverse | 2026102977 |
| 2 | nearest_one | [2, 1] | 1.832150 | 1.545450 | 1.276129 | 1.076796..1.629589 | favorable | 2026103972 |
| 2 | nearest_eight | [2, 1] | 2.075450 | 1.784600 | 1.146878 | 1.024295..1.288414 | favorable | 2026103973 |
| 2 | nearest_wide | [2, 1] | 3.221150 | 2.906550 | 1.150442 | 1.073364..1.229716 | favorable | 2026103974 |
| 2 | radius | [2, 1] | 1.935250 | 1.557050 | 1.226364 | 1.107735..1.357309 | favorable | 2026103975 |
| 2 | mixed_nearest | [2, 4] | 3.081150 | 6.884950 | 0.453927 | 0.418569..0.493392 | adverse | 2026103976 |
| 2 | all_programs | [2, 5] | 4.346200 | 9.747150 | 0.461771 | 0.440834..0.482415 | adverse | 2026103977 |

The initial experiment applied smaller arrays to every nonempty batch, including 65-query cases. It produced repeated adverse small-batch GPU intervals (roughly 1%..4.5% more time for k=8/16), which motivated the 1024-query guard. Both complete original surveys are retained above: 108 warm and 12 fresh-cache summaries, with original seeds 2026101971/2026101972 and 686 source/manifest/executable hashes. Initial validation passed warning-denying Clippy, four targeted tests and the 49-test nearest suite; full downstream validation is recorded for the final build. Initial fresh-cache jobs used 65 queries; final fresh-cache jobs use 1025. Initial executable SHA-256: `2425d40fb0c62680732ac3776d8a326c6046ca11b70a0cf9fa218d1290bcdcd5`. These initial results are separate from the final build and were not selectively rerun or discarded.

Fixture patterns: 0 is a reversed regular 3D grid, 1 a line, 2 duplicate points and 3 repeated planar ties. Queries include exact input coordinates and exact fractional offsets. Array extents reduce declared candidates only; all output capacities remain the requested k or radius capacity. Mixed first-use costs must be considered with warm results.
