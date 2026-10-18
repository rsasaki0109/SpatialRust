# Reusing saturated CPU voxel digit probes

Date: 2026-10-18. Named slice: saturated CPU voxel digit probes.
Immediate parent: `b8ac692b5a34e83ea7e140e229e20750da4b9fba` (merged PR #279).

## Change and exactness proof

The previous private planner used a lightweight global XOR/OR scan only when
its first 256 entries contained one sign per axis. A mixed-sign input with fixed
middle digits still paid for a full indexed sign-class scan even if its probe
already identified every required digit. Pattern 5 in the previous sign-family
evidence exposed a resident-build regression; this family measures it again
against the immediate parent, without assuming a fixed timing result.

The probe already has AND/OR summaries for both signs of every axis. Combining
class OR by OR and class AND by AND gives the complete probe's global variation:
`(pos.any OR neg.any) XOR (pos.all AND neg.all)`. An empty class's (MAX,0)
identity preserves the populated class. The probe is nonempty on this branch.

Let P be the unpadded sign-class digit mask of the probe, C the unpadded complete
sign-class mask, and G the complete global varying-digit mask. Within-class
variation and sign presence only grow, so `P subset C subset G`. When the probe's
own global mask equals P, run the cheap complete global scan. Return it only if
`G == P`, proving `P == C == G`. Both membership and positions must agree; equal
digit counts alone are insufficient. Apply the original parity padding only
after this proof. The previous complete one-sign proof also remains valid if
later same-sign entries introduce more digits. Every other case falls through
to the unchanged complete sign-class scan, including late digit/sign changes.
The returned mask is bit-identical to the immediate parent for every input.

The 256-point bound, original clamp/preparation, bounded worker fold, useful
131,072-point grains, 524,288 parallel threshold and at most four native workers
including the caller remain. A failed new probe candidate can add a complete
cheap scan before the class scan, so adversarial late inputs are measured too.
There is no per-point planning allocation or global worker budget. WASM and
small/one-worker calls remain serial; full-width probes still return early.

## Contracts and validation

Only the private CPU-key digit planner changes production behavior. Public APIs,
admission errors, prepared entry words/source indices, selected GPU digit masks,
WGSL/pipelines/radix scheduling, explicit `16*N` uploads, nonempty four-byte
construction synchronization, metadata D2H, GPU-key route and compaction are
unchanged. GPU work and output allocation stay exact. This is a host scheduling
optimization, not fewer GPU passes or physical memory/bandwidth measurements.

Two new native regressions compare complete parent masks with an independent
per-sign nibble oracle, stable CPU LSD orders and forced 0/1/2/3/4/12/maximum
worker availability. They cover empty/singleton inputs, every signed width,
256-point probe and worker thresholds, saturation, fixed high/middle digits,
late-only digit changes, late opposite signs and clamp extremes. Complete
parent/current resident and small host-returning APIs plus raw owned sort words
match independent CPU expectations. Default/tiny/zero pool budgets retain the
same requests, actual passes and query bytes. GPU-key stages, inputs and errors
remain exact. Concurrent results survive host mutation, pool clears and changed
reuse. Existing tests remain unchanged apart from native test registration.

Native-only reference modules reproduce the immediate-parent planner and complete
CPU/device-key API bodies, with import/path changes only. No core, feature,
dependency, unsafe or shader changes are made.

- VOXEL_PROBE_OUTPUT verified 140094232 parent/current/CPU raw words across 56 complete fixture pairs; unchanged masks/passes/queries/requests, ownership, budgets, device-key input/admission, concurrent builds, mutation and changed reuse.

- VOXEL_PROBE_PLAN verified 2487 parent/oracle/worker masks and stable signed LSD orders; empty classes, every signed width, exact raw-bit membership, 256-point probe and four-worker boundaries, saturated probes and late fallback.

Final validation: 462 native test executions, excluding repeated targeted prechecks.

| Check | Tests passed |
| --- | ---: |
| clippy | 0 |
| image | 182 |
| voxel | 238 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| cpu-only | 0 |
| wasm-minimal | 0 |
| wasm-ci | 0 |
| fmt | 0 |
| diff | 0 |
| facade-gaussian | 1 |

All-feature/all-target Clippy denies warnings. CPU-only, minimal GPU WASM and
CI-feature GPU/image WASM, format and diff checks pass. 618
original GPU/image source files are unchanged; 630 current source,
manifest and executable hashes match across precheck, measurement and validation.
Published files/PR text contain zero forbidden local adapter-name matches.

## Measurement method and limitations

Two consecutive release surveys each use 24 synthetic fixtures, three warmups
and 40 measured pairs on resident `build`, host-returning `full` and host-only
preparation/planning. Each complete old/current call uses its own planner.
Resident time includes original preparation, planning, upload, sorting,
compaction, scalar synchronization and completion. Full time includes explicit
metadata D2H/host conversion. Balanced XOR rotation covers all four old/current
build/full positions. Host diagnostics alternate variants; both input and
selected mask are black-boxed within timing. Output correctness checks and
returned-result destruction are excluded. Identical masks imply identical GPU
schedules; no GPU-stage speedup is claimed from these wall timings.

Six fresh-runtime scenarios each use 20 alternating pairs in two surveys, no
warmup, and a fresh runtime/device per variant created outside timing. The first
resident construction is timed, including its CPU planner and GPU pipeline use;
checking/destruction are excluded. No explicit driver-cache reset is requested.
These are not process-cold or hardware-cold guarantees; runtime creation cost,
physical memory, energy and other devices are unmeasured.

Ratios are paired geometric means parent/current (>1 favors this change).
Each individual 95% interval uses 10,000 paired log bootstrap draws, sorted
indices 249/9749, with arithmetic means computed as math.fsum(values)/N.
Warm seed=2026101841+1000*survey+10*case_index+route_index (1-based surveys,
0-based cases/routes, build/full/host route order). Cold seed=2026101842+
1000*survey+case_index. There is no multiple-comparison correction, epsilon,
outlier filtering, dropped pair or selected rerun. All rows are retained.
Medians are distinct from paired estimates. Shared-host pre/post CPU/GPU
samples do not prove isolation during every pair or portable application speed.
This task's builds/tests and measurement families do not overlap timed surveys.

Patterns 5/21/22 exercise saturated mixed-sign probes with fixed middle digits.
Patterns 23/24 introduce a previously absent digit at the last point, forcing the
additional complete global scan to fall back to class planning. Other controls
include bounded zero-crossing, full-width random, positive/negative-only,
constant and late opposite-sign data. Point counts span 257 through 1,000,003,
including both sides of the four-worker threshold. Empty/singleton cases are
correctness-tested, not timed. Exact generators are in the source paths below.

| Route | Group | Comparisons | Paired ratio range |
| --- | --- | --- | --- |
| build | saturated | 12 favorable / 10 inconclusive / 0 adverse / 0 unresolved | 1.002964..1.120671 |
| build | late fallback | 1 favorable / 3 inconclusive / 4 adverse / 0 unresolved | 0.935811..1.042706 |
| build | control | 1 favorable / 17 inconclusive / 0 adverse / 0 unresolved | 0.973298..1.025316 |
| full | saturated | 7 favorable / 15 inconclusive / 0 adverse / 0 unresolved | 0.982156..1.056728 |
| full | late fallback | 0 favorable / 4 inconclusive / 4 adverse / 0 unresolved | 0.953223..1.010765 |
| full | control | 0 favorable / 17 inconclusive / 1 adverse / 0 unresolved | 0.970903..1.012168 |
| host | saturated | 22 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1.150927..1.373375 |
| host | late fallback | 0 favorable / 1 inconclusive / 7 adverse / 0 unresolved | 0.788529..0.931008 |
| host | control | 1 favorable / 16 inconclusive / 1 adverse / 0 unresolved | 0.958885..1.082120 |

Fresh-runtime comparisons: 0 favorable / 12 inconclusive / 0 adverse / 0 unresolved.

Targeted host preparation/planning takes 13.1–27.2% less time across all 22 paired comparisons; GPU digit masks and work remain identical. Most full-call and all first-construction intervals remain inconclusive, and late-fallback adverse results remain below.

All individually adverse complete-call observations:

- full, survey 1, N=1000003, pattern 6 (control): 1.60% more time, ratio 0.984274 (0.969809..0.998639).

- build, survey 1, N=1000003, pattern 23 (late fallback): 6.48% more time, ratio 0.939151 (0.918383..0.959175).

- full, survey 1, N=1000003, pattern 23 (late fallback): 4.91% more time, ratio 0.953223 (0.932416..0.972475).

- build, survey 1, N=1000003, pattern 24 (late fallback): 6.75% more time, ratio 0.936780 (0.917349..0.955958).

- full, survey 1, N=1000003, pattern 24 (late fallback): 3.04% more time, ratio 0.970528 (0.951480..0.989285).

- build, survey 2, N=1000003, pattern 23 (late fallback): 6.43% more time, ratio 0.939550 (0.924114..0.955458).

- full, survey 2, N=1000003, pattern 23 (late fallback): 4.73% more time, ratio 0.954877 (0.932868..0.977133).

- build, survey 2, N=1000003, pattern 24 (late fallback): 6.86% more time, ratio 0.935811 (0.915364..0.957255).

- full, survey 2, N=1000003, pattern 24 (late fallback): 4.84% more time, ratio 0.953843 (0.925195..0.983943).

Cold results and adverse/inconclusive controls do not establish a cause from these shared-host observations.

| Survey | Kind | Quiet observed | Busy counts pre/post | Maximum busy CPU cores pre/post | GPU samples pre/post |
| ---: | --- | --- | --- | --- | --- |
| 1 | warm | False | [5, 4] | [2.572, 2.447] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 2 | warm | False | [4, 3] | [2.506, 2.473] | `[[[[0, 0]], [[0, 0]]], [[[1, 0]], [[2, 0]]]]` |
| 1 | fresh-runtime | False | [3, 1] | [1.309, 0.994] | `[[[[0, 0]], [[0, 0]]], [[[2, 0]], [[0, 0]]]]` |
| 2 | fresh-runtime | False | [1, 1] | [0.969, 0.957] | `[[[[0, 0]], [[0, 0]]], [[[2, 0]], [[0, 0]]]]` |

Environment labels without local adapter model:

- warm 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=40 warmups=3 wall_order=balanced host_order=alternating checks_drop_excluded=true`

- warm 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=40 warmups=3 wall_order=balanced host_order=alternating checks_drop_excluded=true`

- fresh-runtime 1: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`

- fresh-runtime 2: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`

## Every warm comparison

| Survey | N | Pattern | Digits | Route | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 5 | 10 | build | 2.680400 | 2.557650 | 1.013607 | 0.953985..1.075407 | inconclusive | 2026102841 |
| 1 | 257 | 5 | 10 | full | 4.203950 | 4.350650 | 1.000876 | 0.952935..1.053794 | inconclusive | 2026102842 |
| 1 | 257 | 5 | 10 | host | 0.004500 | 0.003200 | 1.271506 | 1.120380..1.432284 | favorable | 2026102843 |
| 1 | 65537 | 5 | 10 | build | 4.713600 | 4.399300 | 1.066001 | 1.025660..1.110209 | favorable | 2026102851 |
| 1 | 65537 | 5 | 10 | full | 7.907100 | 7.536100 | 1.040304 | 1.007000..1.074041 | favorable | 2026102852 |
| 1 | 65537 | 5 | 10 | host | 0.897500 | 0.687300 | 1.321617 | 1.274452..1.367818 | favorable | 2026102853 |
| 1 | 524287 | 5 | 10 | build | 25.530850 | 23.747250 | 1.085399 | 1.067817..1.111869 | favorable | 2026102861 |
| 1 | 524287 | 5 | 10 | full | 41.381700 | 40.497450 | 1.024728 | 1.006069..1.041263 | favorable | 2026102862 |
| 1 | 524287 | 5 | 10 | host | 6.514350 | 4.786600 | 1.373375 | 1.336727..1.412226 | favorable | 2026102863 |
| 1 | 524288 | 5 | 10 | build | 17.167450 | 17.195250 | 1.021611 | 1.004098..1.041073 | favorable | 2026102871 |
| 1 | 524288 | 5 | 10 | full | 27.944250 | 27.793300 | 1.024901 | 1.004960..1.046352 | favorable | 2026102872 |
| 1 | 524288 | 5 | 10 | host | 3.136950 | 2.690350 | 1.150927 | 1.126611..1.174839 | favorable | 2026102873 |
| 1 | 524289 | 5 | 10 | build | 12.862050 | 12.025050 | 1.050684 | 1.018743..1.099077 | favorable | 2026102881 |
| 1 | 524289 | 5 | 10 | full | 27.426250 | 26.177900 | 1.017913 | 1.001496..1.034794 | favorable | 2026102882 |
| 1 | 524289 | 5 | 10 | host | 3.136350 | 2.670300 | 1.171047 | 1.139266..1.203436 | favorable | 2026102883 |
| 1 | 1000003 | 5 | 10 | build | 21.391300 | 20.266350 | 1.017094 | 0.966096..1.061855 | inconclusive | 2026102891 |
| 1 | 1000003 | 5 | 10 | full | 44.006500 | 42.886600 | 1.024536 | 1.003373..1.046283 | favorable | 2026102892 |
| 1 | 1000003 | 5 | 10 | host | 5.077550 | 4.302600 | 1.194642 | 1.165146..1.226430 | favorable | 2026102893 |
| 1 | 65537 | 21 | 6 | build | 10.545000 | 10.260100 | 1.040802 | 1.000013..1.086765 | favorable | 2026102901 |
| 1 | 65537 | 21 | 6 | full | 18.726850 | 18.338150 | 1.013428 | 0.984444..1.043264 | inconclusive | 2026102902 |
| 1 | 65537 | 21 | 6 | host | 0.917700 | 0.687800 | 1.348470 | 1.293505..1.402214 | favorable | 2026102903 |
| 1 | 1000003 | 21 | 6 | build | 20.151700 | 19.319350 | 1.044929 | 1.014339..1.072168 | favorable | 2026102911 |
| 1 | 1000003 | 21 | 6 | full | 33.450650 | 31.904600 | 1.022172 | 0.999983..1.044226 | inconclusive | 2026102912 |
| 1 | 1000003 | 21 | 6 | host | 5.372250 | 4.457150 | 1.205429 | 1.172094..1.239107 | favorable | 2026102913 |
| 1 | 65537 | 22 | 10 | build | 10.937050 | 10.711950 | 1.006947 | 0.964436..1.049009 | inconclusive | 2026102921 |
| 1 | 65537 | 22 | 10 | full | 20.780300 | 20.323000 | 1.008483 | 0.983284..1.034301 | inconclusive | 2026102922 |
| 1 | 65537 | 22 | 10 | host | 0.941400 | 0.706100 | 1.332627 | 1.298144..1.366061 | favorable | 2026102923 |
| 1 | 1000003 | 22 | 10 | build | 22.425350 | 21.430100 | 1.051634 | 1.030094..1.074776 | favorable | 2026102931 |
| 1 | 1000003 | 22 | 10 | full | 46.090900 | 45.574300 | 1.010317 | 0.986684..1.035025 | inconclusive | 2026102932 |
| 1 | 1000003 | 22 | 10 | host | 5.662400 | 4.796550 | 1.184596 | 1.146881..1.224326 | favorable | 2026102933 |
| 1 | 65537 | 0 | 8 | build | 12.105150 | 12.336500 | 0.978172 | 0.933928..1.023648 | inconclusive | 2026102941 |
| 1 | 65537 | 0 | 8 | full | 20.875050 | 20.896050 | 1.008663 | 0.953784..1.057023 | inconclusive | 2026102942 |
| 1 | 65537 | 0 | 8 | host | 0.975800 | 0.951900 | 1.082120 | 1.024660..1.155377 | favorable | 2026102943 |
| 1 | 1000003 | 0 | 8 | build | 21.905650 | 21.322800 | 1.005831 | 0.970913..1.041348 | inconclusive | 2026102951 |
| 1 | 1000003 | 0 | 8 | full | 34.520550 | 34.773950 | 0.970903 | 0.935528..1.005481 | inconclusive | 2026102952 |
| 1 | 1000003 | 0 | 8 | host | 5.719700 | 5.819100 | 0.999864 | 0.954374..1.044343 | inconclusive | 2026102953 |
| 1 | 65537 | 6 | 24 | build | 13.738100 | 13.927300 | 0.973298 | 0.940284..1.008332 | inconclusive | 2026102961 |
| 1 | 65537 | 6 | 24 | full | 23.732850 | 23.550500 | 0.983435 | 0.956142..1.010949 | inconclusive | 2026102962 |
| 1 | 65537 | 6 | 24 | host | 0.605850 | 0.605600 | 0.991477 | 0.958733..1.026527 | inconclusive | 2026102963 |
| 1 | 1000003 | 6 | 24 | build | 27.303650 | 27.410500 | 0.999012 | 0.986558..1.010856 | inconclusive | 2026102971 |
| 1 | 1000003 | 6 | 24 | full | 48.552950 | 49.594200 | 0.984274 | 0.969809..0.998639 | adverse | 2026102972 |
| 1 | 1000003 | 6 | 24 | host | 3.046200 | 3.151650 | 0.992542 | 0.970180..1.016801 | inconclusive | 2026102973 |
| 1 | 65537 | 8 | 6 | build | 9.677950 | 9.685150 | 0.984146 | 0.955189..1.012859 | inconclusive | 2026102981 |
| 1 | 65537 | 8 | 6 | full | 17.027750 | 17.163150 | 1.005909 | 0.982698..1.031220 | inconclusive | 2026102982 |
| 1 | 65537 | 8 | 6 | host | 0.728700 | 0.722400 | 1.035728 | 0.987716..1.095443 | inconclusive | 2026102983 |
| 1 | 1000003 | 8 | 6 | build | 21.225850 | 20.613150 | 1.005022 | 0.983368..1.027846 | inconclusive | 2026102991 |
| 1 | 1000003 | 8 | 6 | full | 31.540500 | 32.012250 | 0.982286 | 0.956597..1.007624 | inconclusive | 2026102992 |
| 1 | 1000003 | 8 | 6 | host | 4.499150 | 4.528550 | 1.002178 | 0.978680..1.027830 | inconclusive | 2026102993 |
| 1 | 65537 | 23 | 10 | build | 9.807700 | 9.711250 | 0.995588 | 0.963351..1.028442 | inconclusive | 2026103001 |
| 1 | 65537 | 23 | 10 | full | 17.734850 | 17.742900 | 0.997640 | 0.971225..1.025804 | inconclusive | 2026103002 |
| 1 | 65537 | 23 | 10 | host | 0.878250 | 0.975550 | 0.905942 | 0.881889..0.930878 | adverse | 2026103003 |
| 1 | 1000003 | 23 | 10 | build | 20.002900 | 21.672250 | 0.939151 | 0.918383..0.959175 | adverse | 2026103011 |
| 1 | 1000003 | 23 | 10 | full | 40.312800 | 42.972000 | 0.953223 | 0.932416..0.972475 | adverse | 2026103012 |
| 1 | 1000003 | 23 | 10 | host | 5.008100 | 6.191900 | 0.805917 | 0.789117..0.821986 | adverse | 2026103013 |
| 1 | 65537 | 24 | 6 | build | 9.514050 | 9.260300 | 1.042706 | 1.002828..1.085629 | favorable | 2026103021 |
| 1 | 65537 | 24 | 6 | full | 16.480050 | 16.335450 | 1.009182 | 0.983899..1.035288 | inconclusive | 2026103022 |
| 1 | 65537 | 24 | 6 | host | 0.919200 | 1.029450 | 0.901814 | 0.877229..0.928871 | adverse | 2026103023 |
| 1 | 1000003 | 24 | 6 | build | 18.520100 | 20.567600 | 0.936780 | 0.917349..0.955958 | adverse | 2026103031 |
| 1 | 1000003 | 24 | 6 | full | 29.528000 | 30.052250 | 0.970528 | 0.951480..0.989285 | adverse | 2026103032 |
| 1 | 1000003 | 24 | 6 | host | 5.252550 | 6.395600 | 0.827835 | 0.804818..0.851237 | adverse | 2026103033 |
| 1 | 257 | 7 | 0 | build | 7.683500 | 7.552150 | 1.021753 | 0.979643..1.064577 | inconclusive | 2026103041 |
| 1 | 257 | 7 | 0 | full | 14.264150 | 14.050500 | 1.001181 | 0.974935..1.025979 | inconclusive | 2026103042 |
| 1 | 257 | 7 | 0 | host | 0.004200 | 0.004000 | 1.074999 | 0.969753..1.197012 | inconclusive | 2026103043 |
| 1 | 65537 | 25 | 8 | build | 11.453650 | 11.551850 | 0.973801 | 0.934469..1.014454 | inconclusive | 2026103051 |
| 1 | 65537 | 25 | 8 | full | 20.195400 | 20.835500 | 0.974583 | 0.945945..1.005507 | inconclusive | 2026103052 |
| 1 | 65537 | 25 | 8 | host | 1.056050 | 1.057300 | 0.988758 | 0.947008..1.030563 | inconclusive | 2026103053 |
| 1 | 257 | 21 | 6 | build | 9.140750 | 8.841900 | 1.017755 | 0.981275..1.054696 | inconclusive | 2026103061 |
| 1 | 257 | 21 | 6 | full | 17.030700 | 17.470500 | 0.982156 | 0.948262..1.016025 | inconclusive | 2026103062 |
| 1 | 257 | 21 | 6 | host | 0.004600 | 0.003750 | 1.334124 | 1.192726..1.490556 | favorable | 2026103063 |
| 1 | 1000003 | 27 | 6 | build | 18.922650 | 18.866900 | 1.025316 | 1.004770..1.046671 | favorable | 2026103071 |
| 1 | 1000003 | 27 | 6 | full | 30.797450 | 30.497950 | 0.989694 | 0.958379..1.019987 | inconclusive | 2026103072 |
| 1 | 1000003 | 27 | 6 | host | 4.582550 | 4.597900 | 0.981489 | 0.957194..1.004813 | inconclusive | 2026103073 |
| 2 | 257 | 5 | 10 | build | 1.778500 | 1.773800 | 1.026893 | 0.994861..1.061354 | inconclusive | 2026103841 |
| 2 | 257 | 5 | 10 | full | 2.924700 | 2.921700 | 1.008359 | 0.982156..1.032734 | inconclusive | 2026103842 |
| 2 | 257 | 5 | 10 | host | 0.002900 | 0.002450 | 1.286283 | 1.158871..1.430018 | favorable | 2026103843 |
| 2 | 65537 | 5 | 10 | build | 3.041350 | 3.001350 | 1.026680 | 0.986264..1.066837 | inconclusive | 2026103851 |
| 2 | 65537 | 5 | 10 | full | 5.409500 | 5.222050 | 1.023619 | 0.997373..1.048652 | inconclusive | 2026103852 |
| 2 | 65537 | 5 | 10 | host | 0.582800 | 0.459400 | 1.251093 | 1.213726..1.287138 | favorable | 2026103853 |
| 2 | 524287 | 5 | 10 | build | 22.956800 | 21.357350 | 1.120671 | 1.083970..1.173769 | favorable | 2026103861 |
| 2 | 524287 | 5 | 10 | full | 34.626550 | 33.890400 | 1.056728 | 1.036194..1.078162 | favorable | 2026103862 |
| 2 | 524287 | 5 | 10 | host | 5.601550 | 4.113400 | 1.347293 | 1.298435..1.392971 | favorable | 2026103863 |
| 2 | 524288 | 5 | 10 | build | 20.697650 | 20.628950 | 1.043108 | 1.011902..1.091765 | favorable | 2026103871 |
| 2 | 524288 | 5 | 10 | full | 34.423750 | 33.815400 | 1.013866 | 0.998354..1.029113 | inconclusive | 2026103872 |
| 2 | 524288 | 5 | 10 | host | 3.070750 | 2.652200 | 1.157768 | 1.123615..1.192539 | favorable | 2026103873 |
| 2 | 524289 | 5 | 10 | build | 18.266150 | 16.784250 | 1.040871 | 0.987554..1.091547 | inconclusive | 2026103881 |
| 2 | 524289 | 5 | 10 | full | 31.826700 | 30.630150 | 1.005641 | 0.987203..1.023919 | inconclusive | 2026103882 |
| 2 | 524289 | 5 | 10 | host | 3.094300 | 2.688750 | 1.176077 | 1.147456..1.205217 | favorable | 2026103883 |
| 2 | 1000003 | 5 | 10 | build | 24.214150 | 22.408700 | 1.023696 | 0.980494..1.057020 | inconclusive | 2026103891 |
| 2 | 1000003 | 5 | 10 | full | 45.884850 | 45.917950 | 1.017565 | 0.996133..1.040458 | inconclusive | 2026103892 |
| 2 | 1000003 | 5 | 10 | host | 4.939900 | 4.249650 | 1.174836 | 1.150003..1.198973 | favorable | 2026103893 |
| 2 | 65537 | 21 | 6 | build | 8.779800 | 8.645250 | 1.007658 | 0.971554..1.044186 | inconclusive | 2026103901 |
| 2 | 65537 | 21 | 6 | full | 16.771300 | 16.246700 | 1.024191 | 0.995272..1.050994 | inconclusive | 2026103902 |
| 2 | 65537 | 21 | 6 | host | 0.876500 | 0.634100 | 1.363913 | 1.329877..1.397426 | favorable | 2026103903 |
| 2 | 1000003 | 21 | 6 | build | 19.184250 | 17.834500 | 1.062501 | 1.045736..1.081236 | favorable | 2026103911 |
| 2 | 1000003 | 21 | 6 | full | 30.776650 | 29.101500 | 1.046897 | 1.019701..1.074865 | favorable | 2026103912 |
| 2 | 1000003 | 21 | 6 | host | 5.054100 | 4.204100 | 1.207873 | 1.175924..1.241153 | favorable | 2026103913 |
| 2 | 65537 | 22 | 10 | build | 9.441050 | 8.997100 | 1.030523 | 1.001866..1.059061 | favorable | 2026103921 |
| 2 | 65537 | 22 | 10 | full | 17.867300 | 17.427700 | 1.017734 | 0.990664..1.046866 | inconclusive | 2026103922 |
| 2 | 65537 | 22 | 10 | host | 0.843550 | 0.632850 | 1.343229 | 1.291695..1.398401 | favorable | 2026103923 |
| 2 | 1000003 | 22 | 10 | build | 23.220650 | 22.316900 | 1.035205 | 1.012630..1.058512 | favorable | 2026103931 |
| 2 | 1000003 | 22 | 10 | full | 45.144850 | 44.317700 | 1.022113 | 0.997495..1.053869 | inconclusive | 2026103932 |
| 2 | 1000003 | 22 | 10 | host | 5.074650 | 4.276450 | 1.171258 | 1.140512..1.203238 | favorable | 2026103933 |
| 2 | 65537 | 0 | 8 | build | 9.858750 | 10.219200 | 0.976255 | 0.941035..1.014665 | inconclusive | 2026103941 |
| 2 | 65537 | 0 | 8 | full | 17.903400 | 18.266650 | 1.002399 | 0.972568..1.033370 | inconclusive | 2026103942 |
| 2 | 65537 | 0 | 8 | host | 0.898250 | 0.911650 | 0.995832 | 0.961073..1.031159 | inconclusive | 2026103943 |
| 2 | 1000003 | 0 | 8 | build | 17.969800 | 18.263950 | 0.988911 | 0.968006..1.010116 | inconclusive | 2026103951 |
| 2 | 1000003 | 0 | 8 | full | 27.954650 | 28.572900 | 0.994981 | 0.976904..1.012831 | inconclusive | 2026103952 |
| 2 | 1000003 | 0 | 8 | host | 4.683650 | 4.599750 | 1.002448 | 0.985535..1.019253 | inconclusive | 2026103953 |
| 2 | 65537 | 6 | 24 | build | 9.035350 | 9.012550 | 0.996737 | 0.961092..1.033149 | inconclusive | 2026103961 |
| 2 | 65537 | 6 | 24 | full | 15.952350 | 16.190000 | 0.979140 | 0.956776..1.004049 | inconclusive | 2026103962 |
| 2 | 65537 | 6 | 24 | host | 0.526800 | 0.517050 | 1.001585 | 0.974903..1.029733 | inconclusive | 2026103963 |
| 2 | 1000003 | 6 | 24 | build | 27.708700 | 27.588950 | 1.006587 | 0.993763..1.018685 | inconclusive | 2026103971 |
| 2 | 1000003 | 6 | 24 | full | 49.108050 | 48.865150 | 1.011692 | 0.993428..1.030298 | inconclusive | 2026103972 |
| 2 | 1000003 | 6 | 24 | host | 3.156300 | 3.119200 | 0.992762 | 0.956629..1.025402 | inconclusive | 2026103973 |
| 2 | 65537 | 8 | 6 | build | 9.001550 | 8.870700 | 0.986842 | 0.956468..1.017681 | inconclusive | 2026103981 |
| 2 | 65537 | 8 | 6 | full | 15.802100 | 15.887150 | 0.991811 | 0.967633..1.016755 | inconclusive | 2026103982 |
| 2 | 65537 | 8 | 6 | host | 0.647800 | 0.643600 | 0.998079 | 0.961243..1.036968 | inconclusive | 2026103983 |
| 2 | 1000003 | 8 | 6 | build | 19.724400 | 19.215650 | 0.999288 | 0.974963..1.024988 | inconclusive | 2026103991 |
| 2 | 1000003 | 8 | 6 | full | 31.296750 | 31.870650 | 1.005235 | 0.971655..1.041881 | inconclusive | 2026103992 |
| 2 | 1000003 | 8 | 6 | host | 4.374650 | 4.532350 | 0.958885 | 0.926748..0.990975 | adverse | 2026103993 |
| 2 | 65537 | 23 | 10 | build | 10.172000 | 10.120600 | 1.028875 | 0.974070..1.100457 | inconclusive | 2026104001 |
| 2 | 65537 | 23 | 10 | full | 19.027500 | 18.780150 | 1.010765 | 0.980423..1.044353 | inconclusive | 2026104002 |
| 2 | 65537 | 23 | 10 | host | 0.883800 | 0.967600 | 0.931008 | 0.864269..1.026913 | inconclusive | 2026104003 |
| 2 | 1000003 | 23 | 10 | build | 20.961200 | 22.274750 | 0.939550 | 0.924114..0.955458 | adverse | 2026104011 |
| 2 | 1000003 | 23 | 10 | full | 42.269800 | 44.164900 | 0.954877 | 0.932868..0.977133 | adverse | 2026104012 |
| 2 | 1000003 | 23 | 10 | host | 5.129700 | 6.423900 | 0.797613 | 0.767369..0.833053 | adverse | 2026104013 |
| 2 | 65537 | 24 | 6 | build | 9.450650 | 9.238600 | 1.012097 | 0.972585..1.051054 | inconclusive | 2026104021 |
| 2 | 65537 | 24 | 6 | full | 16.832500 | 16.773200 | 1.008775 | 0.983825..1.034418 | inconclusive | 2026104022 |
| 2 | 65537 | 24 | 6 | host | 0.876100 | 0.981900 | 0.909636 | 0.874718..0.950530 | adverse | 2026104023 |
| 2 | 1000003 | 24 | 6 | build | 19.391600 | 21.222000 | 0.935811 | 0.915364..0.957255 | adverse | 2026104031 |
| 2 | 1000003 | 24 | 6 | full | 31.353750 | 31.827100 | 0.953843 | 0.925195..0.983943 | adverse | 2026104032 |
| 2 | 1000003 | 24 | 6 | host | 5.118800 | 6.362950 | 0.788529 | 0.754205..0.817623 | adverse | 2026104033 |
| 2 | 257 | 7 | 0 | build | 7.224550 | 7.169650 | 1.003684 | 0.967089..1.039296 | inconclusive | 2026104041 |
| 2 | 257 | 7 | 0 | full | 14.340600 | 14.258900 | 1.012168 | 0.981464..1.044950 | inconclusive | 2026104042 |
| 2 | 257 | 7 | 0 | host | 0.003350 | 0.003200 | 1.051294 | 0.931744..1.182863 | inconclusive | 2026104043 |
| 2 | 65537 | 25 | 8 | build | 10.268500 | 10.050700 | 1.015132 | 0.965056..1.070627 | inconclusive | 2026104051 |
| 2 | 65537 | 25 | 8 | full | 17.789450 | 17.404650 | 1.000397 | 0.972492..1.028694 | inconclusive | 2026104052 |
| 2 | 65537 | 25 | 8 | host | 0.951950 | 0.955150 | 0.999054 | 0.970829..1.031141 | inconclusive | 2026104053 |
| 2 | 257 | 21 | 6 | build | 8.183150 | 8.278250 | 1.002964 | 0.975655..1.031268 | inconclusive | 2026104061 |
| 2 | 257 | 21 | 6 | full | 15.321500 | 15.423800 | 0.993904 | 0.968920..1.020172 | inconclusive | 2026104062 |
| 2 | 257 | 21 | 6 | host | 0.004250 | 0.003200 | 1.287705 | 1.165303..1.429137 | favorable | 2026104063 |
| 2 | 1000003 | 27 | 6 | build | 19.353150 | 19.191300 | 1.012779 | 0.986653..1.041836 | inconclusive | 2026104071 |
| 2 | 1000003 | 27 | 6 | full | 29.590450 | 30.292950 | 0.983259 | 0.952174..1.011617 | inconclusive | 2026104072 |
| 2 | 1000003 | 27 | 6 | host | 4.433500 | 4.413850 | 0.997484 | 0.975248..1.020923 | inconclusive | 2026104073 |

## Every fresh-runtime resident comparison

| Survey | N | Pattern | Digits | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 5 | 10 | 36.424700 | 34.824250 | 1.039204 | 0.958612..1.124946 | inconclusive | 2026102842 |
| 1 | 65537 | 5 | 10 | 35.102350 | 32.520250 | 1.038463 | 0.960451..1.118016 | inconclusive | 2026102843 |
| 1 | 65537 | 22 | 10 | 31.466150 | 31.884300 | 1.034505 | 0.955614..1.123542 | inconclusive | 2026102844 |
| 1 | 65537 | 23 | 10 | 28.579100 | 27.368550 | 0.991988 | 0.903818..1.082163 | inconclusive | 2026102845 |
| 1 | 65537 | 6 | 24 | 26.521800 | 26.376250 | 1.019474 | 0.922810..1.127268 | inconclusive | 2026102846 |
| 1 | 524289 | 5 | 10 | 38.214800 | 37.213050 | 1.005339 | 0.903393..1.126193 | inconclusive | 2026102847 |
| 2 | 257 | 5 | 10 | 22.924750 | 22.373050 | 1.054589 | 0.979282..1.134394 | inconclusive | 2026103842 |
| 2 | 65537 | 5 | 10 | 25.087250 | 23.881200 | 1.020986 | 0.905100..1.172595 | inconclusive | 2026103843 |
| 2 | 65537 | 22 | 10 | 25.067250 | 23.920050 | 0.986367 | 0.893949..1.086603 | inconclusive | 2026103844 |
| 2 | 65537 | 23 | 10 | 24.782350 | 23.058700 | 1.008356 | 0.939623..1.080673 | inconclusive | 2026103845 |
| 2 | 65537 | 6 | 24 | 23.069850 | 22.858000 | 1.004341 | 0.894752..1.123794 | inconclusive | 2026103846 |
| 2 | 524289 | 5 | 10 | 34.929350 | 32.815350 | 1.016883 | 0.923305..1.104708 | inconclusive | 2026103847 |

## Source and local evidence

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sign_digits_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_probe_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_probe_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_probe_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_probe_build_reference.rs`

Ignored raw logs, samples, identities, validation, hashes and delivery receipts remain under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-probe-*`. Published diagnostics contain counts/capabilities without process identities or adapter names. These tests do not establish a competitor maturity ranking or a release gate on other devices.
