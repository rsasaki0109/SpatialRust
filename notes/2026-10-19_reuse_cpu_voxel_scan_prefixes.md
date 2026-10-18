# Reusing checked CPU voxel digit-scan prefixes

Date: 2026-10-19. Named slice: checked CPU voxel digit-scan prefixes.
Immediate parent: `094abffdd9ff0708365138705d3d59453bea7648` (merged PR #280).

## Change and correctness argument

The parent's saturated-probe fast path scans all entries with a global XOR/OR
summary. If an omitted digit changes late and signs are mixed, it then discards
that scan and summarizes every entry again by coordinate sign. This family
retains the checked prefix so fallback does not repeat the complete scan.

Admission remains the same: a nonempty 256-point probe either proves all 24
required digits, handles the complete small input, or compares its global and
sign-class raw masks. Unequal masks use the original complete class summary.
For equal masks, P is the probe's unpadded class mask and is a lower bound on
the complete class mask. Expand its selected nibbles into three permitted-bit
masks. Each worker uses its own band's actual first entry and reads blocks of
at most 16,384 entries with the lightweight XOR/OR fold.

As long as a checked block differs only in permitted bits, every omitted nibble
in that band's checked prefix equals the first entry's. The sign bits of its
accumulated differences establish exactly which coordinate signs occurred.
Retain one representative for each observed sign; a second sign toggles only
bit 31. These representatives preserve AND/OR values of all omitted nibbles
exactly. Values inside P may be summarized coarsely because P is already a
proven lower bound and is ORed into the final raw mask.

When a block first varies in an omitted bit, summarize that block and the
remaining suffix by sign and merge the representatives of earlier checked
blocks. No earlier block is rescanned. A fast band with no fallback
returns only its sign representatives. Merging bands preserves omitted-nibble
AND/OR and observed sign presence, including differences between individually
constant bands. Thus `P OR digits(merged_summary)` is the exact original class
mask, before applying the unchanged even-pass padding. This requires exact
membership, not equality of digit counts. The argument also covers one-sign
probes acquiring an opposite sign before fallback, worker-local first entries,
empty sign classes and omitted digits with nonzero constant values.

Each worker repeats at most one already-checked block, rather than its whole
band. The fallback suffix is traversed once with the class summary. Early or
middle changes still require class work on the remaining suffix; this is not
an unconditional one-pass or speed guarantee. There is no per-point planner
allocation. The existing 524,288-point parallel threshold, useful 131,072-point
grains and at most four native workers including the caller remain. WASM and
small/one-worker calls remain serial.

## Contracts and validation

Only the private CPU-key planner changes production behavior. The clamped
16-byte entries and global source indices, public APIs/admission, selected GPU
digit masks, stable raw order, shaders/pipelines/sort/compaction, explicit
16*N uploads, nonempty four-byte scalar synchronization and metadata D2H remain
unchanged. GPU-resident key input still uses all 24 digits without CPU inspection.
No core, dependency, feature, unsafe or GPU scheduling changes are introduced.
This is a CPU planning optimization; physical memory, bandwidth, energy and
GPU-stage speed improvements are not inferred from wall time.

Two new native regressions compare immediate-parent masks with an independent
per-sign nibble oracle and stable CPU LSD order, including forced 0/1/2/3/4/12/
maximum worker availability. Cases cover all signed widths, empty/singleton,
probe/block/worker thresholds, sparse and saturated digits, nonzero omitted
constants, clamp extremes and early/middle/last/worker-boundary digit or sign
changes. Forced adaptive summaries are also checked outside normal admission.
Complete parent/current resident and small host-returning outputs and raw sorted
entry words match independent CPU expectations. Default/tiny/zero pool budgets,
actual stages/query bytes/requests, owned output, GPU-key input/admission and
concurrent results surviving host mutation, pool clears and changed reuse are
checked. Existing tests change only to register the native module.

Native-only frozen references reproduce the immediate-parent planner and full
CPU/device-key API bodies, changing imports/paths only.

- VOXEL_SCAN_OUTPUT verified 165234016 parent/current/CPU raw words across 72 complete fixture pairs; unchanged masks/passes/queries/requests, ownership, budgets, device-key input/admission, concurrent builds, mutation and changed reuse.

- VOXEL_SCAN_PLAN verified 3278 parent/oracle/worker masks and stable signed LSD orders; empty classes, every signed width, exact raw-bit membership, 256-point probe and four-worker/block boundaries, saturated probes and early/middle/late fallback.

Final validation: 282 native test executions, excluding repeated targeted prechecks.

| Check | Tests passed |
| --- | ---: |
| clippy | 0 |
| voxel | 240 |
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

All-feature/all-target Clippy denies warnings. CPU-only, minimal GPU WASM,
CI-feature GPU/image WASM, formatting and diff checks pass. 622
original GPU/image sources are unchanged; 634 current source,
manifest and executable hashes match across precheck, measurements and final
validation. Published files and PR text have zero forbidden adapter-name matches.

## Measurement method and limitations

Two consecutive release surveys each cover 24 synthetic cases, three warmups
and 40 measured pairs on complete resident build, host-returning full calls and
host-only original preparation plus planning. Each variant uses its own planner.
Resident timing includes preparation, planning, upload, sorting, compaction,
four-byte synchronization and completion. Full timing also includes metadata
D2H and host conversion. Checks and result destruction are outside timing.
Balanced four-position XOR rotation alternates old/current build/full ordering;
host variants alternate and black-box inputs and selected masks within timing.

Six fresh-runtime scenarios each use 20 alternating pairs in two surveys, zero
warmups and a fresh runtime/device per variant created outside timing. First
resident construction includes its planner and pipeline use; checks and returned
result destruction are excluded. No explicit driver-cache reset is requested.
These are not process-cold or hardware-cold guarantees. Runtime creation cost,
other devices and application workloads are unmeasured.

Ratios are paired geometric means parent/current (>1 favors the change). Each
individual 95% interval uses 10,000 paired log bootstrap draws, sorted indices
249/9749, with math.fsum(values)/N means. Warm seeds are 2026101941 + 1000*survey
+ 10*case_index + route_index (one-based surveys, zero-based cases/routes;
build/full/host order). Cold seeds are 2026101942 + 1000*survey + case_index.
No multiple-comparison correction, epsilon, outlier removal, dropped pair or
selected rerun is applied. All rows, controls and adverse cases are retained.
Shared-host pre/post samples cannot establish isolation during every pair or
portable application speed. This task's builds/tests do not overlap measurements.

Patterns 23/24 introduce a previously omitted digit at the last point. Patterns
29/30 use the pattern-23 base but change the middle or index 257 respectively.
Correctness-only pattern 31 changes the first entry of the last worker band;
pattern 32 adds a middle opposite sign and late clamp extremes to positive data.
Patterns 5/21/22 exercise saturated probes without fallback. Controls cover
bounded mixed-sign, full-width random, positive-only, constant and late opposite
signs. Point counts span 257..1,000,003 and both sides of the worker threshold;
empty/singleton and 16,384-block boundaries are correctness-tested. Exact
fixture definitions and raw evidence paths are listed below.

| Route | Group | Comparisons | Paired ratio range |
| --- | --- | --- | --- |
| build | fallback | 13 favorable / 5 inconclusive / 0 adverse / 0 unresolved | 0.984538..1.119606 |
| build | saturated | 0 favorable / 12 inconclusive / 2 adverse / 0 unresolved | 0.946348..1.024873 |
| build | control | 0 favorable / 15 inconclusive / 1 adverse / 0 unresolved | 0.963899..1.030142 |
| full | fallback | 12 favorable / 6 inconclusive / 0 adverse / 0 unresolved | 1.002292..1.083089 |
| full | saturated | 0 favorable / 13 inconclusive / 1 adverse / 0 unresolved | 0.975011..1.021944 |
| full | control | 0 favorable / 15 inconclusive / 1 adverse / 0 unresolved | 0.970191..1.029071 |
| host | fallback | 18 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1.361400..1.478893 |
| host | saturated | 0 favorable / 10 inconclusive / 4 adverse / 0 unresolved | 0.921057..1.001719 |
| host | control | 2 favorable / 14 inconclusive / 0 adverse / 0 unresolved | 0.944823..1.435601 |

Fresh-runtime comparisons: 2 favorable / 9 inconclusive / 1 adverse / 0 unresolved.

For all 18 fallback host preparation/planning comparisons, the individual
interval favors this change: paired elapsed reductions span 26.55%..32.38%.
This host diagnostic is separate from complete-call time. Complete fallback
resident calls have 13 favorable/5 inconclusive intervals, and host-returning
calls have 12 favorable/6 inconclusive intervals. Ordinary saturated/control
calls include three adverse resident and two adverse host-returning observations,
listed below. The tradeoff is retained rather than claiming every input improves.

Fresh-runtime results have two favorable, nine inconclusive and one adverse
interval. The adverse N=257/pattern-5 observation in survey 1 has ratio 0.893312
(0.819292..0.973930), about 11.94% more time. Survey 2 for the same case is
inconclusive. These observations do not establish a stable first-use benefit
or identify a cause on the shared host.

All individually adverse warm complete-call observations:

- build, survey 1, N=65537, pattern 5 (saturated): 5.67% more time, ratio 0.946348 (0.912730..0.977269).

- build, survey 1, N=1000003, pattern 5 (saturated): 1.40% more time, ratio 0.986150 (0.974264..0.998336).

- build, survey 1, N=257, pattern 7 (control): 3.75% more time, ratio 0.963899 (0.931829..0.996174).

- full, survey 2, N=65537, pattern 21 (saturated): 2.56% more time, ratio 0.975011 (0.952887..0.997071).

- full, survey 2, N=65537, pattern 8 (control): 3.07% more time, ratio 0.970191 (0.943901..0.997140).

No cause is established from adverse/inconclusive shared-host observations.

| Survey | Kind | Quiet observed | Busy counts pre/post | Maximum busy CPU cores pre/post | GPU samples pre/post |
| ---: | --- | --- | --- | --- | --- |
| 1 | warm | False | [3, 1] | [1.301, 0.982] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 2 | warm | False | [1, 1] | [0.976, 0.964] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 1 | fresh-runtime | False | [3, 1] | [1.396, 0.969] | `[[[[0, 0]], [[0, 0]]], [[[1, 0]], [[0, 0]]]]` |
| 2 | fresh-runtime | False | [1, 1] | [0.958, 0.982] | `[[[[0, 0]], [[0, 0]]], [[[1, 0]], [[0, 0]]]]` |

Environment labels without local adapter model:

- warm 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=40 warmups=3 wall_order=balanced host_order=alternating checks_drop_excluded=true`

- warm 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=40 warmups=3 wall_order=balanced host_order=alternating checks_drop_excluded=true`

- fresh-runtime 1: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`

- fresh-runtime 2: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`

## Every warm comparison

| Survey | N | Pattern | Digits | Route | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 5 | 10 | build | 1.715650 | 1.727400 | 1.008349 | 0.971676..1.047069 | inconclusive | 2026102941 |
| 1 | 257 | 5 | 10 | full | 2.917800 | 2.792850 | 1.021944 | 0.984654..1.064416 | inconclusive | 2026102942 |
| 1 | 257 | 5 | 10 | host | 0.002350 | 0.002500 | 0.921057 | 0.813473..1.041151 | inconclusive | 2026102943 |
| 1 | 65537 | 5 | 10 | build | 2.654200 | 2.649500 | 0.946348 | 0.912730..0.977269 | adverse | 2026102951 |
| 1 | 65537 | 5 | 10 | full | 4.805050 | 4.835750 | 0.994515 | 0.961356..1.029668 | inconclusive | 2026102952 |
| 1 | 65537 | 5 | 10 | host | 0.417750 | 0.441200 | 0.949384 | 0.915551..0.985161 | adverse | 2026102953 |
| 1 | 1000003 | 5 | 10 | build | 17.022700 | 17.568350 | 0.986150 | 0.974264..0.998336 | adverse | 2026102961 |
| 1 | 1000003 | 5 | 10 | full | 32.367950 | 31.857350 | 1.007728 | 0.995594..1.020172 | inconclusive | 2026102962 |
| 1 | 1000003 | 5 | 10 | host | 3.331100 | 3.397000 | 0.986282 | 0.963441..1.013066 | inconclusive | 2026102963 |
| 1 | 65537 | 21 | 6 | build | 7.458900 | 7.157250 | 1.024873 | 0.985778..1.063418 | inconclusive | 2026102971 |
| 1 | 65537 | 21 | 6 | full | 13.986950 | 13.654550 | 0.999820 | 0.972740..1.028324 | inconclusive | 2026102972 |
| 1 | 65537 | 21 | 6 | host | 0.498200 | 0.552150 | 0.966667 | 0.939642..0.993238 | adverse | 2026102973 |
| 1 | 1000003 | 21 | 6 | build | 13.822650 | 14.086150 | 0.996823 | 0.969573..1.031324 | inconclusive | 2026102981 |
| 1 | 1000003 | 21 | 6 | full | 22.174650 | 21.458100 | 1.013640 | 0.984704..1.040287 | inconclusive | 2026102982 |
| 1 | 1000003 | 21 | 6 | host | 3.458650 | 3.519200 | 0.989498 | 0.969230..1.010323 | inconclusive | 2026102983 |
| 1 | 65537 | 22 | 10 | build | 7.336750 | 7.262550 | 0.986837 | 0.952820..1.022002 | inconclusive | 2026102991 |
| 1 | 65537 | 22 | 10 | full | 13.916300 | 14.056300 | 1.008415 | 0.982099..1.036223 | inconclusive | 2026102992 |
| 1 | 65537 | 22 | 10 | host | 0.476200 | 0.494600 | 0.978571 | 0.942812..1.016362 | inconclusive | 2026102993 |
| 1 | 1000003 | 22 | 10 | build | 16.586150 | 16.660550 | 0.993897 | 0.983084..1.005027 | inconclusive | 2026103001 |
| 1 | 1000003 | 22 | 10 | full | 31.312550 | 31.556750 | 1.004170 | 0.985995..1.022115 | inconclusive | 2026103002 |
| 1 | 1000003 | 22 | 10 | host | 3.358100 | 3.417650 | 0.998845 | 0.980855..1.019373 | inconclusive | 2026103003 |
| 1 | 65537 | 0 | 8 | build | 7.553700 | 7.624200 | 0.998417 | 0.953692..1.044776 | inconclusive | 2026103011 |
| 1 | 65537 | 0 | 8 | full | 13.638350 | 13.878850 | 0.993974 | 0.966439..1.022223 | inconclusive | 2026103012 |
| 1 | 65537 | 0 | 8 | host | 0.672250 | 0.686950 | 0.993210 | 0.959735..1.025562 | inconclusive | 2026103013 |
| 1 | 1000003 | 0 | 8 | build | 15.328900 | 15.419850 | 1.008486 | 0.992189..1.026137 | inconclusive | 2026103021 |
| 1 | 1000003 | 0 | 8 | full | 23.288650 | 23.476200 | 1.008482 | 0.988974..1.030537 | inconclusive | 2026103022 |
| 1 | 1000003 | 0 | 8 | host | 4.270300 | 4.260500 | 1.003234 | 0.985008..1.021557 | inconclusive | 2026103023 |
| 1 | 65537 | 6 | 24 | build | 7.965250 | 7.603750 | 1.030142 | 0.991622..1.070067 | inconclusive | 2026103031 |
| 1 | 65537 | 6 | 24 | full | 14.014450 | 14.276950 | 0.985505 | 0.956303..1.014684 | inconclusive | 2026103032 |
| 1 | 65537 | 6 | 24 | host | 0.390300 | 0.445400 | 0.970088 | 0.922633..1.020705 | inconclusive | 2026103033 |
| 1 | 1000003 | 6 | 24 | build | 20.659050 | 21.010800 | 0.991833 | 0.983333..1.001277 | inconclusive | 2026103041 |
| 1 | 1000003 | 6 | 24 | full | 34.846750 | 34.710800 | 0.997339 | 0.981547..1.013020 | inconclusive | 2026103042 |
| 1 | 1000003 | 6 | 24 | host | 2.462950 | 2.427250 | 1.002372 | 0.985886..1.019604 | inconclusive | 2026103043 |
| 1 | 65537 | 8 | 6 | build | 6.928350 | 6.940400 | 0.974981 | 0.941652..1.009178 | inconclusive | 2026103051 |
| 1 | 65537 | 8 | 6 | full | 12.344450 | 12.992850 | 0.977220 | 0.938017..1.014197 | inconclusive | 2026103052 |
| 1 | 65537 | 8 | 6 | host | 0.538700 | 0.513850 | 0.979299 | 0.944276..1.017187 | inconclusive | 2026103053 |
| 1 | 1000003 | 8 | 6 | build | 14.715250 | 14.648250 | 0.979000 | 0.937493..1.012353 | inconclusive | 2026103061 |
| 1 | 1000003 | 8 | 6 | full | 22.508850 | 22.333650 | 0.995597 | 0.976616..1.015452 | inconclusive | 2026103062 |
| 1 | 1000003 | 8 | 6 | host | 3.688200 | 3.730000 | 0.985312 | 0.960089..1.009252 | inconclusive | 2026103063 |
| 1 | 65537 | 23 | 10 | build | 8.473350 | 7.934500 | 1.040743 | 0.996987..1.084933 | inconclusive | 2026103071 |
| 1 | 65537 | 23 | 10 | full | 15.531250 | 15.478950 | 1.002292 | 0.972804..1.030391 | inconclusive | 2026103072 |
| 1 | 65537 | 23 | 10 | host | 0.796100 | 0.578500 | 1.402768 | 1.360888..1.441046 | favorable | 2026103073 |
| 1 | 524287 | 23 | 10 | build | 25.022650 | 22.923200 | 1.062647 | 1.015770..1.094569 | favorable | 2026103081 |
| 1 | 524287 | 23 | 10 | full | 39.359150 | 38.424400 | 1.027622 | 1.001036..1.047097 | favorable | 2026103082 |
| 1 | 524287 | 23 | 10 | host | 6.276900 | 4.446500 | 1.466800 | 1.413509..1.524847 | favorable | 2026103083 |
| 1 | 524288 | 23 | 10 | build | 12.841850 | 11.531000 | 1.086984 | 1.061744..1.113208 | favorable | 2026103091 |
| 1 | 524288 | 23 | 10 | full | 25.568500 | 24.084950 | 1.055102 | 1.027051..1.087910 | favorable | 2026103092 |
| 1 | 524288 | 23 | 10 | host | 3.506450 | 2.501050 | 1.416645 | 1.380515..1.455331 | favorable | 2026103093 |
| 1 | 524289 | 23 | 10 | build | 21.993800 | 21.235900 | 1.061982 | 1.034800..1.092585 | favorable | 2026103101 |
| 1 | 524289 | 23 | 10 | full | 35.673000 | 35.458200 | 1.043937 | 1.010101..1.079105 | favorable | 2026103102 |
| 1 | 524289 | 23 | 10 | host | 3.418550 | 2.386850 | 1.421068 | 1.387514..1.455265 | favorable | 2026103103 |
| 1 | 1000003 | 23 | 10 | build | 19.326450 | 18.220400 | 1.083601 | 1.023871..1.141720 | favorable | 2026103111 |
| 1 | 1000003 | 23 | 10 | full | 35.908750 | 34.185800 | 1.082773 | 1.033674..1.144533 | favorable | 2026103112 |
| 1 | 1000003 | 23 | 10 | host | 5.153700 | 3.500500 | 1.455217 | 1.418279..1.495508 | favorable | 2026103113 |
| 1 | 65537 | 24 | 6 | build | 7.716800 | 7.787750 | 1.026006 | 0.984183..1.069278 | inconclusive | 2026103121 |
| 1 | 65537 | 24 | 6 | full | 14.120200 | 13.698750 | 1.003505 | 0.968037..1.040699 | inconclusive | 2026103122 |
| 1 | 65537 | 24 | 6 | host | 0.748550 | 0.502500 | 1.400300 | 1.358636..1.445189 | favorable | 2026103123 |
| 1 | 1000003 | 24 | 6 | build | 16.704050 | 14.974900 | 1.119606 | 1.087137..1.151563 | favorable | 2026103131 |
| 1 | 1000003 | 24 | 6 | full | 26.211400 | 24.386350 | 1.054992 | 1.006038..1.098589 | favorable | 2026103132 |
| 1 | 1000003 | 24 | 6 | host | 5.418550 | 3.836800 | 1.457299 | 1.417150..1.507723 | favorable | 2026103133 |
| 1 | 1000003 | 29 | 10 | build | 21.076000 | 19.440000 | 1.088104 | 1.052628..1.117862 | favorable | 2026103141 |
| 1 | 1000003 | 29 | 10 | full | 37.957250 | 35.712300 | 1.031899 | 1.007762..1.053995 | favorable | 2026103142 |
| 1 | 1000003 | 29 | 10 | host | 5.480900 | 3.803650 | 1.432646 | 1.371440..1.486165 | favorable | 2026103143 |
| 1 | 1000003 | 30 | 10 | build | 21.730050 | 20.217900 | 1.057422 | 1.007075..1.091583 | favorable | 2026103151 |
| 1 | 1000003 | 30 | 10 | full | 39.209200 | 37.956850 | 1.036543 | 1.016065..1.058126 | favorable | 2026103152 |
| 1 | 1000003 | 30 | 10 | host | 5.458700 | 3.945250 | 1.382240 | 1.353346..1.411964 | favorable | 2026103153 |
| 1 | 257 | 7 | 0 | build | 7.042150 | 7.338400 | 0.963899 | 0.931829..0.996174 | adverse | 2026103161 |
| 1 | 257 | 7 | 0 | full | 13.712200 | 13.839900 | 0.997545 | 0.967475..1.028660 | inconclusive | 2026103162 |
| 1 | 257 | 7 | 0 | host | 0.004300 | 0.004350 | 0.946716 | 0.801347..1.119140 | inconclusive | 2026103163 |
| 1 | 65537 | 25 | 8 | build | 9.985000 | 9.751200 | 1.025473 | 0.981097..1.072151 | inconclusive | 2026103171 |
| 1 | 65537 | 25 | 8 | full | 17.615700 | 17.151550 | 1.029071 | 0.999990..1.059882 | inconclusive | 2026103172 |
| 1 | 65537 | 25 | 8 | host | 0.910800 | 0.620150 | 1.435601 | 1.400836..1.471464 | favorable | 2026103173 |
| 2 | 257 | 5 | 10 | build | 1.533700 | 1.545200 | 0.993213 | 0.976078..1.010154 | inconclusive | 2026103941 |
| 2 | 257 | 5 | 10 | full | 2.580200 | 2.592050 | 0.998639 | 0.982063..1.015843 | inconclusive | 2026103942 |
| 2 | 257 | 5 | 10 | host | 0.001950 | 0.002100 | 0.943410 | 0.863768..1.031958 | inconclusive | 2026103943 |
| 2 | 65537 | 5 | 10 | build | 2.637550 | 2.675750 | 1.005101 | 0.977988..1.033363 | inconclusive | 2026103951 |
| 2 | 65537 | 5 | 10 | full | 4.857650 | 4.846050 | 0.992449 | 0.967282..1.017154 | inconclusive | 2026103952 |
| 2 | 65537 | 5 | 10 | host | 0.402600 | 0.418000 | 0.953253 | 0.925226..0.981104 | adverse | 2026103953 |
| 2 | 1000003 | 5 | 10 | build | 18.528900 | 18.881500 | 0.994425 | 0.980862..1.007950 | inconclusive | 2026103961 |
| 2 | 1000003 | 5 | 10 | full | 34.283050 | 34.514450 | 0.994884 | 0.974648..1.013372 | inconclusive | 2026103962 |
| 2 | 1000003 | 5 | 10 | host | 3.545800 | 3.598550 | 0.986353 | 0.965352..1.008394 | inconclusive | 2026103963 |
| 2 | 65537 | 21 | 6 | build | 7.393550 | 7.570550 | 0.977055 | 0.942338..1.010780 | inconclusive | 2026103971 |
| 2 | 65537 | 21 | 6 | full | 13.860350 | 14.021700 | 0.975011 | 0.952887..0.997071 | adverse | 2026103972 |
| 2 | 65537 | 21 | 6 | host | 0.530450 | 0.526850 | 0.985340 | 0.955761..1.016392 | inconclusive | 2026103973 |
| 2 | 1000003 | 21 | 6 | build | 15.157100 | 15.027450 | 1.000391 | 0.971307..1.035455 | inconclusive | 2026103981 |
| 2 | 1000003 | 21 | 6 | full | 24.214850 | 23.760200 | 1.013221 | 0.993365..1.033104 | inconclusive | 2026103982 |
| 2 | 1000003 | 21 | 6 | host | 3.956900 | 4.126650 | 0.973993 | 0.953757..0.994267 | adverse | 2026103983 |
| 2 | 65537 | 22 | 10 | build | 8.061800 | 8.077250 | 1.018660 | 0.974155..1.066046 | inconclusive | 2026103991 |
| 2 | 65537 | 22 | 10 | full | 16.024550 | 15.753300 | 0.997125 | 0.959477..1.030960 | inconclusive | 2026103992 |
| 2 | 65537 | 22 | 10 | host | 0.516350 | 0.496600 | 1.001719 | 0.958596..1.045073 | inconclusive | 2026103993 |
| 2 | 1000003 | 22 | 10 | build | 16.797750 | 17.281400 | 0.985055 | 0.967148..1.001436 | inconclusive | 2026104001 |
| 2 | 1000003 | 22 | 10 | full | 32.362450 | 32.172250 | 0.999975 | 0.980026..1.019815 | inconclusive | 2026104002 |
| 2 | 1000003 | 22 | 10 | host | 3.480600 | 3.488700 | 0.988716 | 0.966514..1.010818 | inconclusive | 2026104003 |
| 2 | 65537 | 0 | 8 | build | 7.758750 | 7.724550 | 0.990233 | 0.958973..1.021730 | inconclusive | 2026104011 |
| 2 | 65537 | 0 | 8 | full | 13.875350 | 13.857300 | 0.986296 | 0.949554..1.019537 | inconclusive | 2026104012 |
| 2 | 65537 | 0 | 8 | host | 0.645250 | 0.686950 | 0.971803 | 0.931484..1.010601 | inconclusive | 2026104013 |
| 2 | 1000003 | 0 | 8 | build | 15.109800 | 14.992150 | 0.991159 | 0.969768..1.012926 | inconclusive | 2026104021 |
| 2 | 1000003 | 0 | 8 | full | 23.173350 | 23.098050 | 0.999608 | 0.979193..1.020260 | inconclusive | 2026104022 |
| 2 | 1000003 | 0 | 8 | host | 4.235050 | 4.266700 | 0.989530 | 0.968548..1.011847 | inconclusive | 2026104023 |
| 2 | 65537 | 6 | 24 | build | 8.544350 | 8.209700 | 1.010014 | 0.964262..1.057556 | inconclusive | 2026104031 |
| 2 | 65537 | 6 | 24 | full | 15.351650 | 14.735450 | 1.009832 | 0.972463..1.044537 | inconclusive | 2026104032 |
| 2 | 65537 | 6 | 24 | host | 0.452350 | 0.461300 | 0.990631 | 0.955841..1.025830 | inconclusive | 2026104033 |
| 2 | 1000003 | 6 | 24 | build | 21.605250 | 21.811500 | 0.998889 | 0.989818..1.008343 | inconclusive | 2026104041 |
| 2 | 1000003 | 6 | 24 | full | 36.680000 | 37.112350 | 0.997332 | 0.980100..1.013922 | inconclusive | 2026104042 |
| 2 | 1000003 | 6 | 24 | host | 2.534700 | 2.509000 | 0.987828 | 0.966794..1.008874 | inconclusive | 2026104043 |
| 2 | 65537 | 8 | 6 | build | 7.044300 | 6.900000 | 1.007790 | 0.972961..1.044182 | inconclusive | 2026104051 |
| 2 | 65537 | 8 | 6 | full | 12.700150 | 12.902150 | 0.970191 | 0.943901..0.997140 | adverse | 2026104052 |
| 2 | 65537 | 8 | 6 | host | 0.481600 | 0.496600 | 0.993882 | 0.966119..1.024167 | inconclusive | 2026104053 |
| 2 | 1000003 | 8 | 6 | build | 14.473250 | 14.205150 | 1.013351 | 0.977272..1.060753 | inconclusive | 2026104061 |
| 2 | 1000003 | 8 | 6 | full | 22.145500 | 21.798450 | 0.992812 | 0.975756..1.009131 | inconclusive | 2026104062 |
| 2 | 1000003 | 8 | 6 | host | 3.632600 | 3.612350 | 0.990649 | 0.968973..1.013147 | inconclusive | 2026104063 |
| 2 | 65537 | 23 | 10 | build | 7.553550 | 7.642000 | 0.984538 | 0.953753..1.017235 | inconclusive | 2026104071 |
| 2 | 65537 | 23 | 10 | full | 14.244700 | 13.940550 | 1.007099 | 0.972978..1.041678 | inconclusive | 2026104072 |
| 2 | 65537 | 23 | 10 | host | 0.704300 | 0.516750 | 1.361400 | 1.318559..1.406612 | favorable | 2026104073 |
| 2 | 524287 | 23 | 10 | build | 13.470400 | 12.307000 | 1.104776 | 1.068704..1.142441 | favorable | 2026104081 |
| 2 | 524287 | 23 | 10 | full | 25.079150 | 23.786700 | 1.046722 | 1.013477..1.076850 | favorable | 2026104082 |
| 2 | 524287 | 23 | 10 | host | 4.731400 | 3.219000 | 1.478893 | 1.414347..1.545098 | favorable | 2026104083 |
| 2 | 524288 | 23 | 10 | build | 12.237800 | 11.477550 | 1.072135 | 1.049304..1.096637 | favorable | 2026104091 |
| 2 | 524288 | 23 | 10 | full | 24.676050 | 24.039300 | 1.015009 | 0.983510..1.043181 | inconclusive | 2026104092 |
| 2 | 524288 | 23 | 10 | host | 3.331150 | 2.362150 | 1.393431 | 1.347793..1.438432 | favorable | 2026104093 |
| 2 | 524289 | 23 | 10 | build | 13.555100 | 13.147150 | 1.033717 | 0.994557..1.066584 | inconclusive | 2026104101 |
| 2 | 524289 | 23 | 10 | full | 28.562450 | 27.109250 | 1.040714 | 1.011503..1.070649 | favorable | 2026104102 |
| 2 | 524289 | 23 | 10 | host | 3.281950 | 2.311000 | 1.435354 | 1.401754..1.470188 | favorable | 2026104103 |
| 2 | 1000003 | 23 | 10 | build | 20.516250 | 18.838700 | 1.093602 | 1.078102..1.108238 | favorable | 2026104111 |
| 2 | 1000003 | 23 | 10 | full | 37.219850 | 36.160900 | 1.039678 | 1.021674..1.057344 | favorable | 2026104112 |
| 2 | 1000003 | 23 | 10 | host | 5.043200 | 3.572450 | 1.418007 | 1.388661..1.448871 | favorable | 2026104113 |
| 2 | 65537 | 24 | 6 | build | 7.470350 | 7.335050 | 1.028533 | 0.997549..1.059402 | inconclusive | 2026104121 |
| 2 | 65537 | 24 | 6 | full | 13.800250 | 13.129900 | 1.026259 | 0.997709..1.056006 | inconclusive | 2026104122 |
| 2 | 65537 | 24 | 6 | host | 0.700450 | 0.521800 | 1.385955 | 1.339345..1.432659 | favorable | 2026104123 |
| 2 | 1000003 | 24 | 6 | build | 16.278450 | 14.679450 | 1.093328 | 1.065102..1.120808 | favorable | 2026104131 |
| 2 | 1000003 | 24 | 6 | full | 25.663950 | 23.616550 | 1.083089 | 1.058732..1.108330 | favorable | 2026104132 |
| 2 | 1000003 | 24 | 6 | host | 5.279350 | 3.687200 | 1.447492 | 1.415212..1.481007 | favorable | 2026104133 |
| 2 | 1000003 | 29 | 10 | build | 20.282850 | 18.232100 | 1.117549 | 1.081216..1.173424 | favorable | 2026104141 |
| 2 | 1000003 | 29 | 10 | full | 36.454450 | 35.433750 | 1.044299 | 1.023078..1.065944 | favorable | 2026104142 |
| 2 | 1000003 | 29 | 10 | host | 5.011550 | 3.457450 | 1.462419 | 1.432438..1.500043 | favorable | 2026104143 |
| 2 | 1000003 | 30 | 10 | build | 21.157050 | 19.463950 | 1.083363 | 1.067916..1.098350 | favorable | 2026104151 |
| 2 | 1000003 | 30 | 10 | full | 38.004650 | 36.761450 | 1.017128 | 0.995471..1.038536 | inconclusive | 2026104152 |
| 2 | 1000003 | 30 | 10 | host | 5.472050 | 3.994250 | 1.366241 | 1.338050..1.394344 | favorable | 2026104153 |
| 2 | 257 | 7 | 0 | build | 6.612650 | 6.598100 | 1.010644 | 0.973958..1.049750 | inconclusive | 2026104161 |
| 2 | 257 | 7 | 0 | full | 12.879450 | 12.778950 | 0.989021 | 0.960915..1.015422 | inconclusive | 2026104162 |
| 2 | 257 | 7 | 0 | host | 0.003350 | 0.004000 | 0.944823 | 0.776132..1.149928 | inconclusive | 2026104163 |
| 2 | 65537 | 25 | 8 | build | 8.898700 | 9.117500 | 0.980466 | 0.947010..1.017922 | inconclusive | 2026104171 |
| 2 | 65537 | 25 | 8 | full | 16.046500 | 15.776300 | 1.012388 | 0.981712..1.043369 | inconclusive | 2026104172 |
| 2 | 65537 | 25 | 8 | host | 0.750800 | 0.546650 | 1.424840 | 1.385938..1.464398 | favorable | 2026104173 |

## Every fresh-runtime resident comparison

| Survey | N | Pattern | Digits | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 5 | 10 | 24.314550 | 27.660950 | 0.893312 | 0.819292..0.973930 | adverse | 2026102942 |
| 1 | 65537 | 22 | 10 | 28.045650 | 26.199150 | 1.054480 | 1.010930..1.100628 | favorable | 2026102943 |
| 1 | 65537 | 23 | 10 | 30.997350 | 28.437500 | 1.220297 | 1.053632..1.474676 | favorable | 2026102944 |
| 1 | 65537 | 24 | 6 | 35.513200 | 30.104200 | 1.288686 | 0.953096..1.756723 | inconclusive | 2026102945 |
| 1 | 65537 | 6 | 24 | 26.118250 | 27.239300 | 0.872082 | 0.710907..1.037102 | inconclusive | 2026102946 |
| 1 | 524289 | 23 | 10 | 39.006200 | 37.063100 | 0.982066 | 0.814445..1.142913 | inconclusive | 2026102947 |
| 2 | 257 | 5 | 10 | 24.206450 | 22.638000 | 1.056722 | 0.990388..1.121197 | inconclusive | 2026103942 |
| 2 | 65537 | 22 | 10 | 24.680700 | 23.680350 | 1.029544 | 0.966271..1.095107 | inconclusive | 2026103943 |
| 2 | 65537 | 23 | 10 | 25.890650 | 25.380500 | 1.010836 | 0.908111..1.138869 | inconclusive | 2026103944 |
| 2 | 65537 | 24 | 6 | 25.217800 | 24.882500 | 0.998752 | 0.898958..1.100805 | inconclusive | 2026103945 |
| 2 | 65537 | 6 | 24 | 25.189400 | 25.145700 | 1.042428 | 0.946056..1.150096 | inconclusive | 2026103946 |
| 2 | 524289 | 23 | 10 | 34.665900 | 35.770500 | 0.993080 | 0.939610..1.049229 | inconclusive | 2026103947 |

## Source and local evidence

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_probe_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_scan_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_scan_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_scan_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_digit_scan_build_reference.rs`

Ignored raw logs, samples, process identities, validation, hashes and delivery receipts remain under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-scan-*`. Published diagnostics contain anonymous counts/capabilities. These tests do not establish a competitor speed ranking or a release gate on other devices.
