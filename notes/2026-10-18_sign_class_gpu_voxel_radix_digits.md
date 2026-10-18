# Sign-class constant digits in CPU-key GPU voxel sorting

Date: 2026-10-18. Named slice: sign-class CPU voxel radix digits.
Immediate parent: `de3934a602efd46e164cb548b1f97a896da3860d` (merged PR #278).

## Change and correctness argument

The previous global varying-nibble summary retained sign-extension digits for
bounded coordinates crossing zero. The private CPU-key planner now tracks
bitwise AND/OR separately for nonnegative and negative coordinates of each axis.
Within one class `AND XOR OR` identifies exactly the varying bits. Empty classes
start with AND=MAX and OR=0; a class is populated exactly when AND<=OR. These
identities merge componentwise by AND/OR, including empty worker-local classes.
Each sign-class summary is 48 bytes, independent of input length.

Keep every digit varying within either sign class. Also keep the most-significant
nibble whenever both signs occur, even if it is constant within each class.
The existing sign-bit flip in that final nibble orders negatives first. After
its stable pass, lower omitted digits cannot change within-class order because
they are constant there. This equals complete signed axis sorting. Applying the
original z/y/x LSD axis order preserves lexicographic keys and source-index order
for ties. Coordinates still use the unchanged original i32 clamp/preparation.

An odd selected count retains the lowest omitted digit for even ping-pong parity.
That digit can differ between signs and need not be globally an identity pass.
It is constant inside each class, and a later retained sign-separating digit
restores sign order. If that most-significant digit is omitted, only one sign
class is present and it is globally constant. Padding also preserves order;
results finish in the same original owned upload allocation without a final copy.
The selected count cannot exceed the previous even global count.

A 256-entry probe returns immediately if all 24 digits are necessary. Variation/
presence are monotone, so later entries cannot invalidate this decision. Otherwise
the complete input is summarized; at least 524,288 native points use at most four
workers including the caller, bounded by available CPUs and 131,072 points per
useful worker. Small, single-worker and WASM calls remain serial. Handles are
bounded, with no per-point planner vector or global thread budget. Additional
CPU work is included in the complete-call measurements below.

If the probe contains only one sign per axis, the production path first uses a
cheap three-word global XOR/OR summary against the first entry, with the same worker cap.
It returns that mask only after the complete summary proves no sign bit differs.
A late opposite sign falls through to the complete class summary, potentially
running two summary phases. The probe never establishes a one-sign guarantee.
Small inputs already completed by the probe retain the direct class result.

## Contracts and accounting

Only private host planning changes production behavior. Public signatures, empty/
count/device admission, original clamp/source indices, `16*N` upload bytes,
four-byte nonempty construction synchronization and later explicit metadata
readback retain their contracts. No hidden GPU-to-CPU inspection is added.
GPU-key-buffer calls still sort all 24 digits and preserve borrowed inputs.
Every WGSL shader, radix scheduling/control code, pipeline layout/variant,
original preparation, scan, compaction, dependency and core API is unchanged.

For k actual digits the existing sorter performs `3*k` compute passes and actual
timestamp collection transfers `48*k` bytes; tests assert these counts. Nonzero
sorts retain four scratch requests and sizes: `16*N` ping-pong, `4*N` ranks and
two `64*ceil(N/256)` histograms/offsets. Zero-digit sorts return before radix
resources/submission; compaction remains. Batched controls still initialize all
24 aligned uniforms when supported; the existing fallback binds selected digits.
Descriptor/query counts do not measure physical memory, bandwidth, energy or
pipeline compilation counts. Those metrics and other devices are unmeasured.

## Validation and provenance

Four new native regressions cover independent class/nibble masks and stable CPU
LSD order for every signed bit width, probe/worker boundaries, forced
0/1/2/3/4/12/maximum worker availability, late-only opposite signs, classes absent
inside individual workers, disjoint high digits, saturation, ties and actual
6,8,...,22-digit masks. Complete parent/current public APIs and raw entries match
independent BTreeMap/stable CPU expectations. Original owned handles, profiles/
query bytes, default/tiny/zero pool budgets, unchanged GPU-key stages/input and
admission errors are checked. Concurrent results remain correct after host
mutation, pool clears and changed-input reuse.

Previous digit-family tests remain, extending their independent mask oracle to
sign classes and sharing helpers. No correctness test was removed. Frozen
native-only references reproduce the complete immediate-parent planner and
CPU/device-key API bodies, with only import paths changed; radix code is shared
because it is byte-identical to the parent.

- VOXEL_SIGN_DIGITS_OUTPUT verified 133099828 raw parent/current/CPU words across 74 complete fixture pairs; stable signed lexicographic order, saturation, ties, source ownership and 6..22 actual digits.

- VOXEL_SIGN_DIGITS_CONCURRENT verified independent mixed-sign/late-sign/saturated/full-width results under simultaneous builds, host mutation, pool clear and changed reuse.

- VOXEL_SIGN_DIGITS_RESOURCES verified three stages and 48 query bytes per digit, original owned output, default/tiny/zero pool budgets, four nonzero-sort requests and unchanged GPU-key passes/input.

- VOXEL_SIGN_DIGITS_PLAN verified 2888 independent mask/order/worker cases; every signed width, absent classes, late opposite signs, isolated worker signs, saturation, 6..22 actual digits and 48-byte summaries.

Final validation executes 460 native tests (targeted prechecks are not counted twice):

| Check | Passed tests |
| --- | ---: |
| clippy | 0 |
| image | 182 |
| voxel | 236 |
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
CI-feature GPU/image WASM compile checks, format and diff checks pass. Audit
confirms 613 original GPU/image sources unchanged and
626 source/manifest/executable SHA-256 hashes identical across
precheck, measurements and final validation. Published files and PR text contain
zero forbidden local adapter-name matches.

## Paired methods and limits

Two consecutive release surveys each execute 28 fixtures, three warmups and
30 measured parent/current pairs on four routes. Resident `build` includes
original CPU preparation, planning, upload, sort, compaction, completion and its
four-byte scalar synchronization. `full` also includes public host conversion
and explicit metadata D2H. Checks, metadata checking after a resident build and
output destruction are outside timers. Balanced XOR rotation covers all four
old/current build/full positions. GPU-stage timestamp and host preparation-plus-
planner diagnostics alternate old/current order; checks/destruction are excluded.
Both host variants include their own planner. Stage timing excludes upload,
planning and compaction and is diagnostic only.

Six fresh-runtime scenarios each run two surveys of 20 alternating pairs, no
warmups, with a new runtime/device per variant created outside timing. The timed
resident call includes its first construction. Checks/destruction are excluded.
No explicit driver-cache reset is requested: these are not process-cold or
hardware-cold guarantees. Runtime/device creation costs are unmeasured.

Ratios are paired geometric means parent/current (>1 favors this change), with
10,000 paired log bootstrap draws and percentile indices 249/9749 for individual
95% intervals. Warm seed=2026101831+1000*survey+10*case_index+route_index, with
1-based surveys and 0-based cases/routes, route order gpu/build/full/host.
Cold seed=2026101832+1000*survey+case_index. Seeds are listed below. Zero samples
make ratios undefined; no epsilon, dropped pair, outlier removal or selected rerun
is used. No multiple-comparison correction is applied. Medians are separate from
the paired estimate. All results, including adverse controls, are retained.

There is no overlap with this task's builds/tests or other measurement families.
Boundary process CPU deltas/GPU utilization cannot prove quiet conditions through
every pair. Shared-host diagnostics do not imply portable application speed or
competitor superiority.

| Route | Reduced-digit comparisons | Ratio range | Same-count controls | Ratio range |
| --- | --- | --- | --- | --- |
| gpu | 40 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1.049893..5.848124 | 0 favorable / 14 inconclusive / 0 adverse / 2 unresolved | 0.987860..1.014267 |
| build | 35 favorable / 5 inconclusive / 0 adverse / 0 unresolved | 0.993507..1.798141 | 0 favorable / 15 inconclusive / 1 adverse / 0 unresolved | 0.947121..1.031605 |
| full | 33 favorable / 7 inconclusive / 0 adverse / 0 unresolved | 0.985529..1.551217 | 0 favorable / 14 inconclusive / 2 adverse / 0 unresolved | 0.950743..1.024174 |
| host | 0 favorable / 0 inconclusive / 40 adverse / 0 unresolved | 0.498398..0.774300 | 0 favorable / 8 inconclusive / 8 adverse / 0 unresolved | 0.668664..1.059686 |

The adopted-source reduced-digit resident/full groups have no individually adverse intervals, but some remain inconclusive. Same-count controls are not a zero-overhead promise: every adverse complete-call result is retained below. The individual 95% intervals have no multiplicity correction; shared-host scheduling and unchanged GPU-stage controls do not establish a causal source for these differences.

- Same-count build, survey 1, N=65537, pattern 5: parent/current ratio 0.947121 (5.58% more time), interval 0.910904..0.984943.

- Same-count full, survey 1, N=1000003, pattern 6: parent/current ratio 0.957477 (4.44% more time), interval 0.910692..0.999489.

- Same-count full, survey 2, N=65537, pattern 8: parent/current ratio 0.950743 (5.18% more time), interval 0.909237..0.992277.

Reduced-digit host-only preparation/planning has 0 favorable / 0 inconclusive / 40 adverse / 0 unresolved; observed extra host time ranges 29.1%..100.6%. This cost is already inside resident/full timings, not an additional uncounted charge.

Fresh-runtime resident comparisons: 1 favorable / 10 inconclusive / 1 adverse / 0 unresolved. Reduced-digit scenarios: 1 favorable / 7 inconclusive / 0 adverse / 0 unresolved; same-count controls: 0 favorable / 3 inconclusive / 1 adverse / 0 unresolved. Most first-construction intervals remain inconclusive.

Fresh-runtime adverse control: survey 1, N=65537, pattern 6, 24/24 digits; ratio 0.937098 (6.71% more time), interval 0.870956..0.997236. Retained without attributing a cause from these shared-host observations.

| Survey | Method | Boundary quiet observed | Busy counts before/after | Max CPU cores before/after | GPU utilization samples before/after |
| ---: | --- | --- | --- | --- | --- |
| 1 | warm | False | [3, 2] | [0.989, 1.0] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 2 | warm | False | [3, 2] | [0.994, 0.987] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 1 | fresh-runtime | False | [2, 1] | [1.418, 0.982] | `[[[[0, 0]], [[0, 0]]], [[[2, 0]], [[0, 0]]]]` |
| 2 | fresh-runtime | False | [1, 1] | [0.994, 0.97] | `[[[[0, 0]], [[0, 0]]], [[[2, 0]], [[0, 0]]]]` |

Backend/capability labels, without adapter model names:

- warm 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=30 warmups=3 wall_order=balanced gpu_host_order=alternating checks_drop_excluded=true`

- warm 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=30 warmups=3 wall_order=balanced gpu_host_order=alternating checks_drop_excluded=true`

- fresh-runtime 1: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`

- fresh-runtime 2: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`

Patterns 0..5 target bounded/disjoint/late/worker-separated/saturated/holey signed coordinates; 6 is full-width random, 7 constant, 8 positive-only. Patterns 10..18 exercise every even 6..22 selected-digit count. Timed point counts cross the 256 probe and 524,288 worker thresholds through 1,000,003 points. Empty/singleton behavior is correctness-tested, not timed.

## Every warm comparison

| Survey | N | Pattern | Parent/new digits | Route | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 0 | 24/8 | gpu | 0.281232 | 0.095920 | 2.924056 | 2.895689..2.944546 | favorable | 2026102831 |
| 1 | 257 | 0 | 24/8 | build | 2.229650 | 1.680100 | 1.302772 | 1.242703..1.361150 | favorable | 2026102832 |
| 1 | 257 | 0 | 24/8 | full | 3.285600 | 2.860350 | 1.143810 | 1.095107..1.192518 | favorable | 2026102833 |
| 1 | 257 | 0 | 24/8 | host | 0.001450 | 0.002850 | 0.516371 | 0.421956..0.632191 | adverse | 2026102834 |
| 1 | 257 | 2 | 8/2 | gpu | 0.094048 | 0.023920 | 3.949945 | 3.900594..3.999526 | favorable | 2026102841 |
| 1 | 257 | 2 | 8/2 | build | 1.682650 | 1.515800 | 1.115167 | 1.062458..1.172588 | favorable | 2026102842 |
| 1 | 257 | 2 | 8/2 | full | 2.784700 | 2.599650 | 1.086932 | 1.052342..1.123684 | favorable | 2026102843 |
| 1 | 257 | 2 | 8/2 | host | 0.001850 | 0.003450 | 0.541129 | 0.459552..0.637019 | adverse | 2026102844 |
| 1 | 257 | 6 | 24/24 | gpu | 0.287328 | 0.286784 | 1.004253 | 0.998781..1.010068 | inconclusive | 2026102851 |
| 1 | 257 | 6 | 24/24 | build | 1.912650 | 1.916400 | 0.985564 | 0.959660..1.007878 | inconclusive | 2026102852 |
| 1 | 257 | 6 | 24/24 | full | 2.979300 | 2.877550 | 1.024174 | 0.999206..1.049783 | inconclusive | 2026102853 |
| 1 | 257 | 6 | 24/24 | host | 0.001400 | 0.001850 | 0.704659 | 0.567790..0.875776 | adverse | 2026102854 |
| 1 | 257 | 7 | 0/0 | gpu | -0.000000 | -0.000000 | undefined | undefined..undefined | unresolved | 2026102861 |
| 1 | 257 | 7 | 0/0 | build | 1.241750 | 1.202700 | 1.009790 | 0.943256..1.071984 | inconclusive | 2026102862 |
| 1 | 257 | 7 | 0/0 | full | 2.211450 | 2.285250 | 0.973937 | 0.931777..1.008625 | inconclusive | 2026102863 |
| 1 | 257 | 7 | 0/0 | host | 0.002050 | 0.002750 | 0.724459 | 0.614103..0.854577 | adverse | 2026102864 |
| 1 | 257 | 8 | 6/6 | gpu | 0.146432 | 0.145888 | 1.006325 | 0.990856..1.028567 | inconclusive | 2026102871 |
| 1 | 257 | 8 | 6/6 | build | 1.449200 | 1.461600 | 0.999575 | 0.950449..1.049695 | inconclusive | 2026102872 |
| 1 | 257 | 8 | 6/6 | full | 2.537450 | 2.628900 | 0.982854 | 0.949655..1.014358 | inconclusive | 2026102873 |
| 1 | 257 | 8 | 6/6 | host | 0.001950 | 0.002500 | 0.733390 | 0.613787..0.878423 | adverse | 2026102874 |
| 1 | 65537 | 0 | 24/8 | gpu | 2.120112 | 0.818944 | 2.588899 | 2.574692..2.603112 | favorable | 2026102881 |
| 1 | 65537 | 0 | 24/8 | build | 5.534000 | 3.574200 | 1.559820 | 1.516492..1.608379 | favorable | 2026102882 |
| 1 | 65537 | 0 | 24/8 | full | 7.541250 | 5.408350 | 1.369351 | 1.327595..1.411362 | favorable | 2026102883 |
| 1 | 65537 | 0 | 24/8 | host | 0.427000 | 0.688950 | 0.629482 | 0.595611..0.666351 | adverse | 2026102884 |
| 1 | 65537 | 1 | 22/10 | gpu | 2.094256 | 1.072688 | 1.953091 | 1.943538..1.962122 | favorable | 2026102891 |
| 1 | 65537 | 1 | 22/10 | build | 5.733400 | 3.922900 | 1.453284 | 1.401155..1.503573 | favorable | 2026102892 |
| 1 | 65537 | 1 | 22/10 | full | 8.093450 | 6.519250 | 1.241011 | 1.195484..1.285496 | favorable | 2026102893 |
| 1 | 65537 | 1 | 22/10 | host | 0.516250 | 0.708850 | 0.747465 | 0.715231..0.783065 | adverse | 2026102894 |
| 1 | 65537 | 3 | 18/6 | gpu | 1.631424 | 0.628176 | 2.600050 | 2.580701..2.619251 | favorable | 2026102901 |
| 1 | 65537 | 3 | 18/6 | build | 5.282500 | 3.498000 | 1.435809 | 1.299972..1.532208 | favorable | 2026102902 |
| 1 | 65537 | 3 | 18/6 | full | 7.379800 | 5.620750 | 1.266590 | 1.224977..1.309905 | favorable | 2026102903 |
| 1 | 65537 | 3 | 18/6 | host | 0.451750 | 0.701600 | 0.675848 | 0.650458..0.702975 | adverse | 2026102904 |
| 1 | 65537 | 4 | 24/4 | gpu | 1.998272 | 0.378240 | 5.293145 | 5.258593..5.328253 | favorable | 2026102911 |
| 1 | 65537 | 4 | 24/4 | build | 5.527800 | 3.171450 | 1.771463 | 1.703263..1.841117 | favorable | 2026102912 |
| 1 | 65537 | 4 | 24/4 | full | 7.320500 | 4.750400 | 1.551217 | 1.503155..1.602319 | favorable | 2026102913 |
| 1 | 65537 | 4 | 24/4 | host | 0.421750 | 0.693050 | 0.631778 | 0.602213..0.665094 | adverse | 2026102914 |
| 1 | 65537 | 5 | 10/10 | gpu | 0.981264 | 0.983440 | 1.001602 | 0.997137..1.005849 | inconclusive | 2026102921 |
| 1 | 65537 | 5 | 10/10 | build | 3.776100 | 4.041800 | 0.947121 | 0.910904..0.984943 | adverse | 2026102922 |
| 1 | 65537 | 5 | 10/10 | full | 6.337800 | 6.630950 | 0.975782 | 0.943433..1.009752 | inconclusive | 2026102923 |
| 1 | 65537 | 5 | 10/10 | host | 0.457350 | 0.599400 | 0.763216 | 0.735454..0.794220 | adverse | 2026102924 |
| 1 | 65537 | 6 | 24/24 | gpu | 1.292464 | 1.294576 | 0.999346 | 0.997954..1.000847 | inconclusive | 2026102931 |
| 1 | 65537 | 6 | 24/24 | build | 6.029250 | 5.983600 | 0.997528 | 0.967706..1.029574 | inconclusive | 2026102932 |
| 1 | 65537 | 6 | 24/24 | full | 8.488950 | 8.593050 | 0.981471 | 0.955004..1.008621 | inconclusive | 2026102933 |
| 1 | 65537 | 6 | 24/24 | host | 0.439250 | 0.428800 | 0.993895 | 0.941294..1.050636 | inconclusive | 2026102934 |
| 1 | 65537 | 8 | 6/6 | gpu | 0.245552 | 0.245536 | 0.998939 | 0.995393..1.002560 | inconclusive | 2026102941 |
| 1 | 65537 | 8 | 6/6 | build | 2.814500 | 2.856950 | 1.001933 | 0.948653..1.057123 | inconclusive | 2026102942 |
| 1 | 65537 | 8 | 6/6 | full | 4.586800 | 4.611800 | 0.971586 | 0.922979..1.016633 | inconclusive | 2026102943 |
| 1 | 65537 | 8 | 6/6 | host | 0.510950 | 0.495750 | 1.006522 | 0.947595..1.069540 | inconclusive | 2026102944 |
| 1 | 524287 | 0 | 24/8 | gpu | 4.904576 | 1.983872 | 2.446264 | 2.428966..2.463780 | favorable | 2026102951 |
| 1 | 524287 | 0 | 24/8 | build | 11.565700 | 10.008800 | 1.234929 | 1.166810..1.312371 | favorable | 2026102952 |
| 1 | 524287 | 0 | 24/8 | full | 14.854350 | 13.073300 | 1.128737 | 1.085817..1.173645 | favorable | 2026102953 |
| 1 | 524287 | 0 | 24/8 | host | 2.813650 | 4.543950 | 0.615345 | 0.579165..0.649439 | adverse | 2026102954 |
| 1 | 524288 | 0 | 24/8 | gpu | 5.835056 | 2.352048 | 2.486587 | 2.480370..2.494323 | favorable | 2026102961 |
| 1 | 524288 | 0 | 24/8 | build | 11.883350 | 8.649600 | 1.361400 | 1.337560..1.384750 | favorable | 2026102962 |
| 1 | 524288 | 0 | 24/8 | full | 15.351750 | 12.763650 | 1.210658 | 1.185393..1.234812 | favorable | 2026102963 |
| 1 | 524288 | 0 | 24/8 | host | 1.508550 | 2.628350 | 0.569006 | 0.554867..0.582991 | adverse | 2026102964 |
| 1 | 524289 | 0 | 24/8 | gpu | 5.941776 | 2.473856 | 2.403975 | 2.398132..2.409772 | favorable | 2026102971 |
| 1 | 524289 | 0 | 24/8 | build | 12.085650 | 9.108950 | 1.314724 | 1.289259..1.339011 | favorable | 2026102972 |
| 1 | 524289 | 0 | 24/8 | full | 16.343500 | 13.179200 | 1.220321 | 1.194250..1.246206 | favorable | 2026102973 |
| 1 | 524289 | 0 | 24/8 | host | 1.663250 | 2.755300 | 0.591208 | 0.573323..0.609645 | adverse | 2026102974 |
| 1 | 1000003 | 0 | 24/8 | gpu | 11.329136 | 4.716768 | 2.398161 | 2.391870..2.402032 | favorable | 2026102981 |
| 1 | 1000003 | 0 | 24/8 | build | 23.977050 | 18.084000 | 1.331427 | 1.312637..1.349284 | favorable | 2026102982 |
| 1 | 1000003 | 0 | 24/8 | full | 32.044500 | 26.070400 | 1.237267 | 1.202671..1.281827 | favorable | 2026102983 |
| 1 | 1000003 | 0 | 24/8 | host | 2.727800 | 4.691000 | 0.601010 | 0.585120..0.617531 | adverse | 2026102984 |
| 1 | 1000003 | 4 | 24/4 | gpu | 9.535136 | 1.638912 | 5.848124 | 5.819007..5.901010 | favorable | 2026102991 |
| 1 | 1000003 | 4 | 24/4 | build | 19.904100 | 12.810900 | 1.552327 | 1.522534..1.583745 | favorable | 2026102992 |
| 1 | 1000003 | 4 | 24/4 | full | 25.797350 | 19.163250 | 1.329490 | 1.293728..1.365332 | favorable | 2026102993 |
| 1 | 1000003 | 4 | 24/4 | host | 2.607850 | 4.400450 | 0.595253 | 0.582971..0.607527 | adverse | 2026102994 |
| 1 | 1000003 | 6 | 24/24 | gpu | 20.921328 | 20.929408 | 0.989887 | 0.976462..1.000615 | inconclusive | 2026103001 |
| 1 | 1000003 | 6 | 24/24 | build | 26.772400 | 26.894700 | 1.028043 | 0.960221..1.131736 | inconclusive | 2026103002 |
| 1 | 1000003 | 6 | 24/24 | full | 46.035350 | 46.642100 | 0.957477 | 0.910692..0.999489 | adverse | 2026103003 |
| 1 | 1000003 | 6 | 24/24 | host | 4.149700 | 4.228600 | 0.986141 | 0.922723..1.049532 | inconclusive | 2026103004 |
| 1 | 1000003 | 8 | 6/6 | gpu | 6.028272 | 6.026784 | 1.014044 | 0.997677..1.043958 | inconclusive | 2026103011 |
| 1 | 1000003 | 8 | 6/6 | build | 17.104750 | 17.066500 | 1.012756 | 0.951418..1.099097 | inconclusive | 2026103012 |
| 1 | 1000003 | 8 | 6/6 | full | 27.176800 | 27.498300 | 0.994367 | 0.951558..1.033022 | inconclusive | 2026103013 |
| 1 | 1000003 | 8 | 6/6 | host | 4.470050 | 4.461150 | 1.059686 | 0.997118..1.170982 | inconclusive | 2026103014 |
| 1 | 65537 | 10 | 24/6 | gpu | 4.996576 | 1.508784 | 3.309730 | 3.294173..3.325193 | favorable | 2026103021 |
| 1 | 65537 | 10 | 24/6 | build | 10.675050 | 9.298300 | 1.125141 | 1.034205..1.203687 | favorable | 2026103022 |
| 1 | 65537 | 10 | 24/6 | full | 18.420600 | 16.790200 | 1.099388 | 1.061124..1.139485 | favorable | 2026103023 |
| 1 | 65537 | 10 | 24/6 | host | 0.579450 | 0.897100 | 0.666108 | 0.629227..0.711101 | adverse | 2026103024 |
| 1 | 65537 | 11 | 24/8 | gpu | 5.139680 | 2.043968 | 2.512724 | 2.505210..2.520266 | favorable | 2026103031 |
| 1 | 65537 | 11 | 24/8 | build | 13.022150 | 10.327750 | 1.256181 | 1.202251..1.315683 | favorable | 2026103032 |
| 1 | 65537 | 11 | 24/8 | full | 21.231900 | 19.038900 | 1.117382 | 1.075281..1.157881 | favorable | 2026103033 |
| 1 | 65537 | 11 | 24/8 | host | 0.541500 | 0.801550 | 0.670034 | 0.642048..0.701265 | adverse | 2026103034 |
| 1 | 65537 | 12 | 24/10 | gpu | 5.067296 | 2.483136 | 2.035098 | 2.026869..2.043120 | favorable | 2026103041 |
| 1 | 65537 | 12 | 24/10 | build | 12.688300 | 9.504350 | 1.304943 | 1.247737..1.365579 | favorable | 2026103042 |
| 1 | 65537 | 12 | 24/10 | full | 20.625900 | 17.467200 | 1.184894 | 1.144260..1.225803 | favorable | 2026103043 |
| 1 | 65537 | 12 | 24/10 | host | 0.520400 | 0.776750 | 0.659987 | 0.634732..0.687037 | adverse | 2026103044 |
| 1 | 65537 | 13 | 24/12 | gpu | 4.920384 | 2.853584 | 1.730251 | 1.722685..1.737704 | favorable | 2026103051 |
| 1 | 65537 | 13 | 24/12 | build | 12.263550 | 9.114150 | 1.353740 | 1.307100..1.399960 | favorable | 2026103052 |
| 1 | 65537 | 13 | 24/12 | full | 20.287700 | 17.089000 | 1.191433 | 1.151426..1.234363 | favorable | 2026103053 |
| 1 | 65537 | 13 | 24/12 | host | 0.490450 | 0.745250 | 0.657595 | 0.631230..0.688320 | adverse | 2026103054 |
| 1 | 65537 | 14 | 24/14 | gpu | 5.161312 | 3.394224 | 1.535610 | 1.528378..1.542864 | favorable | 2026103061 |
| 1 | 65537 | 14 | 24/14 | build | 10.388650 | 9.337000 | 1.090949 | 1.030627..1.154505 | favorable | 2026103062 |
| 1 | 65537 | 14 | 24/14 | full | 18.029450 | 17.026100 | 1.053662 | 1.009596..1.098704 | favorable | 2026103063 |
| 1 | 65537 | 14 | 24/14 | host | 0.496850 | 0.780700 | 0.650808 | 0.620873..0.680618 | adverse | 2026103064 |
| 1 | 65537 | 15 | 24/16 | gpu | 2.385408 | 1.721808 | 1.380406 | 1.373638..1.386404 | favorable | 2026103071 |
| 1 | 65537 | 15 | 24/16 | build | 11.465400 | 9.717250 | 1.075266 | 1.007654..1.143500 | favorable | 2026103072 |
| 1 | 65537 | 15 | 24/16 | full | 19.061650 | 17.097950 | 1.061206 | 1.021727..1.103502 | favorable | 2026103073 |
| 1 | 65537 | 15 | 24/16 | host | 0.516750 | 0.793350 | 0.647797 | 0.618651..0.679701 | adverse | 2026103074 |
| 1 | 65537 | 16 | 24/18 | gpu | 5.355360 | 4.288960 | 1.248792 | 1.243788..1.253639 | favorable | 2026103081 |
| 1 | 65537 | 16 | 24/18 | build | 8.713600 | 8.605800 | 1.058349 | 1.005597..1.113670 | favorable | 2026103082 |
| 1 | 65537 | 16 | 24/18 | full | 16.202000 | 16.293550 | 1.016953 | 0.983092..1.051665 | inconclusive | 2026103083 |
| 1 | 65537 | 16 | 24/18 | host | 0.478700 | 0.761850 | 0.658656 | 0.629267..0.690257 | adverse | 2026103084 |
| 1 | 65537 | 17 | 24/20 | gpu | 5.467520 | 4.754208 | 1.149234 | 1.145014..1.153457 | favorable | 2026103091 |
| 1 | 65537 | 17 | 24/20 | build | 9.211900 | 9.105900 | 1.049604 | 0.993383..1.111142 | inconclusive | 2026103092 |
| 1 | 65537 | 17 | 24/20 | full | 16.285850 | 16.790100 | 1.003755 | 0.966513..1.039425 | inconclusive | 2026103093 |
| 1 | 65537 | 17 | 24/20 | host | 0.459800 | 0.710000 | 0.646691 | 0.625255..0.670212 | adverse | 2026103094 |
| 1 | 65537 | 18 | 24/22 | gpu | 2.595104 | 2.424896 | 1.049893 | 1.010031..1.071943 | favorable | 2026103101 |
| 1 | 65537 | 18 | 24/22 | build | 8.212300 | 8.306400 | 0.997358 | 0.944810..1.053518 | inconclusive | 2026103102 |
| 1 | 65537 | 18 | 24/22 | full | 15.115500 | 15.242450 | 0.986874 | 0.953860..1.020514 | inconclusive | 2026103103 |
| 1 | 65537 | 18 | 24/22 | host | 0.460900 | 0.662600 | 0.672877 | 0.639166..0.708045 | adverse | 2026103104 |
| 2 | 257 | 0 | 24/8 | gpu | 0.279456 | 0.096464 | 2.903440 | 2.888924..2.918905 | favorable | 2026103831 |
| 2 | 257 | 0 | 24/8 | build | 2.050700 | 1.560950 | 1.298622 | 1.260953..1.336370 | favorable | 2026103832 |
| 2 | 257 | 0 | 24/8 | full | 3.062850 | 2.614900 | 1.164028 | 1.133439..1.193086 | favorable | 2026103833 |
| 2 | 257 | 0 | 24/8 | host | 0.001450 | 0.002900 | 0.498398 | 0.405946..0.609221 | adverse | 2026103834 |
| 2 | 257 | 2 | 8/2 | gpu | 0.093552 | 0.023856 | 3.943288 | 3.905809..3.981768 | favorable | 2026103841 |
| 2 | 257 | 2 | 8/2 | build | 1.403900 | 1.275200 | 1.116946 | 1.087102..1.147809 | favorable | 2026103842 |
| 2 | 257 | 2 | 8/2 | full | 2.310050 | 2.209700 | 1.076941 | 1.042847..1.109861 | favorable | 2026103843 |
| 2 | 257 | 2 | 8/2 | host | 0.002000 | 0.003050 | 0.549335 | 0.461281..0.653499 | adverse | 2026103844 |
| 2 | 257 | 6 | 24/24 | gpu | 0.286496 | 0.286096 | 1.000935 | 0.996012..1.005895 | inconclusive | 2026103851 |
| 2 | 257 | 6 | 24/24 | build | 2.070950 | 2.003800 | 1.031605 | 0.994757..1.071018 | inconclusive | 2026103852 |
| 2 | 257 | 6 | 24/24 | full | 3.132400 | 3.071350 | 0.996753 | 0.969041..1.027700 | inconclusive | 2026103853 |
| 2 | 257 | 6 | 24/24 | host | 0.001450 | 0.001950 | 0.688823 | 0.556111..0.856460 | adverse | 2026103854 |
| 2 | 257 | 7 | 0/0 | gpu | -0.000000 | -0.000000 | undefined | undefined..undefined | unresolved | 2026103861 |
| 2 | 257 | 7 | 0/0 | build | 1.134150 | 1.149800 | 0.974654 | 0.911814..1.025216 | inconclusive | 2026103862 |
| 2 | 257 | 7 | 0/0 | full | 2.084450 | 2.074250 | 0.982728 | 0.950604..1.006306 | inconclusive | 2026103863 |
| 2 | 257 | 7 | 0/0 | host | 0.002050 | 0.002650 | 0.668664 | 0.517843..0.833153 | adverse | 2026103864 |
| 2 | 257 | 8 | 6/6 | gpu | 0.070736 | 0.070944 | 1.014267 | 0.997806..1.033982 | inconclusive | 2026103871 |
| 2 | 257 | 8 | 6/6 | build | 1.718450 | 1.638850 | 0.995321 | 0.951414..1.039440 | inconclusive | 2026103872 |
| 2 | 257 | 8 | 6/6 | full | 2.820050 | 2.880150 | 0.992222 | 0.961373..1.021957 | inconclusive | 2026103873 |
| 2 | 257 | 8 | 6/6 | host | 0.002600 | 0.003050 | 0.741700 | 0.614759..0.897729 | adverse | 2026103874 |
| 2 | 65537 | 0 | 24/8 | gpu | 1.942448 | 0.765728 | 2.544348 | 2.530096..2.558772 | favorable | 2026103881 |
| 2 | 65537 | 0 | 24/8 | build | 5.514100 | 3.462350 | 1.492089 | 1.399343..1.577639 | favorable | 2026103882 |
| 2 | 65537 | 0 | 24/8 | full | 7.203300 | 5.123600 | 1.364473 | 1.305610..1.426944 | favorable | 2026103883 |
| 2 | 65537 | 0 | 24/8 | host | 0.431450 | 0.638850 | 0.662580 | 0.628394..0.697862 | adverse | 2026103884 |
| 2 | 65537 | 1 | 22/10 | gpu | 1.943408 | 0.994288 | 1.955319 | 1.946298..1.964438 | favorable | 2026103891 |
| 2 | 65537 | 1 | 22/10 | build | 5.560150 | 4.005900 | 1.390413 | 1.338216..1.443763 | favorable | 2026103892 |
| 2 | 65537 | 1 | 22/10 | full | 7.910650 | 6.222150 | 1.258458 | 1.226903..1.290029 | favorable | 2026103893 |
| 2 | 65537 | 1 | 22/10 | host | 0.489050 | 0.646100 | 0.774300 | 0.739389..0.813583 | adverse | 2026103894 |
| 2 | 65537 | 3 | 18/6 | gpu | 1.586032 | 0.607488 | 2.622554 | 2.602256..2.643282 | favorable | 2026103901 |
| 2 | 65537 | 3 | 18/6 | build | 4.728200 | 3.234550 | 1.463965 | 1.405245..1.524458 | favorable | 2026103902 |
| 2 | 65537 | 3 | 18/6 | full | 6.661400 | 5.043100 | 1.296948 | 1.265245..1.328231 | favorable | 2026103903 |
| 2 | 65537 | 3 | 18/6 | host | 0.480150 | 0.667850 | 0.704175 | 0.672253..0.739103 | adverse | 2026103904 |
| 2 | 65537 | 4 | 24/4 | gpu | 2.003136 | 0.379312 | 5.264434 | 5.225833..5.303209 | favorable | 2026103911 |
| 2 | 65537 | 4 | 24/4 | build | 5.328800 | 2.975500 | 1.798141 | 1.720353..1.878485 | favorable | 2026103912 |
| 2 | 65537 | 4 | 24/4 | full | 7.303250 | 4.761200 | 1.495820 | 1.443301..1.551251 | favorable | 2026103913 |
| 2 | 65537 | 4 | 24/4 | host | 0.437650 | 0.617050 | 0.677051 | 0.641641..0.714888 | adverse | 2026103914 |
| 2 | 65537 | 5 | 10/10 | gpu | 1.033920 | 1.037088 | 0.987860 | 0.966323..1.000727 | inconclusive | 2026103921 |
| 2 | 65537 | 5 | 10/10 | build | 4.164300 | 4.200050 | 0.993531 | 0.962230..1.029105 | inconclusive | 2026103922 |
| 2 | 65537 | 5 | 10/10 | full | 6.589300 | 6.664600 | 1.016531 | 0.992551..1.043943 | inconclusive | 2026103923 |
| 2 | 65537 | 5 | 10/10 | host | 0.474250 | 0.615400 | 0.761685 | 0.729803..0.795789 | adverse | 2026103924 |
| 2 | 65537 | 6 | 24/24 | gpu | 1.926640 | 1.929216 | 1.001825 | 0.998997..1.005054 | inconclusive | 2026103931 |
| 2 | 65537 | 6 | 24/24 | build | 6.419900 | 6.310650 | 1.008780 | 0.976958..1.040999 | inconclusive | 2026103932 |
| 2 | 65537 | 6 | 24/24 | full | 8.893800 | 8.917100 | 0.999952 | 0.972285..1.027786 | inconclusive | 2026103933 |
| 2 | 65537 | 6 | 24/24 | host | 0.421100 | 0.419550 | 1.011753 | 0.944865..1.083708 | inconclusive | 2026103934 |
| 2 | 65537 | 8 | 6/6 | gpu | 0.253200 | 0.252800 | 1.002499 | 0.998920..1.005904 | inconclusive | 2026103941 |
| 2 | 65537 | 8 | 6/6 | build | 2.862350 | 2.949200 | 0.987279 | 0.937500..1.038778 | inconclusive | 2026103942 |
| 2 | 65537 | 8 | 6/6 | full | 4.446600 | 4.629600 | 0.950743 | 0.909237..0.992277 | adverse | 2026103943 |
| 2 | 65537 | 8 | 6/6 | host | 0.487950 | 0.494200 | 1.006086 | 0.949273..1.069529 | inconclusive | 2026103944 |
| 2 | 524287 | 0 | 24/8 | gpu | 6.299264 | 2.588016 | 2.430192 | 2.424747..2.435136 | favorable | 2026103951 |
| 2 | 524287 | 0 | 24/8 | build | 13.771500 | 11.487400 | 1.190976 | 1.154154..1.230919 | favorable | 2026103952 |
| 2 | 524287 | 0 | 24/8 | full | 17.094950 | 14.620700 | 1.146465 | 1.114429..1.180283 | favorable | 2026103953 |
| 2 | 524287 | 0 | 24/8 | host | 2.972650 | 5.086950 | 0.611923 | 0.588172..0.638163 | adverse | 2026103954 |
| 2 | 524288 | 0 | 24/8 | gpu | 6.113536 | 2.473664 | 2.471244 | 2.469541..2.472902 | favorable | 2026103961 |
| 2 | 524288 | 0 | 24/8 | build | 12.902950 | 8.972700 | 1.415720 | 1.391089..1.441181 | favorable | 2026103962 |
| 2 | 524288 | 0 | 24/8 | full | 16.731600 | 13.282650 | 1.269822 | 1.238401..1.300981 | favorable | 2026103963 |
| 2 | 524288 | 0 | 24/8 | host | 1.599700 | 2.745400 | 0.580868 | 0.565834..0.595853 | adverse | 2026103964 |
| 2 | 524289 | 0 | 24/8 | gpu | 6.614128 | 2.772704 | 2.386713 | 2.382911..2.390371 | favorable | 2026103971 |
| 2 | 524289 | 0 | 24/8 | build | 13.024700 | 9.410750 | 1.362136 | 1.332124..1.393981 | favorable | 2026103972 |
| 2 | 524289 | 0 | 24/8 | full | 17.694050 | 14.758950 | 1.221082 | 1.197410..1.244522 | favorable | 2026103973 |
| 2 | 524289 | 0 | 24/8 | host | 1.684000 | 2.807750 | 0.585397 | 0.571562..0.599797 | adverse | 2026103974 |
| 2 | 1000003 | 0 | 24/8 | gpu | 8.303872 | 3.380432 | 2.463283 | 2.456203..2.473278 | favorable | 2026103981 |
| 2 | 1000003 | 0 | 24/8 | build | 19.546650 | 14.533350 | 1.337152 | 1.317779..1.356359 | favorable | 2026103982 |
| 2 | 1000003 | 0 | 24/8 | full | 25.746200 | 20.764150 | 1.225323 | 1.200337..1.250558 | favorable | 2026103983 |
| 2 | 1000003 | 0 | 24/8 | host | 2.732550 | 4.388200 | 0.607296 | 0.594205..0.620462 | adverse | 2026103984 |
| 2 | 1000003 | 4 | 24/4 | gpu | 10.143584 | 1.733664 | 5.841102 | 5.827943..5.853196 | favorable | 2026103991 |
| 2 | 1000003 | 4 | 24/4 | build | 19.551800 | 13.205350 | 1.477785 | 1.450593..1.505728 | favorable | 2026103992 |
| 2 | 1000003 | 4 | 24/4 | full | 25.872500 | 19.759400 | 1.308422 | 1.283080..1.334579 | favorable | 2026103993 |
| 2 | 1000003 | 4 | 24/4 | host | 2.779500 | 4.582150 | 0.593233 | 0.579802..0.607275 | adverse | 2026103994 |
| 2 | 1000003 | 6 | 24/24 | gpu | 12.065712 | 12.175792 | 0.999940 | 0.997369..1.003252 | inconclusive | 2026104001 |
| 2 | 1000003 | 6 | 24/24 | build | 25.823850 | 25.904900 | 0.992440 | 0.980804..1.002901 | inconclusive | 2026104002 |
| 2 | 1000003 | 6 | 24/24 | full | 44.209350 | 44.689850 | 1.006250 | 0.985202..1.026965 | inconclusive | 2026104003 |
| 2 | 1000003 | 6 | 24/24 | host | 2.807800 | 2.757300 | 1.006391 | 0.981267..1.035508 | inconclusive | 2026104004 |
| 2 | 1000003 | 8 | 6/6 | gpu | 4.259552 | 4.309360 | 1.001704 | 0.996135..1.007474 | inconclusive | 2026104011 |
| 2 | 1000003 | 8 | 6/6 | build | 15.265750 | 15.187400 | 1.003880 | 0.970240..1.038883 | inconclusive | 2026104012 |
| 2 | 1000003 | 8 | 6/6 | full | 25.271500 | 24.260900 | 1.006169 | 0.972624..1.038325 | inconclusive | 2026104013 |
| 2 | 1000003 | 8 | 6/6 | host | 3.975900 | 3.967550 | 0.989030 | 0.960217..1.019987 | inconclusive | 2026104014 |
| 2 | 65537 | 10 | 24/6 | gpu | 4.950928 | 1.496560 | 3.270878 | 3.191723..3.318609 | favorable | 2026104021 |
| 2 | 65537 | 10 | 24/6 | build | 8.501650 | 7.945900 | 1.051392 | 1.016773..1.089209 | favorable | 2026104022 |
| 2 | 65537 | 10 | 24/6 | full | 15.417150 | 14.461650 | 1.056546 | 1.024250..1.093104 | favorable | 2026104023 |
| 2 | 65537 | 10 | 24/6 | host | 0.455150 | 0.652750 | 0.655782 | 0.630625..0.682406 | adverse | 2026104024 |
| 2 | 65537 | 11 | 24/8 | gpu | 4.946464 | 1.999760 | 2.475888 | 2.465648..2.485917 | favorable | 2026104031 |
| 2 | 65537 | 11 | 24/8 | build | 12.227200 | 9.466250 | 1.305720 | 1.223665..1.378270 | favorable | 2026104032 |
| 2 | 65537 | 11 | 24/8 | full | 20.337200 | 17.277450 | 1.168253 | 1.132443..1.204266 | favorable | 2026104033 |
| 2 | 65537 | 11 | 24/8 | host | 0.441250 | 0.729700 | 0.622546 | 0.587210..0.656141 | adverse | 2026104034 |
| 2 | 65537 | 12 | 24/10 | gpu | 4.870976 | 2.421936 | 2.014752 | 2.006506..2.023131 | favorable | 2026104041 |
| 2 | 65537 | 12 | 24/10 | build | 12.162350 | 8.728050 | 1.374758 | 1.315538..1.438008 | favorable | 2026104042 |
| 2 | 65537 | 12 | 24/10 | full | 19.991600 | 17.102750 | 1.163228 | 1.131868..1.194449 | favorable | 2026104043 |
| 2 | 65537 | 12 | 24/10 | host | 0.445950 | 0.705750 | 0.641152 | 0.614337..0.668035 | adverse | 2026104044 |
| 2 | 65537 | 13 | 24/12 | gpu | 2.336528 | 1.315568 | 1.772489 | 1.765854..1.779156 | favorable | 2026104051 |
| 2 | 65537 | 13 | 24/12 | build | 11.758850 | 8.685150 | 1.296485 | 1.244397..1.348040 | favorable | 2026104052 |
| 2 | 65537 | 13 | 24/12 | full | 19.116550 | 16.838400 | 1.110452 | 1.072964..1.148227 | favorable | 2026104053 |
| 2 | 65537 | 13 | 24/12 | host | 0.428300 | 0.672900 | 0.656422 | 0.628070..0.684378 | adverse | 2026104054 |
| 2 | 65537 | 14 | 24/14 | gpu | 2.384128 | 1.541776 | 1.587608 | 1.543647..1.675097 | favorable | 2026104061 |
| 2 | 65537 | 14 | 24/14 | build | 9.867400 | 8.968450 | 1.117034 | 1.048035..1.189744 | favorable | 2026104062 |
| 2 | 65537 | 14 | 24/14 | full | 17.696950 | 15.868300 | 1.081640 | 1.040151..1.123258 | favorable | 2026104063 |
| 2 | 65537 | 14 | 24/14 | host | 0.456350 | 0.706250 | 0.649224 | 0.623949..0.676652 | adverse | 2026104064 |
| 2 | 65537 | 15 | 24/16 | gpu | 5.211152 | 3.828848 | 1.408974 | 1.370446..1.484470 | favorable | 2026104071 |
| 2 | 65537 | 15 | 24/16 | build | 8.757600 | 8.638800 | 1.054635 | 1.002468..1.107226 | favorable | 2026104072 |
| 2 | 65537 | 15 | 24/16 | full | 15.666100 | 15.642450 | 1.030851 | 0.993393..1.068128 | inconclusive | 2026104073 |
| 2 | 65537 | 15 | 24/16 | host | 0.473500 | 0.630400 | 0.676408 | 0.646612..0.710209 | adverse | 2026104074 |
| 2 | 65537 | 16 | 24/18 | gpu | 2.504592 | 1.980624 | 1.286986 | 1.250092..1.357528 | favorable | 2026104081 |
| 2 | 65537 | 16 | 24/18 | build | 8.159950 | 8.423800 | 0.993507 | 0.949061..1.039284 | inconclusive | 2026104082 |
| 2 | 65537 | 16 | 24/18 | full | 14.475750 | 15.290150 | 1.002309 | 0.969077..1.036371 | inconclusive | 2026104083 |
| 2 | 65537 | 16 | 24/18 | host | 0.402250 | 0.605650 | 0.649665 | 0.627651..0.673258 | adverse | 2026104084 |
| 2 | 65537 | 17 | 24/20 | gpu | 2.541792 | 2.199776 | 1.156128 | 1.152659..1.159484 | favorable | 2026104091 |
| 2 | 65537 | 17 | 24/20 | build | 7.856500 | 8.251650 | 0.994711 | 0.931031..1.058262 | inconclusive | 2026104092 |
| 2 | 65537 | 17 | 24/20 | full | 13.977550 | 14.396450 | 0.985529 | 0.962342..1.011223 | inconclusive | 2026104093 |
| 2 | 65537 | 17 | 24/20 | host | 0.393600 | 0.611350 | 0.641733 | 0.620447..0.662969 | adverse | 2026104094 |
| 2 | 65537 | 18 | 24/22 | gpu | 2.523936 | 2.360448 | 1.126797 | 1.067059..1.221375 | favorable | 2026104101 |
| 2 | 65537 | 18 | 24/22 | build | 8.121850 | 8.145350 | 1.000946 | 0.951258..1.053666 | inconclusive | 2026104102 |
| 2 | 65537 | 18 | 24/22 | full | 14.684750 | 14.355100 | 1.011264 | 0.979209..1.044630 | inconclusive | 2026104103 |
| 2 | 65537 | 18 | 24/22 | host | 0.415450 | 0.638250 | 0.655083 | 0.629933..0.683685 | adverse | 2026104104 |

## Every fresh-runtime resident comparison

| Survey | N | Pattern | Parent/new digits | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 0 | 24/8 | 25.257450 | 22.969700 | 1.068743 | 1.021195..1.120190 | favorable | 2026102832 |
| 1 | 65537 | 0 | 24/8 | 24.545550 | 22.049550 | 1.088183 | 0.990634..1.195156 | inconclusive | 2026102833 |
| 1 | 65537 | 4 | 24/4 | 22.537600 | 21.629000 | 1.091697 | 0.958713..1.266613 | inconclusive | 2026102834 |
| 1 | 65537 | 6 | 24/24 | 23.772950 | 25.393650 | 0.937098 | 0.870956..0.997236 | adverse | 2026102835 |
| 1 | 65537 | 8 | 6/6 | 23.109350 | 22.493500 | 0.920709 | 0.771969..1.061267 | inconclusive | 2026102836 |
| 1 | 524289 | 0 | 24/8 | 30.466750 | 29.176100 | 1.067526 | 0.997215..1.139512 | inconclusive | 2026102837 |
| 2 | 257 | 0 | 24/8 | 24.099400 | 22.466900 | 1.072575 | 0.984261..1.166747 | inconclusive | 2026103832 |
| 2 | 65537 | 0 | 24/8 | 25.273750 | 24.818550 | 1.032330 | 0.957760..1.109282 | inconclusive | 2026103833 |
| 2 | 65537 | 4 | 24/4 | 26.516000 | 25.763850 | 1.030126 | 0.945844..1.129466 | inconclusive | 2026103834 |
| 2 | 65537 | 6 | 24/24 | 23.023450 | 24.361950 | 0.930935 | 0.817392..1.048089 | inconclusive | 2026103835 |
| 2 | 65537 | 8 | 6/6 | 23.811300 | 21.829150 | 1.042936 | 0.980781..1.106629 | inconclusive | 2026103836 |
| 2 | 524289 | 0 | 24/8 | 30.726450 | 28.967100 | 1.053917 | 0.962372..1.151542 | inconclusive | 2026103837 |

## Rejected unconditional class-summary candidate

The initial candidate used the class scan even for one-sign inputs. It passed the same four regressions, but its first complete 224-row survey set exposed resident/full one-sign regressions. The final source adds the proven global one-sign path above; the entire unchanged 28-case matrix is then repeated twice, not just favorable cases. All original results remain below, clearly separate from the delivered-source measurements. Both versions use the same frozen immediate parent. The initial source snapshot, executable/source hash receipt and raw logs are retained in ignored local evidence. Initial results do not describe the delivered source, and reused bootstrap seeds do not establish independence or a direct before/after comparison between candidates.

| Route | Reduced-digit comparisons | Same-count controls |
| --- | --- | --- |
| gpu | 40 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1 favorable / 13 inconclusive / 0 adverse / 2 unresolved |
| build | 29 favorable / 11 inconclusive / 0 adverse / 0 unresolved | 1 favorable / 13 inconclusive / 2 adverse / 0 unresolved |
| full | 27 favorable / 12 inconclusive / 1 adverse / 0 unresolved | 0 favorable / 15 inconclusive / 1 adverse / 0 unresolved |
| host | 0 favorable / 0 inconclusive / 40 adverse / 0 unresolved | 1 favorable / 5 inconclusive / 10 adverse / 0 unresolved |

| Survey | N | Pattern | Parent/new digits | Route | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 257 | 0 | 24/8 | gpu | 0.282368 | 0.096784 | 2.923627 | 2.906353..2.940807 | favorable | 2026102831 |
| 1 | 257 | 0 | 24/8 | build | 2.011700 | 1.505050 | 1.172230 | 0.960998..1.336235 | inconclusive | 2026102832 |
| 1 | 257 | 0 | 24/8 | full | 3.101250 | 2.512450 | 1.198886 | 1.065279..1.314676 | favorable | 2026102833 |
| 1 | 257 | 0 | 24/8 | host | 0.001650 | 0.002900 | 0.511559 | 0.438576..0.597399 | adverse | 2026102834 |
| 1 | 257 | 2 | 8/2 | gpu | 0.095888 | 0.023888 | 4.022338 | 3.977460..4.069285 | favorable | 2026102841 |
| 1 | 257 | 2 | 8/2 | build | 1.454750 | 1.259450 | 1.147527 | 1.117675..1.181720 | favorable | 2026102842 |
| 1 | 257 | 2 | 8/2 | full | 2.499900 | 2.293350 | 1.075109 | 1.042616..1.116377 | favorable | 2026102843 |
| 1 | 257 | 2 | 8/2 | host | 0.001550 | 0.002800 | 0.555354 | 0.491915..0.625913 | adverse | 2026102844 |
| 1 | 257 | 6 | 24/24 | gpu | 0.288128 | 0.287360 | 1.003037 | 0.999710..1.006557 | inconclusive | 2026102851 |
| 1 | 257 | 6 | 24/24 | build | 2.064850 | 2.136050 | 1.027694 | 0.930119..1.133687 | inconclusive | 2026102852 |
| 1 | 257 | 6 | 24/24 | full | 3.136850 | 3.181200 | 0.996736 | 0.928390..1.075022 | inconclusive | 2026102853 |
| 1 | 257 | 6 | 24/24 | host | 0.001650 | 0.002000 | 0.794025 | 0.653118..0.968024 | adverse | 2026102854 |
| 1 | 257 | 7 | 0/0 | gpu | -0.000000 | -0.000000 | undefined | undefined..undefined | unresolved | 2026102861 |
| 1 | 257 | 7 | 0/0 | build | 1.177350 | 1.169200 | 0.983118 | 0.932523..1.022529 | inconclusive | 2026102862 |
| 1 | 257 | 7 | 0/0 | full | 2.125350 | 2.162050 | 0.986493 | 0.964890..1.009378 | inconclusive | 2026102863 |
| 1 | 257 | 7 | 0/0 | host | 0.002050 | 0.003500 | 0.592909 | 0.503513..0.717074 | adverse | 2026102864 |
| 1 | 257 | 8 | 6/6 | gpu | 0.166704 | 0.166688 | 1.002453 | 0.996192..1.009124 | inconclusive | 2026102871 |
| 1 | 257 | 8 | 6/6 | build | 1.717400 | 1.693000 | 0.895320 | 0.739573..1.028484 | inconclusive | 2026102872 |
| 1 | 257 | 8 | 6/6 | full | 3.147300 | 3.048900 | 1.041430 | 0.987661..1.108299 | inconclusive | 2026102873 |
| 1 | 257 | 8 | 6/6 | host | 0.002150 | 0.003750 | 0.568716 | 0.498202..0.648817 | adverse | 2026102874 |
| 1 | 65537 | 0 | 24/8 | gpu | 2.133040 | 0.824272 | 2.587731 | 2.575555..2.599170 | favorable | 2026102881 |
| 1 | 65537 | 0 | 24/8 | build | 5.982600 | 3.931300 | 1.580424 | 1.495664..1.691539 | favorable | 2026102882 |
| 1 | 65537 | 0 | 24/8 | full | 8.383300 | 6.164400 | 1.360361 | 1.291635..1.430026 | favorable | 2026102883 |
| 1 | 65537 | 0 | 24/8 | host | 0.424700 | 0.690900 | 0.642800 | 0.616872..0.668400 | adverse | 2026102884 |
| 1 | 65537 | 1 | 22/10 | gpu | 1.969936 | 1.004800 | 1.954329 | 1.942359..1.966116 | favorable | 2026102891 |
| 1 | 65537 | 1 | 22/10 | build | 5.500350 | 4.014700 | 1.428332 | 1.368540..1.496571 | favorable | 2026102892 |
| 1 | 65537 | 1 | 22/10 | full | 8.427050 | 6.703400 | 1.309521 | 1.224044..1.415611 | favorable | 2026102893 |
| 1 | 65537 | 1 | 22/10 | host | 0.516400 | 0.690500 | 0.753818 | 0.736470..0.772623 | adverse | 2026102894 |
| 1 | 65537 | 3 | 18/6 | gpu | 1.691104 | 0.656288 | 2.576715 | 2.560291..2.593504 | favorable | 2026102901 |
| 1 | 65537 | 3 | 18/6 | build | 5.357150 | 3.555300 | 1.494927 | 1.439871..1.552715 | favorable | 2026102902 |
| 1 | 65537 | 3 | 18/6 | full | 7.227500 | 5.776050 | 1.321432 | 1.216151..1.467672 | favorable | 2026102903 |
| 1 | 65537 | 3 | 18/6 | host | 0.487700 | 0.620300 | 0.768412 | 0.734861..0.813275 | adverse | 2026102904 |
| 1 | 65537 | 4 | 24/4 | gpu | 2.041824 | 0.386496 | 5.261407 | 5.211849..5.308376 | favorable | 2026102911 |
| 1 | 65537 | 4 | 24/4 | build | 5.631600 | 3.032750 | 1.886605 | 1.799232..1.965817 | favorable | 2026102912 |
| 1 | 65537 | 4 | 24/4 | full | 7.523550 | 4.891950 | 1.644091 | 1.508261..1.876712 | favorable | 2026102913 |
| 1 | 65537 | 4 | 24/4 | host | 0.392800 | 0.647000 | 0.612022 | 0.596024..0.628829 | adverse | 2026102914 |
| 1 | 65537 | 5 | 10/10 | gpu | 2.499056 | 2.501424 | 1.000848 | 0.994788..1.006707 | inconclusive | 2026102921 |
| 1 | 65537 | 5 | 10/10 | build | 4.233100 | 4.451450 | 1.059701 | 0.970130..1.198066 | inconclusive | 2026102922 |
| 1 | 65537 | 5 | 10/10 | full | 7.476600 | 7.428900 | 0.937504 | 0.837891..1.009793 | inconclusive | 2026102923 |
| 1 | 65537 | 5 | 10/10 | host | 0.731950 | 0.982150 | 0.958449 | 0.674519..1.457407 | inconclusive | 2026102924 |
| 1 | 65537 | 6 | 24/24 | gpu | 1.240400 | 1.240512 | 1.000178 | 0.998609..1.001712 | inconclusive | 2026102931 |
| 1 | 65537 | 6 | 24/24 | build | 6.208700 | 6.271900 | 1.030772 | 0.940173..1.185975 | inconclusive | 2026102932 |
| 1 | 65537 | 6 | 24/24 | full | 8.723950 | 9.037500 | 1.013373 | 0.969033..1.070912 | inconclusive | 2026102933 |
| 1 | 65537 | 6 | 24/24 | host | 0.386150 | 0.403650 | 0.972774 | 0.924681..1.025690 | inconclusive | 2026102934 |
| 1 | 65537 | 8 | 6/6 | gpu | 0.236160 | 0.236800 | 0.996391 | 0.992384..1.000131 | inconclusive | 2026102941 |
| 1 | 65537 | 8 | 6/6 | build | 2.997200 | 2.879550 | 1.019216 | 0.966024..1.084505 | inconclusive | 2026102942 |
| 1 | 65537 | 8 | 6/6 | full | 4.689250 | 4.811350 | 0.968076 | 0.929064..1.006815 | inconclusive | 2026102943 |
| 1 | 65537 | 8 | 6/6 | host | 0.473850 | 0.628600 | 0.764861 | 0.750673..0.780793 | adverse | 2026102944 |
| 1 | 524287 | 0 | 24/8 | gpu | 5.573824 | 2.241184 | 2.462560 | 2.446796..2.475276 | favorable | 2026102951 |
| 1 | 524287 | 0 | 24/8 | build | 11.972100 | 10.431700 | 1.117943 | 1.001192..1.197017 | favorable | 2026102952 |
| 1 | 524287 | 0 | 24/8 | full | 15.705850 | 14.153100 | 1.077243 | 0.999480..1.155613 | inconclusive | 2026102953 |
| 1 | 524287 | 0 | 24/8 | host | 2.688300 | 4.362700 | 0.621018 | 0.591722..0.650088 | adverse | 2026102954 |
| 1 | 524288 | 0 | 24/8 | gpu | 5.740096 | 2.301552 | 2.494871 | 2.491889..2.497802 | favorable | 2026102961 |
| 1 | 524288 | 0 | 24/8 | build | 12.216500 | 8.651250 | 1.272553 | 1.100592..1.411783 | favorable | 2026102962 |
| 1 | 524288 | 0 | 24/8 | full | 15.574900 | 12.383350 | 1.309890 | 1.230501..1.424848 | favorable | 2026102963 |
| 1 | 524288 | 0 | 24/8 | host | 1.459350 | 2.420900 | 0.586447 | 0.567923..0.606088 | adverse | 2026102964 |
| 1 | 524289 | 0 | 24/8 | gpu | 6.242416 | 2.606976 | 2.391807 | 2.388266..2.394542 | favorable | 2026102971 |
| 1 | 524289 | 0 | 24/8 | build | 12.225200 | 9.023200 | 1.323522 | 1.271032..1.369987 | favorable | 2026102972 |
| 1 | 524289 | 0 | 24/8 | full | 16.585400 | 13.171050 | 1.151417 | 1.013736..1.261247 | favorable | 2026102973 |
| 1 | 524289 | 0 | 24/8 | host | 1.548250 | 2.596400 | 0.578592 | 0.520047..0.622307 | adverse | 2026102974 |
| 1 | 1000003 | 0 | 24/8 | gpu | 9.257632 | 3.701120 | 2.494947 | 2.490304..2.499383 | favorable | 2026102981 |
| 1 | 1000003 | 0 | 24/8 | build | 20.330350 | 15.143050 | 1.207363 | 1.032237..1.352196 | favorable | 2026102982 |
| 1 | 1000003 | 0 | 24/8 | full | 26.365200 | 21.949400 | 1.200126 | 1.169989..1.229296 | favorable | 2026102983 |
| 1 | 1000003 | 0 | 24/8 | host | 2.496550 | 4.092350 | 0.605360 | 0.590459..0.620811 | adverse | 2026102984 |
| 1 | 1000003 | 4 | 24/4 | gpu | 9.008256 | 1.529264 | 5.884169 | 5.871289..5.897855 | favorable | 2026102991 |
| 1 | 1000003 | 4 | 24/4 | build | 19.224950 | 12.970400 | 1.536306 | 1.446427..1.655702 | favorable | 2026102992 |
| 1 | 1000003 | 4 | 24/4 | full | 25.015400 | 18.312350 | 1.352043 | 1.309817..1.397520 | favorable | 2026102993 |
| 1 | 1000003 | 4 | 24/4 | host | 2.485500 | 4.034600 | 0.570804 | 0.498198..0.633241 | adverse | 2026102994 |
| 1 | 1000003 | 6 | 24/24 | gpu | 9.976704 | 9.976848 | 1.000680 | 0.998833..1.002859 | inconclusive | 2026103001 |
| 1 | 1000003 | 6 | 24/24 | build | 21.296050 | 21.579950 | 1.007269 | 0.975717..1.042318 | inconclusive | 2026103002 |
| 1 | 1000003 | 6 | 24/24 | full | 38.226150 | 37.861600 | 1.033950 | 0.982478..1.104893 | inconclusive | 2026103003 |
| 1 | 1000003 | 6 | 24/24 | host | 2.669950 | 2.589600 | 1.037834 | 1.016654..1.063055 | favorable | 2026103004 |
| 1 | 1000003 | 8 | 6/6 | gpu | 4.051680 | 4.059280 | 1.000846 | 0.995082..1.006963 | inconclusive | 2026103011 |
| 1 | 1000003 | 8 | 6/6 | build | 14.407500 | 15.149850 | 0.935051 | 0.899431..0.971472 | adverse | 2026103012 |
| 1 | 1000003 | 8 | 6/6 | full | 22.908650 | 23.451250 | 0.977989 | 0.957407..1.002253 | inconclusive | 2026103013 |
| 1 | 1000003 | 8 | 6/6 | host | 4.048550 | 4.791400 | 0.912041 | 0.841625..1.016278 | inconclusive | 2026103014 |
| 1 | 65537 | 10 | 24/6 | gpu | 4.867376 | 1.557680 | 3.136550 | 3.114863..3.160305 | favorable | 2026103021 |
| 1 | 65537 | 10 | 24/6 | build | 8.658700 | 8.571050 | 1.053159 | 0.991585..1.109594 | inconclusive | 2026103022 |
| 1 | 65537 | 10 | 24/6 | full | 15.374200 | 15.056850 | 1.030587 | 0.989086..1.075702 | inconclusive | 2026103023 |
| 1 | 65537 | 10 | 24/6 | host | 0.476350 | 0.740800 | 0.659753 | 0.616664..0.709338 | adverse | 2026103024 |
| 1 | 65537 | 11 | 24/8 | gpu | 4.618336 | 1.956256 | 2.398249 | 2.353701..2.477748 | favorable | 2026103031 |
| 1 | 65537 | 11 | 24/8 | build | 11.755500 | 8.724550 | 1.363158 | 1.306734..1.418632 | favorable | 2026103032 |
| 1 | 65537 | 11 | 24/8 | full | 19.115950 | 15.885900 | 1.181971 | 1.126633..1.229799 | favorable | 2026103033 |
| 1 | 65537 | 11 | 24/8 | host | 0.446050 | 0.683350 | 0.622938 | 0.604404..0.640527 | adverse | 2026103034 |
| 1 | 65537 | 12 | 24/10 | gpu | 4.710464 | 2.431344 | 1.939662 | 1.928648..1.950086 | favorable | 2026103041 |
| 1 | 65537 | 12 | 24/10 | build | 11.858100 | 8.462800 | 1.331380 | 1.272618..1.392327 | favorable | 2026103042 |
| 1 | 65537 | 12 | 24/10 | full | 19.860250 | 16.793750 | 1.203704 | 1.140206..1.285217 | favorable | 2026103043 |
| 1 | 65537 | 12 | 24/10 | host | 0.425300 | 0.694350 | 0.636951 | 0.622248..0.653351 | adverse | 2026103044 |
| 1 | 65537 | 13 | 24/12 | gpu | 4.964016 | 2.948256 | 1.684286 | 1.677200..1.691799 | favorable | 2026103051 |
| 1 | 65537 | 13 | 24/12 | build | 9.171650 | 8.244150 | 1.126133 | 1.059811..1.197962 | favorable | 2026103052 |
| 1 | 65537 | 13 | 24/12 | full | 16.239700 | 16.011150 | 1.026008 | 0.988446..1.064889 | inconclusive | 2026103053 |
| 1 | 65537 | 13 | 24/12 | host | 0.391500 | 0.644000 | 0.619361 | 0.595712..0.643825 | adverse | 2026103054 |
| 1 | 65537 | 14 | 24/14 | gpu | 4.987952 | 3.383056 | 1.524375 | 1.477922..1.612211 | favorable | 2026103061 |
| 1 | 65537 | 14 | 24/14 | build | 8.285650 | 8.060200 | 1.056993 | 1.004362..1.116202 | favorable | 2026103062 |
| 1 | 65537 | 14 | 24/14 | full | 15.322550 | 15.116100 | 1.031451 | 0.995840..1.069053 | inconclusive | 2026103063 |
| 1 | 65537 | 14 | 24/14 | host | 0.380050 | 0.610450 | 0.627313 | 0.612823..0.642024 | adverse | 2026103064 |
| 1 | 65537 | 15 | 24/16 | gpu | 5.111520 | 3.803856 | 1.341015 | 1.335694..1.346314 | favorable | 2026103071 |
| 1 | 65537 | 15 | 24/16 | build | 8.442350 | 8.246300 | 1.061181 | 0.997195..1.133436 | inconclusive | 2026103072 |
| 1 | 65537 | 15 | 24/16 | full | 14.852100 | 15.039250 | 0.998987 | 0.971813..1.027483 | inconclusive | 2026103073 |
| 1 | 65537 | 15 | 24/16 | host | 0.410400 | 0.668550 | 0.623960 | 0.605277..0.641401 | adverse | 2026103074 |
| 1 | 65537 | 16 | 24/18 | gpu | 2.455584 | 3.091968 | 1.212737 | 1.147285..1.249903 | favorable | 2026103081 |
| 1 | 65537 | 16 | 24/18 | build | 7.846750 | 8.043700 | 0.985589 | 0.942148..1.032384 | inconclusive | 2026103082 |
| 1 | 65537 | 16 | 24/18 | full | 13.908800 | 14.406500 | 0.955168 | 0.917474..0.987899 | adverse | 2026103083 |
| 1 | 65537 | 16 | 24/18 | host | 0.383350 | 0.593450 | 0.636349 | 0.615392..0.657332 | adverse | 2026103084 |
| 1 | 65537 | 17 | 24/20 | gpu | 2.542464 | 2.209984 | 1.151707 | 1.148640..1.154968 | favorable | 2026103091 |
| 1 | 65537 | 17 | 24/20 | build | 7.723050 | 8.148800 | 0.994234 | 0.940686..1.049275 | inconclusive | 2026103092 |
| 1 | 65537 | 17 | 24/20 | full | 14.526450 | 14.656800 | 1.005845 | 0.971156..1.041993 | inconclusive | 2026103093 |
| 1 | 65537 | 17 | 24/20 | host | 0.403800 | 0.582650 | 0.659376 | 0.636339..0.687583 | adverse | 2026103094 |
| 1 | 65537 | 18 | 24/22 | gpu | 2.602912 | 2.425872 | 1.098265 | 1.067396..1.159536 | favorable | 2026103101 |
| 1 | 65537 | 18 | 24/22 | build | 7.581250 | 8.057950 | 0.979941 | 0.924752..1.043799 | inconclusive | 2026103102 |
| 1 | 65537 | 18 | 24/22 | full | 14.612050 | 15.043650 | 1.004802 | 0.971993..1.039541 | inconclusive | 2026103103 |
| 1 | 65537 | 18 | 24/22 | host | 0.374450 | 0.582950 | 0.635082 | 0.610442..0.658995 | adverse | 2026103104 |
| 2 | 257 | 0 | 24/8 | gpu | 0.280080 | 0.096416 | 2.914538 | 2.894573..2.934624 | favorable | 2026103831 |
| 2 | 257 | 0 | 24/8 | build | 2.058100 | 1.568200 | 1.318466 | 1.269074..1.379467 | favorable | 2026103832 |
| 2 | 257 | 0 | 24/8 | full | 3.073900 | 2.619300 | 1.187509 | 1.149050..1.231558 | favorable | 2026103833 |
| 2 | 257 | 0 | 24/8 | host | 0.002400 | 0.003850 | 0.540303 | 0.453827..0.643018 | adverse | 2026103834 |
| 2 | 257 | 2 | 8/2 | gpu | 0.095424 | 0.024112 | 3.951323 | 3.907963..3.996767 | favorable | 2026103841 |
| 2 | 257 | 2 | 8/2 | build | 1.655500 | 1.451100 | 1.166919 | 1.120650..1.217341 | favorable | 2026103842 |
| 2 | 257 | 2 | 8/2 | full | 2.849050 | 2.550000 | 1.160754 | 1.115700..1.214855 | favorable | 2026103843 |
| 2 | 257 | 2 | 8/2 | host | 0.002150 | 0.003300 | 0.623327 | 0.517563..0.772512 | adverse | 2026103844 |
| 2 | 257 | 6 | 24/24 | gpu | 0.285632 | 0.286768 | 0.998566 | 0.993414..1.003597 | inconclusive | 2026103851 |
| 2 | 257 | 6 | 24/24 | build | 2.093650 | 2.030250 | 1.022714 | 0.979338..1.071029 | inconclusive | 2026103852 |
| 2 | 257 | 6 | 24/24 | full | 3.204500 | 3.186100 | 0.980467 | 0.937986..1.021667 | inconclusive | 2026103853 |
| 2 | 257 | 6 | 24/24 | host | 0.001850 | 0.002100 | 0.809710 | 0.649311..1.010648 | inconclusive | 2026103854 |
| 2 | 257 | 7 | 0/0 | gpu | -0.000000 | -0.000000 | undefined | undefined..undefined | unresolved | 2026103861 |
| 2 | 257 | 7 | 0/0 | build | 1.231300 | 1.234100 | 0.939970 | 0.846228..1.022192 | inconclusive | 2026103862 |
| 2 | 257 | 7 | 0/0 | full | 2.306500 | 2.238900 | 1.018178 | 0.990219..1.047430 | inconclusive | 2026103863 |
| 2 | 257 | 7 | 0/0 | host | 0.002300 | 0.003750 | 0.566000 | 0.489089..0.652324 | adverse | 2026103864 |
| 2 | 257 | 8 | 6/6 | gpu | 0.159136 | 0.158224 | 1.006106 | 1.000764..1.011322 | favorable | 2026103871 |
| 2 | 257 | 8 | 6/6 | build | 1.612150 | 1.564850 | 1.010859 | 0.968726..1.055152 | inconclusive | 2026103872 |
| 2 | 257 | 8 | 6/6 | full | 2.641950 | 2.729400 | 0.999179 | 0.968947..1.031209 | inconclusive | 2026103873 |
| 2 | 257 | 8 | 6/6 | host | 0.002300 | 0.003550 | 0.604799 | 0.499895..0.735973 | adverse | 2026103874 |
| 2 | 65537 | 0 | 24/8 | gpu | 2.158880 | 0.842640 | 2.564838 | 2.552147..2.577603 | favorable | 2026103881 |
| 2 | 65537 | 0 | 24/8 | build | 5.907150 | 3.736050 | 1.569228 | 1.527244..1.610320 | favorable | 2026103882 |
| 2 | 65537 | 0 | 24/8 | full | 7.718050 | 5.586850 | 1.382936 | 1.348835..1.420873 | favorable | 2026103883 |
| 2 | 65537 | 0 | 24/8 | host | 0.388700 | 0.629500 | 0.626944 | 0.605073..0.650252 | adverse | 2026103884 |
| 2 | 65537 | 1 | 22/10 | gpu | 1.991056 | 1.017424 | 1.963263 | 1.951909..1.974587 | favorable | 2026103891 |
| 2 | 65537 | 1 | 22/10 | build | 6.178650 | 4.510300 | 1.407267 | 1.363174..1.450802 | favorable | 2026103892 |
| 2 | 65537 | 1 | 22/10 | full | 9.299850 | 7.303400 | 1.265630 | 1.227968..1.305467 | favorable | 2026103893 |
| 2 | 65537 | 1 | 22/10 | host | 0.575950 | 0.754400 | 0.770693 | 0.747393..0.795081 | adverse | 2026103894 |
| 2 | 65537 | 3 | 18/6 | gpu | 1.582640 | 0.608176 | 2.599199 | 2.583845..2.615542 | favorable | 2026103901 |
| 2 | 65537 | 3 | 18/6 | build | 5.095300 | 3.492500 | 1.446046 | 1.394035..1.502434 | favorable | 2026103902 |
| 2 | 65537 | 3 | 18/6 | full | 7.142400 | 5.658650 | 1.289596 | 1.240681..1.347062 | favorable | 2026103903 |
| 2 | 65537 | 3 | 18/6 | host | 0.482400 | 0.598250 | 0.791106 | 0.757630..0.828763 | adverse | 2026103904 |
| 2 | 65537 | 4 | 24/4 | gpu | 2.102384 | 0.396032 | 5.393213 | 5.142201..5.757675 | favorable | 2026103911 |
| 2 | 65537 | 4 | 24/4 | build | 5.819200 | 3.265400 | 1.803479 | 1.737253..1.873185 | favorable | 2026103912 |
| 2 | 65537 | 4 | 24/4 | full | 7.615300 | 4.991700 | 1.517566 | 1.473277..1.560531 | favorable | 2026103913 |
| 2 | 65537 | 4 | 24/4 | host | 0.449800 | 0.689800 | 0.637970 | 0.624384..0.651054 | adverse | 2026103914 |
| 2 | 65537 | 5 | 10/10 | gpu | 1.054512 | 1.053664 | 0.998335 | 0.993731..1.002440 | inconclusive | 2026103921 |
| 2 | 65537 | 5 | 10/10 | build | 4.167450 | 4.237000 | 0.973887 | 0.944036..1.003834 | inconclusive | 2026103922 |
| 2 | 65537 | 5 | 10/10 | full | 7.098900 | 7.236200 | 0.970424 | 0.939685..1.002491 | inconclusive | 2026103923 |
| 2 | 65537 | 5 | 10/10 | host | 0.497700 | 0.721700 | 0.744275 | 0.714038..0.778834 | adverse | 2026103924 |
| 2 | 65537 | 6 | 24/24 | gpu | 2.502144 | 2.500192 | 1.001721 | 0.999529..1.004059 | inconclusive | 2026103931 |
| 2 | 65537 | 6 | 24/24 | build | 6.642650 | 6.572750 | 1.003353 | 0.982017..1.024350 | inconclusive | 2026103932 |
| 2 | 65537 | 6 | 24/24 | full | 9.344000 | 9.226000 | 1.017076 | 0.991457..1.044197 | inconclusive | 2026103933 |
| 2 | 65537 | 6 | 24/24 | host | 0.474400 | 0.483200 | 0.929476 | 0.880586..0.977454 | adverse | 2026103934 |
| 2 | 65537 | 8 | 6/6 | gpu | 0.559088 | 0.556144 | 0.998817 | 0.986398..1.011720 | inconclusive | 2026103941 |
| 2 | 65537 | 8 | 6/6 | build | 2.955250 | 3.482050 | 0.905052 | 0.862003..0.950841 | adverse | 2026103942 |
| 2 | 65537 | 8 | 6/6 | full | 5.331200 | 5.340900 | 0.965769 | 0.930682..1.001077 | inconclusive | 2026103943 |
| 2 | 65537 | 8 | 6/6 | host | 0.499600 | 0.651950 | 0.757042 | 0.732690..0.781037 | adverse | 2026103944 |
| 2 | 524287 | 0 | 24/8 | gpu | 5.881056 | 2.397760 | 2.455867 | 2.451125..2.461073 | favorable | 2026103951 |
| 2 | 524287 | 0 | 24/8 | build | 12.569950 | 10.852700 | 1.190251 | 1.152587..1.229172 | favorable | 2026103952 |
| 2 | 524287 | 0 | 24/8 | full | 15.847450 | 13.833250 | 1.162047 | 1.124404..1.209860 | favorable | 2026103953 |
| 2 | 524287 | 0 | 24/8 | host | 2.468750 | 4.763200 | 0.580366 | 0.554111..0.606116 | adverse | 2026103954 |
| 2 | 524288 | 0 | 24/8 | gpu | 5.699744 | 2.286944 | 2.490425 | 2.483759..2.495490 | favorable | 2026103961 |
| 2 | 524288 | 0 | 24/8 | build | 11.737700 | 8.430100 | 1.388369 | 1.365408..1.411368 | favorable | 2026103962 |
| 2 | 524288 | 0 | 24/8 | full | 15.755450 | 12.425550 | 1.251125 | 1.212110..1.287893 | favorable | 2026103963 |
| 2 | 524288 | 0 | 24/8 | host | 1.396900 | 2.369800 | 0.592074 | 0.572792..0.612576 | adverse | 2026103964 |
| 2 | 524289 | 0 | 24/8 | gpu | 5.645376 | 2.321792 | 2.431359 | 2.428080..2.434988 | favorable | 2026103971 |
| 2 | 524289 | 0 | 24/8 | build | 11.610900 | 8.623200 | 1.347944 | 1.325190..1.371067 | favorable | 2026103972 |
| 2 | 524289 | 0 | 24/8 | full | 15.653950 | 12.752500 | 1.218806 | 1.193458..1.243566 | favorable | 2026103973 |
| 2 | 524289 | 0 | 24/8 | host | 1.475050 | 2.524900 | 0.590515 | 0.570705..0.612388 | adverse | 2026103974 |
| 2 | 1000003 | 0 | 24/8 | gpu | 9.928944 | 4.038784 | 2.459501 | 2.456899..2.462368 | favorable | 2026103981 |
| 2 | 1000003 | 0 | 24/8 | build | 19.057150 | 14.229350 | 1.310363 | 1.266072..1.345068 | favorable | 2026103982 |
| 2 | 1000003 | 0 | 24/8 | full | 25.702850 | 21.043950 | 1.206644 | 1.172534..1.241895 | favorable | 2026103983 |
| 2 | 1000003 | 0 | 24/8 | host | 2.574350 | 4.423750 | 0.594704 | 0.583205..0.606646 | adverse | 2026103984 |
| 2 | 1000003 | 4 | 24/4 | gpu | 9.656752 | 1.665472 | 5.828685 | 5.817363..5.840170 | favorable | 2026103991 |
| 2 | 1000003 | 4 | 24/4 | build | 18.915650 | 12.528400 | 1.497055 | 1.445139..1.544292 | favorable | 2026103992 |
| 2 | 1000003 | 4 | 24/4 | full | 24.597950 | 18.082550 | 1.365840 | 1.335422..1.397183 | favorable | 2026103993 |
| 2 | 1000003 | 4 | 24/4 | host | 2.479300 | 4.040850 | 0.600510 | 0.589143..0.611802 | adverse | 2026103994 |
| 2 | 1000003 | 6 | 24/24 | gpu | 9.980016 | 9.978928 | 1.000147 | 0.998542..1.001858 | inconclusive | 2026104001 |
| 2 | 1000003 | 6 | 24/24 | build | 23.323200 | 23.215600 | 1.020382 | 1.002347..1.043163 | favorable | 2026104002 |
| 2 | 1000003 | 6 | 24/24 | full | 41.519500 | 40.375300 | 1.015961 | 0.983540..1.052314 | inconclusive | 2026104003 |
| 2 | 1000003 | 6 | 24/24 | host | 2.718650 | 2.617250 | 1.018899 | 0.991476..1.045182 | inconclusive | 2026104004 |
| 2 | 1000003 | 8 | 6/6 | gpu | 3.700320 | 3.665392 | 0.998063 | 0.993072..1.002151 | inconclusive | 2026104011 |
| 2 | 1000003 | 8 | 6/6 | build | 14.408050 | 15.243500 | 0.964101 | 0.913874..1.013828 | inconclusive | 2026104012 |
| 2 | 1000003 | 8 | 6/6 | full | 22.937300 | 24.194950 | 0.928108 | 0.888676..0.961540 | adverse | 2026104013 |
| 2 | 1000003 | 8 | 6/6 | host | 3.763150 | 4.376200 | 0.843288 | 0.821720..0.865119 | adverse | 2026104014 |
| 2 | 65537 | 10 | 24/6 | gpu | 2.739840 | 0.820048 | 3.334151 | 3.325133..3.343041 | favorable | 2026104021 |
| 2 | 65537 | 10 | 24/6 | build | 8.097400 | 8.060800 | 1.027893 | 0.982431..1.074975 | inconclusive | 2026104022 |
| 2 | 65537 | 10 | 24/6 | full | 15.111600 | 14.103700 | 1.052306 | 1.020240..1.084732 | favorable | 2026104023 |
| 2 | 65537 | 10 | 24/6 | host | 0.432300 | 0.694000 | 0.643823 | 0.623087..0.666048 | adverse | 2026104024 |
| 2 | 65537 | 11 | 24/8 | gpu | 4.659648 | 1.964912 | 2.367797 | 2.355964..2.379244 | favorable | 2026104031 |
| 2 | 65537 | 11 | 24/8 | build | 11.593300 | 8.705800 | 1.358685 | 1.311784..1.410304 | favorable | 2026104032 |
| 2 | 65537 | 11 | 24/8 | full | 18.873950 | 15.751100 | 1.191511 | 1.152220..1.231166 | favorable | 2026104033 |
| 2 | 65537 | 11 | 24/8 | host | 0.479100 | 0.762650 | 0.631185 | 0.610685..0.654798 | adverse | 2026104034 |
| 2 | 65537 | 12 | 24/10 | gpu | 4.693792 | 2.412352 | 1.956156 | 1.941497..1.977737 | favorable | 2026104041 |
| 2 | 65537 | 12 | 24/10 | build | 11.676450 | 9.075650 | 1.313904 | 1.262748..1.367463 | favorable | 2026104042 |
| 2 | 65537 | 12 | 24/10 | full | 19.447950 | 16.639400 | 1.183665 | 1.153705..1.216551 | favorable | 2026104043 |
| 2 | 65537 | 12 | 24/10 | host | 0.392750 | 0.661750 | 0.618907 | 0.601920..0.635243 | adverse | 2026104044 |
| 2 | 65537 | 13 | 24/12 | gpu | 4.901792 | 2.927296 | 1.676926 | 1.667352..1.686644 | favorable | 2026104051 |
| 2 | 65537 | 13 | 24/12 | build | 8.502000 | 7.969250 | 1.055156 | 1.016797..1.097345 | favorable | 2026104052 |
| 2 | 65537 | 13 | 24/12 | full | 14.890500 | 14.852700 | 1.016001 | 0.983804..1.049411 | inconclusive | 2026104053 |
| 2 | 65537 | 13 | 24/12 | host | 0.413150 | 0.658550 | 0.637580 | 0.615573..0.661475 | adverse | 2026104054 |
| 2 | 65537 | 14 | 24/14 | gpu | 2.375696 | 1.537344 | 1.534690 | 1.526735..1.542246 | favorable | 2026104061 |
| 2 | 65537 | 14 | 24/14 | build | 10.698850 | 8.979300 | 1.150533 | 1.080167..1.225896 | favorable | 2026104062 |
| 2 | 65537 | 14 | 24/14 | full | 18.032950 | 16.514950 | 1.098280 | 1.064737..1.132282 | favorable | 2026104063 |
| 2 | 65537 | 14 | 24/14 | host | 0.378900 | 0.600850 | 0.629462 | 0.607204..0.652892 | adverse | 2026104064 |
| 2 | 65537 | 15 | 24/16 | gpu | 5.145744 | 3.850688 | 1.387535 | 1.344075..1.469054 | favorable | 2026104071 |
| 2 | 65537 | 15 | 24/16 | build | 8.498900 | 8.018150 | 1.040141 | 0.984727..1.095912 | inconclusive | 2026104072 |
| 2 | 65537 | 15 | 24/16 | full | 14.821400 | 15.004950 | 1.007322 | 0.974240..1.043784 | inconclusive | 2026104073 |
| 2 | 65537 | 15 | 24/16 | host | 0.380300 | 0.640100 | 0.620367 | 0.609390..0.631195 | adverse | 2026104074 |
| 2 | 65537 | 16 | 24/18 | gpu | 2.479024 | 1.982000 | 1.253599 | 1.250163..1.257265 | favorable | 2026104081 |
| 2 | 65537 | 16 | 24/18 | build | 8.219000 | 8.867050 | 0.981465 | 0.922344..1.044500 | inconclusive | 2026104082 |
| 2 | 65537 | 16 | 24/18 | full | 15.529150 | 15.782350 | 0.989190 | 0.952465..1.029159 | inconclusive | 2026104083 |
| 2 | 65537 | 16 | 24/18 | host | 0.382800 | 0.603150 | 0.622635 | 0.595196..0.651229 | adverse | 2026104084 |
| 2 | 65537 | 17 | 24/20 | gpu | 2.559904 | 2.216384 | 1.180570 | 1.147717..1.245487 | favorable | 2026104091 |
| 2 | 65537 | 17 | 24/20 | build | 7.887150 | 8.115850 | 0.997321 | 0.947508..1.048416 | inconclusive | 2026104092 |
| 2 | 65537 | 17 | 24/20 | full | 14.477750 | 14.339600 | 1.015855 | 0.973236..1.058839 | inconclusive | 2026104093 |
| 2 | 65537 | 17 | 24/20 | host | 0.399500 | 0.643350 | 0.617963 | 0.603440..0.632545 | adverse | 2026104094 |
| 2 | 65537 | 18 | 24/22 | gpu | 2.548272 | 2.382016 | 1.069184 | 1.064343..1.073403 | favorable | 2026104101 |
| 2 | 65537 | 18 | 24/22 | build | 7.374300 | 7.774000 | 0.975632 | 0.915158..1.041848 | inconclusive | 2026104102 |
| 2 | 65537 | 18 | 24/22 | full | 14.311800 | 14.003850 | 1.022476 | 0.983317..1.063208 | inconclusive | 2026104103 |
| 2 | 65537 | 18 | 24/22 | host | 0.363000 | 0.597950 | 0.623118 | 0.606282..0.639052 | adverse | 2026104104 |

## Source and local evidence

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sort.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sign_digits_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sign_digits_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sign_digits_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sign_digits_build_reference.rs`

Ignored raw logs/samples, boundary diagnostics, validation, hashes and delivery receipts are under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-sign-digits-*`. Raw process identities remain local. These measurements do not establish competitor maturity or a release gate on other hardware.
