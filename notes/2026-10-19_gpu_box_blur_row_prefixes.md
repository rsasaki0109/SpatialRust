# GPU box blur with workgroup-local row prefixes

Date: 2026-10-19. Named slice: GPU box-blur row prefixes.
Immediate parent: `a2aeef2b250bb3fd2553b8023b6b7dc97128fd0e` (merged PR #281).

## Change and exact integer proof

The previous unpacked horizontal pass shares adjacent windows but repeatedly
sums their valid row samples. The new private alternative scans each row once
inside a workgroup and answers each horizontal window by subtracting prefixes.
Its work is bounded by row width, rather than valid window width per output.
There is no new global prefix buffer or compute pass.

Eligible validated calls have widths 512..4096, at least 128 rows, kernel widths
at least 259 and at least 16,384 bytes of workgroup storage. This conservative
private heuristic is not a portable crossover guarantee. Original admission,
packed selection and all fallback shaders remain byte-identical. Smaller,
thinner or wider-than-4096 textures use the original schedule
when outside these explicit bounds. The original separable area/sample policy
still determines one versus two passes. Limited workgroup storage selects the
existing unpacked pipeline without adding an admission failure.

A 256-lane workgroup owns one complete row and initializes every element up to
its next power of two, including zero padding. A work-efficient Blelloch upsweep
and downsweep produce exclusive u32 prefixes in a static 4096-word workgroup
array. Disjoint tree nodes and barriers separate each level. Every subtree and
prefix is at most `4096*255 = 1,044,480`, below 2^20, including all-maximum rows.
For inclusive valid endpoints lo/hi, the exact window sum is
`prefix[hi] + source[hi] - prefix[lo]`; this avoids an extra prefix element for
the full-width endpoint. The monotone prefix prevents subtraction underflow.
Replicate borders add the original edge value times left/right mass; zero
borders omit those terms. The original area/rounding admission bounds every
horizontal sum and the unchanged vertical sum in u32, even for oversized kernels.
Final half-up rounding occurs exactly once in the original vertical shader.

Workgroup IDs own disjoint rows and advance by the existing dispatch group count
when needed. An extra barrier completes all prefix reads before the next row
reinitializes shared memory. Idle workgroups return uniformly before barriers.
The subtraction-based row advance prevents overflow. Validated u32 pixel counts
bound every row*width+x output index. Every logical output is written once;
no atomics, float approximations or cross-workgroup synchronization are added.

## Contracts, resources and tests

The sixth pipeline is lazy and used only by eligible unpacked horizontal calls.
It has the same uniform/source-texture/output-storage bindings, first-stage name
and original dispatch group count. The five existing programs are unchanged.
Original horizontal scratch requests, output texture extents, logical receipts,
metadata, one/two passes and timestamp query counts remain exact. Workgroup
memory increases to 16KiB in the new variant; compilation/cache/occupancy costs
can increase. Physical memory, driver transactions, bandwidth and energy are
unmeasured. Public API bodies, input/runtime/channel errors, core, dependencies,
feature flags and unsafe boundaries are unchanged. All host/device transfers
remain explicit; no host scan or readback is introduced into the resident blur.

Three new native regressions cover capability/shape boundaries and every CPU
scan width 1..4096, then compare raw GPU sums with complete immediate-parent
and independent u64 row-prefix references. Both borders, all-maximum/varied
bytes, nonzero unused RGBA channels, power-of-two padding, one-pixel widths,
maximum admitted integer mass, poisoned output guards, idle groups and forced
multiple-row reuse are covered. Complete parent/current images match independent
u64 rectangular means and full RGBA channels, with exact receipts/metadata,
profiles, lazy cache identities, budgets, rejection before initialization,
concurrent first use and retained results after pool clears/changed reuse.
Existing tests change only to register the native module. The frozen parent
copies the full original host path and references its five unchanged shaders;
only test visibility/imports and explicit cache ownership differ.

- BOX_ROW_PREFIX_PLAN checked 11184811 exclusive-prefix values for every width 1..4096; exact capability/shape boundaries, 16KiB storage and bounded u32 tree sums.

- BOX_ROW_PREFIX_PUBLIC checked 45949656 complete parent/CPU/RGBA bytes in 48 fixtures; unchanged receipts/pass/query counts, capability/shape fallback, lazy six-program cache, budgets, errors, concurrent first use and live output ownership.

- BOX_ROW_PREFIX_RAW checked 3061960 immediate-parent/current/u64/guard words in 548 fixtures; all scan powers, padding, maximum mass, both borders, source RGBA poison, idle groups and repeated rows per workgroup.

Final validation: 188 native test executions, excluding repeated targeted prechecks.

| Check | Tests passed |
| --- | ---: |
| clippy | 0 |
| image | 185 |
| facade | 2 |
| cpu-only | 0 |
| wasm-minimal | 0 |
| wasm-ci | 0 |
| fmt | 0 |
| diff | 0 |
| facade-gaussian | 1 |

All-feature/all-target warning-denying Clippy, CPU-only, minimal GPU WASM,
CI-feature GPU/image WASM, formatting and diff checks pass. 626
original GPU/image sources are unchanged; 638 current source,
manifest and executable hashes match across precheck, measurement and final
validation. Published files/PR text contain zero forbidden adapter-name matches.

## Measurement method and limitations

Two consecutive release surveys each use 24 synthetic cases, three warmups and
30 measured pairs. GPU `pass` time is the actual first stage's timestamp in a
complete blur call; query collection and correctness checks are outside timing.
`resident` measures the complete public blur on a retained uploaded input plus
wait_idle. `full` includes explicit upload, the complete blur and u8 readback.
Checks, returned-result destruction and explicit pool recycling are excluded.
The four old/current resident/full positions use balanced XOR rotation; GPU
first-stage variants alternate. Both variants use their complete host path,
not an unchanged host plan with a substituted measured dispatch.

Five fresh application-cache scenarios use 20 alternating pairs each in two
surveys: small, large, small-then-large, large-then-small and all-programs. The
new cache and frozen parent cache are reset between variants on a retained
device. Upload/runtime creation, validation and explicit recycling are outside
timing; the whole sequence of resident calls and completion is timed. Pipeline
counts reflect actual lazy programs: mixed shapes require three instead of two,
and all-program sequences require six instead of five.
There is no explicit driver-cache reset or process-/hardware-cold guarantee.

Ratios are paired geometric means parent/current (>1 favors this change).
Individual 95% intervals use 10,000 paired log bootstrap draws with sorted
indices 249/9749 and math.fsum(values)/N means. Warm seeds are 2026101951 +
1000*survey + 10*case_index + route_index (one-based surveys, zero-based case/
route indices; pass/resident/full order). Cold seeds are 2026101952 + 1000*survey
+ case_index. No multiple-comparison correction, epsilon, removed outlier,
dropped pair or selected rerun is used within the two fixed final surveys. All rows and controls are retained.
Shared-host pre/post samples cannot prove isolation during every pair or
portable application speed. This task's builds/tests do not overlap measurements.
No competitor or other-device speed comparison is inferred.

| Route | Eligible prefix | Comparisons | Paired ratio range |
| --- | --- | --- | --- |
| pass | True | 36 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 3.605186..92.418222 |
| pass | False | 0 favorable / 12 inconclusive / 0 adverse / 0 unresolved | 0.991340..1.019552 |
| resident | True | 36 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1.190382..13.622804 |
| resident | False | 0 favorable / 12 inconclusive / 0 adverse / 0 unresolved | 0.964270..1.054365 |
| full | True | 24 favorable / 12 inconclusive / 0 adverse / 0 unresolved | 0.976392..3.230136 |
| full | False | 1 favorable / 11 inconclusive / 0 adverse / 0 unresolved | 0.946402..1.054570 |

Fresh application-cache comparisons: 1 favorable / 4 inconclusive / 5 adverse / 0 unresolved.

Eligible pass paired ratios span 3.605186..92.418222; 36 favorable / 0 inconclusive / 0 adverse / 0 unresolved.

Eligible resident paired ratios span 1.190382..13.622804; 36 favorable / 0 inconclusive / 0 adverse / 0 unresolved.

Eligible full paired ratios span 0.976392..3.230136; 24 favorable / 12 inconclusive / 0 adverse / 0 unresolved.

Both mixed first-use orders grow from two cached parent programs to three current programs. Across the two surveys, all four mixed intervals are adverse, with 33.00%..36.08% more elapsed time. All-program sequences grow from five to six programs, with one adverse and one inconclusive observation. The resident steady-state benefit is retained with this explicitly recorded first-use tradeoff, rather than claiming every call improves.

The parent receives a dedicated cache explicitly; its cache object is constructed outside timing. The current public call initializes its runtime cache inside first-use timing. Individual pipeline construction, encoding, submission and completion remain inside both timed calls.

All individually adverse complete-call observations:

No individually adverse warm complete-call interval was observed; inconclusive results do not prove equality.

- Fresh cache, survey 1, small_then_large: 36.08% more time; ratio 0.734840 (0.652254..0.833375).

- Fresh cache, survey 1, large_then_small: 33.93% more time; ratio 0.746640 (0.698809..0.796933).

- Fresh cache, survey 2, small_then_large: 33.00% more time; ratio 0.751862 (0.698010..0.811834).

- Fresh cache, survey 2, large_then_small: 35.48% more time; ratio 0.738122 (0.689174..0.793765).

- Fresh cache, survey 2, all_programs: 12.34% more time; ratio 0.890175 (0.831623..0.959447).

Shared-host observations do not identify a cause or guarantee performance elsewhere.

| Survey | Kind | Quiet observed | Busy counts pre/post | Maximum busy CPU cores pre/post | GPU samples pre/post |
| ---: | --- | --- | --- | --- | --- |
| 1 | exploratory | False | [2, 1] | [1.376, 0.994] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 1 | warm | False | [4, 6] | [0.972, 1.011] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 2 | warm | False | [5, 6] | [0.994, 0.963] | `[[[[0, 0]], [[0, 0]]], [[[0, 0]], [[0, 0]]]]` |
| 1 | fresh-cache | False | [6, 4] | [1.006, 0.933] | `[[[[3, 0]], [[3, 0]]], [[[0, 0]], [[1, 0]]]]` |
| 2 | fresh-cache | False | [5, 6] | [0.946, 0.979] | `[[[[3, 0]], [[0, 0]]], [[[1, 0]], [[0, 0]]]]` |

Environment without local adapter model:

- exploratory 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`

- warm 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`

- warm 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`

- fresh-cache 1: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`

- fresh-cache 2: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`

## Exploratory survey retained separately

One exploratory survey used the same production and correctness code, three warmups
and 30 pairs for all 24 cases/routes. Its cold-cache benchmark assertion still
expected the old program counts; that unused assertion was corrected before the
final binary was rebuilt. No production threshold or shader was changed. The
initial source/executable snapshot is hash-checked, and these 72 comparisons are
reported separately from the two final surveys. Initial seeds are 2026101901 +
1000*survey + 10*case_index + route_index, with the same individual interval method.

All 18 eligible first-stage and resident intervals favored the candidate. Full
transfer-including calls had 13 favorable, four inconclusive and one adverse
eligible result; the 513x129/259x31 adverse ratio was 0.962907
(0.927207..0.997908), about 3.85% more time. Retain the conservative scope and
measure it again on the final binary without removing that observation.

| Survey | Image | Kernel | Zero | Prefix | Route | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 17x19 | 3x3 | False | False | pass | 0.005824 | 0.005696 | 1.029467 | 1.002446..1.058350 | favorable | 2026102901 |
| 1 | 17x19 | 3x3 | False | False | resident | 0.208500 | 0.215050 | 1.011809 | 0.971968..1.053276 | inconclusive | 2026102902 |
| 1 | 17x19 | 3x3 | False | False | full | 1.227900 | 1.232500 | 1.008832 | 0.973311..1.045957 | inconclusive | 2026102903 |
| 1 | 257x259 | 257x31 | False | False | pass | 0.072160 | 0.072128 | 1.001940 | 0.998057..1.005896 | inconclusive | 2026102911 |
| 1 | 257x259 | 257x31 | False | False | resident | 0.330200 | 0.325350 | 1.040655 | 1.010644..1.077552 | favorable | 2026102912 |
| 1 | 257x259 | 257x31 | False | False | full | 1.390150 | 1.407850 | 0.996172 | 0.971186..1.020087 | inconclusive | 2026102913 |
| 1 | 511x129 | 259x9 | False | False | pass | 0.075520 | 0.075456 | 1.000575 | 0.996413..1.004787 | inconclusive | 2026102921 |
| 1 | 511x129 | 259x9 | False | False | resident | 0.333000 | 0.333150 | 1.009565 | 0.982989..1.040835 | inconclusive | 2026102922 |
| 1 | 511x129 | 259x9 | False | False | full | 1.391600 | 1.400700 | 1.023026 | 0.983980..1.079785 | inconclusive | 2026102923 |
| 1 | 512x127 | 259x9 | False | False | pass | 0.125088 | 0.124864 | 1.001015 | 0.998073..1.004074 | inconclusive | 2026102931 |
| 1 | 512x127 | 259x9 | False | False | resident | 0.404800 | 0.412800 | 0.989223 | 0.956584..1.027680 | inconclusive | 2026102932 |
| 1 | 512x127 | 259x9 | False | False | full | 1.492250 | 1.458650 | 1.035857 | 1.005762..1.069211 | favorable | 2026102933 |
| 1 | 4097x128 | 259x9 | False | False | pass | 0.525952 | 0.525648 | 1.000309 | 0.998481..1.002026 | inconclusive | 2026102941 |
| 1 | 4097x128 | 259x9 | False | False | resident | 0.900100 | 0.880200 | 1.030809 | 0.996518..1.084201 | inconclusive | 2026102942 |
| 1 | 4097x128 | 259x9 | False | False | full | 2.974500 | 3.010100 | 1.001494 | 0.978376..1.025439 | inconclusive | 2026102943 |
| 1 | 1025x257 | 257x31 | False | False | pass | 0.353696 | 0.351296 | 1.001846 | 0.998721..1.005198 | inconclusive | 2026102951 |
| 1 | 1025x257 | 257x31 | False | False | resident | 0.696950 | 0.725100 | 1.025884 | 0.965759..1.125937 | inconclusive | 2026102952 |
| 1 | 1025x257 | 257x31 | False | False | full | 2.314700 | 2.471950 | 0.956293 | 0.897249..1.006847 | inconclusive | 2026102953 |
| 1 | 512x128 | 259x1 | False | True | pass | 0.107008 | 0.023856 | 4.453379 | 4.419841..4.486954 | favorable | 2026102961 |
| 1 | 512x128 | 259x1 | False | True | resident | 0.377700 | 0.298500 | 1.215719 | 1.174289..1.257433 | favorable | 2026102962 |
| 1 | 512x128 | 259x1 | False | True | full | 1.447050 | 1.456250 | 1.000104 | 0.966156..1.039984 | inconclusive | 2026102963 |
| 1 | 513x129 | 259x31 | False | True | pass | 0.126512 | 0.035312 | 3.585225 | 3.519550..3.629892 | favorable | 2026102971 |
| 1 | 513x129 | 259x31 | False | True | resident | 0.523750 | 0.418300 | 1.264026 | 1.204375..1.328596 | favorable | 2026102972 |
| 1 | 513x129 | 259x31 | False | True | full | 1.661100 | 1.695950 | 0.962907 | 0.927207..0.997908 | adverse | 2026102973 |
| 1 | 1025x257 | 259x9 | False | True | pass | 0.541792 | 0.125104 | 4.304941 | 4.270810..4.338413 | favorable | 2026102981 |
| 1 | 1025x257 | 259x9 | False | True | resident | 0.968400 | 0.571450 | 1.723459 | 1.662324..1.805038 | favorable | 2026102982 |
| 1 | 1025x257 | 259x9 | False | True | full | 2.368000 | 2.223850 | 1.083413 | 1.060891..1.107084 | favorable | 2026102983 |
| 1 | 1025x257 | 259x9 | True | True | pass | 0.555472 | 0.127488 | 4.347902 | 4.326078..4.368461 | favorable | 2026102991 |
| 1 | 1025x257 | 259x9 | True | True | resident | 0.957850 | 0.531600 | 1.781693 | 1.740963..1.822053 | favorable | 2026102992 |
| 1 | 1025x257 | 259x9 | True | True | full | 2.210250 | 2.037300 | 1.100824 | 1.070442..1.131100 | favorable | 2026102993 |
| 1 | 1920x1080 | 259x1 | False | True | pass | 2.659008 | 0.320736 | 8.291143 | 8.285049..8.297260 | favorable | 2026103001 |
| 1 | 1920x1080 | 259x1 | False | True | resident | 3.252600 | 0.901100 | 3.611736 | 3.513380..3.703554 | favorable | 2026103002 |
| 1 | 1920x1080 | 259x1 | False | True | full | 9.970400 | 7.589600 | 1.341148 | 1.306002..1.376097 | favorable | 2026103003 |
| 1 | 1920x1080 | 259x31 | False | True | pass | 2.614992 | 0.315472 | 8.302500 | 8.285512..8.331591 | favorable | 2026103011 |
| 1 | 1920x1080 | 259x31 | False | True | resident | 3.477450 | 1.135050 | 2.995308 | 2.863570..3.083410 | favorable | 2026103012 |
| 1 | 1920x1080 | 259x31 | False | True | full | 9.652900 | 7.378400 | 1.308861 | 1.288766..1.328545 | favorable | 2026103013 |
| 1 | 1920x1080 | 511x31 | False | True | pass | 3.208976 | 0.208288 | 15.401558 | 15.385176..15.417971 | favorable | 2026103021 |
| 1 | 1920x1080 | 511x31 | False | True | resident | 4.369350 | 0.971050 | 4.636989 | 4.525597..4.755243 | favorable | 2026103022 |
| 1 | 1920x1080 | 511x31 | False | True | full | 10.672350 | 7.107300 | 1.527812 | 1.493162..1.563536 | favorable | 2026103023 |
| 1 | 1920x1080 | 4099x9 | False | True | pass | 5.735840 | 0.187936 | 30.529720 | 30.507723..30.552844 | favorable | 2026103031 |
| 1 | 1920x1080 | 4099x9 | False | True | resident | 6.308750 | 0.727550 | 8.866360 | 8.674454..9.064123 | favorable | 2026103032 |
| 1 | 1920x1080 | 4099x9 | False | True | full | 12.896350 | 7.288800 | 1.776551 | 1.709908..1.831830 | favorable | 2026103033 |
| 1 | 1920x1080 | 4099x9 | True | True | pass | 5.714800 | 0.187744 | 30.454672 | 30.436939..30.474877 | favorable | 2026103041 |
| 1 | 1920x1080 | 4099x9 | True | True | resident | 6.217250 | 0.691500 | 8.944763 | 8.728255..9.171727 | favorable | 2026103042 |
| 1 | 1920x1080 | 4099x9 | True | True | full | 12.516300 | 6.922350 | 1.782676 | 1.727957..1.831275 | favorable | 2026103043 |
| 1 | 3840x2160 | 259x3 | False | True | pass | 6.542016 | 0.663648 | 9.880863 | 9.841510..9.950461 | favorable | 2026103051 |
| 1 | 3840x2160 | 259x3 | False | True | resident | 10.281800 | 4.413150 | 2.321794 | 2.301405..2.341423 | favorable | 2026103052 |
| 1 | 3840x2160 | 259x3 | False | True | full | 27.040550 | 21.572000 | 1.191200 | 1.094256..1.262906 | favorable | 2026103053 |
| 1 | 3840x2160 | 259x31 | False | True | pass | 6.567344 | 0.665136 | 9.858143 | 9.820524..9.879860 | favorable | 2026103061 |
| 1 | 3840x2160 | 259x31 | False | True | resident | 10.994150 | 5.137350 | 2.110553 | 2.041637..2.171380 | favorable | 2026103062 |
| 1 | 3840x2160 | 259x31 | False | True | full | 28.147000 | 22.095950 | 1.277051 | 1.230934..1.326530 | favorable | 2026103063 |
| 1 | 3840x2160 | 4099x3 | False | True | pass | 60.505776 | 0.660960 | 91.600738 | 91.556943..91.643222 | favorable | 2026103071 |
| 1 | 3840x2160 | 4099x3 | False | True | resident | 64.444950 | 4.490900 | 14.208519 | 13.891599..14.481285 | favorable | 2026103072 |
| 1 | 3840x2160 | 4099x3 | False | True | full | 82.535450 | 23.983300 | 3.458745 | 3.353165..3.558479 | favorable | 2026103073 |
| 1 | 3840x2160 | 4099x31 | False | True | pass | 60.549520 | 0.661088 | 91.611426 | 91.567283..91.658845 | favorable | 2026103081 |
| 1 | 3840x2160 | 4099x31 | False | True | resident | 65.082750 | 5.136900 | 12.577307 | 12.406347..12.724827 | favorable | 2026103082 |
| 1 | 3840x2160 | 4099x31 | False | True | full | 81.751650 | 21.876750 | 3.683213 | 3.553856..3.788294 | favorable | 2026103083 |
| 1 | 4096x128 | 259x9 | False | True | pass | 0.416032 | 0.058736 | 6.575731 | 6.027350..6.931811 | favorable | 2026103091 |
| 1 | 4096x128 | 259x9 | False | True | resident | 0.877200 | 0.541300 | 1.643969 | 1.568931..1.726155 | favorable | 2026103092 |
| 1 | 4096x128 | 259x9 | False | True | full | 8.658750 | 8.931600 | 0.977240 | 0.942402..1.013705 | inconclusive | 2026103093 |
| 1 | 4096x129 | 4099x31 | True | True | pass | 6.175968 | 0.081056 | 76.226715 | 75.811191..76.643432 | favorable | 2026103101 |
| 1 | 4096x129 | 4099x31 | True | True | resident | 8.153050 | 0.659450 | 12.301426 | 11.938351..12.677932 | favorable | 2026103102 |
| 1 | 4096x129 | 4099x31 | True | True | full | 10.738850 | 7.849300 | 1.416799 | 1.347973..1.496330 | favorable | 2026103103 |
| 1 | 512x129 | 65537x1 | False | True | pass | 0.246080 | 0.037888 | 6.315221 | 6.198347..6.420129 | favorable | 2026103111 |
| 1 | 512x129 | 65537x1 | False | True | resident | 0.477550 | 0.386350 | 1.251915 | 1.168406..1.341033 | favorable | 2026103112 |
| 1 | 512x129 | 65537x1 | False | True | full | 6.332900 | 6.464700 | 0.979028 | 0.944349..1.014489 | inconclusive | 2026103113 |
| 1 | 512x128 | 259x81 | False | True | pass | 0.313808 | 0.085296 | 3.689840 | 3.659668..3.721214 | favorable | 2026103121 |
| 1 | 512x128 | 259x81 | False | True | resident | 1.215300 | 1.180850 | 1.111770 | 1.005016..1.232783 | favorable | 2026103122 |
| 1 | 512x128 | 259x81 | False | True | full | 6.797150 | 6.774450 | 1.005175 | 0.981073..1.030931 | inconclusive | 2026103123 |
| 1 | 1920x1080 | 259x81 | False | True | pass | 2.883712 | 0.347968 | 8.290165 | 8.283456..8.296728 | favorable | 2026103131 |
| 1 | 1920x1080 | 259x81 | False | True | resident | 5.731650 | 2.619750 | 2.194010 | 2.141058..2.245265 | favorable | 2026103132 |
| 1 | 1920x1080 | 259x81 | False | True | full | 14.361800 | 12.767650 | 1.128654 | 1.100786..1.157290 | favorable | 2026103133 |

## Every warm comparison

| Survey | Image | Kernel | Zero | Prefix | Route | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 17x19 | 3x3 | False | False | pass | 0.005728 | 0.005792 | 0.994189 | 0.971511..1.017773 | inconclusive | 2026102951 |
| 1 | 17x19 | 3x3 | False | False | resident | 0.262650 | 0.255250 | 1.020912 | 0.982937..1.057652 | inconclusive | 2026102952 |
| 1 | 17x19 | 3x3 | False | False | full | 1.461550 | 1.427650 | 0.996120 | 0.968371..1.024765 | inconclusive | 2026102953 |
| 1 | 257x259 | 257x31 | False | False | pass | 0.071552 | 0.071616 | 0.997458 | 0.993115..1.001665 | inconclusive | 2026102961 |
| 1 | 257x259 | 257x31 | False | False | resident | 0.372750 | 0.370050 | 0.995924 | 0.975763..1.015894 | inconclusive | 2026102962 |
| 1 | 257x259 | 257x31 | False | False | full | 1.562650 | 1.578800 | 0.946402 | 0.880947..1.000713 | inconclusive | 2026102963 |
| 1 | 511x129 | 259x9 | False | False | pass | 0.074864 | 0.074928 | 0.998718 | 0.994366..1.002897 | inconclusive | 2026102971 |
| 1 | 511x129 | 259x9 | False | False | resident | 0.398000 | 0.407850 | 0.972834 | 0.922833..1.019384 | inconclusive | 2026102972 |
| 1 | 511x129 | 259x9 | False | False | full | 1.670200 | 1.687700 | 0.975292 | 0.896008..1.059422 | inconclusive | 2026102973 |
| 1 | 512x127 | 259x9 | False | False | pass | 0.123872 | 0.123664 | 0.999734 | 0.997467..1.002009 | inconclusive | 2026102981 |
| 1 | 512x127 | 259x9 | False | False | resident | 0.413150 | 0.421000 | 1.020783 | 0.989454..1.055101 | inconclusive | 2026102982 |
| 1 | 512x127 | 259x9 | False | False | full | 1.562000 | 1.544200 | 1.019621 | 0.990252..1.047675 | inconclusive | 2026102983 |
| 1 | 4097x128 | 259x9 | False | False | pass | 0.521248 | 0.521280 | 0.991340 | 0.972579..1.001662 | inconclusive | 2026102991 |
| 1 | 4097x128 | 259x9 | False | False | resident | 0.928700 | 0.924150 | 1.005395 | 0.986847..1.022422 | inconclusive | 2026102992 |
| 1 | 4097x128 | 259x9 | False | False | full | 3.357750 | 3.343900 | 1.006211 | 0.962460..1.043125 | inconclusive | 2026102993 |
| 1 | 1025x257 | 257x31 | False | False | pass | 0.387680 | 0.388224 | 0.999609 | 0.995054..1.004967 | inconclusive | 2026103001 |
| 1 | 1025x257 | 257x31 | False | False | resident | 0.792350 | 0.781650 | 1.038928 | 0.987033..1.125362 | inconclusive | 2026103002 |
| 1 | 1025x257 | 257x31 | False | False | full | 2.532500 | 2.558500 | 0.996738 | 0.968595..1.032114 | inconclusive | 2026103003 |
| 1 | 512x128 | 259x1 | False | True | pass | 0.120656 | 0.027056 | 4.460539 | 4.433393..4.495751 | favorable | 2026103011 |
| 1 | 512x128 | 259x1 | False | True | resident | 0.495000 | 0.399500 | 1.217257 | 1.172881..1.260941 | favorable | 2026103012 |
| 1 | 512x128 | 259x1 | False | True | full | 1.756700 | 1.771650 | 0.997124 | 0.966247..1.032008 | inconclusive | 2026103013 |
| 1 | 513x129 | 259x31 | False | True | pass | 0.145232 | 0.040720 | 3.605186 | 3.587445..3.623552 | favorable | 2026103021 |
| 1 | 513x129 | 259x31 | False | True | resident | 0.540550 | 0.437200 | 1.244784 | 1.202335..1.291296 | favorable | 2026103022 |
| 1 | 513x129 | 259x31 | False | True | full | 1.750500 | 1.822600 | 1.008218 | 0.962503..1.074822 | inconclusive | 2026103023 |
| 1 | 1025x257 | 259x9 | False | True | pass | 0.621824 | 0.144304 | 4.373674 | 4.341183..4.402755 | favorable | 2026103031 |
| 1 | 1025x257 | 259x9 | False | True | resident | 1.105100 | 0.672550 | 1.632825 | 1.544567..1.710278 | favorable | 2026103032 |
| 1 | 1025x257 | 259x9 | False | True | full | 2.718950 | 2.732900 | 1.002231 | 0.957996..1.043504 | inconclusive | 2026103033 |
| 1 | 1025x257 | 259x9 | True | True | pass | 0.731328 | 0.176448 | 4.147130 | 4.120557..4.172241 | favorable | 2026103041 |
| 1 | 1025x257 | 259x9 | True | True | resident | 1.494800 | 0.971450 | 1.553198 | 1.512285..1.596278 | favorable | 2026103042 |
| 1 | 1025x257 | 259x9 | True | True | full | 3.424950 | 3.016500 | 1.125674 | 1.087088..1.163702 | favorable | 2026103043 |
| 1 | 1920x1080 | 259x1 | False | True | pass | 2.724960 | 0.328384 | 8.382235 | 8.296868..8.549613 | favorable | 2026103051 |
| 1 | 1920x1080 | 259x1 | False | True | resident | 4.521650 | 1.548700 | 2.958321 | 2.784838..3.132265 | favorable | 2026103052 |
| 1 | 1920x1080 | 259x1 | False | True | full | 12.665950 | 9.812750 | 1.229232 | 1.144202..1.289837 | favorable | 2026103053 |
| 1 | 1920x1080 | 259x31 | False | True | pass | 2.803328 | 0.337664 | 8.361727 | 8.296406..8.487857 | favorable | 2026103061 |
| 1 | 1920x1080 | 259x31 | False | True | resident | 3.664450 | 1.238400 | 2.845936 | 2.654519..2.990001 | favorable | 2026103062 |
| 1 | 1920x1080 | 259x31 | False | True | full | 11.344850 | 8.825950 | 1.275528 | 1.241865..1.306179 | favorable | 2026103063 |
| 1 | 1920x1080 | 511x31 | False | True | pass | 4.169296 | 0.270640 | 15.402916 | 15.391686..15.414294 | favorable | 2026103071 |
| 1 | 1920x1080 | 511x31 | False | True | resident | 5.447700 | 1.147450 | 4.816938 | 4.644112..5.056420 | favorable | 2026103072 |
| 1 | 1920x1080 | 511x31 | False | True | full | 12.920600 | 8.482250 | 1.494428 | 1.452498..1.535203 | favorable | 2026103073 |
| 1 | 1920x1080 | 4099x9 | False | True | pass | 5.739664 | 0.188208 | 30.506098 | 30.471570..30.536742 | favorable | 2026103081 |
| 1 | 1920x1080 | 4099x9 | False | True | resident | 6.678200 | 0.805800 | 8.709060 | 8.449903..8.988111 | favorable | 2026103082 |
| 1 | 1920x1080 | 4099x9 | False | True | full | 14.230900 | 8.879450 | 1.626280 | 1.587403..1.667912 | favorable | 2026103083 |
| 1 | 1920x1080 | 4099x9 | True | True | pass | 5.714672 | 0.187568 | 30.467424 | 30.442143..30.493074 | favorable | 2026103091 |
| 1 | 1920x1080 | 4099x9 | True | True | resident | 6.320400 | 0.777100 | 8.134488 | 7.837641..8.426048 | favorable | 2026103092 |
| 1 | 1920x1080 | 4099x9 | True | True | full | 13.734800 | 8.679350 | 1.557963 | 1.509383..1.603638 | favorable | 2026103093 |
| 1 | 3840x2160 | 259x3 | False | True | pass | 8.009712 | 0.811024 | 9.888664 | 9.847983..9.937023 | favorable | 2026103101 |
| 1 | 3840x2160 | 259x3 | False | True | resident | 11.051800 | 4.807750 | 2.276189 | 2.216044..2.331716 | favorable | 2026103102 |
| 1 | 3840x2160 | 259x3 | False | True | full | 31.927800 | 26.748500 | 1.185021 | 1.154646..1.215114 | favorable | 2026103103 |
| 1 | 3840x2160 | 259x31 | False | True | pass | 7.248320 | 0.732864 | 9.903224 | 9.867615..9.943838 | favorable | 2026103111 |
| 1 | 3840x2160 | 259x31 | False | True | resident | 11.772200 | 5.799450 | 2.075751 | 2.035274..2.114993 | favorable | 2026103112 |
| 1 | 3840x2160 | 259x31 | False | True | full | 31.986450 | 27.444950 | 1.201353 | 1.165622..1.236477 | favorable | 2026103113 |
| 1 | 3840x2160 | 4099x3 | False | True | pass | 60.563520 | 0.660816 | 92.418222 | 91.796656..93.192666 | favorable | 2026103121 |
| 1 | 3840x2160 | 4099x3 | False | True | resident | 64.497900 | 4.700100 | 13.622804 | 13.370936..13.853890 | favorable | 2026103122 |
| 1 | 3840x2160 | 4099x3 | False | True | full | 84.616750 | 26.147750 | 3.230136 | 3.168510..3.291568 | favorable | 2026103123 |
| 1 | 3840x2160 | 4099x31 | False | True | pass | 60.535488 | 0.660848 | 91.824358 | 91.577137..92.272659 | favorable | 2026103131 |
| 1 | 3840x2160 | 4099x31 | False | True | resident | 65.360350 | 5.471450 | 11.826108 | 11.567331..12.059201 | favorable | 2026103132 |
| 1 | 3840x2160 | 4099x31 | False | True | full | 85.647350 | 27.102900 | 3.163555 | 3.088632..3.240514 | favorable | 2026103133 |
| 1 | 4096x128 | 259x9 | False | True | pass | 1.246880 | 0.197296 | 6.011913 | 5.411436..6.546850 | favorable | 2026103141 |
| 1 | 4096x128 | 259x9 | False | True | resident | 0.911850 | 0.570550 | 1.609874 | 1.554353..1.663826 | favorable | 2026103142 |
| 1 | 4096x128 | 259x9 | False | True | full | 9.914450 | 9.529150 | 1.034346 | 0.984866..1.091658 | inconclusive | 2026103143 |
| 1 | 4096x129 | 4099x31 | True | True | pass | 7.224064 | 0.094000 | 72.664474 | 69.305189..75.569432 | favorable | 2026103151 |
| 1 | 4096x129 | 4099x31 | True | True | resident | 8.030500 | 0.774450 | 10.466371 | 9.876581..11.077256 | favorable | 2026103152 |
| 1 | 4096x129 | 4099x31 | True | True | full | 11.508300 | 9.403800 | 1.289015 | 1.214011..1.374045 | favorable | 2026103153 |
| 1 | 512x129 | 65537x1 | False | True | pass | 0.411312 | 0.072352 | 5.673450 | 5.524536..5.782519 | favorable | 2026103161 |
| 1 | 512x129 | 65537x1 | False | True | resident | 0.753550 | 0.590300 | 1.292447 | 1.231796..1.358647 | favorable | 2026103162 |
| 1 | 512x129 | 65537x1 | False | True | full | 8.168900 | 8.060800 | 0.997332 | 0.967721..1.027452 | inconclusive | 2026103163 |
| 1 | 512x128 | 259x81 | False | True | pass | 0.338944 | 0.092320 | 3.912904 | 3.690003..4.365307 | favorable | 2026103171 |
| 1 | 512x128 | 259x81 | False | True | resident | 1.401750 | 1.165900 | 1.253899 | 1.166451..1.351068 | favorable | 2026103172 |
| 1 | 512x128 | 259x81 | False | True | full | 8.977700 | 9.098950 | 0.976392 | 0.925146..1.029047 | inconclusive | 2026103173 |
| 1 | 1920x1080 | 259x81 | False | True | pass | 3.361600 | 0.485360 | 6.902832 | 6.612478..7.129450 | favorable | 2026103181 |
| 1 | 1920x1080 | 259x81 | False | True | resident | 3.206000 | 1.442200 | 2.271288 | 2.203464..2.383402 | favorable | 2026103182 |
| 1 | 1920x1080 | 259x81 | False | True | full | 14.119750 | 13.949650 | 0.991474 | 0.966775..1.016217 | inconclusive | 2026103183 |
| 2 | 17x19 | 3x3 | False | False | pass | 0.005808 | 0.005792 | 1.019552 | 0.994570..1.047277 | inconclusive | 2026103951 |
| 2 | 17x19 | 3x3 | False | False | resident | 0.413850 | 0.400700 | 1.054365 | 0.978152..1.155611 | inconclusive | 2026103952 |
| 2 | 17x19 | 3x3 | False | False | full | 2.138350 | 2.204100 | 0.962667 | 0.910548..1.014450 | inconclusive | 2026103953 |
| 2 | 257x259 | 257x31 | False | False | pass | 0.071728 | 0.071632 | 0.999806 | 0.995912..1.003723 | inconclusive | 2026103961 |
| 2 | 257x259 | 257x31 | False | False | resident | 0.511750 | 0.501100 | 0.964270 | 0.891981..1.035105 | inconclusive | 2026103962 |
| 2 | 257x259 | 257x31 | False | False | full | 2.152450 | 2.208350 | 0.971651 | 0.917731..1.024812 | inconclusive | 2026103963 |
| 2 | 511x129 | 259x9 | False | False | pass | 0.075040 | 0.074848 | 1.003637 | 0.999130..1.008564 | inconclusive | 2026103971 |
| 2 | 511x129 | 259x9 | False | False | resident | 0.489900 | 0.480650 | 0.972655 | 0.898125..1.040390 | inconclusive | 2026103972 |
| 2 | 511x129 | 259x9 | False | False | full | 2.186400 | 2.281600 | 1.029277 | 0.979776..1.084533 | inconclusive | 2026103973 |
| 2 | 512x127 | 259x9 | False | False | pass | 0.324832 | 0.324528 | 1.000023 | 0.997356..1.002972 | inconclusive | 2026103981 |
| 2 | 512x127 | 259x9 | False | False | resident | 0.568700 | 0.529800 | 1.041715 | 0.995089..1.093471 | inconclusive | 2026103982 |
| 2 | 512x127 | 259x9 | False | False | full | 2.062250 | 1.944850 | 1.054570 | 1.009993..1.104624 | favorable | 2026103983 |
| 2 | 4097x128 | 259x9 | False | False | pass | 1.028208 | 1.030080 | 0.998718 | 0.997372..1.000099 | inconclusive | 2026103991 |
| 2 | 4097x128 | 259x9 | False | False | resident | 1.821800 | 1.843600 | 0.978463 | 0.935100..1.005803 | inconclusive | 2026103992 |
| 2 | 4097x128 | 259x9 | False | False | full | 4.823250 | 4.730600 | 0.996592 | 0.967897..1.017907 | inconclusive | 2026103993 |
| 2 | 1025x257 | 257x31 | False | False | pass | 0.529984 | 0.530896 | 1.000246 | 0.996711..1.003789 | inconclusive | 2026104001 |
| 2 | 1025x257 | 257x31 | False | False | resident | 1.106900 | 1.091700 | 1.035074 | 0.992353..1.112366 | inconclusive | 2026104002 |
| 2 | 1025x257 | 257x31 | False | False | full | 2.922700 | 3.000350 | 0.983474 | 0.960049..1.006588 | inconclusive | 2026104003 |
| 2 | 512x128 | 259x1 | False | True | pass | 0.137344 | 0.030528 | 4.471009 | 4.362575..4.540371 | favorable | 2026104011 |
| 2 | 512x128 | 259x1 | False | True | resident | 0.573650 | 0.477400 | 1.233474 | 1.195342..1.274509 | favorable | 2026104012 |
| 2 | 512x128 | 259x1 | False | True | full | 1.985850 | 1.976800 | 0.998380 | 0.970218..1.026949 | inconclusive | 2026104013 |
| 2 | 513x129 | 259x31 | False | True | pass | 0.221472 | 0.060528 | 3.656550 | 3.614307..3.707410 | favorable | 2026104021 |
| 2 | 513x129 | 259x31 | False | True | resident | 0.841550 | 0.715350 | 1.199195 | 1.162505..1.235692 | favorable | 2026104022 |
| 2 | 513x129 | 259x31 | False | True | full | 2.295450 | 2.289100 | 1.015377 | 0.968676..1.085250 | inconclusive | 2026104023 |
| 2 | 1025x257 | 259x9 | False | True | pass | 0.932624 | 0.214272 | 4.333322 | 4.300560..4.359032 | favorable | 2026104031 |
| 2 | 1025x257 | 259x9 | False | True | resident | 1.797500 | 1.125400 | 1.616891 | 1.585156..1.650384 | favorable | 2026104032 |
| 2 | 1025x257 | 259x9 | False | True | full | 4.011100 | 3.496650 | 1.147257 | 1.125887..1.169480 | favorable | 2026104033 |
| 2 | 1025x257 | 259x9 | True | True | pass | 0.943936 | 0.217904 | 4.319230 | 4.288277..4.345218 | favorable | 2026104041 |
| 2 | 1025x257 | 259x9 | True | True | resident | 1.792900 | 1.078650 | 1.671762 | 1.649207..1.695527 | favorable | 2026104042 |
| 2 | 1025x257 | 259x9 | True | True | full | 3.786100 | 3.127100 | 1.190822 | 1.152706..1.229311 | favorable | 2026104043 |
| 2 | 1920x1080 | 259x1 | False | True | pass | 3.443920 | 0.485312 | 7.013837 | 6.950629..7.074462 | favorable | 2026104051 |
| 2 | 1920x1080 | 259x1 | False | True | resident | 5.320800 | 1.779800 | 2.948519 | 2.844395..3.059056 | favorable | 2026104052 |
| 2 | 1920x1080 | 259x1 | False | True | full | 14.178200 | 10.795800 | 1.328226 | 1.292005..1.370526 | favorable | 2026104053 |
| 2 | 1920x1080 | 259x31 | False | True | pass | 2.886240 | 0.347808 | 8.299599 | 8.293193..8.305832 | favorable | 2026104061 |
| 2 | 1920x1080 | 259x31 | False | True | resident | 4.490650 | 1.603600 | 2.716061 | 2.624189..2.803419 | favorable | 2026104062 |
| 2 | 1920x1080 | 259x31 | False | True | full | 13.407150 | 10.564050 | 1.260726 | 1.242398..1.279337 | favorable | 2026104063 |
| 2 | 1920x1080 | 511x31 | False | True | pass | 5.163968 | 0.333744 | 15.471569 | 15.461518..15.481875 | favorable | 2026104071 |
| 2 | 1920x1080 | 511x31 | False | True | resident | 6.433850 | 1.366550 | 4.608390 | 4.504639..4.705958 | favorable | 2026104072 |
| 2 | 1920x1080 | 511x31 | False | True | full | 15.165450 | 10.716900 | 1.441878 | 1.408023..1.474567 | favorable | 2026104073 |
| 2 | 1920x1080 | 4099x9 | False | True | pass | 7.047728 | 0.231424 | 30.494827 | 30.459715..30.536440 | favorable | 2026104081 |
| 2 | 1920x1080 | 4099x9 | False | True | resident | 8.300950 | 0.975800 | 8.646922 | 8.343206..8.962230 | favorable | 2026104082 |
| 2 | 1920x1080 | 4099x9 | False | True | full | 17.234450 | 10.544800 | 1.603154 | 1.543086..1.663280 | favorable | 2026104083 |
| 2 | 1920x1080 | 4099x9 | True | True | pass | 5.904032 | 0.193504 | 30.531017 | 30.509618..30.553372 | favorable | 2026104091 |
| 2 | 1920x1080 | 4099x9 | True | True | resident | 7.048450 | 0.799500 | 8.769227 | 8.450185..9.089127 | favorable | 2026104092 |
| 2 | 1920x1080 | 4099x9 | True | True | full | 14.902350 | 9.570650 | 1.569453 | 1.530745..1.607130 | favorable | 2026104093 |
| 2 | 3840x2160 | 259x3 | False | True | pass | 8.862128 | 0.894976 | 9.879973 | 9.818989..9.945906 | favorable | 2026104101 |
| 2 | 3840x2160 | 259x3 | False | True | resident | 12.008300 | 5.226750 | 2.225037 | 2.151660..2.297750 | favorable | 2026104102 |
| 2 | 3840x2160 | 259x3 | False | True | full | 32.695450 | 28.158950 | 1.155282 | 1.121477..1.195258 | favorable | 2026104103 |
| 2 | 3840x2160 | 259x31 | False | True | pass | 8.263376 | 0.828240 | 9.881329 | 9.826780..9.940284 | favorable | 2026104111 |
| 2 | 3840x2160 | 259x31 | False | True | resident | 11.832150 | 5.752650 | 2.099595 | 2.056344..2.138854 | favorable | 2026104112 |
| 2 | 3840x2160 | 259x31 | False | True | full | 32.557750 | 27.324450 | 1.186562 | 1.146303..1.225936 | favorable | 2026104113 |
| 2 | 3840x2160 | 4099x3 | False | True | pass | 60.509936 | 0.661024 | 88.601869 | 82.565938..91.967017 | favorable | 2026104121 |
| 2 | 3840x2160 | 4099x3 | False | True | resident | 64.736750 | 4.776400 | 13.161913 | 12.764546..13.535856 | favorable | 2026104122 |
| 2 | 3840x2160 | 4099x3 | False | True | full | 84.987600 | 26.881550 | 3.137831 | 3.039192..3.226558 | favorable | 2026104123 |
| 2 | 3840x2160 | 4099x31 | False | True | pass | 61.042640 | 0.666256 | 91.858677 | 91.614273..92.305770 | favorable | 2026104131 |
| 2 | 3840x2160 | 4099x31 | False | True | resident | 65.770950 | 5.583100 | 11.813816 | 11.583660..12.041458 | favorable | 2026104132 |
| 2 | 3840x2160 | 4099x31 | False | True | full | 86.415200 | 27.380700 | 3.088253 | 2.965365..3.206045 | favorable | 2026104133 |
| 2 | 4096x128 | 259x9 | False | True | pass | 1.223136 | 0.197856 | 5.995485 | 5.511039..6.457019 | favorable | 2026104141 |
| 2 | 4096x128 | 259x9 | False | True | resident | 0.948550 | 0.601900 | 1.584583 | 1.528097..1.644487 | favorable | 2026104142 |
| 2 | 4096x128 | 259x9 | False | True | full | 9.687750 | 9.493200 | 1.028912 | 1.003205..1.056682 | favorable | 2026104143 |
| 2 | 4096x129 | 4099x31 | True | True | pass | 7.939440 | 0.102272 | 71.604433 | 68.218846..74.736969 | favorable | 2026104151 |
| 2 | 4096x129 | 4099x31 | True | True | resident | 8.741000 | 0.749600 | 11.032850 | 10.274333..11.743140 | favorable | 2026104152 |
| 2 | 4096x129 | 4099x31 | True | True | full | 12.067650 | 9.487200 | 1.332196 | 1.266116..1.406267 | favorable | 2026104153 |
| 2 | 512x129 | 65537x1 | False | True | pass | 0.414240 | 0.072128 | 5.772222 | 5.730264..5.812241 | favorable | 2026104161 |
| 2 | 512x129 | 65537x1 | False | True | resident | 0.758450 | 0.582400 | 1.355281 | 1.276659..1.444142 | favorable | 2026104162 |
| 2 | 512x129 | 65537x1 | False | True | full | 7.893150 | 8.197250 | 1.010107 | 0.978202..1.043724 | inconclusive | 2026104163 |
| 2 | 512x128 | 259x81 | False | True | pass | 0.339472 | 0.092592 | 3.709024 | 3.680730..3.737838 | favorable | 2026104171 |
| 2 | 512x128 | 259x81 | False | True | resident | 1.311000 | 1.103500 | 1.190382 | 1.108217..1.282219 | favorable | 2026104172 |
| 2 | 512x128 | 259x81 | False | True | full | 8.510550 | 8.811600 | 0.981966 | 0.943623..1.026652 | inconclusive | 2026104173 |
| 2 | 1920x1080 | 259x81 | False | True | pass | 4.001424 | 0.510160 | 7.747714 | 7.663180..7.829805 | favorable | 2026104181 |
| 2 | 1920x1080 | 259x81 | False | True | resident | 6.199650 | 2.853050 | 2.173425 | 2.095904..2.290341 | favorable | 2026104182 |
| 2 | 1920x1080 | 259x81 | False | True | full | 15.816150 | 15.374000 | 1.043626 | 0.983003..1.101053 | inconclusive | 2026104183 |

## Every fresh-cache comparison

| Survey | Scenario | Parent/new pipeline counts | Parent median ms | New median ms | Paired ratio | 95% interval | Result | Seed |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | small | [2, 2] | 2.828650 | 2.763750 | 1.141263 | 0.998280..1.358649 | inconclusive | 2026102952 |
| 1 | large | [2, 2] | 2.819050 | 2.467400 | 1.109070 | 1.015497..1.205141 | favorable | 2026102953 |
| 1 | small_then_large | [2, 3] | 3.146400 | 4.287200 | 0.734840 | 0.652254..0.833375 | adverse | 2026102954 |
| 1 | large_then_small | [2, 3] | 3.136200 | 4.062850 | 0.746640 | 0.698809..0.796933 | adverse | 2026102955 |
| 1 | all_programs | [5, 6] | 7.188750 | 7.669250 | 0.945242 | 0.858101..1.051350 | inconclusive | 2026102956 |
| 2 | small | [2, 2] | 2.974450 | 2.903300 | 1.136299 | 0.996657..1.372262 | inconclusive | 2026103952 |
| 2 | large | [2, 2] | 2.903700 | 2.748900 | 1.053976 | 0.947483..1.173638 | inconclusive | 2026103953 |
| 2 | small_then_large | [2, 3] | 3.108000 | 4.139100 | 0.751862 | 0.698010..0.811834 | adverse | 2026103954 |
| 2 | large_then_small | [2, 3] | 3.210000 | 4.439450 | 0.738122 | 0.689174..0.793765 | adverse | 2026103955 |
| 2 | all_programs | [5, 6] | 7.243100 | 8.419750 | 0.890175 | 0.831623..0.959447 | adverse | 2026103956 |

Measured executable SHA-256: initial `8d8a1ab132a3e1266da87d708d6e7fea4482290886d45b657014536e9ff91369`; final `5b53eaaccf4c3f07382be4735e12c00666500699f778dba0306a65ee67520cd4`. Native-only cold benchmark assertions differ; all production/correctness sources match the retained exploratory snapshot.

## Source and local evidence

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_row_prefix.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_row_prefix_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_row_prefix_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_row_prefix_benchmark.rs`

Ignored logs, raw paired samples, process identities, source/executable hashes and delivery receipts remain under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/box-row-prefix-*`. Published diagnostics use anonymous counts/capabilities. These results do not establish a competitor maturity ranking or a cross-device release gate.
