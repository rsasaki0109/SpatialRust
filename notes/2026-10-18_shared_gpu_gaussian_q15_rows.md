# Shared row calculations in GPU Gaussian Q15 vertical filtering

Date: 2026-10-18. Named slice: shared Q15 Gaussian rows. Parent:
`791894f00ac4d30fdd0f97ffdf3c2b8f9b4378f2` (merged PR #267).

## Change and exactness

The Q15 multi-channel vertical Gaussian stage no longer repeats row mapping
and weight selection separately for every channel. A private two-lane shader
handles two channels; a private four-lane shader handles three/four channels.
They gather the same uncompressed horizontal u32 sums and perform the original
componentwise base-65536 digit accumulation, symmetric tap pairing, one carry,
half-up shift and final byte clamp. Inactive physical output components remain
zero, including constant-border cases whose ignored constants are nonzero.

Q15 horizontal sums are <2^23. Nonnegative weight mass 32768 bounds each low
digit sum by 65535*32768 (<2^31) and each high digit sum by 127*32768 (<2^22).
Vector operations are componentwise: each channel keeps the original exact
u32 partials and `(hi + (lo >> 16) + 8192) >> 14` rounding. No floating-point
pixel math, approximate arithmetic or cross-channel carries are introduced.
Mapped rows are shared per channel pair or all active channels, while each
active source word is still loaded once per tap. The change reduces coordinate
and address calculation, not the number of logical active sum words consumed.

The planner, quantization, coordinate mapping helper, original WGSL shaders,
horizontal passes, scratch extents, three-binding layouts, uniforms, two passes,
submission, receipts, ownership and explicit upload/readback contracts are
unchanged. Q8 and one-channel calls return the original vertical pipeline by
identity. Public APIs, dependencies, core, point-cloud algorithms and automatic
GPU admission/placement are unchanged. New helper shaders are selected only
for Q15 multi-channel plans; all errors and profiler capacity checks still
precede runtime cache/resource creation.

## Cache and cold-call tradeoff

Two optional per-runtime OnceLocks cache the private two-/four-lane vertical
pipelines. Only selected variants compile, and concurrent first use shares
initialization. This preserves the complete original shader programs for
controls. The existing original unpacked pair (including its original vertical
pipeline) is still initialized for the horizontal stage. Consequently the
first affected layout also creates its selected private vertical pipeline;
a mixed runtime can retain up to two additional vertical pipelines. Shader
compilation and driver pipeline memory are not counted as scratch pool bytes.

The final paired timings below are warm-call measurements. Cold compilation
latency and physical pipeline/VRAM cost were not measured, so the results do not
establish that a one-off cold image call improves. Temporary buffer/texture
descriptors and pool budgets retain their original behavior.

## Correctness and provenance

Four new native tests compare the active pipeline selector against a complete
frozen immediate-parent planner/cache/call, independent u64 vertical sums and
the CPU image implementation. Raw fixtures include all 1..=4 channel layouts,
five borders, 3/5/7 vertical kernels, mixed Q15/Q8 axes, tiny/narrow/odd images,
extreme sums/digit carries and nonzero ignored constant components. Every RGBA
output component is checked. Complete parent/CPU calls cover the same border/
kernel/channel families and large 1025x1027 images with grid-stride work.
Receipts and metadata match. Existing ROI/metadata and full Gaussian fixtures
also run through the changed public implementation.

Additional checks verify unchanged Q8/mono pipeline identity, lazy private
cache selection, rejection before cache/allocation, exact two-pass/query counts,
fresh/warm/zero-budget scratch and texture behavior, concurrent cold-cache
initialization and independent retained outputs after pool clear/reuse with a
different input. The frozen reference changes names, cache injection and test
visibility only; orphaned public-wrapper documentation is removed. Its original
shaders are shared only after independent source identity checks.

GAUSSIAN_ROWS_RAW: verified 174291200 physical RGBA bytes across 1208 paired parent/u64 fixtures, full u16/23-bit inputs, dirty inactive halves, all borders and one-group stride.

GAUSSIAN_ROWS_PUBLIC: verified 79164450 complete parent/CPU bytes, both scratch layouts, mixed axes, receipts and metadata.

There are 565 unchanged original GPU/image source files and 577 source/manifest/executable SHA-256 values. The finalized executable, GPU/image sources, manifests and resolved local Cargo.lock are identical before/after both surveys and broad validation. Cargo.lock is intentionally ignored in this library workspace; its local resolved bytes are hashed. All original shaders remain byte-identical, and both final helper shader bodies match the explored candidates. Repository/PR text privacy scans find zero forbidden adapter-name matches. Raw local receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 155 |
| voxel | 213 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 410 |

Clippy with all features/targets and warnings denied, CPU-only check, minimal/CI-feature WASM checks with warnings denied, workspace formatting and diff checks pass. Hardware checks include ignored tests and exclude explicit benchmarks only.

## Exploratory GPU comparison

Before production changes, an optimized standalone probe loaded the parent,
two-lane and four-lane shader text at runtime. Each 1920x1080 raw input contains
deterministic arbitrary 23-bit horizontal sums, with three warmup rounds and
30 measured rounds. Variant order rotates so every variant occupies each
position ten times. All three physical RGBA outputs match byte for byte.
Only pass timestamps are measured, with no complete-call speed inference.
There is one exploratory run without controlled workload isolation, and no
confidence intervals are inferred from these sketch medians. Every case is
retained below, including the slower one-channel candidates that are not used.
Two-channel work selects the two-lane candidate; three/four-channel work selects
the four-lane candidate. Values are median milliseconds.

| Channels | Vertical kernel | Border (2=Reflect101, 4=Constant) | Parent | Two lanes | Four lanes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 2 | 0.193424 | 0.230800 | 0.268816 |
| 1 | 7 | 4 | 0.193456 | 0.231072 | 0.268912 |
| 2 | 3 | 2 | 0.198464 | 0.156832 | 0.179440 |
| 2 | 3 | 4 | 0.198624 | 0.157200 | 0.179872 |
| 2 | 5 | 2 | 0.309680 | 0.216848 | 0.243824 |
| 2 | 5 | 4 | 0.310192 | 0.217072 | 0.243808 |
| 2 | 7 | 2 | 0.450688 | 0.270816 | 0.301744 |
| 2 | 7 | 4 | 0.447680 | 0.268656 | 0.298960 |
| 3 | 3 | 2 | 0.350928 | 0.266768 | 0.200656 |
| 3 | 3 | 4 | 0.351360 | 0.267168 | 0.201232 |
| 3 | 5 | 2 | 0.706128 | 0.500192 | 0.272336 |
| 3 | 5 | 4 | 0.552608 | 0.391952 | 0.220704 |
| 3 | 7 | 2 | 0.798368 | 0.568464 | 0.292080 |
| 3 | 7 | 4 | 0.801184 | 0.571200 | 0.293216 |
| 4 | 3 | 2 | 0.561984 | 0.343312 | 0.202192 |
| 4 | 3 | 4 | 0.566272 | 0.345792 | 0.203360 |
| 4 | 5 | 2 | 2.121552 | 1.100416 | 0.847840 |
| 4 | 5 | 4 | 2.114960 | 1.098176 | 0.846640 |
| 4 | 7 | 2 | 3.101520 | 1.849984 | 0.963152 |
| 4 | 7 | 4 | 3.102048 | 1.848848 | 0.962480 |

## Final paired measurement method

Two consecutive surveys use the same finalized release executable and one
headless timestamp-enabled runtime per survey. Each has 13 cases and three
routes, three warmup pairs and 30 measured parent/current pairs. Each pair is
recorded parent-first and current-first 15 times. Resident/full variants rotate
through all four positions; raw pass order alternates. Inputs, CPU expectations,
pipeline creation and result checks/destruction/recycling are outside timing.
All timed results are checked. No builds or tests overlap timed surveys; no
case is filtered or retried. Both affected and original-shader controls remain.

- `pass`: vertical GPU timestamps only, using the same horizontal input prepared
  by the unchanged horizontal shader outside timing; complete physical output
  is checked against the CPU result afterward.
- `resident`: completed public resident Gaussian call, including host planning,
  resource setup and both passes; explicit idle wait completes the GPU work.
  Payload upload/readback for checking is excluded.
- `full`: completed upload, public Gaussian call and payload readback, including
  host transfer preparation. This is the transfer-inclusive diagnostic.

The frozen complete parent shares only independently unchanged dependencies;
its pre-created parent cache replaces the original runtime cache lookup.
All resource descriptors, pass recording and receipt construction otherwise
retain the original code. Three warmup pairs exclude compilation for both
variants. Results apply to these warmed shared-host comparisons only.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times using seed
`2026101803 + 1000*survey + 10*case_index + route_index` (zero-based indices)
and sorted draws 249/9749. Entirely above/below one is favorable/adverse;
otherwise inconclusive. There is no multiplicity correction. Pre/post host
observations do not establish isolation during every pair or portable speed.

| Route | All 26 comparisons | Affected Q15 multi-channel 20 comparisons |
| --- | --- | --- |
| pass | 20 favorable / 6 inconclusive / 0 adverse | 20 favorable / 0 inconclusive / 0 adverse |
| resident | 17 favorable / 8 inconclusive / 1 adverse | 17 favorable / 3 inconclusive / 0 adverse |
| full | 12 favorable / 13 inconclusive / 1 adverse | 12 favorable / 8 inconclusive / 0 adverse |

All 20 affected pass comparisons favor the shared-row variants, with paired ratios 1.118–3.530. Affected resident calls have 17 favorable and three inconclusive comparisons; transfer-inclusive calls have 12 favorable and eight inconclusive comparisons. The favorable full-call ratios span 1.029–1.296. Both 3840x2160 three-channel 7x7 full-call comparisons favor current, at 1.097 and 1.088. Smaller images and short vertical kernels retain inconclusive full-call results. These observations support the warmed multi-channel path without establishing improvement for every complete image call.

Two original-program control comparisons are adverse: survey 1, 1920x1080 one-channel 7x7 resident, ratio 0.971411 (95% interval 0.940097–0.999021); survey 2, 1920x1080 one-channel 5x5 full, ratio 0.983171 (0.968458–0.998101). Their GPU programs are byte-identical; host selection work and shared-host variation are not separated by this survey, so the cause is unresolved. No control comparison is dropped.

Published observations exclude process identities:

- Survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=vertical_pass_resident_and_full wall_order=balanced checks_drop_recycle_excluded=true`.
  pre: 1 busy processes above 0.25 CPU cores, estimated cores [0.99]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [0.29]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
- Survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=vertical_pass_resident_and_full wall_order=balanced checks_drop_recycle_excluded=true`.
  pre: 1 busy processes above 0.25 CPU cores, estimated cores [0.32]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [0.26]; GPU/memory-controller utilization samples [[[0, 0]], [[0, 0]]]; observed quiet=False.

All 78 final comparisons follow (times in milliseconds):

| Survey | Image | C | Kernel | Constant | Route | Parent median | Current median | Paired ratio | 95% interval | Result |
| ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | 16x17 | 4 | 7x7 | False | pass | 0.008672 | 0.006416 | 1.341794 | 1.285462–1.400261 | favorable |
| 1 | 16x17 | 4 | 7x7 | False | resident | 0.264600 | 0.266350 | 0.971772 | 0.893379–1.050347 | inconclusive |
| 1 | 16x17 | 4 | 7x7 | False | full | 1.276400 | 1.278650 | 1.022696 | 0.991303–1.058030 | inconclusive |
| 1 | 257x259 | 3 | 5x7 | False | pass | 0.039024 | 0.020720 | 1.893440 | 1.861800–1.925461 | favorable |
| 1 | 257x259 | 3 | 5x7 | False | resident | 0.314200 | 0.307550 | 1.044268 | 1.005010–1.085919 | favorable |
| 1 | 257x259 | 3 | 5x7 | False | full | 1.530850 | 1.465500 | 1.018449 | 0.976972–1.060169 | inconclusive |
| 1 | 1920x1080 | 1 | 5x5 | False | pass | 0.162336 | 0.162496 | 0.999705 | 0.994155–1.005260 | inconclusive |
| 1 | 1920x1080 | 1 | 5x5 | False | resident | 0.708800 | 0.722550 | 0.997567 | 0.969254–1.027158 | inconclusive |
| 1 | 1920x1080 | 1 | 5x5 | False | full | 7.498500 | 7.384350 | 1.005159 | 0.980506–1.033547 | inconclusive |
| 1 | 1920x1080 | 4 | 5x5 | False | pass | 0.421184 | 0.422352 | 0.998560 | 0.995884–1.001193 | inconclusive |
| 1 | 1920x1080 | 4 | 5x5 | False | resident | 0.974100 | 0.985300 | 0.974430 | 0.917971–1.023819 | inconclusive |
| 1 | 1920x1080 | 4 | 5x5 | False | full | 5.553750 | 5.540200 | 1.000997 | 0.971201–1.025617 | inconclusive |
| 1 | 1920x1080 | 1 | 7x7 | False | pass | 0.515056 | 0.516992 | 0.997124 | 0.990586–1.003488 | inconclusive |
| 1 | 1920x1080 | 1 | 7x7 | False | resident | 1.488100 | 1.490550 | 0.971411 | 0.940097–0.999021 | adverse |
| 1 | 1920x1080 | 1 | 7x7 | False | full | 8.964350 | 8.978300 | 0.990460 | 0.976047–1.004024 | inconclusive |
| 1 | 1920x1080 | 2 | 7x3 | False | pass | 0.258128 | 0.209536 | 1.230145 | 1.225080–1.235111 | favorable |
| 1 | 1920x1080 | 2 | 7x3 | False | resident | 1.047000 | 0.989050 | 1.033160 | 1.001019–1.065035 | favorable |
| 1 | 1920x1080 | 2 | 7x3 | False | full | 7.504750 | 7.682800 | 0.988151 | 0.962242–1.012396 | inconclusive |
| 1 | 1920x1080 | 2 | 7x7 | False | pass | 1.011984 | 0.904544 | 1.118133 | 1.114139–1.122176 | favorable |
| 1 | 1920x1080 | 2 | 7x7 | False | resident | 1.427750 | 1.099050 | 1.249939 | 1.211075–1.285004 | favorable |
| 1 | 1920x1080 | 2 | 7x7 | False | full | 8.093850 | 7.754850 | 1.028954 | 1.012824–1.044824 | favorable |
| 1 | 1920x1080 | 3 | 7x5 | False | pass | 0.953872 | 0.368512 | 2.588180 | 2.584406–2.592083 | favorable |
| 1 | 1920x1080 | 3 | 7x5 | False | resident | 1.855250 | 1.284650 | 1.419323 | 1.333837–1.478103 | favorable |
| 1 | 1920x1080 | 3 | 7x5 | False | full | 8.480400 | 7.860100 | 1.079679 | 1.059831–1.099430 | favorable |
| 1 | 1920x1080 | 3 | 7x7 | False | pass | 1.438272 | 0.526112 | 2.733270 | 2.729509–2.737070 | favorable |
| 1 | 1920x1080 | 3 | 7x7 | False | resident | 2.449950 | 1.487750 | 1.437182 | 1.332853–1.544482 | favorable |
| 1 | 1920x1080 | 3 | 7x7 | False | full | 9.783050 | 9.320450 | 1.063589 | 1.035225–1.090022 | favorable |
| 1 | 1920x1080 | 4 | 7x3 | False | pass | 1.011792 | 0.350832 | 2.881740 | 2.876989–2.886267 | favorable |
| 1 | 1920x1080 | 4 | 7x3 | False | resident | 2.179100 | 1.495900 | 1.453680 | 1.431007–1.477663 | favorable |
| 1 | 1920x1080 | 4 | 7x3 | False | full | 6.693950 | 6.023250 | 1.120356 | 1.111334–1.129185 | favorable |
| 1 | 1920x1080 | 4 | 7x7 | False | pass | 2.254656 | 0.641104 | 3.530468 | 3.522330–3.538711 | favorable |
| 1 | 1920x1080 | 4 | 7x7 | False | resident | 3.953900 | 1.991800 | 1.993072 | 1.976595–2.010801 | favorable |
| 1 | 1920x1080 | 4 | 7x7 | False | full | 8.675350 | 6.664000 | 1.296146 | 1.280877–1.310621 | favorable |
| 1 | 3840x2160 | 3 | 7x7 | False | pass | 5.381344 | 2.600400 | 2.045763 | 2.000037–2.079791 | favorable |
| 1 | 3840x2160 | 3 | 7x7 | False | resident | 12.401600 | 10.057000 | 1.255021 | 1.180574–1.335219 | favorable |
| 1 | 3840x2160 | 3 | 7x7 | False | full | 33.656450 | 30.984000 | 1.096971 | 1.081058–1.115201 | favorable |
| 1 | 1025x257 | 3 | 3x7 | True | pass | 0.110400 | 0.046368 | 2.399484 | 2.376366–2.422696 | favorable |
| 1 | 1025x257 | 3 | 3x7 | True | resident | 0.538100 | 0.472850 | 1.133605 | 1.082193–1.187808 | favorable |
| 1 | 1025x257 | 3 | 3x7 | True | full | 6.278100 | 6.389100 | 0.983233 | 0.948551–1.018930 | inconclusive |
| 2 | 16x17 | 4 | 7x7 | False | pass | 0.008736 | 0.006608 | 1.338019 | 1.281115–1.397226 | favorable |
| 2 | 16x17 | 4 | 7x7 | False | resident | 0.261250 | 0.274300 | 0.937544 | 0.848161–1.017214 | inconclusive |
| 2 | 16x17 | 4 | 7x7 | False | full | 1.130250 | 1.148950 | 1.001070 | 0.973710–1.031734 | inconclusive |
| 2 | 257x259 | 3 | 5x7 | False | pass | 0.038896 | 0.020768 | 1.860434 | 1.828790–1.893091 | favorable |
| 2 | 257x259 | 3 | 5x7 | False | resident | 0.255650 | 0.229950 | 1.074459 | 1.027210–1.119938 | favorable |
| 2 | 257x259 | 3 | 5x7 | False | full | 1.213400 | 1.192300 | 1.003434 | 0.980174–1.028757 | inconclusive |
| 2 | 1920x1080 | 1 | 5x5 | False | pass | 0.162560 | 0.162512 | 1.000520 | 0.995834–1.005306 | inconclusive |
| 2 | 1920x1080 | 1 | 5x5 | False | resident | 0.672150 | 0.683050 | 0.995631 | 0.967127–1.026457 | inconclusive |
| 2 | 1920x1080 | 1 | 5x5 | False | full | 6.160750 | 6.268100 | 0.983171 | 0.968458–0.998101 | adverse |
| 2 | 1920x1080 | 4 | 5x5 | False | pass | 0.417648 | 0.417552 | 0.998969 | 0.995821–1.002059 | inconclusive |
| 2 | 1920x1080 | 4 | 5x5 | False | resident | 0.934900 | 0.960250 | 0.975890 | 0.949046–1.003118 | inconclusive |
| 2 | 1920x1080 | 4 | 5x5 | False | full | 5.169500 | 5.073800 | 1.041322 | 0.975748–1.116207 | inconclusive |
| 2 | 1920x1080 | 1 | 7x7 | False | pass | 0.520400 | 0.520800 | 0.996764 | 0.991534–1.002408 | inconclusive |
| 2 | 1920x1080 | 1 | 7x7 | False | resident | 0.990750 | 0.983550 | 1.046672 | 0.997052–1.102701 | inconclusive |
| 2 | 1920x1080 | 1 | 7x7 | False | full | 6.990650 | 6.977650 | 0.985468 | 0.934239–1.037681 | inconclusive |
| 2 | 1920x1080 | 2 | 7x3 | False | pass | 0.258128 | 0.210000 | 1.226915 | 1.218843–1.234311 | favorable |
| 2 | 1920x1080 | 2 | 7x3 | False | resident | 1.075400 | 1.045250 | 1.018322 | 0.991474–1.048027 | inconclusive |
| 2 | 1920x1080 | 2 | 7x3 | False | full | 7.107750 | 7.130150 | 1.004413 | 0.977304–1.030168 | inconclusive |
| 2 | 1920x1080 | 2 | 7x7 | False | pass | 0.632800 | 0.362656 | 1.749491 | 1.745125–1.753802 | favorable |
| 2 | 1920x1080 | 2 | 7x7 | False | resident | 1.414850 | 1.142300 | 1.232723 | 1.202539–1.263281 | favorable |
| 2 | 1920x1080 | 2 | 7x7 | False | full | 7.090250 | 6.907750 | 1.034573 | 1.021507–1.048306 | favorable |
| 2 | 1920x1080 | 3 | 7x5 | False | pass | 0.966128 | 0.371936 | 2.597376 | 2.594201–2.600497 | favorable |
| 2 | 1920x1080 | 3 | 7x5 | False | resident | 1.981150 | 1.379200 | 1.248541 | 1.156041–1.339184 | favorable |
| 2 | 1920x1080 | 3 | 7x5 | False | full | 8.142750 | 7.388850 | 1.069046 | 1.045826–1.094515 | favorable |
| 2 | 1920x1080 | 3 | 7x7 | False | pass | 1.335808 | 0.489712 | 2.730482 | 2.725930–2.734991 | favorable |
| 2 | 1920x1080 | 3 | 7x7 | False | resident | 2.253750 | 1.424050 | 1.582407 | 1.552247–1.614354 | favorable |
| 2 | 1920x1080 | 3 | 7x7 | False | full | 8.189650 | 7.443050 | 1.101669 | 1.085670–1.118402 | favorable |
| 2 | 1920x1080 | 4 | 7x3 | False | pass | 0.868464 | 0.303616 | 2.859307 | 2.854362–2.864141 | favorable |
| 2 | 1920x1080 | 4 | 7x3 | False | resident | 2.041450 | 1.493150 | 1.365595 | 1.339775–1.392436 | favorable |
| 2 | 1920x1080 | 4 | 7x3 | False | full | 6.278850 | 5.781400 | 1.096147 | 1.084739–1.107446 | favorable |
| 2 | 1920x1080 | 4 | 7x7 | False | pass | 1.682992 | 0.480480 | 3.529904 | 3.522276–3.537098 | favorable |
| 2 | 1920x1080 | 4 | 7x7 | False | resident | 3.275450 | 1.666050 | 1.988856 | 1.960485–2.021899 | favorable |
| 2 | 1920x1080 | 4 | 7x7 | False | full | 7.625050 | 5.970050 | 1.270915 | 1.259235–1.282683 | favorable |
| 2 | 3840x2160 | 3 | 7x7 | False | pass | 5.211120 | 2.606704 | 1.990934 | 1.962192–2.019012 | favorable |
| 2 | 3840x2160 | 3 | 7x7 | False | resident | 11.755900 | 8.877750 | 1.299518 | 1.234525–1.367317 | favorable |
| 2 | 3840x2160 | 3 | 7x7 | False | full | 31.925900 | 29.378200 | 1.088023 | 1.075171–1.102341 | favorable |
| 2 | 1025x257 | 3 | 3x7 | True | pass | 0.110192 | 0.045968 | 2.411337 | 2.387508–2.436199 | favorable |
| 2 | 1025x257 | 3 | 3x7 | True | resident | 0.532600 | 0.452300 | 1.145996 | 1.101277–1.192782 | favorable |
| 2 | 1025x257 | 3 | 3x7 | True | full | 6.417850 | 6.357250 | 0.995994 | 0.966329–1.028477 | inconclusive |

Original-shader controls can fluctuate even with identical GPU computation. Pass-only gains do not establish full-call gains; all adverse and inconclusive results remain visible. Cold compilation, pipeline-memory cost and shared-host contention limit the interpretation above.
