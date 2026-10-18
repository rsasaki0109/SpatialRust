# Reusing adjacent GPU box blur sums

Date: 2026-10-18. Named slice: adjacent GPU box blur windows. Immediate parent:
`ce6f3a57f61ffda95f5b09769878eaf47d31b4b2` (merged PR #269).

## Change and exact arithmetic

The existing packed horizontal pass owns two adjacent logical pixels per word.
It previously summed both complete windows separately. For two pixels on the
same row and an odd kernel width greater than one, the second sum now uses
`H(x+1,y) = H(x,y) - S(x-r,y) + S(x+r+1,y)`, where S applies the original
replicate or constant-zero border and r is the kernel radius. Two sample loads
replace a repeated clipped window loop and its edge mass calculations. These
are source operations, not measured physical memory transactions.

The removed sample occurs in the first complete sum, or is zero, so unsigned
subtraction cannot underflow. Each result is a sum of at most 257 byte samples:
`0 <= H <= 257*255 = 65535`. Neither intermediate addition nor the two packed
u16 halves can overflow or carry into the other pixel. Only same-row pairs
take this path. At odd-width row crossings the original complete sum is used;
an unpaired final pixel retains a zero upper half. Width-one kernels use their
exact source byte directly, avoiding all window/edge calculations.

The outgoing coordinate is at least -128. For a reused pair x <= width-2, so
the incoming coordinate `x+r+1 <= width-1+r` retains the existing planner's
signed bound. All dimensions, arithmetic admission and original capacity/error
precedence remain. The first multi-tap sum otherwise retains its exact loop
and edge masses; the vertical integer sum and once-only half-up division are
unchanged. No approximate arithmetic, floating-point pixel math, extra scratch
or new device feature is introduced.

Only the existing packed horizontal WGSL changes in production. The host
planner/encoder/API, serial/unpacked/vertical shaders, uniforms/layouts, caches,
one/two-pass policy, buffer/texture requests, ownership and explicit payload
transfers/receipts are unchanged. Packed selection remains separable width
<=257, with the original area/sample cutoffs. Pipeline variant count is the
same; driver compilation cost and physical memory were not measured. Timings
below apply to warmed calls, not cold-runtime initialization.

## Correctness and provenance

Four new native tests compare the actual production shader against a complete
immediate-parent planner/cache/encoder/call and the original packed horizontal
WGSL. Independent u64 row/column prefixes and small brute-force 2D windows
check exact mathematical outputs, rather than duplicating the sliding update.

Raw packed probes inspect both halves, dirty guards, nonzero unused source
channels, one-tap and 3/5/9/31/255/257 widths, both borders, maximal u16 sums,
one/two-pixel strips, even/odd row crossings, odd tails and forced one-group
grid-stride work through 1025x1025 images. Complete calls cover serial and
unpacked controls, the 257/259 boundary, mixed axes, a valid 4099x4099 kernel,
receipts, metadata and every physical RGBA component. Existing ROI and transfer
fixtures also run through the public implementation.

Additional checks compare original pass/query counts, scratch/texture request
counts, warmed reuse, tiny/zero budgets, errors before cache creation, concurrent
first use and retained outputs after pool clear and changed-input reuse. The
native complete reference changes names, visibility and cache injection only;
it aliases the unchanged public border enum and freezes the one changed shader.
Its other dependencies are shared only after independent source identity checks.

BOX_WINDOW_RAW: verified 28026984 parent/current/u64 horizontal/guard words across 344 cases; full u16 mass, nonzero unused source channels, both borders, one-tap/row crossings/odd tails and one-group stride.

BOX_WINDOW_FULL: verified 34968936 complete parent/CPU/physical output bytes across 244 cases; unchanged packed/serial/wide policy, round-half-up results, receipts and metadata.

BOX_WINDOW_RESOURCES: verified original pipeline/pass/query counts, scratch/texture allocation requests and warm reuse, tiny/zero budgets and pre-allocation errors.

BOX_WINDOW_CONCURRENT: verified simultaneous packed/one-tap/wide initialization and independent retained outputs after pool clear and changed-input reuse.

There are 576 unchanged original GPU/image source files and 588 source/manifest/executable SHA-256 values. The finalized release unit executable, GPU/image sources, manifests and resolved local ignored Cargo.lock are identical before/after both timed surveys and broad validation. All other original shaders and all production Rust host code are unchanged. The final production shader body matches the explored candidate; the frozen packed shader matches the immediate parent. Repository/PR privacy scans find zero forbidden adapter-name matches; raw local receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 159 |
| voxel | 218 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 419 |

All-features/all-targets Clippy with warnings denied, CPU-only check, minimal/CI-feature WASM checks with warnings denied, workspace formatting and diff checks pass. Hardware checks include ignored correctness tests and exclude explicit benchmarks.

## Exploratory GPU comparisons

An optimized standalone probe loaded parent/candidate shader text at runtime.
Each prototype has 18 raw cases, three warmup pairs and 30 measured pairs in
alternating order. All packed words, dirty guards and odd-tail upper halves
match the parent. Only horizontal GPU timestamps are measured, with no
complete-call or portable-speed inference. Shared-host isolation was not
controlled; these sketch medians have no inferred confidence intervals.

The first prototype retains complete one-tap sums and has slightly higher
one-tap medians. The final prototype adds the exact one-tap direct load, then
rechecks every original case. Both code phases and every result are retained
separately below; their values are not pooled into final survey statistics.

| Phase | Image | Kernel width | Border (0=replicate, 1=zero) | Parent median ms | Candidate median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| first | 1920x1080 | 1 | 0 | 0.115520 | 0.119632 |
| first | 1920x1080 | 1 | 1 | 0.101328 | 0.102336 |
| first | 1920x1080 | 3 | 0 | 0.153728 | 0.136848 |
| first | 1920x1080 | 3 | 1 | 0.136224 | 0.127920 |
| first | 1920x1080 | 9 | 0 | 0.218448 | 0.165968 |
| first | 1920x1080 | 9 | 1 | 0.194832 | 0.155808 |
| first | 1920x1080 | 31 | 0 | 0.555712 | 0.324128 |
| first | 1920x1080 | 31 | 1 | 0.532576 | 0.317344 |
| first | 1920x1080 | 257 | 0 | 3.439792 | 1.759968 |
| first | 1920x1080 | 257 | 1 | 3.021040 | 1.549488 |
| first | 257x259 | 9 | 0 | 0.009952 | 0.008896 |
| first | 257x259 | 9 | 1 | 0.009392 | 0.008176 |
| first | 257x259 | 257 | 0 | 0.101520 | 0.057200 |
| first | 257x259 | 257 | 1 | 0.100240 | 0.056768 |
| first | 17x19 | 31 | 0 | 0.005376 | 0.005504 |
| first | 17x19 | 31 | 1 | 0.005376 | 0.005376 |
| first | 1x521 | 257 | 0 | 0.003936 | 0.004272 |
| first | 1x521 | 257 | 1 | 0.004320 | 0.004192 |
| final | 1920x1080 | 1 | 0 | 0.116624 | 0.069808 |
| final | 1920x1080 | 1 | 1 | 0.100240 | 0.069040 |
| final | 1920x1080 | 3 | 0 | 0.153168 | 0.100512 |
| final | 1920x1080 | 3 | 1 | 0.135232 | 0.089616 |
| final | 1920x1080 | 9 | 0 | 0.216352 | 0.138800 |
| final | 1920x1080 | 9 | 1 | 0.193376 | 0.125856 |
| final | 1920x1080 | 31 | 0 | 0.549920 | 0.303232 |
| final | 1920x1080 | 31 | 1 | 0.527488 | 0.293520 |
| final | 1920x1080 | 257 | 0 | 3.833792 | 1.948688 |
| final | 1920x1080 | 257 | 1 | 3.014976 | 1.536896 |
| final | 257x259 | 9 | 0 | 0.009840 | 0.008720 |
| final | 257x259 | 9 | 1 | 0.009632 | 0.007744 |
| final | 257x259 | 257 | 0 | 0.100304 | 0.056208 |
| final | 257x259 | 257 | 1 | 0.099632 | 0.055776 |
| final | 17x19 | 31 | 0 | 0.005776 | 0.005136 |
| final | 17x19 | 31 | 1 | 0.005296 | 0.005056 |
| final | 1x521 | 257 | 0 | 0.003936 | 0.003744 |
| final | 1x521 | 257 | 1 | 0.003936 | 0.004000 |

## Final paired measurement method

Two consecutive surveys use the same finalized release executable. Each uses
one headless timestamp-enabled runtime, 13 cases, three warmup pairs and 30
measured parent/current pairs. Resident/full variants rotate through all four
positions; each pair is parent-first/current-first 15 times. Timestamp pairs
alternate in the same balanced directions. Input creation, CPU expectations,
pipeline compilation and checks/destruction/recycling are outside timing.
No builds/tests initiated by this workflow overlap either timed survey; no
case is retried or filtered. Other shared-host work is not controlled.

- `pass`: first-stage GPU timestamp from a completed profiled public call.
  For separable calls this is the horizontal stage; for the serial control it
  is the original single whole-filter pass. Every stage still runs, all query
  bytes/counts are checked, and complete outputs are checked afterward.
- `resident`: completed public resident box blur call plus explicit idle
  wait, excluding payload upload/readback for checking.
- `full`: upload, public box blur and readback, including host transfer
  preparation. This is the transfer-inclusive diagnostic.

The complete parent uses a pre-created private parent cache in place of the
original runtime cache lookup. All planning/resources/pass recording/receipt
construction otherwise retain the original body. Three warmup pairs exclude
pipeline compilation; no cold-runtime latency or compiler-cost claim is made.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times with seed
`2026101805 + 1000*survey + 10*case_index + route_index` (zero-based case/route)
and sorted draws 249/9749. Entirely above/below one is favorable/adverse,
otherwise inconclusive. There is no multiplicity correction. Host pre/post
samples cannot establish isolation during every pair or application speed.

| Route | All 26 comparisons | Packed shader 22 comparisons |
| --- | --- | --- |
| pass | 21 favorable / 4 inconclusive / 1 adverse | 21 favorable / 1 inconclusive / 0 adverse |
| resident | 16 favorable / 10 inconclusive / 0 adverse | 15 favorable / 7 inconclusive / 0 adverse |
| full | 5 favorable / 21 inconclusive / 0 adverse | 5 favorable / 17 inconclusive / 0 adverse |

The packed set includes one-tap direct loads and the width-one row-crossing fallback, in addition to reused same-row windows. Original-program serial/unpacked controls, all adverse/inconclusive rows and all first-stage/full-call differences remain visible. First-stage gains do not establish full-call gains.

Packed first-stage results are 21 favorable and one inconclusive, with favorable ratios 1.068–1.963. The inconclusive first-stage case is the width-one fallback in survey 1. Packed resident results are 15 favorable and seven inconclusive; full calls are five favorable and 17 inconclusive, with favorable full ratios 1.037–1.231. Both 1920x1080 257x3 full calls favor current (1.231 and 1.230), as do both 31x31 full calls (1.079 and 1.061). All four 3840x2160 full-call comparisons remain inconclusive. No final full-call interval is wholly adverse.

One unchanged unpacked-shader control is adverse: survey 1, 257x259 with 259x3, first-stage ratio 0.998385 (95% interval 0.997356–0.999476). The same unchanged control also favors current for survey 2 resident calls, ratio 1.072699 (1.024473–1.127774). Their shader text is identical; this survey does not resolve the cause of either control shift. Both remain visible and reinforce the shared-host limits.

Published host observations exclude process identities:

- Survey 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.
  pre: 2 busy processes above 0.25 CPU cores, estimated cores [1.36, 0.56]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [0.36]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.
- Survey 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.
  pre: 0 busy processes above 0.25 CPU cores, estimated cores []; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=True.
  post: 1 busy processes above 0.25 CPU cores, estimated cores [0.89]; GPU/memory-controller samples [[[0, 0]], [[0, 0]]]; observed quiet=False.

All 78 comparisons follow (times in milliseconds):

| Survey | Image | Kernel | Zero border | Packed | Route | Parent median | Current median | Paired ratio | 95% interval | Result |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | 16x17 | 3x3 | False | False | pass | 0.005440 | 0.005456 | 0.995559 | 0.989077–1.001722 | inconclusive |
| 1 | 16x17 | 3x3 | False | False | resident | 0.220450 | 0.208050 | 0.987532 | 0.940422–1.035883 | inconclusive |
| 1 | 16x17 | 3x3 | False | False | full | 1.123750 | 1.127150 | 0.982862 | 0.953072–1.011890 | inconclusive |
| 1 | 257x259 | 9x9 | False | True | pass | 0.013312 | 0.010960 | 1.219756 | 1.208730–1.230517 | favorable |
| 1 | 257x259 | 9x9 | False | True | resident | 0.232550 | 0.234100 | 1.008237 | 0.986667–1.028863 | inconclusive |
| 1 | 257x259 | 9x9 | False | True | full | 1.215700 | 1.186750 | 1.019239 | 0.994240–1.046105 | inconclusive |
| 1 | 1920x1080 | 3x27 | False | True | pass | 0.158784 | 0.101760 | 1.560499 | 1.554713–1.566649 | favorable |
| 1 | 1920x1080 | 3x27 | False | True | resident | 0.815700 | 0.756400 | 1.074443 | 1.053284–1.096016 | favorable |
| 1 | 1920x1080 | 3x27 | False | True | full | 6.694850 | 6.546650 | 1.015502 | 0.982105–1.047123 | inconclusive |
| 1 | 1920x1080 | 9x9 | False | True | pass | 0.422096 | 0.322496 | 1.378354 | 1.324028–1.451864 | favorable |
| 1 | 1920x1080 | 9x9 | False | True | resident | 0.866100 | 0.799400 | 1.098826 | 1.068533–1.128274 | favorable |
| 1 | 1920x1080 | 9x9 | False | True | full | 6.865050 | 6.767050 | 0.999081 | 0.964377–1.033724 | inconclusive |
| 1 | 1920x1080 | 31x31 | False | True | pass | 0.772544 | 0.422752 | 1.826809 | 1.825045–1.828539 | favorable |
| 1 | 1920x1080 | 31x31 | False | True | resident | 1.709250 | 1.367500 | 1.253666 | 1.237147–1.269808 | favorable |
| 1 | 1920x1080 | 31x31 | False | True | full | 7.783500 | 7.447650 | 1.078673 | 1.045706–1.120694 | favorable |
| 1 | 1920x1080 | 257x3 | False | True | pass | 3.032560 | 1.545408 | 1.962053 | 1.961127–1.962953 | favorable |
| 1 | 1920x1080 | 257x3 | False | True | resident | 3.689500 | 2.165900 | 1.750376 | 1.713254–1.797698 | favorable |
| 1 | 1920x1080 | 257x3 | False | True | full | 9.864050 | 8.143400 | 1.231128 | 1.204426–1.262993 | favorable |
| 1 | 1920x1080 | 1x81 | False | True | pass | 0.095136 | 0.057184 | 1.661632 | 1.654296–1.668847 | favorable |
| 1 | 1920x1080 | 1x81 | False | True | resident | 1.053000 | 1.024500 | 1.029889 | 1.018244–1.042368 | favorable |
| 1 | 1920x1080 | 1x81 | False | True | full | 7.242850 | 7.081400 | 0.999196 | 0.969397–1.023211 | inconclusive |
| 1 | 257x259 | 259x3 | False | False | pass | 0.128736 | 0.128944 | 0.998385 | 0.997356–0.999476 | adverse |
| 1 | 257x259 | 259x3 | False | False | resident | 0.376500 | 0.384950 | 0.980964 | 0.930636–1.032819 | inconclusive |
| 1 | 257x259 | 259x3 | False | False | full | 2.115100 | 2.104000 | 0.984374 | 0.939987–1.027364 | inconclusive |
| 1 | 3840x2160 | 9x9 | False | True | pass | 0.967584 | 0.629040 | 1.538082 | 1.537148–1.538922 | favorable |
| 1 | 3840x2160 | 9x9 | False | True | resident | 2.656900 | 2.256100 | 1.187925 | 1.139634–1.257497 | favorable |
| 1 | 3840x2160 | 9x9 | False | True | full | 23.810700 | 23.607850 | 1.060007 | 0.977513–1.146597 | inconclusive |
| 1 | 3840x2160 | 31x31 | False | True | pass | 1.781808 | 0.991312 | 1.797394 | 1.796684–1.798098 | favorable |
| 1 | 3840x2160 | 31x31 | False | True | resident | 4.188750 | 3.417700 | 1.232131 | 1.221216–1.243998 | favorable |
| 1 | 3840x2160 | 31x31 | False | True | full | 23.109050 | 22.044200 | 1.023821 | 0.962234–1.084373 | inconclusive |
| 1 | 1025x257 | 9x9 | True | True | pass | 0.041824 | 0.029408 | 1.415889 | 1.388777–1.441767 | favorable |
| 1 | 1025x257 | 9x9 | True | True | resident | 0.390000 | 0.399600 | 0.974046 | 0.928554–1.021346 | inconclusive |
| 1 | 1025x257 | 9x9 | True | True | full | 5.789300 | 5.825750 | 1.012290 | 0.972468–1.059517 | inconclusive |
| 1 | 17x19 | 31x31 | False | True | pass | 0.015552 | 0.014832 | 1.067863 | 1.042662–1.096627 | favorable |
| 1 | 17x19 | 31x31 | False | True | resident | 0.378100 | 0.367100 | 0.990130 | 0.910672–1.057618 | inconclusive |
| 1 | 17x19 | 31x31 | False | True | full | 5.305300 | 5.311700 | 1.008565 | 0.988312–1.029693 | inconclusive |
| 1 | 1x521 | 257x3 | False | True | pass | 0.019088 | 0.018928 | 1.023901 | 0.963681–1.077693 | inconclusive |
| 1 | 1x521 | 257x3 | False | True | resident | 0.594400 | 0.514350 | 1.096050 | 0.972981–1.243484 | inconclusive |
| 1 | 1x521 | 257x3 | False | True | full | 5.439000 | 5.529600 | 0.997327 | 0.970586–1.024615 | inconclusive |
| 2 | 16x17 | 3x3 | False | False | pass | 0.005472 | 0.005472 | 1.001969 | 0.994467–1.009317 | inconclusive |
| 2 | 16x17 | 3x3 | False | False | resident | 0.296700 | 0.291950 | 0.984089 | 0.914858–1.055311 | inconclusive |
| 2 | 16x17 | 3x3 | False | False | full | 1.488250 | 1.420650 | 1.014790 | 0.978367–1.050772 | inconclusive |
| 2 | 257x259 | 9x9 | False | True | pass | 0.013232 | 0.011152 | 1.187763 | 1.176948–1.199006 | favorable |
| 2 | 257x259 | 9x9 | False | True | resident | 0.301150 | 0.288000 | 1.047182 | 1.001601–1.096318 | favorable |
| 2 | 257x259 | 9x9 | False | True | full | 1.445450 | 1.463550 | 0.994952 | 0.969342–1.021267 | inconclusive |
| 2 | 1920x1080 | 3x27 | False | True | pass | 0.159504 | 0.101776 | 1.593580 | 1.570098–1.633693 | favorable |
| 2 | 1920x1080 | 3x27 | False | True | resident | 0.831750 | 0.781400 | 1.052301 | 1.038376–1.066061 | favorable |
| 2 | 1920x1080 | 3x27 | False | True | full | 7.262000 | 7.099250 | 1.037182 | 0.987424–1.092414 | inconclusive |
| 2 | 1920x1080 | 9x9 | False | True | pass | 0.438432 | 0.326304 | 1.338951 | 1.333047–1.344381 | favorable |
| 2 | 1920x1080 | 9x9 | False | True | resident | 1.429950 | 1.310050 | 1.083197 | 1.056742–1.108336 | favorable |
| 2 | 1920x1080 | 9x9 | False | True | full | 9.114600 | 8.825850 | 1.036737 | 1.013749–1.063196 | favorable |
| 2 | 1920x1080 | 31x31 | False | True | pass | 0.751264 | 0.411584 | 1.825072 | 1.823042–1.827085 | favorable |
| 2 | 1920x1080 | 31x31 | False | True | resident | 1.876400 | 1.624900 | 1.221182 | 1.190189–1.249847 | favorable |
| 2 | 1920x1080 | 31x31 | False | True | full | 9.067250 | 8.328650 | 1.061046 | 1.039066–1.086756 | favorable |
| 2 | 1920x1080 | 257x3 | False | True | pass | 3.034416 | 1.545808 | 1.962939 | 1.961908–1.963959 | favorable |
| 2 | 1920x1080 | 257x3 | False | True | resident | 3.742950 | 2.176150 | 1.743935 | 1.717362–1.782921 | favorable |
| 2 | 1920x1080 | 257x3 | False | True | full | 9.743600 | 8.094150 | 1.229903 | 1.191057–1.271033 | favorable |
| 2 | 1920x1080 | 1x81 | False | True | pass | 0.094704 | 0.057168 | 1.657885 | 1.651727–1.663598 | favorable |
| 2 | 1920x1080 | 1x81 | False | True | resident | 1.117400 | 1.059650 | 1.044851 | 1.030219–1.058037 | favorable |
| 2 | 1920x1080 | 1x81 | False | True | full | 7.056950 | 7.031250 | 0.990218 | 0.953634–1.026783 | inconclusive |
| 2 | 257x259 | 259x3 | False | False | pass | 0.133424 | 0.133152 | 1.003645 | 0.999079–1.010074 | inconclusive |
| 2 | 257x259 | 259x3 | False | False | resident | 0.454350 | 0.458200 | 1.072699 | 1.024473–1.127774 | favorable |
| 2 | 257x259 | 259x3 | False | False | full | 2.303300 | 2.371450 | 1.004078 | 0.940049–1.079357 | inconclusive |
| 2 | 3840x2160 | 9x9 | False | True | pass | 1.008352 | 0.652928 | 1.539311 | 1.527858–1.549951 | favorable |
| 2 | 3840x2160 | 9x9 | False | True | resident | 2.270350 | 1.928950 | 1.168937 | 1.154529–1.183294 | favorable |
| 2 | 3840x2160 | 9x9 | False | True | full | 23.302750 | 22.120250 | 1.006622 | 0.954158–1.061421 | inconclusive |
| 2 | 3840x2160 | 31x31 | False | True | pass | 1.781936 | 0.991584 | 1.797273 | 1.796558–1.797980 | favorable |
| 2 | 3840x2160 | 31x31 | False | True | resident | 4.244400 | 3.423400 | 1.237475 | 1.229230–1.245317 | favorable |
| 2 | 3840x2160 | 31x31 | False | True | full | 23.688700 | 22.768650 | 1.030368 | 0.984267–1.077732 | inconclusive |
| 2 | 1025x257 | 9x9 | True | True | pass | 0.045120 | 0.031296 | 1.438937 | 1.400530–1.489022 | favorable |
| 2 | 1025x257 | 9x9 | True | True | resident | 0.435000 | 0.416750 | 1.026739 | 0.980686–1.074543 | inconclusive |
| 2 | 1025x257 | 9x9 | True | True | full | 6.051850 | 6.082750 | 0.985106 | 0.950945–1.019431 | inconclusive |
| 2 | 17x19 | 31x31 | False | True | pass | 0.017536 | 0.016192 | 1.075700 | 1.060418–1.091190 | favorable |
| 2 | 17x19 | 31x31 | False | True | resident | 0.464050 | 0.427900 | 1.041868 | 0.983980–1.104351 | inconclusive |
| 2 | 17x19 | 31x31 | False | True | full | 5.619050 | 5.519850 | 1.028900 | 0.987405–1.073159 | inconclusive |
| 2 | 1x521 | 257x3 | False | True | pass | 0.022016 | 0.020416 | 1.067965 | 1.028789–1.111303 | favorable |
| 2 | 1x521 | 257x3 | False | True | resident | 0.979750 | 0.871750 | 1.055285 | 0.942790–1.185185 | inconclusive |
| 2 | 1x521 | 257x3 | False | True | full | 6.609250 | 6.459950 | 1.019709 | 0.959162–1.087636 | inconclusive |
