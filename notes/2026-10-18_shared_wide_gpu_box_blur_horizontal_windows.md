# Reusing adjacent wide GPU box blur horizontal windows

Date: 2026-10-18. Named slice: shared wide GPU box blur horizontal windows.
Immediate parent: `7f2aba9770f7ca656b2054ed338763362f82503f` (merged PR #274).

## Change and exact arithmetic

For unpacked separable box blur calls with at least 65,536 pixels, one invocation
owns two adjacent linear output indices. The first original scalar window is
computed without rounding, then the next same-row sum is
`H(x+1,y)=H(x,y)-I(x-r,y)+I(x+r+1,y)`, with the original replicate or zero border.
Pairs crossing an odd-width row boundary compute two original complete sums.
An odd final pixel writes once. Each result is a complete u32 horizontal sum;
there is no packing, truncation, approximate arithmetic or cross-channel carry.
The original vertical stage performs the same final half-up rounding.

The outgoing byte belongs to the first nonnegative window or is zero, so the
subtraction cannot underflow. Both results obey `sum <= kernel_width*255`,
bounded by the original full-area/rounding admission. For a same-row second
pixel x<=width-2, the incoming coordinate is at most width-1+r, already bounded
by the original signed admission; outgoing is at least -r. Pair count uses
`pixel_count/2+pixel_count%2`; doubling a valid pair index and testing index+1
remain safe even at u32::MAX pixels. The original subtraction-based stride
guard prevents wraparound. Disjoint output ownership adds no atomics.

Below 65,536 pixels, the original one-pixel scalar schedule and complete window
arithmetic are retained inside the existing shader. This is a conservative
private heuristic selected after the initial probe's small-image regressions,
not a portable crossover guarantee. Large thin inputs and other devices are not
comprehensively timed by this survey. The compiled shader and its uniform branch
still change for fallback calls; zero overhead is not claimed.

Only the existing unpacked horizontal WGSL changes in production. All production
Rust host code, public API, admission, one/two-pass and packed selection, dispatch
requests, five lazy variants, layouts, other shaders, scratch/pools, metadata and
explicit transfer receipts retain their parent contracts. No dependency, feature,
pipeline or buffer is added. Driver transactions, bandwidth, physical memory,
compilation cost and cold runtime latency are unmeasured.

## Exploratory candidates

Two standalone timestamp phases use actual R8 source bytes in RGBA textures with
nonzero unused channels and poisoned pixels beyond the logical image. Both
parent/current full u32 outputs match independent u64 row prefixes; output guard
words remain unchanged. Three warmup pairs and 30 measured alternating pairs
exclude CPU, transfers, initialization and compilation. These medians have no
confidence intervals or isolation claim and are not pooled into final statistics.
The initial always-paired candidate improves large inputs but regresses small
images. The adopted candidate keeps the scalar schedule below 65,536 pixels;
26 probes include 65,535/65,536 pixel boundaries. All 22 initial and 26 adopted
cases remain below, including slower fallback medians. Both phases ran against
the unchanged parent production shader before adoption.

| Phase | Image | kw | Zero border | Parent median ms | Candidate median ms | Ratio of medians |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| always-paired | 1920x1080 | 259 | False | 3.820304 | 2.000864 | 1.909327 |
| always-paired | 1920x1080 | 259 | True | 2.978064 | 1.566112 | 1.901565 |
| always-paired | 1920x1080 | 511 | False | 5.717328 | 2.956384 | 1.933892 |
| always-paired | 1920x1080 | 511 | True | 5.680592 | 2.934016 | 1.936115 |
| always-paired | 1920x1080 | 4099 | False | 22.640864 | 11.350848 | 1.994641 |
| always-paired | 1920x1080 | 4099 | True | 22.617456 | 11.330624 | 1.996135 |
| always-paired | 3840x2160 | 259 | False | 12.155584 | 6.525056 | 1.862909 |
| always-paired | 3840x2160 | 259 | True | 12.076544 | 6.497568 | 1.858625 |
| always-paired | 3840x2160 | 511 | False | 23.663936 | 12.338240 | 1.917934 |
| always-paired | 3840x2160 | 511 | True | 23.682304 | 12.422496 | 1.906405 |
| always-paired | 257x259 | 259 | False | 0.086976 | 0.057872 | 1.502903 |
| always-paired | 257x259 | 259 | True | 0.086144 | 0.057136 | 1.507701 |
| always-paired | 257x259 | 4099 | False | 0.103952 | 0.080688 | 1.288320 |
| always-paired | 257x259 | 4099 | True | 0.103584 | 0.079792 | 1.298175 |
| always-paired | 17x19 | 259 | False | 0.004304 | 0.004736 | 0.908784 |
| always-paired | 17x19 | 259 | True | 0.004032 | 0.004752 | 0.848485 |
| always-paired | 1x521 | 4099 | False | 0.003024 | 0.003856 | 0.784232 |
| always-paired | 1x521 | 4099 | True | 0.003440 | 0.004096 | 0.839844 |
| always-paired | 521x1 | 4099 | False | 0.026448 | 0.026144 | 1.011628 |
| always-paired | 521x1 | 4099 | True | 0.026240 | 0.026016 | 1.008610 |
| always-paired | 2x521 | 259 | False | 0.003520 | 0.003856 | 0.912863 |
| always-paired | 2x521 | 259 | True | 0.003184 | 0.003568 | 0.892377 |
| bounded | 1920x1080 | 259 | False | 3.788640 | 1.984336 | 1.909273 |
| bounded | 1920x1080 | 259 | True | 2.976832 | 1.565712 | 1.901264 |
| bounded | 1920x1080 | 511 | False | 5.719888 | 2.955424 | 1.935387 |
| bounded | 1920x1080 | 511 | True | 5.682448 | 2.933472 | 1.937107 |
| bounded | 1920x1080 | 4099 | False | 22.638688 | 11.351520 | 1.994331 |
| bounded | 1920x1080 | 4099 | True | 22.620800 | 11.330320 | 1.996484 |
| bounded | 3840x2160 | 259 | False | 12.164992 | 6.531808 | 1.862423 |
| bounded | 3840x2160 | 259 | True | 12.074032 | 6.482992 | 1.862417 |
| bounded | 3840x2160 | 511 | False | 23.672304 | 12.404688 | 1.908335 |
| bounded | 3840x2160 | 511 | True | 23.590768 | 12.359568 | 1.908705 |
| bounded | 257x259 | 259 | False | 0.087696 | 0.058208 | 1.506597 |
| bounded | 257x259 | 259 | True | 0.086704 | 0.057760 | 1.501108 |
| bounded | 257x259 | 4099 | False | 0.104672 | 0.081184 | 1.289318 |
| bounded | 257x259 | 4099 | True | 0.103856 | 0.080640 | 1.287897 |
| bounded | 17x19 | 259 | False | 0.004528 | 0.003920 | 1.155102 |
| bounded | 17x19 | 259 | True | 0.004192 | 0.003936 | 1.065041 |
| bounded | 1x521 | 4099 | False | 0.003264 | 0.003360 | 0.971429 |
| bounded | 1x521 | 4099 | True | 0.003776 | 0.003408 | 1.107981 |
| bounded | 521x1 | 4099 | False | 0.026560 | 0.026816 | 0.990453 |
| bounded | 521x1 | 4099 | True | 0.026448 | 0.026560 | 0.995783 |
| bounded | 2x521 | 259 | False | 0.003344 | 0.003632 | 0.920705 |
| bounded | 2x521 | 259 | True | 0.003616 | 0.003120 | 1.158974 |
| bounded | 255x257 | 259 | False | 0.086320 | 0.086560 | 0.997227 |
| bounded | 255x257 | 259 | True | 0.085776 | 0.085888 | 0.998696 |
| bounded | 256x256 | 259 | False | 0.086896 | 0.053232 | 1.632402 |
| bounded | 256x256 | 259 | True | 0.086304 | 0.052272 | 1.651056 |

## Correctness and provenance

Five new native tests compare actual parent/current raw u32 sums against independent
u64 row prefixes, complete public calls against independent 2D prefixes and small
brute windows, physical inactive channels, metadata and transfer receipts. Maximum
rounding-safe odd kernels, full u32 mass above/below the cutoff, poisoned guards,
nonzero unused source channels, row crossings, odd tails, forced one-group stride,
full-u32 index bounds and signed sample coordinates are included. Pool/profiler
requests, warm reuse, zero/tiny budgets, pre-allocation errors, concurrent first use
and live outputs after pool clear/changed reuse are checked.

The full native parent freezes the original planner/cache/encoder/call, aliases
the unchanged border enum and injects a private pre-created cache in place of
the original lookup. Only the changed horizontal shader is frozen separately;
all shared dependencies are identity-checked. The original first scalar sum
and fallback schedule are independently normalized against the immediate parent.

BOX_WIDE_HORIZONTAL_PLAN: verified 264534 pixel owners, 397 row-crossing fallbacks, forced strides, 65535/65536/65537 and maximum-u32 counts, maximum rounding-safe kernels and signed samples.

BOX_WIDE_HORIZONTAL_RAW: verified 39038292 parent/current/u64 horizontal/guard words across 428 cases; full u32 mass and scalar/pair threshold boundaries, nonzero unused source channels, both borders, one-tap/row crossings/odd tails and one-group stride.

BOX_WIDE_HORIZONTAL_FULL: verified 76717440 complete parent/CPU/physical output bytes across 448 cases; unchanged packed/serial/wide policy, round-half-up results, receipts and metadata.

BOX_WIDE_HORIZONTAL_RESOURCES: verified original pipeline/pass/query counts, scratch/texture allocation requests and warm reuse, tiny/zero budgets and pre-allocation errors.

BOX_WIDE_HORIZONTAL_CONCURRENT: verified simultaneous packed/one-tap/wide initialization and independent retained outputs after pool clear and changed-input reuse.

There are 597 unchanged original GPU/image sources and 609 source/manifest/executable SHA-256 values. The final shader matches the adopted explored candidate. Hashes agree before/after both surveys and broad validation, including the resolved local ignored lockfile and executable. Repository/PR privacy scans find zero forbidden adapter-name matches; raw process names/PIDs remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 174 |
| voxel | 228 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 444 |

All-features/all-targets Clippy with warnings denied, CPU-only and minimal/CI-feature WASM checks with warnings denied, formatting and diff checks pass. Native correctness suites include ignored tests and skip explicit benchmarks.

## Final paired method

Reproduce the native diagnostic from the repository root:

```text
cargo test -p spatialrust-gpu --all-features --release --lib benchmark_complete_box_wide_horizontal_calls -- --ignored --nocapture --test-threads=1
```

Two consecutive initialized-runtime surveys use the same release executable,
21 cases, three warmup pairs and 30 measured parent/current pairs per route.
Resident/full variants rotate evenly through four positions; GPU pairs alternate.
Every completed output is checked after timing; checks, preparation, destruction
and recycling are excluded. No build/test started by this workflow overlaps timed
surveys. Shared-host work remains uncontrolled; no case is retried or filtered.

- `pass`: first-stage GPU timestamp in the complete profiled public call. This
  is the horizontal stage for separable calls or the single stage for serial
  controls. Original stage labels, timestamp counts and query bytes are checked.
- `resident`: completed public resident blur call plus idle wait, excluding
  input upload and checking readback.
- `full`: explicit upload, public blur and readback, including host transfer
  preparation and payload transfers.

Fourteen cases per survey use the paired wide shader, including the exact cutoff.
Five wide cases use its retained scalar schedule, including the cutoff-minus-one;
these fallback controls still use a changed compiled shader. Two original-program
controls use unchanged packed or serial shaders. No GPU causal benefit is assigned
to original-program controls. Private parent cache injection, code layout, timer
resolution and host load may contribute to whole-call/control differences.

Ratios are paired geometric means of parent/current (>1 favors current). The
95% interval uses 10,000 bootstrap resamples of 30 paired log ratios, seed
2026101812+1000*survey+10*case_index+route_index (zero-based indices), sorted draws
249/9749. Intervals above/below one are favorable/adverse, otherwise inconclusive.
Zeros remain undefined without epsilon substitution. There is no multiplicity
correction. Boundary observations cannot establish isolation during every pair,
portable speed or application performance. GPU-stage gains do not establish a
uniform transfer-inclusive benefit; all adverse and inconclusive results remain.

| Route | Paired: 28 comparisons | Scalar fallback: 10 comparisons | Original programs: 4 comparisons |
| --- | --- | --- | --- |
| pass | 28 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 6 adverse / 0 unresolved | 1 favorable / 3 inconclusive / 0 adverse / 0 unresolved |
| resident | 28 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 10 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |
| full | 22 favorable / 5 inconclusive / 1 adverse / 0 unresolved | 0 favorable / 10 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |

For paired pass, favorable ratios range 1.506-1.995.

For paired resident, favorable ratios range 1.126-1.948.

For paired full, favorable ratios range 1.140-1.628.

The retained scalar schedule does not eliminate compiled-shader overhead: the small 17x19, one-column 1x521 and cutoff-minus-one 255x257 GPU stages are adverse in both surveys. Scalar-fallback resident/full comparisons remain inconclusive; zero overhead is not established. One paired full comparison (1025x257, 259x9 zero border, survey 1) is adverse despite favorable GPU/resident intervals; survey 2 is inconclusive (ratio 0.983161, interval 0.948405-1.018152), with both point estimates below one. All repetitions and control results remain below.

All adverse observations:

- Survey 1, 1025x257, 259x9, zero=True, paired, full: ratio 0.968364, interval 0.944449-0.990919.

- Survey 1, 17x19, 259x31, zero=False, scalar-fallback, pass: ratio 0.936061, interval 0.908797-0.963208.

- Survey 1, 1x521, 4099x257, zero=False, scalar-fallback, pass: ratio 0.958331, interval 0.937755-0.978950.

- Survey 1, 255x257, 259x31, zero=False, scalar-fallback, pass: ratio 0.996259, interval 0.994326-0.998245.

- Survey 2, 17x19, 259x31, zero=False, scalar-fallback, pass: ratio 0.927920, interval 0.877840-0.970174.

- Survey 2, 1x521, 4099x257, zero=False, scalar-fallback, pass: ratio 0.945494, interval 0.926209-0.964535.

- Survey 2, 255x257, 259x31, zero=False, scalar-fallback, pass: ratio 0.995784, interval 0.993057-0.998022.

## Pre/post observations

Two-second boundary samples publish busy process counts/CPU cores only. GPU/controller values are utilization percentages, not allocated memory. No boundary sample observes every timed pair.

| Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Observed quiet |
| --- | --- | ---: | --- | --- | --- |
| 1 | pre | 1 | 0.420 | [[[0,0]],[[0,0]]] | False |
| 1 | post | 1 | 0.276 | [[[0,0]],[[0,0]]] | False |

Environment 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.
| 2 | pre | 0 | none | [[[0,0]],[[0,0]]] | True |
| 2 | post | 2 | 0.254, 0.497 | [[[0,0]],[[0,0]]] | False |

Environment 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

## All final comparisons

Medians retain all samples; intervals use paired log ratios. Zero counts are parent/current.

| Survey | Image | kw | kh | Zero border | Group | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16x17 | 3 | 3 | False | original-program | pass | 0.005440 | 0.005440 | 1.002675 | 0.960761-1.043384 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | original-program | resident | 0.178800 | 0.177250 | 1.002911 | 0.963753-1.044228 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | original-program | full | 1.030750 | 1.026350 | 1.001606 | 0.989962-1.013313 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | original-program | pass | 0.072464 | 0.072704 | 0.997536 | 0.993699-1.001374 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | original-program | resident | 0.288050 | 0.282200 | 1.018661 | 0.992751-1.045570 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | original-program | full | 1.182400 | 1.179900 | 1.008715 | 0.986715-1.031905 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | paired | pass | 2.999616 | 1.579936 | 1.901158 | 1.899830-1.902457 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 1 | False | paired | resident | 3.495100 | 2.074700 | 1.696009 | 1.682885-1.708612 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 1 | False | paired | full | 9.426950 | 8.106450 | 1.185376 | 1.172231-1.198419 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 3 | False | paired | pass | 2.990128 | 1.572608 | 1.901187 | 1.900218-1.902265 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 3 | False | paired | resident | 3.492900 | 2.060850 | 1.692417 | 1.679420-1.705309 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 3 | False | paired | full | 9.512200 | 8.041750 | 1.177822 | 1.163589-1.192326 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 9 | False | paired | pass | 2.988800 | 1.571792 | 1.900805 | 1.899599-1.901897 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 9 | False | paired | resident | 3.509600 | 2.086450 | 1.682309 | 1.667760-1.696499 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 9 | False | paired | full | 9.532700 | 8.194200 | 1.139845 | 1.113611-1.162178 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | False | paired | pass | 2.988976 | 1.571440 | 1.901762 | 1.900954-1.902559 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | False | paired | resident | 3.657850 | 2.226350 | 1.659181 | 1.628949-1.700013 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | False | paired | full | 9.784400 | 8.353950 | 1.173312 | 1.161778-1.184784 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 81 | False | paired | pass | 2.988448 | 1.572608 | 1.900675 | 1.899682-1.901697 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 81 | False | paired | resident | 3.967350 | 2.539350 | 1.562411 | 1.550521-1.574484 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 81 | False | paired | full | 10.121900 | 8.569600 | 1.175923 | 1.163546-1.188795 | 0/0 | favorable |
| 1 | 1920x1080 | 511 | 31 | False | paired | pass | 5.658000 | 2.928688 | 1.932309 | 1.931723-1.932915 | 0/0 | favorable |
| 1 | 1920x1080 | 511 | 31 | False | paired | resident | 6.349250 | 3.616700 | 1.757276 | 1.747860-1.766933 | 0/0 | favorable |
| 1 | 1920x1080 | 511 | 31 | False | paired | full | 12.687400 | 9.959000 | 1.270427 | 1.251370-1.289502 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | paired | pass | 22.637664 | 11.346224 | 1.995232 | 1.995147-1.995321 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | paired | resident | 23.216700 | 11.893300 | 1.947688 | 1.943371-1.951940 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | paired | full | 29.524550 | 18.139750 | 1.627503 | 1.617147-1.637800 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | paired | pass | 2.991872 | 1.572912 | 1.901892 | 1.901086-1.902669 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | paired | resident | 3.631550 | 2.216800 | 1.638905 | 1.625476-1.652518 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | paired | full | 9.783800 | 8.228950 | 1.190931 | 1.176243-1.205683 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 3 | False | paired | pass | 12.158736 | 6.534160 | 1.860480 | 1.859384-1.861498 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 3 | False | paired | resident | 12.971650 | 7.375050 | 1.759596 | 1.754748-1.763981 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 3 | False | paired | full | 31.278100 | 25.681250 | 1.214434 | 1.202782-1.225949 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 31 | False | paired | pass | 12.160304 | 6.540800 | 1.859488 | 1.858275-1.860707 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 31 | False | paired | resident | 13.687900 | 8.071600 | 1.690057 | 1.680063-1.698236 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 31 | False | paired | full | 32.234000 | 26.646500 | 1.227569 | 1.197726-1.262231 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 81 | False | paired | pass | 12.160160 | 6.540112 | 1.858911 | 1.857960-1.859881 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 81 | False | paired | resident | 15.225300 | 9.621150 | 1.576220 | 1.566272-1.585392 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 81 | False | paired | full | 33.651750 | 28.133350 | 1.202694 | 1.152231-1.256080 | 0/0 | favorable |
| 1 | 1025x257 | 259 | 9 | True | paired | pass | 0.383552 | 0.211808 | 1.810675 | 1.806546-1.814644 | 0/0 | favorable |
| 1 | 1025x257 | 259 | 9 | True | paired | resident | 0.793900 | 0.590150 | 1.344773 | 1.304314-1.386679 | 0/0 | favorable |
| 1 | 1025x257 | 259 | 9 | True | paired | full | 5.735450 | 5.848500 | 0.968364 | 0.944449-0.990919 | 0/0 | adverse |
| 1 | 257x259 | 259 | 257 | False | paired | pass | 0.304784 | 0.204272 | 1.510589 | 1.498789-1.524281 | 0/0 | favorable |
| 1 | 257x259 | 259 | 257 | False | paired | resident | 0.749200 | 0.654350 | 1.129974 | 1.084648-1.174197 | 0/0 | favorable |
| 1 | 257x259 | 259 | 257 | False | paired | full | 5.301550 | 5.216350 | 1.007315 | 0.985954-1.027291 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.017792 | 0.018672 | 0.936061 | 0.908797-0.963208 | 0/0 | adverse |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.499150 | 0.507850 | 0.853868 | 0.707745-1.021311 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | full | 5.729650 | 5.686150 | 1.021393 | 0.979177-1.075352 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.016544 | 0.017216 | 0.958331 | 0.937755-0.978950 | 0/0 | adverse |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.538100 | 0.563000 | 0.915659 | 0.779589-1.079642 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.609950 | 5.643900 | 0.986175 | 0.956346-1.015177 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.136080 | 0.137184 | 1.004446 | 0.974323-1.035553 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.770500 | 0.778500 | 1.004953 | 0.892998-1.128714 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 6.293700 | 6.136650 | 1.000394 | 0.959257-1.050288 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.018240 | 0.018176 | 0.985401 | 0.935178-1.025520 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.839250 | 1.006750 | 0.989408 | 0.852746-1.156358 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | full | 6.572900 | 6.292150 | 1.061149 | 0.999663-1.148042 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.500080 | 0.501456 | 0.996259 | 0.994326-0.998245 | 0/0 | adverse |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.656600 | 1.678800 | 0.990083 | 0.919083-1.067193 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | full | 6.303700 | 6.319750 | 1.001435 | 0.969241-1.033688 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | paired | pass | 0.504864 | 0.311360 | 1.620451 | 1.616659-1.624035 | 0/0 | favorable |
| 1 | 256x256 | 259 | 31 | False | paired | resident | 1.649300 | 1.478400 | 1.163257 | 1.063781-1.275904 | 0/0 | favorable |
| 1 | 256x256 | 259 | 31 | False | paired | full | 6.413300 | 6.344800 | 0.995469 | 0.950172-1.043873 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | original-program | pass | 0.005472 | 0.005408 | 1.020812 | 0.984876-1.058296 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | original-program | resident | 0.191800 | 0.188400 | 1.002867 | 0.956814-1.053360 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | original-program | full | 1.093550 | 1.108850 | 0.978097 | 0.912475-1.023944 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | original-program | pass | 0.071808 | 0.071488 | 1.005458 | 1.000769-1.010333 | 0/0 | favorable |
| 2 | 257x259 | 257 | 31 | False | original-program | resident | 0.312700 | 0.307550 | 1.013266 | 0.994632-1.031269 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | original-program | full | 1.405400 | 1.411200 | 1.004913 | 0.987440-1.023162 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | paired | pass | 2.988848 | 1.571776 | 1.901602 | 1.900907-1.902316 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 1 | False | paired | resident | 3.494850 | 2.066550 | 1.688818 | 1.665715-1.708590 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 1 | False | paired | full | 9.634100 | 8.083700 | 1.181498 | 1.167429-1.195785 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 3 | False | paired | pass | 2.990704 | 1.574416 | 1.901038 | 1.900181-1.901906 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 3 | False | paired | resident | 3.605700 | 2.184150 | 1.646708 | 1.621756-1.672387 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 3 | False | paired | full | 9.819000 | 8.320500 | 1.184826 | 1.169649-1.201063 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 9 | False | paired | pass | 3.012816 | 1.584256 | 1.901619 | 1.900688-1.902486 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 9 | False | paired | resident | 3.675650 | 2.200300 | 1.655510 | 1.630900-1.680013 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 9 | False | paired | full | 9.766600 | 8.227650 | 1.171484 | 1.147451-1.193658 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | False | paired | pass | 3.012384 | 1.584560 | 1.901285 | 1.900573-1.902012 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | False | paired | resident | 3.742800 | 2.331200 | 1.636246 | 1.598270-1.686510 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | False | paired | full | 9.947700 | 8.555500 | 1.153192 | 1.110023-1.185847 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 81 | False | paired | pass | 3.013472 | 1.584672 | 1.901460 | 1.900544-1.902382 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 81 | False | paired | resident | 3.999500 | 2.558200 | 1.558497 | 1.548183-1.569166 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 81 | False | paired | full | 10.082350 | 8.634800 | 1.156202 | 1.141060-1.170113 | 0/0 | favorable |
| 2 | 1920x1080 | 511 | 31 | False | paired | pass | 5.705408 | 2.950064 | 1.933897 | 1.933149-1.934622 | 0/0 | favorable |
| 2 | 1920x1080 | 511 | 31 | False | paired | resident | 6.360150 | 3.601500 | 1.764762 | 1.757253-1.771872 | 0/0 | favorable |
| 2 | 1920x1080 | 511 | 31 | False | paired | full | 12.351700 | 9.581450 | 1.289972 | 1.274105-1.307992 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | paired | pass | 22.637712 | 11.346256 | 1.995203 | 1.995112-1.995296 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | paired | resident | 23.205700 | 11.916950 | 1.946148 | 1.942839-1.949336 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | paired | full | 29.487700 | 18.222150 | 1.617156 | 1.602388-1.630668 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | True | paired | pass | 2.992400 | 1.573712 | 1.901202 | 1.900283-1.902087 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | True | paired | resident | 3.648650 | 2.233200 | 1.641949 | 1.622931-1.661299 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | True | paired | full | 10.261100 | 8.660850 | 1.178188 | 1.161814-1.194296 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 3 | False | paired | pass | 12.163504 | 6.540912 | 1.859995 | 1.858869-1.861154 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 3 | False | paired | resident | 12.988200 | 7.357400 | 1.765093 | 1.760745-1.769505 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 3 | False | paired | full | 31.948200 | 26.294050 | 1.218152 | 1.205165-1.231898 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 31 | False | paired | pass | 12.260432 | 6.594128 | 1.858882 | 1.857809-1.859907 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 31 | False | paired | resident | 13.688400 | 8.054550 | 1.693678 | 1.679496-1.702985 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 31 | False | paired | full | 32.291200 | 26.679500 | 1.214044 | 1.203257-1.225032 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 81 | False | paired | pass | 12.259472 | 6.592064 | 1.859654 | 1.858475-1.860830 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 81 | False | paired | resident | 15.182900 | 9.502700 | 1.596262 | 1.592464-1.599882 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 81 | False | paired | full | 33.521650 | 28.038350 | 1.204340 | 1.193831-1.215531 | 0/0 | favorable |
| 2 | 1025x257 | 259 | 9 | True | paired | pass | 0.386496 | 0.212720 | 1.815080 | 1.811329-1.818945 | 0/0 | favorable |
| 2 | 1025x257 | 259 | 9 | True | paired | resident | 0.767850 | 0.602550 | 1.279914 | 1.243504-1.318075 | 0/0 | favorable |
| 2 | 1025x257 | 259 | 9 | True | paired | full | 6.598500 | 6.603650 | 0.983161 | 0.948405-1.018152 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | paired | pass | 0.353696 | 0.238464 | 1.505791 | 1.494497-1.518443 | 0/0 | favorable |
| 2 | 257x259 | 259 | 257 | False | paired | resident | 0.827250 | 0.717450 | 1.125880 | 1.091011-1.159102 | 0/0 | favorable |
| 2 | 257x259 | 259 | 257 | False | paired | full | 6.010250 | 6.025900 | 0.998138 | 0.963449-1.032854 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.018288 | 0.019632 | 0.927920 | 0.877840-0.970174 | 0/0 | adverse |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 1.010250 | 1.066200 | 0.929932 | 0.805449-1.070998 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | full | 6.436600 | 6.457900 | 0.974970 | 0.937911-1.013775 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.016608 | 0.017312 | 0.945494 | 0.926209-0.964535 | 0/0 | adverse |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 1.061550 | 1.099300 | 0.920913 | 0.796950-1.064932 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 6.590000 | 6.614600 | 1.007154 | 0.971692-1.044702 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.131296 | 0.137056 | 0.986708 | 0.956054-1.017638 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.772950 | 0.715750 | 1.005374 | 0.874067-1.151328 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 6.429250 | 6.208600 | 1.014003 | 0.975881-1.048781 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.018368 | 0.018128 | 1.027087 | 0.981047-1.080148 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.656900 | 0.767750 | 0.919222 | 0.789681-1.071509 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | full | 6.656100 | 6.568150 | 1.004556 | 0.956410-1.055854 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.500144 | 0.501568 | 0.995784 | 0.993057-0.998022 | 0/0 | adverse |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.730400 | 1.678300 | 1.002412 | 0.927989-1.078894 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | full | 6.653850 | 6.598550 | 1.013341 | 0.978473-1.047843 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | paired | pass | 0.502352 | 0.310784 | 1.619246 | 1.616004-1.622514 | 0/0 | favorable |
| 2 | 256x256 | 259 | 31 | False | paired | resident | 1.606600 | 1.507600 | 1.134503 | 1.046369-1.233207 | 0/0 | favorable |
| 2 | 256x256 | 259 | 31 | False | paired | full | 6.663900 | 6.807750 | 0.974950 | 0.934062-1.018366 | 0/0 | inconclusive |

## Source locations

- [box_blur_horizontal.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_horizontal.wgsl)
- [box_blur_wide_hwindow_tests.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hwindow_tests.rs)
- [box_blur_wide_hwindow_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hwindow_reference.rs)
- [box_blur_wide_hwindow_horizontal_reference.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hwindow_horizontal_reference.wgsl)
- [box_blur_wide_hwindow_benchmark.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hwindow_benchmark.rs)
