# Reusing adjacent wide GPU box blur vertical windows

Date: 2026-10-18. Named slice: shared wide GPU box blur vertical windows. Immediate
parent: `2dbea243fa36947ec5ad4d652bd345706e0f5f09` (merged PR #273).

## Change and exact arithmetic

The existing unpacked vertical shader previously sums a complete horizontal-sum
window for every output pixel. A native GPU invocation now owns one x coordinate
and two adjacent rows. It computes the original first window at even y, then
uses `V(x,y+1)=V(x,y)-H(x,y-r)+H(x,y+r+1)` for the second row, where H applies
the original replicate or constant-zero border and r=kernel_height/2. A one-tap
height loads each horizontal sum directly. An odd final row writes only its own
pixel. Original per-pixel rounding and RGBA output writes remain exact.

The outgoing sample belongs to the first nonnegative window or is zero; the
subtraction cannot underflow. Each second window retains the original bound
`sum <= kernel_width*kernel_height*255`, and the original planner also bounds
`sum + area/2 <= u32::MAX`. There are no approximate, floating-point or packed-
carry sums. Both scalar values round half upward once with the original area.
The first clipped sum/edge masses and final division/store body retain their
original arithmetic. Each input is the complete u32 horizontal sum; values beyond
65535 and source/output guard storage are checked without packing or truncation.

For a valid second row y<=height-2, the incoming signed coordinate is at most
height-1+r, already bounded by the original planner. The outgoing coordinate is
at least -r. Pair ownership uses `width*ceil(height/2)`, computed as
`width*(height/2+height%2)` without adding width to the full pixel count. It is
at most the original bounded pixel count. First/second pixel indices remain
within that count, and the original subtraction-based stride guard prevents
wraparound. Width-one columns, odd dimensions and the final unpaired row use
the same ownership formula. No overlapping pixel writes or atomics are added.

Only the existing unpacked vertical WGSL changes in production. Host code, public
API, admission/error precedence, uniforms/layouts, lazy variant cache and pipeline
count, original workgroup/dispatch requests, one/two-pass selection, packed
threshold, all other shaders, resources/pools, metadata, explicit transfers and
receipts retain their parent contracts. Source window work is shared, but driver
transactions, bandwidth, physical allocation/residency and compilation cost are
not measured. These initialized-runtime diagnostics do not establish cold latency.

## Exploratory GPU candidate

A standalone timestamp probe compares the immediate parent and row-pair candidate
using synthetic bounded u32 horizontal sums at kw=511. Three warmup pairs and
30 measured pairs use balanced alternating order. All output bytes, inactive
channels and sentinel pixels outside the logical image match. The full-width
input range exceeds 16 bits. These raw vertical GPU medians exclude CPU,
transfer, runtime initialization and compilation; confidence or host isolation
is not inferred. All 24 cases are retained separately from final statistics.
The probe ran before production adoption against the unchanged parent shader.

| Image | kw | kh | Border (0=replicate, 1=zero) | Parent median ms | Candidate median ms | Ratio of medians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1920x1080 | 511 | 1 | 0 | 0.125584 | 0.083392 | 1.505948 |
| 1920x1080 | 511 | 1 | 1 | 0.104416 | 0.083296 | 1.253554 |
| 1920x1080 | 511 | 3 | 0 | 0.157248 | 0.105616 | 1.488865 |
| 1920x1080 | 511 | 3 | 1 | 0.137360 | 0.094832 | 1.448456 |
| 1920x1080 | 511 | 9 | 0 | 0.203808 | 0.134448 | 1.515887 |
| 1920x1080 | 511 | 9 | 1 | 0.175328 | 0.118944 | 1.474038 |
| 1920x1080 | 511 | 31 | 0 | 0.561344 | 0.291952 | 1.922727 |
| 1920x1080 | 511 | 31 | 1 | 0.519456 | 0.270080 | 1.923341 |
| 1920x1080 | 511 | 81 | 0 | 2.715344 | 1.382480 | 1.964111 |
| 1920x1080 | 511 | 81 | 1 | 2.640224 | 1.347744 | 1.958995 |
| 1920x1080 | 511 | 257 | 0 | 4.933536 | 1.781152 | 2.769857 |
| 1920x1080 | 511 | 257 | 1 | 4.624528 | 1.666512 | 2.774974 |
| 257x259 | 511 | 31 | 0 | 0.014928 | 0.009664 | 1.544702 |
| 257x259 | 511 | 31 | 1 | 0.015104 | 0.009408 | 1.605442 |
| 257x259 | 511 | 257 | 0 | 0.037088 | 0.024256 | 1.529024 |
| 257x259 | 511 | 257 | 1 | 0.036496 | 0.023856 | 1.529846 |
| 17x19 | 511 | 31 | 0 | 0.004288 | 0.004272 | 1.003745 |
| 17x19 | 511 | 31 | 1 | 0.004528 | 0.003952 | 1.145749 |
| 1x521 | 511 | 257 | 0 | 0.009184 | 0.007472 | 1.229122 |
| 1x521 | 511 | 257 | 1 | 0.009104 | 0.007712 | 1.180498 |
| 2x521 | 511 | 257 | 0 | 0.037920 | 0.035712 | 1.061828 |
| 2x521 | 511 | 257 | 1 | 0.042256 | 0.040208 | 1.050935 |
| 3840x2160 | 511 | 31 | 0 | 3.378336 | 1.314256 | 2.570531 |
| 3840x2160 | 511 | 31 | 1 | 3.269408 | 1.207424 | 2.707755 |

## Correctness and provenance

Five new native tests cover unique pixel ownership, full-u32 count/index limits,
forced grid strides, signed sample coordinates and maximum rounding-safe odd
kernels. Raw probes instrument only the destination of the actual parent/current
vertical shader to expose unrounded u32 sums, comparing every sum with independent
u64 column prefixes, including full-width source words/guards and output sentinels.
Complete public calls compare a frozen immediate parent, independent u64 2D
prefixes and small brute-force windows; every physical RGBA byte, metadata and
transfer receipt is checked. Packed/serial/unpacked paths, maximum integer mass,
one-tap axes, both borders, odd tails, thin strips and large inputs are included.

Resource checks compare pass/query names/counts, scratch/texture request deltas,
warm reuse, tiny/zero budgets and invalid pre-allocation errors. Concurrent first
use and live outputs after pool clear/changed input reuse remain exact. The full
native parent freezes the original planner/cache/encoder/call, aliases only the
unchanged public border enum, and injects a private parent cache in place of the
runtime lookup. Only the changed vertical shader is frozen separately; all shared
dependencies are identity-checked. Warmups exclude pipeline compilation.

BOX_WIDE_VERTICAL_PLAN: verified 132724 pixel owners and extreme indices, forced grid strides, maximum odd kernels/rounding bounds and signed outgoing/incoming coordinates.

BOX_WIDE_VERTICAL_RAW: verified 55047840 parent/current/u64 raw u32 sums and source/output guard words across 640 cases; full-width input words and dirty guards, maximum rounding-safe kernels, both borders and one-group row-pair stride.

BOX_WIDE_VERTICAL_FULL: verified 38182560 complete parent/CPU/physical output bytes across 350 cases; unchanged packed/serial/wide policy, round-half-up results, receipts and metadata.

BOX_WIDE_VERTICAL_RESOURCES: verified original pipeline/pass/query counts, scratch/texture allocation requests and warm reuse, tiny/zero budgets and pre-allocation errors.

BOX_WIDE_VERTICAL_CONCURRENT: verified simultaneous packed/one-tap/wide initialization and independent retained outputs after pool clear and changed-input reuse.

There are 593 unchanged original GPU/image source files and 605 source/manifest/executable SHA-256 values. All production Rust host code and other original shaders are byte-identical. The production vertical shader matches the final explored candidate; the original first sum and final rounding/store body are independently verified. The finalized executable, GPU/image sources, manifests and resolved local ignored lockfile remain identical before/after timing and broad validation. Repository/PR privacy scans find zero forbidden adapter-name matches; raw local process receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 169 |
| voxel | 228 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 439 |

All-features/all-targets Clippy with warnings denied, CPU-only check, minimal and CI-feature WASM checks with warnings denied, formatting and diff checks pass. Native suites include ignored correctness tests and exclude explicit benchmarks.

## Final paired method

Two consecutive surveys use the same finalized release executable, 19 cases,
three warmup pairs and 30 measured parent/current pairs per route. Wall-clock
parent/current resident/full variants rotate through all four positions; GPU
timestamp pairs alternate evenly. Input/oracle preparation, runtime initialization,
checks and output destruction/recycling are excluded. No build/test started by
this workflow overlaps the timed surveys. Other shared-host work is uncontrolled;
no case is retried or filtered. Every completed output is checked after timing.

- `pass`: GPU timestamp for the last stage of a completed profiled public call.
  It is the changed vertical stage for wide calls, the original packed
  vertical stage for the packed control, or the single original stage for
  serial calls. All stages run, and timestamp query counts/bytes are checked.
- `resident`: completed public resident box blur call plus idle wait, excluding
  input upload and output readback used for checking.
- `full`: explicit upload, public blur, output readback and host transfer
  preparation. This route includes payload transfers.

Seventeen cases per survey select the changed unpacked shader, including one-tap
height, one-row/column strips and odd final rows. Two original-program controls
select serial or packed code. The parent injects a pre-created private cache
in place of the original lookup; other host bodies remain frozen. No additional
pipeline variant or feature is introduced, but compiler cost is unmeasured.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times with seed
2026101810 + 1000*survey + 10*case_index + route_index (zero-based indices) and
sorted draws 249/9749. Intervals entirely above/below one are favorable/adverse;
otherwise inconclusive. Zero samples retain undefined ratios/intervals without
epsilon substitution. There is no multiplicity correction. Host boundary samples
do not establish isolation during every pair or portable/application speed.

| Route | Changed wide shader: 34 comparisons | Original-program controls: 4 comparisons |
| --- | --- | --- |
| pass | 30 favorable / 1 inconclusive / 3 adverse / 0 unresolved | 1 favorable / 3 inconclusive / 0 adverse / 0 unresolved |
| resident | 15 favorable / 19 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |
| full | 5 favorable / 27 inconclusive / 2 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |

For wide pass, favorable paired ratios range 1.068-2.766.

For wide resident, favorable paired ratios range 1.011-1.297.

For wide full, favorable paired ratios range 1.026-1.116.

All adverse wide or original-control comparisons are explicitly retained:

- Survey 1, 3840x2160, 259x3, zero=False, changed=True, full: ratio 0.935469, interval 0.872840-0.990379.

- Survey 1, 17x19, 259x31, zero=False, changed=True, pass: ratio 0.865782, interval 0.802985-0.936073.

- Survey 1, 2x521, 259x81, zero=True, changed=True, pass: ratio 0.953704, interval 0.927331-0.981781.

- Survey 2, 521x1, 4099x257, zero=False, changed=True, full: ratio 0.931628, interval 0.853371-0.995441.

- Survey 2, 2x521, 259x81, zero=True, changed=True, pass: ratio 0.984792, interval 0.978478-0.989920.

The 2x521, 259x81 constant-zero GPU pass is adverse in both surveys (ratios 0.953704 and 0.984792); thin-strip GPU latency is a repeated limitation. The 17x19, 259x31 replicate GPU pass is adverse in survey 1 and favorable in survey 2. Full 4K 259x3 is adverse in survey 1 and inconclusive in survey 2; the 521x1, 4099x257 full call is inconclusive in survey 1 and adverse in survey 2. Both full-call conditions have point estimates below one in both surveys. No adverse resident interval is observed. There is no uniform transfer-inclusive speed benefit.

All adverse/inconclusive and original-control results remain below. GPU-stage gains do not imply full-call gains. Differences in original-program controls are unresolved and may include cache lookup/code layout, timer resolution and shared-host load; no causal benefit is attributed to them.

## Pre/post observations

Two-second boundary samples publish busy process counts/CPU cores only; names/PIDs stay in ignored receipts. GPU pairs are utilization/memory-controller utilization percentages, not allocated-memory percentages. No sample observes every timed pair.

| Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Observed quiet |
| --- | --- | ---: | --- | --- | --- |
| 1 | pre | 2 | 0.371, 1.197 | [[[0,0]],[[0,0]]] | False |
| 1 | post | 0 | none | [[[0,0]],[[0,0]]] | True |
| 2 | pre | 4 | 0.275, 0.571, 0.324, 0.907 | [[[0,0]],[[0,0]]] | False |
| 2 | post | 0 | none | [[[0,0]],[[0,0]]] | True |

Environment 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=last_pass_resident_full wall_order=balanced last_order=alternating checks_drop_recycle_excluded=true`.

Environment 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=last_pass_resident_full wall_order=balanced last_order=alternating checks_drop_recycle_excluded=true`.

## All final comparisons

Medians retain every measured sample; ratios/intervals use paired log statistics. Zero counts are parent/current.

| Survey | Image | kw | kh | Zero border | Packed | Changed | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16x17 | 3 | 3 | False | False | False | pass | 0.005536 | 0.005568 | 1.005357 | 0.969278-1.044441 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | False | False | resident | 0.266600 | 0.236250 | 0.907821 | 0.754377-1.051222 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | False | False | full | 1.215850 | 1.237100 | 0.959547 | 0.859020-1.048740 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | True | False | pass | 0.010512 | 0.010832 | 0.974685 | 0.946373-1.002866 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | True | False | resident | 0.332500 | 0.334600 | 1.004569 | 0.973543-1.032663 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | True | False | full | 1.427850 | 1.430550 | 0.995173 | 0.968435-1.026604 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | False | True | pass | 0.103104 | 0.068800 | 1.502022 | 1.497930-1.506416 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 1 | False | False | True | resident | 3.512600 | 3.485200 | 1.002638 | 0.980645-1.019387 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | False | True | full | 9.955500 | 10.061450 | 1.115017 | 1.002869-1.300026 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 3 | False | False | True | pass | 0.129104 | 0.086272 | 1.497948 | 1.494623-1.501369 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 3 | False | False | True | resident | 3.561250 | 3.558200 | 0.975217 | 0.922003-1.013823 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | False | True | full | 10.436800 | 10.392150 | 0.948151 | 0.850756-1.009757 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | False | True | pass | 0.167136 | 0.108784 | 1.585158 | 1.532244-1.691959 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 9 | False | False | True | resident | 3.609050 | 3.580000 | 0.968198 | 0.906998-1.030196 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | False | True | full | 9.927400 | 9.946250 | 0.944854 | 0.823006-1.026655 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | False | False | True | pass | 0.438688 | 0.231328 | 1.893340 | 1.887664-1.898995 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | False | False | True | resident | 3.932150 | 3.754050 | 1.057120 | 1.039076-1.083864 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | False | False | True | full | 10.249200 | 10.116350 | 0.972534 | 0.863140-1.049291 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 81 | False | False | True | pass | 1.094592 | 0.553616 | 1.976411 | 1.972243-1.980581 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 81 | False | False | True | resident | 4.633600 | 4.094450 | 1.130700 | 1.122369-1.138289 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 81 | False | False | True | full | 10.862700 | 10.401050 | 1.000602 | 0.914155-1.052874 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | False | True | pass | 0.439632 | 0.232560 | 1.891545 | 1.885244-1.897421 | 0/0 | favorable |
| 1 | 1920x1080 | 511 | 31 | False | False | True | resident | 6.661400 | 6.449900 | 1.033998 | 1.028157-1.039773 | 0/0 | favorable |
| 1 | 1920x1080 | 511 | 31 | False | False | True | full | 12.999850 | 12.762850 | 0.941499 | 0.838586-1.021024 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | False | False | True | pass | 0.168096 | 0.109360 | 1.535906 | 1.532549-1.539117 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | False | True | resident | 23.443750 | 23.414450 | 0.991169 | 0.970431-1.004250 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | False | False | True | full | 31.077000 | 30.292200 | 0.946640 | 0.857426-1.022028 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | True | False | True | pass | 0.413728 | 0.218240 | 1.894921 | 1.891099-1.898672 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | False | True | resident | 3.855200 | 3.694200 | 1.034325 | 1.007549-1.051667 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | False | True | full | 9.917450 | 9.678300 | 1.090764 | 1.009852-1.232987 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 3 | False | False | True | pass | 0.514688 | 0.337952 | 1.523277 | 1.518491-1.528148 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 3 | False | False | True | resident | 13.288850 | 13.102250 | 1.014337 | 1.008946-1.019135 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 3 | False | False | True | full | 31.590100 | 31.839800 | 0.935469 | 0.872840-0.990379 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 31 | False | False | True | pass | 1.935328 | 1.007088 | 1.917697 | 1.909148-1.926019 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 31 | False | False | True | resident | 14.786300 | 13.760950 | 1.072935 | 1.065182-1.081226 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 31 | False | False | True | full | 33.622050 | 31.947550 | 1.029335 | 0.901600-1.161942 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | False | True | pass | 6.630272 | 2.393184 | 2.765846 | 2.753811-2.777399 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 81 | False | False | True | resident | 19.511100 | 15.209450 | 1.297188 | 1.278913-1.330347 | 0/0 | favorable |
| 1 | 3840x2160 | 259 | 81 | False | False | True | full | 38.097800 | 33.620900 | 1.064525 | 0.970583-1.141188 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | False | True | pass | 0.026256 | 0.019216 | 1.361824 | 1.343974-1.379094 | 0/0 | favorable |
| 1 | 1025x257 | 259 | 9 | True | False | True | resident | 0.724700 | 0.741400 | 1.041937 | 0.954216-1.200518 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | False | True | full | 6.451700 | 6.262050 | 1.026083 | 0.983357-1.069578 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | False | True | pass | 0.141600 | 0.090096 | 1.560216 | 1.545506-1.574803 | 0/0 | favorable |
| 1 | 257x259 | 259 | 257 | False | False | True | resident | 0.766950 | 0.721850 | 1.051229 | 1.014024-1.086805 | 0/0 | favorable |
| 1 | 257x259 | 259 | 257 | False | False | True | full | 5.492750 | 5.552750 | 0.972342 | 0.901428-1.039209 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | False | True | pass | 0.018336 | 0.021552 | 0.865782 | 0.802985-0.936073 | 0/0 | adverse |
| 1 | 17x19 | 259 | 31 | False | False | True | resident | 0.624500 | 0.726450 | 0.928943 | 0.814383-1.060336 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | False | True | full | 5.618600 | 5.663450 | 0.966124 | 0.919357-1.004893 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | False | True | pass | 0.043872 | 0.039056 | 1.130390 | 1.112416-1.149199 | 0/0 | favorable |
| 1 | 1x521 | 4099 | 257 | False | False | True | resident | 0.790150 | 0.931100 | 1.034346 | 0.879981-1.216524 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | False | True | full | 6.520150 | 6.506300 | 1.008573 | 0.958827-1.063186 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | False | True | pass | 0.014736 | 0.015024 | 1.022551 | 0.920218-1.136751 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | False | True | resident | 0.779450 | 0.965050 | 0.901631 | 0.763961-1.055338 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | False | True | full | 6.387800 | 6.507550 | 0.970892 | 0.927685-1.012566 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | False | True | pass | 0.026448 | 0.028752 | 0.953704 | 0.927331-0.981781 | 0/0 | adverse |
| 1 | 2x521 | 259 | 81 | True | False | True | resident | 0.792800 | 0.888450 | 0.972773 | 0.864441-1.092239 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | False | True | full | 6.047400 | 6.092450 | 0.945719 | 0.847076-1.018586 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | False | False | pass | 0.005472 | 0.005376 | 1.031191 | 1.005871-1.059919 | 0/0 | favorable |
| 2 | 16x17 | 3 | 3 | False | False | False | resident | 0.221250 | 0.218250 | 1.005432 | 0.969886-1.042888 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | False | False | full | 1.209100 | 1.214000 | 0.987698 | 0.956046-1.015309 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | True | False | pass | 0.011056 | 0.010768 | 1.016244 | 0.989482-1.043989 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | True | False | resident | 0.312300 | 0.315650 | 0.993724 | 0.969071-1.018613 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | True | False | full | 1.341700 | 1.309700 | 1.013581 | 0.983094-1.043401 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | False | True | pass | 0.103488 | 0.068992 | 1.499281 | 1.495400-1.503133 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 1 | False | False | True | resident | 3.580400 | 3.551150 | 1.011444 | 1.005193-1.017554 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 1 | False | False | True | full | 9.604350 | 9.718850 | 0.997618 | 0.984413-1.010958 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | False | True | pass | 0.129936 | 0.086976 | 1.451986 | 1.368084-1.497375 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 3 | False | False | True | resident | 3.588350 | 3.545400 | 1.008095 | 0.999799-1.015866 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | False | True | full | 9.811550 | 9.776200 | 1.010290 | 0.999071-1.022463 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | False | True | pass | 0.168288 | 0.109888 | 1.531524 | 1.527559-1.535366 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 9 | False | False | True | resident | 3.644050 | 3.573700 | 1.018197 | 1.004136-1.033885 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 9 | False | False | True | full | 10.050900 | 9.727300 | 1.025837 | 1.007096-1.049291 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | False | False | True | pass | 0.440848 | 0.233952 | 1.898394 | 1.879098-1.931187 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | False | False | True | resident | 3.922450 | 3.686250 | 1.095218 | 1.053683-1.164730 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | False | False | True | full | 10.465200 | 9.877650 | 1.007760 | 0.957123-1.051976 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 81 | False | False | True | pass | 1.104576 | 0.557520 | 1.968299 | 1.942936-1.982761 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 81 | False | False | True | resident | 4.579750 | 4.025000 | 1.142905 | 1.133403-1.154182 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 81 | False | False | True | full | 11.577000 | 10.915450 | 1.116317 | 1.031843-1.261664 | 0/0 | favorable |
| 2 | 1920x1080 | 511 | 31 | False | False | True | pass | 0.440192 | 0.232912 | 1.837743 | 1.759158-1.893329 | 0/0 | favorable |
| 2 | 1920x1080 | 511 | 31 | False | False | True | resident | 6.666900 | 6.489000 | 1.033135 | 1.026669-1.040331 | 0/0 | favorable |
| 2 | 1920x1080 | 511 | 31 | False | False | True | full | 13.989450 | 13.650950 | 1.017997 | 0.919806-1.086491 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | False | False | True | pass | 0.169200 | 0.110608 | 1.527314 | 1.522502-1.531700 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | False | True | resident | 23.667050 | 24.134100 | 0.977317 | 0.937968-1.015687 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | False | False | True | full | 34.300650 | 34.028500 | 0.966325 | 0.884020-1.036976 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | False | True | pass | 0.417280 | 0.220256 | 1.914490 | 1.890433-1.958690 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | True | False | True | resident | 3.968800 | 3.807700 | 1.005175 | 0.928205-1.051773 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | False | True | full | 11.214300 | 11.097900 | 1.005262 | 0.984995-1.026037 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | False | True | pass | 0.517680 | 0.340640 | 1.520477 | 1.516352-1.524640 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 3 | False | False | True | resident | 13.411750 | 13.179000 | 1.006067 | 0.954545-1.048147 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | False | True | full | 34.725800 | 34.204100 | 0.989646 | 0.939210-1.021609 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 31 | False | False | True | pass | 1.945424 | 1.014944 | 1.897259 | 1.824050-1.973019 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 31 | False | False | True | resident | 14.873200 | 13.871950 | 1.074390 | 1.060809-1.088435 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 31 | False | False | True | full | 35.898800 | 34.817450 | 1.012763 | 0.943790-1.074220 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | False | True | pass | 6.640352 | 2.406736 | 2.754186 | 2.742778-2.764183 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 81 | False | False | True | resident | 19.592300 | 15.305150 | 1.285455 | 1.277772-1.297460 | 0/0 | favorable |
| 2 | 3840x2160 | 259 | 81 | False | False | True | full | 41.418050 | 37.791500 | 1.070136 | 1.011091-1.108167 | 0/0 | favorable |
| 2 | 1025x257 | 259 | 9 | True | False | True | pass | 0.091072 | 0.065472 | 1.360586 | 1.342870-1.376644 | 0/0 | favorable |
| 2 | 1025x257 | 259 | 9 | True | False | True | resident | 0.955300 | 0.954450 | 0.993248 | 0.946268-1.035953 | 0/0 | inconclusive |
| 2 | 1025x257 | 259 | 9 | True | False | True | full | 10.016850 | 9.894550 | 0.956588 | 0.871557-1.037717 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | False | True | pass | 0.241408 | 0.122624 | 1.956558 | 1.934442-1.978374 | 0/0 | favorable |
| 2 | 257x259 | 259 | 257 | False | False | True | resident | 1.484450 | 1.438400 | 0.960859 | 0.865176-1.048082 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | False | True | full | 7.833050 | 7.873250 | 0.986241 | 0.890640-1.082519 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | False | True | pass | 0.017440 | 0.016400 | 1.068023 | 1.019753-1.109156 | 0/0 | favorable |
| 2 | 17x19 | 259 | 31 | False | False | True | resident | 0.853250 | 0.980150 | 0.858121 | 0.723806-1.002636 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | False | True | full | 7.789950 | 7.631850 | 1.011447 | 0.938859-1.089169 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | False | True | pass | 0.044352 | 0.035312 | 1.241946 | 1.226344-1.255375 | 0/0 | favorable |
| 2 | 1x521 | 4099 | 257 | False | False | True | resident | 0.984150 | 0.906400 | 1.009097 | 0.882899-1.131275 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | False | True | full | 7.786450 | 7.655900 | 1.013820 | 0.941708-1.092747 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | False | True | pass | 0.022624 | 0.020016 | 1.131473 | 1.117834-1.144548 | 0/0 | favorable |
| 2 | 521x1 | 4099 | 257 | False | False | True | resident | 1.518650 | 1.591300 | 1.044392 | 0.921734-1.191048 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | False | True | full | 7.388300 | 7.723600 | 0.931628 | 0.853371-0.995441 | 0/0 | adverse |
| 2 | 2x521 | 259 | 81 | True | False | True | pass | 0.046432 | 0.046752 | 0.984792 | 0.978478-0.989920 | 0/0 | adverse |
| 2 | 2x521 | 259 | 81 | True | False | True | resident | 1.421300 | 1.364650 | 1.002516 | 0.864669-1.131728 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | False | True | full | 7.022200 | 7.273650 | 0.965409 | 0.929732-1.001231 | 0/0 | inconclusive |

## Source locations

- [box_blur_vertical.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_vertical.wgsl)
- [box_blur_wide_vwindow_tests.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_vwindow_tests.rs)
- [box_blur_wide_vwindow_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_vwindow_reference.rs)
- [box_blur_wide_vwindow_vertical_reference.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_vwindow_vertical_reference.wgsl)
- [box_blur_wide_vwindow_benchmark.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_vwindow_benchmark.rs)
