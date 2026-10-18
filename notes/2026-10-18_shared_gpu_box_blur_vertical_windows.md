# Reusing adjacent GPU box blur vertical windows

Date: 2026-10-18. Named slice: shared GPU box blur vertical windows. Immediate
parent: `ef9805eb2e9df1b936c8b1fc171c041dd587c135` (merged PR #272).

## Change and exact arithmetic

The existing packed vertical shader previously sums a complete horizontal-sum
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
original code. Each input horizontal sum is decoded from its u16 half before
arithmetic; unused high halves and guard storage are not sampled.

For a valid second row y<=height-2, the incoming signed coordinate is at most
height-1+r, already bounded by the original planner. The outgoing coordinate is
at least -r. Pair ownership uses `width*ceil(height/2)`, computed as
`width*(height/2+height%2)` without adding width to the full pixel count. It is
at most the original bounded pixel count. First/second pixel indices remain
within that count, and the original subtraction-based stride guard prevents
wraparound. Width-one columns, odd dimensions and the final unpaired row use
the same ownership formula. No overlapping pixel writes or atomics are added.

Only the existing packed vertical WGSL changes in production. Host code, public
API, admission/error precedence, uniforms/layouts, lazy variant cache and pipeline
count, original workgroup/dispatch requests, one/two-pass selection, packed
threshold, all other shaders, resources/pools, metadata, explicit transfers and
receipts retain their parent contracts. Source window work is shared, but driver
transactions, bandwidth, physical allocation/residency and compilation cost are
not measured. These initialized-runtime diagnostics do not establish cold latency.

## Exploratory GPU candidates

A standalone timestamp probe compares a parent and candidate on real packed
horizontal sums. Each phase uses three warmup pairs and 30 measured pairs with
balanced alternating order. All logical output bytes, inactive channels and
sentinel pixels outside the logical image match. Source unused packed halves
and guard words are dirty. These are raw GPU vertical timings, excluding CPU
transfer/runtime/compilation work; no confidence interval or controlled host
isolation is inferred for the exploratory medians.

The first candidate combines adjacent columns with vec2 arithmetic and shared
row mapping. It becomes substantially slower for longer kernels and small
images and is rejected. The second candidate shares adjacent vertical windows
with the exact outgoing/incoming update and is adopted for full testing. All
24 results per phase, including adverse small-image observations, are retained;
phases are not pooled into final paired statistics. All probes ran against the
unchanged immediate-parent shader before production adoption.

| Phase | Image | kw | kh | Border (0=replicate, 1=zero) | Parent median ms | Candidate median ms | Ratio of medians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| columns-rejected | 1920x1080 | 257 | 1 | 0 | 0.123760 | 0.090896 | 1.361556 |
| columns-rejected | 1920x1080 | 257 | 1 | 1 | 0.102976 | 0.089792 | 1.146828 |
| columns-rejected | 1920x1080 | 257 | 3 | 0 | 0.156544 | 0.114160 | 1.371268 |
| columns-rejected | 1920x1080 | 257 | 3 | 1 | 0.136176 | 0.108736 | 1.252354 |
| columns-rejected | 1920x1080 | 257 | 9 | 0 | 0.197744 | 0.173568 | 1.139288 |
| columns-rejected | 1920x1080 | 257 | 9 | 1 | 0.174608 | 0.159840 | 1.092392 |
| columns-rejected | 1920x1080 | 257 | 31 | 0 | 0.371792 | 0.422688 | 0.879590 |
| columns-rejected | 1920x1080 | 257 | 31 | 1 | 0.349392 | 0.412448 | 0.847118 |
| columns-rejected | 1920x1080 | 257 | 81 | 0 | 0.738496 | 0.870384 | 0.848471 |
| columns-rejected | 1920x1080 | 257 | 81 | 1 | 0.718016 | 0.857120 | 0.837708 |
| columns-rejected | 1920x1080 | 257 | 257 | 0 | 3.194832 | 3.434992 | 0.930084 |
| columns-rejected | 1920x1080 | 257 | 257 | 1 | 2.475536 | 2.536720 | 0.975881 |
| columns-rejected | 257x259 | 257 | 31 | 0 | 0.012816 | 0.019248 | 0.665835 |
| columns-rejected | 257x259 | 257 | 31 | 1 | 0.012352 | 0.018288 | 0.675416 |
| columns-rejected | 257x259 | 257 | 257 | 0 | 0.046832 | 0.083248 | 0.562560 |
| columns-rejected | 257x259 | 257 | 257 | 1 | 0.046560 | 0.082272 | 0.565928 |
| columns-rejected | 17x19 | 257 | 31 | 0 | 0.004496 | 0.006752 | 0.665877 |
| columns-rejected | 17x19 | 257 | 31 | 1 | 0.004576 | 0.006480 | 0.706173 |
| columns-rejected | 1x521 | 257 | 257 | 0 | 0.010016 | 0.009312 | 1.075601 |
| columns-rejected | 1x521 | 257 | 257 | 1 | 0.029696 | 0.027200 | 1.091765 |
| columns-rejected | 2x521 | 257 | 257 | 0 | 0.037264 | 0.071248 | 0.523018 |
| columns-rejected | 2x521 | 257 | 257 | 1 | 0.038992 | 0.075456 | 0.516751 |
| columns-rejected | 3840x2160 | 257 | 31 | 0 | 2.123584 | 1.911200 | 1.111126 |
| columns-rejected | 3840x2160 | 257 | 31 | 1 | 1.944496 | 1.814480 | 1.071655 |
| vertical-windows | 1920x1080 | 257 | 1 | 0 | 0.123744 | 0.079808 | 1.550521 |
| vertical-windows | 1920x1080 | 257 | 1 | 1 | 0.102960 | 0.079504 | 1.295029 |
| vertical-windows | 1920x1080 | 257 | 3 | 0 | 0.156608 | 0.119184 | 1.314002 |
| vertical-windows | 1920x1080 | 257 | 3 | 1 | 0.137568 | 0.108864 | 1.263668 |
| vertical-windows | 1920x1080 | 257 | 9 | 0 | 0.197744 | 0.139728 | 1.415207 |
| vertical-windows | 1920x1080 | 257 | 9 | 1 | 0.174528 | 0.129232 | 1.350501 |
| vertical-windows | 1920x1080 | 257 | 31 | 0 | 0.371616 | 0.221616 | 1.676846 |
| vertical-windows | 1920x1080 | 257 | 31 | 1 | 0.351328 | 0.211040 | 1.664746 |
| vertical-windows | 1920x1080 | 257 | 81 | 0 | 0.745312 | 0.397872 | 1.873246 |
| vertical-windows | 1920x1080 | 257 | 81 | 1 | 0.724256 | 0.390384 | 1.855240 |
| vertical-windows | 1920x1080 | 257 | 257 | 0 | 2.475952 | 1.063328 | 2.328493 |
| vertical-windows | 1920x1080 | 257 | 257 | 1 | 2.497824 | 1.091360 | 2.288726 |
| vertical-windows | 257x259 | 257 | 31 | 0 | 0.015680 | 0.010976 | 1.428571 |
| vertical-windows | 257x259 | 257 | 31 | 1 | 0.014960 | 0.011024 | 1.357039 |
| vertical-windows | 257x259 | 257 | 257 | 0 | 0.057136 | 0.032560 | 1.754791 |
| vertical-windows | 257x259 | 257 | 257 | 1 | 0.056240 | 0.031824 | 1.767220 |
| vertical-windows | 17x19 | 257 | 31 | 0 | 0.005552 | 0.005776 | 0.961219 |
| vertical-windows | 17x19 | 257 | 31 | 1 | 0.006112 | 0.005456 | 1.120235 |
| vertical-windows | 1x521 | 257 | 257 | 0 | 0.011264 | 0.010352 | 1.088099 |
| vertical-windows | 1x521 | 257 | 257 | 1 | 0.011552 | 0.010400 | 1.110769 |
| vertical-windows | 2x521 | 257 | 257 | 0 | 0.013648 | 0.012944 | 1.054388 |
| vertical-windows | 2x521 | 257 | 257 | 1 | 0.040864 | 0.039824 | 1.026115 |
| vertical-windows | 3840x2160 | 257 | 31 | 0 | 2.143824 | 1.025824 | 2.089856 |
| vertical-windows | 3840x2160 | 257 | 31 | 1 | 1.947792 | 0.955824 | 2.037814 |

## Correctness and provenance

Five new native tests cover unique pixel ownership, full-u32 count/index limits,
forced grid strides, signed sample coordinates and maximum rounding-safe odd
kernels. Raw probes instrument only the destination of the actual parent/current
vertical shader to expose unrounded u32 sums, comparing every sum with independent
u64 column prefixes, including dirty source halves/guards and output sentinels.
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

BOX_VERTICAL_PLAN: verified 132724 pixel owners and extreme indices, forced grid strides, maximum odd kernels/rounding bounds and signed outgoing/incoming coordinates.

BOX_VERTICAL_RAW: verified 48166296 parent/current/u64 raw u32 sums and source/output guard words across 604 cases; dirty unused packed halves, maximum rounding-safe kernels, both borders and one-group row-pair stride.

BOX_VERTICAL_FULL: verified 38182344 complete parent/CPU/physical output bytes across 344 cases; unchanged packed/serial/wide policy, round-half-up results, receipts and metadata.

BOX_VERTICAL_RESOURCES: verified original pipeline/pass/query counts, scratch/texture allocation requests and warm reuse, tiny/zero budgets and pre-allocation errors.

BOX_VERTICAL_CONCURRENT: verified simultaneous packed/one-tap/wide initialization and independent retained outputs after pool clear and changed-input reuse.

There are 589 unchanged original GPU/image source files and 601 source/manifest/executable SHA-256 values. All production Rust host code and other original shaders are byte-identical. The production vertical shader matches the final explored candidate; the original first sum and final rounding/store body are independently verified. The finalized executable, GPU/image sources, manifests and resolved local ignored lockfile remain identical before/after timing and broad validation. Repository/PR privacy scans find zero forbidden adapter-name matches; raw local process receipts remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 164 |
| voxel | 228 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native tests | 434 |

All-features/all-targets Clippy with warnings denied, CPU-only check, minimal and CI-feature WASM checks with warnings denied, formatting and diff checks pass. Native suites include ignored correctness tests and exclude explicit benchmarks.

## Final paired method

Two consecutive surveys use the same finalized release executable, 18 cases,
three warmup pairs and 30 measured parent/current pairs per route. Wall-clock
parent/current resident/full variants rotate through all four positions; GPU
timestamp pairs alternate evenly. Input/oracle preparation, runtime initialization,
checks and output destruction/recycling are excluded. No build/test started by
this workflow overlaps the timed surveys. Other shared-host work is uncontrolled;
no case is retried or filtered. Every completed output is checked after timing.

- `pass`: GPU timestamp for the last stage of a completed profiled public call.
  It is the changed vertical stage for packed calls, the original unpacked
  vertical stage for the wide-kernel control, or the single original stage for
  serial calls. All stages run, and timestamp query counts/bytes are checked.
- `resident`: completed public resident box blur call plus idle wait, excluding
  input upload and output readback used for checking.
- `full`: explicit upload, public blur, output readback and host transfer
  preparation. This route includes payload transfers.

Sixteen cases per survey select the changed packed shader, including one-tap
height, one-row/column strips and odd final rows. Two original-program controls
select serial or unpacked code. The parent injects a pre-created private cache
in place of the original lookup; other host bodies remain frozen. No additional
pipeline variant or feature is introduced, but compiler cost is unmeasured.

Ratios are paired geometric means of parent/current (>1 favors current).
The 95% interval resamples 30 paired log ratios 10,000 times with seed
2026101808 + 1000*survey + 10*case_index + route_index (zero-based indices) and
sorted draws 249/9749. Intervals entirely above/below one are favorable/adverse;
otherwise inconclusive. Zero samples retain undefined ratios/intervals without
epsilon substitution. There is no multiplicity correction. Host boundary samples
do not establish isolation during every pair or portable/application speed.

| Route | Changed packed shader: 32 comparisons | Original-program controls: 4 comparisons |
| --- | --- | --- |
| pass | 29 favorable / 2 inconclusive / 1 adverse / 0 unresolved | 1 favorable / 3 inconclusive / 0 adverse / 0 unresolved |
| resident | 18 favorable / 14 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |
| full | 6 favorable / 26 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |

For packed pass, favorable paired ratios range 1.070-2.906.

For packed resident, favorable paired ratios range 1.029-2.364.

For packed full, favorable paired ratios range 1.057-1.109.

The 2x521, 9x81 constant-zero packed vertical pass is adverse in survey 1: ratio 0.927189, interval 0.913499-0.941621. The second survey has ratio 0.958516, interval 0.918209-1.001143 (inconclusive). Both point estimates are below one; thin-strip GPU latency remains a limitation rather than being hidden by favorable packed-stage summaries. No resident or full comparison has an adverse interval. One unchanged unpacked GPU control is favorable (1.059) in survey 2, which is not attributed to the changed shader.

All adverse/inconclusive and original-control results remain below. GPU-stage gains do not imply full-call gains. Differences in original-program controls are unresolved and may include cache lookup/code layout, timer resolution and shared-host load; no causal benefit is attributed to them.

## Pre/post observations

Two-second boundary samples publish busy process counts/CPU cores only; names/PIDs stay in ignored receipts. GPU pairs are utilization/memory-controller utilization percentages, not allocated-memory percentages. No sample observes every timed pair.

| Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Observed quiet |
| --- | --- | ---: | --- | --- | --- |
| 1 | pre | 1 | 0.923 | [[[0,0]],[[0,0]]] | False |
| 1 | post | 2 | 0.268, 0.929 | [[[3,0]],[[5,0]]] | False |
| 2 | pre | 1 | 0.906 | [[[3,0]],[[5,0]]] | False |
| 2 | post | 2 | 0.908, 0.897 | [[[4,0]],[[5,0]]] | False |

Environment 1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=last_pass_resident_full wall_order=balanced last_order=alternating checks_drop_recycle_excluded=true`.

Environment 2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=last_pass_resident_full wall_order=balanced last_order=alternating checks_drop_recycle_excluded=true`.

## All final comparisons

Medians retain every measured sample; ratios/intervals use paired log statistics. Zero counts are parent/current.

| Survey | Image | kw | kh | Zero border | Packed | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16x17 | 3 | 3 | False | False | pass | 0.005440 | 0.005472 | 0.978056 | 0.943412-1.015554 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | False | resident | 0.225850 | 0.222150 | 1.022815 | 0.970549-1.074244 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | False | full | 1.238650 | 1.273750 | 0.991077 | 0.969191-1.015545 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 3 | False | False | pass | 0.009920 | 0.009968 | 0.982971 | 0.957216-1.008229 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 3 | False | False | resident | 0.361900 | 0.360750 | 1.005515 | 0.980024-1.036671 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 3 | False | False | full | 1.422950 | 1.407550 | 1.004769 | 0.990006-1.020381 | 0/0 | inconclusive |
| 1 | 1920x1080 | 9 | 9 | False | True | pass | 0.199424 | 0.141104 | 1.419954 | 1.415453-1.424664 | 0/0 | favorable |
| 1 | 1920x1080 | 9 | 9 | False | True | resident | 0.668300 | 0.598650 | 1.108637 | 1.079094-1.140925 | 0/0 | favorable |
| 1 | 1920x1080 | 9 | 9 | False | True | full | 6.985650 | 6.886300 | 1.019899 | 0.998674-1.045043 | 0/0 | inconclusive |
| 1 | 1920x1080 | 31 | 31 | False | True | pass | 0.491392 | 0.290256 | 1.694033 | 1.692151-1.696061 | 0/0 | favorable |
| 1 | 1920x1080 | 31 | 31 | False | True | resident | 1.399800 | 1.207100 | 1.174787 | 1.141352-1.212813 | 0/0 | favorable |
| 1 | 1920x1080 | 31 | 31 | False | True | full | 8.200400 | 8.055800 | 1.073660 | 1.002841-1.197280 | 0/0 | favorable |
| 1 | 1920x1080 | 1 | 81 | False | True | pass | 1.541056 | 0.848128 | 1.825258 | 1.807353-1.856226 | 0/0 | favorable |
| 1 | 1920x1080 | 1 | 81 | False | True | resident | 1.631250 | 1.154800 | 1.460286 | 1.411935-1.509420 | 0/0 | favorable |
| 1 | 1920x1080 | 1 | 81 | False | True | full | 8.200050 | 7.593850 | 1.109486 | 1.054015-1.207955 | 0/0 | favorable |
| 1 | 1920x1080 | 257 | 3 | False | True | pass | 0.158368 | 0.120496 | 1.313700 | 1.311151-1.315982 | 0/0 | favorable |
| 1 | 1920x1080 | 257 | 3 | False | True | resident | 2.981500 | 2.936200 | 0.999809 | 0.937963-1.050483 | 0/0 | inconclusive |
| 1 | 1920x1080 | 257 | 3 | False | True | full | 9.656900 | 9.611800 | 1.020183 | 0.988971-1.070280 | 0/0 | inconclusive |
| 1 | 1920x1080 | 257 | 1 | False | True | pass | 0.121248 | 0.079776 | 1.522335 | 1.517665-1.527018 | 0/0 | favorable |
| 1 | 1920x1080 | 257 | 1 | False | True | resident | 2.534350 | 2.463600 | 0.992313 | 0.944779-1.028593 | 0/0 | inconclusive |
| 1 | 1920x1080 | 257 | 1 | False | True | full | 9.020650 | 9.203900 | 1.029955 | 0.964079-1.136592 | 0/0 | inconclusive |
| 1 | 1920x1080 | 9 | 257 | False | True | pass | 2.709456 | 1.196432 | 2.296856 | 2.263593-2.339239 | 0/0 | favorable |
| 1 | 1920x1080 | 9 | 257 | False | True | resident | 3.207450 | 1.724450 | 1.844587 | 1.801215-1.883539 | 0/0 | favorable |
| 1 | 1920x1080 | 9 | 257 | False | True | full | 10.286000 | 8.979550 | 1.101314 | 0.989329-1.176910 | 0/0 | inconclusive |
| 1 | 1920x1080 | 31 | 31 | True | True | pass | 0.791248 | 0.508848 | 1.578030 | 1.562397-1.595261 | 0/0 | favorable |
| 1 | 1920x1080 | 31 | 31 | True | True | resident | 1.279650 | 1.120900 | 1.138284 | 1.111348-1.165017 | 0/0 | favorable |
| 1 | 1920x1080 | 31 | 31 | True | True | full | 7.960100 | 8.000750 | 0.972821 | 0.885684-1.030210 | 0/0 | inconclusive |
| 1 | 3840x2160 | 9 | 9 | False | True | pass | 1.036640 | 0.700704 | 1.573507 | 1.496037-1.708873 | 0/0 | favorable |
| 1 | 3840x2160 | 9 | 9 | False | True | resident | 2.179350 | 1.814400 | 1.192514 | 1.120192-1.249585 | 0/0 | favorable |
| 1 | 3840x2160 | 9 | 9 | False | True | full | 25.047800 | 24.329250 | 1.092544 | 0.986701-1.240326 | 0/0 | inconclusive |
| 1 | 3840x2160 | 31 | 31 | False | True | pass | 2.526464 | 1.212976 | 2.101096 | 2.069214-2.135535 | 0/0 | favorable |
| 1 | 3840x2160 | 31 | 31 | False | True | resident | 4.535750 | 3.215950 | 1.396202 | 1.367370-1.415889 | 0/0 | favorable |
| 1 | 3840x2160 | 31 | 31 | False | True | full | 27.433950 | 26.812550 | 1.089131 | 0.968191-1.241505 | 0/0 | inconclusive |
| 1 | 3840x2160 | 1 | 81 | False | True | pass | 6.252464 | 2.148256 | 2.901981 | 2.872971-2.932084 | 0/0 | favorable |
| 1 | 3840x2160 | 1 | 81 | False | True | resident | 7.204500 | 3.083750 | 2.363883 | 2.310770-2.438736 | 0/0 | favorable |
| 1 | 3840x2160 | 1 | 81 | False | True | full | 28.911800 | 24.210900 | 1.097470 | 0.974694-1.206320 | 0/0 | inconclusive |
| 1 | 1025x257 | 9 | 9 | True | True | pass | 0.041600 | 0.029728 | 1.386131 | 1.342882-1.417725 | 0/0 | favorable |
| 1 | 1025x257 | 9 | 9 | True | True | resident | 0.518700 | 0.509700 | 0.975987 | 0.880394-1.051777 | 0/0 | inconclusive |
| 1 | 1025x257 | 9 | 9 | True | True | full | 7.466300 | 7.276100 | 1.051848 | 0.987954-1.136216 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 257 | False | True | pass | 0.161760 | 0.094832 | 1.698816 | 1.688903-1.708885 | 0/0 | favorable |
| 1 | 257x259 | 257 | 257 | False | True | resident | 0.714050 | 0.693650 | 1.089302 | 1.049368-1.137602 | 0/0 | favorable |
| 1 | 257x259 | 257 | 257 | False | True | full | 6.329000 | 6.288150 | 0.982107 | 0.920743-1.031181 | 0/0 | inconclusive |
| 1 | 17x19 | 31 | 31 | False | True | pass | 0.026336 | 0.021856 | 1.162078 | 1.074927-1.254724 | 0/0 | favorable |
| 1 | 17x19 | 31 | 31 | False | True | resident | 0.834700 | 0.904150 | 1.036295 | 0.909316-1.188889 | 0/0 | inconclusive |
| 1 | 17x19 | 31 | 31 | False | True | full | 6.288450 | 6.337350 | 0.954036 | 0.889424-1.004414 | 0/0 | inconclusive |
| 1 | 1x521 | 257 | 257 | False | True | pass | 0.048112 | 0.044272 | 1.077720 | 1.036665-1.119695 | 0/0 | favorable |
| 1 | 1x521 | 257 | 257 | False | True | resident | 1.204050 | 1.216450 | 1.045858 | 0.914492-1.197011 | 0/0 | inconclusive |
| 1 | 1x521 | 257 | 257 | False | True | full | 6.868100 | 6.642200 | 1.057164 | 1.015149-1.102902 | 0/0 | favorable |
| 1 | 521x1 | 257 | 257 | False | True | pass | 0.020624 | 0.015536 | 1.283435 | 1.209361-1.360511 | 0/0 | favorable |
| 1 | 521x1 | 257 | 257 | False | True | resident | 0.972500 | 1.066350 | 0.927689 | 0.830415-1.035275 | 0/0 | inconclusive |
| 1 | 521x1 | 257 | 257 | False | True | full | 6.823200 | 6.683550 | 1.032314 | 0.992319-1.079814 | 0/0 | inconclusive |
| 1 | 2x521 | 9 | 81 | True | True | pass | 0.030672 | 0.033104 | 0.927189 | 0.913499-0.941621 | 0/0 | adverse |
| 1 | 2x521 | 9 | 81 | True | True | resident | 1.069850 | 1.058800 | 0.921774 | 0.712228-1.132349 | 0/0 | inconclusive |
| 1 | 2x521 | 9 | 81 | True | True | full | 6.664700 | 6.755250 | 0.940941 | 0.850556-1.005662 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | False | pass | 0.005408 | 0.005424 | 0.993527 | 0.949458-1.036037 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | False | resident | 0.228250 | 0.224550 | 0.960648 | 0.872686-1.030557 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | False | full | 1.318150 | 1.316550 | 0.994790 | 0.971816-1.019897 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 3 | False | False | pass | 0.009888 | 0.009120 | 1.058740 | 1.029146-1.090199 | 0/0 | favorable |
| 2 | 257x259 | 259 | 3 | False | False | resident | 0.358900 | 0.362400 | 0.996351 | 0.975604-1.019276 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 3 | False | False | full | 1.441550 | 1.428000 | 1.016425 | 0.994492-1.039223 | 0/0 | inconclusive |
| 2 | 1920x1080 | 9 | 9 | False | True | pass | 0.197808 | 0.139360 | 1.403236 | 1.372019-1.420357 | 0/0 | favorable |
| 2 | 1920x1080 | 9 | 9 | False | True | resident | 0.648900 | 0.597400 | 1.086807 | 1.058085-1.113480 | 0/0 | favorable |
| 2 | 1920x1080 | 9 | 9 | False | True | full | 6.791350 | 6.895150 | 0.950948 | 0.858672-1.011745 | 0/0 | inconclusive |
| 2 | 1920x1080 | 31 | 31 | False | True | pass | 0.510016 | 0.300304 | 1.697428 | 1.695626-1.699214 | 0/0 | favorable |
| 2 | 1920x1080 | 31 | 31 | False | True | resident | 1.459200 | 1.286900 | 1.238757 | 1.155303-1.383879 | 0/0 | favorable |
| 2 | 1920x1080 | 31 | 31 | False | True | full | 8.094750 | 8.590800 | 0.944771 | 0.841078-1.031755 | 0/0 | inconclusive |
| 2 | 1920x1080 | 1 | 81 | False | True | pass | 1.049376 | 0.551232 | 1.904189 | 1.902370-1.905990 | 0/0 | favorable |
| 2 | 1920x1080 | 1 | 81 | False | True | resident | 1.670300 | 1.169100 | 1.456305 | 1.421092-1.503320 | 0/0 | favorable |
| 2 | 1920x1080 | 1 | 81 | False | True | full | 8.165750 | 7.567550 | 1.082521 | 1.061848-1.105849 | 0/0 | favorable |
| 2 | 1920x1080 | 257 | 3 | False | True | pass | 0.159920 | 0.121696 | 1.314176 | 1.311905-1.316225 | 0/0 | favorable |
| 2 | 1920x1080 | 257 | 3 | False | True | resident | 2.964650 | 2.871500 | 1.028984 | 1.002892-1.062568 | 0/0 | favorable |
| 2 | 1920x1080 | 257 | 3 | False | True | full | 9.462100 | 9.441200 | 1.051250 | 0.966016-1.177986 | 0/0 | inconclusive |
| 2 | 1920x1080 | 257 | 1 | False | True | pass | 0.124496 | 0.081664 | 1.528895 | 1.522213-1.536920 | 0/0 | favorable |
| 2 | 1920x1080 | 257 | 1 | False | True | resident | 2.562900 | 2.539600 | 1.001240 | 0.974516-1.022419 | 0/0 | inconclusive |
| 2 | 1920x1080 | 257 | 1 | False | True | full | 9.234400 | 9.214100 | 1.002482 | 0.987783-1.017360 | 0/0 | inconclusive |
| 2 | 1920x1080 | 9 | 257 | False | True | pass | 2.552272 | 1.104336 | 2.308316 | 2.304424-2.312191 | 0/0 | favorable |
| 2 | 1920x1080 | 9 | 257 | False | True | resident | 3.155800 | 1.722450 | 1.840572 | 1.821369-1.861524 | 0/0 | favorable |
| 2 | 1920x1080 | 9 | 257 | False | True | full | 9.808600 | 8.364700 | 1.097234 | 1.000467-1.168535 | 0/0 | favorable |
| 2 | 1920x1080 | 31 | 31 | True | True | pass | 0.482944 | 0.287584 | 1.673297 | 1.660567-1.680489 | 0/0 | favorable |
| 2 | 1920x1080 | 31 | 31 | True | True | resident | 1.204600 | 1.043550 | 1.194056 | 1.131967-1.291649 | 0/0 | favorable |
| 2 | 1920x1080 | 31 | 31 | True | True | full | 8.215050 | 8.284600 | 0.987812 | 0.937297-1.031109 | 0/0 | inconclusive |
| 2 | 3840x2160 | 9 | 9 | False | True | pass | 0.806400 | 0.550064 | 1.466822 | 1.461810-1.471775 | 0/0 | favorable |
| 2 | 3840x2160 | 9 | 9 | False | True | resident | 1.831050 | 1.573050 | 1.079114 | 0.943839-1.176948 | 0/0 | inconclusive |
| 2 | 3840x2160 | 9 | 9 | False | True | full | 23.046050 | 23.754950 | 0.981586 | 0.895336-1.072159 | 0/0 | inconclusive |
| 2 | 3840x2160 | 31 | 31 | False | True | pass | 2.516768 | 1.195264 | 2.103763 | 2.086545-2.122318 | 0/0 | favorable |
| 2 | 3840x2160 | 31 | 31 | False | True | resident | 4.004800 | 2.820500 | 1.406003 | 1.362757-1.433762 | 0/0 | favorable |
| 2 | 3840x2160 | 31 | 31 | False | True | full | 27.135600 | 25.074300 | 1.097588 | 1.024809-1.194854 | 0/0 | favorable |
| 2 | 3840x2160 | 1 | 81 | False | True | pass | 6.247232 | 2.131024 | 2.906301 | 2.879970-2.932775 | 0/0 | favorable |
| 2 | 3840x2160 | 1 | 81 | False | True | resident | 7.024700 | 2.937900 | 2.311903 | 2.144670-2.411175 | 0/0 | favorable |
| 2 | 3840x2160 | 1 | 81 | False | True | full | 27.589150 | 23.303850 | 1.106537 | 0.989667-1.204799 | 0/0 | inconclusive |
| 2 | 1025x257 | 9 | 9 | True | True | pass | 0.041568 | 0.029696 | 1.416890 | 1.396693-1.439103 | 0/0 | favorable |
| 2 | 1025x257 | 9 | 9 | True | True | resident | 0.470400 | 0.447200 | 1.005497 | 0.866996-1.135152 | 0/0 | inconclusive |
| 2 | 1025x257 | 9 | 9 | True | True | full | 7.028000 | 6.894100 | 0.998178 | 0.966642-1.032873 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 257 | False | True | pass | 0.160960 | 0.093200 | 1.712320 | 1.702029-1.724441 | 0/0 | favorable |
| 2 | 257x259 | 257 | 257 | False | True | resident | 0.738500 | 0.688600 | 1.065613 | 1.035745-1.091142 | 0/0 | favorable |
| 2 | 257x259 | 257 | 257 | False | True | full | 6.448700 | 6.425500 | 1.015625 | 0.993907-1.040166 | 0/0 | inconclusive |
| 2 | 17x19 | 31 | 31 | False | True | pass | 0.021536 | 0.021504 | 0.992040 | 0.926530-1.062868 | 0/0 | inconclusive |
| 2 | 17x19 | 31 | 31 | False | True | resident | 0.904550 | 0.844150 | 0.951747 | 0.784726-1.128141 | 0/0 | inconclusive |
| 2 | 17x19 | 31 | 31 | False | True | full | 6.468850 | 6.566300 | 1.030041 | 0.947118-1.169687 | 0/0 | inconclusive |
| 2 | 1x521 | 257 | 257 | False | True | pass | 0.046640 | 0.043872 | 1.069738 | 1.033341-1.107950 | 0/0 | favorable |
| 2 | 1x521 | 257 | 257 | False | True | resident | 1.005100 | 1.095650 | 0.884127 | 0.760758-1.017126 | 0/0 | inconclusive |
| 2 | 1x521 | 257 | 257 | False | True | full | 7.147750 | 7.202400 | 0.996448 | 0.930891-1.060522 | 0/0 | inconclusive |
| 2 | 521x1 | 257 | 257 | False | True | pass | 0.021104 | 0.016688 | 1.220848 | 1.119517-1.333490 | 0/0 | favorable |
| 2 | 521x1 | 257 | 257 | False | True | resident | 1.079250 | 1.117700 | 0.978073 | 0.847987-1.128909 | 0/0 | inconclusive |
| 2 | 521x1 | 257 | 257 | False | True | full | 6.986350 | 7.221550 | 0.964766 | 0.896771-1.024867 | 0/0 | inconclusive |
| 2 | 2x521 | 9 | 81 | True | True | pass | 0.030960 | 0.030240 | 0.958516 | 0.918209-1.001143 | 0/0 | inconclusive |
| 2 | 2x521 | 9 | 81 | True | True | resident | 1.020050 | 0.896750 | 1.025414 | 0.902504-1.160736 | 0/0 | inconclusive |
| 2 | 2x521 | 9 | 81 | True | True | full | 7.207500 | 6.972350 | 1.001122 | 0.944945-1.058770 | 0/0 | inconclusive |

## Source locations

- [box_blur_packed_vertical.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_packed_vertical.wgsl)
- [box_blur_vwindow_tests.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_vwindow_tests.rs)
- [box_blur_vwindow_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_vwindow_reference.rs)
- [box_blur_vwindow_vertical_reference.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_vwindow_vertical_reference.wgsl)
- [box_blur_vwindow_benchmark.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_vwindow_benchmark.rs)
