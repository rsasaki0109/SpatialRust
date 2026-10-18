# Four adjacent wide GPU box blur horizontal windows

Date: 2026-10-18. Named slice: four wide GPU box blur horizontal windows.
Immediate parent: `40828ce315e94dce8f20ddb7e6323ecf09f9ee92` (merged PR #275).

## Change and exactness

For unpacked separable calls with at least 65,536 pixels and kernel width at least
4,099, one invocation now owns up to four consecutive linear output indices
instead of two. It computes one original complete horizontal sum, then each
same-row successor uses
`H(x+1,y)=H(x,y)-I(x-r,y)+I(x+r+1,y)` with the original border rule. A row crossing
recomputes the complete sum. Final blocks with one/two/three valid pixels write
each exactly once; unused logical/physical channels retain their contracts.

The three successor updates are explicitly written to keep control flow simple.
The outgoing byte is part of the previous nonnegative window or zero; subtract
then add cannot underflow and each full u32 result obeys kw*255. Full-area and
rounding admission still bounds each horizontal sum. For a same-row update,
x<=width-2, so incoming<=width-1+r and outgoing>=-r, already covered by signed
admission. Block count uses count/4 plus a remainder flag, without adding three
to count. A valid block's first index is at most 0xfffffffc, so adding lanes 0..3
cannot wrap. The subtraction-based remaining-lane and stride guards avoid
wraparound at maximum u32 counts. Disjoint owners need no atomics or barriers.
The unchanged vertical stage retains exact final half-up rounding.

Below the existing 65,536-pixel cutoff, original scalar source and schedule are
retained inside the combined shader. For kernel widths below 4,099 the complete original
adjacent-pair schedule remains byte-identical. Both paths use a changed compiled
program, so unchanged scheduling is not zero overhead. The additional width cutoff
is chosen conservatively after the exploratory regressions below. These private
heuristics are not portable crossover guarantees. Large thin shapes and other
devices are not comprehensively timed. No public API, feature or dependency changes.

Only the existing unpacked horizontal WGSL changes in production. All production
Rust host code, admission, packed/serial selection, dispatch requests, five lazy
variants, layouts, pass/query counts, scratch/texture requests, pools, metadata
and explicit transfers retain their parent contracts. No buffer or pipeline is
added. Bandwidth, physical memory and driver compiler internals remain unmeasured.

## Rejected separate-program candidate

Before the four-output candidate, a host-selected six-variant prototype restored
the original small scalar shader and separated the previous two-output shader.
Four new tests plus five existing related tests passed (the four new tests also
ran separately); broad regression was not run for this rejected prototype.
It used the same 21-case warm method and five fresh-cache scenarios below.
Its small GPU comparisons were 2 favorable, 6 inconclusive and 2 adverse, failing
to establish the intended general small-input improvement. Mixed small/large
fresh-cache calls created three wide pipelines instead of two; both orders were
adverse in both surveys, parent/current ratios 0.817878-0.850561 (about 18-22% more
elapsed time). All-program cold calls also grew from five to six pipelines and
were adverse in both surveys. This did not justify additional retained objects.
The prototype was not adopted; its sources, executable and before/after hashes
are preserved in ignored local evidence. All 126 warm and 10 fresh-cache summaries,
including its favorable small-only cold cases and all adverse/control rows, are
retained in the appendix. Measurements are never pooled with the adopted source.

## Correctness and provenance

Four new tests verify four-output ownership, row crossings, forced strides,
all partial-tail lengths, exact cutoff and maximum-u32 index arithmetic; actual
parent/CPU outputs, profiler/query bytes, shared lazy caches, stable pipeline
identities, budgets, early errors, concurrent first use and retained owned outputs
after clearing pools and changed reuse. An additional raw matrix covers two-pixel
tails, widths 4,097/4,099/4,101 around the cutoff, both borders and maximum-width
full-u32 sums with dirty unused channels.
The existing raw 428-case and complete 448-case matrices continue to exercise
selected production shaders versus independent u64/prefix/brute references,
physical channels, guards, maximum rounding-safe kernels, receipts and metadata.

The immediate parent freezes the complete host planner/cache/encoder/call and
the original combined shader. Only private cache injection/test visibility
differ. Source audit verifies original production host/other shaders byte-for-byte,
the exact scalar prefix/fallback, original adjacent-pair body and final candidate
identity. Each rejected snapshot's source/executable digests match its measurement receipt separately.

BOX_WIDE_QUAD_PLAN: verified 264334 pixel owners, 49757 row-crossing fallbacks, forced strides, one/two/three-pixel tails, 65535/65536/65537 and maximum-u32 indices.

BOX_WIDE_QUAD_CACHE: verified shared scalar/quad lazy cache, exact cutoff, row crossings and one/two/three-pixel tails, 28 immediate-parent/CPU outputs, 3183072 additional raw/u64/guard words, two stages and unchanged receipts/metadata.

BOX_WIDE_QUAD_RESOURCES: verified unchanged mixed cache growth 2/2/3/5, stable pipeline identities, warm scratch/texture reuse, tiny/zero budgets and cold rejection.

BOX_WIDE_QUAD_CONCURRENT: verified simultaneous scalar/paired/packed/serial initialization, independent retained outputs and changed-input pool reuse.

There are 601 unchanged original GPU/image sources, 613 current source/manifest/executable SHA-256 values, 12 separately verified split snapshot files, 7 loop snapshot files and 8 probe artifacts. Current hashes agree before/after both warm/fresh-cache surveys and broad validation, including the ignored lockfile and executable. Repository/PR adapter-name matches=0; process names/PIDs remain ignored.

## Validation

| Check | Passed |
| --- | ---: |
| image | 178 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native test executions | 181 |

All-features/all-targets Clippy with warnings denied, CPU-only checks, minimal/CI-feature WASM with warnings denied, formatting and diff checks pass. Native suites include ignored correctness tests and skip explicit benchmarks.

## Measurement method

```text
cargo test -p spatialrust-gpu --all-features --release --lib benchmark_complete_box_wide_quad_calls -- --ignored --nocapture --test-threads=1
cargo test -p spatialrust-gpu --all-features --release --lib benchmark_cold_box_wide_quad_calls -- --ignored --nocapture --test-threads=1
```

Two consecutive initialized-runtime surveys use the same release executable,
27 cases, three warmup pairs and 30 measured parent/current pairs per route.
Resident/full order rotates through four balanced positions; GPU pairs alternate.
Every output is checked after timing. Preparation, checking, destruction and
recycling are excluded. No workflow build/test overlaps any timed survey;
shared-host work remains uncontrolled. No case is retried or filtered.

- `pass`: first-stage GPU timestamp in the complete profiled public call;
  horizontal for separable calls or single stage for serial controls.
- `resident`: completed public resident blur plus idle wait, excluding upload
  and checking readback.
- `full`: explicit upload, public blur and readback, including host transfer
  preparation and payload transfers.

Seven cases per survey use four-output wide work, thirteen retain the paired
schedule, five retain the scalar schedule, and two are unchanged packed/serial
controls. Both retained wide schedules still use a changed compiled program.
No GPU causal benefit is assigned to unchanged controls. Parent calls inject a
private pre-created cache; code layout, timer resolution and host load may affect
whole-call/control differences. This is not a same-condition competitor benchmark.

Ratios are paired geometric means of parent/current (>1 favors current). 95%
intervals bootstrap 10,000 resamples of 30 paired log ratios, seed
2026101817+1000*survey+10*case_index+route_index, zero-based indices, draws 249/9749.
Intervals above/below one are favorable/adverse, otherwise inconclusive. Zeros
remain undefined without epsilon and there is no multiplicity correction.
Boundary observations cannot establish isolation during every pair or portable,
universal non-regression/application-speed guarantees. All comparisons remain.

| Route | Four-output: 14 comparisons | Paired schedule: 26 | Scalar fallback: 10 | Original controls: 4 |
| --- | --- | --- | --- | --- |
| pass | 14 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1 favorable / 25 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 7 inconclusive / 3 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |
| resident | 13 favorable / 1 inconclusive / 0 adverse / 0 unresolved | 1 favorable / 25 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 9 inconclusive / 1 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |
| full | 8 favorable / 6 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 24 inconclusive / 2 adverse / 0 unresolved | 1 favorable / 9 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 4 inconclusive / 0 adverse / 0 unresolved |

All adverse adopted warm observations:

- Survey 1, 521x1, 4099x257, zero=False, scalar-fallback, pass: ratio 0.965320, interval 0.933077-0.995176.

- Survey 1, 255x257, 259x31, zero=False, scalar-fallback, pass: ratio 0.997865, interval 0.996533-0.999138.

- Survey 2, 1920x1080, 259x1, zero=False, paired-schedule, full: ratio 0.961324, interval 0.921941-0.995027.

- Survey 2, 1025x257, 259x9, zero=True, paired-schedule, full: ratio 0.960881, interval 0.922622-0.995601.

- Survey 2, 17x19, 259x31, zero=False, scalar-fallback, pass: ratio 0.959670, interval 0.939024-0.980467.

- Survey 2, 255x257, 259x31, zero=False, scalar-fallback, resident: ratio 0.910521, interval 0.840207-0.987678.

## Fresh application caches

Two separate surveys use five scenarios and 20 alternating measured pairs each,
without warmup or retries. Application pipeline cache and resource pools are
cleared before each variant outside timing. Upload and expected bytes are also
prepared outside timing. The timed region contains all public resident calls
and each idle wait; outputs stay owned until after timing. Runtime/device
creation, checking readback, destruction and recycling are excluded. The device
is retained and driver caches are not reset: process-cold startup, physical
pipeline memory and absolute first-process compilation costs are unmeasured.
The large jobs use 256x256, 4099x31 while small jobs use 17x19, 259x31. The
cache-count assertions remain 2/2 for one/mixed wide scenarios and 5/5 for
all-program scenarios. Bootstrap uses 20 paired logs, 10,000 draws, seed
2026101818+1000*survey+scenario_index, draws 249/9749 and the same limitations.

| Survey | Scenario | Parent/current pipelines | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | small | 2/2 | 1.492450 | 1.670500 | 0.968924 | 0.865131-1.175234 | 0/0 | inconclusive |
| 1 | large | 2/2 | 1.669600 | 1.767950 | 0.933136 | 0.904423-0.963124 | 0/0 | adverse |
| 1 | small_then_large | 2/2 | 1.876600 | 2.016850 | 0.918675 | 0.902383-0.933705 | 0/0 | adverse |
| 1 | large_then_small | 2/2 | 1.810050 | 1.991050 | 0.903780 | 0.867141-0.932780 | 0/0 | adverse |
| 1 | all_programs | 5/5 | 4.166050 | 4.242550 | 0.984416 | 0.939946-1.024416 | 0/0 | inconclusive |
| 2 | small | 2/2 | 1.468850 | 1.677000 | 0.965209 | 0.874931-1.160600 | 0/0 | inconclusive |
| 2 | large | 2/2 | 1.605500 | 1.759500 | 0.912954 | 0.881451-0.947591 | 0/0 | adverse |
| 2 | small_then_large | 2/2 | 1.764200 | 1.990150 | 0.893594 | 0.874621-0.912838 | 0/0 | adverse |
| 2 | large_then_small | 2/2 | 1.726450 | 1.944350 | 0.886059 | 0.873237-0.899545 | 0/0 | adverse |
| 2 | all_programs | 5/5 | 4.070500 | 4.242600 | 0.954799 | 0.914414-0.986866 | 0/0 | adverse |

Unchanged pipeline counts do not imply zero initialization cost. Fresh-cache comparisons are 7 adverse and 3 inconclusive: the cutoff-sized large scenario and both mixed orders are adverse in both surveys (parent/current ratios 0.886059-0.933136, about 7-13% more elapsed time); all-program survey 2 is also adverse. The small-only scenarios are inconclusive. These measurements do not isolate compiler work from other first-call work and do not time first use at every image size.

## Boundary diagnostics

Two-second observations publish busy process counts/CPU cores only. GPU/controller values are utilization percentages, not allocated memory. No sample covers every timed pair.

| Candidate/phase | Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Quiet |
| --- | --- | --- | ---: | --- | --- | --- |
| adopted/warm | 1 | pre | 1 | 0.494 | [[[0,0]],[[0,0]]] | False |
| adopted/warm | 1 | post | 1 | 0.286 | [[[0,0]],[[0,0]]] | False |
| adopted/warm | 2 | pre | 1 | 0.300 | [[[0,0]],[[0,0]]] | False |
| adopted/warm | 2 | post | 1 | 0.268 | [[[0,0]],[[0,0]]] | False |
| adopted/fresh-cache | 1 | pre | 1 | 0.877 | [[[0,0]],[[0,0]]] | False |
| adopted/fresh-cache | 1 | post | 1 | 0.274 | [[[3,0]],[[0,0]]] | False |
| adopted/fresh-cache | 2 | pre | 1 | 0.348 | [[[0,0]],[[0,0]]] | False |
| adopted/fresh-cache | 2 | post | 0 | none | [[[3,0]],[[0,0]]] | True |
| rejected/warm | 1 | pre | 0 | none | [[[0,0]],[[0,0]]] | True |
| rejected/warm | 1 | post | 1 | 0.441 | [[[0,0]],[[0,0]]] | False |
| rejected/warm | 2 | pre | 1 | 0.696 | [[[0,0]],[[0,0]]] | False |
| rejected/warm | 2 | post | 1 | 0.253 | [[[0,0]],[[0,0]]] | False |
| rejected/fresh-cache | 1 | pre | 1 | 0.300 | [[[0,0]],[[0,0]]] | False |
| rejected/fresh-cache | 1 | post | 1 | 0.267 | [[[2,0]],[[0,0]]] | False |
| rejected/fresh-cache | 2 | pre | 0 | none | [[[0,0]],[[0,0]]] | True |
| rejected/fresh-cache | 2 | post | 1 | 0.331 | [[[4,0]],[[0,0]]] | False |
| rejected-loop/warm | 1 | pre | 1 | 0.385 | [[[0,0]],[[0,0]]] | False |
| rejected-loop/warm | 1 | post | 1 | 0.286 | [[[0,0]],[[0,0]]] | False |
| rejected-loop/warm | 2 | pre | 1 | 0.463 | [[[0,0]],[[0,0]]] | False |
| rejected-loop/warm | 2 | post | 0 | none | [[[0,0]],[[0,0]]] | True |

Environment adopted/warm/1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

Environment adopted/warm/2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

Environment adopted/fresh-cache/1: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`.

Environment adopted/fresh-cache/2: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`.

Environment rejected/warm/1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

Environment rejected/warm/2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

Environment rejected/fresh-cache/1: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`.

Environment rejected/fresh-cache/2: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`.

Environment rejected-loop/warm/1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

Environment rejected-loop/warm/2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=first_pass_resident_full wall_order=balanced first_order=alternating checks_drop_recycle_excluded=true`.

## All adopted warm comparisons

| Survey | Image | kw | kh | Zero border | Group | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16x17 | 3 | 3 | False | control | pass | 0.005408 | 0.005440 | 0.989990 | 0.964648-1.016716 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | control | resident | 0.166650 | 0.165150 | 1.012691 | 0.969051-1.056483 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | control | full | 0.957800 | 0.958800 | 0.968609 | 0.887154-1.021934 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | pass | 0.072256 | 0.072624 | 0.997528 | 0.993467-1.001698 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | resident | 0.263000 | 0.260600 | 1.003690 | 0.987417-1.019813 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | full | 1.087000 | 1.097050 | 0.997657 | 0.979463-1.015699 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | paired-schedule | pass | 1.669648 | 1.669248 | 0.999640 | 0.998601-1.000368 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | paired-schedule | resident | 2.264850 | 2.323250 | 0.998659 | 0.986195-1.012225 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | paired-schedule | full | 8.014550 | 8.037050 | 1.004886 | 0.993128-1.018838 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | paired-schedule | pass | 1.577232 | 1.578000 | 0.999865 | 0.999323-1.000406 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | paired-schedule | resident | 2.126700 | 2.124950 | 1.000396 | 0.984917-1.016229 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | paired-schedule | full | 7.955500 | 8.017850 | 0.994349 | 0.983143-1.007144 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | paired-schedule | pass | 1.577312 | 1.577488 | 0.999337 | 0.998588-1.000036 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | paired-schedule | resident | 2.127950 | 2.146950 | 0.992872 | 0.978503-1.007345 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | paired-schedule | full | 8.049000 | 8.088750 | 1.012988 | 0.990244-1.048044 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | False | paired-schedule | pass | 1.577088 | 1.577728 | 1.000003 | 0.999368-1.000617 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | False | paired-schedule | resident | 2.286450 | 2.319050 | 0.998189 | 0.972542-1.038537 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | False | paired-schedule | full | 8.150050 | 8.209400 | 1.006069 | 0.981994-1.035511 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 81 | False | paired-schedule | pass | 1.576768 | 1.577296 | 0.999650 | 0.999074-1.000221 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 81 | False | paired-schedule | resident | 2.604450 | 2.606050 | 0.991663 | 0.977342-1.005279 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 81 | False | paired-schedule | full | 8.443300 | 8.472400 | 0.972748 | 0.930339-1.006389 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | paired-schedule | pass | 2.925392 | 2.925680 | 0.999945 | 0.999459-1.000449 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | paired-schedule | resident | 3.686550 | 3.708550 | 0.996763 | 0.986105-1.007129 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | paired-schedule | full | 9.741050 | 9.746050 | 0.999480 | 0.989165-1.009650 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | pass | 11.254544 | 5.689104 | 1.978291 | 1.978129-1.978459 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | resident | 11.979850 | 6.417250 | 1.866103 | 1.857400-1.875526 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | full | 18.624200 | 13.028650 | 1.420085 | 1.404011-1.435945 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | paired-schedule | pass | 1.560736 | 1.560272 | 1.000667 | 1.000135-1.001197 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | paired-schedule | resident | 2.250850 | 2.225850 | 1.004094 | 0.992921-1.015022 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | True | paired-schedule | full | 8.078200 | 8.096150 | 0.988705 | 0.976100-1.000197 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 3 | False | paired-schedule | pass | 6.486144 | 6.485984 | 0.999822 | 0.998863-1.000667 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 3 | False | paired-schedule | resident | 7.428050 | 7.418200 | 0.997377 | 0.986661-1.008227 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 3 | False | paired-schedule | full | 25.519050 | 25.809650 | 0.982596 | 0.935261-1.032082 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 31 | False | paired-schedule | pass | 6.533792 | 6.538368 | 0.999549 | 0.998705-1.000375 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 31 | False | paired-schedule | resident | 8.161700 | 8.204750 | 0.992396 | 0.984149-1.000542 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 31 | False | paired-schedule | full | 26.486050 | 26.107150 | 1.031809 | 0.977731-1.090140 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | paired-schedule | pass | 6.541280 | 6.536832 | 0.999999 | 0.999342-1.000628 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | paired-schedule | resident | 9.546850 | 9.545300 | 1.001120 | 0.988848-1.014698 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | paired-schedule | full | 28.321600 | 28.238400 | 1.010533 | 0.968410-1.051408 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | paired-schedule | pass | 0.211488 | 0.211632 | 0.999846 | 0.997364-1.002424 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | paired-schedule | resident | 0.572450 | 0.562700 | 1.007816 | 0.984985-1.031375 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | paired-schedule | full | 5.191800 | 5.175550 | 1.005423 | 0.987198-1.024850 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | paired-schedule | pass | 0.158592 | 0.157472 | 1.004218 | 0.994158-1.013572 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | paired-schedule | resident | 0.567800 | 0.561500 | 1.022334 | 0.981079-1.069287 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | paired-schedule | full | 4.801900 | 4.828800 | 1.004141 | 0.978339-1.033853 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.016000 | 0.016240 | 0.984116 | 0.954433-1.015715 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.465600 | 0.466050 | 1.004167 | 0.858951-1.174359 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | full | 4.944050 | 5.039650 | 0.998884 | 0.963313-1.031715 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.014144 | 0.014208 | 0.969594 | 0.919408-1.002674 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.594800 | 0.629000 | 0.906492 | 0.799167-1.024423 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.247650 | 5.227900 | 1.017261 | 0.989051-1.044485 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.127872 | 0.128848 | 0.965320 | 0.933077-0.995176 | 0/0 | adverse |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.694800 | 0.696400 | 0.989338 | 0.950468-1.027487 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 5.376050 | 5.445150 | 1.003397 | 0.963963-1.047804 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.015120 | 0.015264 | 0.976215 | 0.926307-1.009209 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.638600 | 0.636750 | 0.904241 | 0.792956-1.027690 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | full | 5.423600 | 5.572500 | 0.978473 | 0.950081-1.007062 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.475664 | 0.477040 | 0.997865 | 0.996533-0.999138 | 0/0 | adverse |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.130350 | 1.132600 | 0.978794 | 0.942173-1.004715 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | full | 5.425700 | 5.492650 | 0.979996 | 0.946623-1.012552 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | paired-schedule | pass | 0.288128 | 0.288304 | 0.999613 | 0.998304-1.000928 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | paired-schedule | resident | 1.028150 | 0.994850 | 1.066819 | 1.013883-1.145942 | 0/0 | favorable |
| 1 | 256x256 | 259 | 31 | False | paired-schedule | full | 5.464500 | 5.595200 | 0.983232 | 0.953491-1.016496 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | True | four-output | pass | 11.325984 | 5.714624 | 1.981901 | 1.981741-1.982065 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | True | four-output | resident | 11.952050 | 6.375450 | 1.881038 | 1.863297-1.898095 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | True | four-output | full | 19.415650 | 13.409250 | 1.403559 | 1.305682-1.509935 | 0/0 | favorable |
| 1 | 3840x2160 | 4099 | 3 | False | four-output | pass | 72.690976 | 60.012752 | 1.211078 | 1.210606-1.211556 | 0/0 | favorable |
| 1 | 3840x2160 | 4099 | 3 | False | four-output | resident | 73.874750 | 61.157050 | 1.207536 | 1.206057-1.208993 | 0/0 | favorable |
| 1 | 3840x2160 | 4099 | 3 | False | four-output | full | 96.698450 | 82.185500 | 1.171274 | 1.151591-1.190789 | 0/0 | favorable |
| 1 | 3840x2160 | 4099 | 31 | False | four-output | pass | 72.711888 | 60.077536 | 1.209894 | 1.209171-1.210544 | 0/0 | favorable |
| 1 | 3840x2160 | 4099 | 31 | False | four-output | resident | 74.384000 | 61.767850 | 1.204193 | 1.202229-1.206068 | 0/0 | favorable |
| 1 | 3840x2160 | 4099 | 31 | False | four-output | full | 96.148350 | 83.004650 | 1.158765 | 1.144426-1.173709 | 0/0 | favorable |
| 1 | 1025x257 | 4099 | 9 | True | four-output | pass | 0.858992 | 0.512032 | 1.676435 | 1.662342-1.690936 | 0/0 | favorable |
| 1 | 1025x257 | 4099 | 9 | True | four-output | resident | 1.237350 | 0.885700 | 1.390552 | 1.351971-1.424852 | 0/0 | favorable |
| 1 | 1025x257 | 4099 | 9 | True | four-output | full | 5.202850 | 5.159800 | 1.009656 | 0.980511-1.039235 | 0/0 | inconclusive |
| 1 | 257x259 | 4099 | 257 | False | four-output | pass | 0.233392 | 0.168768 | 1.361887 | 1.316883-1.393021 | 0/0 | favorable |
| 1 | 257x259 | 4099 | 257 | False | four-output | resident | 0.537350 | 0.490300 | 1.090766 | 1.043187-1.141277 | 0/0 | favorable |
| 1 | 257x259 | 4099 | 257 | False | four-output | full | 5.431800 | 5.409650 | 1.025144 | 0.993241-1.058918 | 0/0 | inconclusive |
| 1 | 256x256 | 4099 | 31 | False | four-output | pass | 0.266544 | 0.186992 | 1.441768 | 1.411893-1.487109 | 0/0 | favorable |
| 1 | 256x256 | 4099 | 31 | False | four-output | resident | 0.800400 | 0.770850 | 1.084028 | 0.985554-1.198351 | 0/0 | inconclusive |
| 1 | 256x256 | 4099 | 31 | False | four-output | full | 5.723050 | 5.749050 | 0.992393 | 0.951040-1.034105 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | pass | 0.005472 | 0.005440 | 1.000029 | 0.972140-1.027887 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | resident | 0.170350 | 0.174450 | 0.994522 | 0.955073-1.033383 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | full | 0.987550 | 0.995050 | 1.008546 | 0.985442-1.031023 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | pass | 0.071488 | 0.071728 | 0.999231 | 0.995127-1.003399 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | resident | 0.279950 | 0.283450 | 1.011802 | 0.980589-1.045867 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | full | 1.173350 | 1.169150 | 1.004072 | 0.989921-1.017814 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | paired-schedule | pass | 1.616544 | 1.616512 | 1.000461 | 0.999636-1.001495 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | paired-schedule | resident | 2.283250 | 2.324050 | 0.987458 | 0.968272-1.005496 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | paired-schedule | full | 7.934700 | 8.083500 | 0.961324 | 0.921941-0.995027 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 3 | False | paired-schedule | pass | 1.590976 | 1.591648 | 0.999580 | 0.998898-1.000258 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | paired-schedule | resident | 2.131950 | 2.135900 | 1.007118 | 0.995615-1.019076 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | paired-schedule | full | 7.940050 | 7.929950 | 0.999486 | 0.986600-1.012967 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | paired-schedule | pass | 1.590736 | 1.591152 | 0.999671 | 0.999065-1.000265 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | paired-schedule | resident | 2.214000 | 2.208700 | 1.006474 | 0.990413-1.023571 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | paired-schedule | full | 8.137850 | 8.247100 | 0.987448 | 0.967550-1.005686 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | False | paired-schedule | pass | 1.590272 | 1.591344 | 0.999445 | 0.998734-1.000157 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | False | paired-schedule | resident | 2.273900 | 2.271900 | 1.013068 | 0.986698-1.053773 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | False | paired-schedule | full | 8.186600 | 8.156150 | 1.008934 | 0.929576-1.089285 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 81 | False | paired-schedule | pass | 1.584528 | 1.583824 | 1.000249 | 0.999648-1.000853 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 81 | False | paired-schedule | resident | 2.614800 | 2.622150 | 0.999382 | 0.990781-1.008433 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 81 | False | paired-schedule | full | 8.501750 | 8.459700 | 1.006225 | 0.994327-1.019543 | 0/0 | inconclusive |
| 2 | 1920x1080 | 511 | 31 | False | paired-schedule | pass | 2.949824 | 2.949568 | 1.000028 | 0.999579-1.000461 | 0/0 | inconclusive |
| 2 | 1920x1080 | 511 | 31 | False | paired-schedule | resident | 3.725650 | 3.753050 | 0.992946 | 0.984390-1.001275 | 0/0 | inconclusive |
| 2 | 1920x1080 | 511 | 31 | False | paired-schedule | full | 9.782850 | 9.752650 | 1.005439 | 0.988754-1.025974 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | pass | 11.347248 | 5.736352 | 1.979187 | 1.978198-1.980405 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | resident | 12.003200 | 6.425500 | 1.874835 | 1.866659-1.883316 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | full | 18.201850 | 12.700050 | 1.437534 | 1.426417-1.448040 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | True | paired-schedule | pass | 1.574480 | 1.574400 | 0.999986 | 0.999460-1.000529 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | paired-schedule | resident | 2.264300 | 2.246700 | 1.000875 | 0.980480-1.018185 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | paired-schedule | full | 8.097150 | 8.104050 | 1.003283 | 0.989855-1.017762 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | paired-schedule | pass | 6.542512 | 6.542624 | 0.999921 | 0.999025-1.000833 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | paired-schedule | resident | 7.508800 | 7.477100 | 1.010227 | 0.995640-1.024953 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | paired-schedule | full | 25.619200 | 25.828850 | 1.014294 | 0.955337-1.077980 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 31 | False | paired-schedule | pass | 6.546112 | 6.541792 | 1.000647 | 0.999792-1.001512 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 31 | False | paired-schedule | resident | 8.161700 | 8.164150 | 0.993131 | 0.982655-1.003968 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 31 | False | paired-schedule | full | 26.301650 | 26.150950 | 0.984721 | 0.930196-1.038883 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | paired-schedule | pass | 6.546912 | 6.542384 | 1.000440 | 0.999714-1.001124 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | paired-schedule | resident | 9.553800 | 9.596950 | 0.993109 | 0.982623-1.003872 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | paired-schedule | full | 27.943750 | 27.876500 | 0.999489 | 0.944847-1.055838 | 0/0 | inconclusive |
| 2 | 1025x257 | 259 | 9 | True | paired-schedule | pass | 0.211856 | 0.211968 | 0.999427 | 0.996912-1.002153 | 0/0 | inconclusive |
| 2 | 1025x257 | 259 | 9 | True | paired-schedule | resident | 0.576600 | 0.580750 | 0.997708 | 0.967330-1.030954 | 0/0 | inconclusive |
| 2 | 1025x257 | 259 | 9 | True | paired-schedule | full | 5.627000 | 5.795150 | 0.960881 | 0.922622-0.995601 | 0/0 | adverse |
| 2 | 257x259 | 259 | 257 | False | paired-schedule | pass | 0.196208 | 0.196160 | 0.999787 | 0.994001-1.006072 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | paired-schedule | resident | 0.663600 | 0.659750 | 1.008709 | 0.966826-1.049903 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | paired-schedule | full | 5.272300 | 5.144250 | 1.031005 | 0.999308-1.074928 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.017824 | 0.018688 | 0.959670 | 0.939024-0.980467 | 0/0 | adverse |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.482700 | 0.461250 | 0.986166 | 0.850935-1.124744 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | full | 5.403150 | 5.336150 | 1.016341 | 0.983400-1.050905 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.014192 | 0.014240 | 0.990118 | 0.960331-1.023962 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.841200 | 0.727000 | 1.033139 | 0.914442-1.161491 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.598300 | 5.682450 | 0.994142 | 0.964355-1.031788 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.136768 | 0.135648 | 1.024765 | 0.986801-1.062491 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.722300 | 0.711800 | 1.065250 | 0.992317-1.177357 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 5.452850 | 5.550400 | 0.976159 | 0.921092-1.014641 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.018000 | 0.018016 | 1.006636 | 0.975522-1.040511 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.521750 | 0.527000 | 1.011794 | 0.916283-1.119308 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | full | 5.607100 | 5.593700 | 1.058416 | 1.000279-1.142201 | 0/0 | favorable |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.475568 | 0.475840 | 0.999928 | 0.999037-1.000966 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.251450 | 1.452000 | 0.910521 | 0.840207-0.987678 | 0/0 | adverse |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | full | 5.663650 | 5.613200 | 1.011145 | 0.988627-1.034490 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | paired-schedule | pass | 0.287776 | 0.288048 | 0.999892 | 0.998745-1.001046 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | paired-schedule | resident | 1.041200 | 0.998750 | 1.045145 | 0.987717-1.139731 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | paired-schedule | full | 5.651300 | 5.634350 | 1.013823 | 0.980506-1.049270 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | True | four-output | pass | 11.326160 | 5.715536 | 1.982106 | 1.981575-1.983012 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | True | four-output | resident | 11.997800 | 6.385850 | 1.868252 | 1.836420-1.894700 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | True | four-output | full | 18.977750 | 12.837300 | 1.417808 | 1.311605-1.526539 | 0/0 | favorable |
| 2 | 3840x2160 | 4099 | 3 | False | four-output | pass | 73.301664 | 60.535840 | 1.210557 | 1.209917-1.211188 | 0/0 | favorable |
| 2 | 3840x2160 | 4099 | 3 | False | four-output | resident | 73.823650 | 61.103400 | 1.207583 | 1.206102-1.209114 | 0/0 | favorable |
| 2 | 3840x2160 | 4099 | 3 | False | four-output | full | 95.105200 | 81.151350 | 1.166240 | 1.149291-1.182367 | 0/0 | favorable |
| 2 | 3840x2160 | 4099 | 31 | False | four-output | pass | 73.283792 | 60.547504 | 1.210268 | 1.209678-1.210856 | 0/0 | favorable |
| 2 | 3840x2160 | 4099 | 31 | False | four-output | resident | 75.050400 | 62.250300 | 1.204246 | 1.201941-1.206360 | 0/0 | favorable |
| 2 | 3840x2160 | 4099 | 31 | False | four-output | full | 97.225550 | 83.612600 | 1.159090 | 1.138730-1.179343 | 0/0 | favorable |
| 2 | 1025x257 | 4099 | 9 | True | four-output | pass | 0.869232 | 0.519248 | 1.681162 | 1.671549-1.691168 | 0/0 | favorable |
| 2 | 1025x257 | 4099 | 9 | True | four-output | resident | 1.270950 | 0.907400 | 1.378995 | 1.339325-1.411481 | 0/0 | favorable |
| 2 | 1025x257 | 4099 | 9 | True | four-output | full | 5.274900 | 5.298650 | 0.962559 | 0.896445-1.009043 | 0/0 | inconclusive |
| 2 | 257x259 | 4099 | 257 | False | four-output | pass | 0.221968 | 0.163536 | 1.351957 | 1.342889-1.360884 | 0/0 | favorable |
| 2 | 257x259 | 4099 | 257 | False | four-output | resident | 0.490450 | 0.443350 | 1.116585 | 1.076576-1.158931 | 0/0 | favorable |
| 2 | 257x259 | 4099 | 257 | False | four-output | full | 5.145450 | 5.152850 | 1.012566 | 0.998201-1.026842 | 0/0 | inconclusive |
| 2 | 256x256 | 4099 | 31 | False | four-output | pass | 0.248400 | 0.176576 | 1.439495 | 1.407121-1.488320 | 0/0 | favorable |
| 2 | 256x256 | 4099 | 31 | False | four-output | resident | 0.798050 | 0.719200 | 1.129440 | 1.056103-1.225006 | 0/0 | favorable |
| 2 | 256x256 | 4099 | 31 | False | four-output | full | 5.538100 | 5.480400 | 1.009802 | 0.978983-1.042588 | 0/0 | inconclusive |

## Rejected all-width four-output loop candidate

This preliminary candidate used four-output work for every large wide input. Four new and five legacy related tests passed, with the new tests also run separately. Broad regression and cold measurements were not run. All 126 warm comparisons use the same 21 cases and method, seed 2026101815+1000*survey+10*case_index+route_index. GPU/resident/full comparisons include30/22/20 adverse intervals respectively. These observations led to width-limited work rather than a universal four-output rule.

| Survey | Image | kw | kh | Zero border | Group | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16x17 | 3 | 3 | False | control | pass | 0.005440 | 0.005408 | 1.010818 | 0.976586-1.044933 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | control | resident | 0.207600 | 0.209300 | 1.011220 | 0.970098-1.054014 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | control | full | 1.177450 | 1.199950 | 0.959426 | 0.900276-1.004992 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | pass | 0.072400 | 0.072496 | 0.999535 | 0.995291-1.004034 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | resident | 0.263250 | 0.261900 | 0.990297 | 0.961785-1.018180 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | full | 1.097650 | 1.100700 | 1.004391 | 0.981862-1.031208 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | four-output | pass | 1.662592 | 2.103968 | 0.789105 | 0.788071-0.790112 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 1 | False | four-output | resident | 2.523400 | 2.999650 | 0.835380 | 0.822357-0.848262 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 1 | False | four-output | full | 8.647650 | 9.152250 | 0.938767 | 0.890730-0.990998 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 3 | False | four-output | pass | 1.577056 | 1.994080 | 0.790706 | 0.789706-0.791725 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 3 | False | four-output | resident | 2.145000 | 2.515750 | 0.852869 | 0.839971-0.868151 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 3 | False | four-output | full | 8.030000 | 8.468900 | 0.926452 | 0.886843-0.952561 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 9 | False | four-output | pass | 1.577072 | 1.994416 | 0.790699 | 0.789896-0.791503 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 9 | False | four-output | resident | 2.174300 | 2.630350 | 0.827676 | 0.813631-0.842021 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 9 | False | four-output | full | 8.395950 | 8.730100 | 0.944380 | 0.899967-0.985893 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 31 | False | four-output | pass | 1.576944 | 1.994384 | 0.790745 | 0.790084-0.791401 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 31 | False | four-output | resident | 2.319900 | 2.702050 | 0.863104 | 0.841707-0.897540 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 31 | False | four-output | full | 8.344750 | 8.802000 | 0.954444 | 0.937649-0.970068 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 81 | False | four-output | pass | 1.576336 | 1.995248 | 0.790022 | 0.789194-0.790845 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 81 | False | four-output | resident | 2.617700 | 3.078000 | 0.853555 | 0.844474-0.862671 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 81 | False | four-output | full | 8.558500 | 9.077900 | 0.951547 | 0.940801-0.962344 | 0/0 | adverse |
| 1 | 1920x1080 | 511 | 31 | False | four-output | pass | 2.925024 | 3.411120 | 0.857489 | 0.856695-0.858233 | 0/0 | adverse |
| 1 | 1920x1080 | 511 | 31 | False | four-output | resident | 3.675500 | 4.127850 | 0.890145 | 0.883555-0.896472 | 0/0 | adverse |
| 1 | 1920x1080 | 511 | 31 | False | four-output | full | 9.702600 | 10.221400 | 0.949830 | 0.938009-0.961179 | 0/0 | adverse |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | pass | 11.254464 | 5.688736 | 1.978942 | 1.978253-1.980120 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | resident | 11.910800 | 6.369750 | 1.871317 | 1.865239-1.877691 | 0/0 | favorable |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | full | 18.191200 | 12.575100 | 1.446786 | 1.431963-1.460402 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 31 | True | four-output | pass | 1.567312 | 1.988608 | 0.787920 | 0.787019-0.788826 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 31 | True | four-output | resident | 2.264150 | 2.675850 | 0.844332 | 0.829282-0.860035 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 31 | True | four-output | full | 8.273650 | 8.712200 | 0.945231 | 0.924516-0.962767 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 3 | False | four-output | pass | 6.492176 | 8.224144 | 0.789330 | 0.788376-0.790169 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 3 | False | four-output | resident | 7.422650 | 9.177050 | 0.808619 | 0.803246-0.814397 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 3 | False | four-output | full | 25.740550 | 27.671650 | 0.948227 | 0.901832-0.996517 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 31 | False | four-output | pass | 6.535168 | 8.283920 | 0.789265 | 0.788676-0.789906 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 31 | False | four-output | resident | 8.154050 | 9.927300 | 0.821848 | 0.815109-0.828347 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 31 | False | four-output | full | 26.439050 | 28.017500 | 0.959845 | 0.922720-0.998064 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 81 | False | four-output | pass | 6.538320 | 8.283936 | 0.789253 | 0.788632-0.789864 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 81 | False | four-output | resident | 9.545050 | 11.314000 | 0.845876 | 0.838875-0.853093 | 0/0 | adverse |
| 1 | 3840x2160 | 259 | 81 | False | four-output | full | 27.811700 | 29.644800 | 0.934521 | 0.897354-0.972277 | 0/0 | adverse |
| 1 | 1025x257 | 259 | 9 | True | four-output | pass | 0.211968 | 0.247248 | 0.848052 | 0.840427-0.854855 | 0/0 | adverse |
| 1 | 1025x257 | 259 | 9 | True | four-output | resident | 0.596100 | 0.619500 | 0.968465 | 0.918382-1.030569 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | four-output | full | 5.339150 | 5.466950 | 0.948131 | 0.907104-0.986310 | 0/0 | adverse |
| 1 | 257x259 | 259 | 257 | False | four-output | pass | 0.200416 | 0.213024 | 0.931713 | 0.925528-0.938735 | 0/0 | adverse |
| 1 | 257x259 | 259 | 257 | False | four-output | resident | 0.463150 | 0.463600 | 0.983886 | 0.945243-1.028042 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | four-output | full | 4.945150 | 4.914900 | 1.003144 | 0.968960-1.039247 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.017472 | 0.018272 | 0.971479 | 0.955754-0.987545 | 0/0 | adverse |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.594400 | 0.578350 | 1.095060 | 0.907147-1.305421 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | full | 5.268900 | 5.401450 | 0.982388 | 0.955883-1.008948 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.013904 | 0.014208 | 0.982256 | 0.965553-1.000852 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.626200 | 0.629050 | 1.118413 | 0.987302-1.272360 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.423350 | 5.438850 | 1.014186 | 0.981939-1.048048 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.127776 | 0.128032 | 0.991947 | 0.975958-1.007611 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.709050 | 0.698200 | 1.011022 | 0.978588-1.044203 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 5.307650 | 5.307350 | 1.000034 | 0.974052-1.029593 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.018192 | 0.018176 | 0.996206 | 0.971679-1.022217 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.632150 | 0.642950 | 0.988539 | 0.873209-1.123911 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | full | 5.526050 | 5.357050 | 1.040342 | 0.996875-1.085848 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.475632 | 0.475728 | 1.000055 | 0.999440-1.000724 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.245900 | 1.249700 | 1.055510 | 0.973037-1.146044 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | full | 5.444350 | 5.445300 | 1.013596 | 0.969928-1.065304 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | four-output | pass | 0.287936 | 0.310256 | 0.927252 | 0.923941-0.929508 | 0/0 | adverse |
| 1 | 256x256 | 259 | 31 | False | four-output | resident | 1.041950 | 1.035600 | 1.031084 | 0.983283-1.105395 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | four-output | full | 5.460500 | 5.449850 | 1.010662 | 0.979969-1.041941 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | pass | 0.005424 | 0.005424 | 1.006580 | 0.985071-1.027639 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | resident | 0.176250 | 0.175200 | 0.996599 | 0.955983-1.033233 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | full | 1.000600 | 1.007100 | 0.988446 | 0.969353-1.007833 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | pass | 0.071504 | 0.071728 | 0.997417 | 0.992643-1.002244 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | resident | 0.263250 | 0.265000 | 0.986804 | 0.953856-1.014936 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | full | 1.114200 | 1.107200 | 1.000846 | 0.989317-1.012876 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | four-output | pass | 1.580000 | 1.998800 | 0.790754 | 0.789863-0.791802 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 1 | False | four-output | resident | 2.332450 | 2.751950 | 0.824258 | 0.815315-0.833905 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 1 | False | four-output | full | 8.148150 | 8.793750 | 0.949505 | 0.923145-0.983906 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 3 | False | four-output | pass | 1.577072 | 1.997936 | 0.789957 | 0.789147-0.790773 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 3 | False | four-output | resident | 2.163150 | 2.585950 | 0.845098 | 0.829685-0.863076 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 3 | False | four-output | full | 8.191050 | 8.604850 | 0.932824 | 0.894512-0.962850 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 9 | False | four-output | pass | 1.580080 | 1.998112 | 0.790139 | 0.788999-0.791163 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 9 | False | four-output | resident | 2.167700 | 2.616700 | 0.842687 | 0.815934-0.879098 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 9 | False | four-output | full | 8.224200 | 8.609750 | 0.957357 | 0.934302-0.983480 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 31 | False | four-output | pass | 1.590448 | 2.012544 | 0.789517 | 0.788471-0.790557 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 31 | False | four-output | resident | 2.293000 | 2.720100 | 0.854509 | 0.832196-0.889649 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 31 | False | four-output | full | 8.245300 | 8.690200 | 0.943872 | 0.928855-0.958651 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 81 | False | four-output | pass | 1.589344 | 2.010256 | 0.790640 | 0.789946-0.791315 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 81 | False | four-output | resident | 2.669150 | 3.109300 | 0.864280 | 0.848054-0.884010 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 81 | False | four-output | full | 8.831650 | 9.296150 | 0.964108 | 0.948181-0.984188 | 0/0 | adverse |
| 2 | 1920x1080 | 511 | 31 | False | four-output | pass | 2.948896 | 3.444432 | 0.856309 | 0.855326-0.857237 | 0/0 | adverse |
| 2 | 1920x1080 | 511 | 31 | False | four-output | resident | 3.731600 | 4.232350 | 0.874170 | 0.863616-0.884638 | 0/0 | adverse |
| 2 | 1920x1080 | 511 | 31 | False | four-output | full | 9.916750 | 10.373000 | 0.935628 | 0.902518-0.957547 | 0/0 | adverse |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | pass | 11.346176 | 5.734944 | 1.978406 | 1.978245-1.978584 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | resident | 11.877100 | 6.277850 | 1.890828 | 1.887121-1.894610 | 0/0 | favorable |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | full | 17.889200 | 12.275200 | 1.458279 | 1.448311-1.468453 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 31 | True | four-output | pass | 1.573280 | 1.997536 | 0.787511 | 0.786649-0.788358 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 31 | True | four-output | resident | 2.166150 | 2.589000 | 0.838078 | 0.831260-0.844975 | 0/0 | adverse |
| 2 | 1920x1080 | 259 | 31 | True | four-output | full | 7.972450 | 8.366050 | 0.949014 | 0.936637-0.962388 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 3 | False | four-output | pass | 6.534160 | 8.290320 | 0.788641 | 0.788075-0.789226 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 3 | False | four-output | resident | 7.377650 | 9.119350 | 0.810750 | 0.806161-0.816458 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 3 | False | four-output | full | 25.316550 | 27.043750 | 0.916957 | 0.898333-0.932324 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 31 | False | four-output | pass | 6.542160 | 8.288416 | 0.789252 | 0.788687-0.789816 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 31 | False | four-output | resident | 8.167550 | 9.959800 | 0.816923 | 0.809680-0.824138 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 31 | False | four-output | full | 26.856700 | 28.818000 | 0.962656 | 0.914601-1.014470 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | four-output | pass | 6.540272 | 8.282272 | 0.789138 | 0.788539-0.789704 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 81 | False | four-output | resident | 9.566100 | 11.309800 | 0.847401 | 0.841458-0.853873 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 81 | False | four-output | full | 27.788300 | 29.511650 | 0.948820 | 0.912255-0.989865 | 0/0 | adverse |
| 2 | 1025x257 | 259 | 9 | True | four-output | pass | 0.212224 | 0.249648 | 0.839400 | 0.831108-0.847113 | 0/0 | adverse |
| 2 | 1025x257 | 259 | 9 | True | four-output | resident | 0.572250 | 0.643850 | 0.889309 | 0.854315-0.925356 | 0/0 | adverse |
| 2 | 1025x257 | 259 | 9 | True | four-output | full | 5.441350 | 5.391100 | 1.015933 | 0.967120-1.090327 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | four-output | pass | 0.203824 | 0.211680 | 0.936611 | 0.921492-0.949883 | 0/0 | adverse |
| 2 | 257x259 | 259 | 257 | False | four-output | resident | 0.624100 | 0.628700 | 0.987026 | 0.946460-1.027502 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | four-output | full | 5.103550 | 5.275150 | 0.984277 | 0.962202-1.006525 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.017488 | 0.018272 | 0.955281 | 0.938281-0.972473 | 0/0 | adverse |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.602350 | 0.589350 | 1.031844 | 0.837218-1.270329 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | full | 5.612650 | 5.577300 | 0.987568 | 0.949919-1.023957 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.016368 | 0.016992 | 0.954868 | 0.934975-0.975626 | 0/0 | adverse |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.658200 | 0.662500 | 1.030614 | 0.893154-1.191344 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.639800 | 5.471700 | 0.971791 | 0.902273-1.022097 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.136624 | 0.137872 | 0.972445 | 0.938341-1.006201 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.707750 | 0.741400 | 0.893113 | 0.808823-0.986103 | 0/0 | adverse |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 5.412550 | 5.397800 | 1.002283 | 0.976797-1.029057 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.017008 | 0.018448 | 0.920096 | 0.896837-0.944392 | 0/0 | adverse |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.631950 | 0.614750 | 1.021303 | 0.885769-1.175990 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | full | 5.489900 | 5.444100 | 1.006392 | 0.983421-1.030059 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.501152 | 0.501856 | 1.000816 | 0.998479-1.003450 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.246500 | 1.256750 | 0.986499 | 0.908121-1.071193 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | full | 5.609850 | 5.584650 | 1.031902 | 0.992792-1.084949 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | four-output | pass | 0.309216 | 0.329840 | 0.937028 | 0.934276-0.939668 | 0/0 | adverse |
| 2 | 256x256 | 259 | 31 | False | four-output | resident | 1.070700 | 1.101300 | 0.963474 | 0.848490-1.108897 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | four-output | full | 5.722050 | 5.680350 | 0.988534 | 0.915631-1.040133 | 0/0 | inconclusive |

## Standalone shader probes

Both unbounded explicit-unroll and bounded-width candidates use three warmup and 30 alternating timestamp pairs. They compare actual full-u32 outputs/guard words against parent and independent u64 row prefixes with dirty unused channels and poisoned physical pixels beyond logical bounds. Timing excludes host, transfers, initialization and compilation. These median-only probes have no confidence/isolation claim and are not pooled with final surveys. Explicit unrolling alone did not remove narrow-kernel regressions. The bounded candidate uses the exact adopted shader and retains original pairs below width 4,099. All 52 observations follow.

| Phase | Image | kw | Zero border | Parent median ms | Candidate median ms | Ratio of medians |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| unrolled | 1920x1080 | 259 | False | 1.999568 | 2.512800 | 0.795753 |
| unrolled | 1920x1080 | 259 | True | 1.556400 | 1.960400 | 0.793920 |
| unrolled | 1920x1080 | 511 | False | 2.925696 | 3.410032 | 0.857967 |
| unrolled | 1920x1080 | 511 | True | 2.909312 | 3.394288 | 0.857120 |
| unrolled | 1920x1080 | 4099 | False | 11.259424 | 5.697024 | 1.976369 |
| unrolled | 1920x1080 | 4099 | True | 11.238512 | 5.677200 | 1.979587 |
| unrolled | 3840x2160 | 259 | False | 6.473040 | 8.162528 | 0.793019 |
| unrolled | 3840x2160 | 259 | True | 6.437888 | 8.145872 | 0.790325 |
| unrolled | 3840x2160 | 511 | False | 12.220736 | 14.557008 | 0.839509 |
| unrolled | 3840x2160 | 511 | True | 12.256720 | 14.608448 | 0.839016 |
| unrolled | 257x259 | 259 | False | 0.057792 | 0.062224 | 0.928773 |
| unrolled | 257x259 | 259 | True | 0.057152 | 0.061632 | 0.927310 |
| unrolled | 257x259 | 4099 | False | 0.080992 | 0.060496 | 1.338799 |
| unrolled | 257x259 | 4099 | True | 0.079968 | 0.059456 | 1.344995 |
| unrolled | 17x19 | 259 | False | 0.003968 | 0.004048 | 0.980237 |
| unrolled | 17x19 | 259 | True | 0.004176 | 0.003808 | 1.096639 |
| unrolled | 1x521 | 4099 | False | 0.003424 | 0.003536 | 0.968326 |
| unrolled | 1x521 | 4099 | True | 0.002976 | 0.003680 | 0.808696 |
| unrolled | 521x1 | 4099 | False | 0.026176 | 0.026368 | 0.992718 |
| unrolled | 521x1 | 4099 | True | 0.026256 | 0.026208 | 1.001832 |
| unrolled | 2x521 | 259 | False | 0.003296 | 0.003648 | 0.903509 |
| unrolled | 2x521 | 259 | True | 0.003056 | 0.003392 | 0.900943 |
| unrolled | 255x257 | 259 | False | 0.085760 | 0.086048 | 0.996653 |
| unrolled | 255x257 | 259 | True | 0.085168 | 0.085088 | 1.000940 |
| unrolled | 256x256 | 259 | False | 0.052416 | 0.056048 | 0.935198 |
| unrolled | 256x256 | 259 | True | 0.051824 | 0.055600 | 0.932086 |
| bounded | 1920x1080 | 259 | False | 2.000128 | 1.999888 | 1.000120 |
| bounded | 1920x1080 | 259 | True | 1.552416 | 1.553680 | 0.999186 |
| bounded | 1920x1080 | 511 | False | 2.929760 | 2.930672 | 0.999689 |
| bounded | 1920x1080 | 511 | True | 2.909712 | 2.910848 | 0.999610 |
| bounded | 1920x1080 | 4099 | False | 11.260320 | 5.697568 | 1.976338 |
| bounded | 1920x1080 | 4099 | True | 11.238176 | 5.676448 | 1.979790 |
| bounded | 3840x2160 | 259 | False | 6.475616 | 6.477472 | 0.999713 |
| bounded | 3840x2160 | 259 | True | 6.430976 | 6.440304 | 0.998552 |
| bounded | 3840x2160 | 511 | False | 12.351392 | 12.339168 | 1.000991 |
| bounded | 3840x2160 | 511 | True | 12.355808 | 12.359344 | 0.999714 |
| bounded | 257x259 | 259 | False | 0.057808 | 0.057840 | 0.999447 |
| bounded | 257x259 | 259 | True | 0.057616 | 0.057424 | 1.003344 |
| bounded | 257x259 | 4099 | False | 0.080992 | 0.060704 | 1.334212 |
| bounded | 257x259 | 4099 | True | 0.080560 | 0.059552 | 1.352767 |
| bounded | 17x19 | 259 | False | 0.003952 | 0.004112 | 0.961089 |
| bounded | 17x19 | 259 | True | 0.004192 | 0.003904 | 1.073770 |
| bounded | 1x521 | 4099 | False | 0.003360 | 0.003392 | 0.990566 |
| bounded | 1x521 | 4099 | True | 0.002992 | 0.003680 | 0.813043 |
| bounded | 521x1 | 4099 | False | 0.026432 | 0.026176 | 1.009780 |
| bounded | 521x1 | 4099 | True | 0.026672 | 0.026656 | 1.000600 |
| bounded | 2x521 | 259 | False | 0.003408 | 0.003632 | 0.938326 |
| bounded | 2x521 | 259 | True | 0.003472 | 0.003120 | 1.112821 |
| bounded | 255x257 | 259 | False | 0.086176 | 0.086000 | 1.002047 |
| bounded | 255x257 | 259 | True | 0.085152 | 0.085472 | 0.996256 |
| bounded | 256x256 | 259 | False | 0.052352 | 0.052656 | 0.994227 |
| bounded | 256x256 | 259 | True | 0.052160 | 0.051904 | 1.004932 |

## Rejected separate-program warm appendix

The four-output group label below denotes the same large-case classification only; this rejected source used two outputs per invocation. Same warm method, bootstrap seed 2026101813+1000*survey+10*case_index+route_index. All 126 observations remain.

| Survey | Image | kw | kh | Zero border | Group | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16x17 | 3 | 3 | False | control | pass | 0.005408 | 0.005408 | 1.023337 | 0.984135-1.063253 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | control | resident | 0.171400 | 0.168650 | 1.009232 | 0.950389-1.069724 | 0/0 | inconclusive |
| 1 | 16x17 | 3 | 3 | False | control | full | 0.954750 | 0.963650 | 0.986232 | 0.973858-0.998411 | 0/0 | adverse |
| 1 | 257x259 | 257 | 31 | False | control | pass | 0.072480 | 0.072112 | 1.003706 | 0.999538-1.007863 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | resident | 0.267050 | 0.270700 | 1.035430 | 0.968142-1.155635 | 0/0 | inconclusive |
| 1 | 257x259 | 257 | 31 | False | control | full | 1.092350 | 1.114650 | 1.008124 | 0.965735-1.075089 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | four-output | pass | 1.649696 | 1.648816 | 0.999297 | 0.997311-1.000535 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | four-output | resident | 2.478450 | 2.472500 | 1.000773 | 0.972587-1.026291 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 1 | False | four-output | full | 8.238850 | 8.402250 | 1.001657 | 0.943641-1.059431 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | four-output | pass | 1.577472 | 1.576400 | 1.000345 | 0.999873-1.000829 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | four-output | resident | 2.154450 | 2.140700 | 1.011245 | 0.998125-1.024354 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 3 | False | four-output | full | 8.044600 | 8.065950 | 1.001524 | 0.991430-1.011929 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | four-output | pass | 1.578048 | 1.578528 | 1.000010 | 0.999231-1.000925 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | four-output | resident | 2.167450 | 2.165300 | 0.999665 | 0.982567-1.016937 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 9 | False | four-output | full | 8.067150 | 8.149100 | 0.975544 | 0.946756-0.997225 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 31 | False | four-output | pass | 1.577376 | 1.577888 | 0.999615 | 0.999087-1.000128 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | False | four-output | resident | 2.270800 | 2.275650 | 1.032530 | 0.998432-1.081142 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | False | four-output | full | 8.181650 | 8.306400 | 0.981915 | 0.961429-0.998959 | 0/0 | adverse |
| 1 | 1920x1080 | 259 | 81 | False | four-output | pass | 1.578048 | 1.577296 | 1.000927 | 1.000204-1.001709 | 0/0 | favorable |
| 1 | 1920x1080 | 259 | 81 | False | four-output | resident | 2.648050 | 2.603450 | 1.009969 | 0.998040-1.022075 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 81 | False | four-output | full | 8.562350 | 8.524750 | 1.002375 | 0.986189-1.018714 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | four-output | pass | 2.925920 | 2.924528 | 1.000343 | 0.999784-1.000869 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | four-output | resident | 3.704750 | 3.657600 | 1.005869 | 0.995993-1.015333 | 0/0 | inconclusive |
| 1 | 1920x1080 | 511 | 31 | False | four-output | full | 9.935750 | 9.925900 | 0.994423 | 0.973605-1.016742 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | pass | 11.255296 | 11.254416 | 0.999819 | 0.999366-1.000077 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | resident | 11.986800 | 11.999200 | 0.999220 | 0.996143-1.002369 | 0/0 | inconclusive |
| 1 | 1920x1080 | 4099 | 9 | False | four-output | full | 18.971600 | 18.880000 | 0.996720 | 0.988811-1.004317 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | True | four-output | pass | 1.567424 | 1.567664 | 1.000074 | 0.999446-1.000699 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | True | four-output | resident | 2.266200 | 2.287400 | 0.992907 | 0.976082-1.010736 | 0/0 | inconclusive |
| 1 | 1920x1080 | 259 | 31 | True | four-output | full | 8.201100 | 8.143500 | 1.006167 | 0.989978-1.024869 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 3 | False | four-output | pass | 6.533552 | 6.529312 | 1.000673 | 0.999758-1.001637 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 3 | False | four-output | resident | 7.397500 | 7.401600 | 0.998161 | 0.986410-1.009798 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 3 | False | four-output | full | 25.550550 | 25.658450 | 0.955699 | 0.909386-1.001733 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 31 | False | four-output | pass | 6.535920 | 6.539680 | 0.999281 | 0.998373-1.000150 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 31 | False | four-output | resident | 8.000450 | 8.017000 | 0.999888 | 0.997188-1.002729 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 31 | False | four-output | full | 25.888200 | 26.043350 | 0.996282 | 0.988149-1.004012 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | four-output | pass | 6.536704 | 6.535232 | 1.000431 | 0.999723-1.001157 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | four-output | resident | 9.573350 | 9.550850 | 1.002331 | 0.993956-1.010612 | 0/0 | inconclusive |
| 1 | 3840x2160 | 259 | 81 | False | four-output | full | 27.934000 | 28.035250 | 0.984404 | 0.947440-1.018527 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | four-output | pass | 0.211712 | 0.211200 | 1.001800 | 0.998866-1.004493 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | four-output | resident | 0.575200 | 0.578100 | 1.008302 | 0.968766-1.048567 | 0/0 | inconclusive |
| 1 | 1025x257 | 259 | 9 | True | four-output | full | 5.578200 | 5.490050 | 1.004235 | 0.978180-1.032503 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | four-output | pass | 0.211408 | 0.209168 | 1.006836 | 1.002203-1.011653 | 0/0 | favorable |
| 1 | 257x259 | 259 | 257 | False | four-output | resident | 0.531450 | 0.455900 | 1.025345 | 0.958293-1.121628 | 0/0 | inconclusive |
| 1 | 257x259 | 259 | 257 | False | four-output | full | 5.109600 | 5.091600 | 0.979923 | 0.943547-1.011784 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.017776 | 0.018608 | 0.974274 | 0.951829-0.996341 | 0/0 | adverse |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.563900 | 0.561700 | 0.977407 | 0.835738-1.121864 | 0/0 | inconclusive |
| 1 | 17x19 | 259 | 31 | False | scalar-fallback | full | 5.365450 | 5.304700 | 1.003489 | 0.984767-1.023617 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.016576 | 0.017584 | 0.963310 | 0.944515-0.984212 | 0/0 | adverse |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.636250 | 0.665700 | 0.945377 | 0.814807-1.099137 | 0/0 | inconclusive |
| 1 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.704050 | 5.453200 | 1.010540 | 0.980472-1.041651 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.137504 | 0.136400 | 1.007527 | 0.965908-1.051186 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.714450 | 0.750400 | 0.958556 | 0.837369-1.095979 | 0/0 | inconclusive |
| 1 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 5.601850 | 5.544750 | 1.019868 | 0.985403-1.057206 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.015264 | 0.015360 | 1.011204 | 0.992703-1.033407 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.639300 | 0.671850 | 0.904282 | 0.764284-1.070136 | 0/0 | inconclusive |
| 1 | 2x521 | 259 | 81 | True | scalar-fallback | full | 5.670800 | 5.529350 | 1.035188 | 1.004363-1.074413 | 0/0 | favorable |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.475696 | 0.475360 | 1.000750 | 1.000224-1.001260 | 0/0 | favorable |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.184950 | 1.218350 | 0.999436 | 0.968598-1.037661 | 0/0 | inconclusive |
| 1 | 255x257 | 259 | 31 | False | scalar-fallback | full | 5.594150 | 5.683950 | 0.973033 | 0.937302-1.002436 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | four-output | pass | 0.287984 | 0.287344 | 1.002582 | 1.001277-1.003813 | 0/0 | favorable |
| 1 | 256x256 | 259 | 31 | False | four-output | resident | 1.027700 | 1.016050 | 1.038555 | 0.991983-1.113105 | 0/0 | inconclusive |
| 1 | 256x256 | 259 | 31 | False | four-output | full | 5.553300 | 5.524200 | 0.999655 | 0.978248-1.018445 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | pass | 0.005424 | 0.005440 | 1.006185 | 0.984860-1.030040 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | resident | 0.180950 | 0.180200 | 0.986290 | 0.938005-1.035848 | 0/0 | inconclusive |
| 2 | 16x17 | 3 | 3 | False | control | full | 1.003200 | 0.996250 | 0.997709 | 0.941497-1.076113 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | pass | 0.072464 | 0.072752 | 0.998146 | 0.992934-1.003488 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | resident | 0.263650 | 0.263700 | 1.008829 | 0.975713-1.045838 | 0/0 | inconclusive |
| 2 | 257x259 | 257 | 31 | False | control | full | 1.110000 | 1.103950 | 1.002173 | 0.987437-1.017195 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | four-output | pass | 1.695168 | 1.695232 | 1.000606 | 0.999654-1.001770 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | four-output | resident | 2.451450 | 2.422850 | 1.002042 | 0.987732-1.016087 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 1 | False | four-output | full | 8.123100 | 8.219700 | 1.017565 | 0.954609-1.088718 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | four-output | pass | 1.578560 | 1.578640 | 1.000231 | 0.999559-1.000917 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | four-output | resident | 2.221800 | 2.222950 | 0.997559 | 0.977710-1.017779 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 3 | False | four-output | full | 8.098400 | 8.066950 | 1.026621 | 0.994026-1.072136 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | four-output | pass | 1.577840 | 1.578368 | 1.000032 | 0.999456-1.000630 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | four-output | resident | 2.154950 | 2.145300 | 0.991099 | 0.957082-1.015264 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 9 | False | four-output | full | 8.078600 | 8.055250 | 1.015075 | 0.996974-1.042120 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | False | four-output | pass | 1.578784 | 1.578352 | 1.000102 | 0.999641-1.000582 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | False | four-output | resident | 2.294550 | 2.319000 | 1.016376 | 0.985987-1.058744 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | False | four-output | full | 8.095500 | 8.169050 | 0.994590 | 0.983855-1.006114 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 81 | False | four-output | pass | 1.579600 | 1.578320 | 1.000161 | 0.999594-1.000728 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 81 | False | four-output | resident | 2.582250 | 2.562550 | 1.011736 | 1.002941-1.020462 | 0/0 | favorable |
| 2 | 1920x1080 | 259 | 81 | False | four-output | full | 8.507250 | 8.577550 | 1.002849 | 0.983957-1.022799 | 0/0 | inconclusive |
| 2 | 1920x1080 | 511 | 31 | False | four-output | pass | 2.949712 | 2.948912 | 1.000123 | 0.999499-1.000744 | 0/0 | inconclusive |
| 2 | 1920x1080 | 511 | 31 | False | four-output | resident | 3.680900 | 3.741600 | 0.988859 | 0.979493-0.997711 | 0/0 | adverse |
| 2 | 1920x1080 | 511 | 31 | False | four-output | full | 9.819350 | 9.842000 | 0.998152 | 0.986349-1.010077 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | pass | 11.346208 | 11.345760 | 1.000054 | 0.999992-1.000119 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | resident | 12.062000 | 12.064700 | 1.002092 | 0.999492-1.004851 | 0/0 | inconclusive |
| 2 | 1920x1080 | 4099 | 9 | False | four-output | full | 18.635950 | 18.981700 | 0.989616 | 0.975409-1.002510 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | four-output | pass | 1.572800 | 1.572800 | 1.000194 | 0.999657-1.000710 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | four-output | resident | 2.252550 | 2.268350 | 0.989443 | 0.974786-1.004102 | 0/0 | inconclusive |
| 2 | 1920x1080 | 259 | 31 | True | four-output | full | 8.207400 | 8.106900 | 1.008598 | 0.992996-1.024428 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | four-output | pass | 6.540304 | 6.541904 | 0.999629 | 0.998816-1.000447 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | four-output | resident | 7.488000 | 7.494650 | 0.996329 | 0.985360-1.007119 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 3 | False | four-output | full | 26.140000 | 27.030200 | 0.987302 | 0.940601-1.037733 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 31 | False | four-output | pass | 6.543104 | 6.538720 | 1.000536 | 0.999671-1.001361 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 31 | False | four-output | resident | 8.023800 | 8.055150 | 0.994462 | 0.992095-0.996824 | 0/0 | adverse |
| 2 | 3840x2160 | 259 | 31 | False | four-output | full | 25.827150 | 25.993300 | 0.995491 | 0.986787-1.003793 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | four-output | pass | 6.544000 | 6.541088 | 1.000250 | 0.999592-1.000881 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | four-output | resident | 9.481550 | 9.477700 | 0.999210 | 0.997211-1.001236 | 0/0 | inconclusive |
| 2 | 3840x2160 | 259 | 81 | False | four-output | full | 28.683050 | 28.847950 | 1.034428 | 0.991534-1.098054 | 0/0 | inconclusive |
| 2 | 1025x257 | 259 | 9 | True | four-output | pass | 0.211696 | 0.210960 | 1.003172 | 1.000510-1.005876 | 0/0 | favorable |
| 2 | 1025x257 | 259 | 9 | True | four-output | resident | 0.590000 | 0.576700 | 1.009770 | 0.976588-1.045972 | 0/0 | inconclusive |
| 2 | 1025x257 | 259 | 9 | True | four-output | full | 6.270000 | 6.211100 | 1.004407 | 0.971242-1.037504 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | four-output | pass | 0.230192 | 0.227600 | 1.007891 | 1.001711-1.013336 | 0/0 | favorable |
| 2 | 257x259 | 259 | 257 | False | four-output | resident | 0.624750 | 0.613400 | 1.023071 | 0.990489-1.058817 | 0/0 | inconclusive |
| 2 | 257x259 | 259 | 257 | False | four-output | full | 5.676750 | 5.737650 | 1.003129 | 0.970208-1.039063 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | pass | 0.019936 | 0.020240 | 0.987108 | 0.970278-1.003953 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | resident | 0.569600 | 0.594900 | 0.948636 | 0.827634-1.098477 | 0/0 | inconclusive |
| 2 | 17x19 | 259 | 31 | False | scalar-fallback | full | 6.063000 | 6.021700 | 1.025536 | 0.989443-1.065975 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | pass | 0.014144 | 0.014336 | 0.991072 | 0.966391-1.016583 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | resident | 0.758550 | 0.721250 | 1.008701 | 0.926719-1.093809 | 0/0 | inconclusive |
| 2 | 1x521 | 4099 | 257 | False | scalar-fallback | full | 5.722100 | 5.718200 | 1.005981 | 0.981241-1.033605 | 0/0 | inconclusive |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | pass | 0.127664 | 0.127488 | 1.002166 | 1.000360-1.004323 | 0/0 | favorable |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | resident | 0.722250 | 0.919500 | 0.851634 | 0.760664-0.951537 | 0/0 | adverse |
| 2 | 521x1 | 4099 | 257 | False | scalar-fallback | full | 5.748250 | 5.684200 | 1.019591 | 0.991937-1.053075 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | pass | 0.019216 | 0.018688 | 1.005823 | 0.997949-1.014071 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | resident | 0.822550 | 0.780250 | 0.986507 | 0.836555-1.167830 | 0/0 | inconclusive |
| 2 | 2x521 | 259 | 81 | True | scalar-fallback | full | 5.545650 | 5.577500 | 0.992476 | 0.967522-1.018532 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | pass | 0.475840 | 0.475696 | 1.000413 | 0.999707-1.001159 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | resident | 1.197050 | 1.226200 | 0.966577 | 0.925005-1.003827 | 0/0 | inconclusive |
| 2 | 255x257 | 259 | 31 | False | scalar-fallback | full | 5.579050 | 5.835850 | 0.982045 | 0.951709-1.012717 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | four-output | pass | 0.288208 | 0.287472 | 1.001501 | 0.996295-1.005069 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | four-output | resident | 0.994000 | 1.005300 | 1.029618 | 0.979469-1.117020 | 0/0 | inconclusive |
| 2 | 256x256 | 259 | 31 | False | four-output | full | 5.722450 | 5.703100 | 0.990324 | 0.962153-1.018732 | 0/0 | inconclusive |

## Rejected separate-program fresh-cache appendix

Same retained-device fresh-cache method, seed 2026101814+1000*survey+scenario_index. All 10 observations remain.

| Survey | Scenario | Parent/current pipelines | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | small | 2/2 | 1.450700 | 1.275200 | 1.242193 | 1.113954-1.505773 | 0/0 | favorable |
| 1 | large | 2/2 | 1.566150 | 1.536150 | 1.013980 | 0.998928-1.028307 | 0/0 | inconclusive |
| 1 | small_then_large | 2/3 | 1.812900 | 2.197300 | 0.817878 | 0.794775-0.838786 | 0/0 | adverse |
| 1 | large_then_small | 2/3 | 1.846400 | 2.163400 | 0.850561 | 0.829406-0.872110 | 0/0 | adverse |
| 1 | all_programs | 5/6 | 3.940950 | 4.267350 | 0.901591 | 0.878271-0.923907 | 0/0 | adverse |
| 2 | small | 2/2 | 1.617600 | 1.458850 | 1.242845 | 1.112510-1.508228 | 0/0 | favorable |
| 2 | large | 2/2 | 1.653050 | 1.633800 | 1.016367 | 0.963081-1.063460 | 0/0 | inconclusive |
| 2 | small_then_large | 2/3 | 2.102450 | 2.560100 | 0.819782 | 0.801048-0.838424 | 0/0 | adverse |
| 2 | large_then_small | 2/3 | 2.077150 | 2.456650 | 0.838820 | 0.823226-0.854057 | 0/0 | adverse |
| 2 | all_programs | 5/6 | 4.238750 | 4.629000 | 0.909784 | 0.883934-0.935937 | 0/0 | adverse |

## Source locations

- [box_blur_horizontal.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_horizontal.wgsl)
- [box_blur_wide_hquad_tests.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hquad_tests.rs)
- [box_blur_wide_hquad_reference.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hquad_reference.rs)
- [box_blur_wide_hquad_horizontal_reference.wgsl](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hquad_horizontal_reference.wgsl)
- [box_blur_wide_hquad_benchmark.rs](C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_wide_hquad_benchmark.rs)
