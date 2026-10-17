# Symmetric GPU Gaussian vertical taps

Parent: `32502457acae208e92a355677eaf9d64c807ab85` (merged PR 260).

The packed Q8 and unpacked Q8/Q15 vertical stages read the center once and
share each mirrored pair's exact weight. Q15 splits both horizontal values
into base-65536 digits before adding the pair. Packed Q8 decodes each word's
two channels before addition. Only the two production vertical tap bodies
change. The host planner/quantizer, horizontal stages, pipeline cache, resource
sizes, dispatch, ownership, metadata, errors, profiling and explicit transfers
retain their parent behavior. No API, feature, dependency, core or unsafe change.

## Eligibility and exact arithmetic

Opposite integer offsets have equal binary64 squares and exponent arguments.
The unchanged host quantizer normalizes and rounds them identically; its mass
correction affects only the center. Mirrored weights are exactly equal for
every supported finite positive sigma. This structural argument is supported
by 3,132 quantizer configurations, including subnormal/tiny/huge sigma values
and 512 deterministic finite-positive binary64 patterns per size/precision.
A 7-tap axis still selects Q15 for both axes; other calls use Q8.

For weight mass M and symmetric weight w, `2*w <= M`. Q8 horizontal components
are at most 65,280 in production; the tests admit the entire u16 range. Pairing
before multiplication is exact and every partial sum is bounded by `65535*256`.
The original `+32768` then shift by 16 half-up rounding fits u32 unchanged.
Packed words are decoded before adding, so carries cannot cross channel halves.

Q15 horizontal components are below 2^23. For `a=65536*ah+al` and
`b=65536*bh+bl`, the pair contributes `(al+bl)*w` to lo and `(ah+bh)*w` to hi.
Each pair product and every partial sum stay within `lo <= 65535*32768 < 2^31`
and `hi <= 127*32768 < 2^22`. Thus `lo + (hi << 16)` reconstructed in u64 is
the original weighted sum. The unchanged final `(hi+(lo>>16)+8192)>>14`
implements exact division by 2^30 with half-up rounding; low discarded bits
cannot alter that result. Final clamp to 255 is unchanged. No intermediate
rounding, full-word-pair overflow, or inter-channel carry is introduced.

The center row is in bounds after the original nonempty-image/index guards.
Each paired side independently uses the original border mapper. Constant,
replicate, wrap, reflect and reflect101 therefore remain exact, including
height-one/two strips that need repeated reflection. Signed-coordinate limits,
grid-stride tails and inactive output channels retain their original handling.

| Source work per channel (unpacked) or channel pair (packed) | 3 taps | 5 taps | 7 taps |
| --- | ---: | ---: | ---: |
| Original weighted terms / coefficient accesses | 3 | 5 | 7 |
| Paired weighted terms / coefficient accesses | 2 | 3 | 4 |
| Original loop iterations | 3 | 5 | 7 |
| Pair-loop iterations, center outside | 1 | 2 | 3 |
| Logical tap positions read, both versions | 3 | 5 | 7 |

Q15 has two digit products per weighted term. These are source counts, not
generated-instruction or physical memory-traffic measurements. Pairing adds
digit/vector additions; it does not reduce the number of logical values read.
Packed Q8 retains ceil(C/2) words per pixel and other layouts C words. Both
compute passes, scratch size, output size and transfer boundaries are unchanged.

## Correctness and integration

- **377 native correctness tests:** 151 image, 189 voxel/readback, 8 RANSAC,
  8 segmentation, 18 filtering, 2 transfer facade and 1 Gaussian facade.
  Ignored GPU correctness tests are included and benchmarks excluded.
  All-feature/all-target Clippy with warnings denied, CPU-only, minimal/CI
  WASM with warnings denied, formatting and whitespace checks pass.
- **50,112 checked-arithmetic fixtures** cover 3/5/7 taps, Q8/Q15 and
  522 sigma values, with sixteen maximum/alternating/digit-boundary/random
  input patterns each. They check exact symmetry/mass, reconstructed u64 sums,
  split-digit bounds and final half-up rounding using checked u32 products/adds.
- **174,291,200 physical RGBA bytes** in **1,208 paired raw fixtures**
  (2,416 vertical dispatches) match both the frozen parent and an independent
  sequential u64 tap oracle. C1..4, packed/unpacked, Q8/Q15, all five borders,
  narrow strips, odd dimensions, forced one-group grid strides, full u16/23-bit
  words, dirty inactive high halves and 1025x1025 maximum inputs are included.
  Physical output channels that are inactive must remain zero. Input suffix
  poison exposes invalid tail reads when they affect output parity; this
  read-only input is not an output guard-write test.
- **60,216,300 complete parent/CPU bytes** across 1,080 paired conditions
  (2,160 blur calls) cover six dimensions, C1..4, all nine mixed axis sizes
  and five borders, including receipts and metadata. Existing Gaussian tests
  also cover ROI/strides, extreme sigmas, validation/errors, profiling, cache
  concurrency, retained outputs, chaining and pool reuse.
- **530 original GPU/image sources unchanged** and
  **540 source/executable hashes**. The two frozen vertical shaders
  match the immediate parent byte-for-byte. The complete parent host is frozen
  with type/name/visibility, shader-reference and external-cache substitutions
  only. The three targeted tests, all measurements and final broad validation
  use the identical source set and native executable hash.

## Paired completed-call diagnostics

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=vertical_pass_resident_and_full wall_order=balanced checks_drop_recycle_excluded=true`. Adapter model names are omitted.

Two surveys each record three warmups and thirty measured parent/current pairs
across thirteen cases: 16x17 C4 3x3, 257x259 C3 5x7, 1080p C1..4 for 5x5 Q8
and 7x7 Q15, 4K C3 7x7, 1025x257 C3 constant-border 3x7, and 1080p C4 3x3.
All **78 summary rows** are retained below, including adverse and unresolved
results. Every vertical path is affected; small/mixed cases are separate loads.

- Vertical GPU pass: **18 favorable / 5 inconclusive / 3 adverse**.
- Completed resident Gaussian call: **13 favorable / 12 inconclusive / 1 adverse**.
- Upload/blur/readback: **4 favorable / 22 inconclusive / 0 adverse**.

Adverse pointwise intervals: run 1 257x259 C3 5x7 pass, 0.961557 [0.948243, 0.975031]; run 1 1025x257 C3 3x7 resident, 0.967802 [0.939597, 0.998254]; run 2 257x259 C3 5x7 pass, 0.935680 [0.922718, 0.948348]; run 2 1920x1080 C4 7x7 pass, 0.990431 [0.972914, 0.999955].

For 1080p C4 5x5, vertical medians are 0.479392 -> 0.399088 ms and 0.411984 -> 0.335200 ms; resident
medians are 1.240550 -> 1.165600 ms and 1.103150 -> 1.030600 ms; full medians are
5.924400 -> 5.850550 ms and 5.746950 -> 5.682450 ms. At 4K C3 7x7, vertical medians are
6.460912 -> 5.240880 ms and 6.434736 -> 5.230032 ms; resident medians are
12.707800 -> 11.607800 ms and 12.437650 -> 11.316200 ms; full medians are
36.004750 -> 34.818000 ms and 35.602800 -> 34.592850 ms.

The 257x259 C3 5x7 vertical pass regresses in both surveys (roughly 1.6/2.2
microseconds in medians). The 1080p C4 Q15 pass has one narrowly adverse
interval, and one constant-border completed resident row is adverse. The change
is adopted for its bounded arithmetic and repeated gains on the larger Q8 and
selected Q15 cases, including 4K C3. These results disclose a workload tradeoff;
the implementation introduces no size/device threshold inferred from this host.

Timing boundaries:

- Vertical GPU timestamps use a shared precomputed horizontal buffer. Texture,
  scratch, uniform, bind groups, pipeline creation and horizontal preparation
  are outside the timestamps. Parent/current run in one submission with order
  alternating each round. Each has fifteen parent-first/current-first measured
  pairs. Two query intervals read 32 bytes of timestamp data per pair. Output
  readback and CPU verification occur after timing.
- Resident wall time starts with input already uploaded and includes the
  complete original/new Gaussian call, its host encoding/submission/resource
  requests and both passes, then waits for GPU completion. Payload readback and
  CPU equality checks are outside the timed interval.
- Full wall time starts before upload, includes both Gaussian stages and ends
  after the real output readback. CPU equality checks, receipts and recycling
  occur after timing. Setup/reference construction are excluded. All timed
  outputs are CPU-verified afterward, including each resident call.
- Four wall variants use `variant=offset^(round%4)`. Each variant occupies all
  four positions seven/eight times after warmups; each route has exactly fifteen
  measured parent-first and fifteen current-first pairs. Both production and
  parent pipelines/pools are warmed with the three discarded rounds.

Intervals use paired log(old/new) ratios, a geometric-mean estimate and 10,000
paired bootstrap resamples. Seeds are `2026101711+1000*run+10*case+routeIndex`
(case zero-based; routes pass/resident/full indexed 0/1/2), with sorted bounds
249/9749. Ratios above one favor the change. These are pointwise percentile
95% intervals with no multiple-comparison correction. A zero duration prevents
ratio estimation for that row; raw counts remain disclosed. All medians retain
all thirty samples; no outlier filtering, automatic retrial or selective discard.
Bootstrap analysis runs only after both timed surveys finish.

Pre/post host observations:

- Run 1: observed-quiet=False; pre={"busy_processes": [{"cpu_cores": 0.337406753055503, "id": 17516, "name": "Discord"}, {"cpu_cores": 1.0060855909291362, "id": 46660, "name": "python"}], "gpu_samples": [[[1, 0]], [[0, 0]]], "observed_quiet": false}; post={"busy_processes": [{"cpu_cores": 0.9576221877500468, "id": 37228, "name": "python"}, {"cpu_cores": 0.9879266873623901, "id": 46660, "name": "python"}], "gpu_samples": [[[1, 0]], [[2, 0]]], "observed_quiet": false}.

- Run 2: observed-quiet=False; pre={"busy_processes": [{"cpu_cores": 0.9941548486244186, "id": 37228, "name": "python"}, {"cpu_cores": 0.9941548486244186, "id": 46660, "name": "python"}], "gpu_samples": [[[2, 0]], [[6, 0]]], "observed_quiet": false}; post={"busy_processes": [{"cpu_cores": 1.0001437332100591, "id": 45232, "name": "python"}, {"cpu_cores": 1.0001437332100591, "id": 46660, "name": "python"}], "gpu_samples": [[[1, 0]], [[1, 0]]], "observed_quiet": false}.

The two-second process/GPU observations cannot establish isolation during every
timed pair. This is one shared host/backend/device class with fixed order and
workloads. Timestamp granularity, thermals, background activity, upload/readback
cost and scheduling remain limitations. The results justify the bounded source
change on these workloads; they establish no portable, transfer-inclusive or
application-wide speed guarantee. All adverse/inconclusive rows remain visible.

| Run | WxH | C | Kernel | Constant | Route | Parent ms | Current ms | Paired old/new | Pointwise 95% interval | Result | Seed | Zero old/new |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| 1 | 16x17 | 4 | 3x3 | False | pass | 0.005728 | 0.005376 | 1.027595 | [0.964977, 1.093987] | inconclusive | 2026102711 | 0/0 |
| 1 | 16x17 | 4 | 3x3 | False | resident | 0.253200 | 0.253100 | 0.972655 | [0.875833, 1.053605] | inconclusive | 2026102712 | 0/0 |
| 1 | 16x17 | 4 | 3x3 | False | full | 1.266700 | 1.282950 | 0.994474 | [0.956277, 1.032417] | inconclusive | 2026102713 | 0/0 |
| 1 | 257x259 | 3 | 5x7 | False | pass | 0.036800 | 0.038416 | 0.961557 | [0.948243, 0.975031] | adverse | 2026102721 | 0/0 |
| 1 | 257x259 | 3 | 5x7 | False | resident | 0.301150 | 0.305750 | 0.972992 | [0.939316, 1.002747] | inconclusive | 2026102722 | 0/0 |
| 1 | 257x259 | 3 | 5x7 | False | full | 1.455300 | 1.481900 | 0.996736 | [0.973482, 1.019342] | inconclusive | 2026102723 | 0/0 |
| 1 | 1920x1080 | 1 | 5x5 | False | pass | 0.217584 | 0.161504 | 1.341607 | [1.335496, 1.347388] | favorable | 2026102731 | 0/0 |
| 1 | 1920x1080 | 1 | 5x5 | False | resident | 0.712800 | 0.667200 | 1.081606 | [1.050792, 1.112456] | favorable | 2026102732 | 0/0 |
| 1 | 1920x1080 | 1 | 5x5 | False | full | 7.051150 | 7.143350 | 0.978797 | [0.951602, 1.005400] | inconclusive | 2026102733 | 0/0 |
| 1 | 1920x1080 | 2 | 5x5 | False | pass | 0.222848 | 0.167760 | 1.323728 | [1.319614, 1.327905] | favorable | 2026102741 | 0/0 |
| 1 | 1920x1080 | 2 | 5x5 | False | resident | 0.766500 | 0.697600 | 1.098362 | [1.057817, 1.138619] | favorable | 2026102742 | 0/0 |
| 1 | 1920x1080 | 2 | 5x5 | False | full | 7.321550 | 7.345200 | 1.004627 | [0.984597, 1.025646] | inconclusive | 2026102743 | 0/0 |
| 1 | 1920x1080 | 3 | 5x5 | False | pass | 1.044768 | 0.924528 | 1.103964 | [1.041495, 1.139130] | favorable | 2026102751 | 0/0 |
| 1 | 1920x1080 | 3 | 5x5 | False | resident | 2.454700 | 2.379900 | 1.035324 | [0.963700, 1.106404] | inconclusive | 2026102752 | 0/0 |
| 1 | 1920x1080 | 3 | 5x5 | False | full | 10.325800 | 10.139300 | 1.020766 | [1.005776, 1.036032] | favorable | 2026102753 | 0/0 |
| 1 | 1920x1080 | 4 | 5x5 | False | pass | 0.479392 | 0.399088 | 1.209439 | [1.199739, 1.226921] | favorable | 2026102761 | 0/0 |
| 1 | 1920x1080 | 4 | 5x5 | False | resident | 1.240550 | 1.165600 | 1.073876 | [1.020822, 1.139139] | favorable | 2026102762 | 0/0 |
| 1 | 1920x1080 | 4 | 5x5 | False | full | 5.924400 | 5.850550 | 1.013592 | [0.996911, 1.029223] | inconclusive | 2026102763 | 0/0 |
| 1 | 1920x1080 | 1 | 7x7 | False | pass | 0.412368 | 0.302048 | 1.364252 | [1.360879, 1.367605] | favorable | 2026102771 | 0/0 |
| 1 | 1920x1080 | 1 | 7x7 | False | resident | 1.009350 | 0.930500 | 1.080875 | [1.053681, 1.105563] | favorable | 2026102772 | 0/0 |
| 1 | 1920x1080 | 1 | 7x7 | False | full | 7.486200 | 7.328850 | 1.017037 | [0.996821, 1.036790] | inconclusive | 2026102773 | 0/0 |
| 1 | 1920x1080 | 2 | 7x7 | False | pass | 0.648208 | 0.597904 | 1.083361 | [1.082041, 1.084725] | favorable | 2026102781 | 0/0 |
| 1 | 1920x1080 | 2 | 7x7 | False | resident | 2.708250 | 2.425200 | 1.056754 | [1.001344, 1.103852] | favorable | 2026102782 | 0/0 |
| 1 | 1920x1080 | 2 | 7x7 | False | full | 10.220900 | 9.986350 | 1.020347 | [0.999938, 1.044608] | inconclusive | 2026102783 | 0/0 |
| 1 | 1920x1080 | 3 | 7x7 | False | pass | 1.061040 | 1.036208 | 1.024148 | [1.022659, 1.025434] | favorable | 2026102791 | 0/0 |
| 1 | 1920x1080 | 3 | 7x7 | False | resident | 2.214000 | 2.193850 | 0.978577 | [0.890115, 1.031720] | inconclusive | 2026102792 | 0/0 |
| 1 | 1920x1080 | 3 | 7x7 | False | full | 8.950200 | 8.860450 | 1.000460 | [0.982653, 1.016927] | inconclusive | 2026102793 | 0/0 |
| 1 | 1920x1080 | 4 | 7x7 | False | pass | 1.499296 | 1.498320 | 0.999930 | [0.948970, 1.055183] | inconclusive | 2026102801 | 0/0 |
| 1 | 1920x1080 | 4 | 7x7 | False | resident | 2.722300 | 2.699400 | 1.021146 | [0.968497, 1.074998] | inconclusive | 2026102802 | 0/0 |
| 1 | 1920x1080 | 4 | 7x7 | False | full | 7.593000 | 7.616500 | 1.040109 | [0.984082, 1.138908] | inconclusive | 2026102803 | 0/0 |
| 1 | 3840x2160 | 3 | 7x7 | False | pass | 6.460912 | 5.240880 | 1.239692 | [1.219641, 1.259441] | favorable | 2026102811 | 0/0 |
| 1 | 3840x2160 | 3 | 7x7 | False | resident | 12.707800 | 11.607800 | 1.097307 | [1.024543, 1.177308] | favorable | 2026102812 | 0/0 |
| 1 | 3840x2160 | 3 | 7x7 | False | full | 36.004750 | 34.818000 | 1.033649 | [1.016126, 1.051521] | favorable | 2026102813 | 0/0 |
| 1 | 1025x257 | 3 | 3x7 | True | pass | 0.110112 | 0.110240 | 0.998607 | [0.992933, 1.004381] | inconclusive | 2026102821 | 0/0 |
| 1 | 1025x257 | 3 | 3x7 | True | resident | 0.578100 | 0.610950 | 0.967802 | [0.939597, 0.998254] | adverse | 2026102822 | 0/0 |
| 1 | 1025x257 | 3 | 3x7 | True | full | 7.565300 | 7.633550 | 0.993166 | [0.958852, 1.025057] | inconclusive | 2026102823 | 0/0 |
| 1 | 1920x1080 | 4 | 3x3 | False | pass | 0.477904 | 0.372480 | 1.282899 | [1.280632, 1.285212] | favorable | 2026102831 | 0/0 |
| 1 | 1920x1080 | 4 | 3x3 | False | resident | 1.230950 | 1.095250 | 1.112866 | [1.078494, 1.146495] | favorable | 2026102832 | 0/0 |
| 1 | 1920x1080 | 4 | 3x3 | False | full | 9.923850 | 9.621400 | 1.040943 | [0.992332, 1.104250] | inconclusive | 2026102833 | 0/0 |
| 2 | 16x17 | 4 | 3x3 | False | pass | 0.005616 | 0.005440 | 1.019975 | [0.957766, 1.086343] | inconclusive | 2026103711 | 0/0 |
| 2 | 16x17 | 4 | 3x3 | False | resident | 0.280900 | 0.280500 | 0.987445 | [0.899304, 1.073477] | inconclusive | 2026103712 | 0/0 |
| 2 | 16x17 | 4 | 3x3 | False | full | 1.395300 | 1.362500 | 1.028851 | [1.001419, 1.058534] | favorable | 2026103713 | 0/0 |
| 2 | 257x259 | 3 | 5x7 | False | pass | 0.036688 | 0.038912 | 0.935680 | [0.922718, 0.948348] | adverse | 2026103721 | 0/0 |
| 2 | 257x259 | 3 | 5x7 | False | resident | 0.309100 | 0.309250 | 1.016159 | [0.985876, 1.048914] | inconclusive | 2026103722 | 0/0 |
| 2 | 257x259 | 3 | 5x7 | False | full | 1.518050 | 1.519300 | 1.000968 | [0.969520, 1.032246] | inconclusive | 2026103723 | 0/0 |
| 2 | 1920x1080 | 1 | 5x5 | False | pass | 0.217072 | 0.161616 | 1.345769 | [1.340705, 1.350698] | favorable | 2026103731 | 0/0 |
| 2 | 1920x1080 | 1 | 5x5 | False | resident | 0.724050 | 0.689600 | 1.066246 | [1.040569, 1.093794] | favorable | 2026103732 | 0/0 |
| 2 | 1920x1080 | 1 | 5x5 | False | full | 7.418650 | 7.512450 | 1.006730 | [0.978990, 1.041641] | inconclusive | 2026103733 | 0/0 |
| 2 | 1920x1080 | 2 | 5x5 | False | pass | 0.706128 | 0.597200 | 1.256574 | [1.185167, 1.373376] | favorable | 2026103741 | 0/0 |
| 2 | 1920x1080 | 2 | 5x5 | False | resident | 0.791750 | 0.743700 | 1.067178 | [0.989892, 1.143205] | inconclusive | 2026103742 | 0/0 |
| 2 | 1920x1080 | 2 | 5x5 | False | full | 7.690500 | 7.627400 | 0.996472 | [0.978499, 1.015409] | inconclusive | 2026103743 | 0/0 |
| 2 | 1920x1080 | 3 | 5x5 | False | pass | 1.038720 | 0.931472 | 1.122049 | [1.115786, 1.128153] | favorable | 2026103751 | 0/0 |
| 2 | 1920x1080 | 3 | 5x5 | False | resident | 2.399500 | 2.298650 | 1.077679 | [1.045649, 1.129074] | favorable | 2026103752 | 0/0 |
| 2 | 1920x1080 | 3 | 5x5 | False | full | 10.010650 | 9.912350 | 0.999405 | [0.983859, 1.014649] | inconclusive | 2026103753 | 0/0 |
| 2 | 1920x1080 | 4 | 5x5 | False | pass | 0.411984 | 0.335200 | 1.230076 | [1.227781, 1.232344] | favorable | 2026103761 | 0/0 |
| 2 | 1920x1080 | 4 | 5x5 | False | resident | 1.103150 | 1.030600 | 1.055144 | [1.009036, 1.090613] | favorable | 2026103762 | 0/0 |
| 2 | 1920x1080 | 4 | 5x5 | False | full | 5.746950 | 5.682450 | 1.008874 | [0.985342, 1.031511] | inconclusive | 2026103763 | 0/0 |
| 2 | 1920x1080 | 1 | 7x7 | False | pass | 0.757344 | 0.518128 | 1.459258 | [1.454247, 1.464307] | favorable | 2026103771 | 0/0 |
| 2 | 1920x1080 | 1 | 7x7 | False | resident | 0.960200 | 0.911250 | 1.075191 | [0.999764, 1.127239] | inconclusive | 2026103772 | 0/0 |
| 2 | 1920x1080 | 1 | 7x7 | False | full | 7.735000 | 7.902200 | 1.003075 | [0.981687, 1.024344] | inconclusive | 2026103773 | 0/0 |
| 2 | 1920x1080 | 2 | 7x7 | False | pass | 0.655408 | 0.605152 | 1.081485 | [1.079705, 1.083210] | favorable | 2026103781 | 0/0 |
| 2 | 1920x1080 | 2 | 7x7 | False | resident | 1.387700 | 1.365200 | 1.008929 | [0.952140, 1.048055] | inconclusive | 2026103782 | 0/0 |
| 2 | 1920x1080 | 2 | 7x7 | False | full | 7.861500 | 7.738100 | 1.035115 | [0.993768, 1.101389] | inconclusive | 2026103783 | 0/0 |
| 2 | 1920x1080 | 3 | 7x7 | False | pass | 1.471616 | 1.437008 | 1.023848 | [1.022528, 1.025094] | favorable | 2026103791 | 0/0 |
| 2 | 1920x1080 | 3 | 7x7 | False | resident | 2.369750 | 2.337750 | 1.015620 | [1.005889, 1.025204] | favorable | 2026103792 | 0/0 |
| 2 | 1920x1080 | 3 | 7x7 | False | full | 9.014700 | 9.134800 | 1.003187 | [0.988931, 1.018163] | inconclusive | 2026103793 | 0/0 |
| 2 | 1920x1080 | 4 | 7x7 | False | pass | 1.871792 | 1.873568 | 0.990431 | [0.972914, 0.999955] | adverse | 2026103801 | 0/0 |
| 2 | 1920x1080 | 4 | 7x7 | False | resident | 2.933200 | 2.938250 | 0.987847 | [0.968688, 1.003523] | inconclusive | 2026103802 | 0/0 |
| 2 | 1920x1080 | 4 | 7x7 | False | full | 7.619700 | 7.702850 | 0.993304 | [0.977453, 1.008821] | inconclusive | 2026103803 | 0/0 |
| 2 | 3840x2160 | 3 | 7x7 | False | pass | 6.434736 | 5.230032 | 1.231682 | [1.205771, 1.258790] | favorable | 2026103811 | 0/0 |
| 2 | 3840x2160 | 3 | 7x7 | False | resident | 12.437650 | 11.316200 | 1.082669 | [1.000887, 1.166666] | favorable | 2026103812 | 0/0 |
| 2 | 3840x2160 | 3 | 7x7 | False | full | 35.602800 | 34.592850 | 1.037165 | [1.017437, 1.061973] | favorable | 2026103813 | 0/0 |
| 2 | 1025x257 | 3 | 3x7 | True | pass | 0.110816 | 0.110352 | 1.002675 | [0.997608, 1.007824] | inconclusive | 2026103821 | 0/0 |
| 2 | 1025x257 | 3 | 3x7 | True | resident | 0.592550 | 0.578250 | 1.019029 | [0.977910, 1.061438] | inconclusive | 2026103822 | 0/0 |
| 2 | 1025x257 | 3 | 3x7 | True | full | 7.546200 | 7.383900 | 1.023890 | [0.994300, 1.053399] | inconclusive | 2026103823 | 0/0 |
| 2 | 1920x1080 | 4 | 3x3 | False | pass | 0.275648 | 0.208432 | 1.319964 | [1.316844, 1.322681] | favorable | 2026103831 | 0/0 |
| 2 | 1920x1080 | 4 | 3x3 | False | resident | 0.851200 | 0.815800 | 1.051358 | [1.020153, 1.081890] | favorable | 2026103832 | 0/0 |
| 2 | 1920x1080 | 4 | 3x3 | False | full | 9.546800 | 9.586150 | 1.001576 | [0.979276, 1.026052] | inconclusive | 2026103833 | 0/0 |

## Reproduction and local evidence

Native test sources, frozen parents and benchmark are under
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/`.
Run the `gaussian::tests::vmirror::` correctness filter with all features in
release mode, including ignored tests and excluding `benchmark`. Run
`benchmark_complete_gaussian_vmirror_calls` alone with `--ignored --nocapture
--test-threads=1` for the timestamp/resident/full matrix.

Local ignored receipts and complete thirty-pair arrays are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`: the
`gaussian-vmirror-` precheck, validation, provenance, measurements and survey
JSON files and corresponding logs. They contain no adapter model names in the
published benchmark text. This note retains every aggregate row; local receipts
are machine-specific evidence and are not required by the production library.
