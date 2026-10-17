# Symmetric GPU Gaussian horizontal taps

Parent: `b77751514a95685316f5748f6390dbfc7e5ca901` (merged PR 259).

Packed Q8 and unpacked Q8/Q15 horizontal filters now read the center once and
combine mirrored pixels before applying their shared weight. Only the two
horizontal tap bodies change. The full host planner/weight quantizer/API,
vertical shaders, pipeline cache, dispatch, scratch, pass count, ownership,
metadata, validation/error precedence, profiling and explicit transfers remain.
No public API, dependency, core or unsafe boundary changes.

## Exact eligibility, arithmetic and borders

The original host computes each tap from the square of an exactly represented
integer offset. Opposite offsets have identical binary64 squares, exponent
arguments and raw values. Every tap uses the same normalization denominator
and rounding; center correction affects only the unpaired center. Thus opposite
quantized weights are exactly equal, including extreme finite positive sigmas.
The quantizer and its original order are unchanged. A 7-tap axis still selects
Q15 for both axes; other calls retain Q8.

For mirrored pixels a/b and equal weight w, `(a+b)*w` equals `a*w+b*w` exactly.
Pixel components are in 0..=255, so the pair is at most 510. Nonnegative total
weight mass M is 256 or 32768, and `2*w <= M`. Each pair product and every
partial sum are bounded by `255*M`: 65,280 for Q8, 8,355,840 for Q15, both below
2^23 and therefore u32 capacity. Reordering introduces no overflow or rounding.
There is still no intermediate rounding, and the unchanged vertical shaders
apply exactly the original final half-up rounding.

The center coordinate is always in bounds because the original grid loop
checks pixel count and uses validated nonzero width/height. Each mirrored side
independently calls the same original mapped-coordinate helper; constant
substitution, replicate, wrap and both reflections remain, even on width-one
or width-two strips that need repeated reflection. Signed coordinate bounds
are unchanged. Every logical output and packed inactive odd-channel high half
matches the parent; output indexing and grid-stride tail logic are unchanged.

| Source work per pixel | 3 taps | 5 taps | 7 taps |
| --- | ---: | ---: | ---: |
| Original weighted vec4 terms / coefficient accesses | 3 | 5 | 7 |
| Symmetric weighted vec4 terms / coefficient accesses | 2 | 3 | 4 |
| Original loop iterations | 3 | 5 | 7 |
| Symmetric pair-loop iterations, center outside | 1 | 2 | 3 |
| Logical tap values including center, both versions | 3 | 5 | 7 |

These source counts do not measure generated instructions or texture traffic.
The source adds a pixel-pair addition and removes repeated multiplies/loop work;
it does not reduce the number of sampled pixels. The packed layout still uses
ceil(C/2) u32 words per pixel, with inactive odd high halves zero. Other layouts
still use C u32 words. No pass, buffer request, upload or readback is added.

## Correctness and integration

- **374 native correctness tests:** 148 image, 189 voxel/readback, 8 RANSAC,
  8 segmentation, 18 filtering, 2 image-transfer facade and 1 Gaussian facade.
  Ignored GPU correctness tests are included and benchmarks excluded.
  Clippy all-features/all-targets with warnings denied, CPU-only, minimal/CI
  WASM with warnings denied, formatting and whitespace pass.
- **3,132 quantizer cases** check symmetry, nonnegative bounded
  weights, exact mass and zero padding for 3/5/7 taps with Q8/Q15. Include ten
  deliberate tiny/subnormal/overflowing-denominator/ordinary/huge sigmas and
  512 deterministic finite-positive binary64 bit patterns per size/precision.
  These tests support the structural equality argument above, not an exhaustive
  enumeration of all binary64 inputs.
- **86,465,776 exact parent/u64 horizontal/guard words** in 1,456 paired
  raw fixtures (2,912 kernel executions). Cover C1..4, both scratch layouts,
  3/5/7 horizontal widths, Q8/Q15 mixed vertical selection, five borders,
  sigmas 0.17/7, one/two-pixel strips, odd dimensions, 257x259, forced one-group
  grid strides and suffix guards. Four maximum-input fixtures per channel at
  1025x257 cover Q8/Q15 bounds and constant-255 borders. The oracle sums taps
  independently in original order using u64, then packs words independently.
  Every logical word, unused odd high half and eight-word suffix is checked.
- **60,216,300 complete parent/CPU bytes** in 1,080 paired conditions
  (2,160 complete blur calls). Six dimensions, all nine mixed axis sizes,
  C1..4 and five borders check final bytes, metadata and original transfer/
  compute receipts. Existing Gaussian regressions also cover extreme sigmas,
  nontrivial weighted filters, ROI/strided views, invalid limits/options,
  profiling, cache concurrency, retained outputs, chaining and pool reuse.
- **525 original GPU/image sources unchanged**, with **535 source/executable
  hashes**. Both parent horizontal shaders are frozen; the full parent
  planner/quantizer/pipelines/compute call is frozen with type/name/visibility,
  shader-reference and external-cache substitutions only. The production Rust
  file is unchanged; its existing native test module only appends a submodule.
  The executable and all GPU/image sources measured after the three targeted
  tests match the final broad-validation manifest exactly.

## Paired completed-call diagnostics

Final environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=horizontal_pass_resident_and_full wall_order=balanced checks_drop_recycle_excluded=true`. Adapter model names are omitted.

Two initial rotation surveys and two final balanced surveys each use three
warmups and thirty measured pairs across thirteen cases. All **156** rows are
retained with phase labels. Cases include 16x17 C4 3x3;
257x259 C3 5x7; 1080p C1/C2/C3/C4 for both 5x5 Q8 and 7x7 Q15; 4K C3 7x7;
1025x257 C3 constant-border 3x7; and 1080p C4 3x3. Every horizontal path is
affected; tiny/mixed-axis cases are separate workloads, not unchanged controls.

- Final balanced horizontal GPU pass: **22 favorable / 4 inconclusive / 0 adverse**.
- Final balanced completed resident Gaussian call: **13 favorable / 13 inconclusive / 0 adverse**.
- Final balanced upload/blur/readback: **0 favorable / 24 inconclusive / 2 adverse**.

Initial rotation phase: horizontal **20 favorable / 6 inconclusive / 0 adverse**;
resident **10 favorable / 16 inconclusive / 0 adverse**; full
**1 favorable / 24 inconclusive / 1 adverse**. These observations remain separately
reported; they are not pooled into the final balanced estimates.

Adverse pointwise intervals are retained: run 1, 1920x1080 C4 7x7, full, ratio 0.938428 [0.871479, 0.997530]; run 3, 1920x1080 C1 5x5, full, ratio 0.978515 [0.963754, 0.992340]; run 3, 1920x1080 C4 7x7, full, ratio 0.966366 [0.916793, 0.997740].

For 1080p C4 5x5, horizontal medians are 0.291696 -> 0.258032 ms and 0.285360 -> 0.252272 ms; resident medians are
1.244200 -> 1.212500 ms and 1.264850 -> 1.219800 ms; full medians are 6.243600 -> 6.288400 ms and 6.565300 -> 6.504900 ms. For 1080p C4 7x7,
horizontal medians are 0.535216 -> 0.472832 ms and 0.477872 -> 0.424416 ms. At 4K C3 7x7, horizontal medians are
1.049184 -> 0.898464 ms and 1.047376 -> 0.897584 ms; resident medians are
12.770450 -> 13.768900 ms and 13.321900 -> 12.845400 ms; full medians are
34.741000 -> 34.660950 ms and 34.187300 -> 34.210000 ms. Each boundary must be interpreted
independently; these shared-host diagnostics do not establish a portable
application or transfer-inclusive guarantee.

- `pass`: one horizontal compute-pass duration. Source texture, control buffer,
  separate guarded parent/current sums and bind groups are prepared outside
  timing. One command buffer alternates parent/current order by round. Explicit
  timestamp collection returns 32 bytes per pair after execution. Every final
  horizontal word matches the frozen parent, including guarded suffixes. This
  boundary excludes vertical work, gaps, host planning, allocation, uploads,
  payload/query readback and CPU waits.
- `resident`: the complete Gaussian callable starting from a resident image;
  includes original host plan/weights/cache/scratch/output/control/group/encoder
  construction, both GPU stages, submission and explicit completion wait. No
  profiler is attached; final output readback/assertions/recycle follow timing.
  The parent is the complete frozen callable with a prebuilt external cache;
  production retains its original lazy runtime cache, warm before measured work.
- `full`: image upload, complete Gaussian call and original owned image readback;
  includes transfer allocation/packing, map/wait and CPU copy. Input, source and
  output recycling/destruction and assertions follow timing. Both routes retain
  original padded RGBA transfer plans and receipts. This is a Gaussian chain,
  not an entire vision application.

Initial resident/full variants rotate by round, which gives each variant all
four positions but leaves parent first in 23/30 resident and 22/30 full pairs.
Final variants use `offset XOR (round mod 4)`, giving orders 0/1/2/3, 1/0/3/2,
2/3/0/1 and 3/2/1/0. The parent runs first in 15 pairs and the new version first
in 15 pairs; each variant visits every position seven or eight times. Horizontal
GPU pairs already alternate first direction 15/15 in every survey.

Both initial complete surveys and all 535 validated source/executable files are
archived against their preserved hash manifest. Only the native benchmark's
wall-order permutation/environment label and executable differ between phases;
production shaders, parent host reference and correctness fixtures match.
Both phases pass 374 native correctness tests. Do not pool wall estimates
across these distinct order protocols.

CPU expectations, pipeline compilation and setup are outside measured rounds. GPU
pass profiling is separate from the unprofiled completed-call timers. Defaults
for bounded scratch/texture retention remain; allocations required by those
budgets are part of complete call timing for both variants.

Ratios are geometric means of paired parent/new samples; above one favors the
new route. Pointwise percentile intervals resample paired log ratios 10,000
times, seed `2026101710 + 1000*run + 10*case + route_index`, route order
pass/resident/full, sorted bounds 249/9749. No multiple-comparison correction
is applied. Zero samples would suppress ratios; every raw sample remains.

Run 1 (rotation-prototype): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=horizontal_pass_resident_and_full checks_drop_recycle_excluded=true`; observed-quiet **false**; busy-process counts 2 before / 2 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[1, 0]], [[0, 0]]]` after.

Run 2 (rotation-prototype): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=horizontal_pass_resident_and_full checks_drop_recycle_excluded=true`; observed-quiet **false**; busy-process counts 1 before / 0 after. GPU/memory utilization samples `[[[0, 0]], [[3, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 3 (balanced-final): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=horizontal_pass_resident_and_full wall_order=balanced checks_drop_recycle_excluded=true`; observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization samples `[[[5, 0]], [[5, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 4 (balanced-final): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=horizontal_pass_resident_and_full wall_order=balanced checks_drop_recycle_excluded=true`; observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization samples `[[[0, 0]], [[5, 0]]]` before / `[[[2, 0]], [[5, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all sampled
GPU/memory utilization at most 5%. Snapshots are two seconds apart; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post observations
cannot prove isolation throughout each pair. These are shared-host diagnostics
with no portable throughput or whole-application latency guarantee. Process
identities remain in ignored receipts.

| Phase / run | Size / channels / horizontal×vertical / constant | Boundary | Parent median ms | New median ms | Paired parent/new ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| rotation-prototype / 1 | 16×17 / 4 / 3×3 / false | pass | 0.005856 | 0.005952 | 0.985344 [0.911918, 1.064854] |
| rotation-prototype / 1 | 16×17 / 4 / 3×3 / false | resident | 0.273200 | 0.285900 | 0.946671 [0.869544, 1.009170] |
| rotation-prototype / 1 | 16×17 / 4 / 3×3 / false | full | 1.368200 | 1.387950 | 0.991057 [0.951177, 1.034986] |
| rotation-prototype / 1 | 257×259 / 3 / 5×7 / false | pass | 0.016144 | 0.014704 | 1.104401 [1.065697, 1.144145] |
| rotation-prototype / 1 | 257×259 / 3 / 5×7 / false | resident | 0.368850 | 0.347450 | 1.072261 [1.018712, 1.128737] |
| rotation-prototype / 1 | 257×259 / 3 / 5×7 / false | full | 1.637200 | 1.605050 | 1.005277 [0.968057, 1.045982] |
| rotation-prototype / 1 | 1920×1080 / 1 / 5×5 / false | pass | 0.191648 | 0.159440 | 1.201446 [1.197824, 1.204961] |
| rotation-prototype / 1 | 1920×1080 / 1 / 5×5 / false | resident | 0.813250 | 0.784150 | 1.036237 [1.001293, 1.069831] |
| rotation-prototype / 1 | 1920×1080 / 1 / 5×5 / false | full | 8.265700 | 8.250300 | 1.004001 [0.982614, 1.026280] |
| rotation-prototype / 1 | 1920×1080 / 2 / 5×5 / false | pass | 0.193600 | 0.163648 | 1.181711 [1.177413, 1.185849] |
| rotation-prototype / 1 | 1920×1080 / 2 / 5×5 / false | resident | 0.918750 | 0.929400 | 1.002236 [0.969993, 1.036319] |
| rotation-prototype / 1 | 1920×1080 / 2 / 5×5 / false | full | 9.084300 | 9.241200 | 1.003903 [0.975258, 1.033292] |
| rotation-prototype / 1 | 1920×1080 / 3 / 5×5 / false | pass | 0.218528 | 0.189920 | 1.149913 [1.144974, 1.154520] |
| rotation-prototype / 1 | 1920×1080 / 3 / 5×5 / false | resident | 1.174050 | 1.152200 | 0.999758 [0.979248, 1.021714] |
| rotation-prototype / 1 | 1920×1080 / 3 / 5×5 / false | full | 9.976550 | 9.776650 | 1.016534 [0.981032, 1.056983] |
| rotation-prototype / 1 | 1920×1080 / 4 / 5×5 / false | pass | 0.854976 | 0.852880 | 1.003772 [0.999874, 1.007903] |
| rotation-prototype / 1 | 1920×1080 / 4 / 5×5 / false | resident | 1.267950 | 1.167150 | 1.021397 [0.988034, 1.055780] |
| rotation-prototype / 1 | 1920×1080 / 4 / 5×5 / false | full | 8.658650 | 8.847200 | 0.989124 [0.961384, 1.015957] |
| rotation-prototype / 1 | 1920×1080 / 1 / 7×7 / false | pass | 0.509280 | 0.463344 | 1.101183 [1.096770, 1.105437] |
| rotation-prototype / 1 | 1920×1080 / 1 / 7×7 / false | resident | 2.246000 | 2.165400 | 1.034130 [1.017767, 1.050975] |
| rotation-prototype / 1 | 1920×1080 / 1 / 7×7 / false | full | 11.270050 | 11.108150 | 1.010387 [0.995259, 1.025143] |
| rotation-prototype / 1 | 1920×1080 / 2 / 7×7 / false | pass | 0.859888 | 0.854560 | 1.006169 [1.002778, 1.009546] |
| rotation-prototype / 1 | 1920×1080 / 2 / 7×7 / false | resident | 3.244650 | 3.238500 | 0.998917 [0.987784, 1.010205] |
| rotation-prototype / 1 | 1920×1080 / 2 / 7×7 / false | full | 12.717050 | 12.502200 | 1.012878 [0.996003, 1.030398] |
| rotation-prototype / 1 | 1920×1080 / 3 / 7×7 / false | pass | 0.998544 | 0.995280 | 1.002207 [0.999115, 1.005379] |
| rotation-prototype / 1 | 1920×1080 / 3 / 7×7 / false | resident | 4.211950 | 4.224900 | 0.997976 [0.991715, 1.004293] |
| rotation-prototype / 1 | 1920×1080 / 3 / 7×7 / false | full | 13.749300 | 13.673450 | 1.002901 [0.983761, 1.021795] |
| rotation-prototype / 1 | 1920×1080 / 4 / 7×7 / false | pass | 0.575808 | 0.509856 | 1.115484 [1.096219, 1.130702] |
| rotation-prototype / 1 | 1920×1080 / 4 / 7×7 / false | resident | 3.745650 | 3.686200 | 1.017603 [1.001182, 1.033785] |
| rotation-prototype / 1 | 1920×1080 / 4 / 7×7 / false | full | 10.507700 | 10.872450 | 0.938428 [0.871479, 0.997530] |
| rotation-prototype / 1 | 3840×2160 / 3 / 7×7 / false | pass | 1.065552 | 0.911904 | 1.178250 [1.167506, 1.197645] |
| rotation-prototype / 1 | 3840×2160 / 3 / 7×7 / false | resident | 15.146550 | 13.386750 | 1.118572 [1.083290, 1.151623] |
| rotation-prototype / 1 | 3840×2160 / 3 / 7×7 / false | full | 37.868850 | 38.412400 | 0.994418 [0.979951, 1.009079] |
| rotation-prototype / 1 | 1025×257 / 3 / 3×7 / true | pass | 0.065440 | 0.058608 | 1.097318 [1.074257, 1.120999] |
| rotation-prototype / 1 | 1025×257 / 3 / 3×7 / true | resident | 0.586100 | 0.565550 | 1.046971 [0.986804, 1.118736] |
| rotation-prototype / 1 | 1025×257 / 3 / 3×7 / true | full | 8.096950 | 7.977550 | 1.000330 [0.959997, 1.040939] |
| rotation-prototype / 1 | 1920×1080 / 4 / 3×3 / false | pass | 0.860432 | 0.854816 | 1.018590 [1.007417, 1.031885] |
| rotation-prototype / 1 | 1920×1080 / 4 / 3×3 / false | resident | 1.474350 | 1.438750 | 1.008060 [0.971398, 1.042174] |
| rotation-prototype / 1 | 1920×1080 / 4 / 3×3 / false | full | 11.017100 | 11.096550 | 0.990720 [0.954019, 1.028786] |
| rotation-prototype / 2 | 16×17 / 4 / 3×3 / false | pass | 0.005792 | 0.006032 | 0.980290 [0.908571, 1.060257] |
| rotation-prototype / 2 | 16×17 / 4 / 3×3 / false | resident | 0.265350 | 0.263500 | 0.982250 [0.908761, 1.037853] |
| rotation-prototype / 2 | 16×17 / 4 / 3×3 / false | full | 1.249300 | 1.296350 | 0.982599 [0.964756, 1.000355] |
| rotation-prototype / 2 | 257×259 / 3 / 5×7 / false | pass | 0.016384 | 0.014784 | 1.100674 [1.060101, 1.142843] |
| rotation-prototype / 2 | 257×259 / 3 / 5×7 / false | resident | 0.289600 | 0.283450 | 1.019752 [0.976838, 1.063437] |
| rotation-prototype / 2 | 257×259 / 3 / 5×7 / false | full | 1.367200 | 1.327750 | 1.014281 [0.989804, 1.039504] |
| rotation-prototype / 2 | 1920×1080 / 1 / 5×5 / false | pass | 0.193632 | 0.161536 | 1.200033 [1.196434, 1.203336] |
| rotation-prototype / 2 | 1920×1080 / 1 / 5×5 / false | resident | 0.835850 | 0.811050 | 1.027570 [1.002707, 1.053890] |
| rotation-prototype / 2 | 1920×1080 / 1 / 5×5 / false | full | 7.849750 | 7.663800 | 1.010152 [0.986826, 1.034439] |
| rotation-prototype / 2 | 1920×1080 / 2 / 5×5 / false | pass | 0.587440 | 0.581104 | 1.008315 [0.999984, 1.016558] |
| rotation-prototype / 2 | 1920×1080 / 2 / 5×5 / false | resident | 0.840050 | 0.813400 | 1.006533 [0.983396, 1.031323] |
| rotation-prototype / 2 | 1920×1080 / 2 / 5×5 / false | full | 7.936250 | 8.116750 | 0.990540 [0.962016, 1.016371] |
| rotation-prototype / 2 | 1920×1080 / 3 / 5×5 / false | pass | 0.724432 | 0.721088 | 1.003734 [0.995917, 1.011800] |
| rotation-prototype / 2 | 1920×1080 / 3 / 5×5 / false | resident | 2.797900 | 2.775200 | 1.006568 [0.993764, 1.021177] |
| rotation-prototype / 2 | 1920×1080 / 3 / 5×5 / false | full | 10.824700 | 10.845600 | 1.010656 [0.993988, 1.028909] |
| rotation-prototype / 2 | 1920×1080 / 4 / 5×5 / false | pass | 0.854384 | 0.850464 | 1.004670 [1.000131, 1.009380] |
| rotation-prototype / 2 | 1920×1080 / 4 / 5×5 / false | resident | 2.856850 | 2.849050 | 1.013080 [0.993581, 1.041739] |
| rotation-prototype / 2 | 1920×1080 / 4 / 5×5 / false | full | 8.939100 | 8.802750 | 1.029676 [1.004440, 1.069851] |
| rotation-prototype / 2 | 1920×1080 / 1 / 7×7 / false | pass | 0.308752 | 0.255168 | 1.210506 [1.208957, 1.212043] |
| rotation-prototype / 2 | 1920×1080 / 1 / 7×7 / false | resident | 1.105200 | 1.067150 | 1.045628 [1.018638, 1.076696] |
| rotation-prototype / 2 | 1920×1080 / 1 / 7×7 / false | full | 8.945900 | 8.764250 | 1.011711 [0.982218, 1.043931] |
| rotation-prototype / 2 | 1920×1080 / 2 / 7×7 / false | pass | 0.860112 | 0.853200 | 1.006334 [1.003482, 1.009286] |
| rotation-prototype / 2 | 1920×1080 / 2 / 7×7 / false | resident | 2.765500 | 2.780600 | 1.020781 [0.998879, 1.044230] |
| rotation-prototype / 2 | 1920×1080 / 2 / 7×7 / false | full | 10.567850 | 10.706400 | 0.993985 [0.966589, 1.019410] |
| rotation-prototype / 2 | 1920×1080 / 3 / 7×7 / false | pass | 0.433296 | 0.380112 | 1.140335 [1.136475, 1.144197] |
| rotation-prototype / 2 | 1920×1080 / 3 / 7×7 / false | resident | 3.875650 | 3.861900 | 0.997937 [0.965469, 1.018856] |
| rotation-prototype / 2 | 1920×1080 / 3 / 7×7 / false | full | 11.899200 | 11.858400 | 1.014946 [0.993982, 1.040393] |
| rotation-prototype / 2 | 1920×1080 / 4 / 7×7 / false | pass | 0.539392 | 0.479008 | 1.127226 [1.121045, 1.133649] |
| rotation-prototype / 2 | 1920×1080 / 4 / 7×7 / false | resident | 4.014600 | 3.935800 | 1.049841 [1.013780, 1.100310] |
| rotation-prototype / 2 | 1920×1080 / 4 / 7×7 / false | full | 10.964250 | 10.888200 | 1.010766 [0.977467, 1.041651] |
| rotation-prototype / 2 | 3840×2160 / 3 / 7×7 / false | pass | 1.047584 | 0.897744 | 1.167850 [1.165318, 1.170387] |
| rotation-prototype / 2 | 3840×2160 / 3 / 7×7 / false | resident | 14.417700 | 12.355650 | 1.135023 [1.095378, 1.171861] |
| rotation-prototype / 2 | 3840×2160 / 3 / 7×7 / false | full | 35.621800 | 35.270800 | 0.999901 [0.976291, 1.021532] |
| rotation-prototype / 2 | 1025×257 / 3 / 3×7 / true | pass | 0.034480 | 0.032208 | 1.075266 [1.054293, 1.096672] |
| rotation-prototype / 2 | 1025×257 / 3 / 3×7 / true | resident | 0.517800 | 0.518500 | 1.025266 [0.956260, 1.119161] |
| rotation-prototype / 2 | 1025×257 / 3 / 3×7 / true | full | 6.858400 | 6.995300 | 1.001187 [0.963024, 1.042315] |
| rotation-prototype / 2 | 1920×1080 / 4 / 3×3 / false | pass | 0.373840 | 0.336736 | 1.109855 [1.105955, 1.113876] |
| rotation-prototype / 2 | 1920×1080 / 4 / 3×3 / false | resident | 1.282850 | 1.262300 | 1.018654 [1.000750, 1.036480] |
| rotation-prototype / 2 | 1920×1080 / 4 / 3×3 / false | full | 9.329400 | 9.757050 | 0.968213 [0.933235, 1.000823] |
| balanced-final / 3 | 16×17 / 4 / 3×3 / false | pass | 0.005872 | 0.006016 | 0.984511 [0.909611, 1.066608] |
| balanced-final / 3 | 16×17 / 4 / 3×3 / false | resident | 0.241950 | 0.243450 | 0.979150 [0.905755, 1.035096] |
| balanced-final / 3 | 16×17 / 4 / 3×3 / false | full | 1.220000 | 1.232150 | 0.985543 [0.958096, 1.011259] |
| balanced-final / 3 | 257×259 / 3 / 5×7 / false | pass | 0.016368 | 0.014912 | 1.101390 [1.053525, 1.151336] |
| balanced-final / 3 | 257×259 / 3 / 5×7 / false | resident | 0.290500 | 0.286850 | 1.018161 [0.997928, 1.041420] |
| balanced-final / 3 | 257×259 / 3 / 5×7 / false | full | 1.378750 | 1.395850 | 0.985761 [0.949357, 1.018427] |
| balanced-final / 3 | 1920×1080 / 1 / 5×5 / false | pass | 0.193968 | 0.162272 | 1.196201 [1.193700, 1.198703] |
| balanced-final / 3 | 1920×1080 / 1 / 5×5 / false | resident | 0.712450 | 0.676200 | 1.046991 [1.020900, 1.074838] |
| balanced-final / 3 | 1920×1080 / 1 / 5×5 / false | full | 6.488800 | 6.610650 | 0.978515 [0.963754, 0.992340] |
| balanced-final / 3 | 1920×1080 / 2 / 5×5 / false | pass | 0.196272 | 0.166688 | 1.178996 [1.175268, 1.181884] |
| balanced-final / 3 | 1920×1080 / 2 / 5×5 / false | resident | 0.738500 | 0.698300 | 1.019490 [0.961398, 1.068413] |
| balanced-final / 3 | 1920×1080 / 2 / 5×5 / false | full | 6.656750 | 6.698350 | 0.988721 [0.971924, 1.004996] |
| balanced-final / 3 | 1920×1080 / 3 / 5×5 / false | pass | 0.291952 | 0.253248 | 1.135945 [1.073873, 1.239622] |
| balanced-final / 3 | 1920×1080 / 3 / 5×5 / false | resident | 1.335350 | 1.291200 | 1.040299 [1.007935, 1.090583] |
| balanced-final / 3 | 1920×1080 / 3 / 5×5 / false | full | 7.770250 | 7.885550 | 0.996575 [0.987002, 1.006074] |
| balanced-final / 3 | 1920×1080 / 4 / 5×5 / false | pass | 0.291696 | 0.258032 | 1.128183 [1.123303, 1.133010] |
| balanced-final / 3 | 1920×1080 / 4 / 5×5 / false | resident | 1.244200 | 1.212500 | 1.019552 [0.994841, 1.042233] |
| balanced-final / 3 | 1920×1080 / 4 / 5×5 / false | full | 6.243600 | 6.288400 | 0.996916 [0.983699, 1.012094] |
| balanced-final / 3 | 1920×1080 / 1 / 7×7 / false | pass | 0.511392 | 0.465776 | 1.096454 [1.092309, 1.100594] |
| balanced-final / 3 | 1920×1080 / 1 / 7×7 / false | resident | 1.746050 | 1.735850 | 0.995467 [0.920523, 1.041618] |
| balanced-final / 3 | 1920×1080 / 1 / 7×7 / false | full | 8.868250 | 8.974300 | 1.000087 [0.978021, 1.028724] |
| balanced-final / 3 | 1920×1080 / 2 / 7×7 / false | pass | 0.332592 | 0.280832 | 1.183324 [1.180899, 1.185632] |
| balanced-final / 3 | 1920×1080 / 2 / 7×7 / false | resident | 2.053750 | 1.462800 | 1.041537 [1.005374, 1.101414] |
| balanced-final / 3 | 1920×1080 / 2 / 7×7 / false | full | 8.634600 | 8.500600 | 1.000778 [0.983349, 1.019957] |
| balanced-final / 3 | 1920×1080 / 3 / 7×7 / false | pass | 0.423520 | 0.372704 | 1.140487 [1.135948, 1.145244] |
| balanced-final / 3 | 1920×1080 / 3 / 7×7 / false | resident | 2.338850 | 2.305250 | 1.017722 [1.012149, 1.023885] |
| balanced-final / 3 | 1920×1080 / 3 / 7×7 / false | full | 9.017250 | 8.820250 | 1.005896 [0.992819, 1.019038] |
| balanced-final / 3 | 1920×1080 / 4 / 7×7 / false | pass | 0.535216 | 0.472832 | 1.127598 [1.122980, 1.132352] |
| balanced-final / 3 | 1920×1080 / 4 / 7×7 / false | resident | 3.768250 | 3.697600 | 1.019372 [1.013693, 1.026129] |
| balanced-final / 3 | 1920×1080 / 4 / 7×7 / false | full | 8.723050 | 8.803750 | 0.966366 [0.916793, 0.997740] |
| balanced-final / 3 | 3840×2160 / 3 / 7×7 / false | pass | 1.049184 | 0.898464 | 1.167096 [1.164524, 1.169617] |
| balanced-final / 3 | 3840×2160 / 3 / 7×7 / false | resident | 12.770450 | 13.768900 | 0.967110 [0.903889, 1.032233] |
| balanced-final / 3 | 3840×2160 / 3 / 7×7 / false | full | 34.741000 | 34.660950 | 1.010379 [0.994530, 1.030326] |
| balanced-final / 3 | 1025×257 / 3 / 3×7 / true | pass | 0.034608 | 0.032016 | 1.073726 [1.051585, 1.096809] |
| balanced-final / 3 | 1025×257 / 3 / 3×7 / true | resident | 0.567900 | 0.561900 | 0.981163 [0.945844, 1.017215] |
| balanced-final / 3 | 1025×257 / 3 / 3×7 / true | full | 7.175450 | 7.343150 | 0.991213 [0.967156, 1.014492] |
| balanced-final / 3 | 1920×1080 / 4 / 3×3 / false | pass | 0.373568 | 0.336448 | 1.103817 [1.098807, 1.108778] |
| balanced-final / 3 | 1920×1080 / 4 / 3×3 / false | resident | 1.282050 | 1.272850 | 1.046497 [1.012315, 1.099731] |
| balanced-final / 3 | 1920×1080 / 4 / 3×3 / false | full | 8.834400 | 8.910150 | 0.988639 [0.968329, 1.011115] |
| balanced-final / 4 | 16×17 / 4 / 3×3 / false | pass | 0.005776 | 0.005888 | 0.983336 [0.909637, 1.062794] |
| balanced-final / 4 | 16×17 / 4 / 3×3 / false | resident | 0.245500 | 0.247250 | 0.964623 [0.877145, 1.047128] |
| balanced-final / 4 | 16×17 / 4 / 3×3 / false | full | 1.246700 | 1.278050 | 0.987223 [0.965637, 1.009645] |
| balanced-final / 4 | 257×259 / 3 / 5×7 / false | pass | 0.016112 | 0.014912 | 1.094806 [1.055358, 1.135521] |
| balanced-final / 4 | 257×259 / 3 / 5×7 / false | resident | 0.297000 | 0.302300 | 0.990480 [0.942305, 1.035469] |
| balanced-final / 4 | 257×259 / 3 / 5×7 / false | full | 1.447000 | 1.461250 | 1.000289 [0.971715, 1.029496] |
| balanced-final / 4 | 1920×1080 / 1 / 5×5 / false | pass | 0.192912 | 0.161296 | 1.195781 [1.193051, 1.198650] |
| balanced-final / 4 | 1920×1080 / 1 / 5×5 / false | resident | 0.712900 | 0.693750 | 1.021063 [0.995263, 1.046930] |
| balanced-final / 4 | 1920×1080 / 1 / 5×5 / false | full | 6.591650 | 6.591050 | 1.011016 [0.988348, 1.038475] |
| balanced-final / 4 | 1920×1080 / 2 / 5×5 / false | pass | 0.195136 | 0.165360 | 1.178268 [1.175214, 1.181438] |
| balanced-final / 4 | 1920×1080 / 2 / 5×5 / false | resident | 0.755700 | 0.720800 | 1.034430 [0.998257, 1.071043] |
| balanced-final / 4 | 1920×1080 / 2 / 5×5 / false | full | 6.717600 | 6.626500 | 1.008036 [0.989617, 1.026629] |
| balanced-final / 4 | 1920×1080 / 3 / 5×5 / false | pass | 0.743584 | 0.744288 | 1.002831 [0.993974, 1.011911] |
| balanced-final / 4 | 1920×1080 / 3 / 5×5 / false | resident | 1.325650 | 1.301250 | 1.026294 [1.013507, 1.040583] |
| balanced-final / 4 | 1920×1080 / 3 / 5×5 / false | full | 7.711450 | 7.591500 | 1.009475 [0.986337, 1.043045] |
| balanced-final / 4 | 1920×1080 / 4 / 5×5 / false | pass | 0.285360 | 0.252272 | 1.131539 [1.126756, 1.136222] |
| balanced-final / 4 | 1920×1080 / 4 / 5×5 / false | resident | 1.264850 | 1.219800 | 1.027035 [1.010884, 1.043929] |
| balanced-final / 4 | 1920×1080 / 4 / 5×5 / false | full | 6.565300 | 6.504900 | 1.005306 [0.991004, 1.020014] |
| balanced-final / 4 | 1920×1080 / 1 / 7×7 / false | pass | 0.536080 | 0.484704 | 1.105454 [1.102216, 1.108789] |
| balanced-final / 4 | 1920×1080 / 1 / 7×7 / false | resident | 1.802850 | 1.458050 | 1.055324 [1.025413, 1.100320] |
| balanced-final / 4 | 1920×1080 / 1 / 7×7 / false | full | 9.089100 | 9.078000 | 0.999692 [0.976151, 1.021671] |
| balanced-final / 4 | 1920×1080 / 2 / 7×7 / false | pass | 0.338080 | 0.285392 | 1.184958 [1.182175, 1.187699] |
| balanced-final / 4 | 1920×1080 / 2 / 7×7 / false | resident | 2.644050 | 2.612500 | 1.035223 [1.009437, 1.071541] |
| balanced-final / 4 | 1920×1080 / 2 / 7×7 / false | full | 10.006350 | 9.991850 | 0.995537 [0.979701, 1.010751] |
| balanced-final / 4 | 1920×1080 / 3 / 7×7 / false | pass | 1.027216 | 1.021232 | 1.004350 [0.998509, 1.010351] |
| balanced-final / 4 | 1920×1080 / 3 / 7×7 / false | resident | 2.422950 | 2.363700 | 1.022926 [1.013551, 1.031651] |
| balanced-final / 4 | 1920×1080 / 3 / 7×7 / false | full | 8.951900 | 8.924500 | 1.017518 [0.985414, 1.055991] |
| balanced-final / 4 | 1920×1080 / 4 / 7×7 / false | pass | 0.477872 | 0.424416 | 1.123757 [1.118230, 1.129153] |
| balanced-final / 4 | 1920×1080 / 4 / 7×7 / false | resident | 3.404000 | 3.336200 | 1.017934 [1.012650, 1.022845] |
| balanced-final / 4 | 1920×1080 / 4 / 7×7 / false | full | 8.898150 | 8.927150 | 1.000051 [0.986120, 1.014478] |
| balanced-final / 4 | 3840×2160 / 3 / 7×7 / false | pass | 1.047376 | 0.897584 | 1.167569 [1.165326, 1.169789] |
| balanced-final / 4 | 3840×2160 / 3 / 7×7 / false | resident | 13.321900 | 12.845400 | 1.027159 [0.975404, 1.079655] |
| balanced-final / 4 | 3840×2160 / 3 / 7×7 / false | full | 34.187300 | 34.210000 | 0.999563 [0.990541, 1.007927] |
| balanced-final / 4 | 1025×257 / 3 / 3×7 / true | pass | 0.034384 | 0.032144 | 1.072057 [1.050167, 1.093506] |
| balanced-final / 4 | 1025×257 / 3 / 3×7 / true | resident | 0.548500 | 0.552500 | 0.988871 [0.961091, 1.019551] |
| balanced-final / 4 | 1025×257 / 3 / 3×7 / true | full | 7.024200 | 6.940350 | 0.998293 [0.968113, 1.025548] |
| balanced-final / 4 | 1920×1080 / 4 / 3×3 / false | pass | 0.339856 | 0.307952 | 1.106892 [1.101820, 1.112432] |
| balanced-final / 4 | 1920×1080 / 4 / 3×3 / false | resident | 1.260500 | 1.216650 | 1.057528 [1.023282, 1.115083] |
| balanced-final / 4 | 1920×1080 / 4 / 3×3 / false | full | 8.893700 | 8.863750 | 1.005978 [0.990462, 1.021272] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Production: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_horizontal.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_packed_horizontal.wgsl`.
Native fixtures: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_mirror_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_mirror_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_horizontal_mirror_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_packed_horizontal_mirror_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_mirror_benchmark.rs`.
Ignored validation/raw measurements/provenance: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gaussian-mirror-*`.
