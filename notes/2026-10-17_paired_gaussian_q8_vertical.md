# Read packed Q8 Gaussian channels in pairs

Parent: `df4309672738fc4c1c5f169ef0443a8391d16e50` (merged PR 254).

The packed Q8 vertical shader previously looped over logical channels and read
the same packed word separately for its low and high halves. It now reads each
word once per pair, unpacks both u16 components and accumulates two independent
integer sums. The final store masks the inactive high component for odd channel
counts, preserving zero unused physical output lanes.

## Source work and exact arithmetic

The unchanged planner selects this shader only for 2..=4 channels with both
kernel axes 3 or 5 and Q8 weights. A 7-tap axis selects the original unpacked Q15
pair of shaders; one-channel calls also retain their original shaders. The host
options, f64 coefficient generation, validation/error order, lazy caches, scratch
capacity, API body, metadata, ownership and receipts are unchanged.

| Logical channels | Original storage-word loads per in-bounds vertical tap | New loads |
| --- | ---: | ---: |
| 2 | 2 | 1 |
| 3 | 3 | 2 |
| 4 | 4 | 2 |

These are source-level load expressions, not physical memory-transaction or
bandwidth counts. A five-tap, four-channel pixel with all rows in bounds uses
10 rather than 20 word loads. Border coordinate and weight work is shared per
pair; the odd pair may compute an unused high sum which is never stored.
Compiler caches, coalescing and code generation affect actual device work.

For production Q8, each nonnegative horizontal sum is at most 255*256 = 65,280.
Vertical weights have mass 256, so each sum is at most 16,711,680 and adding the
half-up rounding offset gives at most 16,744,448. Both fit in u32. Thus the old
Q8 low limb has no carry, and `(sum + 32768) >> 16`, clamped to 255, equals the
previous wide-result formula exactly. The two vec2 components retain the same
per-channel tap order. The unreachable Q15 product/high-limb code is removed
only from this packed-Q8 shader; the real Q15 shader is byte-for-byte unchanged.

There are still two ordered passes in one submission, with the same temporary
buffer and output allocation. No payload transfer, readback, synchronization,
pool, dependency, core or unsafe change is introduced. No automatic CPU/GPU
placement or public API is added.

## Correctness and integration

- **348 native correctness tests:** 143 image, 184 voxel/readback, 18 filtering,
  2 transfer facade and 1 Gaussian facade, including ignored GPU tests and
  excluding benchmarks. All-feature/all-target Clippy with warnings denied,
  CPU-only, minimal/CI WASM with warnings denied, formatting and whitespace pass.
- **57,330,360 original/u64-CPU physical byte comparisons** in 363 raw
  vertical fixtures. Arbitrary packed u16 values cover a wider range than valid
  horizontal sums, including all-one hot words, dirty unused odd-channel high
  halves, poisoned trailing input, all five borders, tiny/odd/strip inputs and
  1025x1025 one-group grid-stride work. Both complete physical outputs match an
  independent u64 weighted-sum/division oracle, including inactive output lanes.
  Even 65,535*256 + 32,768 fits u32 for these extended probes.
- **46,830,700 complete-parent/CPU logical bytes** across 700 paired
  API fixtures (1,400 executions), all channels, five borders, mixed 3/5/7 axes,
  tiny/odd/strip dimensions, matching metadata and receipts before/after readback.
  Q15 and one-channel controls retain their old outputs. Existing weight-mass,
  numeric/rounding, capacity/error, ROI/metadata, concurrent lazy cache,
  profiling, scratch reuse and live-output tests remain unchanged and pass.
- **504 unchanged original GPU/image Rust/WGSL sources** and **512 validated
  source/executable hashes**. Only the packed vertical shader and an appended
  native module declaration differ in original sources. The complete immediate-
  parent plan, coefficients, pipeline cache and compute call are frozen, with
  explicit type/visibility/name and external-cache substitutions. The original
  packed vertical shader is retained separately, byte-for-byte. The complete
  production Gaussian host file, horizontal/parameter and unpacked shaders,
  image transfers and all point-cloud sources compare unchanged.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=vertical_pass_resident_and_full checks_drop_recycle_excluded=true`. Adapter model names are omitted.

After all builds/tests, two sequential surveys use three warm pairs and thirty
measured pairs per ten cases, alternating old/new order. Four resident/full
variants rotate in order each round; their samples remain paired by round.
There are eight affected packed-Q8 cases and two unchanged controls (one-channel
Q8 and three-channel Q15). Every one of the **60** boundary/result rows is below.
CPU references, fixture/pipeline creation and compilation, assertions, output
destruction and recycling are outside timing.

- `pass`: timestamps bracket only the old/new vertical GPU passes. The unchanged
  horizontal stage prepares a common scratch input before timing. Both vertical
  passes share one command buffer with alternating order, equivalent uniforms,
  layouts, dispatch and independently owned outputs. Explicit collection reads
  32 query bytes outside the timestamps; both images are checked against CPU.
  This excludes horizontal compute and is not API or frame latency.
- `resident`: complete two-pass Gaussian API call and `wait_idle` with an already
  uploaded image. Verification readback and output checks follow timing.
- `full`: upload, complete two-pass Gaussian and readback. Channel conversion,
  allocation and transfer/synchronization costs are included; assertions,
  destruction and recycling follow timing.

Ratios are geometric means of paired original/new sample ratios; above one
favors the new route. Pointwise percentile intervals resample paired log ratios
10,000 times, seed `2026101705 + 100*run + 10*case + route_index`, route order
pass/resident/full, with sorted bounds 249/9749. No multiple-comparison correction
is applied. Zero durations would suppress ratios; every raw sample is retained.

All **16/16** affected vertical-pass intervals favor the new shader. Affected
resident calls have **11 favorable / 5 inconclusive / 0 adverse** intervals;
full calls have **4 favorable / 12 inconclusive / 0 adverse**. Unchanged vertical
controls all cross one. One unchanged one-channel resident control favors the
new route only in run two; that result is not attributed to this optimization.

For 1920x1080 four-channel, five-tap input, vertical medians are
0.879632 -> 0.480384 ms and 0.880736 -> 0.480944 ms (about 45% lower).
Complete resident medians are 1.773350 -> 1.376900 ms and
1.647250 -> 1.234550 ms. For 3840x2160 three-channel, five-tap input,
vertical medians are 2.387312 -> 1.714464 ms and 2.574304 -> 1.856528 ms;
resident medians are 7.180650 -> 6.551800 ms and 7.295600 -> 6.633550 ms.
These local observations do not establish a general transfer-inclusive speedup.

Run 1: observed-quiet **true**; busy-process counts 0 before / 0 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 2: observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all GPU/memory
samples at most 5%. Surveys wait two seconds between snapshots; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post sampling cannot
prove isolation during every timed pair, including a run meeting those limits.
These are shared-host diagnostic observations, not portable throughput or
application-latency guarantees. Process identities remain in ignored receipts.

Nonconstant cases use Reflect101; the constant case uses [19,73,129,251]. Both
sigmas are 1.2 in timing fixtures. Correctness covers all five borders and mixed
axes/extreme weights independently. First pipeline compilation stays lazy and
outside warm timing.

| Run | Shape / channels / kernel / constant border | Boundary | Original median ms | New median ms | Paired original/new ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 16x17 / 4 / 3 / false | pass | 0.006656 | 0.005568 | 1.219784 [1.156074, 1.286925] |
| 1 | 16x17 / 4 / 3 / false | resident | 0.204250 | 0.208550 | 0.963693 [0.875568, 1.037741] |
| 1 | 16x17 / 4 / 3 / false | full | 1.022150 | 1.021700 | 0.989744 [0.961341, 1.018833] |
| 1 | 257x259 / 3 / 5 / false | pass | 0.021680 | 0.016176 | 1.338511 [1.305919, 1.371637] |
| 1 | 257x259 / 3 / 5 / false | resident | 0.223800 | 0.216950 | 1.030139 [0.992587, 1.065125] |
| 1 | 257x259 / 3 / 5 / false | full | 1.141350 | 1.153350 | 1.002806 [0.979008, 1.027358] |
| 1 | 1920x1080 / 2 / 3 / false | pass | 0.254384 | 0.157776 | 1.612758 [1.609883, 1.615589] |
| 1 | 1920x1080 / 2 / 3 / false | resident | 0.798650 | 0.686950 | 1.153800 [1.098852, 1.207158] |
| 1 | 1920x1080 / 2 / 3 / false | full | 6.369750 | 6.214050 | 1.039567 [0.985791, 1.092987] |
| 1 | 1920x1080 / 3 / 5 / false | pass | 0.545184 | 0.382624 | 1.424846 [1.422705, 1.426986] |
| 1 | 1920x1080 / 3 / 5 / false | resident | 1.205100 | 1.061000 | 1.120949 [1.086150, 1.153575] |
| 1 | 1920x1080 / 3 / 5 / false | full | 7.090450 | 6.895750 | 1.046953 [0.988541, 1.110987] |
| 1 | 1920x1080 / 4 / 5 / false | pass | 0.879632 | 0.480384 | 1.830083 [1.828587, 1.831583] |
| 1 | 1920x1080 / 4 / 5 / false | resident | 1.773350 | 1.376900 | 1.240670 [1.166966, 1.306088] |
| 1 | 1920x1080 / 4 / 5 / false | full | 6.181900 | 5.783800 | 1.058259 [0.994382, 1.112616] |
| 1 | 3840x2160 / 4 / 3 / false | pass | 2.097904 | 1.284768 | 1.630536 [1.624087, 1.634303] |
| 1 | 3840x2160 / 4 / 3 / false | resident | 8.220650 | 7.209300 | 1.097631 [1.039609, 1.143268] |
| 1 | 3840x2160 / 4 / 3 / false | full | 26.595950 | 25.630550 | 1.099221 [1.020665, 1.185699] |
| 1 | 3840x2160 / 3 / 5 / false | pass | 2.387312 | 1.714464 | 1.383587 [1.381504, 1.385742] |
| 1 | 3840x2160 / 3 / 5 / false | resident | 7.180650 | 6.551800 | 1.098716 [1.070656, 1.125058] |
| 1 | 3840x2160 / 3 / 5 / false | full | 28.569100 | 27.453800 | 1.060208 [1.009306, 1.107916] |
| 1 | 1025x257 / 3 / 3 / true | pass | 0.041712 | 0.031712 | 1.314144 [1.298086, 1.330036] |
| 1 | 1025x257 / 3 / 3 / true | resident | 0.385450 | 0.378650 | 0.965328 [0.853507, 1.049864] |
| 1 | 1025x257 / 3 / 3 / true | full | 5.337950 | 5.315800 | 1.000696 [0.977324, 1.024757] |
| 1 | 1920x1080 / 1 / 5 / false | pass | 0.287904 | 0.288064 | 1.000975 [0.998421, 1.003520] |
| 1 | 1920x1080 / 1 / 5 / false | resident | 0.821000 | 0.818300 | 0.992237 [0.947300, 1.039577] |
| 1 | 1920x1080 / 1 / 5 / false | full | 9.282850 | 9.412950 | 0.978484 [0.930449, 1.024549] |
| 1 | 1920x1080 / 3 / 7 / false | pass | 1.424944 | 1.425072 | 1.000060 [0.998923, 1.001234] |
| 1 | 1920x1080 / 3 / 7 / false | resident | 2.399600 | 2.401000 | 1.004651 [0.968501, 1.041128] |
| 1 | 1920x1080 / 3 / 7 / false | full | 9.764950 | 9.754500 | 1.012456 [0.984091, 1.048875] |
| 2 | 16x17 / 4 / 3 / false | pass | 0.006656 | 0.005456 | 1.220956 [1.157316, 1.286642] |
| 2 | 16x17 / 4 / 3 / false | resident | 0.190350 | 0.187000 | 1.014385 [0.926950, 1.091928] |
| 2 | 16x17 / 4 / 3 / false | full | 0.983950 | 0.975150 | 1.019947 [0.993948, 1.049197] |
| 2 | 257x259 / 3 / 5 / false | pass | 0.021568 | 0.016192 | 1.335719 [1.301505, 1.372168] |
| 2 | 257x259 / 3 / 5 / false | resident | 0.211450 | 0.205350 | 1.038465 [1.016594, 1.060709] |
| 2 | 257x259 / 3 / 5 / false | full | 1.127750 | 1.129650 | 0.998118 [0.980472, 1.016362] |
| 2 | 1920x1080 / 2 / 3 / false | pass | 0.254352 | 0.158032 | 1.610477 [1.607674, 1.613326] |
| 2 | 1920x1080 / 2 / 3 / false | resident | 0.793250 | 0.703400 | 1.146616 [1.090955, 1.202781] |
| 2 | 1920x1080 / 2 / 3 / false | full | 6.289750 | 6.250050 | 1.033347 [0.967746, 1.105971] |
| 2 | 1920x1080 / 3 / 5 / false | pass | 0.545088 | 0.382512 | 1.425347 [1.423085, 1.427581] |
| 2 | 1920x1080 / 3 / 5 / false | resident | 1.205600 | 1.075050 | 1.121810 [1.092319, 1.148117] |
| 2 | 1920x1080 / 3 / 5 / false | full | 6.946200 | 6.816400 | 0.970683 [0.905464, 1.034993] |
| 2 | 1920x1080 / 4 / 5 / false | pass | 0.880736 | 0.480944 | 1.830054 [1.827819, 1.832037] |
| 2 | 1920x1080 / 4 / 5 / false | resident | 1.647250 | 1.234550 | 1.318816 [1.295752, 1.341923] |
| 2 | 1920x1080 / 4 / 5 / false | full | 6.155000 | 5.803800 | 1.068919 [1.053780, 1.083904] |
| 2 | 3840x2160 / 4 / 3 / false | pass | 2.634944 | 1.605904 | 1.650588 [1.637580, 1.675380] |
| 2 | 3840x2160 / 4 / 3 / false | resident | 9.046200 | 8.044550 | 1.124480 [1.098220, 1.151592] |
| 2 | 3840x2160 / 4 / 3 / false | full | 34.971900 | 28.077450 | 1.178413 [1.051316, 1.319580] |
| 2 | 3840x2160 / 3 / 5 / false | pass | 2.574304 | 1.856528 | 1.387358 [1.386304, 1.388465] |
| 2 | 3840x2160 / 3 / 5 / false | resident | 7.295600 | 6.633550 | 1.097931 [1.068904, 1.125894] |
| 2 | 3840x2160 / 3 / 5 / false | full | 28.707850 | 28.160700 | 1.009895 [0.956743, 1.062488] |
| 2 | 1025x257 / 3 / 3 / true | pass | 0.041984 | 0.032128 | 1.314560 [1.298260, 1.331242] |
| 2 | 1025x257 / 3 / 3 / true | resident | 0.401200 | 0.395650 | 0.980110 [0.850471, 1.091752] |
| 2 | 1025x257 / 3 / 3 / true | full | 5.806800 | 5.645850 | 1.024472 [0.992378, 1.056535] |
| 2 | 1920x1080 / 1 / 5 / false | pass | 0.316640 | 0.316752 | 1.000174 [0.998562, 1.001779] |
| 2 | 1920x1080 / 1 / 5 / false | resident | 0.792700 | 0.779000 | 1.041458 [1.004964, 1.076516] |
| 2 | 1920x1080 / 1 / 5 / false | full | 9.613950 | 9.715150 | 0.990038 [0.965248, 1.018752] |
| 2 | 1920x1080 / 3 / 7 / false | pass | 1.469504 | 1.468704 | 0.999917 [0.998518, 1.001230] |
| 2 | 1920x1080 / 3 / 7 / false | resident | 3.885050 | 3.843350 | 1.010889 [0.992709, 1.027814] |
| 2 | 1920x1080 / 3 / 7 / false | full | 11.366000 | 11.299550 | 1.005569 [0.974189, 1.035426] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Production:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_packed_vertical.wgsl`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_pairs_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_pairs_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_pairs_vertical_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_pairs_benchmark.rs`.
Ignored validation, raw samples and provenance:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gaussian-pairs-*`.
