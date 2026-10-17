# Split Q15 Gaussian vertical sums before accumulation

Parent: `66dc377c94259a4259f24043bfab17bafe7b5a6b` (merged PR 255).

The unpacked Gaussian vertical shader previously formed each Q15 product with
a general two-u32 multiply and propagated its carry on every tap. It now sums
the horizontal input's lower and upper base-65536 digits independently and
combines them once after the tap loop. The full 38-bit weighted sum remains
exact without a native shader u64 type or intermediate rounding.

## Arithmetic and scope

The unchanged planner selects Q15 for both axes when either kernel is seven
taps, for all 1..=4 logical channels. Each horizontal input H is at most
255*32768 = 8,355,840 < 2^23, including the constant-border value. The host
coefficients are nonnegative integers with mass 32768. Write H = a + 65536*b,
where 0 <= a <= 65535 and 0 <= b <= 127. Then accumulate:

```
L = sum(a_i * w_i) <= 65535*32768 = 2,147,450,880 < 2^31
U = sum(b_i * w_i) <= 127*32768   = 4,161,536 < 2^22
S = L + 65536*U
round_half_up(S / 2^30) = (U + floor(L/65536) + 8192) >> 14
```

Both digit sums and the final addition fit u32. The last identity is exact
because the discarded remainder of L is below 65536 and the denominator and
half-up offset are multiples of 65536. It introduces no premature rounding.
The result is clamped to 255 exactly as before; unused physical output lanes
stay zero. Relative to the source's general multiply, each Q15 tap replaces
four limb multiplications and carry logic with two bounded products and sums.
These are source operations, not device instruction or bandwidth counts.

This unpacked shader also serves one-channel Q8. Its sum has mass 256 and
horizontal maximum 65,280, hence S+32768 <= 16,744,448 fits u32. The final
`(S+32768)>>16` replaces the same old limb formula exactly. Packed multi-channel
Q8 shaders are byte-for-byte unchanged and provide an unchanged timing control.

The entire production Gaussian host file, parameter and horizontal shaders,
coefficient generation, validation/error precedence, lazy pipeline cache,
scratch size, two passes/one submission, allocations, metadata, ownership,
receipts and explicit transfer/wait boundaries remain unchanged. No public API,
dependency, core, pool or unsafe change is introduced.

## Correctness and integration

- **350 native correctness tests:** 145 image, 184 voxel/readback, 18 filtering,
  2 transfer facade and 1 Gaussian facade, including ignored GPU tests and
  excluding benchmarks. All-feature/all-target Clippy with warnings denied,
  CPU-only, minimal/CI WASM with warnings denied, formatting and whitespace pass.
- **97,850,720 parent/u64-CPU physical byte comparisons** across 724 raw
  vertical fixtures. Inputs sample the full 23-bit range, slightly beyond valid
  horizontal sums, with half-up boundaries, lower-digit carry boundaries,
  maximum/clamp values and poisoned trailing words. Tests cover all channels,
  all five borders, vertical 3/5/7 taps with a seven-tap horizontal axis, narrow
  and broad weights, tiny/odd/strip dimensions and forced one-group grid strides
  through 1025x1025. The oracle uses independent border mapping, direct u64
  weighted sums, addition and division, rather than the split-digit formula.
  Both complete physical outputs are checked, including unused channels.
- **60,210,900 complete-parent/CPU logical byte comparisons** across
  900 paired API fixtures (1,800 executions), all nine combinations of 3/5/7
  axes, all channels/borders and five shapes, with matching metadata and receipts
  before/after readback. Existing extreme-sigma, mass, numeric/rounding,
  validation/capacity, ROI, concurrency, profiling, scratch and ownership checks
  remain unchanged and pass. One-channel and packed Q8 controls retain outputs.
- **508 original GPU/image Rust/WGSL sources unchanged**, with **516 validated
  source/executable hashes**. Only the unpacked vertical WGSL and an appended
  native test-module declaration differ in original sources. The immediate-parent
  complete planner, coefficients, pipelines and compute body are frozen with
  explicit type/name/visibility/external-cache substitutions; its original
  unpacked vertical shader is frozen separately, byte-for-byte. All image
  transfers and point-cloud sources compare unchanged.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=vertical_pass_resident_and_full checks_drop_recycle_excluded=true`. Adapter model names are omitted.

After all builds/tests, two sequential surveys each use three warm pairs and
thirty measured pairs for ten cases. Eight use seven-tap Q15, one uses the
also-affected one-channel Q8 path, and one uses the unchanged packed Q8 path.
All **60** rows are retained below. The affected Q15 rows have:

- Vertical pass: **13 favorable / 2 inconclusive / 1 adverse**.
- Complete resident API: **6 favorable / 10 inconclusive / 0 adverse**.
- Upload/filter/readback: **0 favorable / 15 inconclusive / 1 adverse**.

The two Q8 controls together have vertical **2 favorable / 2 inconclusive / 0 adverse**;
resident **2 favorable / 2 inconclusive / 0 adverse**; full **0 favorable / 3 inconclusive / 1 adverse**.
One-channel Q8 shares the changed shader's final rounding simplification;
multi-channel packed Q8 retains the original shader. Their individual rows
are shown rather than attributing all observed differences to Q15 arithmetic.

For 1920x1080 four-channel seven-tap input, vertical medians are
1.614576 -> 1.606832 ms and 1.567696 -> 1.562368 ms; resident medians are
2.695150 -> 2.667800 ms and 2.732850 -> 2.689150 ms. For 3840x2160 three-channel seven-tap input,
vertical medians are 6.570176 -> 6.420032 ms and 6.520048 -> 6.412240 ms; resident medians are
14.038550 -> 12.502750 ms and 12.415500 -> 12.242000 ms. These local observations do not establish
a general transfer-inclusive speedup.

The clearest repeated Q15 gains are at 1920x1080: one-channel vertical
medians are 0.316112 -> 0.274528 ms and 0.313920 -> 0.273152 ms (about 13% lower), and two-channel
medians are 0.722176 -> 0.645552 ms and 0.560528 -> 0.500608 ms (about 10% lower). Three/four-channel
changes are smaller. The second survey's 1080p three-channel vertical interval
is adverse, with median 1.309344 -> 1.312176 ms and paired ratio about 0.986.
The second survey's 257x259 three-channel full interval is also adverse, despite
its lower median. An unchanged packed-Q8 full interval is adverse in survey one.
All are retained; none is erased or counted as a Q15 gain. No per-device or
per-channel dispatch heuristic is inferred from these local observations.

- `pass`: timestamps bracket only the old/new vertical GPU passes. The unchanged
  horizontal stage prepares a common scratch input before timing. Vertical
  passes share one command buffer with alternating order, equal uniforms,
  dispatch/layouts and independently owned outputs. Explicit collection reads
  32 query bytes outside timestamps; both images are checked against CPU.
- `resident`: complete two-pass Gaussian API and `wait_idle`, starting with a
  resident source. Verification readback and output checks follow timing.
- `full`: upload, complete two-pass Gaussian and readback. Allocation, conversion
  and synchronization are included; assertions, destruction and recycling follow.

Resident/full order rotates four variants per round; old/new samples stay
paired by round. CPU reference construction, pipeline compilation, assertions,
destruction and recycling are excluded. Ratios are geometric means of paired
original/new samples; above one favors the new route. Pointwise percentile
intervals resample paired log ratios 10,000 times, seed
`2026101706 + 100*run + 10*case + route_index`, route order pass/resident/full,
with sorted bounds 249/9749. No multiple-comparison correction is applied.
Zero times would suppress ratios; all raw samples are retained.

Run 1: observed-quiet **false**; busy-process counts 2 before / 1 after. GPU/memory utilization samples `[[[1, 0]], [[2, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 2: observed-quiet **true**; busy-process counts 0 before / 0 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all GPU/memory
samples at most 5%. Surveys wait two seconds between snapshots; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post samples cannot
establish isolation throughout every timed pair. These are shared-host diagnostic
observations, with no portable throughput or application-latency guarantee.
Process identities remain in ignored receipts. Nonconstant timing cases use
Reflect101, the constant case uses [19,73,129,251], and both sigmas are 1.2.

| Run | Shape / channels / kernel / constant border | Boundary | Original median ms | New median ms | Paired original/new ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 16x17 / 4 / 7 / false | pass | 0.010928 | 0.009728 | 1.125413 [1.089750, 1.162157] |
| 1 | 16x17 / 4 / 7 / false | resident | 0.291850 | 0.313950 | 0.945936 [0.864469, 1.018138] |
| 1 | 16x17 / 4 / 7 / false | full | 1.223700 | 1.263300 | 0.987810 [0.933056, 1.031024] |
| 1 | 257x259 / 3 / 7 / false | pass | 0.038288 | 0.037072 | 1.029836 [1.015143, 1.044389] |
| 1 | 257x259 / 3 / 7 / false | resident | 0.326300 | 0.318950 | 0.981714 [0.937399, 1.027224] |
| 1 | 257x259 / 3 / 7 / false | full | 1.379550 | 1.404250 | 0.984909 [0.956431, 1.013959] |
| 1 | 1920x1080 / 1 / 7 / false | pass | 0.316112 | 0.274528 | 1.152512 [1.150029, 1.155937] |
| 1 | 1920x1080 / 1 / 7 / false | resident | 0.896950 | 0.849000 | 1.066666 [1.033331, 1.099725] |
| 1 | 1920x1080 / 1 / 7 / false | full | 7.083000 | 7.150600 | 0.997471 [0.964773, 1.031849] |
| 1 | 1920x1080 / 2 / 7 / false | pass | 0.722176 | 0.645552 | 1.118378 [1.116962, 1.119798] |
| 1 | 1920x1080 / 2 / 7 / false | resident | 1.170250 | 1.117450 | 1.060878 [1.040495, 1.083356] |
| 1 | 1920x1080 / 2 / 7 / false | full | 7.668850 | 7.502600 | 0.999234 [0.971704, 1.028572] |
| 1 | 1920x1080 / 3 / 7 / false | pass | 1.421328 | 1.426656 | 0.998883 [0.997599, 1.000129] |
| 1 | 1920x1080 / 3 / 7 / false | resident | 2.314800 | 2.290600 | 1.012169 [0.999103, 1.026194] |
| 1 | 1920x1080 / 3 / 7 / false | full | 8.958850 | 8.779100 | 1.004274 [0.985837, 1.023944] |
| 1 | 1920x1080 / 4 / 7 / false | pass | 1.614576 | 1.606832 | 1.005147 [1.004029, 1.006284] |
| 1 | 1920x1080 / 4 / 7 / false | resident | 2.695150 | 2.667800 | 1.011595 [0.998976, 1.028757] |
| 1 | 1920x1080 / 4 / 7 / false | full | 7.112150 | 7.067300 | 1.003104 [0.987812, 1.018260] |
| 1 | 3840x2160 / 3 / 7 / false | pass | 6.570176 | 6.420032 | 1.037334 [1.018529, 1.056644] |
| 1 | 3840x2160 / 3 / 7 / false | resident | 14.038550 | 12.502750 | 1.080795 [1.037685, 1.123143] |
| 1 | 3840x2160 / 3 / 7 / false | full | 35.335400 | 34.821750 | 1.007567 [0.992470, 1.022519] |
| 1 | 1025x257 / 3 / 7 / true | pass | 0.112736 | 0.110896 | 1.019105 [1.014584, 1.023708] |
| 1 | 1025x257 / 3 / 7 / true | resident | 0.534100 | 0.544450 | 0.932914 [0.830975, 1.011652] |
| 1 | 1025x257 / 3 / 7 / true | full | 6.548100 | 6.681700 | 0.989798 [0.958136, 1.021289] |
| 1 | 1920x1080 / 1 / 5 / false | pass | 0.342848 | 0.338448 | 1.014008 [1.011845, 1.016101] |
| 1 | 1920x1080 / 1 / 5 / false | resident | 0.915800 | 0.870150 | 1.026727 [1.001547, 1.053401] |
| 1 | 1920x1080 / 1 / 5 / false | full | 10.916650 | 11.089600 | 0.980145 [0.952278, 1.008064] |
| 1 | 1920x1080 / 4 / 5 / false | pass | 0.759552 | 0.759488 | 0.999909 [0.998746, 1.001044] |
| 1 | 1920x1080 / 4 / 5 / false | resident | 1.568750 | 1.540400 | 1.011149 [0.997456, 1.025499] |
| 1 | 1920x1080 / 4 / 5 / false | full | 8.171700 | 8.278900 | 0.951037 [0.894490, 0.993475] |
| 2 | 16x17 / 4 / 7 / false | pass | 0.010848 | 0.009664 | 1.130531 [1.093808, 1.168690] |
| 2 | 16x17 / 4 / 7 / false | resident | 0.221000 | 0.223000 | 0.978047 [0.881432, 1.054398] |
| 2 | 16x17 / 4 / 7 / false | full | 1.107600 | 1.116850 | 0.989834 [0.964508, 1.015578] |
| 2 | 257x259 / 3 / 7 / false | pass | 0.037872 | 0.036512 | 1.029917 [1.014324, 1.046193] |
| 2 | 257x259 / 3 / 7 / false | resident | 0.257950 | 0.271550 | 0.980032 [0.944116, 1.016255] |
| 2 | 257x259 / 3 / 7 / false | full | 1.252900 | 1.239800 | 0.962598 [0.922177, 0.993642] |
| 2 | 1920x1080 / 1 / 7 / false | pass | 0.313920 | 0.273152 | 1.149977 [1.148489, 1.151480] |
| 2 | 1920x1080 / 1 / 7 / false | resident | 0.931550 | 0.920100 | 1.039379 [1.010927, 1.072965] |
| 2 | 1920x1080 / 1 / 7 / false | full | 6.899450 | 6.696450 | 1.017796 [0.970185, 1.076126] |
| 2 | 1920x1080 / 2 / 7 / false | pass | 0.560528 | 0.500608 | 1.118794 [1.117418, 1.120226] |
| 2 | 1920x1080 / 2 / 7 / false | resident | 1.202050 | 1.132300 | 1.042175 [1.009137, 1.072749] |
| 2 | 1920x1080 / 2 / 7 / false | full | 6.786600 | 6.752600 | 1.024654 [0.976889, 1.076307] |
| 2 | 1920x1080 / 3 / 7 / false | pass | 1.309344 | 1.312176 | 0.986081 [0.964325, 0.997727] |
| 2 | 1920x1080 / 3 / 7 / false | resident | 2.248000 | 2.218900 | 0.998845 [0.971027, 1.023190] |
| 2 | 1920x1080 / 3 / 7 / false | full | 8.354600 | 8.382350 | 0.992178 [0.964193, 1.016429] |
| 2 | 1920x1080 / 4 / 7 / false | pass | 1.567696 | 1.562368 | 1.004204 [1.002249, 1.006834] |
| 2 | 1920x1080 / 4 / 7 / false | resident | 2.732850 | 2.689150 | 1.017544 [1.005055, 1.030263] |
| 2 | 1920x1080 / 4 / 7 / false | full | 6.998450 | 7.011100 | 0.998786 [0.985364, 1.012232] |
| 2 | 3840x2160 / 3 / 7 / false | pass | 6.520048 | 6.412240 | 1.019719 [0.997087, 1.041430] |
| 2 | 3840x2160 / 3 / 7 / false | resident | 12.415500 | 12.242000 | 1.028330 [0.990012, 1.075408] |
| 2 | 3840x2160 / 3 / 7 / false | full | 36.522200 | 36.237600 | 1.001271 [0.981006, 1.022107] |
| 2 | 1025x257 / 3 / 7 / true | pass | 0.113024 | 0.110976 | 1.020691 [1.017615, 1.023722] |
| 2 | 1025x257 / 3 / 7 / true | resident | 0.495150 | 0.515900 | 0.954445 [0.853962, 1.033121] |
| 2 | 1025x257 / 3 / 7 / true | full | 6.105150 | 6.024400 | 1.004225 [0.976504, 1.031231] |
| 2 | 1920x1080 / 1 / 5 / false | pass | 0.331776 | 0.327552 | 1.013132 [1.011460, 1.014866] |
| 2 | 1920x1080 / 1 / 5 / false | resident | 0.803600 | 0.796950 | 1.034212 [1.009004, 1.062288] |
| 2 | 1920x1080 / 1 / 5 / false | full | 10.416750 | 10.436450 | 1.013148 [0.971059, 1.068068] |
| 2 | 1920x1080 / 4 / 5 / false | pass | 0.773888 | 0.773824 | 0.999260 [0.998028, 1.000498] |
| 2 | 1920x1080 / 4 / 5 / false | resident | 1.654050 | 1.662150 | 0.994046 [0.948799, 1.026432] |
| 2 | 1920x1080 / 4 / 5 / false | full | 9.957600 | 9.980200 | 1.004763 [0.964538, 1.048192] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Production: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_vertical.wgsl`.
Native evidence: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_split_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_split_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_split_vertical_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gaussian_split_benchmark.rs`.
Ignored validation, raw samples and provenance: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gaussian-split-*`.
