# Exact power-of-two GPU CLAHE geometry

Date: 2026-10-12. Named slice: GPU exact geometry ratios.

## Result and mathematical boundary

CLAHE's finite non-negative binary64 ratio helper now recognizes positive
power-of-two divisors. For a nonzero u32 numerator it normalizes that exact
integer using the existing `fp_uint`, then subtracts the denominator's base-two
exponent. The ratio's encoded binary64 words are identical to correctly rounded
CPU division. Zero numerators retain their existing zero return. Other positive
divisors retain the original long-division loop, normalization and ties-to-even
rounding, including exact integer ratios with non-power-of-two divisors.

An integer below 2^32 is exactly representable with binary64's 53-bit significand.
Multiplying it by 2^-k for k in 0..=31 changes only the unbiased exponent, not the
significand. The minimum nonzero result is 2^-31, well above the normal/subnormal
boundary. Consequently this path needs no rounding or fractional division steps.
Validated tile widths/heights are positive; the existing ratio helper continues
to be an internal normal-or-zero subset rather than a general IEEE-754 divider.

This changes only geometry's ratio calculation in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/binary64_positive.wgsl`.
Axis index/weight operation order, clipping, lookup rounding and interpolation
are unchanged. Both axes and every geometry entry are still generated on the
GPU each call. The shared helper is included in the application shader, whose
entry point does not call `fp_ratio`; all arithmetic it uses remains unchanged.

No pipeline family, shader binding, workgroup, uniform size, allocation,
retention budget, feature/dependency or public signature is added. Direct work
still submits three passes and collects 48 query bytes; split histograms submit
four passes and collect 64 query bytes. Axis storage remains 16*(width+height)
bytes in the existing submitted scratch leases. Image payload uploads/readbacks
remain explicit, and metadata/transfer history is preserved.

## Survey and retention decision

Before changing the ratio helper, the same paired diagnostic ran on two copies
of the previous shader across 24 conditions. At 256x256 / 8x8, geometry's median
share of summed GPU pass timestamps was 12.92% for mixed pixels and 24.07% for
hot pixels (7.152/6.784 us). At 3840x2160 / 8x8 it was 0.39%/0.99%
(26.640/40.192 us). These are ratios of GPU pass times, not fractions of host wall
time; allocation, submission and waits also affect completed calls.

A new geometry-retention policy was deferred: its ownership, limits and changing
pass/receipt contracts would need evidence of material completed-call savings.
The existing bounded scratch already reuses allocations, and shortening exact
geometry arithmetic offers a focused change without retained immutable tables.
ROADMAP records that deferral and this completed arithmetic slice separately.

An initial prototype also normalized exact integer ratios for general divisors.
Its extra test slightly worsened several general-divisor geometry conditions,
so it was removed. Final non-power-of-two ratios keep the original algorithm.
The private `gpu-clahe-geometry-integer-pilot-*` files retain that adverse pilot;
its timings are not the final results below.

## Correctness and compilation

Final native suite: 137 passing tests, comprising 110 unit, four CDF, six CLAHE,
five equalization, three histogram and nine shared stage tests. Nine native
performance diagnostics remain ignored unless explicitly invoked. Image/point
regressions cover retained outputs, reused/cleared scratch, runtime ownership,
ROI/metadata, byte/word parity, transfer history, dispatch limits and timestamp
capacity. No browser execution is claimed.

The new independent ratio/axis proof in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_geometry_ratio_tests.rs`
checks 332,917 cases against both native CPU f64 and the frozen previous shader.
Each emits two ratio words, two axis indices and two weight words: 1,997,502
exact result words per GPU variant, 3,995,004 comparisons overall. It covers:

- Numerators 0..=1,024 with every denominator 1..=256.
- All 32 power-of-two u32 divisors, zero/below/at/above denominator cases,
  u32::MAX boundaries, 128 deterministic random numerators per divisor, and
  single/maximal tile counts.
- Exact integer ratios and adjacent numerators for non-power-of-two tile sizes,
  including 80/135/240/270/480 and large divisors through u32::MAX.
- 65,536 deterministic general u32 ratios, zeros and independently chosen
  positive tile counts.

The ratio output structure's WGSL vec4 alignment is respected: its 32-byte rows
contain two ignored padding words, which are excluded from the six comparisons.
GPU and CPU compare raw words, not a numerical tolerance or only final bytes.

All-feature/all-target strict Clippy, default-feature strict Clippy and two
WASM builds (`gpu-wgpu,gpu-image` and `gpu-image,gpu-aoso-staging`) pass. WASM is
compilation-only. Workspace formatting and diff checks pass. No CI check name
or feature matrix job count changes; the existing 114 named checks gate merge.

Private regression/build logs are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:
`gpu-clahe-geometry-baseline-tests.log`, `gpu-clahe-geometry-regressions.log`,
`gpu-clahe-geometry-clippy.log`, `gpu-clahe-geometry-default-clippy.log`,
`gpu-clahe-geometry-wasm-minimal.log`, `gpu-clahe-geometry-wasm-combined.log`,
`gpu-clahe-geometry-baseline-build.log` and `gpu-clahe-geometry-release-build.log`.

## Completed resident and GPU-stage measurement

Release Windows native, Rust/Cargo 1.98.1, wgpu 24.0.5, Vulkan discrete backend.
No adapter model is published. The ignored paired diagnostic is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_geometry_benchmark.rs`.
Reproduce with:

```powershell
cargo test -p spatialrust-gpu --all-features --release --lib paired_previous_current_clahe_geometry -- --ignored --nocapture --test-threads=1
```

Each of 24 resolution/grid/distribution conditions has four warmups and 20
alternating pairs. Clip limit is 2. Both variants share planning, input/profile
validation, scratch acquisition, uniform/bind/encoder setup, current lookup and
application shaders, submission and a device-idle wait through the same private
CLAHE executor. The sole shader difference is frozen versus current geometry.
This is a controlled resident host-executor comparison, not two historical
public API binaries. Pipeline construction and lazy cache initialization are
warmed before recorded pairs; its cold cost is outside these timings.

The frozen helper is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/clahe_binary64_previous.wgsl`,
byte-checked against parent main 6c8cb2f's helper after line-ending normalization. Geometry/parameter shaders are
unchanged. The benchmark's second pipeline family is test-only; production
pipeline counts and laziness remain unchanged.

Completed timers include validation/planning, allocation/reuse, encoding,
submission and explicit idle waiting. They exclude runtime/pipeline setup,
image upload, profiler creation/query collection, output readback, CPU output
verification and recycling. Native GPU regressions and other agent-owned builds
finished before the measured diagnostic; no concurrent agent-owned benchmark ran.
Every warmup and measured output is byte/metadata-checked against CPU CLAHE,
with unchanged three/four-stage recording and 48/64-byte explicit query reads.

GPU stage times exclude host work, copies, gaps and waits; improved geometry
alone does not establish improved completed-call latency. The table includes
all unchanged/general-divisor and adverse conditions. Timings are local samples,
not portable speed guarantees. GPU clock/state and desktop load are uncontrolled.

Private raw logs are
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-geometry-survey.log`
and `gpu-clahe-geometry-paired.log` in the same directory. Corresponding
`*-estimates.json` and `*-table.md`, plus `summarize-clahe-geometry.py`, retain
all per-metric timing ranges. Reported ratios are medians of per-pair ratios,
not quotients of separate timing medians. Intervals use seeded 10,000-resample
paired-median bootstrap 95% estimates (seed 20261012).

All 16 conditions with two power-of-two axis divisors improved the geometry
stage: median paired ratios ranged from 1.170 to 1.396, with every reported
geometry interval above 1. General-divisor geometry ranged from 0.954 to 1.012;
the 1920x1080 and 3840x2160 conditions were slower at that stage, including a
0.954 [0.942, 0.966] mixed 3840x2160 ratio. The added divisor check and shader
code generation can affect these cases even though their arithmetic is unchanged.
At 3840x2160 / 8x8, previous geometry accounted for only 0.45%/0.97% of summed
GPU stage time in this final run. These adverse cases are retained below.

Completed resident-call paired ratios ranged from 0.948 to 1.075, and every
completed-call interval includes 1. This experiment therefore establishes a
localized geometry-stage improvement for power-of-two tile dimensions, not a
whole-call improvement or a benefit for arbitrary dimensions. Ratio greater
than 1 means the current variant was faster; ratio below 1 means it was slower.

| Size / grid | Pattern | Old geometry us / share | New geometry us | Geometry ratio [95% CI] | Old / new completed ms | Completed ratio [95% CI] |
| --- | --- | --- | ---: | --- | --- | --- |
| 1x257 / 1x257 | mixed | 7.968 / 18.40% | 6.160 | 1.236 [1.219, 1.387] | 0.485 / 0.456 | 0.948 [0.916, 1.034] |
| 1x257 / 1x257 | hot | 7.856 / 20.57% | 6.416 | 1.213 [1.193, 1.257] | 0.523 / 0.539 | 0.966 [0.907, 1.084] |
| 257x1 / 257x1 | mixed | 7.936 / 18.04% | 6.752 | 1.175 [1.172, 1.180] | 0.649 / 0.670 | 1.000 [0.866, 1.310] |
| 257x1 / 257x1 | hot | 7.936 / 20.80% | 6.768 | 1.170 [1.166, 1.175] | 0.606 / 0.532 | 1.045 [0.955, 1.229] |
| 128x128 / 8x8 | mixed | 7.920 / 23.98% | 6.080 | 1.301 [1.285, 1.311] | 0.540 / 0.532 | 1.012 [0.952, 1.091] |
| 128x128 / 8x8 | hot | 7.920 / 31.86% | 6.080 | 1.299 [1.287, 1.327] | 0.511 / 0.501 | 1.011 [0.959, 1.053] |
| 256x256 / 8x8 | mixed | 7.376 / 13.05% | 5.456 | 1.353 [1.323, 1.422] | 0.525 / 0.529 | 1.015 [0.920, 1.040] |
| 256x256 / 8x8 | hot | 7.424 / 25.55% | 5.552 | 1.340 [1.331, 1.390] | 0.486 / 0.485 | 0.987 [0.965, 1.089] |
| 256x256 / 1x1 | mixed | 19.984 / 16.31% | 14.608 | 1.367 [1.347, 1.400] | 0.861 / 0.883 | 0.978 [0.944, 1.010] |
| 256x256 / 1x1 | hot | 20.336 / 16.15% | 14.800 | 1.396 [1.364, 1.425] | 0.690 / 0.688 | 1.003 [0.981, 1.041] |
| 257x259 / 3x5 | mixed | 33.888 / 15.12% | 33.792 | 1.010 [1.001, 1.018] | 0.713 / 0.712 | 0.987 [0.969, 1.026] |
| 257x259 / 3x5 | hot | 35.056 / 28.87% | 34.576 | 1.012 [1.006, 1.019] | 0.681 / 0.710 | 0.986 [0.931, 1.042] |
| 640x480 / 8x8 | mixed | 52.576 / 6.52% | 52.096 | 1.009 [1.006, 1.017] | 1.606 / 1.580 | 1.008 [0.988, 1.024] |
| 640x480 / 8x8 | hot | 53.344 / 19.31% | 52.432 | 1.008 [1.007, 1.020] | 1.118 / 1.113 | 0.975 [0.940, 1.027] |
| 1024x1024 / 8x8 | mixed | 44.112 / 2.74% | 35.024 | 1.255 [1.251, 1.273] | 2.533 / 2.530 | 1.003 [0.977, 1.039] |
| 1024x1024 / 8x8 | hot | 44.064 / 6.32% | 35.168 | 1.254 [1.238, 1.264] | 1.541 / 1.545 | 0.984 [0.962, 1.033] |
| 1920x1080 / 8x8 | mixed | 44.016 / 1.42% | 45.328 | 0.975 [0.962, 0.980] | 4.453 / 4.392 | 0.993 [0.973, 1.051] |
| 1920x1080 / 8x8 | hot | 43.392 / 3.69% | 44.336 | 0.981 [0.971, 0.989] | 2.751 / 2.637 | 1.075 [0.999, 1.142] |
| 3840x2160 / 8x8 | mixed | 38.912 / 0.45% | 40.656 | 0.954 [0.942, 0.966] | 10.414 / 10.434 | 0.997 [0.976, 1.005] |
| 3840x2160 / 8x8 | hot | 38.944 / 0.97% | 40.624 | 0.959 [0.942, 0.972] | 5.760 / 5.867 | 0.985 [0.961, 1.010] |
| 4096x2048 / 8x8 | mixed | 28.128 / 0.47% | 23.232 | 1.209 [1.191, 1.233] | 11.957 / 11.959 | 0.993 [0.991, 1.004] |
| 4096x2048 / 8x8 | hot | 28.128 / 1.50% | 23.296 | 1.197 [1.177, 1.211] | 7.834 / 7.779 | 0.999 [0.994, 1.016] |
| 4096x2048 / 1x1 | mixed | 26.848 / 0.71% | 22.624 | 1.198 [1.184, 1.224] | 9.812 / 9.949 | 0.992 [0.974, 1.005] |
| 4096x2048 / 1x1 | hot | 27.888 / 0.92% | 22.720 | 1.203 [1.166, 1.249] | 8.963 / 9.022 | 1.011 [0.993, 1.026] |

Parent point-graph PR141 is merged at 6c8cb2f with all 114 checks successful.
This slice changes no CPU point-cloud code or GPU transfer policy.
