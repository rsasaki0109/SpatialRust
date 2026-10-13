# Wider GPU Trimmed ICP histogram selection

Date: 2026-10-13. Named slice: one-bin-per-lane integer gathering in the existing
resident stable Trimmed ICP selector. No CPU placement, pose solver, transform,
nearest search, precision choice or public API is changed.

## Execution

The histogram producer still launches min(ceil(N/64),256) 64-lane workgroups.
Each produces 256 exact bins for the active distance/source-index prefix. The
old selector launches one 64-lane group; every lane serially gathers four bins
across all producer groups. The optional wide selector launches one 256-lane
group, assigning one bin to each lane. Each bin sums the same group sequence;
after the same barrier, lane zero makes the same ascending-bin rank decision.
No cross-workgroup convergence, atomic floating sums or reduction reordering is
introduced. The numeric representation and all eight Selection words agree.

Host selection requires at least 64 producer groups (N>=4033 with unchanged
planning), max_compute_invocations_per_workgroup>=256 and
max_compute_workgroup_size_x>=256. Small or device-limited calls retain the
64-lane pipeline and avoid wide-cache compilation. The original limits still
reject unsupported input before production work. The optimized selector is a
separate OnceLock in the existing per-runtime trim cache; concurrent cold callers
compile it once. This is internal kernel scheduling, not automatic CPU/GPU placement.

A WGSL override changes only selector workgroup size and bin stride. The
histogram kernel remains 64 lanes. The same 256-bin workgroup totals, barrier,
32-byte controls, <=256 KiB histogram lease, 32-byte selection lease and
necessary distance/index digit passes remain. No new device buffer, readback,
payload upload or submission is added. Fraction one still delegates to canonical
untrimmed moments; both CompensatedF32 and default DoubleSingle retain their
existing math. Explicit moment collection remains 176 bytes per evaluation.

## Correctness

The comparison fixture freezes the complete selector/histogram shader from
`dbf35de22e583f9f08a962c5fe9439992e54e07a` (merged PR #181). Native baseline
caches compile its actual 64-lane histogram and selector and use the same control,
scratch/submission and moment boundaries. It is not a CPU-simulated baseline.

- The existing independent comparison-sort test now covers 14 sizes, including
  4032/4033 and 4096/4097 scheduling boundaries and 262,145 points, three tied/
  missing/signed-zero/subnormal distance profiles and five trim fractions:
  210 conditions. Every Selection word matches the frozen GPU shader; chosen
  pairs match the independent distance/index sort.
- 896 individual-digit conditions cover all eight digit positions, seven partial
  counts (1/2/63/64/65/255/256), four distributions, four fractions and maximum
  total u32 mass. All 7,168 output words agree with frozen GPU and independent
  u64 bin/prefix/rank arithmetic; tiny fractions retain zero correctly.
- Workgroup-size/invocation boundary checks enforce the lower-limit fallback.
  Small calls leave the optimized pipeline uninitialized. Twelve concurrently
  submitted owned results retain exact selections across three dirty-scratch
  waves and cold cache initialization.
- Existing binary64 fraction emulation, all-gated fitness, independent f64
  centered moments, both precision choices, invalid data/status/ownership and
  canonical fraction-one tests pass. Precision math retains 2,076 independent
  comparisons and the fixed-pose 262,145-point references.

GPU all-feature point/image-kernel regression: 143 passed, 14 explicitly ignored
measurement fixtures. All-feature/all-target GPU Clippy passed. Five release
registration integration tests were explicitly run with ignored GPU cases enabled:
486 small/scale/profile/fraction/stopping conditions preserve the original CPU pose
threshold (maximum scale-normalized difference 1.9073486e-6), and 27 large
4097/65537/262145 trajectories preserve the unchanged 3e-5 limit (maximum
7.6293945e-6). Fraction-one/mixed concurrent precision, rotated initial poses,
validation and insufficient-retained cases also pass. Feature-free GPU, GPU-image
WASM, minimal register-icp-gpu facade and workspace formatting checks pass.
WASM was compile checked; browser execution was not measured.

## Measurement conditions

Two isolated release surveys completed sequentially in 6.53 and 4.89 seconds,
with no overlapping local build/test/benchmark/statistical analysis. Backend is
Vulkan and device class DiscreteGpu; no local adapter model is recorded.
Runtimes use separate default bounded pools, so each route keeps its own retained
buffers. Pipeline construction, fixture generation, warmed expected outputs and
post-timer comparisons are excluded. Three completed warmups and 20 paired rounds
rotate through old full, wide full, old resident and wide resident.

Selection fixtures use 257/4097/16385/65537/262145/1000003 precomputed matches.
Profile 0 has seeded mixed distances with missing/out-of-gate matches; profile 1
uses equal 0.25 distances. Bound is 0.5 and trim fraction 0.8. Full selection
calls include an 8*N match upload, controls, output/scratch acquisition, all digit
passes, completed 32-byte selection readback and dropping caller-owned buffers.
Resident calls start from uploaded matches, include controls/acquisition/
submission/wait_idle, then read/check the 32 bytes outside timing. These private
selector probes do not include transform, nearest search or the CPU pose solver.

Moment fixtures use 257/4097/65537/262145 source points and a SINGLE target at the
origin, with deterministic binary-exact residuals. Profile 0 is fully gated;
profile 1 has periodic outliers. Full calls construct that target tree and upload
original source/target/tree, compute identity-pose default DoubleSingle trimmed
moments, collect 176 bytes and drop owned inputs/output. Prepared calls reuse
uploaded input, time the public resident moment call plus wait_idle, then collect
and compare outside timing. Every decoded moment/status/count/fitness/transfer
field matches the frozen-selector result exactly. These are complete moment
calls for a reduction-focused synthetic geometry, not a measurement of a full
multi-target alignment/iteration loop or CPU-vs-GPU placement.

## Paired results

Times are medians in milliseconds. Ratios are geometric means of twenty paired
old/wide times; values above one favor wide. Intervals use 10,000 paired log-ratio
bootstrap draws, sorted entries 249/9749, with seed
20261013 + 2*record_index + comparison_index. They describe within-survey
variation, not repeat-run or hardware-wide bounds. Every adverse/tail sample is
retained.

| Survey | Work/count | Profile | Old full ms | Wide full ms | Old resident ms | Wide resident ms | Old/wide full (95%) | Old/wide resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | selection/257 | 0 | 1.042 | 1.055 | 0.461 | 0.460 | 0.95x (0.91-0.98) | 0.99x (0.92-1.05) |
| 1 | selection/257 | 1 | 1.039 | 1.043 | 0.446 | 0.443 | 1.00x (0.98-1.02) | 1.00x (0.93-1.08) |
| 1 | selection/4,097 | 0 | 1.062 | 1.059 | 0.524 | 0.453 | 0.99x (0.97-1.01) | 1.18x (1.10-1.27) |
| 1 | selection/4,097 | 1 | 1.063 | 1.029 | 0.552 | 0.454 | 1.01x (0.96-1.04) | 1.19x (1.11-1.29) |
| 1 | selection/16,385 | 0 | 1.118 | 1.076 | 0.780 | 0.552 | 1.02x (1.00-1.04) | 1.48x (1.39-1.57) |
| 1 | selection/16,385 | 1 | 1.101 | 1.096 | 0.779 | 0.550 | 1.01x (0.98-1.03) | 1.46x (1.38-1.55) |
| 1 | selection/65,537 | 0 | 1.242 | 1.167 | 0.891 | 0.629 | 1.07x (1.01-1.13) | 1.45x (1.37-1.54) |
| 1 | selection/65,537 | 1 | 1.275 | 1.144 | 0.886 | 0.649 | 1.09x (1.06-1.11) | 1.42x (1.34-1.52) |
| 1 | selection/262,145 | 0 | 2.152 | 1.893 | 0.964 | 0.710 | 1.12x (1.08-1.15) | 1.40x (1.34-1.47) |
| 1 | selection/262,145 | 1 | 2.156 | 1.835 | 0.946 | 0.686 | 1.15x (1.09-1.19) | 1.36x (1.29-1.43) |
| 1 | selection/1,000,003 | 0 | 3.020 | 2.713 | 1.180 | 0.903 | 1.13x (1.09-1.17) | 1.28x (1.22-1.33) |
| 1 | selection/1,000,003 | 1 | 2.943 | 2.721 | 1.198 | 0.885 | 1.07x (1.00-1.12) | 1.29x (1.23-1.36) |
| 1 | moments/257 | 0 | 1.223 | 1.182 | 0.582 | 0.593 | 1.00x (0.94-1.05) | 0.98x (0.91-1.06) |
| 1 | moments/257 | 1 | 1.081 | 1.092 | 0.546 | 0.540 | 0.99x (0.97-1.01) | 0.98x (0.93-1.04) |
| 1 | moments/4,097 | 0 | 1.188 | 1.132 | 0.846 | 0.786 | 1.03x (1.01-1.06) | 1.09x (1.05-1.13) |
| 1 | moments/4,097 | 1 | 1.211 | 1.197 | 0.849 | 0.802 | 1.03x (0.98-1.10) | 1.04x (1.00-1.09) |
| 1 | moments/65,537 | 0 | 4.315 | 3.067 | 2.183 | 1.929 | 1.45x (1.37-1.53) | 1.10x (1.03-1.17) |
| 1 | moments/65,537 | 1 | 4.304 | 2.987 | 2.040 | 1.852 | 1.45x (1.40-1.50) | 1.12x (1.09-1.16) |
| 1 | moments/262,145 | 0 | 6.410 | 6.337 | 2.775 | 2.482 | 1.00x (0.98-1.02) | 1.11x (1.09-1.14) |
| 1 | moments/262,145 | 1 | 6.390 | 6.312 | 2.822 | 2.458 | 1.02x (0.99-1.04) | 1.16x (1.12-1.21) |
| 2 | selection/257 | 0 | 1.052 | 1.049 | 0.406 | 0.406 | 0.98x (0.95-1.01) | 0.98x (0.92-1.04) |
| 2 | selection/257 | 1 | 1.052 | 1.041 | 0.400 | 0.400 | 1.00x (0.98-1.02) | 1.01x (0.95-1.08) |
| 2 | selection/4,097 | 0 | 1.054 | 1.056 | 0.451 | 0.405 | 0.99x (0.96-1.01) | 1.14x (1.07-1.21) |
| 2 | selection/4,097 | 1 | 1.020 | 1.047 | 0.458 | 0.407 | 0.95x (0.92-0.97) | 1.13x (1.07-1.20) |
| 2 | selection/16,385 | 0 | 1.134 | 1.094 | 0.707 | 0.527 | 1.03x (1.00-1.06) | 1.43x (1.35-1.53) |
| 2 | selection/16,385 | 1 | 1.082 | 1.072 | 0.759 | 0.556 | 1.00x (0.98-1.02) | 1.42x (1.33-1.53) |
| 2 | selection/65,537 | 0 | 1.207 | 1.135 | 0.879 | 0.626 | 1.02x (0.96-1.06) | 1.45x (1.37-1.54) |
| 2 | selection/65,537 | 1 | 1.209 | 1.119 | 0.876 | 0.639 | 1.06x (1.03-1.08) | 1.43x (1.34-1.52) |
| 2 | selection/262,145 | 0 | 2.145 | 1.855 | 0.950 | 0.700 | 1.16x (1.14-1.18) | 1.39x (1.33-1.46) |
| 2 | selection/262,145 | 1 | 2.121 | 1.869 | 0.905 | 0.686 | 1.11x (1.06-1.16) | 1.38x (1.33-1.44) |
| 2 | selection/1,000,003 | 0 | 2.934 | 2.740 | 1.154 | 0.884 | 1.05x (0.97-1.12) | 1.29x (1.25-1.34) |
| 2 | selection/1,000,003 | 1 | 2.886 | 2.609 | 1.100 | 0.883 | 1.05x (0.98-1.11) | 1.32x (1.26-1.42) |
| 2 | moments/257 | 0 | 1.079 | 1.063 | 0.539 | 0.546 | 1.00x (0.97-1.04) | 0.98x (0.93-1.04) |
| 2 | moments/257 | 1 | 1.056 | 1.029 | 0.533 | 0.533 | 1.02x (0.99-1.05) | 1.00x (0.95-1.07) |
| 2 | moments/4,097 | 0 | 1.172 | 1.129 | 0.836 | 0.772 | 1.03x (0.99-1.06) | 1.09x (1.05-1.14) |
| 2 | moments/4,097 | 1 | 1.173 | 1.124 | 0.844 | 0.735 | 1.04x (1.00-1.08) | 1.10x (1.05-1.15) |
| 2 | moments/65,537 | 0 | 4.265 | 2.993 | 2.090 | 1.900 | 1.37x (1.22-1.47) | 1.14x (1.09-1.19) |
| 2 | moments/65,537 | 1 | 4.252 | 2.921 | 1.987 | 1.868 | 1.45x (1.41-1.48) | 1.10x (1.06-1.12) |
| 2 | moments/262,145 | 0 | 6.310 | 6.344 | 2.803 | 2.482 | 0.99x (0.95-1.03) | 1.13x (1.11-1.16) |
| 2 | moments/262,145 | 1 | 6.439 | 6.297 | 2.782 | 2.474 | 1.04x (1.00-1.08) | 1.12x (1.09-1.16) |

For every >=4097 selection condition, resident ratios are 1.13-1.48x in both
surveys, with intervals above one. At >=16385 they are 1.28-1.48x. Upload/readback
comparisons vary: the 4097 tied case in survey 2 regresses (0.95x, interval below
one), and several larger intervals include one. At 262145 full selection gives
1.11-1.16x; at one million it gives 1.05-1.13x with some inconclusive intervals.
The 257 route executes the same 64-lane path yet has a 0.95x adverse full ratio
in survey 1; host/runtime/scheduling variation remains visible.

Prepared moment ratios for >=4097 are 1.04-1.16x, with one survey-1 gated 4097
interval at the one boundary. At 65537/262145, all prepared intervals exclude one
and ratios are 1.10-1.16x. Complete 65537 moment calls give 1.37-1.45x in this
synthetic setup, but 262145 complete ratios are 0.99-1.04x and include one; no
consistent full-call improvement is established there. These completed timings
include host/device resource and completion behavior; the much larger 65537 full
ratio than prepared ratio is not attributed to selector shader throughput.
No full registration, general-device or blanket end-to-end speedup is claimed.

Private raw/derived artifacts remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selector-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selector-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selector-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-icp-trim-selector.py`

Source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/icp_trim.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_selector_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_selector_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/icp_trim_selector_previous.wgsl`

From `C:/Users/rsasa/Workspace/SpatialRust`, compile the release test binary with
--no-run first, then run this exact test alone twice sequentially:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib kernels::icp_reduction::trim::tests::selector_tests::benchmark::completed_selector_and_moment_calls -- --include-ignored --exact --nocapture --test-threads=1
```
