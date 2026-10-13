# GPU Trimmed ICP selection tree

Date: 2026-10-13. Named slice: integer rank selection in the resident stable
Trimmed ICP path. No public API or CPU placement is changed.

## Execution

The capable-device 256-lane selector keeps one lane per histogram bin. It
builds a 255-node binary sum tree in eight synchronized levels, then lane zero
finds the selected byte in eight decisions. Subtracting a left subtree only
when descending right leaves the same within-bin rank as the previous ascending
linear walk. The root supplies the gated count on the first distance byte.
Every internal sum is bounded by the total u32 match count.

The tree reuses the declared producer histogram workgroup array. The wide selector
accesses both arrays (2 KiB rather than the prior selector's 1 KiB), and adds eight
workgroup barriers to the existing gathering barrier. The original planning
requirement of at least 2 KiB workgroup storage still applies. Global scratch,
uniforms, submissions, digit passes and explicit transfers do not change.
This is a local workgroup scheduling change, not a CPU/GPU placement policy.

The same gate requires at least 64 producer groups (N>=4033), 256 invocations
per workgroup and 256 x lanes. Small or lower-limit devices retain the
64-lane linear pipeline and avoid wide-cache compilation. Histogram production
remains 64 lanes. Existing exact binary64 fraction emulation, distance/source-index tie ordering, both moment
precisions, fraction-one delegation and owned 176-byte moment collection remain.
CPU target-tree construction, Horn solving, composition and stopping remain.

## Correctness

The complete PR #182 shader is frozen from
`a55fcce69c9f671c3f0d8b78d25be18e18e72d97`. The native comparison compiles
its actual histogram and optional 256-lane linear selection pipelines. It retains
the 64-lane route below the original threshold. PR #181's actual 64-lane shader
remains a second frozen reference. The historical lane benchmark now explicitly
compares frozen PR #181 and PR #182, so this optimization cannot change its
meaning.

- 210 full-selection conditions cover fourteen sizes including dispatch
  thresholds, 262,145 points, ties/missing/signed-zero/subnormal distances and
  five fractions. Every word agrees with both frozen GPU routes; selected pairs
  agree with an independent comparison sort.
- 896 individual-digit conditions (7,168 words) cover eight bytes, seven
  histogram group counts, four distributions and four fractions. Both frozen
  routes and independent u64 counts/prefix/rank arithmetic agree through maximum
  u32 mass and zero retained counts.
- 8,192 raw endpoint conditions (65,536 words) cover every bin, all eight bytes,
  one and maximum u32 mass, and both rank endpoints. All words match the frozen
  wide GPU route and an independent one-hot reference.
- Capability, lazy initialization, concurrent cold-cache construction, dirty
  scratch reuse, owned output, fraction arithmetic and independent f64 moment
  references remain checked.

GPU all-feature/all-target Clippy passed. The all-feature point/image-kernel
regression completed with 144 passed, zero failed, and 15 explicitly ignored
measurement fixtures. The native trim family separately passed 14 correctness
tests, with two measurement fixtures ignored.

Five release registration integration tests were explicitly run with ignored
GPU cases enabled. The 486 small/scale/profile/fraction/stopping conditions
preserve the original CPU pose threshold (maximum scale-normalized difference
1.9073486e-6). All 27 large 4097/65537/262145 trajectories preserve the unchanged
3e-5 limit (maximum 7.6293945e-6). Fraction-one/mixed concurrent precision,
rotated initial poses, validation and insufficient-retained cases also pass.
Feature-free GPU, GPU-image WASM and the minimal register-icp-gpu facade compile
checks pass. WASM was compile checked; browser execution was not measured.

Workspace formatting also passes.

## Measurement conditions

Two isolated release surveys completed sequentially in 6.22 and 6.02 seconds.
No overlapping local build/test/benchmark was detected during either accepted
run. Statistical analysis followed all surveys. The backend is Vulkan and the
device class is DiscreteGpu. Relevant reported limits are 1024 invocations per
workgroup, 1024 x lanes and 49152 bytes workgroup storage. Both runtimes assert
that the capable-device tree route is available. No adapter model is recorded.
Separate default bounded pools retain buffers independently. Pipeline
construction, fixture generation, warmed references and post-timer comparisons
are excluded. Three completed warmups and twenty paired rounds rotate old full,
tree full, old resident and tree resident stages.

Selection fixtures have 257/4097/16385/65537/262145/1000003 precomputed matches.
Profile 0 has seeded mixed distances and missing/out-of-gate pairs; profile 1
uses equal 0.25 distances. Bound is 0.5 and fraction is 0.8. Full selector calls
include an 8*N match upload, controls, owned output/global scratch, all digit
passes, completed 32-byte selection collection and dropping owned buffers.
Resident calls use uploaded matches and time controls/acquisition/submission/
wait_idle, then collect and check 32 bytes outside timing. These private selector
probes exclude transforms, nearest search and the CPU pose solver.

Public moment fixtures use 257/4097/65537/262145 source points and a SINGLE target
at the origin, with deterministic binary-exact residuals. Profile 0 is fully
gated; profile 1 has periodic outliers. Full calls construct that target tree,
upload source/target/tree, compute identity-pose default DoubleSingle trimmed
moments, collect 176 bytes and drop owned inputs/output. Prepared calls reuse
uploaded input and time the public resident moment call plus wait_idle, then
collect and compare outside timing. Every decoded moment/status/count/fitness/
transfer field agrees with the frozen wide selector result; collection is
asserted to remain 176 bytes. This reduction-focused synthetic geometry is not a
full multi-target alignment/iteration benchmark or a CPU-vs-GPU comparison.

The historical frozen PR #181/#182 paired selector/moment survey was explicitly
re-run alone and passed in 6.06 seconds. Its results do not enter the new table.

## Paired results

Times are medians in milliseconds. Ratios are geometric means of twenty paired
old/tree times; values above one favor the tree. Intervals use 10,000 paired
log-ratio bootstrap draws, sorted entries 249/9749, with seed
20261013 + 2*record_index + comparison_index. These unadjusted intervals describe
within-survey variation, not repeat-run or hardware-wide bounds. All adverse and
tail samples are retained.


| Survey | Work/count | Profile | Old full ms | Tree full ms | Old resident ms | Tree resident ms | Old/tree full (95%) | Old/tree resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | selection/257 | 0 | 1.111 | 1.092 | 0.451 | 0.449 | 0.994x (0.944-1.047) | 0.999x (0.917-1.091) |
| 1 | selection/257 | 1 | 1.085 | 1.099 | 0.459 | 0.455 | 0.984x (0.959-1.009) | 1.000x (0.934-1.072) |
| 1 | selection/4,097 | 0 | 1.018 | 0.995 | 0.455 | 0.442 | 1.032x (1.007-1.059) | 1.016x (0.944-1.093) |
| 1 | selection/4,097 | 1 | 1.089 | 1.093 | 0.484 | 0.460 | 0.956x (0.908-1.003) | 1.036x (0.963-1.114) |
| 1 | selection/16,385 | 0 | 1.182 | 1.166 | 0.578 | 0.570 | 1.031x (1.009-1.055) | 1.021x (0.955-1.091) |
| 1 | selection/16,385 | 1 | 1.070 | 1.056 | 0.558 | 0.550 | 1.011x (0.987-1.036) | 1.019x (0.937-1.109) |
| 1 | selection/65,537 | 0 | 1.118 | 1.127 | 0.613 | 0.621 | 0.982x (0.955-1.008) | 1.037x (0.951-1.133) |
| 1 | selection/65,537 | 1 | 1.126 | 1.140 | 0.657 | 0.630 | 0.964x (0.917-1.006) | 1.051x (0.976-1.138) |
| 1 | selection/262,145 | 0 | 1.785 | 1.777 | 0.721 | 0.711 | 0.981x (0.952-1.014) | 1.011x (0.953-1.074) |
| 1 | selection/262,145 | 1 | 1.941 | 1.907 | 0.742 | 0.709 | 1.024x (0.975-1.066) | 1.010x (0.948-1.073) |
| 1 | selection/1,000,003 | 0 | 2.872 | 2.900 | 1.140 | 1.115 | 0.993x (0.949-1.039) | 0.991x (0.938-1.049) |
| 1 | selection/1,000,003 | 1 | 2.772 | 2.802 | 1.033 | 0.996 | 0.975x (0.934-1.009) | 1.064x (0.991-1.144) |
| 1 | moments/257 | 0 | 1.277 | 1.369 | 0.666 | 0.648 | 0.952x (0.896-1.010) | 1.012x (0.934-1.102) |
| 1 | moments/257 | 1 | 1.189 | 1.186 | 0.633 | 0.620 | 1.002x (0.973-1.035) | 1.028x (0.947-1.118) |
| 1 | moments/4,097 | 0 | 1.516 | 1.386 | 0.903 | 0.877 | 1.035x (0.983-1.092) | 1.026x (0.989-1.065) |
| 1 | moments/4,097 | 1 | 1.187 | 1.214 | 0.869 | 0.859 | 0.979x (0.965-0.993) | 1.015x (0.970-1.060) |
| 1 | moments/65,537 | 0 | 4.382 | 3.288 | 2.161 | 2.037 | 1.301x (1.232-1.365) | 1.045x (1.012-1.083) |
| 1 | moments/65,537 | 1 | 4.078 | 3.000 | 1.956 | 1.884 | 1.326x (1.231-1.393) | 1.075x (0.998-1.195) |
| 1 | moments/262,145 | 0 | 6.370 | 6.280 | 2.475 | 2.446 | 1.010x (0.985-1.036) | 1.016x (0.994-1.039) |
| 1 | moments/262,145 | 1 | 6.294 | 6.349 | 2.545 | 2.435 | 0.994x (0.965-1.026) | 1.044x (1.013-1.078) |
| 2 | selection/257 | 0 | 1.064 | 1.063 | 0.457 | 0.457 | 0.986x (0.947-1.022) | 1.004x (0.934-1.077) |
| 2 | selection/257 | 1 | 1.052 | 1.059 | 0.449 | 0.448 | 1.003x (0.973-1.035) | 0.997x (0.926-1.074) |
| 2 | selection/4,097 | 0 | 0.999 | 1.011 | 0.452 | 0.447 | 0.996x (0.970-1.023) | 1.024x (0.953-1.101) |
| 2 | selection/4,097 | 1 | 1.005 | 0.987 | 0.461 | 0.445 | 0.987x (0.961-1.010) | 1.041x (0.970-1.116) |
| 2 | selection/16,385 | 0 | 1.190 | 1.196 | 0.569 | 0.561 | 0.982x (0.945-1.023) | 1.016x (0.943-1.092) |
| 2 | selection/16,385 | 1 | 1.177 | 1.180 | 0.576 | 0.558 | 0.978x (0.930-1.020) | 1.040x (0.975-1.112) |
| 2 | selection/65,537 | 0 | 1.110 | 1.109 | 0.640 | 0.619 | 1.014x (0.987-1.042) | 1.036x (0.957-1.126) |
| 2 | selection/65,537 | 1 | 1.105 | 1.081 | 0.656 | 0.613 | 1.025x (1.003-1.047) | 1.060x (0.970-1.156) |
| 2 | selection/262,145 | 0 | 1.747 | 1.712 | 0.683 | 0.702 | 1.031x (1.011-1.051) | 1.018x (0.966-1.074) |
| 2 | selection/262,145 | 1 | 1.799 | 1.799 | 0.706 | 0.702 | 1.011x (0.985-1.039) | 1.021x (0.951-1.090) |
| 2 | selection/1,000,003 | 0 | 2.865 | 2.911 | 1.053 | 1.032 | 1.037x (0.972-1.128) | 1.041x (0.987-1.098) |
| 2 | selection/1,000,003 | 1 | 2.965 | 2.890 | 1.139 | 1.051 | 1.021x (0.989-1.052) | 1.018x (0.958-1.082) |
| 2 | moments/257 | 0 | 1.265 | 1.291 | 0.672 | 0.685 | 0.992x (0.951-1.034) | 0.976x (0.928-1.025) |
| 2 | moments/257 | 1 | 1.230 | 1.227 | 0.653 | 0.667 | 1.022x (0.970-1.077) | 0.972x (0.898-1.050) |
| 2 | moments/4,097 | 0 | 1.209 | 1.206 | 0.847 | 0.858 | 1.020x (0.985-1.061) | 1.005x (0.960-1.051) |
| 2 | moments/4,097 | 1 | 1.224 | 1.225 | 0.860 | 0.856 | 1.022x (0.987-1.057) | 1.012x (0.976-1.050) |
| 2 | moments/65,537 | 0 | 4.240 | 3.277 | 2.161 | 2.018 | 1.278x (1.243-1.314) | 1.052x (1.025-1.082) |
| 2 | moments/65,537 | 1 | 3.978 | 2.927 | 1.938 | 1.853 | 1.328x (1.281-1.376) | 1.041x (1.003-1.079) |
| 2 | moments/262,145 | 0 | 6.172 | 6.379 | 2.455 | 2.460 | 0.969x (0.929-1.008) | 1.017x (0.994-1.042) |
| 2 | moments/262,145 | 1 | 6.204 | 6.290 | 2.504 | 2.486 | 0.993x (0.967-1.020) | 1.009x (0.987-1.033) |

Every standalone resident-selector interval includes one; no statistically clear
standalone-selector improvement is established. Estimates for optimized counts
range from 0.991x to 1.064x. Small 257-point calls retain the same linear route
and still show host/runtime variation. Full selector results are mixed: for
example, mixed 16385 points changes from 1.031x to 0.982x between surveys.

Fully gated 65537-point prepared moments give 1.045x (1.012-1.083) and 1.052x
(1.025-1.082), establishing a repeated modest improvement in that synthetic
case. Outlier-profile prepared results are 1.075x (0.998-1.195) and 1.041x
(1.003-1.079); the first is inconclusive. Prepared 262145-point estimates are
1.009-1.044x, but only one of four intervals excludes one. Neither smaller
prepared moments nor larger clouds establish a general improvement.

Complete 65537-point synthetic moment calls give 1.278-1.328x. Those completed
timings include host/device resource and completion behavior; their much larger
ratios are not attributed to selector shader throughput. Complete 262145-point
results are 0.969-1.010x with intervals including one. The 4097-point outlier
full call regresses in survey 1 (0.979x, 0.965-0.993), then is inconclusive in
survey 2 (1.022x, 0.987-1.057). These limitations and the extra shared-memory/
barrier cost remain explicit. Adoption is supported by the repeated fully gated
65537-point prepared-moment result; no blanket selector, complete registration
or general-device acceleration is claimed.

Private raw/derived artifacts remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selection-tree-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selection-tree-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selection-tree-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-icp-trim-selection-tree.py`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/run-gpu-icp-trim-selection-tree-survey.ps1`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trim-selector-historical-recheck-benchmark-1.log`

Source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/icp_trim.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_selector_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_tree_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_tree_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/icp_trim_tree_previous.wgsl`

From `C:/Users/rsasa/Workspace/SpatialRust`, compile the release test binary with
--no-run first, then run this exact test alone twice sequentially:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib kernels::icp_reduction::trim::tests::selector_tests::tree_tests::benchmark::completed_tree_selector_and_moment_calls -- --include-ignored --exact --nocapture --test-threads=1
```
