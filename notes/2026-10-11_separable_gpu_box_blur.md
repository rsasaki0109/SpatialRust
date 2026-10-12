# Exact separable GPU box blur

Workspace: `C:/Users/rsasa/Workspace/SpatialRust`.
Parent main: `d26ed930b399da202cb9b2528229f47f7df13b5e` (PR #136 merged).
Slice: retain the previous integer gray-box mean while reducing large-kernel
device work and validating arithmetic/dispatch limits. Public signatures and
CPU/GPU placement remain unchanged; no dependency is added.

## Implementation and arithmetic

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur.rs`
plans the operation before output/scratch allocation. Positive odd dimensions
are required. Pixel count fits u32, coordinates and the old small-kernel signed
offset arithmetic fit i32, and texture/compute/uniform limits are supported.
With kernel area A, `255*A + floor(A/2) <= u32::MAX` bounds every intermediate,
edge-weight product, accumulated sum and final rounding addition. Checked byte
size and storage binding/buffer limits bound separable scratch. Unsupported
sizes return `InvalidArgument` rather than overflow or invalid dispatch.

Small workloads retain the previous full-box sampling and one final integer
rounding, with bounded grid strides added. Boxes with area at least 81 and at
least 65,536 total pixel/sample visits use two ordered compute passes in one
submission. Selection compares pixel count against ceil(65,536/A), avoiding
multiplication overflow even for unvalidated profiler inputs:

1. Write an unrounded u32 horizontal sum for every pixel.
2. Sum those rows vertically and write `(sum + floor(A/2)) / A` once to RGBA8.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_horizontal.wgsl`
and
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_vertical.wgsl`
scan valid coordinates only. Replicate borders add the exact number of omitted
edge samples as integer weights. A one-pixel axis receives both edge weights.
Constant-zero borders add no out-of-image mass and retain the full box divisor.
Interior sample work falls from kw*kh to kw+kh; oversized kernels scan at most
the image width plus height, with constant additional edge work.

Every invocation owns disjoint pixel residues. At most 4,096 groups (or the
device's lower limit) dispatch 256 lanes. A subtraction guard terminates strides
before u32 wrap. The serial/horizontal/vertical layouts and pipelines compile
independently through private per-runtime OnceLocks, including concurrent cold
calls. The existing runtime image-pipeline family count remains one for blur.

## Ownership, receipts and profiling

The horizontal array uses a private submitted scratch lease and exactly 4*N
bytes for N pixels. Every word is overwritten before the vertical pass reads it;
no reset pass, atomics, floating reassociation or host synchronization is added.
Queue ordering permits reuse after submission, while overlapping encoders hold
exclusive leases. Abandoned encoders retain the existing no-recycle rule.
Caller-owned input and output images remain outside the scratch lease.

Image H2D/D2H behavior is unchanged. The existing 32-byte parameter uniform is
still initialized per call and excluded from the image payload receipt as before.
Small blur workloads record 4*N device-written bytes; separable boxes record 8*N:
`box_blur_horizontal_gpu` followed by `box_blur_gpu`. Output metadata and unused
RGBA components are preserved. Native readback remains an explicit caller action.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/vision_chain.rs`
reserves six profiler stages for small blur workloads and seven for separable
kernels before executing the chain. Insufficient capacity preserves prior
recording. Collection explicitly reads 96 or 112 timestamp bytes, respectively,
separately from image payload accounting. Default 3x3 chains retain six stages.

## Correctness evidence

The exact previous shader is frozen at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/box_blur_previous.wgsl`.
It is verified against the parent Rust literal after newline normalization.
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_tests.rs`
compares both shaders with an independent u64 full-box sampling oracle:
83,592 pixels / 334,368 RGBA component bytes per variant, across seven image
shapes, nine kernel shapes, two distributions and both borders (252 fixtures).
This reference contract is the previous GPU integer mean.

The paired correctness test forces the separable path at area 81 and above
regardless of the work cutoff, forces one workgroup, fills scratch with dirty bytes,
and exercises several stride rounds, 1x1 and single-axis images, non-divisible
dimensions, 7x7/9x9 cutoff cases, rectangular kernels and kernels larger than
the image. Full RGBA comparisons include zero unused channels.
Additional cases cover strided ROI and non-default metadata, adjacent rounding
sums 40/41 and 121/122 for an 81-sample divisor, 4099x4099 edge multiplicities
near the maximum safe sum, exact u32::MAX pixel planning, the 809/810-pixel
9x9 work boundary, ten insufficient device limits, byte-size/coordinate
overflow, invalid kernels/channels/foreign runtimes before allocation, retained
outputs through concurrent calls/runtime moves, and warmed scratch reuse.
The chain test verifies seven ordered timing samples, exact profiled/plain
tensor and receipt equality, and capacity failure before any recording.

## Diagnostic scope and reproduction

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/box_blur_benchmark.rs`
is ignored by default. It compares the frozen shader with production encoding
on resident inputs, separate outputs and warmed pipelines. Both variants use
the same runtime and uniform. There are three warmups and 20 alternating pairs
for each of 26 image/kernel/border fixtures. Final complete images are compared
byte-for-byte after timing.

GPU durations sum one previous pass versus one or two production passes and
exclude uploads, allocation, compilation, CPU command setup, readback and
inter-pass gaps. The separate completion interval starts after encoder finish
and covers submit+wait, including inter-pass transitions. Query collection is
outside both timers and reads 32 or 48 bytes per pair. These measurements
describe resident blur execution, excluding full public-API or Vision-chain
latency. The area/work thresholds are internal specializations, not a new
CPU/GPU dispatch policy or a portable optimal cutoff.

Environment: local GPU, Vulkan backend, Windows, Rust/Cargo 1.98.1, wgpu 24.0.5,
release/thin LTO. Other user computations remain running. Agent GPU correctness
tests finish before diagnostics; timings can still vary with desktop load and
device clocks. The diagnostic prints backend/device type without a model name.

```powershell
$env:PATH = 'C:\Users\rsasa\.cargo\bin;' + $env:PATH
cargo test -p spatialrust-gpu --all-features -- --test-threads=1
cargo test -p spatialrust-gpu --release --features gpu-image paired_previous_current_blur --lib -- --ignored --nocapture --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-image --target wasm32-unknown-unknown --lib -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown --lib -- -D warnings
cargo fmt --all --check
python C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-gpu-blur.py
```

WebAssembly configurations are compile-checked; browser shader execution remains
unverified. Diagnostic raw samples, estimates and validation logs are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` and remain local.

## Final validation

The final full native all-feature run passed 122 tests (97 unit, four CDF,
five CLAHE, five equalization, three histogram and eight stage-timing tests),
with five explicit diagnostics ignored and no failures. The exact new
adapter-free planner command used by CI also passed under minimal `gpu-image`.
Native all-feature/all-target and default Clippy with warnings denied passed;
minimal and combined image/AoSoA WebAssembly Clippy passed, including after the
work-selection adjustment. Formatting, whitespace and frozen-shader checks pass.
The existing GPU-image CI job compiles all image tests and now executes the pure
blur planner without requiring an adapter; required job names are unchanged.

Logs under the absolute workspace `target/automation/` directory:
`gpu-blur-final-regressions.log`, `gpu-blur-final-minimal.log` (six native blur
checks before the workload selection adjustment), `gpu-blur-ci-pure-plan.log`,
`gpu-blur-final-clippy.log`, `gpu-blur-default-clippy.log`,
`gpu-blur-wasm-minimal.log`, and `gpu-blur-wasm-combined.log`.

## Measurement chronology and final results

The first area-only policy used two passes for every area of at least 81. It
improved the 17x19/9x9 GPU stage sum by 1.50/1.67x but its paired completion
ratios were 0.840/0.882 for replicate/zero borders. Those results are preserved
separately in `gpu-blur-area-only-paired.log`,
`gpu-blur-area-only-estimates.json` and `gpu-blur-area-only-table.md`.
The final policy additionally requires at least 65,536 total pixel/sample visits,
retaining a single pass for this small case while still accelerating oversized
kernels. This is a conservative internal threshold, not a claim of an optimal
break-even point across devices. The pure planner checks the 809/810-pixel
boundary for a 9x9 box and safe selection on maximum unvalidated inputs.

The final sweep passed all 26 full-image comparisons with 20 measured pairs per
fixture. GPU/compiler checks completed before it; other user workloads continued.
Final 4K/31x31 GPU paired ratios are 15.793/17.476 and submit+wait paired ratios
15.001/16.489 for replicate/zero borders. Tiny 17x19/9x9 cases now execute one
pass; their GPU ratio intervals include parity. Completion times remain noisy,
and small cases include observed regressions, so no universal speedup is claimed.

`C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-gpu-blur.py`
checks 26 complete fixtures and all four 20-value arrays, saves raw samples,
min/max ranges and paired ratios, and bootstraps the paired median 10,000 times
with seed 20261011. Table ratios are medians of within-round previous/current
ratios; they need not equal the ratio of the displayed medians. The 95% intervals
describe these local alternating samples. Ratios greater than one favor the
current shader. The final data are `gpu-blur-paired.log`,
`gpu-blur-estimates.json` and `gpu-blur-table.md`, separate from the first probe.

| Image | Kernel | Border | Previous/current GPU median (us) | Paired GPU ratio [95% CI] | Previous/current submit+wait median (ms) | Paired completion ratio [95% CI] |
|---|---|---|---:|---:|---:|---:|
| 17x19 | 3x3 | Replicate | 5.312/5.920 | 0.935 [0.783,1.198] | 0.144/0.142 | 1.043 [0.951,1.150] |
| 17x19 | 3x3 | ConstantZero | 5.808/6.304 | 0.920 [0.786,1.241] | 0.388/0.810 | 0.756 [0.431,1.048] |
| 17x19 | 9x9 | Replicate | 15.008/15.232 | 0.968 [0.848,1.076] | 12.349/3.435 | 1.316 [0.731,3.835] |
| 17x19 | 9x9 | ConstantZero | 51.216/50.208 | 1.015 [0.956,1.068] | 0.443/0.407 | 0.905 [0.844,1.186] |
| 640x480 | 3x3 | Replicate | 236.864/240.016 | 0.939 [0.898,1.002] | 9.491/7.389 | 1.191 [0.854,2.112] |
| 640x480 | 3x3 | ConstantZero | 240.640/239.120 | 0.974 [0.880,1.135] | 3.677/3.544 | 1.208 [0.575,2.443] |
| 640x480 | 9x9 | Replicate | 981.632/487.520 | 2.015 [1.992,2.061] | 1.484/1.124 | 1.326 [1.098,1.636] |
| 640x480 | 9x9 | ConstantZero | 1049.856/465.936 | 2.320 [2.220,2.469] | 1.441/0.924 | 1.730 [0.924,1.929] |
| 640x480 | 31x31 | Replicate | 7837.456/659.424 | 12.582 [11.918,12.769] | 8.285/1.077 | 7.663 [7.223,7.882] |
| 640x480 | 31x31 | ConstantZero | 2657.392/143.360 | 18.511 [18.184,18.695] | 2.857/0.323 | 8.840 [8.357,9.183] |
| 1920x1080 | 3x3 | Replicate | 265.280/261.824 | 1.014 [1.004,1.020] | 0.369/0.363 | 1.009 [0.978,1.033] |
| 1920x1080 | 3x3 | ConstantZero | 271.792/262.112 | 1.038 [1.030,1.044] | 0.372/0.364 | 1.021 [1.003,1.049] |
| 1920x1080 | 9x9 | Replicate | 2328.912/414.496 | 5.617 [5.571,5.642] | 2.446/0.540 | 4.459 [4.414,4.556] |
| 1920x1080 | 9x9 | ConstantZero | 1656.448/370.336 | 4.471 [4.457,4.480] | 1.770/0.487 | 3.656 [3.532,3.744] |
| 1920x1080 | 31x31 | Replicate | 16969.040/1033.280 | 16.456 [16.392,16.505] | 17.195/1.212 | 14.105 [13.848,14.212] |
| 1920x1080 | 31x31 | ConstantZero | 14702.704/810.688 | 18.139 [18.099,18.171] | 14.974/0.969 | 15.388 [14.870,15.779] |
| 3840x2160 | 3x3 | Replicate | 918.576/909.360 | 1.010 [1.007,1.013] | 1.793/1.333 | 1.293 [0.989,1.701] |
| 3840x2160 | 3x3 | ConstantZero | 915.472/926.448 | 0.989 [0.986,0.991] | 1.111/1.131 | 0.994 [0.960,1.013] |
| 3840x2160 | 9x9 | Replicate | 6712.832/1332.272 | 5.017 [4.994,5.124] | 6.905/1.490 | 4.629 [4.538,4.781] |
| 3840x2160 | 9x9 | ConstantZero | 5510.368/1176.912 | 4.680 [4.678,4.684] | 8.111/2.376 | 3.942 [2.680,4.232] |
| 3840x2160 | 31x31 | Replicate | 56941.920/3605.536 | 15.793 [15.739,15.831] | 57.153/3.810 | 15.001 [14.899,15.048] |
| 3840x2160 | 31x31 | ConstantZero | 59528.464/3403.840 | 17.476 [17.426,17.515] | 59.777/3.622 | 16.489 [16.408,16.580] |
| 640x480 | 1x81 | Replicate | 230.128/136.992 | 1.678 [1.660,1.695] | 0.390/0.292 | 1.358 [1.251,1.389] |
| 640x480 | 1x81 | ConstantZero | 240.832/128.480 | 1.876 [1.849,1.883] | 0.363/0.262 | 1.399 [1.342,1.442] |
| 640x480 | 81x1 | Replicate | 181.472/173.008 | 1.048 [1.036,1.058] | 0.306/0.297 | 1.032 [0.979,1.104] |
| 640x480 | 81x1 | ConstantZero | 200.704/169.216 | 1.186 [1.177,1.197] | 0.322/0.308 | 1.046 [1.017,1.136] |
