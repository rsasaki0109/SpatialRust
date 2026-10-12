# Exact separable GPU rectangular morphology

Workspace: `C:/Users/rsasa/Workspace/SpatialRust`.
Parent main: `dcabe107ee18748c26a734cba7699f7f0fc64e6e` (PR #137 merged).
Slice: reduce large rectangular erosion/dilation work while preserving the
previous byte results and explicit device residency. No public signature,
dependency, kernel range or CPU/GPU placement policy changes.

## Operation and ownership

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/spatial.rs`
retains the original full-box shader for small work. Odd kernel dimensions in
`1..=31` and single-channel inputs remain required. Boxes with area at least
81 and at least 1,048,576 pixel/sample visits use two compute passes in one
encoder/submission. Selection compares pixel count against ceil(1,048,576/area),
avoiding multiplication overflow for unvalidated profiler options too.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/morphology_separable.wgsl`
writes the horizontal neighborhood minimum/maximum to an exclusive RGBA8Uint
texture, then reduces those values vertically. Extrema are associative and
idempotent: min/max over the Cartesian rectangle factors exactly into its
axes, and repeating a clamped edge value changes neither extremum. Scanning
only valid coordinates therefore preserves replicate borders, including
one-pixel axes and kernels wider/taller than the image. Unused G/B/A bytes are
zero in both passes. Interior sampling drops from kw*kh to kw+kh; a 31x31
rectangle needs at most 62 samples rather than 961.

Output and intermediate storage each consume four bytes per pixel. The
intermediate uses the existing bounded image texture pool and is returned only
after queue submission. A later acquisition is protected by queue ordering,
without a host wait. Abandoned encoding drops its scratch rather than caching
it. Caller-owned source and output textures remain exclusive until explicitly
recycled. Under budget pressure the pool may discard scratch and allocate on a
later call; pooling does not promise allocation-free execution at every size.

The planner rejects invalid kernel, dimension, signed-coordinate, byte-size,
texture, workgroup, dispatch, binding and uniform limits before output/scratch
allocation or cache initialization. The 32-byte uniform is bounded by both
uniform binding and buffer limits. Texture scratch requires no storage buffer.
The horizontal/vertical pipeline pair initializes lazily and concurrently
within the existing spatial family, preserving the public family count.
Resize and Sobel shader code and dispatch remain unchanged.

Payload receipts retain the source upload and add `4*N` device-written bytes
for each selected pass. The two-pass stage order is
`morphology_horizontal_gpu`, then `morphology_gpu`, recording `8*N` written
bytes and no image readback. Uniform setup is outside image payload accounting
as in the previous implementation.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/vision_chain.rs`
reserves six stages plus one for each separable blur/morphology operation, up
to eight. Insufficient capacity rejects before chain execution and preserves
prior recordings. Timestamp collection is explicit and has a separate receipt:
96/112/128 bytes for six/seven/eight stage samples.

## Correctness verification

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/morphology_previous.wgsl`
is the literal parent spatial shader, checked against the parent commit. Tests
compile its original morphology entry separately. Production's small-work
shader remains identical to this fixture.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/morphology_tests.rs`
compares the frozen shader, forced current one-pass and forced current two-pass
against an independent i64-coordinate full-rectangle CPU oracle. The 560 cases
span seven shapes, four patterns, ten rectangular kernels and both operations.
Each GPU variant verifies 185,760 pixels / 743,040 RGBA bytes, including constant
0/255, checkerboard, non-workgroup-aligned dimensions, single-pixel axes,
oversized rectangles and asymmetric 1x31/31x1/3x27/27x3 kernels. Dirty scratch
proves full writes; all unused components are checked.

Separate checks cover every legal odd kernel pair in the pure planner,
12,945/12,946-pixel 9x9 selection boundaries, unvalidated u32 maximum inputs,
signed-coordinate boundaries, fourteen device limit failures, allocation-free
invalid calls, foreign runtimes, strided ROI metadata, retained outputs, cold
concurrent cache initialization, runtime moves, rapid submission/reuse,
reduced/zero pool budgets, the small one-pass path and abandoned encoding.
Vision tests compare profiled/plain outputs and receipts with seven/eight
stages and preserve an existing recording on capacity rejection.

## Measurement method

Native measurements use the local Vulkan discrete adapter, Rust/Cargo 1.98.1
and wgpu 24.0.5. Adapter model names are omitted. Timestamp queries are explicitly
enabled; default runtimes still request none.

The ignored diagnostic in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/morphology_benchmark.rs`
uses resident inputs, outputs, uniforms and intermediate textures. Each of 38
image/kernel/operation conditions gets three warmups then twenty rounds,
alternating old/new submission order. All final RGBA image bytes must match.

GPU time sums the pass timestamp intervals, excluding inter-pass gaps, image
transfers, resource setup, compilation, CPU encoding and query collection.
Completion time starts after encoder.finish() and includes queue submission,
GPU execution/gaps and an explicit wait, excluding the same setup/transfers and
later query collection. These are resident operation measurements, not full
public API, image transfer or Vision-chain latency. The diagnostic's own GPU
tests and benchmark do not overlap; existing user jobs remain untouched.

The table reports old/current timing medians and the median of paired
old/current ratios. The latter need not equal the ratio of the medians. Each
interval bootstraps paired ratios 10,000 times with seed 20261011 and reports
the 2.5/97.5 percentiles. This describes observed local variation and does not
establish cross-device speedups. The selection cutoff is conservative internal
policy, not a device-calibrated optimum.

The initial 65,536-visit cutoff selected two passes for 33x35 / 9x9.
GPU paired ratios were 1.116/1.126 (erosion/dilation), but completion ratios
were 0.970 [0.938,1.008] / 0.915 [0.863,0.940]. The latter exposed submission
and pass overhead for small work. The final cutoff is raised to 1,048,576
visits, restoring this case to the unchanged one-pass shader. This is a
conservative adjustment, not a cross-device tuning claim. The final diagnostic
also includes 33x35 / 31x31 and 128x128 / 9x9, near the selected-work lower range.
Initial measurements are preserved at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-initial-cutoff-paired.log`,
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-initial-cutoff-estimates.json`
and
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-initial-cutoff-table.md`.

Final paired 4K / 31x31 GPU ratios are 14.574 [14.569,14.579] for
erosion and 14.583 [14.581,14.587] for dilation. Completed ratios are
13.653 [13.567,13.724] and 5.136 [3.457,8.651], respectively. Dilation's
completion samples vary substantially despite stable GPU stage intervals;
these measurements do not identify the cause of the additional wall time.
The final 33x35 / 9x9 case uses the unchanged single pass with GPU ratios
1.015/1.022 and intervals spanning parity. The selected 128x128 / 9x9 case
improves GPU intervals by 1.582/1.587, but completion intervals span parity
and median ratios are 1.002/0.903. Default/single-axis cases also show wall-time
variation. No universal completion-latency improvement is claimed.

| Image | Kernel | Operation | Previous/current GPU median (us) | Paired GPU ratio [95% CI] | Previous/current submit+wait median (ms) | Paired completion ratio [95% CI] |
|---|---|---|---:|---:|---:|---:|
| 17x19 | 3x3 | Erode | 5.248/5.200 | 1.040 [0.756,1.301] | 0.155/0.153 | 0.976 [0.943,1.128] |
| 17x19 | 3x3 | Dilate | 4.880/4.784 | 1.030 [0.764,1.303] | 0.163/0.156 | 1.005 [0.930,1.194] |
| 17x19 | 9x9 | Erode | 8.992/8.800 | 1.019 [0.881,1.137] | 0.144/0.152 | 0.960 [0.888,1.065] |
| 17x19 | 9x9 | Dilate | 8.800/8.992 | 1.001 [0.873,1.139] | 0.150/0.155 | 1.000 [0.849,1.099] |
| 33x35 | 9x9 | Erode | 9.568/9.328 | 1.015 [0.885,1.128] | 0.129/0.127 | 0.974 [0.931,1.060] |
| 33x35 | 9x9 | Dilate | 9.264/9.056 | 1.022 [0.882,1.128] | 0.167/0.169 | 0.928 [0.866,1.106] |
| 33x35 | 31x31 | Erode | 56.960/10.816 | 5.270 [4.852,5.468] | 0.192/0.153 | 1.248 [1.180,1.339] |
| 33x35 | 31x31 | Dilate | 57.296/10.592 | 5.411 [5.053,5.585] | 0.203/0.154 | 1.364 [1.193,1.437] |
| 128x128 | 9x9 | Erode | 14.848/9.552 | 1.582 [1.454,1.707] | 0.170/0.159 | 1.002 [0.914,1.132] |
| 128x128 | 9x9 | Dilate | 15.072/9.520 | 1.587 [1.418,1.770] | 0.158/0.172 | 0.903 [0.818,1.044] |
| 640x480 | 3x3 | Erode | 36.224/36.304 | 0.999 [0.953,1.050] | 0.185/0.188 | 0.962 [0.938,0.997] |
| 640x480 | 3x3 | Dilate | 36.016/36.192 | 0.998 [0.952,1.049] | 0.185/0.185 | 1.018 [0.951,1.050] |
| 640x480 | 9x9 | Erode | 190.496/64.528 | 2.965 [2.901,3.030] | 0.355/0.234 | 1.517 [1.392,1.636] |
| 640x480 | 9x9 | Dilate | 191.392/65.360 | 2.930 [2.887,3.046] | 0.321/0.196 | 1.636 [1.540,1.699] |
| 640x480 | 31x31 | Erode | 2121.776/152.496 | 13.989 [13.745,14.052] | 2.291/0.306 | 7.520 [7.139,7.771] |
| 640x480 | 31x31 | Dilate | 2122.576/152.688 | 13.970 [13.766,14.049] | 2.277/0.294 | 7.779 [7.258,7.947] |
| 1920x1080 | 3x3 | Erode | 230.464/230.656 | 1.002 [0.988,1.007] | 0.358/0.365 | 0.973 [0.951,1.003] |
| 1920x1080 | 3x3 | Dilate | 231.024/230.448 | 1.003 [0.993,1.010] | 0.355/0.361 | 0.992 [0.967,1.024] |
| 1920x1080 | 9x9 | Erode | 1255.664/378.288 | 3.319 [3.301,3.334] | 1.417/0.542 | 2.584 [2.490,2.720] |
| 1920x1080 | 9x9 | Dilate | 1255.536/378.432 | 3.319 [3.302,3.332] | 1.412/0.545 | 2.604 [2.569,2.650] |
| 1920x1080 | 31x31 | Erode | 11650.000/803.184 | 14.524 [14.502,14.534] | 11.848/0.976 | 12.164 [12.033,12.300] |
| 1920x1080 | 31x31 | Dilate | 11323.168/782.224 | 14.477 [14.448,14.506] | 11.516/0.940 | 12.260 [11.914,12.374] |
| 3840x2160 | 3x3 | Erode | 762.880/763.792 | 0.999 [0.994,1.003] | 0.898/0.902 | 1.011 [0.990,1.013] |
| 3840x2160 | 3x3 | Dilate | 770.368/766.944 | 1.005 [1.001,1.006] | 1.046/1.066 | 1.015 [0.990,1.055] |
| 3840x2160 | 9x9 | Erode | 4046.656/1234.320 | 3.284 [3.279,3.285] | 4.281/1.475 | 2.912 [2.874,3.009] |
| 3840x2160 | 9x9 | Dilate | 4029.392/1228.672 | 3.282 [3.279,3.287] | 4.271/1.499 | 2.949 [2.909,2.985] |
| 3840x2160 | 31x31 | Erode | 45523.552/3123.040 | 14.574 [14.569,14.579] | 45.777/3.356 | 13.653 [13.567,13.724] |
| 3840x2160 | 31x31 | Dilate | 45535.040/3122.544 | 14.583 [14.581,14.587] | 51.527/10.543 | 5.136 [3.457,8.651] |
| 640x480 | 1x31 | Erode | 109.440/109.904 | 1.003 [0.914,1.095] | 0.276/0.275 | 0.981 [0.895,1.070] |
| 640x480 | 1x31 | Dilate | 109.456/109.728 | 1.005 [0.915,1.092] | 0.257/0.266 | 0.948 [0.899,1.034] |
| 640x480 | 31x1 | Erode | 62.704/62.640 | 1.003 [0.956,1.050] | 0.259/0.234 | 1.208 [1.001,1.263] |
| 640x480 | 31x1 | Dilate | 62.416/62.608 | 0.994 [0.959,1.045] | 0.208/0.210 | 0.940 [0.862,1.058] |
| 640x480 | 3x27 | Erode | 187.312/72.288 | 2.598 [2.575,2.609] | 0.350/0.244 | 1.397 [1.340,1.464] |
| 640x480 | 3x27 | Dilate | 189.632/72.544 | 2.584 [2.570,2.693] | 0.352/0.245 | 1.474 [1.281,1.598] |
| 640x480 | 27x3 | Erode | 153.488/75.584 | 2.026 [2.008,2.134] | 0.336/0.282 | 1.379 [1.263,1.550] |
| 640x480 | 27x3 | Dilate | 456.944/245.968 | 1.855 [1.734,2.014] | 0.703/0.587 | 1.389 [1.236,1.530] |
| 640x480 | 15x15 | Erode | 1436.912/359.104 | 4.140 [3.965,4.331] | 1.746/0.707 | 2.509 [2.317,2.706] |
| 640x480 | 15x15 | Dilate | 1462.576/367.696 | 4.461 [3.931,4.946] | 1.828/0.739 | 2.826 [2.095,3.362] |

## Completed validation

- 128 native tests passed: 102 unit, 4 CDF, 5 CLAHE, 5 equalization,
  3 histogram and 9 stage-timing tests. Six explicit diagnostics remain ignored
  by the normal suite; the morphology diagnostic was separately executed.
- Native all-feature/all-target and default/all-target Clippy passed with
  `-D warnings`. Minimal image and combined image/AoSoA WASM library Clippy
  passed. WASM execution remains unverified.
- The exact minimal-feature CI planner command passed without an adapter.
- The 38-condition release diagnostic completed with all final RGBA bytes
  matching; twenty positive timing pairs per condition were checked.
- Formatting, diff whitespace, frozen shader and publication scans passed.

Local receipts (ignored artifacts, not publication assets):

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-final-regressions.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-final-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-default-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-wasm-minimal.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-wasm-combined.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-ci-pure-plan.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-paired.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-estimates.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-morphology-table.md`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-gpu-morphology.py`

The estimate script validates all sample counts and positive timings, retains
individual paired ratios/ranges and generates the full table above.


## Reproduction

Run from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-image --target wasm32-unknown-unknown --lib -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown --lib -- -D warnings
cargo test -p spatialrust-gpu --features gpu-image morphology_plan_bounds_kernels_coordinates_textures_and_dispatch --lib
cargo test -p spatialrust-gpu --features gpu-image --release --lib paired_previous_current_morphology -- --ignored --nocapture --test-threads=1
cargo fmt --all -- --check
git diff --check
```

CI's existing `spatialrust-gpu-image` job compiles GPU tests and executes the
new adapter-free planner test. The required job inventory remains 114. Native
kernel tests and timed diagnostics require a usable adapter; the planner test
does not. WASM checks compile only; browser kernel execution is not verified.
