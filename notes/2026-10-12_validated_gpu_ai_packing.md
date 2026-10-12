# Validated GPU AI packing

Date: 2026-10-12. This slice follows merged
[PR #146](https://github.com/rsasaki0109/SpatialRust/pull/146), main
`a9ba90e132e1282aa15e8d111af6fa7e5fbd5120`.

## Delivered behavior

`pack_ai_chw_gpu` previously computed output sizes and acquired storage before
checking output-binding/device limits or profiler capacity. Unsupported storage
sizes could reach wgpu validation rather than return a spatial error. Packing
now validates its complete work plan before output/uniform allocation or
AI-pipeline initialization. Valid operations retain the original shader,
pipeline factory, normalization, CHW shape/order and transfer receipts.

The plan requires positive source dimensions, channels in 1..=4, device texture
limits and signed texture coordinates. Widened host arithmetic checks pixel and
element counts against u32 shader indexing and multiplies element count by four
with checked arithmetic. Output bytes must fit both `max_buffer_size` and
`max_storage_buffer_binding_size`. It also checks the existing 16x16/256-lane
workgroup, per-axis dispatch limits, one sampled texture, storage buffer and
uniform buffer, a 64-byte uniform/buffer, one bind group and three bindings.
It imposes no new workgroup-storage or backend-feature requirement.

Runtime ownership and existing finite scale/mean, finite nonzero std validation
remain ahead of execution; all four parameter slots are still validated, even
for fewer logical channels. Negative std and permitted intermediate overflow
retain their previous behavior. Profile ownership and one free stage are
checked before work. Rejection preserves previously recorded stages.

Production still computes `(byte * scale - mean[channel]) / std[channel]` in
the original f32 operation order. There is one runtime-owned cached pipeline,
one GPU pass and no extra scratch or payload transfer. Receipts record
`4 * channels * width * height` GPU write bytes, zero implicit payload readback,
and 16 query bytes on explicit one-stage collection. Public signatures,
dependencies, storage retention and tensor ownership/readback remain unchanged.
The buffer pool bounds retained storage, not active outputs or peak driver memory.

## Correctness and feature validation

Six new tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/ai_pack_tests.rs`
cover:

- **23,857,560 exact f32 output words** compared with a frozen parent GPU
  shader across 192 channel/shape/normalization cases. Twelve shapes include
  singleton axes, 7/8/9 and 15/16/17 boundaries, odd tails and 255x257/256x256/
  257x259 images. Channels 1/2/3/4 and four parameter sets cover identity,
  exact binary scaling/negative std/signed zero, ordinary AI normalization
  and extreme finite parameters producing overflow. Forced lane, forced fixed-
  channel and public-original outputs retain frozen words. Identity and binary
  scaling additionally match an independent CPU CHW lookup.
- Dimension/channel, texture/signed-coordinate, pixel/element overflow, exact
  buffer-binding boundaries, output/uniform buffer, compute/dispatch and binding
  limits. Zero workgroup storage remains acceptable. Large synthetic limits are
  pure planning tests, not claims that giant buffers can be allocated.
- Invalid normalization, foreign image/profile ownership and exhausted capacity
  leave storage-pool counters and cold AI caches unchanged. Capacity rejection
  preserves an earlier Sobel stage and its 16-byte query collection.
- A metadata-only negative probe derives declared output dimensions from the
  real device's buffer limits while backing the probe with one tiny texture.
  It confirms rejection before any work or cache creation without allocating a
  huge source. The mismatched probe is dropped rather than returned to the pool.
- Four concurrent cold calls on a strided 257x259 RGB ROI, independent live
  tensors while the old source texture is reused, rapid submissions, runtime
  movement, allocation reuse, zero retention budget and foreign tensor readback.
- A **4096x4096 complete resident Vision chain** on constant RGB input with
  1x1 blur/morphology and identity packing. All **16,777,216 final f32 words** are
  zero, matching the independent constant-image/Sobel expectation. It records
  six stages, 67,108,864 upload bytes, 402,653,184 GPU write bytes and zero payload
  readback before explicit collection. Query collection separately returns
  96 bytes; explicit tensor readback adds 67,108,864 payload bytes.

The frozen parent shader is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/ai_pack_previous.wgsl`.
The production shader, pipeline factory and tensor ownership/readback block are
checked against parent main. Both experimental shaders live only in native
fixtures; production does not compile, cache or select them.

Final validation: **154 native image/point GPU tests pass**, zero failures and
12 ignored diagnostics (127 unit plus 27 CDF/CLAHE/equalization/histogram/shared-
stage tests). Strict default/all-feature all-target Clippy, two WASM feature
compilation profiles, formatting and diff checks pass. WASM compilation does
not establish browser execution. The original six-to-eight-pass chain regressions
also pass; the new 4096x4096 proof covers the six-pass constant-image configuration.

```text
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-image
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-image,gpu-aoso-staging
cargo fmt --all --check
git diff --check
```

## Why the experimental kernels were deferred

Two native release pilots tested the same 24 conditions: eight shapes crossed
with channels 1/3/4. Each used two warmups and twenty alternating frozen/current
rounds, checking every output against frozen words. Experiments selected RGB/
RGBA at 65,536 pixels or more; twelve cases were selected and twelve retained
the original kernel. Raw samples and all controls are retained privately.

The first prototype assigns channels to separate lanes in an 8x8x4 workgroup.
All twelve selected GPU intervals fell below one: frozen/experimental paired
ratios range **0.551–0.713**. Completed ratios range 0.873–1.329, including noisy
positive samples. The GPU regression was consistent enough to reject this route.
The exact prototype remains
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/ai_pack_channels_experiment.wgsl`.

The second prototype retains 16x16 workgroups and one texture read per pixel,
fixing the loop bound with a pipeline override for RGB/RGBA. At 256x256,
GPU ratios were 1.085 [1.056, 1.103] for RGB and 1.058 [1.039, 1.077] for RGBA,
but completed intervals crossed one (1.010 [0.948, 1.067] and
0.948 [0.886, 1.020]). Full-HD/UHD GPU ratios were about 0.997–1.001; UHD RGBA
had a small adverse GPU interval, 0.997 [0.996, 0.998]. The 17x4097 RGBA
completed ratio was 0.968 [0.938, 0.983]. These results do not establish a
completed-call gain worth additional production pipelines. The prototype remains
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/ai_pack_specialized_experiment.wgsl`.

No speedup is claimed for the shipped validation change. Both original-kernel
controls and adverse experimental samples are retained; no experimental
selection policy or new pipeline is present in production.

## Final resident diagnostic

Release native Windows, Rust/Cargo 1.98.1, wgpu 24.0.5, Vulkan discrete backend
with pass timestamps. No local adapter model is published. No GPU tests or builds
launched by this task overlap any measurement run. Desktop activity, clocks
and external GPU load are not controlled.

The final run compares frozen and current original kernels for all 24 conditions.
Each has two warmups and twenty alternating rounds. Every warmup/measured output
matches frozen words, CHW shape, write/readback accounting and a one-stage,
16-byte query report. Pipeline setup, upload, profiler creation, query collection,
payload readback/checking and recycling are outside timers. Completed timers
include source/plan/profile validation, output/uniform allocation, encoding,
submission and `wait_idle`. The frozen shader shares the current host executor;
this **does not measure historical API overhead or validation cost**.

One runtime and its default 64 MiB retained-storage budget persist across shapes.
The pool is not cleared between conditions; live/output allocation and retained
occupancy affect reuse. Times are variant medians. Ratios are medians of twenty
paired frozen/current times, with 10,000 paired median bootstrap resamples
(seed 20261012; sorted indices 249 and 9749). Above one favors current. These
describe one run's samples rather than independent-repeat confidence.

| Size | Channels | Frozen/current GPU us | GPU ratio [95% CI] | Frozen/current completed ms | Completed ratio [95% CI] |
| --- | ---: | --- | --- | --- | --- |
| 17x19 | 1 | 3.936 / 3.504 | 1.128 [0.944, 1.290] | 0.257 / 0.257 | 1.002 [0.890, 1.059] |
| 17x19 | 3 | 4.160 / 4.128 | 1.000 [0.923, 1.036] | 0.242 / 0.241 | 0.989 [0.948, 1.065] |
| 17x19 | 4 | 4.368 / 4.288 | 1.004 [0.993, 1.041] | 0.221 / 0.222 | 0.991 [0.932, 1.050] |
| 128x128 | 1 | 4.128 / 4.128 | 1.016 [0.953, 1.069] | 0.248 / 0.268 | 0.945 [0.883, 1.086] |
| 128x128 | 3 | 4.416 / 4.448 | 0.971 [0.891, 1.029] | 0.279 / 0.292 | 0.950 [0.898, 1.030] |
| 128x128 | 4 | 4.704 / 4.704 | 1.003 [0.916, 1.065] | 0.310 / 0.301 | 1.030 [0.919, 1.202] |
| 256x256 | 1 | 5.376 / 5.376 | 1.000 [0.953, 1.030] | 0.250 / 0.277 | 0.919 [0.871, 0.964] |
| 256x256 | 3 | 7.296 / 7.328 | 0.996 [0.981, 1.044] | 0.318 / 0.311 | 0.992 [0.928, 1.106] |
| 256x256 | 4 | 8.880 / 8.928 | 0.991 [0.963, 1.000] | 0.329 / 0.342 | 0.961 [0.882, 1.074] |
| 640x480 | 1 | 15.504 / 15.600 | 0.982 [0.964, 1.002] | 0.363 / 0.358 | 0.880 [0.771, 1.097] |
| 640x480 | 3 | 21.536 / 21.760 | 0.993 [0.973, 1.006] | 0.459 / 0.471 | 0.974 [0.889, 1.115] |
| 640x480 | 4 | 28.272 / 28.272 | 1.006 [0.991, 1.018] | 0.474 / 0.467 | 1.002 [0.865, 1.150] |
| 1920x1080 | 1 | 379.904 / 380.512 | 1.001 [0.990, 1.012] | 1.062 / 1.059 | 1.002 [0.972, 1.102] |
| 1920x1080 | 3 | 980.896 / 982.640 | 0.999 [0.998, 1.002] | 2.040 / 1.987 | 1.038 [1.019, 1.056] |
| 1920x1080 | 4 | 1448.000 / 1447.120 | 1.001 [1.000, 1.003] | 3.784 / 3.759 | 0.997 [0.995, 1.013] |
| 3840x2160 | 1 | 4002.096 / 4006.176 | 0.997 [0.995, 1.002] | 13.030 / 13.061 | 0.995 [0.990, 1.005] |
| 3840x2160 | 3 | 9934.272 / 9935.088 | 0.999 [0.999, 1.001] | 26.141 / 25.987 | 1.001 [0.994, 1.010] |
| 3840x2160 | 4 | 12572.480 / 12562.352 | 1.000 [0.999, 1.001] | 33.714 / 33.804 | 0.996 [0.993, 0.999] |
| 17x4097 | 1 | 41.280 / 41.952 | 0.995 [0.907, 1.025] | 1.022 / 1.058 | 0.998 [0.779, 1.119] |
| 17x4097 | 3 | 100.720 / 100.096 | 1.008 [0.986, 1.021] | 1.193 / 1.250 | 0.991 [0.927, 1.013] |
| 17x4097 | 4 | 86.416 / 81.872 | 1.008 [0.987, 1.157] | 1.272 / 1.227 | 0.999 [0.930, 1.052] |
| 4097x17 | 1 | 35.680 / 65.216 | 0.973 [0.543, 1.008] | 0.674 / 1.209 | 0.914 [0.537, 1.053] |
| 4097x17 | 3 | 104.128 / 105.344 | 0.984 [0.952, 1.006] | 1.219 / 1.236 | 0.969 [0.914, 1.048] |
| 4097x17 | 4 | 89.408 / 94.880 | 0.999 [0.986, 1.018] | 1.203 / 1.199 | 0.997 [0.907, 1.042] |

Reproduction and private log paths are absolute:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/ai_tensor.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/ai_pack_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-regressions.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-default-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-wasm.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-wasm-combined.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-release-build.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-paired.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-estimates.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/validated-gpu-ai-pack-table.md`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-validated-gpu-ai-pack.py`

Earlier raw/estimate/table sets use `ai-pack-channel-lanes-*` and
`ai-pack-fixed-channel-*` in that private directory. Estimates include all
sample values, minimum/maximum times and paired intervals.

```text
cargo test -p spatialrust-gpu --all-features --release --lib paired_frozen_and_validated_ai_pack -- --ignored --nocapture --test-threads=1
```
