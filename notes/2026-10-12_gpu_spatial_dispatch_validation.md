# GPU spatial dispatch validation and shared Sobel experiment

This slice follows merged PR [#143](https://github.com/rsasaki0109/SpatialRust/pull/143)
on main `edaf62094dedb7e92fb1e7c875919e0bcd3ea7a7`. The delivered change is
pre-allocation validation for GPU nearest resize and Sobel. Shared Sobel tiling
remains a native test fixture because the initial performance improvement did
not reproduce consistently. No performance improvement is claimed for the
validation change.

## Production behavior

Previously nearest resize rejected zero output dimensions, but dimensions above
device limits reached texture creation. Sobel could initialize the spatial
pipeline cache before detecting a source owned by a different runtime. Both
operations could allocate work before discovering insufficient profiler capacity.
The public entry paths now return `SpatialError::InvalidArgument` for invalid
dispatch dimensions/limits and validate runtime/profiler ownership and capacity
before output allocation or spatial-pipeline initialization.

The shared validator requires positive source/output dimensions supported by the
device's 2D texture limit and signed shader coordinates. It checks the existing
16x16 workgroup, 256 invocations, dispatch counts, one sampled texture, one storage
texture, one uniform buffer, a 32-byte uniform binding/buffer, one bind group and
three bindings. It adds no storage-buffer or workgroup-storage requirement.

Nearest resize also checks each axis with widened host arithmetic:
`u64(output_dimension - 1) * u64(source_dimension) <= u32::MAX`.
This rejects overflow in the existing shader's coordinate multiplication before
division. The exact boundary `65535 * 65537 == u32::MAX` is accepted by the pure
validator; the next output coordinate is rejected. These synthetic limits are
math tests, not claims that the local device supports such large textures.

The production WGSL, bind-group layout, pipeline family and shared executor are
unchanged from the merged parent. Valid resize mapping, clamped L1 Sobel bytes,
metadata, single-pass stage names and explicit transfers remain unchanged. Sobel
writes `[magnitude, 0, 0, 0]` to its physical RGBA texture. Each operation still
records four GPU write bytes per pixel, zero implicit host readback, and 16 query
bytes when one timestamp stage is explicitly collected. There is no new payload
copy, production pipeline, global scratch or intermediate texture. Public
signatures, dependencies and the six-to-eight-pass Vision chain remain unchanged.

## Correctness and validation

Five new native tests cover:

- Sobel: nine shapes, including singleton axes, 15/16/17 boundaries, odd tails
  and rectangular images, crossed with six patterns (mixed, zero, full scale,
  checkerboard, low gradient and impulse). An independent i64 3x3 convolution
  checks frozen/direct/experimental outputs over 1,200,054 pixels and
  **14,400,648 exact RGBA bytes** overall, including the unused channels.
- Resize: seven source shapes crossed with six output shapes give 42 mappings.
  An independent u64 CPU lookup checks both frozen and public outputs over
  **104,888 exact RGBA bytes**, including alpha, metadata and receipt accounting.
- Pure planning: dimension, signed-coordinate, texture, compute, binding and
  uniform limits; zero workgroup/storage-buffer capacities remain acceptable
  for production. Both resize axes test the exact u32 product boundary.
- Rejection: foreign runtimes, color Sobel, invalid resize output sizes and
  exhausted profiler capacity leave texture-pool counters unchanged. Invalid
  calls on a cold runtime leave its spatial-pipeline cache uninitialized.
  Capacity rejection preserves an earlier stage and its 16-byte collection.
- Lifetime/pooling: a 1025x1025 ROI from a padded strided image, four simultaneous
  cold calls, independently live outputs while their old source texture is
  reused, rapid submissions, runtime movement, warm allocation reuse and a
  zero-budget pool all preserve exact results and metadata/receipts.

Final validation completed before the final performance run:

```text
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1
  142 passed, 0 failed, 10 ignored native diagnostics
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-image
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-image,gpu-aoso-staging
cargo fmt --all --check
git diff --check
```

The native total includes 115 unit tests and 27 CDF/CLAHE/equalization/histogram/
shared-stage integration tests, covering image and point paths. WASM compilation
does not establish browser execution or browser performance. The production
shader, frozen fixture and shared executor were also compared exactly with the
parent source.

## Why shared tiles remain experimental

The test-only prototype uses a 16x16 workgroup and an 18x18 i32 halo tile
(1,296 bytes). All lanes load clamped source texels cooperatively, then reach the
barrier before partial-output lanes return. The integer Sobel arithmetic is
exact. Production does not compile or select this shader.

Three release diagnostics were run on a native Vulkan discrete backend with
pass timestamps, using Rust/Cargo 1.98.1 and wgpu 24.0.5. The device supports the
existing 256-lane workgroup and the tested 8192-pixel axes. The local adapter name
is omitted. No other GPU tests or builds launched by this task overlapped these
measurements. Desktop activity, frequency and external GPU load were not
controlled, so these runs do not establish portable speedups.

The first pilot used an experimental 65,536-pixel selection threshold and 24
conditions. Its frozen/selected paired median ratios ranged 0.763–1.090 for GPU
time and 0.940–1.047 for completed calls. Larger images initially improved, while
small and thin images regressed: 256x256 mixed GPU ratio 0.947 [0.917, 0.976],
and 4097x17 mixed/hot ratios about 0.782/0.763.

The second pilot raised the experimental threshold to 1,048,576 pixels and
expanded to 36 conditions. Frozen/selected GPU ratios ranged 0.952–1.020 and
completed-call ratios 0.928–1.746; the large positive wall-time outliers are
retained in this range. Earlier large-image GPU gains did not repeat: 1025x1025
mixed was 0.966 [0.963, 0.968], while 8192x129 mixed/hot was about 0.957/0.952.
The 3840x2160 mixed completed-call ratio was 0.961 [0.901, 0.988] and the
8192x129 hot ratio 0.928 [0.894, 0.991]. These selection policies were removed
before delivery.

The final run below compares frozen direct, forced experimental tile and the
current public direct Sobel path for all 36 conditions. The experimental GPU
ratios range 0.762–1.084, with seven intervals below one and 21 above one; its
completed ratios range 0.932–1.062, with one interval below one and seven above
one. Some large-image improvements returned, but the adverse conditions and
inconsistent earlier repeats prevent promoting this policy.

Both direct variants use identical shader math. Their paired GPU ratios range
0.973–1.042 and completed ratios 0.894–1.057, demonstrating that invocation
order and timing noise also affect unchanged kernels. The frozen variant shares
the current host validation and executor: this is a shader diagnostic, **not a
historical full-API comparison or a measurement of validation overhead**.

## Final paired measurements

The ignored native diagnostic performs four warmups followed by twenty rotating
rounds per condition. Mixed and full-scale hot patterns use the same resident
source for all variants. Each warmup and measured output is checked against the
CPU oracle, metadata, single-stage receipt, zero implicit payload readback and
16-byte explicit query collection. Upload, pipeline setup, profiler creation,
query collection, result readback/checking and recycling are outside timers.
Completed timers include validation, output/uniform allocation, encoding,
submission and `wait_idle`. They exclude those listed setup/check operations.

Times below are per-variant medians. Ratios are medians of twenty paired
frozen/variant sample ratios, with 10,000 paired median bootstrap resamples
(seed 20261012; sorted indices 249 and 9749). A ratio above one favors the
variant. These are descriptive intervals from one run, not independent-run
confidence or a universal speed guarantee. All 36 conditions are included.

| Size | Pattern | Frozen / experimental / current GPU us | Frozen/current GPU ratio [95% CI] | Frozen/experimental GPU ratio [95% CI] | Frozen / experimental / current completed ms | Frozen/current completed ratio [95% CI] | Frozen/experimental completed ratio [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17x19 | mixed | 4.448 / 4.848 / 4.448 | 1.000 [0.982, 1.018] | 0.915 [0.891, 1.041] | 0.230 / 0.240 / 0.244 | 0.964 [0.905, 1.041] | 0.940 [0.912, 1.003] |
| 17x19 | hot | 4.448 / 4.960 / 4.448 | 1.000 [0.986, 1.015] | 0.893 [0.881, 0.903] | 0.239 / 0.233 / 0.225 | 1.008 [0.965, 1.078] | 1.003 [0.951, 1.074] |
| 128x128 | mixed | 5.344 / 5.296 / 5.344 | 1.000 [0.979, 1.021] | 1.021 [0.991, 1.035] | 0.259 / 0.255 / 0.263 | 0.965 [0.916, 1.030] | 1.003 [0.945, 1.044] |
| 128x128 | hot | 5.296 / 5.232 / 5.088 | 1.042 [1.012, 1.086] | 1.009 [0.971, 1.056] | 0.249 / 0.247 / 0.262 | 0.969 [0.859, 1.080] | 0.992 [0.933, 1.073] |
| 255x257 | mixed | 8.832 / 8.752 / 8.752 | 1.002 [0.993, 1.013] | 0.998 [0.978, 1.024] | 0.272 / 0.280 / 0.277 | 1.007 [0.968, 1.061] | 1.022 [0.947, 1.053] |
| 255x257 | hot | 8.752 / 8.736 / 8.656 | 1.015 [0.991, 1.049] | 1.000 [0.978, 1.007] | 0.241 / 0.237 / 0.248 | 0.988 [0.927, 1.037] | 0.992 [0.968, 1.067] |
| 256x256 | mixed | 8.832 / 8.736 / 8.672 | 1.011 [0.986, 1.049] | 1.011 [0.986, 1.017] | 0.268 / 0.272 / 0.260 | 1.020 [0.980, 1.102] | 0.991 [0.874, 1.040] |
| 256x256 | hot | 8.704 / 8.720 / 8.576 | 1.007 [0.982, 1.032] | 1.000 [0.960, 1.011] | 0.248 / 0.241 / 0.249 | 1.032 [0.939, 1.071] | 0.993 [0.933, 1.072] |
| 257x259 | mixed | 9.184 / 9.280 / 9.184 | 1.000 [0.995, 1.005] | 0.984 [0.976, 0.991] | 0.229 / 0.223 / 0.226 | 1.005 [0.926, 1.031] | 1.046 [0.961, 1.098] |
| 257x259 | hot | 9.152 / 9.216 / 9.072 | 1.009 [0.986, 1.016] | 0.997 [0.979, 1.007] | 0.260 / 0.278 / 0.245 | 0.965 [0.887, 1.024] | 0.955 [0.876, 1.024] |
| 640x480 | mixed | 26.768 / 27.984 / 26.720 | 0.999 [0.993, 1.007] | 0.956 [0.951, 0.964] | 0.337 / 0.348 / 0.345 | 0.972 [0.874, 1.032] | 1.004 [0.916, 1.093] |
| 640x480 | hot | 119.968 / 116.368 / 120.224 | 1.000 [0.996, 1.003] | 1.033 [1.027, 1.041] | 0.706 / 0.682 / 0.692 | 0.988 [0.941, 1.036] | 1.008 [0.981, 1.044] |
| 1024x1024 | mixed | 540.256 / 521.424 / 541.264 | 0.999 [0.994, 1.004] | 1.039 [1.034, 1.041] | 1.226 / 1.224 / 1.219 | 1.007 [0.974, 1.027] | 1.008 [0.968, 1.040] |
| 1024x1024 | hot | 436.160 / 414.176 / 434.592 | 1.003 [1.001, 1.006] | 1.054 [1.051, 1.059] | 1.124 / 1.109 / 1.131 | 0.993 [0.977, 1.006] | 1.002 [0.981, 1.042] |
| 1920x1080 | mixed | 1081.488 / 1043.392 / 1079.824 | 1.002 [1.001, 1.004] | 1.039 [1.036, 1.042] | 2.197 / 2.145 / 2.179 | 1.027 [1.006, 1.063] | 1.031 [1.015, 1.052] |
| 1920x1080 | hot | 888.992 / 846.960 / 888.784 | 1.000 [0.996, 1.001] | 1.050 [1.048, 1.051] | 1.964 / 1.942 / 1.967 | 1.005 [0.975, 1.021] | 1.014 [0.952, 1.024] |
| 3840x2160 | mixed | 4568.928 / 4379.024 / 4567.856 | 1.000 [0.999, 1.001] | 1.043 [1.041, 1.044] | 5.913 / 5.728 / 5.875 | 1.007 [1.004, 1.012] | 1.037 [1.026, 1.044] |
| 3840x2160 | hot | 3572.336 / 3348.896 / 3571.904 | 1.000 [0.999, 1.002] | 1.067 [1.066, 1.068] | 4.883 / 4.634 / 4.915 | 0.997 [0.988, 1.005] | 1.053 [1.045, 1.067] |
| 17x4097 | mixed | 61.872 / 67.184 / 61.600 | 1.002 [0.998, 1.014] | 0.923 [0.916, 0.927] | 1.099 / 1.141 / 0.953 | 1.040 [0.973, 1.327] | 1.062 [0.940, 1.110] |
| 17x4097 | hot | 64.128 / 71.024 / 63.920 | 1.003 [0.999, 1.007] | 0.902 [0.898, 0.909] | 0.931 / 0.951 / 0.946 | 0.939 [0.876, 1.022] | 0.964 [0.823, 0.988] |
| 4097x17 | mixed | 49.168 / 64.112 / 49.040 | 1.000 [0.982, 1.021] | 0.772 [0.755, 0.774] | 1.123 / 1.193 / 1.028 | 1.054 [0.963, 1.146] | 0.972 [0.904, 1.026] |
| 4097x17 | hot | 53.840 / 71.840 / 55.744 | 0.973 [0.835, 1.009] | 0.762 [0.749, 0.773] | 0.952 / 1.133 / 1.073 | 0.894 [0.783, 1.007] | 0.932 [0.732, 1.026] |
| 4096x2048 | mixed | 4267.536 / 4059.136 / 4268.464 | 0.999 [0.998, 1.001] | 1.053 [1.052, 1.055] | 11.038 / 10.819 / 11.011 | 1.002 [0.993, 1.004] | 1.014 [1.011, 1.028] |
| 4096x2048 | hot | 3474.016 / 3203.648 / 3471.984 | 1.000 [0.999, 1.001] | 1.084 [1.084, 1.085] | 10.297 / 9.963 / 10.279 | 1.006 [0.998, 1.014] | 1.036 [1.022, 1.047] |
| 1023x1025 | mixed | 519.440 / 505.440 / 517.952 | 1.000 [0.998, 1.003] | 1.030 [1.026, 1.036] | 2.040 / 2.037 / 2.053 | 0.990 [0.973, 1.006] | 0.995 [0.968, 1.019] |
| 1023x1025 | hot | 472.992 / 454.624 / 472.144 | 1.003 [0.999, 1.006] | 1.041 [1.036, 1.043] | 2.072 / 2.048 / 2.090 | 0.997 [0.979, 1.010] | 1.012 [0.968, 1.044] |
| 1025x1025 | mixed | 530.672 / 518.512 / 530.080 | 1.001 [0.996, 1.006] | 1.022 [1.020, 1.025] | 2.101 / 2.112 / 2.110 | 0.996 [0.960, 1.020] | 0.991 [0.976, 1.006] |
| 1025x1025 | hot | 476.432 / 456.112 / 476.528 | 0.999 [0.997, 1.002] | 1.045 [1.044, 1.047] | 2.025 / 2.032 / 2.060 | 0.977 [0.965, 1.002] | 0.995 [0.966, 1.024] |
| 128x8192 | mixed | 543.968 / 530.848 / 544.560 | 0.998 [0.994, 1.001] | 1.024 [1.023, 1.026] | 2.100 / 2.063 / 2.101 | 1.005 [0.982, 1.032] | 1.009 [1.001, 1.041] |
| 128x8192 | hot | 462.720 / 453.264 / 461.696 | 1.001 [1.000, 1.004] | 1.021 [1.018, 1.024] | 2.004 / 1.967 / 2.015 | 0.989 [0.972, 1.012] | 1.018 [0.995, 1.024] |
| 8192x128 | mixed | 557.584 / 541.648 / 557.168 | 1.000 [0.997, 1.002] | 1.029 [1.027, 1.031] | 2.196 / 2.159 / 2.161 | 1.003 [0.987, 1.023] | 1.007 [0.985, 1.036] |
| 8192x128 | hot | 472.128 / 452.016 / 473.808 | 0.997 [0.996, 0.999] | 1.045 [1.042, 1.049] | 2.038 / 1.970 / 2.023 | 1.017 [0.969, 1.034] | 1.024 [0.983, 1.047] |
| 129x8192 | mixed | 559.616 / 544.608 / 560.368 | 0.999 [0.996, 1.001] | 1.026 [1.025, 1.030] | 2.158 / 2.071 / 2.129 | 0.988 [0.975, 1.028] | 1.030 [1.002, 1.050] |
| 129x8192 | hot | 501.392 / 491.568 / 500.736 | 1.001 [0.999, 1.002] | 1.019 [1.015, 1.022] | 2.063 / 2.044 / 2.093 | 0.985 [0.967, 1.025] | 1.017 [0.995, 1.027] |
| 8192x129 | mixed | 595.344 / 591.392 / 594.160 | 1.003 [0.997, 1.005] | 1.007 [1.002, 1.010] | 2.167 / 2.188 / 2.216 | 0.979 [0.966, 1.002] | 0.986 [0.966, 1.014] |
| 8192x129 | hot | 504.864 / 499.632 / 503.728 | 1.003 [0.999, 1.007] | 1.011 [1.009, 1.015] | 2.087 / 2.058 / 2.019 | 1.057 [1.006, 1.070] | 1.013 [0.991, 1.040] |

Source and reproduction paths are absolute:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/spatial.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/sobel_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/sobel_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/sobel_previous.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/sobel_tiled_experiment.wgsl`

```text
cargo test -p spatialrust-gpu --all-features --release --lib paired_previous_experimental_current_sobel -- --ignored --nocapture --test-threads=1
```

Private diagnostics remain outside version control under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`: final
`gpu-spatial-validation-paired.log`, `gpu-spatial-validation-estimates.json`
(all sample ranges and both comparison intervals), `gpu-spatial-validation-table.md`
and `summarize-gpu-spatial-validation.py`; the two earlier raw/estimate/table
sets use `tiled-sobel-early-selection-*` and `tiled-sobel-large-selection-*`.
Published artifacts include no local adapter model.
