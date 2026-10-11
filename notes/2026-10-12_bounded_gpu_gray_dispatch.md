# Bounded GPU RGB-to-gray dispatch

Date: 2026-10-12. This named slice follows merged
[PR #145](https://github.com/rsasaki0109/SpatialRust/pull/145), main
`95fb8e854333272c866b2439727c5fe35c360f72`.

## Problem and resulting behavior

`rgb_to_gray_gpu` previously dispatched `ceil(width * height / 256)` workgroups
on the X axis. With the requested device limit of 65,535 groups per dimension,
the largest legal one-axis workload is 16,776,960 pixels. A valid 4096x4096
texture has 16,777,216 pixels and requires 65,536 groups, so the previous call
reached wgpu dispatch validation with an unsupported X count.

The operation now retains the original linear shader when the group count fits
one axis. Larger inputs use a balanced rectangle of flat 256-lane groups:

```text
pixels = u64(width) * u64(height)
groups = ceil(pixels / 256)
rows = ceil(groups / max_groups_per_dimension)
columns = ceil(groups / rows)
```

Both dispatch axes must fit the device limit. For 4096x4096 the plan is exactly
32,768 columns and two rows, with no padding group. Each group's flat index is
`row * columns + column`. The grid shader rejects padding groups before
multiplying their index by 256, then rejects tail lanes against the pixel count.
This order prevents a padding group at the u32 pixel boundary from wrapping
back to pixel zero. Fewer than `rows` groups are padding; no partial-result
array or second pass is introduced.

The pure planner checks positive dimensions, texture limits, signed texture
coordinates, a pixel count at most `u32::MAX`, the existing 256-lane workgroup,
both dispatch dimensions, sampled/storage textures, uniform/buffer sizes and
bind-group capacity. It requires no workgroup storage or storage buffer. Runtime
ownership, RGB channel count and one free profiler stage are checked before
texture/uniform allocation or cache initialization. Synthetic counter/limit
tests do not imply that such textures can be allocated on the local device.

Linear and grid compute pipelines are independently lazy in one runtime-owned
gray family. They share the original bind layout and 16-byte uniform: two
previously unused words now carry the group-row stride and total group count.
The original linear WGSL matches the frozen parent exactly. Every valid pixel
retains `(77*R + 150*G + 29*B + 128) >> 8`, gray metadata with the original color
range, `[gray, 0, 0, 0]` physical output and the `rgb_to_gray_gpu` receipt stage.
One pass still writes four GPU bytes per pixel; explicit timestamp collection
returns 16 query bytes. There is no implicit payload readback, host wait, global
scratch, new dependency, feature or public signature.

This expands the standalone gray operation's supported dispatch range. Other
kernels retain their own limits; full large-image Vision-chain admission is
not established here. Existing valid six-to-eight-pass Vision-chain regressions
continue to pass. The texture pool still bounds retained bytes, not active
source/output textures or peak driver/process memory.

## Correctness and feature validation

Six new native tests cover:

- **Every one of the 16,777,216 possible RGB values**, packed into the public
  4096x4096 grid path, checked against independent widened integer CPU luma.
  This checks **67,108,864 exact physical RGBA bytes**, including zero unused
  channels. The cold call initializes only the grid pipeline.
- Nine small/odd/singleton shapes crossed with mixed, zero and full-scale
  patterns. Frozen linear and current public outputs match the CPU reference;
  **120 forced grid mappings** under 2/3/7/17/31-group limits check tail/padding
  groups over **3,655,572 exact RGBA bytes** across variants.
- A GPU probe calls the production grid-index function for tiny plans and
  synthetic `u32::MAX`/`u32::MAX-1` pixel counts. **96 exact index/validity words**
  cover first/last/padding groups and lanes 0/1/254/255 without allocating giant
  textures. Returning before multiplication distinguishes padding from pixel zero.
- Pure dimension, texture, pixel-overflow, compute, binding and uniform-limit
  rejection. A two-by-two dispatch admits four groups and rejects five. Zero
  workgroup-storage/storage-buffer capacities remain acceptable.
- Wrong runtime/channel/profile ownership and exhausted capacity leave texture
  counters and cold gray caches unchanged. Rejection preserves an earlier Sobel
  stage and its 16-byte explicit query collection.
- A strided RGB ROI, four concurrent cold linear calls, live outputs while the
  original source texture is reused, rapid submissions, moved runtime, warm
  allocation reuse and zero texture-retention budget preserve exact results.
  The cold small path initializes only the linear pipeline.

The physical-channel test helper temporarily aliases the same texture with
unknown color-space metadata to allow explicit four-channel readback. It
allocates no texture and copies no texture payload; the public output retains
its original gray metadata and logical channel count. Test readback is explicit
and is outside production receipts and benchmark timers.

Final native GPU regressions pass: **148 tests**, zero failures, 11 ignored
diagnostics. This includes 121 unit tests and 27 CDF/CLAHE/equalization/histogram/
shared-stage integration tests, covering images and points. Strict all-feature
and default all-target Clippy, formatting and diff checks pass. Minimal
`gpu-image` and combined `gpu-image,gpu-aoso-staging` WASM compilation pass;
browser execution/performance is not established.

```text
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-image
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-image,gpu-aoso-staging
cargo fmt --all --check
git diff --check
```

## Resident timing diagnostic

Release native Windows, Rust/Cargo 1.98.1 and wgpu 24.0.5, Vulkan discrete
backend with pass timestamps. The adapter model is omitted. No tests or builds
launched by this task overlap measurement; desktop activity, clocks and external
GPU load are not controlled. This change expands dispatch support; no speedup
is claimed.

Seven shapes use two warmups and twenty alternating rounds. Six legal linear
shapes compare frozen and current kernels; the new large shape records only
current results, because its frozen X dispatch is unsupported. Each warmup and
measured result checks all physical output bytes against the CPU oracle,
metadata, the receipt, one pass and 16-byte explicit query collection.

Completed timers include runtime/plan/profiler validation, output/uniform
allocation, encoding, submission and `wait_idle`. Upload, warmed pipeline setup,
profiler creation, query collection, physical readback/checking and recycling
are outside timers. The frozen shader uses the current host executor and
validation, so this **does not measure historical full-API overhead**.
One runtime and its default 64 MiB texture-retention budget persist across all
conditions; the pool is not cleared between shapes. Outputs are recycled after
checking, so shape and retained-pool occupancy can affect allocation reuse.

Times below are variant medians. Ratios are medians of twenty paired
frozen/current samples, with 10,000 paired median bootstrap resamples
(seed 20261012, sorted indices 249 and 9749). Above one favors current. These
are descriptive intervals for this run, not independent-repeat confidence or
a portable performance guarantee. Every condition and sample range is retained.

| Size | Dispatch x/y | Frozen/current GPU us | GPU ratio [95% CI] | Frozen/current completed ms | Completed ratio [95% CI] |
| --- | --- | --- | --- | --- | --- |
| 17x19 | 2/1 | 4.224 / 4.272 | 0.981 [0.974, 1.023] | 0.185 / 0.186 | 0.969 [0.897, 1.015] |
| 128x128 | 64/1 | 4.544 / 4.544 | 1.000 [0.976, 1.025] | 0.180 / 0.184 | 1.001 [0.948, 1.039] |
| 640x480 | 1200/1 | 18.752 / 18.832 | 0.995 [0.989, 1.003] | 0.342 / 0.348 | 0.983 [0.944, 1.116] |
| 1920x1080 | 8100/1 | 94.512 / 94.624 | 1.000 [0.994, 1.004] | 0.605 / 0.585 | 1.014 [0.976, 1.057] |
| 3840x2160 | 32400/1 | 5279.440 / 5290.928 | 0.999 [0.996, 1.002] | 6.750 / 6.835 | 0.996 [0.983, 1.014] |
| 4096x4095 | 65520/1 | 12116.400 / 12109.872 | 1.001 [1.000, 1.002] | 23.599 / 23.741 | 0.993 [0.991, 1.001] |
| 4096x4096 | 32768/2 | N/A / 12098.672 | N/A | N/A / 23.633 | N/A |

Source and reproduction paths are absolute:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gray.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gray_grid.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gray_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/gray_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/gray_previous.wgsl`

```text
cargo test -p spatialrust-gpu --all-features --release --lib paired_frozen_current_and_large_gray -- --ignored --nocapture --test-threads=1
```

Private logs under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:
`bounded-gpu-gray-regressions.log`, `bounded-gpu-gray-clippy.log`,
`bounded-gpu-gray-default-clippy.log`, `bounded-gpu-gray-wasm.log`,
`bounded-gpu-gray-wasm-combined.log`, `bounded-gpu-gray-release-build.log`,
`bounded-gpu-gray-paired.log`, `bounded-gpu-gray-estimates.json`,
`bounded-gpu-gray-table.md` and `summarize-bounded-gpu-gray.py`.
