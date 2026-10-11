# Parallel GPU RANSAC plane scoring

Date: 2026-10-11. Repository: `rsasaki0109/SpatialRust`.
Slice: `GPU RANSAC plane scoring`; branch: `codex/parallel-gpu-ransac-plane`.
Frozen baseline: PR #134 source `d04974198307fa3db19fc34c3f10e78c56342d2b`.

## Implementation and ownership

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_plane.rs`
keeps the public scoring signature/result unchanged. Previously each hypothesis
occupied one invocation and serially traversed all points; the shader/pipeline
were constructed on every call and compute/score copy used two submissions.

From 1,024 points, a 256-lane workgroup constructs one model in lane zero,
classifies disjoint point-index residues and tree-reduces u32 inlier counts.
Plane cross/normalization/offset and each point's distance expression retain the
previous f32 operation order; there is no parallel floating sum or centroid
change. A private count-field marker distinguishes degenerate models, and is
replaced by the actual count in the returned score. The model-reuse barrier
prevents the next hypothesis from overwriting a model still being read.

Below 1,024 points the scalar point loop remains in a separate shader, avoiding
workgroup-reduction setup. Both variants dispatch at most 256 groups and stride
over hypotheses, bounded by the device group limit. Remaining-count guards
prevent u32 wrap when advancing point/hypothesis indices. The 1,024 cutoff is a
conservative private GPU scheduling choice; CPU/GPU placement and the
segmentation policy's existing point threshold are unchanged.

Production shaders:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/ransac_plane_serial.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/ransac_plane_parallel.wgsl`

The parallel kernel uses 1,024 bytes of shared counts plus a 20-byte model and
requires at least 2 KiB of device workgroup storage. There are no global partial
buffers or additional compute passes. Point/hypothesis counts must fit u32;
4*N axis buffers, 16*H hypotheses and 20*H output are checked against buffer and
binding limits, along with 256-lane, uniform and five-storage-buffer limits.
All indices must be within the non-empty XYZ arrays. Errors reject before
pipeline initialization or GPU allocation; equal-length empty hypotheses remain
a no-op. Negative/NaN threshold comparison behavior is preserved.

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/runtime.rs`
owns independently lazy serial/parallel pipelines through OnceLock. Only a used
variant compiles, and concurrent calls share its cache object. Compute and score
copy use one encoder/submission, followed by the existing explicit native map
and wait. XYZ uploads are returned to the bounded upload pool after completion
and bind-group release; output and staging stay call-owned. Existing transfer
accounting remains 12*N+16*H host-to-device and 20*H device-to-host bytes,
excluding the existing 16-byte parameter uniform. CPU plane refinement and
cloud extraction remain the high-level segmenter's existing work.

## Correctness and validation

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_plane_tests.rs`
compares both production variants with the frozen shader in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/ransac_plane_previous.wgsl`.
The fixture matches the baseline Rust string exactly after newline normalization.

There are 110 old/new comparisons, totaling **36,450 complete scores / 182,250
exact u32-word comparisons**, including all model float bits. They cover 1, 3,
255, 256, 257, 1,023, 1,024, 1,025, 8,193, 65,537 and 1,000,003 points;
hypothesis counts beyond the 256-group bound; forced single-group reuse;
axis-aligned, oblique/varied-index, reversed, duplicate and collinear hypotheses;
thresholds zero, 0.25, its next f32 neighbor, negative and NaN. Both variants are
forced at each size to test the algorithms independently of the cutoff.
Destination words start at u32::MAX to expose unwritten scores.

An independent CPU oracle counts controlled axis distances at, below and above
the threshold. Arbitrary oblique model comparisons use the previous GPU shader,
so this is a native legacy-compatibility check, not a claim of universal CPU/GPU
floating-point bit identity. Four cold-cache concurrent public calls share one
pipeline object, retain independent host results, and survive a runtime move.
Warmed XYZ buffers introduce no new pool allocations across repeated calls.
Input/index/limit rejection and empty-hypothesis no-op tests allocate nothing.

The all-feature GPU suite passes **116 tests** (92 unit plus 24 image/timing
integration tests), with four explicit performance diagnostics ignored by
default. Segmentation's default features plus `segment-ransac-plane-gpu` pass
**8 tests**, including CPU/GPU planar segmentation. After adding varied oblique
hypotheses and counter-overflow planning cases, the final RANSAC subset passes
all five tests again. Native all-feature/all-target and default Clippy pass with
warnings denied. Minimal-wgpu native tests compile, and minimal-wgpu plus combined
image/AoSoA WASM library Clippy pass. Formatting/whitespace checks pass.
WASM is compile-only; browser shader execution is unverified.

Logs under the absolute workspace root:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-regressions.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-final-proof.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-segmentation.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-default-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-minimal-native.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-wasm-minimal.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-wasm-combined.log`

## Paired native measurements

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_plane_benchmark.rs`
is an ignored, explicit diagnostic rather than a CI performance assertion.
Hardware: local GPU/Vulkan, NVIDIA driver 596.36, i7-9750H (6 cores/12 logical
CPUs), Windows, Rust/Cargo 1.98.1, wgpu 24.0.5.

Seventeen fixtures have three warmups and 20 alternating paired rounds on one
runtime, matching resident XYZ/hypothesis/uniform buffers and layout, independent
score destinations, and the production dispatch/variant selection. The previous
shader is also compiled once to isolate execution from per-call compilation.
Each variant gets a compute pass, submission and completed-work wait; pass queries
are collected after both timers, reading 32 query bytes per pair. Final scores
match the frozen GPU words and controlled CPU axis counts. GPU regressions and
segmentation completed before measurement; user processes were untouched.

GPU pass times exclude uploads, CPU setup/encoding, compilation, readback and
inter-pass gaps. Completion times include submission and wait only, with resident
inputs. Neither reports full public API latency or complete segmentation/frame
speedup; cache reuse and the merged compute/readback-copy submission are not
attributed to these speed ratios. Hardware/clock behavior can vary.

Raw pairs, observed ranges, medians, paired previous/current ratios and 95%
bootstrap intervals (10,000 draws, seed 20261011) are preserved. Ratios above one
favor the current implementation; a paired-ratio median need not equal the ratio
of separate medians. Small fixtures remain near parity with occasional measured
regressions; their intervals mostly include parity. No universal no-regression
or cross-adapter performance guarantee is claimed.

| Points | Hypotheses | Point parallel | Previous/current GPU median (us) | GPU paired ratio [95% CI] | Previous/current submit+wait median (ms) | Completion paired ratio [95% CI] |
|---:|---:|---|---:|---:|---:|---:|
| 100 | 1 | False | 6.320/6.304 | 1.021 [0.962,1.096] | 0.125/0.131 | 0.989 [0.913,1.060] |
| 100 | 16 | False | 6.144/6.320 | 0.961 [0.907,1.079] | 0.115/0.115 | 0.975 [0.917,1.018] |
| 100 | 513 | False | 6.592/7.008 | 0.952 [0.884,1.022] | 0.094/0.094 | 1.042 [0.987,1.067] |
| 1,023 | 1 | False | 35.968/35.792 | 1.007 [0.991,1.032] | 0.129/0.129 | 1.007 [0.972,1.013] |
| 1,023 | 16 | False | 35.920/35.312 | 1.024 [0.992,1.040] | 0.116/0.116 | 1.030 [0.998,1.043] |
| 1,023 | 513 | False | 36.816/36.128 | 1.017 [1.001,1.039] | 0.122/0.122 | 1.025 [0.966,1.039] |
| 1,024 | 1 | True | 35.888/4.368 | 8.249 [7.522,9.028] | 0.134/0.091 | 1.350 [1.258,1.457] |
| 1,024 | 16 | True | 36.112/4.464 | 8.086 [7.255,8.566] | 0.119/0.087 | 1.330 [1.298,1.404] |
| 1,024 | 513 | True | 36.576/13.552 | 2.693 [2.561,2.762] | 0.151/0.115 | 1.302 [1.149,1.466] |
| 8,193 | 1 | True | 265.344/8.672 | 30.635 [29.065,31.209] | 0.346/0.091 | 3.808 [3.644,3.886] |
| 8,193 | 16 | True | 265.968/9.264 | 28.729 [27.560,29.552] | 0.358/0.097 | 3.778 [3.636,3.820] |
| 8,193 | 513 | True | 318.048/38.832 | 8.195 [8.136,8.233] | 0.415/0.135 | 3.076 [2.920,3.160] |
| 65,537 | 1 | True | 2093.904/44.352 | 47.216 [46.838,47.826] | 2.205/0.152 | 14.700 [13.481,15.131] |
| 65,537 | 16 | True | 2094.304/46.224 | 45.304 [44.816,45.524] | 2.178/0.127 | 17.158 [15.515,17.605] |
| 65,537 | 513 | True | 2514.800/228.976 | 10.984 [10.956,11.026] | 2.666/0.375 | 7.097 [6.832,7.300] |
| 1,000,003 | 1 | True | 40313.888/1181.936 | 34.107 [34.056,34.188] | 40.515/1.360 | 29.803 [29.590,30.107] |
| 1,000,003 | 16 | True | 40348.944/1191.376 | 33.944 [29.744,34.148] | 40.553/1.383 | 29.313 [26.783,29.859] |

At 1,000,003 points, the final sweep reports GPU paired ratios of 34.107 and
33.944 for one and sixteen hypotheses; submit+wait ratios are 29.803 and 29.313.
At 65,537 points with 513 hypotheses the GPU ratio is 10.984 and completion
ratio 7.097. Below the cutoff, measured GPU ratios range roughly 0.95–1.02.
A preliminary sweep with only repeated seed models is retained separately;
final numbers above use varied oblique hypotheses and do not pool the sweeps.

Artifacts:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-paired.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-estimates.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-table.md`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-ransac-parallel.py`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-seed-models-paired.log`

## Reproduction and delivery

From `C:/Users/rsasa/Workspace/SpatialRust`, with Rust tools on PATH:

```powershell
cargo test -p spatialrust-gpu --all-features --lib --tests -- --test-threads=1
cargo test -p spatialrust-gpu --all-features --lib ransac_plane -- --test-threads=1 --nocapture
cargo test -p spatialrust-segmentation --features segment-ransac-plane-gpu --lib -- --test-threads=1
cargo test -p spatialrust-gpu --all-features --release --lib paired_previous_current_ransac_scores -- --ignored --nocapture
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo test -p spatialrust-gpu --no-default-features --features gpu-wgpu --lib --no-run
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-wgpu --target wasm32-unknown-unknown --lib -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown --lib -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Delivery uses one focused commit/PR based on frozen PR #134 source
`d04974198307fa3db19fc34c3f10e78c56342d2b`, targeting main after that dependency
merges. The guard at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-ci-merge.ps1`
freezes both heads and requires all 114 named checks to succeed, none pending or
failed, and MERGEABLE/CLEAN. It preserves user-edited PR text and local branches,
retargets only after the frozen parent merges, and never force-pushes. The guard's
all-green, ten rejection cases, retarget tests and live dry run precede monitoring.
Its receipt is
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-parallel-ci-merge.log`.

PR #133 merged at `f3e73886a6e454e646cf82bd2526c5066945ffb0`, and PR #134
merged at `c069a3652afad489256179b8d211772f61eaab02`. This slice leaves those frozen
source heads and unrelated PRs unchanged.
