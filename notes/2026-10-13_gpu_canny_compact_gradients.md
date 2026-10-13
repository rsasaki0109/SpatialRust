# Compact resident GPU Canny gradients

Date: 2026-10-13. Named slice: lossless private gradient packing for the existing
resident image Canny API. This follows PR #180 without changing its public
options, thresholds, five/six passes, metadata or explicit transfer contract.

This records PR #181. The later [horizontal-packing slice](2026-10-13_gpu_canny_horizontal_packing.md)
reduces 5/7 scratch further to 24*N; figures below retain the historical layout.
The gradient benchmark now compares frozen PR #180/#181 shaders so a later
production layout cannot silently alter that comparison.

## Representation and capacity

A u8 Sobel component is bounded by 255 times the positive derivative-weight
sum times the smoothing-weight sum. The maxima are 1020 (3), 12240 (5), and
163200 raw (7). Aperture 7 divides by 16 with signed ties-to-even before storage,
so its stored maximum is 10200. Each fits signed 16-bit, including negatives.
The shader now packs x in the low half and y in the high half of one u32.
Arithmetic sign extension reconstructs both components before directional NMS.
The separate u32 magnitude retains the full L1 or squared-L2 value. No arithmetic
or threshold precision is reduced; the previous padding word is eliminated.

| Device storage | Previous PR 180 | Compact layout |
| --- | ---: | ---: |
| Gradient binding | 16*N | 8*N |
| Largest storage binding | 16*N | 8*N |
| Private scratch, aperture 3 | 28*N+8 | 20*N+8 |
| Private scratch, apertures 5/7 | 36*N | 28*N |
| Stage destination footprints, aperture 3 | 44*N | 36*N |
| Stage destination footprints, apertures 5/7 | 52*N | 44*N |

N is the pixel count. The three state/parent/strong-flag buffers remain 4*N
each. The horizontal buffer remains eight unused bytes for aperture 3 or 8*N
for 5/7. Five storage bindings and the 96-byte control uniform remain required.
These figures exclude source/output textures, live caller images, control,
readback staging, retained pools and driver overhead. Destination footprints
are receipt accounting, not measured memory bandwidth or repeated atomic traffic.

At 3840x2160, the gradient falls from 132,710,400 to 66,355,200 bytes.
Private scratch falls from 232,243,208 to 165,888,008 bytes (3), or from
298,598,400 to 232,243,200 bytes (5/7). Each call saves 66,355,200 bytes
(63.28125 MiB): 28.57% of aperture-3 scratch, 22.22% of 5/7 scratch.
An individual 64 MiB storage binding can now accommodate a 4K image, subject to
all other checked device/index/texture limits and actual allocation capacity.
This is not a claim that all total device memory fits in 64 MiB.

Controls and explicit host/image bytes are unchanged: 4K upload and final
padded RGBA readback each transfer 33,177,600 bytes. No gradient, class or label
payload crosses to CPU in production. The same queue-ordered exclusive scratch
lease, bounded pooling, optional timestamp capacity and ownership checks remain.
No public precision option, CPU fallback, automatic placement or dependency is
added. CPU Canny and explicit gray/Gaussian preprocessing retain their behavior.

## Correctness evidence

The frozen comparison shader is byte-for-byte from commit
`d5f73a5b263bba9262e2bb3526943168088f02ab` (PR #180); it is compiled only by
native tests and explicit measurements. Its 16-byte gradient buffer is retained
in the test helper. Both layouts execute their actual GPU gradient, NMS and
weak-component graph passes; the old route is not a simulated CPU baseline.

- 864 small complete conditions compare CPU, frozen GPU and compact GPU edge
  images exactly: all 367,056 bytes per route, six geometries including strips,
  four profiles, three apertures, L1/L2 and six threshold pairs.
- Twenty-four owned snapshots compare reconstructed gradients, private states,
  minimum weak-component roots, strong-neighbor flags and edges against both
  the frozen shader and independent direct 2D binary64/BFS references. A forced
  single workgroup exercises strided execution. Snapshot copies precede scratch
  recycling and cannot read a later call's reused buffers.
- All 65,536 signed 16-bit values in each component position round-trip exactly
  on device. Different permuted y values and the independent packed-word check
  detect cross-component contamination, swapped halves and sign-extension loss.
- All 326,401 raw aperture-7 derivatives in [-163200,163200] retain exact signed
  ties-to-even rounding; the probe keeps the raw input in a full i32 word.
- Twenty-four extremal aperture/axis/sign/norm cases reach the positive/negative
  1020/12240/10200 bounds and match independent full gradient snapshots.
- Exact eight-byte binding boundaries, extra-row 4K rejection and fourteen
  reduced device limits validate planning before initialization. Existing long
  weak snakes, disconnected masks, strided ROI/metadata, concurrent retained
  outputs, dirty scratch reuse, profiling and Gaussian/Canny/AI chains pass.

The native GPU-image regression passed 87 tests (15 explicit measurement tests
ignored). All-feature/all-target GPU Clippy, facade chain smoke, feature-free
GPU, GPU-image WASM and workspace formatting checks passed.

## Completed-call measurement protocol

The paired release fixture compares the frozen 16-byte shader with the compact
8-byte shader. It uses separate runtimes/pools on the same backend so one
variant cannot consume the other's 64 MiB scratch/texture retention budget.
Each case clears both pools before warmup to remove other geometries, then uses
three completed warmups and twenty paired rounds with rotating four-stage order:
previous full, compact full, previous resident, compact resident. Default pool
budgets and normal within-case retention stay enabled; allocation/recycling
costs are part of the completed call. Pipeline construction, CPU expected output
and fixtures are outside timing. Calls execute sequentially, never concurrently.

Full calls include upload/packing, controls, scratch/output acquisition, every
GPU pass/submission, completed readback/unpacking, receipt copy and recycling.
Resident calls start from a previously uploaded image and time controls,
acquisition, every pass/submission and explicit wait_idle completion. They
exclude image upload, final readback and recycling; validation/readback follows
immediately outside the timer before the next call. Every result matches CPU
edge bytes/metadata. Full receipts assert physical upload/D2H and each layout's
destination footprints; resident receipts enforce zero intermediate image D2H.

Two isolated surveys completed sequentially in 28.13 and 28.60 seconds, with
no overlapping local compilation, tests, benchmark or statistical analysis.
The native backend is Vulkan, device class DiscreteGpu, with 12 default Rayon
workers; no adapter model or worker override is recorded. An unrelated local
build was allowed to finish before either survey began.

## Paired results

The table gives medians in milliseconds. Ratios are geometric means of twenty
paired previous/compact times; values above one favor compact. Parenthesized
intervals use 10,000 paired log-ratio bootstrap draws, sorted entries 249/9749,
with seed 20261013 + 2*record_index + comparison_index. They describe
within-survey variation, not repeat-run variation or hardware-wide bounds.
All samples, including slow tails and regressions, remain in the analysis.

| Survey | Image/aperture/norm | Profile | Old full ms | Compact full ms | Old resident ms | Compact resident ms | Old/compact full (95%) | Old/compact resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 256x192/A3/L1 | bars | 1.587 | 1.649 | 0.472 | 0.422 | 0.99x (0.96-1.02) | 1.10x (1.01-1.19) |
| 1 | 256x192/A7/L2 | noise | 1.678 | 1.690 | 0.469 | 0.458 | 1.02x (0.98-1.05) | 1.03x (0.96-1.10) |
| 1 | 1920x1080/A3/L1 | bars | 13.049 | 13.273 | 1.311 | 1.446 | 0.98x (0.94-1.03) | 1.09x (0.93-1.26) |
| 1 | 1920x1080/A3/L2 | noise | 14.210 | 14.118 | 1.357 | 1.777 | 1.00x (0.95-1.05) | 0.94x (0.85-1.05) |
| 1 | 1920x1080/A5/L1 | bars | 11.852 | 12.151 | 1.414 | 1.266 | 0.98x (0.94-1.02) | 1.08x (0.96-1.22) |
| 1 | 1920x1080/A7/L2 | noise | 14.037 | 14.218 | 1.747 | 1.788 | 0.99x (0.95-1.03) | 1.05x (0.91-1.20) |
| 1 | 3840x2160/A3/L1 | bars | 51.166 | 47.790 | 14.523 | 9.201 | 1.08x (1.01-1.16) | 1.81x (1.58-2.08) |
| 1 | 3840x2160/A3/L2 | noise | 51.968 | 47.329 | 14.399 | 7.240 | 1.12x (1.04-1.20) | 1.84x (1.61-2.10) |
| 1 | 3840x2160/A5/L2 | bars | 53.734 | 48.632 | 15.297 | 12.499 | 1.07x (1.03-1.12) | 1.22x (1.10-1.37) |
| 1 | 3840x2160/A7/L2 | low contrast | 55.551 | 53.181 | 17.371 | 13.261 | 1.07x (1.02-1.12) | 1.28x (1.15-1.42) |
| 2 | 256x192/A3/L1 | bars | 1.953 | 2.054 | 0.538 | 0.508 | 0.98x (0.95-1.02) | 1.06x (0.96-1.17) |
| 2 | 256x192/A7/L2 | noise | 1.898 | 1.912 | 0.545 | 0.526 | 1.01x (0.98-1.04) | 1.08x (1.01-1.15) |
| 2 | 1920x1080/A3/L1 | bars | 15.431 | 15.081 | 1.338 | 1.110 | 1.01x (0.96-1.06) | 1.29x (1.13-1.47) |
| 2 | 1920x1080/A3/L2 | noise | 14.221 | 14.589 | 1.237 | 1.215 | 1.00x (0.95-1.05) | 1.00x (0.87-1.15) |
| 2 | 1920x1080/A5/L1 | bars | 13.834 | 13.231 | 1.422 | 1.306 | 1.04x (0.99-1.11) | 1.09x (1.01-1.20) |
| 2 | 1920x1080/A7/L2 | noise | 16.196 | 16.239 | 1.554 | 1.417 | 0.99x (0.94-1.04) | 1.09x (0.99-1.21) |
| 2 | 3840x2160/A3/L1 | bars | 50.535 | 46.469 | 9.765 | 6.662 | 1.08x (1.03-1.13) | 1.49x (1.34-1.66) |
| 2 | 3840x2160/A3/L2 | noise | 50.203 | 46.756 | 12.205 | 8.366 | 1.09x (1.06-1.13) | 1.49x (1.46-1.52) |
| 2 | 3840x2160/A5/L2 | bars | 52.517 | 49.147 | 19.460 | 15.437 | 1.07x (1.03-1.10) | 1.26x (1.19-1.34) |
| 2 | 3840x2160/A7/L2 | low contrast | 53.518 | 50.659 | 18.595 | 14.738 | 1.04x (1.02-1.07) | 1.26x (1.21-1.30) |

Every 4K comparison improves in both surveys: resident completed-call ratios
are 1.22-1.84x, and full host-image ratios are 1.04-1.12x. Aperture-3 resident
comparisons give 1.49-1.84x; larger apertures give 1.22-1.28x. All eight 4K
resident and all eight full intervals exclude one in this measured setup.
The exact memory saving holds independently of observed timing.

Small/1080p full-call ratios are 0.98-1.04x and all intervals include one.
Resident ratios are 0.94-1.29x; many intervals include one, and the first
1080p noisy aperture-3 compact median is slower (1.777 vs 1.357 ms).
These observations do not establish a repeatable speed gain at smaller sizes.
The optimization does not establish CPU/GPU placement thresholds; the
[earlier CPU comparison](2026-10-13_gpu_canny.md) remains historical and used a
different pool history. No new direct CPU speed claim is inferred from it.

The old 4K gradient exceeds the default 64 MiB per-pool retention limit; the
compact 66,355,200-byte gradient fits. In isolated aperture-3 cases the old
scratch pool retains two 33,177,600-byte state buffers plus the 8-byte unused
horizontal binding, while compact retains the gradient plus that binding.
For 5/7, old retains the horizontal buffer and compact retains the gradient.
Both layouts retain approximately 63.28 MiB but allocate different scratch
buffers each call. This measurement includes those normal acquisition costs;
it does not isolate shader speed or claim the same gain under a larger/custom
retention budget, cross-geometry cache churn or many simultaneous live calls.

Private raw and derived data remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-compact-gradients-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-compact-gradients-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-compact-gradients-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-canny-compact-gradients.py`

Source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_compact_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/canny_unpacked_previous.wgsl`

Reproduce from `C:/Users/rsasa/Workspace/SpatialRust`: build the release test
binary with --no-run first, then run this exact ignored test alone twice,
sequentially, with no overlapping local builds/tests/benchmarks:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib image::canny::tests::compact_benchmark::completed_compact_gradient_calls -- --include-ignored --exact --nocapture --test-threads=1
```
