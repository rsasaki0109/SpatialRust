# Lossless GPU Canny horizontal packing

Date: 2026-10-13. Named slice: exact unrounded horizontal derivative/smoothing
pairs in the existing resident 5/7-aperture image Canny implementation.

This records PR #183 with separate classification/root flags. The later
[shared-state implementation](2026-10-13_gpu_canny_shared_state_flags.md) removes
that flag allocation. The historical horizontal benchmark now explicitly runs
frozen PR #181 and PR #183 layouts, preserving this comparison independently of
current production code.

## Bounds and execution

For aperture 5, the horizontal derivative's positive coefficient sum is 3 and
the smoothing coefficient sum is 16. Applied to u8 data, derivative magnitude
is at most 765 and smoothing at most 4080. For aperture 7 those sums are 10 and
64, giving bounds 2550 and 16320. Negative derivatives have the same bound;
smoothing is nonnegative. Every value fits signed 16-bit without clipping.

The horizontal pass now writes one u32: derivative in the low half, smoothing
in the high half. The existing signed packing/sign-extension functions recover
both integers in the vertical pass before multiplication/accumulation. No
horizontal scale or rounding is introduced. Aperture 7 retains its final signed
1/16 ties-to-even rounding; gradient magnitude/direction, NMS, thresholds and
complete weak-component hysteresis are unchanged. No shader floating math,
native packed type or additional device capability is required.

| Footprint for N pixels | PR 181 | Horizontal packing |
| --- | ---: | ---: |
| Gradient binding | 8*N | 8*N |
| Horizontal binding, apertures 5/7 | 8*N | 4*N |
| Private scratch, apertures 5/7 | 28*N | 24*N |
| Private scratch, aperture 3 | 20*N+8 | 20*N+8 |
| Destination footprints, apertures 5/7 | 44*N | 40*N |
| Destination footprints, aperture 3 | 36*N | 36*N |

A 3840x2160 5/7 call removes exactly 33,177,600 bytes (31.640625 MiB),
reducing private scratch from 232,243,200 to 199,065,600 bytes (14.29%).
Compared with original PR #180, combined gradient/horizontal packing reduces
scratch from 36*N to 24*N, removing 99,532,800 bytes at 4K (one third).
These figures exclude source/output textures, controls, readback staging,
retained pools, other live calls and driver overhead; they are not measured
process/device peak allocations. The largest storage binding remains 8*N,
so the individual binding's maximum image size does not increase in this slice.

Aperture 3 keeps its original eight-byte unused horizontal binding, direct
integer Sobel and five passes. Apertures 5/7 keep six passes and one submission.
Source/output ownership, exclusive scratch leases, bounded pool retention,
96-byte controls, optional pass timestamps and pre-allocation validation remain.
4K image upload and final padded RGBA readback each still transfer 33,177,600
bytes. No gradient/horizontal/state/label payload crosses to CPU in production.
Stage destination footprints reflect the new buffer but do not measure bandwidth
or repeated in-place atomic traffic. Public/CPU APIs, core and dependencies remain.

## Correctness evidence

The PR #181 shader is frozen verbatim from
`dbf35de22e583f9f08a962c5fe9439992e54e07a` in a native-test-only fixture.
Its compact eight-byte gradients and original eight-byte horizontal pairs execute
actual GPU passes; the PR #180 sixteen-byte gradient fixture is also retained.
Both baseline host helpers allocate their original private layouts explicitly.

- 864 conditions compare CPU and all three GPU layouts exactly: all 367,056
  edge bytes per route, six geometries including tiny/narrow strips, four
  constant/noise/bar/low-contrast profiles, 3/5/7 apertures, L1/L2 and six
  fractional/exact/reversed/maximum threshold pairs.
- Twenty-four owned snapshots compare horizontal integer pairs and complete
  gradients, class states, minimum weak roots, strong flags and final edges
  against both frozen shaders. Horizontal references use independent 1D
  binomial formulas and replicated borders; full gradients use independent
  direct 2D binary64 convolution and graph outputs use independent BFS.
  One forced workgroup exercises grid strides. All private copies are encoded
  before scratch recycling, so later calls cannot overwrite observed data.
- Twelve extremal 5/7 sign/constant/norm fixtures reach derivative +/-765/2550
  and smoothing 4080/16320 and agree with both frozen horizontal layouts and
  independent integer references. The existing exhaustive 65,536 signed pairs
  verify packing/sign extension and bit layout; 326,401 raw aperture-7 values
  preserve final ties-to-even scaling.
- Exact 4K horizontal/scratch sizes and unchanged eight-byte gradient binding
  boundaries are checked. Existing reduced-limit, ownership, ROI/metadata,
  concurrent retained-output, dirty-scratch, timestamp and Gaussian/Canny/AI
  chain tests continue to pass, as do long/broken weak snakes and BFS masks.

Native GPU-image regression: 89 passed, 16 explicit measurement fixtures ignored.
The updated frozen PR #180/#181 gradient benchmark was explicitly executed once
and passed all complete outputs/receipts; its historical storage formulas stay
unchanged. All-feature/all-target GPU Clippy, facade gray/Gaussian/Canny/AI smoke, feature-free
GPU and GPU-image WASM builds pass. WASM is compile checked, not browser executed.

## Completed-call protocol

The paired release measurement compares the actual frozen PR #181 shader with
production horizontal packing. Each case clears both runtimes' scratch/texture
pools before warmup, preventing other image geometries from consuming retention.
Runtimes keep separate default 64 MiB pool budgets; the calls execute sequentially
on the same backend. Three completed warmups and 20 paired rounds rotate through
old full, packed full, old resident and packed resident.

Full calls include upload/packing, controls, scratch/output acquisition, every
GPU pass/submission, complete image readback/unpacking, receipt copy and recycling.
Resident calls begin with uploaded images, include controls/acquisition/passes/
submission and explicit wait_idle completion, and exclude upload, readback and
recycling. Every resident output is downloaded and checked outside timing before
the next call. CPU expected image generation, construction/lazy compilation and
post-timer validation are excluded. All edge bytes/metadata and full physical
H2D/D2H/destination receipts are asserted; resident receipts enforce zero
intermediate image readback.

Ten cases cover 256x192/1920x1080/3840x2160, noise/bars/low contrast, apertures
5/7 and both norms, plus unchanged aperture-3 controls at 1080p and 4K.
Measurements wait for a quiet local build/benchmark window. Light process checks
verify isolation during each survey; an overlapping attempt is excluded in full
and repeated, without trimming samples within any accepted survey. Statistical
analysis runs only after both surveys finish. Results are recorded below.


## Measured results

Both surveys passed in 23.32/22.36 seconds with ten conditions each; no overlap was detected
and no attempt was excluded. The backend was Vulkan, device class DiscreteGpu,
with 12 default Rayon workers and no worker override. No local adapter model is
recorded. An unrelated local benchmark was allowed to finish before measurement.
Minimal process observation was the only local monitoring during accepted runs;
no overlapping build/test/benchmark or statistical analysis ran.

Times are medians in milliseconds. Ratios are geometric means of twenty paired
previous/packed call times, favoring packing above one. Intervals use 10,000
paired log-ratio bootstrap draws, sorted entries 249/9749, seeded with
20261013 + 2*record_index + comparison_index. They describe within-survey
variation, not repeat-run or hardware-wide bounds. All adverse/tail samples in
accepted runs are retained.

| Survey | Image/aperture/norm | Profile | Old full ms | Packed full ms | Old resident ms | Packed resident ms | Old/packed full (95%) | Old/packed resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 256x192/A5/L1 | noise | 1.179 | 1.179 | 0.377 | 0.355 | 1.01x (0.97-1.05) | 1.06x (0.94-1.18) |
| 1 | 256x192/A7/L2 | bars | 1.131 | 1.139 | 0.380 | 0.362 | 1.00x (0.97-1.03) | 1.07x (0.98-1.17) |
| 1 | 1920x1080/A3/L1 | bars | 8.844 | 8.811 | 1.172 | 1.197 | 1.00x (0.99-1.01) | 1.00x (0.95-1.06) |
| 1 | 1920x1080/A5/L1 | bars | 9.401 | 9.286 | 1.421 | 1.335 | 1.01x (0.99-1.02) | 1.09x (1.05-1.14) |
| 1 | 1920x1080/A5/L2 | noise | 9.407 | 9.454 | 1.493 | 1.412 | 1.00x (0.99-1.02) | 1.09x (1.04-1.13) |
| 1 | 1920x1080/A7/L2 | noise | 9.927 | 9.834 | 2.100 | 1.966 | 1.01x (0.99-1.03) | 1.10x (1.06-1.14) |
| 1 | 3840x2160/A3/L1 | bars | 34.530 | 35.205 | 10.639 | 9.447 | 0.94x (0.87-0.99) | 1.11x (1.08-1.14) |
| 1 | 3840x2160/A5/L2 | bars | 36.282 | 38.235 | 12.539 | 10.838 | 0.89x (0.81-0.98) | 1.14x (1.11-1.18) |
| 1 | 3840x2160/A7/L2 | noise | 38.495 | 38.306 | 14.209 | 11.875 | 0.98x (0.88-1.08) | 1.18x (1.15-1.20) |
| 1 | 3840x2160/A7/L2 | low contrast | 38.145 | 36.956 | 13.617 | 11.658 | 1.03x (0.93-1.13) | 1.16x (1.14-1.19) |
| 2 | 256x192/A5/L1 | noise | 1.148 | 1.167 | 0.394 | 0.373 | 1.01x (0.97-1.05) | 1.07x (0.96-1.19) |
| 2 | 256x192/A7/L2 | bars | 1.134 | 1.117 | 0.372 | 0.363 | 1.01x (0.99-1.04) | 1.04x (0.95-1.13) |
| 2 | 1920x1080/A3/L1 | bars | 8.639 | 8.644 | 1.075 | 1.039 | 0.98x (0.90-1.06) | 1.03x (0.96-1.10) |
| 2 | 1920x1080/A5/L1 | bars | 8.977 | 8.880 | 1.300 | 1.177 | 1.02x (0.99-1.06) | 1.09x (1.06-1.12) |
| 2 | 1920x1080/A5/L2 | noise | 9.439 | 9.156 | 1.502 | 1.387 | 1.02x (1.01-1.04) | 1.10x (1.05-1.14) |
| 2 | 1920x1080/A7/L2 | noise | 9.869 | 9.598 | 2.006 | 1.778 | 1.01x (0.97-1.04) | 1.14x (1.09-1.18) |
| 2 | 3840x2160/A3/L1 | bars | 33.573 | 33.424 | 8.434 | 8.472 | 0.97x (0.91-1.03) | 1.00x (0.98-1.03) |
| 2 | 3840x2160/A5/L2 | bars | 37.704 | 37.998 | 13.628 | 12.459 | 0.98x (0.94-1.01) | 1.09x (1.06-1.11) |
| 2 | 3840x2160/A7/L2 | noise | 37.606 | 36.985 | 13.626 | 11.740 | 1.05x (0.97-1.14) | 1.11x (1.07-1.14) |
| 2 | 3840x2160/A7/L2 | low contrast | 39.243 | 38.632 | 15.053 | 12.814 | 1.06x (0.97-1.16) | 1.14x (1.10-1.18) |

For all measured 1080p/4K 5/7 conditions, resident ratios are 1.09-1.18x across
both surveys, with intervals above one. Small 256x192 resident ratios are
1.04-1.07x and all intervals include one. Full upload/readback ratios for 5/7
are 0.89-1.06x: the first 4K aperture-5 bar case regresses (0.89x, interval below
one), while several other intervals include one. This does not establish a
consistent round-trip latency gain. The exact 4*N memory saving is independent
of observed speed.

The unchanged aperture-3 control also varies: first-survey 4K gives 0.94x full
and 1.11x resident despite unchanged algorithm/storage; the second gives
0.97x/1.00x. These controls demonstrate completion/allocation/scheduling variation
and limit attribution of every ratio to horizontal shader work. Pool behavior
and normal acquisition/completion are included, not isolated pass throughput.
In particular, both 4K layouts retain the same 66,355,200-byte gradient under
the 64 MiB default scratch budget; neither retains its horizontal/state buffers.
The smaller horizontal buffer changes the amount allocated each call, while
its pool key now matches state-buffer sizes. No CPU/GPU placement threshold,
custom-budget/concurrent-call speedup or universal end-to-end gain is claimed.

Private raw/derived data remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-horizontal-packing-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-horizontal-packing-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-horizontal-packing-isolation.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-horizontal-packing-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-canny-horizontal-packing.py`

Source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_horizontal_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_horizontal_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/canny_horizontal_previous.wgsl`

From `C:/Users/rsasa/Workspace/SpatialRust`, compile the release test binary with
--no-run first, then run this exact measurement alone twice sequentially:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib image::canny::tests::horizontal_tests::benchmark::completed_horizontal_packing_calls -- --include-ignored --exact --nocapture --test-threads=1
```
