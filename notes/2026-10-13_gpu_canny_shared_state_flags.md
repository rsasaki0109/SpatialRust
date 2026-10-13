# Shared GPU Canny classification/root flags

Date: 2026-10-13. Named slice: remove the separate root-strong storage buffer
while preserving resident Canny's exact gradients, classification and complete
weak-component hysteresis. Public options, metadata and explicit upload/readback
boundaries retain behavior.

## Execution and memory

A single atomic u32 per pixel stores classification in bits 0..1 (0 background,
1 weak, 2 strong) and a root-neighbor flag in bit 2. Classification overwrites the
entire word, clearing stale flags in recycled scratch. Weak unions still use the
same full-width u32 parent indices and minimum-root CAS/path-halving algorithm.
After unions complete, a weak pixel touching a strong pixel atomically ORs bit 2
into its component root's state word. Every classification read masks the low
bits, so simultaneous root marking cannot change weak/strong membership. Root
identities are stable after the connect pass; marker writes never alter low bits.
The final pass retains strong pixels directly and weak components whose root has
bit 2 set. No new convergence, propagation cutoff, pass or submission is added.

Removing binding 6 and its lease leaves four storage buffers instead of five:
gradients, horizontal rows, shared states/flags and parents. Existing binding
indices and max-bindings validation otherwise remain. Required storage-buffer
count falls to four; an artificial three-buffer limit still rejects before work.
No pixel-index bit is reserved, so accepted image/index capacity remains. The
largest binding is still 8*N bytes; controls remain 96 bytes. All private scratch
is exclusively leased through the single submission and every state word is
initialized before any later pass reads it.

| Aperture | Previous scratch | Shared scratch | Previous 4K bytes | Shared 4K bytes | Saved 4K bytes |
| --- | --- | --- | ---: | ---: | ---: |
| 3 | 20*N+8 | 16*N+8 | 165,888,008 | 132,710,408 | 33,177,600 |
| 5/7 | 24*N | 20*N | 199,065,600 | 165,888,000 | 33,177,600 |

N=width*height; 4K means 3840x2160. These are requested private buffer bytes,
excluding input/output textures, controls, driver overhead and pool retention.
Classification destination footprints fall from 12*N to 8*N; total recorded
stage destinations fall from 36*N/40*N to 32*N/36*N. This receipt describes
stage destination storage, not repeated in-place atomic byte traffic. Root
marking now changes the shared state binding; its parent/state footprint remains
8*N. Exact image upload and padded image readback bytes remain unchanged. Against
the initial PR #180 unpacked implementation, cumulative private savings are
99,532,800 bytes for aperture 3 and 132,710,400 bytes for apertures 5/7 at 4K.

## Correctness and validation

The complete PR #183 shader is frozen from
`a9c2d8a2d1529b44663af9a5a96c514c460546c9`. Its actual eight-binding/five-buffer
layout, separate root flags and original scratch/receipt boundaries compile and
execute natively. The PR #180 and #181 references retain their actual unpacked
and partially packed layouts. Frozen execution helpers are shared only in tests;
their standalone strong buffer is never acquired in production. Historical
compact-gradient and horizontal benchmarks explicitly compare frozen sources,
so this change cannot silently redefine their baselines.

- 864 threshold/aperture/norm/profile/strip conditions produce 367,056 exact CPU
  edge bytes, and agree with all three frozen GPU generations.
- 120 complete snapshots over degenerate/odd/grid-stride shapes and all apertures,
  both norms and four profiles compare gradients, horizontal sums, classifications,
  minimum roots, flags and edge images against actual frozen PR #183. All 433,320
  shared words equal old_classification | (old_root_flag << 2). The state scratch
  is deliberately filled with all-one words before classification; no upper
  poison bits or stale flags survive.
- Twelve synthetic masks with one or many strong seeds, separated components and
  unseeded components compare against frozen GPU and independent eight-neighbor
  BFS. Long strips and 257x129 masks span many workgroups; shared markers preserve
  both classification and exact minimum-root flags under contention.
- Existing direct f64 gradients/NMS/BFS references, long weak snakes/diagonals,
  326,401 signed rounding values, 65,536 component packing pairs, extrema, ROI,
  metadata, simultaneous live outputs, profiler receipts and pool-reuse/chain
  ownership tests pass. Four-buffer capability and exact scratch/capacity
  boundaries are checked without changing largest-binding limits.

Native Canny family: 13 passed, zero failed, four measurements ignored. GPU image
regression: 92 passed, zero failed, 17 measurements ignored. All-feature/all-target
GPU Clippy, feature-free GPU, GPU-image WASM, minimal gpu-image facade and formatting
checks pass. WASM was compile checked; browser execution was not measured.

## Completed-call measurement

Two isolated release surveys completed sequentially in 22.65 and 22.99 seconds.
No overlapping local build/test/benchmark was detected during accepted runs;
statistical analysis followed all surveys. Backend is Vulkan, device class
DiscreteGpu, and the global Rayon pool has 12 workers. No local adapter model
is recorded. Separate runtimes keep independent default bounded resource pools.
Each geometry starts with its scratch/texture retention cleared; each route
keeps its own normal 64 MiB scratch retention policy. Fixture generation, CPU
references, pipeline compilation/warmup and post-timer checks are excluded.
Three completed warmups precede twenty paired rounds, rotating old full, shared
full, old resident and shared resident calls.

Ten cases cover 256x192, 1920x1080 and 3840x2160, apertures 3/5/7, both norms,
deterministic noise, bar patterns and low-contrast geometry. Thresholds are
100/200. All calls use the same image geometry and exact specialized
CPU edge reference. Full calls include image upload, controls, scratch/output
acquisition, all passes, completed image readback/unpacking and recycling both
owned images. Resident calls use uploaded sources and time the complete public
call plus wait_idle; validation/readback/recycling occur outside the timer.
There is no intermediate image/label collection. Every measured output and
explicit transfer/destination receipt is checked.

Frozen historical gradient (PR #180/#181) and horizontal (PR #181/#183) surveys
were also run alone and passed in 22.64 and 22.74 seconds respectively. Those
measurements do not enter the new table.

## Paired results

Times are medians in milliseconds. Ratios are geometric means of twenty paired
old/shared times; values above one favor sharing. Intervals use 10,000 paired
log-ratio bootstrap draws, sorted entries 249/9749, with seed
20261013 + 2*record_index + comparison_index. These unadjusted intervals describe
within-survey variation, not repeat-run or hardware-wide bounds. All adverse
and tail samples are retained.

| Survey | Image/aperture/norm | Profile | Old full ms | Shared full ms | Old resident ms | Shared resident ms | Old/shared full (95%) | Old/shared resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 256x192/A5/L1 | noise | 1.165 | 1.211 | 0.364 | 0.344 | 0.978x (0.914-1.046) | 1.068x (0.972-1.186) |
| 1 | 256x192/A7/L2 | bars | 1.146 | 1.110 | 0.372 | 0.355 | 1.030x (1.012-1.046) | 1.055x (0.964-1.155) |
| 1 | 1920x1080/A3/L1 | bars | 8.894 | 8.797 | 1.131 | 1.122 | 1.013x (0.999-1.027) | 1.026x (0.968-1.090) |
| 1 | 1920x1080/A5/L1 | bars | 9.254 | 9.251 | 1.311 | 1.261 | 1.002x (0.982-1.020) | 1.023x (0.986-1.063) |
| 1 | 1920x1080/A5/L2 | noise | 9.502 | 9.585 | 1.389 | 1.398 | 0.999x (0.980-1.018) | 1.022x (0.989-1.057) |
| 1 | 1920x1080/A7/L2 | noise | 10.168 | 9.954 | 1.646 | 1.570 | 1.008x (0.919-1.096) | 1.033x (0.998-1.070) |
| 1 | 3840x2160/A3/L1 | bars | 34.803 | 35.103 | 10.318 | 8.652 | 0.964x (0.881-1.051) | 1.195x (1.142-1.263) |
| 1 | 3840x2160/A5/L2 | bars | 37.869 | 33.979 | 11.511 | 7.717 | 1.137x (1.049-1.228) | 1.504x (1.400-1.599) |
| 1 | 3840x2160/A7/L2 | noise | 39.721 | 36.885 | 15.485 | 9.678 | 1.080x (1.043-1.116) | 1.574x (1.508-1.651) |
| 1 | 3840x2160/A7/L2 | low contrast | 39.206 | 36.067 | 14.895 | 9.536 | 1.091x (1.028-1.160) | 1.500x (1.438-1.558) |
| 2 | 256x192/A5/L1 | noise | 1.121 | 1.098 | 0.363 | 0.352 | 0.990x (0.920-1.037) | 1.043x (0.933-1.160) |
| 2 | 256x192/A7/L2 | bars | 1.103 | 1.119 | 0.371 | 0.352 | 0.994x (0.970-1.020) | 1.075x (0.980-1.183) |
| 2 | 1920x1080/A3/L1 | bars | 8.829 | 8.775 | 1.073 | 0.988 | 1.025x (0.951-1.089) | 1.067x (1.007-1.138) |
| 2 | 1920x1080/A5/L1 | bars | 9.192 | 9.247 | 1.288 | 1.285 | 0.998x (0.986-1.012) | 1.031x (0.991-1.072) |
| 2 | 1920x1080/A5/L2 | noise | 9.354 | 9.266 | 1.372 | 1.295 | 1.000x (0.978-1.021) | 1.093x (0.983-1.283) |
| 2 | 1920x1080/A7/L2 | noise | 9.717 | 9.697 | 1.630 | 1.647 | 0.968x (0.906-1.020) | 1.020x (0.977-1.065) |
| 2 | 3840x2160/A3/L1 | bars | 34.978 | 35.394 | 10.340 | 8.737 | 0.949x (0.880-1.023) | 1.183x (1.109-1.280) |
| 2 | 3840x2160/A5/L2 | bars | 36.656 | 34.227 | 11.955 | 8.268 | 1.004x (0.896-1.113) | 1.456x (1.311-1.583) |
| 2 | 3840x2160/A7/L2 | noise | 39.881 | 36.633 | 15.908 | 10.363 | 1.115x (1.029-1.211) | 1.494x (1.439-1.547) |
| 2 | 3840x2160/A7/L2 | low contrast | 40.116 | 35.578 | 15.585 | 10.062 | 1.070x (0.989-1.153) | 1.517x (1.473-1.560) |

All eight 4K resident intervals exclude one. Aperture-3 bars give 1.195x and
1.183x; aperture-5 bars give 1.504x and 1.456x; aperture-7 noise gives 1.574x and
1.494x; aperture-7 low contrast gives 1.500x and 1.517x. These completed resident
calls include controls, encoding, scratch acquisition and device completion;
the ratios are not attributed solely to shader throughput or memory bandwidth.

Default bounded-pool retention matters. At 4K both routes retain 66,355,208 bytes
for aperture 3 and 66,355,200 for 5/7, under the unchanged 64 MiB policy. That
budget does not retain the additional full-size 33,177,600-byte pixel buffers.
Sharing removes one such live lease/allocation per call. Driver/resource behavior
is therefore part of the observed improvement; changing retention or device can
change timings. This is a private working-footprint reduction, not an assertion
that cached bytes or total process VRAM fall by the same amount.

4K full-call results are mixed. Aperture-7 noise repeatedly improves 1.080x and
1.115x, with intervals above one. Aperture-5 bars changes from 1.137x to 1.004x
(the latter is inconclusive). Low-contrast aperture 7 changes from 1.091x to
1.070x, with survey 2 inconclusive. Aperture-3 full estimates are adverse
(0.964x and 0.949x), but both intervals include one. Full HD and small resident
intervals mostly include one, so no consistent improvement is established there.
A lone Full HD aperture-3 resident interval and one small full interval exclude
one; their repeats do not. No blanket full-call or general-device speedup is
claimed. Memory saving remains exact across all accepted apertures.

Private raw/derived artifacts remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-shared-state-flags-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-shared-state-flags-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-shared-state-flags-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-canny-shared-state-flags.py`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/run-gpu-canny-shared-state-flags-survey.ps1`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-horizontal-historical-recheck-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-compact-historical-recheck-benchmark-1.log`

Source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_frozen.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_state_flags_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_state_flags_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/canny_state_flags_previous.wgsl`

From `C:/Users/rsasa/Workspace/SpatialRust`, compile the release binary with
--no-run first, then run this exact test alone twice sequentially:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib image::canny::tests::state_flags_tests::benchmark::completed_state_flag_calls -- --include-ignored --exact --nocapture --test-threads=1
```
