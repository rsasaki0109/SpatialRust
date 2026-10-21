# Confidence-mask resize (146AT)

The optional camera worker keeps the existing personMask signature and exact
binary output. For different source/output widths, a temporary Uint16 array
computes floor(x*sourceWidth/width) once per column, reusing it across rows.
For equal widths, the exact source column is x and no index array is needed.
Vertical sampling retains floor(y*sourceHeight/height); exclusion still uses
strict confidence >0.5.

Shape/type admission and validation of every source float retain their order,
bounds and error text, including invalid pixels that resizing never samples.
All axes remain in 1..320; every mapped column is 0..319 and fits Uint16.
The temporary horizontal indices need at most 640 logical bytes. No index
cache is retained, and each result remains a fresh independently transferable
Uint8Array. These logical sizes exclude output, input, runtime and allocator
memory and do not guarantee browser RSS.

The worker protocol, model/SDK, explicit confidence/image CPU readbacks, gray
conversion, Rust mask-support tracker, public reports and dependencies remain.
This is CPU indexing inside the existing optional demo, not a GPU kernel.

## Correctness and ownership

The [source-bound receipt](receipts/2026-10-22_camera_mask_resize.json) records
92 passing JS controls on both parent and candidate, including eight new tests:
42 shape/pattern cases, all 102,400 horizontal size pairs, small single-pixel
supports, exact/adjacent Float32 threshold values, unsampled invalid floats,
shape/type priority, caller independence and transfer/detachment ownership.
The independent frozen parent predicate remains unchanged between the tests.

All six asset checks and 86 actual WASM/model browser controls pass. Four new
browser checks load the real module, compare 42 complete masks, reject five
unsampled invalid confidences and transfer output buffers without aliasing or
detaching subsequent results. Existing lifecycle, rotation, deadlines, restart,
timing, model inference and CSP coverage remains. Captures are synthetic;
physical phone and iOS Safari behavior are unqualified.

No Rust compilation is needed. The site reuses the current tested tracker WASM,
preserving the previous shared original WASM as well. Exact-head CI, merge and
the matching public Pages/browser checks are verified after the local receipt.

## Interleaved mask-only timing

The original protocol is
C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-mask-resize-plan.json.
Its four-process observations reversed with order in several conditions; all
are retained below. A prospective follow-up protocol is
C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-mask-resize-pair-plan.json.
Two fresh Node processes each warm both functions with 1024 checked calls per
shape, then run eight baseline/candidate/candidate/baseline rounds. Every sample
contains 16 calls; the tables show per-call medians from 16 samples per function.
All timed and warmup output bytes match the frozen parent outside the timer,
and source confidence bytes remain unchanged.

These are local Windows Node observations. Affinity, CPU frequency, host load
and garbage collection are uncontrolled, with no confidence intervals. Ratios
below 1 favor the candidate. Neither survey proves physical-phone, GPU,
AI-inference or end-to-end camera speed. All conditions, including slower
single-row/downsample observations, remain in the tables and raw receipt.

| Source -> output | Parent 1 us/call | New 1 us/call | New/parent 1 | Parent 2 us/call | New 2 us/call | New/parent 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256 -> 192x144 | 1063.378 | 924.891 | 0.870 | 1310.372 | 1067.575 | 0.815 |
| 256x256 -> 256x192 | 1730.441 | 1499.194 | 0.866 | 1640.900 | 1327.050 | 0.809 |
| 256x256 -> 144x192 | 1312.112 | 1183.625 | 0.902 | 1250.119 | 1083.906 | 0.867 |
| 256x256 -> 192x256 | 1607.606 | 1276.316 | 0.794 | 1740.150 | 1346.703 | 0.774 |
| 320x320 -> 320x320 | 3149.703 | 2472.656 | 0.785 | 2769.863 | 2069.425 | 0.747 |
| 320x320 -> 64x64 | 1440.091 | 1498.372 | 1.040 | 1411.537 | 1408.181 | 0.998 |
| 64x64 -> 320x320 | 880.716 | 455.163 | 0.517 | 954.619 | 491.800 | 0.515 |
| 3x2 -> 5x7 | 0.450 | 0.406 | 0.903 | 0.481 | 0.431 | 0.896 |
| 320x1 -> 192x1 | 6.909 | 7.312 | 1.058 | 3.744 | 4.787 | 1.279 |
| 1x320 -> 1x320 | 11.059 | 8.606 | 0.778 | 12.059 | 10.641 | 0.882 |

Node v24.19.0, V8 13.6.233.17-node.51. Of 20 paired medians, 17 are lower and 3 higher for the candidate.

## Retained initial four-process observations

The initial protocol uses 128 checked warmups per shape, then seven samples
of 16 calls, in separate parent/candidate/candidate/parent processes. The
per-call times and ratios are preserved without pooling with the follow-up.

| Source -> output | Parent 1 us/call | New 1 us/call | New/parent 1 | Parent 2 us/call | New 2 us/call | New/parent 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256 -> 192x144 | 1469.775 | 1214.606 | 0.826 | 1875.438 | 2703.850 | 1.442 |
| 256x256 -> 256x192 | 1762.394 | 1669.144 | 0.947 | 2705.156 | 2214.975 | 0.819 |
| 256x256 -> 144x192 | 1492.806 | 1429.838 | 0.958 | 1776.244 | 2640.588 | 1.487 |
| 256x256 -> 192x256 | 1690.631 | 1655.125 | 0.979 | 2028.463 | 2658.369 | 1.311 |
| 320x320 -> 320x320 | 2985.669 | 2455.431 | 0.822 | 10847.000 | 3123.900 | 0.288 |
| 320x320 -> 64x64 | 1384.394 | 1631.781 | 1.179 | 2875.300 | 2455.619 | 0.854 |
| 64x64 -> 320x320 | 1455.081 | 520.337 | 0.358 | 1120.331 | 1094.475 | 0.977 |
| 3x2 -> 5x7 | 0.719 | 0.369 | 0.513 | 1.350 | 0.744 | 0.551 |
| 320x1 -> 192x1 | 8.925 | 4.944 | 0.554 | 5.231 | 5.844 | 1.117 |
| 1x320 -> 1x320 | 10.319 | 6.906 | 0.669 | 13.250 | 16.038 | 1.210 |

Both public benchmark drivers accept --module and --reference paths:
scripts/benchmark_camera_mask_resize.mjs and
scripts/benchmark_camera_mask_resize_pair.mjs. The reference is the frozen
parent person-mask.mjs. Their inputs, clocks, checks and raw samples are bound
to the receipt's command/source hashes.

The first follow-up launcher rejected an ambiguous source substitution before
Node started. Preserve its script and failure transcript; a fresh wrapper targets
the unique source-list location. Both subsequent owned measurements completed.
This was a diagnostic-launch failure, not an application regression.

## Storage and remaining qualification

The owned E-drive root has a prospective 128MiB limit, zero permitted shared
compiler-cache growth, one owned command at a time and a 900-second deadline.
Nine vendor hardlinks reuse 41,087,603 bytes. There is no sensor, model, SDK,
browser or toolchain acquisition. Preserve all 2,354 prior camera/renderer files,
including the completed isolated Cargo target, 493 scientific JSON, 42 frozen
native artifacts, room/reference outputs and resolved lockfile.

Public verification has a 40MiB streaming limit and retains no downloaded vendor
payload files. Physical phones/Safari, calibrated camera timing, independent
motion, metric/full-room accuracy and operational 80% qualification remain open.
