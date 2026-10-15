# Packed Q8 Gaussian intermediate sums

## Scope and numeric bound

`gaussian_blur_gpu` and its profiled companion now store pairs of horizontal
Q8 component sums in one u32 word for two-to-four logical channels. This is
selected only when both axes use 3/5 taps. Single-channel Q8 and every call
with either 7-tap axis retain the exact previous Q8/Q15 shaders.

The original host quantization normalizes the nonnegative Q8 weights to
256. Each source or constant-border component is a byte, so its horizontal
sum is at most 255*256 = 65280, which fits sixteen bits without rounding.
For finite normalization, the largest central weight is at least 256/5
before rounding and the sum correction is at most two units, so center
correction cannot make a weight negative. Degenerate tiny-sigma normalization
keeps the existing NaN-to-integer/center-correction behavior: a central 256
and zero elsewhere. The original weights and arithmetic are unchanged.
Tests cover 8,212 sigma/mass/bound conditions, including
subnormal/very small/very large sigmas and a broad logarithmic sweep.

One invocation owns every packed word for its pixel, with no atomics or
cross-pixel sharing. Odd channel counts write zero in the unused high half.
Vertical sampling extracts the appropriate halfword before executing the
same weight product, sum and final half-up rounding. Constant-border samples
are formed exactly as before. All original coordinate mapping, loops,
grid-stride bounds and output padding remain.

## Allocation, work receipts and placement

For P pixels and C channels, Q8 scratch requests change from 4*P*C bytes to
4*P*ceil(C/2) bytes. Two/four-channel requests fall by 50%; three-channel
requests fall from 12*P to 8*P bytes (one third less). The one-channel and
Q15 requests remain unchanged. These are logical scratch requests; pool
alignment, retained cache capacity, driver allocation and peak/RSS are not
measured. Smaller requests do not promise an equal peak-memory reduction.

The original planner still validates the full original scratch extent,
including device storage limits, before selecting the smaller request.
Original accepted/rejected calls, error strings and validation order remain;
this change does not broaden device acceptance. Reserved uniform word
padding0 carries the packed words per pixel only for the new kernels, with
the same 128-byte layout. Other paths keep all padding words zero.

Two independent lazy per-runtime variants initialize only the selected old
or packed pair; a packed-only caller does not compile the old pair. Both
weighted passes retain one submission, the same scratch lease/recycle
boundary, stage names and optional two-stage timestamps. Work receipts now
count the actual smaller horizontal output plus the unchanged 4*P RGBA8
texture output. Host uploads, downloads, source/output ownership, metadata,
texture format and explicit placement remain. No extra pass, transfer,
readback, host wait, CPU fallback or unsafe code is introduced.

## Correctness and source audit

The complete host planner, weights, scalar scratch shaders and GPU call are
frozen from 49a0088e4f5e5b1e41da0515050f79d6c43399ae. The comparator changes
only test visibility/type names and takes an explicitly prewarmed original
pipeline cache instead of looking it up in the runtime. Original guards,
weight construction, allocation, bind/dispatch, submit/recycle, receipt and
owned result construction stay exact. Both old shaders and the shared
parameter/coordinate source remain byte-for-byte unchanged. Removing the
new store/load packing restores every line of the old shader math.

Direct horizontal probes check 42,997,770 frozen GPU/independent CPU
words across channels 2/3/4, kernels 3/5, five borders, tiny/ordinary/large
sigmas, singleton/strip/odd/partial shapes and 1025x1025 hot inputs. One
workgroup forces grid-stride iterations. Buffers start poisoned; eight
trailing guard words remain untouched in both layouts. Every component,
including 65280 maximum mass and zero odd high halves, is checked.

Complete calls check 88,570,575 frozen GPU/CPU/physical output
bytes across all logical channels, Q8/Q15 controls, both axis orders, all
borders and million-pixel cases. Full RGBA output inspection checks unused
components. Existing 3240-condition CPU tests, extreme sigmas, ROI metadata,
profile chains and Q15 high limbs remain. Their expected receipts and Q8
scratch-size assertion are updated to the new documented payload only.

New tests also cover equal old planner errors at each capacity boundary;
cold concurrent Q8/Q15 pipelines; same-variant pointer identity; two stage
timestamps; invalid runtime/options/profiler calls leaving pools/cache cold;
five queued calls reusing one scratch allocation; changed inputs, source
recycling and pool clears while independent outputs remain live.

All 121 native image, 90
voxel and 2 public facade tests pass (213
distinct tests), including five new Gaussian tests. All-feature/all-target
Clippy with warnings denied, CPU-only, minimal gpu-wgpu and CI
gpu-image,gpu-aoso-staging WASM with warnings denied, formatting and diff
checks pass. Audit verifies 404 unchanged GPU/image
sources and 413 source/executable hashes. Tracked/new-file and
PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares frozen/current completed resident
calls (filter plus wait) and full calls (upload, filter and owned download).
Host planning/weight generation, resource and command construction, submission,
device work, completion and the route's public transfer remain inside timers.
The frozen comparator uses a separately prewarmed original pipeline cache;
current runtime cache access stays inside the call. Source setup, CPU oracle,
assertions, output destruction/recycling and resident result readback remain
outside timers. Three warmups precede thirty cyclically alternating samples
of all four variant/route combinations. Both routes check every output and
actual scratch/output work receipts. Cold shader compilation is excluded.

Eight (width, height, channels, square kernel) conditions are
(16,17,4,3), (257,259,3,5), (1920,1080,2,3), (1920,1080,4,5),
(1920,1080,3,7), (1920,1080,1,5), (3840,2160,4,3), (3840,2160,3,5).
The Q15 and single-channel conditions control unchanged scratch/math.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and verified after
measurement. Accepted paired log old/new ratios use 10000 bootstrap
resamples, seed 2026101603 + 100*survey + 10*condition_index + JSON_field_index;
sorted indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use gaussian-packed- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions and both routes, checking every output and updated work receipt. All its timing samples are excluded.

