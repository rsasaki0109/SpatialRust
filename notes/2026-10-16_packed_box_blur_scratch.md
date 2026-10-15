# Packed exact box-blur intermediate sums

## Scope and bound

The existing separable grayscale `box_blur_gpu` path now stores two adjacent
pixel horizontal sums in one u32 word when kernel width is at most 257.
Source bytes are at most 255, so the largest sum is 255*257 = 65535, exactly
the unsigned 16-bit maximum. Width 259 can reach 66045 and keeps the original
u32 path. Wider positive odd widths, the original direct small-work path,
the area/sample selection thresholds and the public API remain unchanged.

One invocation computes and owns both pixel halves, including a pair crossing
an odd-width row boundary. Each pixel independently computes its original
coordinates and original horizontal sum. An odd final pixel writes zero in
the high half. The vertical kernel extracts the halfword before the unchanged
integer vertical addition, analytical edge weighting, full kernel divisor
and final half-up rounding. No intermediate rounding or cross-invocation
read/modify/write occurs. No atomics are needed.

The original bounded group count remains. Horizontal work walks
P/2 + P%2 words, avoiding P+1 overflow; guarded pair indices and subtraction
before stride addition remain bounded. Vertical work walks the original P
pixels. Packing changes per-invocation work and memory accesses, so lower
scratch/copy counts alone do not establish a latency improvement.

## Requests, receipts and ownership

For P pixels, packed separable scratch requests change from 4*P bytes to
4*ceil(P/2) bytes: exactly half for even P, with one four-byte tail word
for odd P. A one-pixel request stays four bytes. These are logical requests;
pool alignment, cached capacity, driver allocations and peak/RSS are
unmeasured. Actual peak-memory reduction is not inferred from the formula.

The original planner still checks the complete 4*P separable extent, all
coordinate/sum/device limits and original error ordering before choosing
the smaller request. Accepted/rejected calls and pass-count planning remain.
The same 32-byte uniform and existing three bindings remain; no new device
capability is requested. Two lazy variants join the existing runtime cache.
The original direct and wide shaders stay byte-for-byte unchanged and cold
packed-only calls do not compile those variants.

Both weighted passes keep one submit and the original scratch lease/recycle
boundary. Horizontal work receipts now record 4*ceil(P/2) bytes; the RGBA8
output receipt stays 4*P bytes. Stage names, two optional timestamps, Vision
chain pass capacity, host uploads/downloads, returned image ownership and
metadata remain. No extra pass, staging allocation, host wait, hidden
transfer, CPU fallback, dependency or unsafe code is added.

## Correctness and source audit

The complete parent planner, pipeline cache/encoder and public compute call
are frozen from a8bf0421d54c364182640e4e21b4fe32d1550743. The comparator only
changes test visibility/type names and accepts an explicitly prewarmed
original cache instead of looking it up in the runtime. Original planner
guards/policy and resource/submit/receipt/owned-result boundaries stay exact.
All previous shaders and shared coordinate helpers remain unchanged.
The new horizontal per-pixel function is the original function returning
its sum; the vertical shader only replaces storage reads with extraction.

Pure tests compare 607 complete parent capacity/error outcomes
and exact scratch formulas for odd/even/singleton extents, width 257/259,
small/direct and widest supported controls. Direct horizontal probes check
7,414,036 frozen GPU/independent u64-prefix words with poisoned
storage and eight suffix guards in both layouts. Widths 1/3/9/31/255/257,
both borders, all-hot and patterned bytes, singleton/strip/odd/partial
dimensions, row-crossing pairs and 1025x1025 grid-stride fixtures pass.
One workgroup forces complete scratch coverage. Every odd tail has a zero
high half and maximum-mass width-257 hot sums stay exact.

Complete public calls compare 14,471,193 frozen GPU/independent
prefix-oracle/physical output bytes. The independent u64 row/column prefix
oracle is also checked against the existing brute rectangular oracle on
small cases. Matrices cover both borders, hot/patterned inputs, all threshold
and width controls, 257x257 and 4099x4099 kernels on valid small inputs,
million-pixel grid strides and all physical RGBA padding bytes.

New tests cover cold concurrent direct/packed/wide variants, pointer identity,
pass timestamps/stage names, invalid calls leaving pools/cache cold, five
queued calls reusing one scratch allocation, changed inputs and source/pool
recycling while independently owned outputs remain live. Existing ROI,
rounding-neighbor, retained-output and Vision-chain tests remain. Only their
documented smaller work receipt and lazy-variant expectations change.

All 126 native image, 90
voxel and 2 public facade tests pass (218
distinct tests), including five new box-blur tests. All-feature/all-target
Clippy with warnings denied, CPU-only, minimal gpu-wgpu and CI
gpu-image,gpu-aoso-staging WASM with warnings denied, formatting and diff
checks pass. Audit verifies 410 unchanged GPU/image
sources and 418 source/executable hashes. Tracked/new-file and
PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares frozen/current completed resident
calls (public filter plus wait) and full calls (upload, filter and owned
download). Host planning, resource/command construction, submission,
device execution/completion and the route's public transfers remain inside
timers. The frozen comparator uses an explicitly prewarmed original cache;
current runtime cache access stays inside the call. Source setup, oracle,
assertions, result destruction/recycling and resident result readback remain
outside timers. Three warmups precede thirty cyclically alternating samples
of all four variant/route combinations. Every output and actual work receipt
is checked. Cold pipeline construction is excluded.

Eight (width, height, kernel width, kernel height) conditions are
(16,17,3,3), (257,259,9,9), (1920,1080,9,9), (1920,1080,31,31),
(257,259,259,3), (257,259,1,81), (3840,2160,9,9), (3840,2160,31,31).
The small/direct and wider-than-257 cases control unchanged execution.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and verified after
measurement. Accepted paired log old/new ratios use 10000 bootstrap
resamples, seed 2026101604 + 100*survey + 10*condition_index + JSON_field_index;
sorted indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use blur-packed- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions and both routes, checking every output and work receipt. All its timing samples are excluded.

