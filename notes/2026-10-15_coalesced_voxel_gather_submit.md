# Coalesced voxel gather submissions

## Behavior and scope

`gather_voxel_first_f32_multi_gpu` now coalesces adjacent pairs of its existing
two/four-channel GPU batches. Each bounded window records the same original
compute passes and active output copies into one encoder, submits once, maps
once and directly decodes owned channel vectors. The final partial window
keeps its original two/four-channel shader and inactive input/output slots.
Channel order, float payload bits, empty/descending segment rules, uploads,
pool behavior and public API remain. No WGSL or cached pipeline changes.

The private planner accepts runtime widths two/four, at least one extra
channel, 256 lanes/invocations, existing binding/storage requirements and
the complete cell dispatch. Checked per-channel bytes must fit storage and
the combined mapped staging must fit device max-buffer and host slice-size
limits. Two batches is a fixed maximum: min(channels, 2*batch_width), even
for an arbitrarily large channel count. Rejection uses the complete original
public loop. Zero channels, zero cells and channel-length validation keep
their original order and behavior. Calls fitting a single original batch
keep the old path and skip device-limit queries in the added helper.

The production host limit is isize::MAX bytes so mapped slices and owned
plane allocations remain representable on 32bit and 64bit targets. Pure
tests inject a smaller host cap, including signed-32bit boundary cases,
without allocating large buffers. Combined staging is map/copy-only; its
whole size need not fit max_storage_buffer_binding_size. Only each separate
output binding must fit that storage limit. Requested device limits remain.

## Deterministic accounting and memory tradeoff

For C channels and original width K (two or four), compute passes stay
ceil(C/K); active input uploads/output copies stay C. Submit/map/wait/staging
buffer counts fall from ceil(C/K) to ceil(C/(2*K)) when the plan fits. With
K=4, seven channels use one submission instead of two, eleven use two
instead of three, and seventeen use three instead of five. One-batch and
unsupported-size calls retain their old count.

Host upload payload remains 4*point_count*C and active output/download
payload remains 4*cell_count*C bytes. Per-plane owned lengths/capacities and
total mapped bytes over the call remain unchanged. Uniform controls and
inactive output slots are separate from these payload counts. There is no
host channel repacking, repeated upload, hidden transfer or added readback.

This is a bounded memory/latency tradeoff. One window retains inputs and
outputs for at most two original batches until submission/wait, instead of
one. Its staging allocation contains at most 2*K active planes instead of
K. Thus the live input/output/staging working set can be up to twice the old
per-batch working set; host fixture data and returned outputs are additional.
Backend allocation/staging overhead is unmeasured. Fewer submissions and
objects do not establish peak-memory, RSS or latency improvements.

## Correctness method

The complete public call plus its three dispatch helpers and empty-buffer
constructor are frozen from parent merge
2cf4a7e03ad279038af7f336d4a6b208566617ea. Only test visibility and names change.
Removing the added early branch must restore the whole public source,
including validation, upload labels, fallback, original per-batch submission
and decoder. Whole-source comparison also requires every old shader,
recorder, pipeline/cache, mapped decoder, reduction and combined voxel API
to remain unchanged.

Public comparisons cover 0/1/2/3/4/5/6/7/8/9/11/16/17 channels,
0/1/3/17/255/256/257/65537 cells, then 5/7/8/9-channel cases through 1048577
cells. Four-point cells and the largest one-point cells have independent
first-point CPU references. GPU-sorted 257-cell fixtures check resident
segment use. All owned plane capacities are exact, including dropped inputs
and cleared pools. Length errors in the first, middle and final windows
must match the parent before resources, and zero-cell calls ignore supplied
lengths exactly as before.

Sparse raw-word fixtures include signaling/quiet NaNs, signed zeros,
subnormals, empty cells and descending starts; both frozen parent and an
independent first-index oracle must match exactly. Forced original widths
two/four exercise full and partial windows at 3/5/7/8/9/11/16/17 channels.
This tests the two-wide execution branch on a capable device without
claiming a physically limited-device run. Three concurrent callers check
unchanged results after host input mutation to negative zero, dropped
sources and pool clearing. Planner fixtures cover each exact/one-short
capability, buffer, dispatch and host limit and bounded usize::MAX channel
planning without device allocation.

## Measurement method

The ignored release benchmark compares complete current/frozen public
host-channel calls with resident segments: upload, device resource creation,
record/compute/copy/submit, map/wait, direct owned decoding and return.
Fixture and channel-reference preparation, expected values, assertions and
output destruction remain outside the timer. Three warmups precede thirty
alternating paired samples at (cells, channels, points_per_cell):
(16,4,4), (16,5,4), (4096,7,4), (65537,9,3), (262144,5,4),
(262144,11,4), (1048576,7,4), (1048576,9,4).
The four-channel case controls the unchanged single-batch path on the local
capable backend. Every call checks all output bits and capacities. Sorting
and cold device/pipeline construction are excluded.

Each of two surveys allows a one-minute quiet scheduling window, with
three five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled. Source
and executable hashes are captured after validation and verified afterward.
Accepted paired log old/new ratios use 10000 bootstrap resamples with seed
2026101556 + 100*survey + 10*condition_index + JSON_field_index; sorted
indices 249/9749 give 95% intervals. Ratios above one favor coalescing,
intervals crossing one are inconclusive, and zero samples suppress ratios.
All outcomes are retained. Local receipts use voxel-gather-submit- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

## Completion evidence

All 75 native voxel and 18
voxel-filter regressions passed (93 distinct tests), including five new
planner/public/sparse/forced-width/concurrent tests. Public matrices compared
72,623,494 exact parent/CPU words; sparse fixtures compared
116,850 raw words and forced widths 76,586 words.
All-feature/all-target Clippy with warnings denied, CPU-only, minimal
gpu-wgpu and CI gpu-image,gpu-aoso-staging WASM builds with warnings denied,
formatting and diff checks passed.

The source audit verified 387 unchanged GPU/image
sources, the complete parent/dispatch references and unchanged original
public fallback. All 395 source/executable hashes stayed fixed
after validation and measurements. Tracked/new-file and published-body
privacy match counts were zero.

Neither bounded one-minute quiet window produced an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions, checking every output and capacity. Its timing samples are excluded.

