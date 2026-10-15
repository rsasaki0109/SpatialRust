# Batched GPU voxel float averages

## Scope and execution

`reduce_voxel_average_f32_multi_gpu` batches two-to-four float channels in a
lazy GPU kernel. One lane per cell shares segment and point-index loads,
keeps independent per-channel sequential f32 sums and divides each sum by
f32(count), exactly as the original scalar shader. Empty cells store
positive zero in each active plane. No reciprocal multiplication, tree
reduction, quantization, shared scratch or atomics are introduced.

Calls with more than four channels use bounded four-channel groups in the
same existing single submission/readback. A final one-channel group uses
the original scalar recorder. Zero/one channel calls keep the original path
and skip the new device-limit query. Inactive read-only bindings alias the
first active input, and the shader skips their reads and output stores.
Public APIs, default algorithm placement, float upload labels, validation
precedence and mapped owned decoding remain. No pooled uploads are added.

The private planner checks seven storage/eight total bindings, 256 lanes
and invocations, the complete cell dispatch and checked planar output words.
The largest group output must fit the storage-binding limit and u32 indices;
the complete staging must fit device max-buffer and host isize::MAX bytes.
Synthetic host caps exercise signed-32bit boundaries without allocating
large buffers. Unsupported capabilities/sizes retain the complete original
per-channel public loop. Requested runtime limits remain unchanged. The
pipeline/layout initializes once per runtime only after planning succeeds.

## Deterministic accounting

For C active channels, compute passes, uniform buffers, bind groups, output
buffers and output copies each fall from C to ceil(C/4). Four channels use
one of each instead of four; seven use two instead of seven. A one-channel
tail still uses one scalar pass/output/copy. Submit/map/wait remain one.

Host input payload remains 4*point_count*C bytes. Device output, staging
and download payload each remain 4*cell_count*C bytes. No additional
point-sized host packing, hidden transfer, padding plane or unused output
buffer is introduced. The existing parent already retained all channel
inputs/outputs through its single submission, so live active input/output
and staging payload totals do not increase. Per-group metadata holds at
most four input buffer handles. Uniform count drops, and driver allocation
overhead is unmeasured. Four independent accumulators can affect register
pressure and occupancy. Object/command counts do not establish peak-memory,
RSS or latency improvement.

## Correctness method

The complete parent public function is frozen from merge
a876ea7b0d01c4ceb145dedff5663030b6032d3f, changing only test visibility/name.
Removing the added fast-path branch must restore the entire public source,
including ordered validation, the original per-channel loop, uploads,
submission and direct mapped decoding. Removing the new cache field,
initialization and method restores the parent cache. Whole-source audit
requires original WGSL, scalar recorders, combined XYZ/byte methods,
coalesced gather and readback/pool code to remain unchanged.

Direct GPU comparisons check each active output word against the actual
unchanged scalar recorder, including eight untouched guard words after
the new and each scalar output. Counts two/three/four cover cells
1/3/17/255/256/257/1025/65537, reversed indices and occupancies
0/1/2/3/4/7/17. Finite quarter-valued inputs also compare against independent
f64 sums/means within four f32 relative epsilons (scaled by max(1,abs(mean))).
Nonfinite inputs include signaling/quiet NaNs, infinities, negative zero
and subnormals; frozen GPU words must match exactly. Cancellation-sensitive
inputs include the order-dependent 16777216,1,-16777216,3 sequence. A single
262145-point cell compares original sequential f32 results; no exact f64
mean claim is made for cancellation or long-cell fixtures.

Complete public calls compare parent and independent CPU results for
0/1/2/3/4/5/7/8/9/11/17 channels through 65537 cells, and two/three/four/
five/seven-channel cases through 1048577 cells. One/four-point occupancy
keeps the CPU reference exact. GPU-sorted 257-cell fixtures, partial groups,
exact returned capacities, dropped sources and cleared pools are included.
First/middle/final-group length errors preserve input-order precedence.
Zero-cell calls still ignore supplied input lengths. Changed host values
must upload freshly without changing previous owned results. Three
concurrent cold callers share the same lazy pipeline. Pure planner tests
cover each exact/one-short device/host limit, zero/one channels, checked
usize::MAX channel overflow and u32 bounds. These are synthetic capability
probes, not a claim of a physically limited-device run.

## Measurement method

The ignored release benchmark compares complete current/frozen public
host-channel calls with resident segments: uploads, resources, record/
compute/copy/submit, map/wait, owned decoding and return. Setup, borrowed
channel-reference preparation, expected values, assertions and output
destruction remain outside timers. Three warmups precede thirty alternating
paired samples at (cells, channels, points_per_cell):
(16,1,4), (16,4,4), (4096,7,4), (65537,3,4), (262144,2,4),
(262144,5,4), (1048576,4,4), (1048576,7,4).
The one-channel case controls the unchanged scalar path. Every call checks
all output bits and capacities. Sorting and cold construction are excluded;
warmups absorb first-use pipeline creation.

Each of two surveys allows a one-minute quiet scheduling window, with
three five-second quiet checks and process censuses before/during/after.
Overlapping trials are discarded; only owned benchmark children may be
cancelled. Source/executable hashes are captured after validation and
verified afterward. Accepted paired log old/new ratios use 10000 bootstrap
resamples, seed 2026101557 + 100*survey + 10*condition_index + JSON_field_index;
sorted indices 249/9749 define 95% intervals. Ratios above one favor batching,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use voxel-average-batch- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

## Completion evidence

All 80 native voxel and 18
voxel-filter regressions passed (98 distinct tests), including five new
planner/kernel/public/changed-input/concurrent tests. The native device
accepted batching. Direct fixtures compared 1,818,481 original
GPU words, including empty/non-power/nonfinite/cancellation/long-cell/guard
cases; public matrices checked 52,927,918 parent/CPU words.
All-feature/all-target Clippy with warnings denied, CPU-only, minimal
gpu-wgpu and CI gpu-image,gpu-aoso-staging WASM builds with warnings denied,
formatting and diff checks passed.

The source audit verified 390 unchanged GPU/image
sources, complete frozen parent and unchanged fallback/scalar arithmetic.
All 400 source/executable hashes stayed fixed after validation
and measurement. Tracked/new-file and published-body privacy counts were zero.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions, checking every output bit and owned capacity. Its timing samples are excluded.

