# Batched GPU voxel byte attributes

## Scope and behavior

The four combined resident-XYZ/host-attribute public voxel calls share a
capability/size-gated batch for two-to-four u8 attributes. First-point and
average selection use independent lazy runtime pipelines. The public API,
float processing, validation order, submission/readback and float upload pool
boundaries remain. Zero/one/five-or-more byte attributes and unsupported
devices or output sizes use the complete original per-channel loop.

Each active byte channel keeps its original padded host upload. There is no
point-sized host repacking or merged input allocation. Four read-only channel
bindings alias the first active buffer for unused channels, which the shaders
do not read. A bounded vector retains at most four buffer handles. Each cell
shares segment/index loads across active channels. Average sums preserve the
original per-channel sequential f32 additions, scalar division and WGSL round;
first-point gather retains zero for empty or descending start/end pairs.

Every 256-lane workgroup initializes four 256-u32 scratch arrays, including
inactive tail lanes, then executes one unconditional barrier. Its first 64
lanes each pack four consecutive cells into one word in every active planar
channel. Overflow-safe ceil(cells/4) stride and zero tail bytes preserve the
existing staging decoder. No atomics, extra waits or submissions are added.

The private planner requires seven storage buffers, eight total bindings,
256 x-lanes/invocations, 4096 workgroup-storage bytes and the complete dispatch
count. Checked output words must fit u32 and output bytes must fit both buffer
and storage-binding limits. Rejection precedes byte uploads/output/cache
creation. Requested runtime limits remain unchanged; synthetic tests cover
limited-device fallbacks without claiming a physically limited-device run.

## Deterministic accounting

For U in 2..=4, byte compute passes, uniform buffers, bind groups, output
buffers and output copies each fall from U to one. RGB is three to one.
Together with the existing fused float path, four float attributes plus RGB
need two total compute passes/copies instead of four. Byte uploads remain U,
and the output/download payload remains 4*ceil(cells/4)*U bytes. Float payload,
staging size, final owned capacities and transfer boundary are unchanged.
These payload counts exclude uniform controls. Workgroup scratch rises from
1KiB per original byte pass to fixed 4KiB per batch; register pressure and
occupancy may affect latency. Object counts do not establish peak-memory,
RSS or speed improvements.

## Correctness method

Four complete public functions are frozen from parent merge
8aef55cf1267e3a97dbcf596020c75ae60fe8c23, changing only test visibility/names.
The source audit removes only each batch wrapper and restores its original
fallback, requiring exact normalized whole-function equality. It also checks
unrelated code, old shaders/recorders, the float fusion paths and cache/pool
sources. Removing the two lazy cache fields/methods restores the parent cache.

Direct GPU tests compare every packed word with original per-channel packed
recorders, independently check small-cell integer first/average values, allow
the original GPU half-tie choice and require batch exactness. Counts 2/3/4
cover cell boundaries 1/2/3/4/5/63/64/65/255/256/257/1023/1024/1025/65537,
occupancies 0/1/2/3/4/7/17 and reversed indices. All byte codes and all 32768
ordered two-byte half ties are included. A 262145-point single cell compares
original sequential f32 sums beyond exact integer accumulation; no integer
mean claim is made for that case. Zero tail padding and eight trailing guard
words are checked for both the original and batched outputs.

Complete public calls cover all four selection modes, 0/1/2/3/4/5/7 bytes,
zero/four floats and cells through 1048577, with exact integral CPU means,
GPU-sorted 257-cell fixtures and extra one/three/five-float cases. Zero/one/
five/seven-byte fallbacks are included. Other tests check float-before-byte
length errors before resource use, changed host bytes, dropped inputs,
cleared pools, owned old results and three concurrent cold callers sharing
each independent cache. Existing sparse/descending/raw float tests also run
through the new byte batch where eligible.

## Measurement method

The ignored release benchmark compares complete current and frozen-parent
public calls with resident XYZ/segments and host float/byte attributes.
Uploads, device commands, map/wait, decode and owned return are timed.
Both wrappers prepare their borrowed channel-reference vectors inside the
timer; fixture setup, expected values, assertions and output destruction
remain outside. Three warmups precede thirty alternating pairs in eight
conditions (cells, byte channels, occupancy, selection mode), with four
float attributes throughout:
(16,2,4,0), (16,3,4,1), (4096,4,4,2), (65537,3,4,3),
(262144,1,4,0), (262144,3,4,2), (1048576,3,4,0), (1048576,4,4,2).
Mode 0 is first/first; 1 first/average; 2 centroid/average; 3 centroid/first.
The one-byte condition controls the unchanged fallback. Every invocation
checks exact complete outputs and capacities. Cold creation and sorting are
outside measurements.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject any
overlapping trial; only owned benchmark children may be cancelled. Sources
and executable hashes are captured after validation and verified afterward.
Accepted paired log old/new ratios use 10000 bootstrap resamples, seed
2026101555 + 100*survey + 10*condition_index + JSON_field_index and sorted
indices 249/9749 for 95% intervals. Ratios above one favor batching; intervals
crossing one are inconclusive. Zero samples suppress their ratio. All cases
are retained. Local receipts use voxel-u8-batch- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published artifacts omit the local adapter model.

## Completion evidence

All 70 native voxel and 18
voxel-filter tests passed (88 distinct regressions), including five new
planner/kernel/public/ownership/concurrent tests. The native device accepted
batching. Direct probes verified 1,514,664 cell results against parent GPU
and applicable CPU oracles; public matrices checked 731,120,768 exact
parent/CPU elements. All-feature/all-target Clippy with warnings denied,
CPU-only, minimal gpu-wgpu and CI gpu-image,gpu-aoso-staging WASM builds
with warnings denied, formatting and diff checks passed.

The audit verified 379 unchanged GPU/image
sources, all four complete frozen public calls and unchanged fallback/tail
bodies. All 391 source/executable hashes stayed fixed
after validation and measurement. Tracked/new-file and published-body
privacy match counts were zero.

Neither bounded quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions, checking every output and capacity; its timing samples are excluded.

