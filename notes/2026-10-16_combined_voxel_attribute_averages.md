# Batched attributes in combined voxel average calls

## Scope and behavior

The separate-XYZ branches of `gather_voxel_first_xyz_and_average_multi_gpu`
and `reduce_voxel_centroids_xyz_and_average_multi_gpu` now record float
attribute averages in groups of up to four into their existing staging.
They reuse the previously validated standalone four-channel average shader
and lazy pipeline. The original one-to-four-attribute XYZ fusion keeps
priority. More-than-four attributes, and devices/sizes that cannot use XYZ
fusion but can use the smaller attribute batch, benefit from this change.
Zero/one attribute and rejected batch plans keep the exact original loop.

The new helper validates the target region before input/output allocation:
existing group-output, binding, 256-lane, dispatch, u32 and host limits;
four-byte destination alignment; checked offset+attribute_bytes within the
target; and the complete target within device max-buffer and host isize::MAX.
Synthetic host caps cover signed-32bit boundaries without large allocations.
The caller passes the offset immediately after its three XYZ planes. Float
attributes use one planar output/copy per group; a one-channel tail uses the
original scalar recorder. Input labels and per-channel host uploads remain.
At most four input handles are held in each local metadata vector.

The helper only records commands. It creates no staging buffer, submission,
readback or host wait, and does not change requested runtime limits. XYZ
math, first-point empty rules, byte packing/averaging, validation order,
owned decoding and existing pool boundaries remain. All previous WGSL,
cached layouts/pipelines and standalone average code remain unchanged.
Sequential per-channel f32 sums, scalar division and positive-zero empty
attributes come from the unchanged shared shader/scalar tail.

## Deterministic accounting

In a separate-XYZ call with C attributes and a supported plan, attribute
passes, uniforms, bind groups, output buffers and copies fall from C to
ceil(C/4). XYZ stays one pass and three output copies. Seven attributes
therefore use three float passes instead of eight, and five float copies
instead of ten. With three byte channels using the existing byte batch,
the complete seven-float/RGB call uses four passes and six output copies,
instead of nine passes and eleven output copies.

One-to-four-float fused calls keep their previous counts. Submit/map/wait
stay one. Active host float uploads remain 4*point_count*C bytes; float
outputs/downloads remain 4*cell_count*(3+C) bytes. Packed byte payload and
whole staging/returned capacities remain. Parent calls already retained
all active inputs/outputs through one submission, so active device payload
totals do not grow. Uniform count and object count fall; driver allocation
overhead and peak/RSS are unmeasured. Register-pressure/occupancy tradeoffs
of the reused four-sum shader remain; counts do not establish latency gains.

## Correctness method

Both complete public parents are frozen from merge
28d3e64de5934a299484f03edff1887347c64609, changing only test visibility/names.
Removing each new wrapper must restore the original entire function and
fallback loop, including the XYZ/fusion prefix and byte/submit/map tail.
The shared average module is append-only relative to that parent. Audit
also requires original standalone helpers/planners, scalar recorders,
WGSL, pipeline caches, readback and pool sources to remain unchanged.

Public comparisons cover both first-XYZ/average-attributes and centroid-
XYZ/average-attributes modes, 0/1/2/3/4/5/7/8/9/11/17 float attributes,
0/1/3/5 byte channels and 0/1/3/17/257/65537 cells. Five/seven/eight/nine
float cases with zero/three bytes extend through 1048577 cells. One/four-
point occupancy has exact independent CPU means; 257-cell cases use GPU
sorting. Tests include fused controls, zero/one fallback, full/partial
groups, ordered float-before-byte errors, exact capacities and unchanged
upload pool statistics. Changed host floats/bytes must upload freshly and
previous owned results survive dropped sources and cleared pools.

Direct recorder fixtures compare original scalar GPU outputs at two/three/
four/five/seven/eight/nine/eleven/seventeen attributes and cells
1/3/17/257/65537. Reversed point indices and 0/1/2/3/4/7/17 occupancies include
empty cells. An unused prefix point avoids zero-byte storage bindings in
the all-empty fixture. Inputs include finite quarters, signaling/quiet
NaNs, infinities, signed zero, subnormals and cancellation-sensitive
16777216,1,-16777216,3 sequences. Every output word must match the original
scalar recorder. Five prefix and eight suffix staging words, plus eight
guards after each scalar reference output, remain untouched. Unaligned
and overflowing destination requests reject without changing staging.
The unchanged shared kernel's prior direct output-guard/long-cell/math
regressions also run. Three concurrent cold consumers share the existing
lazy pipeline and preserve both public modes. Synthetic limited-device
tests do not claim a physically limited-device run.

## Measurement method

The ignored release benchmark compares complete current/frozen public
resident-XYZ/segment, host-float/byte calls: channel-reference vectors,
uploads, resources, record/compute/copy/submit, map/wait, decode and owned
return. Setup, expected values, assertions and output destruction remain
outside timers. Three warmups precede thirty alternating pairs at
(cells, floats, bytes, points_per_cell, mode):
(16,4,0,4,1), (16,7,3,4,2), (4096,5,0,4,1), (65537,7,3,4,2),
(262144,5,3,4,1), (262144,9,0,4,2), (1048576,7,3,4,1), (1048576,8,0,4,2).
Mode one is first XYZ and mode two centroid XYZ; both average attributes.
The four-attribute case controls unchanged XYZ fusion. Every call checks
all output bits and owned capacities. Sorting and cold construction are
excluded; warmups absorb first-use pipeline creation.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and verified after
measurement. Accepted paired log old/new ratios use 10000 bootstrap
resamples, seed 2026101601 + 100*survey + 10*condition_index + JSON_field_index;
sorted indices 249/9749 define 95% intervals. Ratios above one favor
batching, intervals crossing one are inconclusive and zero samples suppress
ratios. All outcomes are retained. Local receipts use voxel-combined-average-
under C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

## Completion evidence

All 85 native voxel and 18
voxel-filter regressions passed (103 distinct tests), including five new
planner/public/recorder/changed-input/concurrent tests. The native device
accepted batching. Direct offset fixtures compared 4,343,790
original GPU words with untouched staging/reference guards; public matrices
checked 525,631,692 complete parent/CPU elements. All-feature/all-
target Clippy with warnings denied, CPU-only, minimal gpu-wgpu and CI
gpu-image,gpu-aoso-staging WASM builds with warnings denied, formatting and
diff checks passed.

Audit verified 395 unchanged GPU/image sources,
both complete frozen parents/fallbacks and unchanged standalone average
code. All 403 source/executable hashes stayed fixed after
validation and measurement. Tracked/new-file and published-body privacy
match counts were zero.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions, checking every output and owned capacity. Its timing samples are excluded.

