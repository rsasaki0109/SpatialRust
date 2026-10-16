# Fuse wide voxel first attributes with XYZ

`gather_voxel_first_xyz_and_multi_gpu` and
`reduce_voxel_centroids_xyz_and_gather_first_multi_gpu` reuse existing fused
kernels for XYZ plus the first four float attributes when C>4 and the
existing limits allow it. Remaining attributes reuse the original packed
gather recorder with the runtime's original width and scalar tail.
The four-attribute/no-byte gather specialization and one-to-four-attribute
fusion retain priority. The prior wide-average optimization, standalone APIs,
all byte processing, shaders and caches stay unchanged.

## Checked recording and fallback

The private planner reuses the existing ten-storage/eleven-binding fused
planner for 28*N prefix bytes and original 256-lane/full-dispatch bounds.
It checks bind-group/uniform and compute dimensions, then delegates to the
original complete attribute planner at offset 12*N: original width 2/4,
packed output binding/u32 extent, checked C*4*N plus offset within the whole
destination, device max-buffer and host isize::MAX limits all apply.
The complete target can exceed a storage binding size. Since C>4,
28*N lies inside the already bounded 4*N*(3+C) float extent.

The remaining plan has the same channel bytes and width, with offset 28*N.
Four is divisible by both original widths, so all remaining batch boundaries
are preserved. A scalar final channel fits the already checked seven-plane
binding and uses the original scalar recorder. All new checks precede helper
uploads, output allocation and recording. Rejection appends no commands or
recycle handles and retains the complete previous packed-XYZ/attribute path,
including its own original fallbacks. Public staging/encoder construction
still precedes the helper. Requested features and limits are unchanged;
injected limits do not prove execution on physically limited devices.

The recorder receives the existing encoder, borrowed resident XYZ/segments,
destination and caller-owned recycle list. It uploads exactly four prefix
channels through the original pool/label, correctly handling a nonempty
recycle list, then records one fused prefix and copy. The unchanged packed
attribute recorder uploads each remaining channel once and copies its
original batches at 28*N. The helper never submits, waits, maps or decodes.
The public function recycles all attribute inputs only after successful owned
decoding, as before. Original pool budgets, per-size cap and zero-budget
discard policy remain. There is no extra pool lease or staging allocation.

## Exact semantics and deterministic work

At the production four-wide setting, C>4 float compute passes, output copies
and output buffers each fall from 1+ceil(C/4) to ceil(C/4): seven float
attributes use two instead of three, and nine use three instead of four.
Three packed byte attributes add the unchanged single byte pass/copy, so
the seven-float/three-byte total falls from four to three passes/copies.
Forced two-wide tests use 1+ceil((C-4)/2) instead of 1+ceil(C/2), two fewer
passes/copies; a production fused-capable runtime currently selects width four.

Useful float outputs remain 4*N*(3+C) bytes. Mapped bytes remain
4*N*(3+C) + 4*ceil(N/4)*B for B byte channels. Float/byte host payloads,
byte padding, one submit/map/wait and owned plane capacities are unchanged.
Output objects/commands are reduced; active payload, peak allocation, RSS,
backend alignment and driver overhead are not claimed to decrease.

First XYZ and first attributes keep their original bits. Centroid XYZ keeps
ordered f32 sums followed by reciprocal multiplication and positive-zero
empty cells. Empty first XYZ and multi-channel attributes read input point
zero; the original scalar final attribute writes positive zero. Keeping
the original width is essential: for C=7, width two has a scalar final
attribute, while width four has a three-channel final batch. Prior small
fusion, signed zeros, NaN payloads and attribute/byte order stay intact.

## Correctness and provenance

Complete current public parents are frozen from
6e5c4952ac5a5a431102d260880b1b84d732f640 with only test names/visibility and
module qualifications changed. Removing two narrow guards restores the full
previous public source, including old special/fused priority, validation,
fallbacks, attributes/bytes, readback and recycling. All 434
other GPU/image Rust/WGSL sources are byte-identical. Existing packed gather,
scalar tails, pool, mathematical shaders and caches are unchanged.
The post-validation manifest pins 443 source/executable hashes.

Five new tests cover 733 planner outcomes, 14,475,560
direct recorder/current-parent-XYZ/raw-CPU attribute words and
188,946,314 complete public/current-parent/CPU elements.
Plans cover exact/one-short binding, full target/host and dispatch bounds,
invalid/original widths, zero/small/wide counts, 32-bit host boundaries,
uniform/compute rejection and checked overflow.

Direct recorder probes cover 1/3/257/65537 cells, 5/7/8/9/11 float attributes,
reversed indices and occupancy 0/1/3/5/7, signed zeros, subnormals,
infinities, signaling/quiet NaN bits and cancellation-heavy XYZ sums.
Both forced widths compare every attribute word with an independent CPU
first-point/scalar-tail oracle; XYZ compares with the current parent, and
all words compare with that parent at the runtime's width. Twenty-four
poisoned neighboring byte/padding words remain untouched. Short-target and
<=4-channel helper rejections preserve destinations, recycle handles and
pool counters. Successful recordings append exactly C pooled uploads, using
an existing recycle-list entry to check indexing. Accepted runs exercise
both zero and nonzero retention budgets, recycling only after readback.

Public matrices cover 0/1/3/17/255/256/257/65537 cells, occupancies one/four,
float/byte pairs (0,0)/(1,3)/(4,0)/(4,3)/(5,0)/(7,3)/(8,1)/(9,5)/(11,3)/
(17,1), both XYZ modes, old specialized/fused controls, GPU-sorted 257-cell
segments, multiple batches and byte/scalar tails. Four 1048577-cell cases
cover (float,byte,mode)=(5,3,0)/(7,0,3)/(9,5,0)/(17,3,3). Exact capacities
and owned output survive source destruction and pool clearing. Invalid
first/middle/last float channels still precede invalid bytes; zero-cell
calls still skip length checks. Four cold concurrent callers mutate resident
XYZ and host float/byte inputs and retain both old and new owned outputs.

All 115 native voxel, 131
image, 18 voxel-filter and
2 facade regressions pass (266 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM, CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass. Existing numerical and average-path regressions remain.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_wide_first.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_wide_first_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-wide-first-*.
Published evidence describes backend/capabilities without a local adapter model.

## Complete-call measurement boundary

The ignored benchmark times complete public calls from resident XYZ and
host attributes through uploads/encoding, submit, map/wait and owned decode.
Fixture preparation and expected outputs precede timing; exact assertions,
capacity checks and output destruction follow it. Three warmups precede 30
alternating paired current-parent/new calls. Eight (cells,float,byte,occupancy,
mode) conditions are (16,4,0,4,0), (16,7,3,4,3), (4096,5,0,4,0),
(65537,7,3,4,3), (262144,9,0,4,0), (262144,11,3,4,3),
(1048576,7,3,4,0), (1048576,8,0,4,3). Mode 0 gathers first XYZ; mode 3
averages XYZ; both gather first attributes. The first controls the existing
four-attribute/no-byte specialization on a capable backend.

Two one-minute quiet windows require three five-second quiet checks and
process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Source/executable hashes are captured
after validation and checked after trials. Accepted paired log old/new
ratios use 10000 bootstrap resamples with seed 2026101610 + 100*survey +
10*condition_index + JSON_field_index and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor fusion, intervals
crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
