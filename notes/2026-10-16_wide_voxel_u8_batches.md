# Batch wide voxel byte attributes on GPU

Combined voxel APIs previously selected the existing byte batch only for
two to four channels. Five or more channels fell back to one scalar pass,
uniform, output and staging copy per channel. A private checked wide path
now groups those channels in fours, reusing the existing batched gather or
average recorder. A two-/three-channel final batch uses that same recorder;
a one-channel tail uses the original scalar packed recorder. One private
output is copied to the original staging region before the next overwrite.

This covers all four APIs: `gather_voxel_first_xyz_and_multi_gpu`,
`gather_voxel_first_xyz_and_average_multi_gpu`,
`reduce_voxel_centroids_xyz_and_average_multi_gpu` and
`reduce_voxel_centroids_xyz_and_gather_first_multi_gpu`. Their original XYZ,
float-attribute and byte-readback phases remain. Calls with at most four
byte attributes return false before querying limits and retain their
original small-batch/scalar selection. No public API or dependency changes.

## Bounds, numerical behavior and ownership

The pure planner requires more than four channels and reuses the existing
four-channel planner for packed u32 output indexing, storage/binding sizes,
256-lane dispatch and 4096-byte workgroup storage. It also checks uniform/
bind-group/dimension capabilities, complete staging and host size, aligned
offset, checked channel-byte multiplication and final destination bounds.
Unsupported wide calls return false before new buffers/commands and reach
the original selection/fallback. The new private recorder propagates scalar
recording errors through the existing public Result boundary.

All shaders, pipelines and recorders remain source-identical. Average
accumulation preserves the original sequential f32 sum/division/round/mask
for each channel; first-value gathers keep original point order and empty
zero results. Packed per-channel stride remains S=4*ceil(C/4) bytes, with
unused final byte lanes zero. Every active output word is overwritten;
copies retain each chunk's active stride count, including scalar tails.

The output stays private to one recording operation. Pass/copy order and
wgpu resource tracking preserve each result before reuse. Shared output
introduces a dependency from each copy to the next write; fewer requests
and passes alone establish no completed-call latency gain. Existing input
padding/borrowing is reused once per host channel. A small Vec holds each
batch's buffer handles; there is no new point-sized host pack or hidden
transfer. Submitted resources retain their lifetimes through the existing
single submit/map/wait. No new pool, retention, CPU copy policy or unsafe
code is introduced. The wider shared-memory/binding envelope is guarded;
the original scalar route remains for limited devices.

## Source-verified work and allocation ledger

For admitted A>4 byte channels, B=ceil(A/4) and S=4*ceil(C/4):

| Byte-attribute operation | Original scalar fallback | Wide batch |
| --- | ---: | ---: |
| Input uploads | A | A |
| Compute passes / uniforms / bind groups / staging copies | A each | B each |
| Output buffer creations | A | 1 |
| Summed output descriptor bytes | A*S | 4*S |
| Explicit byte-phase buffer creations | 3*A | A+B+1 |
| Caller submit / readback map / wait | 1 each | 1 each |
| Value H2D payload / packed D2H payload | unchanged | unchanged |

For A=17 and C=1,048,576, byte passes/uniforms/copies change from 17 to 5,
outputs from 17 to 1 and explicit byte-phase buffer creations from 51 to 23.
Output descriptor bytes change from 17,825,792 to 4,194,304, removing
13,631,488 requested bytes (13 MiB). D2H byte payload stays 17,825,792 bytes.
For P source points, value H2D stays 17*4*ceil(P/4); separate uniform H2D
falls from 272 to 80 bytes. XYZ/float buffers, metadata and staging are
excluded from these byte-phase counts and unchanged. These are source
operation counts and summed descriptor sizes, not measured physical
allocation, peak-memory/RSS or latency savings.

## Current parent and validation

The baseline is main commit 88c98533e2290f66009d6c2002a0b0278291c175.
Four complete public parent functions are frozen. Their delegates, including
the original small byte batch/scalar recorders, padding, all shaders and
readers, remain source-identical, so parent calls still select one scalar
pass per channel above four. Source audit removes only the four matching
wide hooks and feature registration to recover the original files. It
checks matching arguments/mode flags and private recorder multiplicities.
The other 471 original GPU/image Rust/WGSL sources
are unchanged; 479 hashes pin sources plus the tested executable.
Published evidence omits the local adapter model name.

Five new tests cover 146 planner cases,
27,596,192 sparse raw parent/CPU cell-byte comparisons,
655,369 rounding/long-cell comparisons and
672,578,280 complete public parent/CPU elements. Planner cases
cover exact/one-short device and host limits, aligned offsets, full target
regions, zero/small/wide counts, u32/usize overflow and signed 32-bit host
capacity. Rejected recorders leave guarded staging unchanged.

Raw cases cover all packed byte remainders, dispatch boundaries, up to
65,537 cells and 5/6/7/8/9/11/12/13/16/17 channels. Reversed permutations,
empty/repeated starts, scalar tails, leading/trailing sentinels and zero
packed padding compare exactly to original scalar GPU outputs. All 65,536
ordered byte pairs are tested across a four-channel batch and scalar tail;
parent GPU bits remain exact and independent integer CPU checks permit
either neighboring result at an exact half tie. A 262,145-point long cell
compares to the original sequential f32 GPU, avoiding an incorrect integer-
sum oracle beyond 2^24 accumulation.

Complete calls cover all four XYZ/attribute modes, zero/small controls,
up to 1,048,577 cells, 17 byte and seven float attributes. Regular dyadic/
multiple-of-four fixtures compare bitwise to frozen parents and independent
CPU first-values/f64 means without half ties. Ordered errors, empty-segment
short circuits, exact capacities, four concurrent callers, changed host
attributes and retained results after source destruction/pool clearing pass.

All 168 voxel, 131 image,
18 filter and 2 facade
native tests pass (319 distinct tests). All-feature/all-target Clippy
denies warnings. CPU-only, minimal gpu-wgpu WASM and CI gpu-image/
gpu-aoso-staging WASM with warnings denied, formatting and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_wide.rs,
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_wide_tests.rs,
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs,
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_reduce.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-u8-wide-*.

## Complete-call timing

The release fixture times complete public calls from resident XYZ/segments
and borrowed host attributes through allocation/uploads, all GPU work,
submit/map/wait and owned output collection. Fixture/reference setup,
borrowed channel-list construction, assertions and output destruction stay
outside the clock. Three warmups precede thirty alternating paired calls.
Original CPU readback policies are shared by both variants.

Conditions (cells,float attributes,byte attributes,points per cell,mode) are
(16,4,4,4,0), (257,4,5,4,1), (65537,7,7,4,2), (65537,7,9,1,3),
(262145,4,17,1,0), (262145,4,17,1,1), (1048577,4,7,1,2), (1048577,4,5,1,3).
Modes 0/1 use first XYZ with first/average attributes; modes 2/3 use centroid
XYZ with average/first attributes. The first condition controls the unchanged
small-byte route; the rest cover wide batches and all four public APIs.

Two one-minute windows require three five-second quiet process censuses
and checks before/during/after. Overlap discards trials and only owned
benchmark children may be cancelled. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101621 + 100*survey +
10*condition_index + JSON_field_index, sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor batching, intervals
crossing one are inconclusive and zero samples suppress ratios.

Both bounded one-minute scheduling windows lacked an isolated trial because competing work remained. No timing sample is accepted or published and no latency gain is claimed. A separate functional invocation passed all eight complete-call conditions and exact owned outputs/capacities; its non-isolated timing samples are excluded.
