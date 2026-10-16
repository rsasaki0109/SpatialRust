# Reuse bounded XYZ copying in combined voxel readback

All four combined first/centroid XYZ and first/average float/u8 attribute
APIs share `read_staging_xyz_and_packed_attributes`. Its final XYZ collector
now reuses `copy_owned_xyz_planes`, introduced for standalone XYZ readback.
The helper and its planner are unchanged. This extends native CPU work
after the original GPU map, without changing any GPU algorithm placement.

## Original phase order and mapped lifetime

The reader still validates the full float and packed-byte source ranges
before attribute copies begin. It completes the original float attribute
copy helper, then original packed-byte decoding, then copies XYZ for the
returned tuple. These phases remain sequential. Attribute children have
joined before the XYZ scope starts, so this change does not nest worker
groups or run both groups simultaneously.

The XYZ helper uses the existing 1,048,576-cell native threshold and checked
three-plane plan. With two available copy workers, one child copies X/Y
while the caller copies Z. With three, two children copy X and Y separately
while the caller copies Z. The XYZ phase uses at most three copy workers
including the caller, and at most two additional threads. One CPU or a
CPU-count query error retains serial copying. Small inputs and WASM retain
the original per-plane `to_vec` copies without a CPU-count query. This is
the existing size policy, not a measured universal latency crossover.

Large float attribute copies retain their original cap of four children
and channel/CPU bounds. Their policy, worker code and allocations do not
change. A large call with at least two float attributes can query CPU count
once for that phase and once for XYZ. The original mapped view and unmap
guard enclose all phases; scoped children join before either can be
released, including unwind. Bounds errors still drop the view before
unmapping, permitting subsequent valid maps.

## Work, ownership and unchanged GPU boundaries

Every result still owns three N-element XYZ Vecs, A N-element float Vecs
and B N-element byte Vecs, with exact N capacities. XYZ copying remains
12*N bytes. Total useful host copying remains 4*N*(3+A)+N*B bytes and
mapped GPU payload remains 4*N*(3+A)+4*ceil(N/4)*B bytes. No outer XYZ Vec,
point-sized intermediate or duplicate plane copy is added. Thread handles,
stacks and metadata add bounded overhead; allocation order may change.
Peak memory, RSS, total allocator/backend work and latency are not claimed
to decrease.

No float arithmetic, unsafe code, dependency, global pool or public API is
added. All raw float bits, byte order/tail exclusion and returned ownership
remain. GPU shaders, fused/wide/packed/scalar selection, uploads (including
borrowed byte payloads), storage/copy commands, staging layout, transfers,
submit/map/wait count and recycling remain. Standalone readers are unchanged.

## Exact current parents and validation

The baseline is main commit 3ad6ec3f5d91fb9822a02a90d0b77a3f6632f302.
All four complete public parents and their original packed reader are
frozen for tests. That reader retains serial XYZ, the current parallel
float attribute helper, original byte decoding and original mapped guard.
Unchanged GPU-only helpers and byte-upload preparation remain shared.
Every old public return reaches the frozen reader, so old calls cannot
pick up the new XYZ selection through a shared delegate. Only test names,
visibility and module qualifications change.

Restoring the original XYZ closure and tuple collector restores the entire
original readback file. The other 453 original
GPU/image Rust/WGSL files are byte-identical, including all public GPU
bodies, shaders, caches, standalone readers and CPU copy helpers. The
post-validation manifest pins 459 source/executable hashes.
Published evidence omits the local adapter model name.

Four new tests check 113,246,480 mapped serial-XYZ-parent/CPU elements
and 447,244,968 complete four-mode current-parent/CPU elements.
Mapped probes cover zero/small/immediately-below/at/above-cutoff and
2,097,153-cell calls; zero/one/four/seven float attributes; absent and
one/three/five byte channels. Float expectations use independent raw words,
including signed zero, NaN payloads, subnormals and arbitrary bits. Byte
expectations independently shift packed words, including nonzero unused
tail bytes. Exact capacities and lifetime after buffer destruction remain.
Caught invalid float and byte extents are each followed by valid remaps.
The existing standalone forced two/three-worker XYZ tests remain unchanged.

Complete public cases cover 0/16/257/65537/1048575/1048576/1048577/2097153
cells, zero/one/four/seven float attributes, absent or one/three/five byte
channels and occupancy one/four. They include XYZ-only combined calls,
single-attribute serial controls, existing fused/wide paths and byte scalar/
batch acceptance. Sorted 257-cell construction reads back only four bytes.
All fields compare bitwise/exactly with both complete current parents and
finite CPU references. Capacities and results survive input destruction
and pool clearing. First/middle/last float-before-byte and byte mismatch
order remain, and empty segments return before validation or pool work.

Four concurrent large calls cover all modes and zero/one/two/three float
attributes with absent and one/three/five byte channels. Earlier outputs
remain after both host attributes and resident XYZ change, subsequent GPU
calls, input destruction and pool clearing. No mapped borrow or child
worker escapes an API return.

All 139 native voxel, 131
image, 18 voxel-filter and
2 facade tests pass (290 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM and CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_combined_xyz_parallel_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-combined-xyz-parallel-*.

## Complete-call timing

The benchmark times all four complete APIs from resident XYZ/segments and
host attributes through validation, uploads/encoding, GPU submission/map/
wait and owned decoding. Input/reference preparation, output/capacity
assertions and destruction are excluded. Three warmups precede 30
alternating paired current-parent/new calls. Conditions
(cells,float,byte,occupancy,mode) are (16,4,0,4,0), (65537,7,3,4,1),
(1048575,0,0,1,2), (1048575,1,1,1,3), (1048576,0,0,1,0),
(1048576,1,3,4,1), (1048577,7,3,1,2), (2097153,0,1,1,3).
Modes 0/1 use first XYZ with first/average attributes; modes 2/3 use
centroid XYZ with average/first attributes. The first four control serial
XYZ selection. Existing parallel attribute phases remain in both routes.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Source/executable hashes are captured
after validation and checked afterward. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101615 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
