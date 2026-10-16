# Bounded CPU copies for packed voxel u8 readback

The four combined first/centroid XYZ and first/average float/u8 voxel APIs
still collected their packed byte attribute planes serially. Their shared
mapped reader now calls `copy_owned_packed_u8_planes`. Native calls with at
least 4,194,304 cells and two byte channels can copy disjoint channel bands
in scoped workers. Small inputs, absent/single byte channels, one CPU and
CPU-count query errors retain serial copying; small/single and WASM calls
do not query CPU count for this phase.

The byte cutoff keeps at least four MiB of useful data per channel, matching
the existing float plane policy's useful bytes rather than its element
count. It is a conservative size policy, not a measured universal latency
crossover. `plan_packed_u8_copy` reuses the original `plan_plane_copy` on
ceil(cells/4) packed words after checking the exact cell threshold. Its
checked source-byte extent, available CPU count and channel count bound the
plan. Each band has ceil(channels/workers) channels; the actual number of
children is no greater than four, available CPUs or byte channels. The
caller waits for this phase. No global pool or dependency is added.

## Exact bytes, ownership and lifetimes

Little-endian workers keep the original borrowed packed-word byte view and
copy only the first N bytes of each channel. Big-endian workers keep the
original integer shift/mask decoding. All N modulo four remainders exclude
unused tail bytes. Endian branches are source-equivalent to the old reader;
Native execution and WASM compilation here cover little endian. The complete
packed source range is checked before any child borrows it. Each result is
an independently owned N-byte Vec with exact N capacity. Source padding and
additional source words are never copied into results.

Float attributes finish first, then byte attributes, then XYZ. Each scope
joins before the next phase begins, so worker groups never overlap within
one call. The original mapped view and unmap guard enclose all phases and
scoped children, including unwind. Bounds panics still drop the view before
unmapping, and subsequent valid maps succeed. Concurrent API calls may each
use their own bounded worker group.

The existing float and XYZ planners/copiers remain byte-identical; large
calls can query CPU count once in each of three phases. Float copies retain
their four-child cap and XYZ retains at most three copy workers including
the caller. Only byte collection and appended byte helpers change. GPU
bodies/shaders, fused/wide/scalar/batched paths, math, borrowed uploads,
layouts, transfers, submit/map/wait count, recycling and public APIs remain.

Useful host copying remains 4*N*(3+A)+N*B bytes for A float and B byte
attributes. Mapped GPU payload remains 4*N*(3+A)+4*ceil(N/4)*B bytes. The same
final vectors and outer channel metadata remain, with no additional
point-sized scratch or duplicate byte copy. Thread metadata/stacks add
bounded overhead and allocation order may change. No peak-memory, RSS,
allocator/backend-work or latency reduction is claimed.

## Current parents and validation

The baseline is main commit 29027d77ec0e4ab640070eac5cdd5b8894eaacfa.
Tests freeze all four complete public parents and their serial-byte reader,
including the current parallel float and XYZ helpers. Every old API return
reaches the frozen reader and cannot adopt new byte selection through a
shared delegate. GPU-only recording helpers and current byte uploads remain
shared. Source normalization permits only test names, visibility, module
qualification, whitespace and trailing commas.

Restoring original byte collection restores the entire original readback
file. Existing CPU helpers are an exact source prefix. The other
455 original GPU/image Rust/WGSL files are unchanged,
including every public GPU body, shader and cache. Independent audit also
checks both endian decoding bodies. The post-validation manifest pins
462 source/executable hashes before and after measurement.
Published evidence omits the local adapter model name.

Six new tests verify 507 planner decisions,
369,099,054 automatic/forced two/three/four-worker host bytes,
377,487,702 mapped current-parent/CPU elements and
1,512,598,448 complete four-mode current-parent/CPU elements. Host
tests cover 0/1/2/3/4/17 cells, the exact cutoff and each tail remainder,
8,388,609 cells, absent/single and up to seven channels, exact capacities,
independent shift/mask expectations, source/output independence and short/
overflow rejection. Planner tests also cover seventeen channels and extreme
CPU counts. Mapped tests include nonfinite float bits and all large byte
tails; invalid float and byte extents each permit a subsequent valid remap.

Complete API cases include empty/16/257/65537 cells, immediately below/at/
above cutoff, all four modes, absent/single/two/three/five/seven byte channels,
zero/one/two/seven float channels and occupancy one/four. Sorted 257-cell
construction reads back only four bytes. All fields match both complete
current parents and finite CPU references bitwise/exactly. First/middle/last
float-before-byte and byte error order, empty early return and exact
capacities remain. Four concurrent large calls cover all modes and all
tail remainders; retained outputs survive host attributes and resident XYZ
mutation, subsequent calls, source destruction and pool clearing.

All 145 native voxel, 131
image, 18 filter and 2
facade tests pass (296 distinct tests). All-feature/all-target Clippy
denies warnings; CPU-only, minimal gpu-wgpu WASM and CI gpu-image/
gpu-aoso-staging WASM with warnings denied, formatting and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback.rs,
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback_planes.rs,
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_parallel_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-u8-parallel-*.

## Complete-call timing

The benchmark includes each complete public call from resident XYZ/segments
and host attributes through validation, allocations, uploads/encoding, GPU
submit/map/wait and owned decoding. Input/reference preparation, output/
capacity assertions and destruction are outside the clock. Three warmups
precede thirty alternating paired calls. Conditions (cells,float,byte,
occupancy,mode) are (16,4,3,4,0), (65537,7,3,4,1), (4194303,0,3,1,2),
(4194304,0,1,1,3), (4194304,0,2,1,0), (4194305,1,3,1,1),
(4194306,0,5,1,2), (4194307,2,7,1,3). Modes 0/1 use first XYZ with first/
average attributes; modes 2/3 use centroid XYZ with average/first attributes.
The first four conditions control serial byte selection. Current parallel
float and XYZ phases are present in both routes.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101616 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs/capacities; its non-isolated timing samples are excluded.
