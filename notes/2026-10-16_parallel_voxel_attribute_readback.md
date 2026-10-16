# Bounded CPU copying of mapped voxel attributes

The four combined GPU voxel XYZ/float/byte APIs share
`read_staging_xyz_and_packed_attributes`. Its owned float attribute copies
now use bounded native CPU workers for at least 1,048,576 cells and two
float attributes. This changes the CPU portion after the existing GPU map
completes. GPU algorithm placement, kernels, uploads, compute/copy commands,
submission, map/wait count, staging layout and payload remain unchanged.
Standalone XYZ/channel readers and byte/XYZ decoding remain unchanged.

## Policy and safe mapped lifetime

Small inputs and fewer than two attributes retain the serial per-plane
`to_vec` loop and skip CPU-count queries. Native large calls use available
parallelism, capped at four and the number of channels; zero/one available
workers or query errors retain serial copying. WASM compiles only the
serial implementation and performs no CPU-count query. This is an initial
size policy, not a measured universal latency crossover.

The pure planner checks cells*channels*4 against usize overflow and the
host isize::MAX bound. Contiguous bands contain ceil(C/worker_cap) planes,
so actual nonempty worker count is at most that cap. Each copied plane has
at least 4 MiB; eligible attributes therefore total at least 8 MiB. The
helper checks the full source slice before spawning. Each child borrows
disjoint mutable output Vec metadata and immutable float planes, then uses
the original `to_vec` copy for every complete plane. Plane order and raw
float bits are unchanged; no arithmetic or unsafe code is added.

All children run in a scoped lifetime and join before the helper returns.
The original mapped view and unmap guard outlive the helper. Original float
and packed-byte destination bounds are still validated before attribute
copying, and XYZ/byte copies keep their original order afterward. Bounds
unwind still drops the view before unmapping. Rust's scope joins children
on unwind before mapped storage can be released. There is no borrowed
mapped result or thread continuing after the API returns.

## Work, memory and ownership

Useful float attribute copying remains 4*N*C CPU bytes into C owned
N-element vectors with exact N capacities. XYZ still copies 12*N bytes and
byte results retain N*B useful owned bytes. Mapped payload remains
4*N*(3+C) + 4*ceil(N/4)*B. No point-sized intermediate or duplicate copy is
introduced. The same result Vec metadata is initialized before workers fill
it; scoped thread handles/stacks and worker metadata add bounded per-call
overhead. Allocation order can change. Peak memory, RSS, stack commitment,
allocator behavior and backend overhead are not claimed to decrease.

This is CPU parallelization, not additional GPU computation or fewer GPU
passes/transfers. The public APIs, pooled attribute upload/recycling policy,
validation order, byte padding, exact returned capacities and ownership
remain. Previously returned results survive later source/staging mutation,
source destruction and pool clearing. Concurrent callers have separate
outputs and at most four copy workers each; no global pool or dependency
is introduced. Parallel overhead and memory bandwidth can affect latency.

## Validation and exact current parents

All four complete public parents and their original serial packed readback
are frozen from 740a423377bf200752830d00d2327ce2f64819e4. Only test names,
visibility, module qualifications and the serial readback call name change.
GPU recording helpers are unchanged and append no hidden reads. The timed
old functions invoke the frozen serial reader, so both full-call routes
include their own original/new CPU copy rather than sharing the new helper.

Replacing the new attribute collection expression with the old serial
collector restores the entire original readback source. Map/wait, guards,
all range checks, XYZ/byte handling and standalone readers are exact. All
439 other original GPU/image Rust/WGSL files are
byte-identical, including every public GPU body, shader and cache. The
post-validation manifest pins 447 source/executable hashes.

Five new tests check 339 plans, 116,392,193 raw host
words, 75,497,946 mapped parent/CPU elements and
104,964,370 complete public/current-parent/CPU elements. Planner
probes cover 0/1/1048575/1048576/1048577/2097153 cells, 0/1/2/3/4/5/7/17
channels and zero/one/two/three/four/many available workers. They check
nonempty complete band coverage, worker/plane caps and checked host extents.

Host copying exercises zero/small/cutoff/large cells with 0/1/2/3/7 planes,
automatic and forced two/three/four-worker plans, exact NaN/subnormal/signed-
zero/random words and capacity. Outputs and inputs remain independent;
short source and overflowing extents reject before worker borrowing.
Direct mapped probes compare the old serial reader and independent raw
float/packed-byte expectations around the cutoff, including one-attribute
controls, byte tails, source destruction, invalid float/byte bounds and
successful remapping after each caught panic.

Complete public cases cover (cells,float,byte,mode): (16,4,0,0),
(4096,7,3,1), (1048575,2,0,2), (1048576,2,3,0), (1048577,7,3,1),
(1048576,4,0,2), (1048577,4,5,3), (1048577,1,1,0), all at occupancy four.
Every output compares with the complete current parent and CPU reference;
capacities and ownership survive input destruction and pool clearing.
Original first/middle/last float-before-byte validation and empty-call order
remain. Four concurrent large mapped callers retain old outputs after buffer
mutation, new readback, buffer/source destruction and pool clearing.

All 120 native voxel, 131
image, 18 voxel-filter and
2 facade regressions pass (271 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM and CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass. Existing numerical and GPU fusion regressions remain.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback_planes.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_parallel_readback_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-parallel-readback-*.
Published evidence omits the local adapter model name.

## Complete-call timing

The benchmark times complete public calls from resident XYZ/segments and
host attributes through uploads/encoding, GPU submission/map/wait and owned
decoding. Fixture preparation, expected results, exact output/capacity checks
and destruction are outside timing. Three warmups precede 30 alternating
paired current-parent/new calls. Eight (cells,float,byte,occupancy,mode)
conditions are (16,4,0,4,0), (262144,7,3,4,1), (1048575,2,0,4,2),
(1048576,1,3,4,3), (1048576,2,0,4,0), (1048577,7,3,4,1),
(1048577,4,0,4,2), (2097153,3,1,4,3). Modes 0/1 use first XYZ with
first/average attributes; modes 2/3 use centroid XYZ with average/first
attributes. The first four conditions control the serial policy.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Source/executable hashes are captured
after validation and checked afterward. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101611 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
