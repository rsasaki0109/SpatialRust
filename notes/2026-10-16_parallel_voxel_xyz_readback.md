# Bounded CPU copying of standalone mapped voxel XYZ

`gather_voxel_first_xyz_gpu_buffers` and
`reduce_voxel_centroids_xyz_gpu_buffers` now copy their large mapped XYZ
planes with bounded native CPU work. Only the final collector in
`read_staging_xyz` selects a new helper; the reader's original full source
extent check, map/wait/view/unmap guard and returned tuple stay in place.
GPU math, packed and split output layouts, compute/copy commands, payload,
submit/map/wait count, capability selection and public APIs remain unchanged.

## Worker policy and scoped lifetime

The additive XYZ helper reuses the existing native threshold of 1,048,576
cells and the original checked plane planner with exactly three channels.
Available CPU count is capped at three copy workers, including the caller.
At least two available CPUs are required; one CPU or a query error retains
serial copying. Small calls and WASM retain the original three per-plane
`to_vec` copies and do not query CPU count. This remains an initial size
policy, not a measured universal latency crossover.

With two copy workers, one child copies X then Y while the caller copies Z.
With three, two children copy X and Y separately while the caller copies Z.
There are at most two additional threads, and no waiting-only coordinator
is counted as a copy worker. All children run in a Rust scope and join before
the helper returns, including unwind. The original mapped view and unmap
guard therefore outlive all reads. All source bounds are validated before
workers can borrow the planes. A bad source range still drops the view
before unmapping and permits a later valid map.

The helper returns the same tuple of three independent owned Vecs. Every
plane uses `to_vec`, preserving all float bits and exactly N-element
capacity. No float arithmetic, unsafe code, dependency or global pool is added.
Existing generic float attribute worker code and its policy are unchanged;
combined XYZ/attribute readers are unchanged by this slice.

## Work and memory

XYZ useful copying remains 12*N CPU bytes, staging payload remains 12*N
bytes, and returned owned storage remains three N-element Vecs. There is
no outer result Vec allocation and no point-sized intermediate or duplicate
plane copy. Output tuple/handle metadata and thread stacks add bounded
per-call overhead; allocation order can change. Peak memory, RSS, allocator
and backend overhead, stack commitment and latency are not claimed to
decrease. Concurrent API callers own separate outputs and each has at most
three active copy workers, including its caller.

## Current parents and validation

The baseline is main commit beecc3b7b7c87af99531889fe77e6169c1f93a26.
Both complete public functions, their full packed XYZ controller/dispatch,
the centroid split-output dispatch, and original serial mapped XYZ reader
are frozen. The gather split-output body is included in its complete public
function. Shared GPU-only planners/recorders are unchanged. Every old packed
and split readback reaches the frozen serial reader, so no delegate can
accidentally use the new copy helper. Only test names, visibility and module
qualifications differ.

Restoring the original tuple collector restores the entire readback source.
The new XYZ functions are append-only in the CPU plane-copy module; all
original generic float worker code remains exact. The other
449 original GPU/image Rust/WGSL files are
byte-identical, including all GPU public bodies, shaders and caches. The
post-validation manifest pins 456 source/executable hashes.
Published evidence omits the local adapter model name.

Five new tests check 94,372,164 automatic/forced-worker host words,
18,874,470 mapped serial-parent/CPU words,
63,704,304 complete public current-parent/CPU words and
12,582,924 direct packed/split GPU current-parent/CPU words.
Host tests cover zero/small/cutoff/large lengths, raw signed-zero/NaN/
subnormal/random bits, exact capacities, source/output independence and
forced two/three-worker topologies. Relevant planner probes check zero/one/
two/three/many available CPUs and caller-inclusive worker caps. Invalid
worker counts, overflowing extents and short sources reject before any
borrowed worker work.

Mapped cases immediately below/at/above the cutoff preserve exact parent
and independent raw bits, then retain prior results after staging writes,
new readbacks and buffer destruction. A caught out-of-range panic is followed
by a successful valid remap. Complete public tests use 0/17/257/65537/
1048575/1048576/1048577/2097153 cells and occupancy one/four, checking both
first values and centroids against current parents and bit-exact finite CPU
references. Sorted 257-cell construction reads back only four bytes. Exact
capacities and owned outputs survive source destruction and pool clearing;
empty and normal calls preserve pool statistics.

Direct original packed and split GPU gather layouts use 1,048,577 cells,
repeated starts, reversed indices and raw nonfinite input values. Both
mapped routes match the serial parent and CPU first-point reference with
point-zero empty-cell semantics; sentinel suffix words remain untouched.
Two concurrent large complete public calls retain prior outputs after
resident input mutation, subsequent GPU calls, source destruction and pool
clearing. No borrowed mapped result or worker escapes an API return.

All 135 native voxel, 131
image, 18 voxel-filter and
2 facade tests pass (286 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM and CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback_planes.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_xyz_parallel_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-xyz-parallel-*.

## Complete-call timing

The benchmark times both complete public APIs from resident XYZ/segments
through validation, allocation/encoding, GPU submission/map/wait and owned
decoding. Input/reference preparation, output/capacity assertions and
destruction are excluded. Three warmups precede 30 alternating paired
current-parent/new calls. Conditions (cells,occupancy,mode) are (16,4,0),
(65537,4,1), (1048575,1,0), (1048575,1,1), (1048576,4,0),
(1048576,4,1), (1048577,1,0), (2097153,1,1). Mode zero gathers first
XYZ and mode one computes centroids. The first four control the serial path.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Source/executable hashes are captured
after validation and checked afterward. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101614 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
