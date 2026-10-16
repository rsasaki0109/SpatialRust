# Bounded CPU copying for standalone GPU voxel channels

The standalone `gather_voxel_first_f32_multi_gpu` and
`reduce_voxel_average_f32_multi_gpu` APIs now reuse the bounded native CPU
plane-copy helper in `read_staging_f32_channels`. Only the owned-plane
collector changes. The helper was introduced for combined XYZ/attribute
readback and is byte-identical to main. All GPU algorithms, shaders,
recorders, uploads, compute/copy commands, staging layout, payloads, submit/
map/wait counts and input validation remain unchanged.

## Eligibility, mapped lifetime and memory

For each mapped buffer, at least 1,048,576 cells and two channels enable
independent complete-plane copies on available CPUs, capped at four and
the channel count. Contiguous bands contain ceil(C/worker_cap) planes;
actual nonempty worker count can be below the cap. Fewer than two available
CPUs or a query error retain serial copying. Small/single-channel calls skip
CPU-count queries; WASM compiles only the original serial per-plane copies.
This is the existing initial size policy, not a measured latency crossover.

First-value gathering keeps the original width-two/four packed batches and
at most two batches per submit/map/wait window, including one-channel final
windows. Eligibility applies to each window, so a one-channel final window
stays serial. Average reduction retains four-channel batches, the original
scalar tail, and one submit/map/wait for all channels. Original coalesced and
scalar/multi-output fallbacks also reach the same reader.

The original reader validates the complete mapped source extent before
calling the helper. The original mapped view and unmap guard remain alive
until all scoped workers join, including unwinding. Each worker borrows
immutable source planes and disjoint output Vec metadata, then uses the
original `to_vec` copy. No arithmetic, unsafe code, global thread pool,
dependency or public API is added. NaN payloads, signed zero, subnormal bits,
channel order and exactly N-element owned vector capacities remain. Bounds
panics still drop the view before unmapping, permitting subsequent maps.

Useful host copying remains 4*N*C bytes and mapped payload remains 4*N*C
bytes in total across the original windows. Every result still owns C
N-element planes. No point-sized intermediate or additional plane copy is
introduced. Initial output Vec metadata remains equivalent; scoped handles,
thread stacks and worker metadata add bounded per-call overhead. Allocation
order can change. Peak memory, RSS, stack commitment, allocator/backend
overhead and latency are not claimed to decrease. Concurrent callers have
independent outputs and at most four copy workers each at any time.

This extends CPU parallelism after GPU completion. It adds no GPU work and
does not change existing combined XYZ/byte readback or standalone XYZ.

## Complete current-parent references and validation

The baseline is main commit 0027d5d5fa6a7cfbfe9b46d70bf6dcc6f10ede85.
Both full public parent functions, packed/coalesced gather controllers,
average controller, every original gather dispatch fallback, and the serial
channel reader are frozen for tests. GPU-only recording helpers remain
shared and byte-identical. All baseline controller and dispatch readbacks
reach the frozen serial reader; none can call the new parallel reader.
Only test visibility, names and necessary module qualifications change.
Freezing just the public entry functions would have shared the new reader
through their delegates and would not compare the intended two routes.

Restoring the original serial collector restores the entire original
readback source. The 444 other original GPU/image
Rust/WGSL files are byte-identical, including the shared CPU helper and all
public GPU bodies, shaders and caches. The post-validation manifest pins
450 source/executable hashes.

Five new tests check 25,166,074 raw mapped current-parent/CPU words,
159,391,128 complete public current-parent/CPU elements and
58,720,312 forced-width packed/coalesced current-parent/CPU words.
Direct mapping probes cover zero cells/channels, small inputs, immediately
below/at/above the cutoff, large single channels, exact raw NaN/subnormal/
signed-zero/random bits, exact capacities and successful remapping after a
caught range panic. The existing helper's forced-worker/planner tests remain.

Complete first-value and average calls exercise 16/257/1048575/1048576/
1048577 cells and 1/2/4/7/9/17 channels, with one/four points per cell.
The 257-cell case uses sorted GPU segment construction with only four bytes
of construction readback. Large cases cross the host parallel cutoff and
first-value multiwindow boundaries. Both current parent and CPU results are
checked bitwise, including exact returned capacities and lifetimes after
input destruction and pool clearing. Pool statistics remain unchanged.

Forced width-two/four packed and coalesced gathers use 1,048,577 cells and
2*width+1 channels. Repeated cell starts produce empty cells, reversed point
indices and raw nonfinite inputs verify original point-zero empty behavior,
and one-channel final windows exercise serial tails. Separate tests retain
scalar/batched non-power-of-two averages and first/middle/last mismatch order,
empty-channel and empty-segment early returns. Two concurrent large complete
public calls retain earlier outputs after changed host inputs, new GPU calls,
source destruction and pool clearing.

Non-power averages remain bitwise identical to the current GPU parent. Their
independent CPU reference accumulates in f64 and uses the existing average
kernel bound, 4*f32::EPSILON*max(abs(expected),1), because GPU division by
three can differ from CPU division by one ULP. Complete occupancy-one/four
fixtures and every host/raw mapped copy retain exact bitwise CPU checks.

All 125 native voxel, 131
image, 18 voxel-filter and
2 facade regressions pass (276 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM and CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_channel_parallel_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-channel-parallel-*.
Published evidence omits the local adapter model name.

## Complete-call timing

The benchmark times complete calls from resident segments and host float
channels through uploads, allocation, encoding, GPU submission/map/wait and
owned decoding. Input preparation, reference generation, output/capacity
assertions and output destruction are outside timing. Three warmups precede
30 alternating paired current-parent/new calls for each condition. The
eight (cells,channels,occupancy,mode) conditions are (16,4,4,0),
(262144,7,4,1), (1048575,2,4,0), (1048576,1,4,1), (1048576,2,4,0),
(1048577,7,4,1), (1048577,9,4,0), (2097153,3,4,1). Mode zero gathers
first values and mode one averages. The first four are serial controls.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Source/executable hashes are captured
after validation and checked afterward. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101612 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
