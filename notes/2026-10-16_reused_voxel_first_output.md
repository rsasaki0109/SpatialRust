# Reuse private voxel first-attribute output buffers

Packed first-attribute recorders previously created one f32 output for each
two-/four-wide batch. They now allocate one output for the largest batch in
that recording operation. Every unchanged compute pass is followed by its
original active-size copy into staging before another batch overwrites the
same output. A partial final batch copies only its active planes.

This covers `gather_voxel_first_f32_multi_gpu`,
`gather_voxel_first_xyz_and_multi_gpu` and
`reduce_voxel_centroids_xyz_and_gather_first_multi_gpu`. The standalone path
keeps its existing at-most-two-batch submission windows, with a fresh private
output per window. Combined wide calls retain the fused XYZ/first-four
prefix output and reuse one tail output. The narrower packed attribute
path also uses this tail recorder. Empty private recorders remain no-ops.

## Preserve original widths, empty semantics and lifetimes

The standalone packed shader gathers input point zero for empty cells,
including one-channel tails. The combined packed recorder keeps its
original width and selects the original scalar shader for a one-channel
tail; that scalar shader returns positive zero for empty cells. No width,
dispatch selection, uniform, binding layout or shader changes. Original
fused-prefix behavior remains. Integer packed gathers retain float words
including NaN payloads; scalar and centroid math use the original kernels.

Each encoder's pass/copy sequence preserves completed batch values before
reuse, with normal wgpu storage/copy transitions. All active cells are
written, including empty ones. Binding a larger output for a partial tail
does not change its uniform-bounded writes or copy length. The output is
never queued for host mutation, pooled, shared across calls or retained
across standalone windows. The existing upload-recycle vector still owns
each input until submission, and original recycling/readback order remains.
Sharing an output requires the next write to follow the preceding copy;
allocation-request savings alone establish no completed-call speed benefit.

Existing planners already bound the largest batch output and the complete
staging/host size; allocating that output once adds no new requirement.
The source input uploads, H2D/D2H payloads, GPU pass/copy counts, submission/
map/wait counts, bounded-window policy, errors, fallback selection and owned
CPU readers are unchanged. No public API, dependency, unsafe code or core
crate changes. Empty guards avoid new zero-sized output allocations.

## Source allocation ledger

Let C be cells, A channels and W the original batch width (two or four).
For standalone admitted calls, K=min(A,2*W) is the unchanged window length.
Original output creations are ceil(A/W), with summed descriptor bytes
4*C*A. New creations are ceil(A/K), with summed bytes
4*C*sum(min(window_length,W)). For wide combined calls, the original
28*C-byte fused prefix remains. With T=A-4 tail attributes, tail creations
change from ceil(T/W) to one and tail descriptor bytes from 4*C*T to
4*C*min(T,W). Counts describe nonempty admitted paths.

For C=262,144, A=17 and W=4:

| Route | Original creations | Reused creations | Original bytes | Reused bytes | Removed requested bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standalone first attributes | 5 | 3 | 17,825,792 | 9,437,184 | 8,388,608 (8 MiB) |
| Wide combined XYZ/first, including prefix | 5 | 2 | 20,971,520 | 11,534,336 | 9,437,184 (9 MiB) |

Standalone windows still contain 8/8/1 channels and perform three submits,
maps and waits. The same 17 input buffers, five uniforms/dispatches/copies
and three staging buffers remain; total explicit buffer creations change
from 30 to 28, excluding pre-existing GPU segments. Combined byte buffers
and staging are excluded from the table and unchanged. Width-two standalone
calls retain 4/4/4/4/1-channel windows (nine outputs become five); raw tests
exercise both widths even on a wider-capability runtime.

These are source create_buffer requests and summed descriptor sizes,
without measured physical allocation, peak-memory or RSS claims. Backend
allocation granularity and retained resources can differ. The ledger alone
does not establish lower latency.

## Current parent and validation

The baseline is main commit 7533330a52d81695ff0a9408c9011ec4ff41ab8a.
Native fixtures freeze all three complete public parents, both original
packed/window modules, the original wide recorder and shared fallback
helpers. Thus old calls cannot pick up the new output allocator through a
live shared helper. Audit restores the original production modules by
reversing only the two output moves and empty guards, and checks frozen
parents after mechanical visibility/path changes. All other
469 original GPU/image Rust/WGSL sources remain
unchanged; 475 hashes pin sources plus the validated executable.
Published evidence omits the local adapter model name.

Four new tests verify 47,647,888 raw parent/CPU words,
30,010,272 complete forced-width parent/CPU words and
496,273,824 complete public parent/CPU elements. Width-two/four raw
cases cover 0/1/2/3/4/5/6/7/8/9/11/12/13/16/17 attributes, up to 65,537
cells, nonzero staging offsets and both adjacent sentinel guards. Reversed
point permutations, repeated starts, empty final cells, all partial widths,
NaN payloads, infinities, signed zeros, subnormals and extreme words compare
exactly with the original GPU and independent integer-word gather rules.
The standalone raw window never exceeds two original batches; complete
forced-width calls also cross multiple original submission windows.

Complete public cases cover 0/1/17/255/256/257/65537/262145/1048577 cells,
all three APIs, partial batches and three byte attributes. Regular fixtures
compare bitwise with complete parents and independent CPU first-values/f64
centroids, retaining exact owned Vec capacities. Error tests preserve the
first invalid float channel across batch boundaries, byte error order and
empty-segment short circuits. Four simultaneous calls retain private
outputs after host attribute mutation, source destruction and pool clearing.

All 163 voxel, 131 image,
18 filter and 2 facade
native tests pass (314 distinct tests). All-feature/all-target Clippy
denies warnings. CPU-only, minimal gpu-wgpu WASM and CI gpu-image/
gpu-aoso-staging WASM with warnings denied, formatting and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_packed.rs,
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_standalone_packed.rs,
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_first_reuse_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-first-reuse-*.

## Complete-call timing

The release fixture times complete public calls from resident XYZ/segments
and borrowed host attributes through buffer creation/uploads, all GPU work,
submit/map/wait and owned output collection. Fixture/reference setup,
borrowed channel-list construction, assertions and output destruction are
outside the clock. Original CPU readback policies remain shared. Three
warmups precede thirty alternating paired calls per condition.

Conditions (cells, attributes, mode) are (16,4,0), (257,5,0), (65537,9,0),
(262145,17,0), (65537,13,1), (262145,17,1), (65537,13,2), (262145,17,2).
Mode 0 is standalone first attributes, 1 first XYZ/first attributes and 2
centroid XYZ/first attributes. Occupancy is one and combined calls carry
three byte attributes. The first condition controls a single unchanged
output; later conditions cover multiple batches and all public routes.

Two one-minute windows require three five-second quiet process censuses
and checks before/during/after. Overlap discards trials; only owned benchmark
children may be cancelled. Accepted paired log old/new ratios use 10000
bootstrap resamples, seed 2026101620 + 100*survey + 10*condition_index +
JSON_field_index, sorted indices 249/9749 for 95% intervals. All outcomes
remain; ratios above one favor reuse, intervals crossing one are
inconclusive and zero samples suppress ratios.

Both bounded one-minute scheduling windows lacked an isolated trial because competing work remained. No timing sample is accepted or published and no latency gain is claimed. A separate functional invocation passed all eight complete-call conditions and exact owned outputs/capacities; its non-isolated timing samples are excluded.
