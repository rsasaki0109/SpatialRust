# Reuse private voxel average output buffers

The standalone multi-channel average and combined XYZ/attribute recorders
previously allocated one f32 output for each group of up to four attributes.
Each recorder now creates one private output large enough for its largest
batch. It records each unchanged compute pass followed by the original
copy into that batch's staging region before the next pass overwrites the
same output. The final one/two/three-channel batch copies only active bytes.

This applies to `reduce_voxel_average_f32_multi_gpu`,
`gather_voxel_first_xyz_and_average_multi_gpu` and
`reduce_voxel_centroids_xyz_and_average_multi_gpu`. The combined wide path
keeps its existing fused XYZ/first-four prefix output and reuses one tail
output. The narrower packed-attribute path uses the same recorder.
Zero channels in the private recorder still do no device work; the
explicit empty guard avoids creating a new zero-sized buffer.

## Ordering, ownership and unchanged contracts

One encoder's pass/copy order preserves every batch before reuse; normal
wgpu resource tracking supplies storage-to-copy transitions. All active
cells, including empty cells, are written by the original shaders. A
partial final batch may bind a larger output, but its uniform channel/cell
counts still bound writes and the copy retains the original active size.
No read depends on the output's initialization or inactive trailing planes.
The staging destinations and offsets remain unchanged and disjoint.

Scratch stays private to the call or recording operation, without a pool,
queue write into an in-flight output, cross-call retention or new unsafe
code. Owned result capacities and mapped readback guards remain unchanged.
Inputs, metadata, shader/uniform/binding layouts, math, GPU dispatch/copy
counts, H2D/D2H payloads, submit/map/wait counts, error order, fallback
selection and CPU readback copy policy are unchanged. Existing planners
already check the largest batch's output size and complete staging/host
bounds; allocating that largest output once adds no new device requirement.
No public API, dependency, core crate or feature boundary changes.

## Source allocation ledger

Let C be cells and A standalone attributes. A nonempty admitted batch route
previously requested ceil(A/4) output buffers with total descriptor bytes
4*C*A. It now requests one output with 4*C*min(A,4) descriptor bytes.
For an admitted wide combined call with A>4, the retained prefix output is
28*C bytes and the tail has T=A-4 attributes. The tail changes from
ceil(T/4) outputs / 4*C*T bytes to one / 4*C*min(T,4) bytes.

For C=262,144 and A=17, the f32 output descriptors are:

| Route | Original creations | Reused creations | Original bytes | Reused bytes | Removed requested bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standalone average | 5 | 1 | 17,825,792 | 4,194,304 | 13,631,488 (13 MiB) |
| Wide combined XYZ/average, including prefix | 5 | 2 | 20,971,520 | 11,534,336 | 9,437,184 (9 MiB) |

The same 17 input uploads, five uniform buffers/dispatches/copies and one
submit/map/wait remain on the standalone route. Its total explicit buffer
creations, excluding pre-existing GPU segments, change from 28 to 24.
Combined byte attribute buffers and staging are excluded from the table
and unchanged. These are source-verified create_buffer requests and summed
descriptor sizes, not measured physical allocation, peak memory or RSS.
Backend allocation granularity and resource lifetime can differ; this
ledger alone establishes no latency gain.

## Current parent and correctness evidence

The baseline is main commit 1a6b61551fecd4c8eca4b63c6e82c4a7dfe65028.
Native fixtures freeze all three complete public parents, the original
batch module and the original wide recorder. Thus the parent cannot pick
up the new allocator through a live shared helper. Unchanged delegates,
including shaders, pipelines, all CPU readers and upload helpers, are
source-identical. The audit reverses only the two output moves and empty
guard to restore the original module and checks the complete frozen parent
bodies after mechanical visibility/path changes. All other
467 original GPU/image Rust/WGSL files remain
unchanged, and 472 hashes pin sources plus the validated executable.
Published evidence omits the local adapter model name.

Four new tests verify 15,005,136 raw parent words and
496,273,824 complete parent/independent-CPU elements. Raw recorder
cases cover 0/1/2/3/4/5/6/7/8/9/11/12/13/16/17 attributes, up to 65,537
cells, nonzero staging offsets and both adjacent sentinel guards. Reversed
point permutations, repeated cell starts, empty final cells and non-power
occupancies exercise every active-plane write; finite means use independent
f64 references within four f32 epsilons times max(abs(mean),1). NaN payloads,
infinities, signed zeros, subnormals and extremes remain bit-exact to the
original GPU path, including scalar one-channel tails.

Complete API cases cover 0/1/17/255/256/257/65537/262145/1048577 cells,
all batch tail widths and three byte attributes. Regular dyadic values
compare bitwise against both complete parent and independent CPU output;
owned Vec capacities remain exact. Validation tests preserve the first
invalid float channel across every batch boundary, byte error order and
empty-segment short circuits. Four simultaneous calls use private outputs,
re-upload mutated host attributes and retain old owned results after source
destruction and pool clearing.

All 159 voxel, 131 image,
18 filter and 2 facade
native tests pass (310 distinct tests). All-feature/all-target Clippy
denies warnings. CPU-only, minimal gpu-wgpu WASM, CI gpu-image/
gpu-aoso-staging WASM with warnings denied, formatting and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_average_batch.rs,
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_average_reuse_tests.rs,
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_average_reuse_reference.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-average-reuse-*.

## Complete-call timing

The release fixture times complete public calls from resident XYZ/segments
and borrowed host attributes through GPU buffer creation/uploads, all
compute/copy work, submit/map/wait and owned output collection. Fixture
setup, borrowed channel-list construction, assertions and output destruction
are outside the clock. Both variants share the original CPU copy policy.
Three warmups precede thirty alternating paired calls per condition.

Conditions (cells, attributes, mode) are (16,4,0), (257,5,0), (65537,9,0),
(262145,17,0), (65537,13,1), (262145,17,1), (65537,13,2), (262145,17,2).
Mode 0 is standalone average, 1 first XYZ/average, 2 centroid XYZ/average;
occupancy is one. Combined calls carry three byte attributes. The first
condition controls an unchanged single output; all three public routes
and multiple batches are covered by later conditions.

Two one-minute scheduling windows require three five-second quiet censuses
plus checks before/during/after. Overlap discards trials and only owned
benchmark children may be cancelled. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101619 + 100*survey +
10*condition_index + JSON_field_index, with sorted indices 249/9749 for
95% intervals. All outcomes are retained; ratios above one favor reuse,
intervals crossing one are inconclusive, zero samples suppress ratios.

Both bounded one-minute scheduling windows lacked an isolated trial because competing work remained. No timing sample is accepted or published and no latency gain is claimed. A separate functional invocation passed all eight complete-call conditions and exact owned outputs/capacities; its non-isolated timing samples are excluded.
