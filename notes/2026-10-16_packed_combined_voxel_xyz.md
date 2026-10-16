# Packed XYZ regions in combined GPU voxel calls

## Scope and branch priority

The four combined voxel APIs now reuse the existing packed XYZ kernels in
their branches that compute XYZ separately from attributes:
`gather_voxel_first_xyz_and_multi_gpu`,
`gather_voxel_first_xyz_and_average_multi_gpu`,
`reduce_voxel_centroids_xyz_and_average_multi_gpu` and
`reduce_voxel_centroids_xyz_and_gather_first_multi_gpu`.
One planar XYZ output and one copy replace three XYZ outputs/copies.

The original four-attribute/no-byte gather fast path and all existing
one-to-four-attribute fused plans keep priority. The added helper runs only
inside the old separate-XYZ branches: zero/more-than-four float attributes
and appropriate device/size fallbacks. Every original XYZ fallback block is
retained. Attribute gather/average batching, scalar tails, packed byte paths,
their ordering and all shader/cache variants remain unchanged.

XYZ first-point gathering still copies raw bits, including input point zero
for empty cells. Centroids retain the original ordered f32 sums, reciprocal
multiplication and positive-zero empty cells through the existing packed
centroid kernel. No arithmetic, input validation, dependency, unsafe code or
public API is added. Standalone XYZ/attribute APIs and image processing stay
unchanged. This slice adds no shader or cached pipeline variant.

## Append-only recording and bounds

The private recorder receives the caller's existing encoder, borrowed XYZ
buffers, resident segments, staging buffer, aligned byte offset and XYZ mode.
Production uses offset zero, leaving float attributes and packed byte regions
at their original offsets. It records one pass and one copy into that region;
it never uploads axes, submits, maps, waits or decodes results.

The planner delegates to the original packed XYZ planner: the original
eight-storage/nine-binding envelope, uniform/compute/dispatch support,
checked three-plane word extent, packed storage binding, device buffer and
host isize::MAX bounds remain. It additionally requires four-byte offset
alignment, checked offset+12*N within the destination, and whole destination
size within device max-buffer and host limits. The destination is copy/map
storage; its whole extent need not fit a storage-buffer binding limit.

These checks precede the helper's output allocation and command recording;
the public call already created its original staging buffer. Rejection
appends nothing and invokes the original three-plane XYZ block. In particular,
individual old 4*N-byte storage bindings can fit when the 12*N packed binding
does not; that case retains the complete old XYZ recording. Requested device
limits/features stay unchanged. Pure injected limits do not establish runs
on physically limited devices.

## Deterministic accounting and ownership

In accepted separate-XYZ branches, XYZ output buffers and copy commands each
fall from three to one. Their useful output payload remains 12*N bytes.
The existing single staging buffer, submit/map/wait and owned decoder remain.
Total mapped bytes stay 4*N*(3+C) + 4*ceil(N/4)*B for C float attributes and
B packed byte channels. Host attribute uploads, padding, plane order, returned
lengths/capacities and input transfers remain identical.

For seven float attributes on a four-wide backend, the first-attribute or
average path keeps three float passes: one XYZ plus two attribute batches.
Float copies fall from five to three. With three batched byte attributes,
the byte pass/copy remains one, so total copies fall from six to four with
the same four compute passes and one submit/map/wait. Fused calls keep their
original count. No active output or staging bytes are removed; this reduces
two resource objects and two copy commands per eligible call.

The command encoder retains temporary output resources through submission.
Pooled first-attribute uploads still recycle only after successful complete
host decoding; average uploads stay unpooled. There is no added pool lease,
staging allocation, hidden transfer or CPU fallback. Results remain owned
after source destruction, later calls and pool clearing. Backend allocation
overhead, alignment and peak/RSS are unmeasured. Object/copy reductions alone
do not establish a latency improvement.

## Correctness and source audit

All four complete public parents are frozen from
582db6e81441b75830737f0b0aa265dcb5c23302. Only test visibility/names and module
qualification change. Removing the four small guards restores the entire
original public source, including fused priority, zero/ordered validation,
all original XYZ blocks, attribute/byte loops and upload recycling. Removing
the two appended private functions restores the original shared XYZ module.
All original GPU/image shaders, caches and mathematical kernels stay exact.

Five new tests compare 32 exact/one-short offset/target/device/
host plans, including signed-32bit host boundaries and checked u64 addition.
Direct recorder probes compare 789,576 original GPU/packed XYZ words
at 1/3/257/65537 cells, with reversed indices and cells containing
0/1/3/5/7 points. They include signed zeros, infinities, subnormals,
signaling/quiet NaNs and cancellation-heavy sums. Raw first-point CPU values
and centroid empty zeros match. Five prefix/eight suffix staging guard words,
old per-axis suffix guards and adjacent regions remain unchanged. Rejected
unaligned/out-of-range/overflow recordings leave the complete destination
and storage-pool counters unchanged after an empty submission.

Complete public comparisons check 178,334,384 current-parent/CPU
elements across all four modes. Matrices cover 0/1/3/17/255/256/257/65537
cells, occupancies one/four, float/byte channel pairs
(0,0)/(0,3)/(1,0)/(4,0)/(4,3)/(5,0)/(7,3)/(9,5)/(17,1).
They exercise fused controls, split paths, byte tails and GPU-sorted 257-cell
segments. Four 1048577-cell conditions cover every mode with empty/many float
attributes and byte mixtures. Original error precedence checks first/middle/
last float channels before bad byte channels; zero-cell calls still ignore
supplied lengths. All owned plane capacities remain exact.

Four cold concurrent complete callers mutate resident XYZ plus host float/
byte inputs, then verify both changed results and previously returned outputs
after dropping sources and clearing pools. Existing attribute scalar-tail,
nonfinite, byte-rounding, shader cache and standalone XYZ regressions remain.

All 105 native voxel, 131 native
image, 18 voxel-filter and 2
public facade regressions pass (256 distinct tests). All-feature/all-target
Clippy with warnings denied, CPU-only, minimal gpu-wgpu and CI
gpu-image,gpu-aoso-staging WASM builds with warnings denied, formatting and
diff checks pass. Audit verifies 427 unchanged GPU/image
sources and 435 source/executable hashes fixed after validation
and measurements. Tracked/new-file and PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares frozen/current complete public calls
with resident XYZ/segments and host float/byte attributes. Attribute upload,
resource/command creation, compute/copy/submit, map/wait, direct owned decoding
and return stay inside timers. Fixture/axis upload/segment construction,
channel-reference preparation, oracle, assertions and output destruction stay
outside. Three warmups precede thirty alternating paired samples. Every
output and exact capacity is checked. Cold device/pipeline setup is excluded.

Eight (cells,float_attributes,byte_attributes,occupancy,mode) conditions are
(16,4,0,4,0), (16,0,3,4,1), (4096,5,0,4,2), (65537,7,3,4,3),
(262144,7,3,4,0), (262144,9,0,4,1), (1048576,7,3,4,2),
(1048576,8,0,4,3). Modes 0/1 use first XYZ with first/average attributes;
modes 2/3 use centroid XYZ with average/first attributes. The first condition
controls the unchanged fused gather path on a capable backend.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and checked afterward.
Accepted paired log old/new ratios use 10000 bootstrap resamples with seed
2026101608 + 100*survey + 10*condition_index + JSON_field_index; sorted
indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use voxel-combined-xyz- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions across the four XYZ/attribute modes, checking every output and capacity. All its timing samples are excluded.

