# Packed resident GPU voxel XYZ outputs

## Behavior and scope

`gather_voxel_first_xyz_gpu_buffers` and
`reduce_voxel_centroids_xyz_gpu_buffers` now write one planar XYZ output and
copy it once into their existing mapped staging layout. XYZ order, the
three independently owned returned vectors and their exact capacities stay
unchanged. Both APIs consume already resident input axes and segments.
There is no host upload, hidden device copy, new public API or dependency.
The separate CPU-segment per-axis entry points, shared original recorders,
combined XYZ/attribute branches and all image processing remain unchanged.

First-point gathering reuses the existing packed integer kernel with three
active channels. Its private recorder accepts owned or borrowed buffer
bindings through Borrow without allocating, copying or uploading axes.
Its resource/uniform/encoder body is otherwise unchanged; existing combined
and standalone attribute callers keep their original behavior. Empty or
descending gather cells retain input point-zero bits, including signed zeros
and NaNs. The inactive fourth read-only binding aliases the first axis and
is skipped. No new gather shader or cache variant is needed.

Centroids use one new lazy six-storage/seven-binding kernel. It is the
complete original XYZ kernel with only three output declarations/stores
replaced by planar storage. Segment traversal, per-axis sequential f32 adds,
positive-zero empty outputs, count conversion, reciprocal and final
sum-times-reciprocal arithmetic stay exact. It does not substitute scalar
sum/count division or change reduction order. Its 16-byte control matches
the original bytes. The old XYZ and attribute kernels/cache variants remain.

## Limits, transfers and deterministic accounting

The private planner retains the original XYZ envelope of eight storage/nine
total bindings, 256 lanes/invocations and full cell dispatch. It also checks
uniform/bind-group/axis support and checked three-plane word/byte extents.
Packed output must fit storage-binding and max-buffer limits; complete
mapped bytes must fit host isize::MAX. Original zero-cell handling stays
before resources and cache access. Rejection uses the complete original
public/dispatcher fallback, including the original mapped decoder.

In particular, individual old 4*N-byte planes can fit a storage-binding
limit when the combined 12*N-byte binding does not; such calls keep three
outputs/copies. Requested device features/limits do not change. Forced pure
limits test this fallback choice without claiming physically limited-device
measurements. Inputs remain borrowed resident buffers and are not repacked.

On accepted calls, output buffers and copy commands each fall from three
to one. One compute pass, one submission, one map/wait and one staging
buffer remain. Useful output storage and device-to-host payload stay 12*N
bytes, and staging stays 12*N bytes. Output plus staging data-buffer objects
fall from four to two; their summed logical payload remains 24*N bytes.
Uniform controls and existing input/segment storage are additional.

The mapped XYZ decoder still directly creates three N-capacity vectors;
there is no new host intermediate. Outputs survive input destruction,
pool clearing and later calls. These are deterministic object/copy counts,
not a reduction in active data bytes. Backend allocations, alignment and
peak/RSS are unmeasured. Fewer objects/commands alone do not establish a
completed-call latency improvement.

## Correctness and source audit

Complete current public parents and the original centroid dispatcher are
frozen from afc62d17005bb296ec9f3d010a1e06e3ae0359aa. Only test visibility/names
change; the current direct owned XYZ decoder stays inside both comparisons.
Removing the two new early branches restores the complete original files,
including every combined/fallback API and shared recorder. The shared
packed gather constructor changes only its private generic buffer borrowing.
All original shaders and cache variants remain unchanged. Source comparison
restores the old centroid shader by replacing the planar declarations/stores,
requiring every segmentation and arithmetic token to match.

Five new tests cover 36 exact/one-short plans, signed-32bit
host boundaries, u32/zero cases and the original separate-plane fallback
envelope. Complete public matrices compare 23,649,858 current
parent/CPU/repeated words for 0/1/3/17/255/256/257/65537 cells and
occupancies 1/3/5/7, plus 1048577 cells at occupancy one. Non-power-of-two
occupancies exercise reciprocal arithmetic. GPU-sorted 257-cell segments,
exact capacities, source destruction and pool-independent results pass.

Direct probes compare 789,576 frozen/public/raw words at
1/3/257/65537 cells with reversed indices and repeated nondecreasing starts.
They cover empty cells, signaling/quiet NaNs, infinities, subnormals and
signed zeros. Eight poisoned output suffix words and five prefix/eight
outside-copy staging suffix words remain unchanged. Gather results also
match independent raw first-index CPU values. Centroid payloads compare
against the exact original GPU path, including positive-zero empty cells.
An additional descending-start gather fixture keeps point-zero semantics;
centroid fixtures retain valid nondecreasing segment layouts.

Four cold concurrent gather/centroid callers verify per-runtime pipeline
pointer identity, changed resident input bytes and independently owned old
results after source/pool destruction. Replacing inputs with negative zero
keeps negative-zero gathers and the original positive-zero centroid sums.
Calls leave the storage pool unchanged and zero-cell calls allocate nothing.
Existing standalone/combined attribute tests protect the borrowed recorder.

All 100 native voxel, 131 native
image, 18 voxel-filter and 2
public facade regressions pass (251 distinct tests). All-feature/all-target
Clippy with warnings denied, CPU-only, minimal gpu-wgpu and CI
gpu-image,gpu-aoso-staging WASM builds with warnings denied, formatting and
diff checks pass. Audit verifies 420 unchanged GPU/image
sources and 432 source/executable hashes fixed after validation
and measurements. Tracked/new-file and PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares frozen/current complete public
resident-input calls for gather and centroids. Planning, resource/bind-group
creation, cache access, record/compute/copy/submit, map/wait, owned XYZ decoding
and return stay inside timers. Input upload, segment construction, CPU oracle,
assertions and output destruction remain outside. Each route has three
warmups then thirty alternating paired samples. Every result checks all bits
and exact capacities. Cold device/pipeline initialization is excluded.

Eight conditions combine cells 16/4096/262144/1048576 with occupancies one
and three. Both routes run on the same resident inputs, with the same
completed public boundary for frozen/current calls. There is no claim that
this boundary includes host input transfer or segment construction.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and checked afterward.
Accepted paired log old/new ratios use 10000 bootstrap resamples with seed
2026101607 + 100*survey + 10*condition_index + JSON_field_index; sorted
indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use voxel-xyz-packed- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions for both XYZ gather and centroid routes, checking every output bit and exact capacity. All its timing samples are excluded.

