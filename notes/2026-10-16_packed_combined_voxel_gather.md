# Packed outputs for combined first-attribute voxel gathering

## Scope and behavior

The shared first-attribute recorder used by `gather_voxel_first_xyz_and_multi_gpu`
and `reduce_voxel_centroids_xyz_and_gather_first_multi_gpu` now packs each
existing two/four-channel batch into one planar GPU output. One copy moves
each batch into the original staging region after XYZ. Both public function
bodies, their higher-priority XYZ/attribute fusion, byte handling, validation,
submission, readback and upload recycling are unchanged. Standalone gathering
and every existing shader/pipeline variant remain unchanged.

The original runtime batch width remains: four on devices supporting the
original four-channel gather, otherwise two. Multi-channel empty cells read
input point zero; a one-channel remainder uses the original scalar shader,
which returns positive zero for empty cells. Changing the batch width could
change which channels use that scalar rule, so this optimization does not
do so. The new shader moves u32 words, preserving coordinate-independent
attribute bits without floating arithmetic. Inactive read-only bindings
alias the first input and are skipped. No dummy output/input is needed.

A private planner rejects fewer than two channels, empty cells, invalid
batch widths, fewer than seven storage/eight total bindings, missing
256-lane support, excessive dispatch, u32 output indexing, oversized packed
storage, misaligned destinations, checked region overflow, incomplete
targets and device/host target limits. Rejection precedes new uploads,
outputs and lazy pipeline construction and retains the complete original
helper. Six-storage devices keep their original two-channel path. Limited
devices are covered synthetically; no physically limited-device run is claimed.

The new pipeline is independently lazy and shared per runtime. Pooled upload
labels, upload order, one handle per attribute and recycle timing remain.
The recorder adds no staging allocation, submit, readback, host wait or
hidden transfer. Public outputs stay independently owned.

## Deterministic accounting

For C attributes and original batch width W, attribute output buffers and
copies fall from C to ceil(C/W); compute passes/uniforms/bind groups keep
their previous ceil(C/W) counts. In the separate-XYZ four-wide case, seven
float attributes therefore keep three total float passes and reduce total
float copies from ten to five. With the existing RGB byte batch the complete
call keeps four passes and reduces eleven copies to six. Fused controls
and rejected plans keep their original counts.

Active attribute upload payload remains 4*point_count*C bytes and useful
attribute output/download payload remains 4*cell_count*C bytes. Whole staging,
XYZ, byte payload and owned result capacities remain. The original helper
also created four cell-sized dummy outputs and a four-byte dummy input for
every three/four-channel chunk, even a full four-channel chunk. These are
removed on the packed path. Seven attributes with width four remove eight
dummy outputs, totaling 32*cell_count device payload bytes, plus two four-byte
dummy inputs. All commands still share one submit/map/wait. Driver allocation
overhead and peak/RSS are unmeasured; counts do not establish latency gains.

## Correctness evidence

Complete public parents and their original shared helper are frozen from
381ecfe849ec3ed9686eccd3a405ebb0c4c1b2e1. Only test names/visibility and the
centroid parent's helper import change, so it resolves to the frozen helper.
Removing the production helper's new early branch restores its entire
original body. Removing one field, initializer and method restores the
entire original pipeline cache. Every previous shader, public body, pool,
readback, standalone kernel and image source remains unchanged.

Both public modes compare complete frozen outputs and independent CPU
references for 0/1/2/3/4/5/7/8/9/11/17 float channels, 0/1/3/5 byte channels
and 0/1/3/17/257/65537 cells. Large 1048577-cell cases use 5/7/8/9 floats
with zero/three bytes. GPU-sorted segments, exact capacities, ordered
float-before-byte errors, equal parent allocation/reuse/discard increments
and cached capacity, fresh changed uploads,
owned output lifetimes and three simultaneous cold pipeline consumers pass.

Direct GPU fixtures force both original widths 2/4 with 2/3/4/5/7/8/9/11/17
attributes and 1/3/17/257/65537 cells. They compare original two/four-channel
and scalar recorders plus independent source-word indexing. Empty and
0/1/2/3/4/7/17 occupancies, reversed point indices, a nonzero unused prefix
point, quiet/signaling NaNs, infinities, signed zero, subnormals and extreme
finite words are included. Five prefix/eight suffix staging words and eight
words after each reference output remain untouched. Unaligned, overflowing
and incomplete destination regions record nothing, allocate no pooled
uploads and preserve the entire staging on remap. Exactly one pooled
upload per attribute remains.

All 90 native voxel and 18
filter regressions passed (108 distinct tests), including five new tests.
Direct fixtures checked 17,375,160 parent/independent words;
public matrices checked 525,631,692 complete parent/CPU elements.
All-feature/all-target Clippy with warnings denied, CPU-only, minimal
gpu-wgpu and CI gpu-image,gpu-aoso-staging WASM with warnings denied,
formatting and diff checks pass. Audit verified 398
unchanged GPU/image sources and 408 source/executable hashes.
Tracked/new-file and PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares complete current/frozen public
resident-XYZ/segment, host-float/byte calls: reference-vector preparation,
pooled uploads, resource creation, record/compute/copy/submit, map/wait,
decode and owned return. Setup, oracles, assertions and output destruction
remain outside timers. Three warmups precede thirty alternating pairs at
(cells, floats, bytes, points_per_cell, mode):
(16,4,0,4,0), (16,7,3,4,3), (4096,5,0,4,0), (65537,7,3,4,3),
(262144,5,3,4,0), (262144,9,0,4,3), (1048576,7,3,4,0), (1048576,8,0,4,3).
Mode zero gathers first XYZ and mode three averages XYZ; both gather first
attributes. The four-attribute case controls unchanged XYZ fusion. Every
call checks all output bits/capacities. Sorting and cold construction are
excluded; warmups absorb first-use pipeline creation.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and verified after
measurement. Accepted paired log old/new ratios use 10000 bootstrap
resamples, seed 2026101602 + 100*survey + 10*condition_index + JSON_field_index;
sorted indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use voxel-packed-gather- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions, checking every output and owned capacity. Its timing samples are excluded.

