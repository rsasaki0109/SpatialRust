# Packed standalone GPU voxel gathering

## Behavior and scope

`gather_voxel_first_f32_multi_gpu` now reuses the existing integer-word packed
gather kernel to write each original two/four-channel batch into one planar
output and copy that output once. The original runtime batch width stays two
or four. For multiple batches, the unchanged coalesced planner keeps windows
bounded to two original batches, with one submit/map/wait per window. Calls
fitting one original batch use one packed output and one submit/map/wait.
There is no new shader, cache variant, dependency, unsafe code or public API.

Standalone empty or descending cells gather input point zero in every active
channel, including a one-channel batch or tail. This differs from the scalar
tail rule in the existing combined-XYZ helper: the standalone controller
always uses the packed multi-gather recorder, even for one channel. Integer
loads/stores preserve float payload bits, signed zeros and NaNs. Inactive
read-only bindings alias the first input and are skipped by the shader.
The combined helper, its scalar-tail rule, XYZ fusion, average/byte kernels
and all image processing remain unchanged.

Original zero-channel/zero-cell returns and ordered channel-length checks
precede the new branch. Each active host channel keeps its original unpooled
DeviceExt upload, storage usage and label. The same mapped decoder returns
independent vectors with exact channel capacities. Input/output payloads,
channel order and source/pool lifetime behavior remain. No hidden transfer,
host packing or extra readback is added.

## Bounds and fallback

The private planner requires runtime width two/four, seven storage/eight
total bindings, the original width's larger binding requirements when
applicable, 256 lanes/invocations and the full original cell dispatch.
Checked cell*active_batch_channel word indices must fit u32 and packed
output bytes must fit the storage-binding limit. Staging bytes must fit
device max-buffer and host isize::MAX slice limits. For more than one
batch, the unchanged original coalesced planner must also accept the call.
At most min(C, 2*W) channels occupy a window independently of total C.
Pure tests inject exact and one-short limits, including signed-32bit host
boundaries and usize::MAX channel planning without large allocations.

Rejection runs the complete original public body: first its original
coalesced controller, then its original per-batch loop when necessary.
In particular, a device accepting only six storage bindings or a size
whose individual old output planes fit but packed binding does not fit
retains the original execution. Requested device features/limits remain.
Offsets in the private recorder are controller-owned; guarded native
probes verify nonzero aligned destinations and untouched prefix/suffix.

## Deterministic accounting

For C active channels and original width W, compute passes stay ceil(C/W).
Output buffers and copies change from at least C and C respectively to
ceil(C/W) each. Useful upload payload stays 4*point_count*C bytes; useful
output/download payload stays 4*cell_count*C bytes. Packed outputs eliminate
inactive output planes while retaining every active word. Staging payload
and submit/map/wait counts remain those of the original coalesced policy.

With W=4, seven channels keep two passes and one submit/map/wait, while
output buffers fall from eight to two and copies from seven to two. Removing
one inactive plane saves a logical 4*N-byte output request, plus the old
four-byte empty input placeholder. Eight channels keep two passes and one
submit/map/wait, with eight output buffers/copies replaced by two each;
their active storage byte total stays the same. Nine channels use windows
eight plus one, retaining three passes and two submit/map/waits while
copies fall from nine to three. The final single channel keeps point-zero
empty-cell bits. Single-channel calls replace two output planes with one.

Inputs/outputs remain live through each window's wait for at most two
original batches. Returned host vectors and fixture/source data are
additional. These are logical request/object/copy counts; backend alignment,
driver allocations and peak/RSS are unmeasured. Fewer objects/copies alone
do not establish lower peak memory or completed-call latency.

## Correctness and source audit

The complete public call, its three private dispatchers and empty-buffer
constructor are frozen from e7e3b6fa56eb6c9b5904fd91dfa65dad075ddb0e.
The whole coalesced module is frozen from the same revision. Only test
visibility/names and module-qualified imports change. Removing the added
early branch restores the complete original public source. The shared
packed recorder only gains crate visibility; its body, integer shader and
lazy cache stay exact. All original fallback recorders, readback, pools,
byte/average algorithms and image sources remain unchanged.

Public matrices cover 0/1/2/3/4/5/6/7/8/9/11/16/17 channels and
0/1/3/17/255/256/257/65537 cells, plus 5/7/8/9 channels at 1048577 cells.
Independent first-index CPU values and frozen completed calls match;
257-cell cases use GPU-built sorted segments. Validation checks first,
middle and final bad channels before resources, original zero-cell rules,
exact owned capacities and outputs live after source/pool destruction.

Sparse and forced-width fixtures include empty/descending starts, reversed
indices, positive/negative zeros, infinities, subnormals and signaling/quiet
NaNs. Widths two/four exercise every channel count 1 through 2*W and
1/3/257/65537 cells. Nonzero point-zero values expose incorrect scalar-tail
substitution. Five prefix and eight suffix sentinel words remain intact.
Pool statistics stay unchanged across unpooled float uploads. Forced-width
runs on a capable device do not establish physically limited-device results.
Three cold concurrent callers check shared pipeline pointer identity,
changed inputs and independently owned old results after pool clearing.

All 95 native voxel, 126 native
image, 18 voxel-filter and 2
public facade regressions pass (241 distinct tests), including five new
tests. Public matrices compare 72,623,494 parent/CPU words,
sparse cases 137,350 raw words and forced widths
9,080,124 raw/parent/CPU words. All-feature/all-target Clippy with
warnings denied, CPU-only, minimal gpu-wgpu and CI gpu-image,gpu-aoso-staging
WASM builds with warnings denied, formatting and diff checks pass.
Audit verifies 413 unchanged GPU/image sources and
423 source/executable hashes fixed after validation and measurement.
Tracked/new-file and PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares frozen/current complete public
host-channel calls with resident segments: uploads, resource creation,
record/compute/copy/submit, map/wait, owned decoding and return. Fixture and
reference preparation, assertions and result destruction remain outside
timers. Three warmups precede thirty alternating paired samples at
(cells, channels, points_per_cell): (16,4,4), (16,5,4), (4096,7,4),
(65537,9,3), (262144,5,4), (262144,11,4), (1048576,7,4), (1048576,9,4).
Every result checks all bits and exact capacities. Segment construction and
cold device/pipeline initialization are excluded.

Each of two surveys allows a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and checked afterward.
Accepted paired log old/new ratios use 10000 bootstrap resamples with seed
2026101605 + 100*survey + 10*condition_index + JSON_field_index; sorted
indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use voxel-standalone-packed- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions, checking every output bit and capacity. All its timing samples are excluded.

