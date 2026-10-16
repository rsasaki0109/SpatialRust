# Fuse wide voxel averages with XYZ

The existing `gather_voxel_first_xyz_and_average_multi_gpu` and
`reduce_voxel_centroids_xyz_and_average_multi_gpu` now fuse XYZ and the first
four float attributes when there are more than four attributes and the
existing fused-kernel limits allow it. Remaining attributes use the original
four-channel average batches, including the original scalar final channel.
One-to-four-attribute fusion keeps priority. First-attribute APIs, standalone
APIs, byte processing, shaders and pipeline caches stay unchanged.

## Recording and bounded fallback

The private planner reuses the existing ten-storage/eleven-binding fused
planner for seven output planes. It also checks uniform and full compute
support, checked 4*N*(3+C) float extent, complete destination size, host
isize::MAX, and the original remaining-channel batch plan at offset 28*N.
A single remaining channel fits within the already checked seven-plane
binding; its output and scalar dispatch are bounded by that same plan.
The whole copy/map destination can exceed the storage binding size.

All new checks precede the helper's uploads, output allocation and recording.
The public function already created its original staging/encoder. Rejection
appends no work and uses the complete previous packed-XYZ/attribute path,
including its own original fallbacks. A backend can therefore run the old
12*N XYZ output and 16*N attribute batch even when the 28*N fused prefix
exceeds its storage binding. Requested limits and features are unchanged.
Injected-limit tests do not establish execution on physically limited devices.

The recorder receives existing borrowed resident axes and segments, encoder,
destination, and the XYZ mode. It uploads each host attribute once with the
original unpooled storage policy and labels, records one fused prefix and
one copy, then appends the original attribute loop at 28*N. It never
submits, maps, waits or decodes. The moved shared attribute loop retains its
original planner/wrapper guard for existing callers; its command recording
is byte-equivalent after substituting the channel-byte parameter.

## Deterministic work accounting

For C>4 on a supported backend, float compute passes and output copies each
fall from 1+ceil(C/4) to ceil(C/4). For C=7 this is three to two; C=9 is four
to three. Float output buffers fall by one. XYZ first-point raw gathering
and centroid accumulation reuse the existing fused pipelines, so one
indices/starts traversal and its resource/copy commands are removed.
There is no new shader, pipeline variant, output payload or CPU compute.

With seven float and three byte attributes, the unchanged batched byte
pass/copy adds one to each: the total falls from four passes/copies to
three. The single staging buffer, submit/map/wait and total mapped bytes
remain 4*N*(3+C) + 4*ceil(N/4)*B. Host float/byte upload payloads and byte
padding are unchanged. Active float output payload remains 4*N*(3+C).
Buffer objects and commands are reduced, without claiming a reduction in
active bytes, peak allocation, RSS or backend alignment overhead.

XYZ centroids keep original ordered sums followed by reciprocal
multiplication. Float attributes keep ordered sums followed by division.
First XYZ preserves bits, including the input-point-zero rule for empty
cells; average empty cells stay positive zero. Output plane order and exact
owned capacities remain unchanged after later calls, input destruction and
pool clearing. No hidden transfer or additional upload pool is introduced.

## Validation and provenance

The two complete public parents are frozen from
6e98210018099ddd47008999acac1a4f3d1d7d50 with only test names, visibility and
module qualifications changed. Removing two narrow guards restores the full
previous public source, including fusion priority, original fallbacks,
float-before-byte validation, byte loops and readback. Moving the attribute
loop back into its unchanged wrapper restores its full previous source.
All 429 other original GPU/image Rust/WGSL files are
byte-identical, including all mathematical shaders and lazy caches.
The post-validation manifest pins 439 source/executable hashes.

Five new tests cover 364 planner outcomes, 7,237,780
direct recorder/current-parent words and 188,946,314 complete
public/current-parent/CPU elements. Planner probes cover exact/one-short
output binding, target, host and dispatch limits, zero/small/wide counts,
32-bit host boundaries, uniform/compute capability rejection and checked
integer overflow. Every new rejection retains the complete previous path.

Direct probes cover 1/3/257/65537 cells, 5/7/8/9/11 float attributes,
reversed indices and occupancy 0/1/3/5/7, signed zeros, subnormals,
infinities, signaling/quiet NaNs and cancellation-heavy sums. They compare
every output word to the complete current parent, including scalar tails.
Empty cells retain their precise first/average rules. Twenty-four poisoned
destination words after float output represent neighboring byte/padding
regions and remain untouched. Short destinations and <=4-channel helper
rejections leave all destination words and pool counters unchanged.

Public matrices cover 0/1/3/17/255/256/257/65537 cells, occupancies one/four,
and float/byte pairs (0,0)/(1,3)/(4,0)/(4,3)/(5,0)/(7,3)/(8,1)/(9,5)/
(11,3)/(17,1) for both modes. Controls preserve existing small fusion;
wide paths include multiple batches, scalar tails, byte tails and GPU-sorted
257-cell segments. Four 1048577-cell cases cover float/byte/mode triples
(5,3,1)/(7,0,2)/(9,5,1)/(17,3,2). Owned results retain exact capacity after
source destruction and pool clearing. First/middle/last invalid float
channels still precede invalid bytes; empty calls still skip length checks.
Four cold concurrent callers mutate resident XYZ and host float/byte data,
then retain both old and changed owned outputs after pool clearing.

All 110 native voxel, 131
image, 18 voxel-filter and
2 facade regressions pass (261 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM, CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass. Existing exact numerical tests remain enabled.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_wide_average.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_wide_average_tests.rs.
Local evidence: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-wide-average-*.
Published artifacts describe backend/capabilities without a local adapter model.

## Complete-call timing boundary

The ignored benchmark times complete public resident-XYZ/host-attribute
calls, including upload/encoding, submit, wait/map and owned decoding.
Fixture/axis preparation and expected results precede timing; exact output/
capacity assertions and output destruction follow it. Three warmups precede
30 alternating paired old/new calls. Eight (cells,float,byte,occupancy,mode)
conditions are (16,4,0,4,1), (16,7,3,4,2), (4096,5,0,4,1),
(65537,7,3,4,2), (262144,9,0,4,1), (262144,11,3,4,2),
(1048576,7,3,4,1), (1048576,8,0,4,2). Mode 1 gathers first XYZ and mode 2
averages XYZ; both average attributes. The first is unchanged fused control.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlapping trials are discarded;
only owned benchmark children may be cancelled. Source/executable hashes
are captured after validation and verified after trials. Accepted paired
log old/new ratios use 10000 bootstrap resamples, seed 2026101609 +
100*survey + 10*condition_index + JSON_field_index, and sorted indices
249/9749 for 95% intervals. All outcomes remain, ratios above one favor
fusion, intervals crossing one are inconclusive and zeros suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; all its non-isolated timing samples are excluded.
