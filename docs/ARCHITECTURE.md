# SpatialRust Master Architecture (v0.1)

Design version: **v0.1 Master Architecture Draft**

North star: **Rust-native spatial intelligence: capture, understand, reconstruct, and act**

## Core decisions

A fixed synthetic [sampling audit](DENSE_SAMPLING_AUDIT.md) binds test-only
native observations and installed CPU reference APIs to source/module hashes.
Pyramid raster, lookup admission and derivative semantics remain separate
from trajectory acceptance. Production algorithms, dependencies, public APIs
and transfers are unchanged; an unobservable external Jacobian stays explicit.

An explicit initial-correspondence intensity policy scales fitting gray planes
once before pyramid construction, while retaining original fitting depth and
raw final admission. Streaming means and independent gain application keep
owned memory and CPU/device boundaries explicit. See the
[fixed pair-mean protocol](PAIR_MEAN_DENSE_RGBD_VALIDATION.md).

A separate gray-only fitting policy preserves original full-resolution depth
while sharing the prior Gaussian3 gray arithmetic and raw final admission.
Typed selection, preflighted owned planes and defaults retain their contracts;
see [the component comparison](GRAY_GAUSSIAN_DENSE_RGBD_VALIDATION.md).

An explicit native dense fitting policy can filter full-resolution inputs while
retaining original raw levels for final admission. Checked preflight charges
all additional planes; the raw default, safe owned Python boundary and core/GPU
contracts remain. See [the prospective Gaussian3 protocol](GAUSSIAN_DENSE_RGBD_VALIDATION.md).

A separately fixed CPU comparator owns its adjacent RGB-D inputs and one causal
relative seed. The existing generator retains full-cloud/world/IO admission;
bounded raw projective validation is separate from the external estimator's
filtered objective. No production dependency, numerical binary or device
ownership change; see [the comparator contract](OPEN3D_DENSE_RGBD_COMPARISON.md).

Explicit headless runtime options retain the primary/high-performance default
and expose GL as an opt-in backend set. Owned sync/async constructors retain
original flags, adapter limits and optional timestamp admission. Explicit GL
requires compiled GL support before instance creation and compute-shader support
before device creation; Primary admission retains its original policy. Requested and
actual backends are separate receipts; raw hardware classes retain their
meaning. Shader work, core dependencies and CPU/GPU transfers are unchanged.
The Windows primary control passes both hardware classes; the fixed Linux GL
attempt with the original system driver fails during device creation, before
application work. Separate pinned
diagnostics bind the internal GLSL dynamic-array failure without admission
overrides. See the [fixed GLES protocol and outcomes](EXPLICIT_GLES_GPU_VALIDATION.md).

A separately frozen driver/WSI study reuses the executable and all numeric,
transfer and pool gates with privately staged Mesa Gallium D3D12/surfaceless GL.
All six cases pass; raw Other remains outside the two-class hardware gate.
Installed system libraries, defaults, kernels and safe ownership stay unchanged;
see [isolated Gallium evidence](ISOLATED_GALLIUM_GLES_VALIDATION.md).

A separately fixed experimental dense adapter retains one owned proper relative
pose as the next initial seed, starting with identity. Native and full-cloud
admission remain; a lost chain makes no later call. Independent auditing checks
every seed against the preceding relative estimate. This reuses the frozen CPU
solver without a runtime/core/GPU change; see the
[causal initialization protocol](CAUSAL_DENSE_RGBD_VALIDATION.md).

Dense projective RGB-D uses a dedicated optional vision feature with checked
borrowed grayscale/metric-depth views for both frames, calibrated pinhole data,
an explicit initial pose and preflighted owned CPU pyramids. Fixed ordered row
tiles bound private CPU workers; the Python adapter owns all inputs before
releasing the GIL. No core dependency, unsafe block, hidden device transfer or
change to sparse tracking defaults is introduced. See the
[dense RGB-D protocol](DENSE_RGBD_ODOMETRY_VALIDATION.md).

Hardware coverage audits consume explicit process receipts and complete fixed
frame sets; finite errors, exact counts/transfers and retained-pool bounds are
checked separately from hardware class coverage. The isolated WSL2 Dozen
attempt retains failed default adapter admission without changing production
selection, core dependencies, shaders or device transfers. Capability-only
enumeration is not operation conformance. See
[WSL2 hardware evidence](WSL_HARDWARE_GPU_CONFORMANCE.md).

A separate experimental RGB-D capture adapter supplies fixed LK window/pyramid
settings to the original stateless and persistent-depth implementations. Existing
callers keep original settings; no global controls are mutated. Native/core/GPU
APIs, camera parameters, transfer ownership and all admission/accuracy gates
retain their contracts. Wider windows and extra pyramid work remain explicit.
See [the fixed capture protocol](WIDE_RGBD_CAPTURE_VALIDATION.md).

Robust PnP keeps its selected proper hypothesis through optional final refitting.
The same inlier-count/Euclidean-error ordering admits improvements; errors and
worse/tied candidates retain the selected pose with consistent original-order
rows. Existing owned output buffers are reused, with a restoration projection
pass on a valid rejected candidate. No public API, threshold, core dependency
or CPU/GPU transfer changes. See
[consensus-preservation evidence](PNP_CONSENSUS_VALIDATION.md).

Experimental RGB-D keyframes retain one owned source depth and original landmark
IDs while last-frame pixels advance through native LK. Metric PnP poses compose
through the same keyframe before world chaining; fixed promotion never resets
the world gauge. A separately fixed revision selects bounded local-depth
consensus landmarks and promotes from the previous frame before input-track
starvation, preserving the same native/geometry/accuracy gates. Source-bound
external inputs can share hard-linked PNGs under
a separately frozen plan, with no silent payload-copy fallback. Defaults and
full-cloud/failure/accuracy gates remain. See
[the fixed keyframe RGB-D protocol](TUM_KEYFRAME_RGBD_VALIDATION.md).

Public canonical frame evidence is an explicit offline adaptation of pinned
publisher configuration. Complete source hashes and every bounded sensor header
are checked before emitting the existing frame schema outside the repository.
It adds no runtime/core dependency, clock correction, or implicit transfer.
Publisher-configured extrinsics do not establish independently measured physical
accuracy; the existing missing-clock registration gate remains closed. See
[public frame provenance](PUBLIC_AUTOWARE_FRAME_EVIDENCE.md).

Public multi-adapter conformance uses fixed full RGB-D frames and explicit
upload/readback calls. Scalar integer images, two-dimensional integral blur,
float64 ordered voxel sums and brute-force radius counts are independent
references. Device class/backend and resource receipts are recorded without
local adapter names. Software execution is distinct from hardware coverage;
no shader, runtime API or production selection changes. See
[the fixed public GPU protocol](PUBLIC_GPU_RGBD_CONFORMANCE.md).

Public RGB-D motion validation may explicitly combine native grayscale,
Shi-Tomasi detection, forward/backward sparse LK and source-depth PnP through
an additive Python adapter. Inputs are owned before native tracking releases
the GIL. RGB/depth checksums and timestamp associations, previous/current pose
direction, original full-cloud support and complete lost-frame timelines are
preserved. Estimates are frozen before independent reference scoring; the
depth-only default and canonical Autoware calibration boundary remain. See
[the fixed image-motion protocol](TUM_RGBD_MOTION_VALIDATION.md).

Parallel neighborhood readback keeps the parent production decoder. A native
test-only direct-row candidate moves unchanged validation and collection into
the worker loop, but complete-call improvement did not repeat across initial
surveys and CPU controls regressed. Production selection, count-only/serial
paths, GPU work and explicit transfers remain unchanged. See
[direct parallel row evidence](../notes/2026-10-19_direct_parallel_neighborhood_rows.md)
for all native/exploratory comparisons and shared-host limitations.

Count-only GPU neighborhood readbacks at 65,536 or more queries use complete
integer-OR header validation before bulk owned outputs. Capacity-zero radius
counts preserve all u32 values; nearest-k requires both headers zero. Smaller
and nonzero-capacity results retain the parent decoder and worker policy. GPU
work, ownership and explicit transfers are unchanged. See
[count-only evidence](../notes/2026-10-19_gpu_neighborhood_count_readback.md)
for all native/exploratory comparisons and host-pass costs.

Explicit GPU neighborhood downloads may decode completed maps with at most four
native CPU workers per call. A bounded density/region probe selects large work;
all headers retain original validation, and every worker joins before unmapping.
Small, sparse, concentrated-sample, one-worker and WASM paths retain the original
serial collector. GPU work, ownership and explicit transfers are unchanged.
See [readback evidence](../notes/2026-10-19_parallel_gpu_neighborhood_readback.md)
for all final/initial measurements, resource costs and heuristic limits.

Resident GPU nearest-k search selects compile-time private candidate extents
1/8/16/64 while preserving all original traversal, distance and stable tie rules.
Batches below 1024 queries retain the original 64-entry array. Each requested nearest/radius program is lazy; mixed workloads may retain five
programs instead of the original eager pair. Admission, row schema, zero padding,
owned outputs and explicit transfers remain unchanged. See
[capacity specialization evidence](../notes/2026-10-19_specialized_gpu_neighborhood_capacity.md)
for complete parent/CPU comparisons and warm/first-use tradeoffs.

Saturated CPU-key voxel digit scans now retain sign representatives of checked
blocks. Omitted nibbles stay constant in each checked prefix; their AND/OR and
observed signs therefore merge exactly with the class-summary suffix and other
workers. Fallback repeats at most the current 16,384-point block per worker.
Exact selected masks, GPU work and explicit transfers remain unchanged. See
[checked-prefix evidence](../notes/2026-10-19_reuse_cpu_voxel_scan_prefixes.md)
for proof, controls and complete-call measurements.

CPU-key voxel sorting summarizes AND/OR variation separately by coordinate sign.
It retains the most-significant digit when both signs occur and one optional
within-class constant digit for even owned-output parity. A bounded probe keeps
full-width sorting; complete summaries use at most four native workers. Explicit
transfers, GPU-key sorting, shaders and compaction retain their contracts. Host
costs and all diagnostics are in [sign-class evidence](../notes/2026-10-18_sign_class_gpu_voxel_radix_digits.md);
see also the [original constant-digit evidence](../notes/2026-10-18_constant_cpu_voxel_radix_digits.md).

Resident GPU Vision chains borrow the uploaded source directly for grayscale
when both requested dimensions match, preserving original resize admission.
Only owned intermediates are recycled. The omitted identity dispatch removes
one texture acquire/write and profiling sample; receipts preserve incoming
history and report actual work with saturating counters. Changed shapes and
standalone resize retain original behavior. See
[identity-chain evidence](../notes/2026-10-18_identity_gpu_vision_chain_resize.md).

Eligible unpacked GPU box-blur columns use interleaved workgroup-local u32
prefixes for width >=128, height 512..4096 and both kernels >=259, with 16KiB
shared storage and `height*kernel_width*255 <= u32::MAX`. A seventh lazy final
pipeline combines with the horizontal choices while retaining original scratch,
two passes, receipts and explicit transfers. Outside these bounds the original
vertical pass remains available. See
[column-prefix evidence](../notes/2026-10-19_gpu_box_blur_column_prefixes.md)
for integer proof, ownership and warm/first-use tradeoffs.

Eligible wide GPU box-blur rows use a workgroup-local exclusive u32 prefix:
widths 512..4096, at least 128 rows and kernel width at least 259, with a 16KiB
workgroup-storage capability guard. A sixth lazy horizontal pipeline retains
the original bindings, dispatch, scratch, two passes and explicit transfers;
all five original programs remain fallback. Exact prefix differences and
original border masses preserve the final integer rounding. See
[row-prefix evidence](../notes/2026-10-19_gpu_box_blur_row_prefixes.md)
for bounds, ownership, complete-call and first-use costs.

Fallback unpacked GPU box blurs with at least 65,536 pixels and kernel widths
of at least 4,099 share one complete u32 horizontal sum across up to four outputs through
exact outgoing/incoming updates. Row crossings recompute complete sums and
partial final blocks write once per pixel. Original scalar/pair schedules, the five original variants, passes/resources
and explicit transfers retain their contracts. Cold-call costs and rejected alternatives
are recorded in [four-window evidence](../notes/2026-10-18_four_wide_gpu_box_blur_horizontal_windows.md).

Fallback large unpacked GPU box blur horizontal passes share exact adjacent
windows by subtracting the outgoing and adding the incoming byte. Row crossings use complete
sums; below 65,536 pixels the original scalar schedule is retained. The cutoff is
a private conservative heuristic. Full u32/signed bounds preserve rounding, host
planners, dispatch/resources and explicit transfers. See
[shared wide horizontal evidence](../notes/2026-10-18_shared_wide_gpu_box_blur_horizontal_windows.md).

Wide GPU box blur vertical passes share the original first window with the
next row by subtracting one outgoing and adding one incoming horizontal sum.
One-tap heights load directly; exact u32 bounds, signed coordinates and unique
row-pair ownership preserve rounding and odd final rows. Host planners, dispatch
requests, caches/resources and explicit transfers retain their contracts. See
[shared wide vertical window evidence](../notes/2026-10-18_shared_wide_gpu_box_blur_vertical_windows.md).

Packed GPU box blur vertical passes share the original first window with the
next row by subtracting one outgoing and adding one incoming horizontal sum.
One-tap heights load directly; exact u32 bounds, signed coordinates and unique
row-pair ownership preserve rounding and odd final rows. Host planners, dispatch
requests, caches/resources and explicit transfers retain their contracts. See
[shared vertical window evidence](../notes/2026-10-18_shared_gpu_box_blur_vertical_windows.md).

Large batched GPU voxel metadata key conversions use disjoint original-order
CPU bands at 524,288 occupied cells, bounded to four native workers including
the caller. One safely initialized final tuple vector adds host writes; small,
one-worker, WASM and serial-readback fallback keep original collectors. Mapping
scopes, staging/cap, other planes, GPU work and explicit transfers retain their
contracts. See [parallel metadata key evidence](../notes/2026-10-18_parallel_gpu_voxel_metadata_keys.md).

Explicit GPU voxel segment readback optionally batches unique keys, cell starts
and point indices into one staging allocation/submit/map/wait. Checked layout,
device limits and a 64MiB cap keep the original serial reader as fallback.
Logical transfer bytes, construction scalar sync and final ordering retain their
contracts; simultaneous staging requests grow, while one host key-record vector
is removed. See [metadata readback evidence](../notes/2026-10-18_batched_gpu_voxel_metadata_readback.md).

Packed GPU box blur horizontal passes reuse the first exact sum for adjacent
same-row pixels by subtracting the outgoing byte and adding the incoming byte.
One-tap widths load the source directly; row-crossing pairs keep complete sums
and odd tails keep zero upper halves. Nonnegative windows and the original
width bound of 257 samples preserve exact u16 packing and final rounding. Host
planners, shaders for other variants, caches, passes, resources and explicit
transfers keep their contracts. See
[adjacent box window evidence](../notes/2026-10-18_reused_gpu_box_blur_windows.md).

Non-singleton GPU voxel radix sorts optionally pack all 24 immutable controls
into one aligned per-invocation uniform buffer, with 16-byte static bindings.
Checked device limits and a 64KiB cap retain individual controls as fallback.
Shaders, pipeline layouts, passes, scratch and point-payload receipts keep their
contracts; alignment padding increases internal initializer bytes while source
uniform creation requests fall from 24 to one. Physical memory is unmeasured.
See [batched radix control evidence](../notes/2026-10-18_batched_gpu_voxel_radix_controls.md).

Q15 multi-channel Gaussian vertical filtering shares row mapping and weight
selection across two lanes for two channels and four lanes for three/four
channels. Exact componentwise base-65536 sums and original half-up rounding
preserve bytes and zero inactive components. Two optional private vertical
pipelines are lazily cached; original Q8/mono shaders, planner, buffers, passes,
receipts and explicit transfers retain their contracts. Warm-call diagnostics
and the additional cold-compilation/cache cost are recorded in
[shared Q15 row evidence](../notes/2026-10-18_shared_gpu_gaussian_q15_rows.md).

Large native CPU-key voxel uploads prepare disjoint original-order entry bands
with at most four workers including the caller, one safely zeroed 16*N host
buffer and worker-bounded metadata. Small, one-worker and WASM paths preserve
the original collector; validation remains before preparation. GPU passes,
scratch, ownership, explicit transfers and frame preflight retain their
contracts. The existing bytemuck dependency enables its safe allocation helper.
Exactness and all diagnostics are in
[parallel CPU entry evidence](../notes/2026-10-18_parallel_cpu_voxel_preparation.md).

CPU-key voxel construction uploads owned 16-byte sort entries with increasing
source indices prepared in the existing signed clamp loop. The prebuilt radix
route consumes that upload directly, removing the separate device entry
allocation and singleton build setup. Upload size, host vector capacity,
scratch descriptors, stable ordering and explicit scalar/payload transfers
retain their contracts; the GPU-key input route and frame preflight stay
unchanged. Exactness, allocation accounting and all paired diagnostics are in
[CPU entry upload evidence](../notes/2026-10-18_uploaded_cpu_voxel_entries.md).

Non-singleton GPU voxel sorts read keys directly for their first digit and
construct source indices with one private cached scatter shader. The complete
original radix shader, signed digits, stable ranks, later passes, owned outputs,
scratch and explicit transfers retain their contracts. One build pass/submission,
uniform/group and intermediate entry write disappear; singleton and empty
routes stay unchanged. Frame preflight records the removed stage per grid.
Exactness and all phased diagnostics are in [radix build evidence](../notes/2026-10-17_fused_gpu_voxel_radix_build.md).

Voxel compaction generates key boundaries inside its first inclusive-scan block.
The original four-way tree, upper hierarchy and prefix-based segment writing
preserve exact outputs. One mark pass/dispatch and its full flag write/read are
removed; distinct bindings, owned outputs, scratch requests and four-byte scalar
synchronization remain. Profiling records one fewer pass per grid, reflected in
frame preflight. Exactness and all paired diagnostics are in
[fused boundary evidence](../notes/2026-10-17_fused_gpu_voxel_boundary_scan.md).

Voxel compaction omits discarded per-point atomic cell counters. Owned segments
already retain keys/indices/starts, and existing readback derives lengths from
adjacent starts. A four-byte unused compatibility binding replaces the private
point-sized counter and its explicit clear; layout, pooling ownership and scalar
synchronization stay. Boundary generation now shares the first scan block.
Exact outputs/storage checks and all paired diagnostics are in
[unused counter evidence](../notes/2026-10-17_unused_gpu_voxel_counts.md).

Gaussian vertical filters share exact symmetric weights across mirrored rows.
Packed Q8 decodes channel halves before addition; Q15 adds separately bounded
base-65536 digits before multiplying. Final rounding, scratch, passes and explicit
transfers retain their parent contracts. Exact raw/full-call checks and all paired
vertical/resident/full timings are in [vertical symmetry evidence](../notes/2026-10-17_symmetric_gpu_gaussian_vertical.md).

Gaussian horizontal filters combine mirrored pixels with their exact shared
integer weight, keeping the center outside the pair loop. Nonnegative mass
bounds preserve u32 sums and packed high halves exactly; both sides retain the
original border mapping. Quantization, caches, resource/pass
counts and explicit transfers remain. Full parent/u64/CPU checks and all paired
diagnostics are in [symmetric Gaussian evidence](../notes/2026-10-17_symmetric_gpu_gaussian_horizontal.md).

Retain binary cached radix prefix trees. The correct four-way candidate halves
source tree barriers but repeats small large-input GPU regressions and does not
establish a completed transfer-inclusive benefit. Parent/candidate fixtures,
maximum-u32 offsets, stable full sorts and all phased measurements remain
native-only; production thresholds, caches, scratch, API and transfers stay
unchanged. See [radix candidate evidence](../notes/2026-10-17_four_way_gpu_radix_prefix.md).

Voxel boundary scans use a four-way exclusive tree before adding the original
value, preserving every inclusive u32 prefix while halving tree-stage barriers.
Tail lanes still participate; hierarchy/block offsets retain their original
integer results. The first block now also generates boundaries; scratch ownership
and scalar synchronization remain. Full parent, u64 and compactor checks
plus all paired timing rows are in
[four-way scan evidence](../notes/2026-10-17_four_way_gpu_voxel_scan.md).

Point-parallel RANSAC count reduction uses four disjoint blocks per stage,
halving its integer reduction barriers from eight to four. Bounded u32 totals
preserve exact scores; model/distance arithmetic, serial fallback, caches,
dispatch/resources and explicit uploads/readback retain their contracts. Raw
u64/parent/API checks and all paired pass/full-call timing rows are recorded in
[RANSAC count evidence](../notes/2026-10-17_four_way_gpu_ransac_counts.md).

Q15 Gaussian vertical filtering accumulates the horizontal input's lower/upper
base-65536 digits independently. Nonnegative mass 32768 bounds both sums to u32;
one final carry and half-up shift reproduce the full weighted result exactly.
One-channel Q8 keeps a bounded u32 sum; packed Q8, host planning/caches, scratch,
passes and explicit transfers retain their contracts. Full parent/u64 checks
and all paired timing rows are in [Q15 sum evidence](../notes/2026-10-17_split_gaussian_q15_sums.md).

Packed Q8 Gaussian vertical filtering reads each word once for two independent
channel sums. Nonnegative Q8 mass bounds make u32 accumulation and half-up
rounding exact; inactive odd-channel output stays zero. Q15 and one-channel
shaders, host planning/API/caches, scratch, passes and transfers remain unchanged.
Full parent/u64-CPU checks and every paired vertical/resident/full diagnostic are
in [paired Gaussian evidence](../notes/2026-10-17_paired_gaussian_q8_vertical.md).

Resident 5x5 median selection uses a padded odd-even merge network, removes
maximal-byte identity comparisons and prunes to the median dependency cone.
Generated compare-exchange pairs fall from 191 to 113; the 3x3 shader and shared
API/planner/sampling/resource/transfer behavior remain. Exhaustive binary and
complete parent/CPU checks plus paired GPU-pass/resident/full diagnostics,
including adverse full-call results, are in
[5x5 median evidence](../notes/2026-10-17_gpu_median_five_selection.md).

The owned RGBA row-append candidate remains native-test-only after shared-host
paired diagnostics failed to establish a latency benefit and included slower
cases. Production row packing, padded readback, borrowed dense uploads and the
contiguous readback shortcut retain the immediate-parent implementation. Frozen
public/CPU comparisons and all diagnostic runs are in
[deferred RGBA row evidence](../notes/2026-10-17_owned_rgba_row_copies.md).

Image readback unpacking copies the validated mapped prefix directly into an
owned vector for four logical channels with no staging row padding. Other
layouts retain the original row helper and workers. The output allocation and
copy remain; the source-level zero initialization and row overwrite loop are
avoided. Transfer plans, mapping lifetime, receipts, metadata and GPU work stay
unchanged. Immediate-parent CPU/public comparisons and paired host/completed-call
diagnostics are recorded in [contiguous RGBA evidence](../notes/2026-10-17_contiguous_rgba_readback.md).

The coalesced first-value fallback reuses consecutive identical 16-byte controls
within each original two-batch window. Its private single-entry cache compares
all uniform words and the descriptor label; changed keys create immutable
buffers while earlier commands retain theirs. Single-pass wrappers keep fresh
controls. Original pipeline availability errors, tails, bindings, output reuse,
window bounds and transfer/submit/map/wait counts remain. The preferred packed
route retains priority. Two full batches request one control instead of two;
width-four nineteen-channel fallback windows use three instead of five. Latency
is unmeasured. See [fallback control evidence](../notes/2026-10-17_reused_legacy_gather_controls.md).

Wide byte-attribute recorders reuse consecutive identical 16-byte uniforms in
both first-value and average modes. A private single-entry cache per invocation
compares every original control word; changed keys create immutable buffers and
earlier queued passes retain theirs. Each pass still selects its original
pipeline. Small-batch wrappers retain fresh controls. Original f32 rounding,
scalar tails, packed padding, planners and transfer/submit/map/wait counts remain.
At nineteen byte channels, wide-recorder uniform requests become five to two;
at twenty, five to one. These are source requests; latency is unmeasured. See
[u8 control evidence](../notes/2026-10-17_reused_voxel_u8_controls.md).

Batched f32 voxel averages reuse consecutive identical 16-byte controls within
each standalone call or combined attribute recorder. A private single-entry
cache compares every original uniform word; a changed key creates a new
immutable buffer so queued passes retain their parameters. Shaders, scalar
tails, planners, input/output ordering and payload/submit/map/wait counts remain.
At a four-wide recorder receiving nineteen attributes, uniform creation requests
become five to two; outer fused prefixes keep their controls. Latency is unmeasured.
See [average control evidence](../notes/2026-10-17_reused_voxel_average_controls.md).

Packed first-attribute recorders reuse consecutive identical 16-byte uniforms
within each combined call or bounded standalone window. A private single-entry
cache compares all original control words; changed parameters create a new
immutable buffer, and queued commands retain earlier resources. Calls/windows
do not share controls. Original packed/scalar tails, shaders, input/output
ordering, planners and payload/submit/map/wait counts remain. Four-wide combined
nineteen-attribute control requests become five to two; latency is unmeasured.
See [packed control evidence](../notes/2026-10-17_reused_packed_gather_controls.md).

Native image histogram and CDF readbacks decode their borrowed mapped bytes
directly into owned `[u32; 256]` arrays using the shared unmap guard and original
little-endian integer loop. Each call removes one intermediate host byte vector
and its 1,024-byte copy; the logical 1,024-byte download, staging/copy/submit/map/
wait counts and success-only receipts remain. The unused byte-copy wrapper is
test-only. This is host allocation/copy cleanup; latency is unmeasured. See
[histogram readback evidence](../notes/2026-10-16_direct_mapped_histogram_readback.md).

The combined XYZ/first-attribute fallback reuses one private active output array
per recorder invocation. Every batch copies its active prefix to the original
unified staging offset before a later batch overwrites it. Original scalar/two/
four-channel kernels, tail rules, uploads and dummy bindings remain; preferred
packed/fused priority and transfer/submit/map/wait counts are unchanged. Active
output requests become `A -> min(A,W)` for attribute count `A` and original width
`W`; descriptor sums do not establish physical memory or latency savings. See
[combined output reuse evidence](../notes/2026-10-16_reused_combined_voxel_outputs.md).

Legacy combined XYZ/first-attribute gathers allocate only missing four-channel
output bindings: one dummy for three active channels and none for four, rather
than four dummies per batch. The extracted private block preserves active
bindings, the existing empty read-only input, shader math and output copies;
packed/fused priority, scalar/two-channel paths and transfer/wait counts remain.
These are fewer source buffer requests rather than measured physical savings.
See [inactive output evidence](../notes/2026-10-16_trimmed_voxel_gather_dummies.md).

The legacy coalesced voxel first-value fallback reuses one two-/four-buffer
output array per bounded two-batch submission window. Active planes copy to
their original staging offsets before overwrite; original kernel selection,
point-zero empty-cell tails, uploads, planners and submit/map/wait counts remain.
The preferred packed route keeps priority. Two full four-channel batches request
four rather than eight outputs; these are source requests rather than measured
physical/peak savings. See
[legacy output reuse evidence](../notes/2026-10-16_reused_legacy_voxel_outputs.md).

Scalar `f32` staging readback shares the existing scope-based unmap guard with
XYZ/attribute readers. After a successful map, the view drops before unmapping
on return or panic unwinding, so an internal bounds panic no longer leaves the
buffer mapped. Mapping errors, valid-prefix bits/capacity, public callers,
shaders and transfer/wait counts remain unchanged. See
[scalar readback lifetime evidence](../notes/2026-10-16_guarded_scalar_gpu_readback.md).

Resize/Sobel and CHW image packing initialize private 32-/64-byte uniforms with
mapped initial contents, as existing blur/morphology kernels do. A resident
resize -> Sobel -> CHW chain removes three explicit `queue.write_buffer` calls
while retaining three uniform buffers, three compute passes and three explicit
submissions. Parameter delivery still occurs; internal backend staging/copies
and latency are unmeasured. Kernels, validation, profiler events, pools, owned
results and image/tensor payload receipts remain unchanged. See
[image uniform initialization evidence](../notes/2026-10-16_initialized_image_uniforms.md).

The optional `register-icp-gpu` correspondence API explicitly stages a borrowed,
backend-neutral CPU KD-tree layout into an immutable runtime-owned `GpuKdTree`.
Target buffers survive all ICP iterations; query uploads and nearest-result
downloads are named and counted. Near-first GPU traversal keeps the tree's
tie order and inclusive distance gate. Pose solving, transforms, trimming and
f64 fitness accumulation stay on CPU. Runtime mismatches and unsupported buffer
or dispatch sizes reject before device resource use; ordinary ICP placement is
unchanged. This is a native synchronous search boundary, not a claim that the
entire registration loop is device resident.

Large public `GpuQueryBatch::upload` calls prepare original-order AoS queries
with bounded native CPU workers: at least 524,288 points, useful 262,144-point
bands and a four-worker/available-CPU cap. Each band validates finite XYZ and
writes unchanged coordinate bits plus positive-zero padding into disjoint
parts of the existing 16*N-byte host payload. Small/one-worker/WASM inputs keep
the original serial scan and collector; small calls avoid CPU-count queries.
Device limits are checked first and all scoped work finishes before device
resource creation. Large invalid inputs can allocate/zero the payload before
rejection. Worker handles/stacks add bounded overhead; no additional point-sized
copy, GPU pass, wait, readback or transfer is introduced. Registration source
uploads share this preparation; it does not change algorithm placement or make
whole registration device resident. See [AoS query staging evidence](../notes/2026-10-15_parallel_aos_query_staging.md).

Large `GpuKdTree::upload` calls validate original coordinates and gather
16-byte XYZ/original-index records in bounded native CPU bands. At least
524,288 points use available CPUs and useful 131,072-point bands, capped at
four workers. Finite validation covers every original coordinate independently
of the tree's point permutation; disjoint writes retain leaf traversal order
and original indices. The existing 16*N point allocation and 32*M node staging
remain, with worker-sized metadata and serial node packing. Small/one-worker/
WASM inputs retain serial finite scans and point gathering. Device limits
precede host packing, and all workers succeed before device buffers are created;
large invalid inputs may first allocate bounded host point storage. Root,
nodes, ties, payloads, transfers and search kernels are preserved. See
[tree input staging evidence](../notes/2026-10-14_parallel_gpu_tree_staging.md).

The additive `GpuKdTree::upload_with_gpu_gather` explicitly uploads original
SoA axes and the CPU tree's permutation, then copies integer words into the
unchanged traversal-order point records on GPU. The companion
`upload_and_queries_with_gpu_gather` produces padded original-order queries in
the same pass. CPU partitioning, bounded finite scans and serial node staging
remain. Combined tree/query host payload becomes 16*N + 32*M bytes, compared
with 32*N + 32*M for separate uploads; tree and query receipts each record
16*N device output bytes. Uniform controls are excluded. Two transient input
buffers add 16*N device bytes beside owned outputs; host point/query packing
vectors are removed. The queue retains transient resources and orders searches
without a host wait or readback. Pipeline variants initialize lazily per runtime.
Empty inputs allocate no resources. Existing buffer/compute requirements apply;
the paired API also checks the complete query dispatch count. Default upload,
search kernels and feature/registration placement are preserved. These explicit
APIs expose a transfer/memory tradeoff whose latency depends on input size; see
[GPU gather evidence](../notes/2026-10-14_gpu_tree_point_gather.md).

`GpuFpfhInput`, `GpuIssInput` and `GpuMlsInput` expose explicit
`upload_with_gpu_gather` constructors using those shared tree/query outputs.
ISS/MLS host payload becomes 16*N + 32*M bytes, with 32*N device output bytes.
Below 1,048,576 points, FPFH combines borrowed original-order coordinates and raw normals in one
24*N-byte mapped input, beside the 4*N-byte permutation. One integer pass
generates traversal-order tree points, queries and unchanged 32-byte
position/normal samples together. Larger inputs retain the previous two-pass
tree/query then sample construction. Host payload is 28*N + 32*M bytes, with 64*N
device output bytes. CPU finite/unit-normal/extent validation and node staging
remain; compute kernels and default upload/high-level placement are preserved.
Feature input validation uses bounded native CPU bands for at least 524,288
points: useful 131,072-point bands, available CPU count and a four-worker cap.
Each band scans every original SoA coordinate for finite values while finding
axis extrema; FPFH unit normals are checked in those same workers using the
original f32 multiplication/addition order and tolerance. Bounds merge before
the original axis-ordered f64 difference/square/sum and finite f32 extent gate.
This adds only worker-sized metadata, with no coordinate copy, device resource,
readback or transfer. Small/one-worker/WASM inputs retain serial scans. Feature
planners and normal-length precedence remain before validation; feature-specific
errors are preserved and device resources are created after successful validation.
See [feature validation evidence](../notes/2026-10-14_parallel_feature_validation.md).

No readback or host wait is added. The fused variant is lazy in the runtime's
shared tree-gather cache; the separate sample stage remains for large inputs.
Each submitted pass retains its inputs. Relative to owned buffers,
transient GPU inputs use 28*N bytes for fused FPFH (up to 28*N for the large
two-pass path) and 16*N for ISS/MLS, while
host packing allocations totaling 64*N and 32*N payload bytes respectively
are removed over construction; these sums are not peak-memory savings. Backend staging
overhead is additional. These constructors require the existing feature device
limits; small inputs may be slower. See
[feature gather evidence](../notes/2026-10-14_gpu_feature_input_gather.md).

The fused FPFH path retains the existing five-storage/six-binding
device requirements. Its private planner checks 32*N sample bytes, which also
bound the combined 24*N input and 16*N outputs, and the full query dispatch
count before allocation. The cutoff preserves the previous large-input path
after unconditional fusion regressed shuffled million-point calls in both
surveys; it is a conservative policy, not a universal performance threshold.
Tree-only/paired query entry points, public methods,
default FPFH upload, unit-normal/extent validation, nodes and descriptor kernels
are preserved. Complete two-pass/fused input comparisons, including adverse and
inconclusive cases, are recorded in
[FPFH pass fusion evidence](../notes/2026-10-14_fused_fpfh_input_gather.md).

The same feature adds `align_with_gpu_reduction` for untrimmed point-to-point
ICP. `GpuIcpInput` explicitly uploads original source, original-order target
and target tree once. Six ordered passes per evaluation transform originals,
perform the unchanged bounded nearest traversal and reduce centroids/fitness
then centered cross-covariance. Actual transformed anchors, normalized
compensated f32 sums and exact counts form an owned 176-byte result; private
transformed/match/partial scratch is leased through one submission. CPU Horn's
small eigensolve, transform composition and stopping remain. Post-update moments
serve the next update and final fitness, avoiding duplicate search/readback.
There is no point/match readback, repeated payload upload or CPU fallback.
f32 transforms/reductions can change matching, pose and stopping near gates or
numerically degenerate geometry. See
[resident ICP evidence](../notes/2026-10-13_gpu_icp_reduction.md).

`GpuIcpInput`, `GpuGicpInput` and `GpuPointToPlaneInput` also expose explicit
`upload_with_gpu_gather` constructors. The shared target tree/query integer
gather uploads 16*T + 32*M bytes and generates 32*T device output bytes,
removing separate target host packing allocations totaling 32*T payload bytes
over construction. Two transient buffers add 16*T device bytes beside owned
outputs; this is a transfer/allocation tradeoff, not a peak-memory saving.
Total host payload is 16*S + 16*T + 32*M for ICP, 48*S + 48*T + 32*M for GICP
and 16*S + 32*T + 32*M for point-to-plane. Source packing, finite/extent/anchor
preparation, CPU target partitioning and node staging remain. GICP SPD
covariance preparation and point-to-plane common normal scaling keep their
order and arithmetic. Outputs and both device output receipts remain owned;
no wait/readback is added. Existing device requirements apply. Default inputs,
high-level placement and reduction kernels remain; small inputs may be slower.
Latency depends on input size/order and backend, and can regress even for large
calls; completed-upload measurements retain those cases without auto-selection.
See [registration gather evidence](../notes/2026-10-15_gpu_registration_input_gather.md).

Both point-to-plane upload routes prepare large raw-normal inputs in two bounded
native CPU phases. From 262,144 rows, available CPU count and useful 65,536-row
bands select at most four workers. Finite absolute maxima merge exactly before
the existing zero/normal-scale gate. Disjoint output bands then retain the
original three f32 divisions, nonzero-component underflow rejection, original
order and zero padding. The 16*T-byte host payload allocation remains; parallel
packing zero-initializes it and adds worker-sized metadata and thread stacks.
All scopes join before device construction. Count/device guards still precede
normal preparation and geometry validation still follows it. Small/one-worker/
WASM calls remain serial. Reduction math, GPU passes, ownership and explicit
receipts are preserved. See
[normal staging evidence](../notes/2026-10-15_parallel_plane_normal_staging.md).

The shared `GpuIcpInput` upload geometry preparation checks finite XYZ and
collects per-axis bounds in bounded native chunks, at most one worker per
65,536 points and available CPU parallelism. Source and target are prepared
sequentially. Below 262,144 points, one worker and WASM retain the original
serial scans and skip the CPU-count query. Each worker returns a 24-byte
bounds summary; no extra point-sized staging vector is introduced. Original f64
midpoint, difference, square and axis accumulation order remain. Derived zero
anchors are now consistently `+0` in serial/native/WASM paths;
input point bits, including signed zeros, remain unchanged. Rust min/max did not
guarantee a stable sign for equal zeros; nonzero anchors and extents stay exact.
Existing count/device checks precede this
work, and geometry errors still precede device input allocation. This shared
boundary also serves GICP, point-to-plane and resident covariance constructors.
Payload, input layouts, kernels, result ownership and receipts remain unchanged.
See [registration geometry evidence](../notes/2026-10-14_parallel_gpu_registration_geometry.md).

Shared registration input upload now prepares target bounds directly from the
CPU KD-tree's borrowed original-order SoA coordinates and packs those coordinates
directly into the existing 16-byte query records. This removes the intermediate
12*T host-byte AoS target vector; the 16*T padded query staging and tree-order
point staging remain. Target geometry retains the same native worker bounds,
serial small/WASM placement, f64 arithmetic and canonical zero anchors. Private
SoA helpers reject unequal coordinate lengths and nonfinite values; count/device
limits precede query packing, and target geometry errors precede device input
allocation. Source staging, original target indices, device layouts and shaders
are unchanged. Upload remains 16*S + 32*T + 32*M bytes, with no additional device
copy or readback. See
[direct target staging evidence](../notes/2026-10-14_direct_soa_registration_upload.md).

`align_with_gpu_trimmed_reduction` and `GpuIcpInput::trimmed_moments_resident`
extend the resident path with stable MSD selection by device distance and original
source index. Four distance bytes and the necessary source-index bytes determine
an exact retained threshold without sorting or copying matches to CPU. Device
integer limbs emulate binary64 multiplication/round-to-nearest-even/floor for
`trim_fraction*gated_count`; casting the fraction to f32 is avoided. Retained
pairs define centroids/covariance, while ALL gated pairs define fitness and its
stopping denominator. A 176-byte result adds the all-gated count in reserved word
2; fraction one delegates to ordinary kernels with both reserved words zero.
Private histogram/selection scratch stays leased through the same submission.
CPU target-tree construction/Horn solving/composition/convergence remain; there
is no payload readback or fallback. GPU distance rounding can change selection
near ties compared with CPU, even when retained counts agree. See
[initial resident trimming evidence](../notes/2026-10-13_gpu_icp_resident_trimming.md).

Trim digit selection uses 256 lanes (one bin per lane) when there are
at least 64 partial histogram groups and the device supports 256 invocations
and x lanes. This separately lazy wide pipeline builds a 255-node integer sum
tree in eight synchronized levels, then selects the byte and within-bin rank
with eight decisions. It reuses the declared histogram workgroup array beside
the bin totals: 2 KiB shared storage under the existing planning requirement,
with eight additional workgroup barriers. Other inputs/devices retain the
64-lane linear rank walk (four bins per lane). Histogram group counts, 256 KiB
maximum global histogram scratch, digit/pass count, tie ordering, binary64
retained-count arithmetic and fixed result readbacks are unchanged. See
[histogram lane evidence](../notes/2026-10-13_gpu_icp_trim_selector.md) and
[selection tree evidence](../notes/2026-10-13_gpu_icp_trim_selection_tree.md).

Resident trimming now defaults to `GpuIcpTrimPrecision::DoubleSingle`. Integer
24x24-bit products retain residuals, and a private runtime zero word in dependent
bitcasts preserves rounded f32 values through compensated arithmetic. Two f32
words carry coordinate differences, normalization, mean division and centered
products, without relying on a fused WGSL `fma`. Explicit `*_with_precision`
methods offer the previous `CompensatedF32` arithmetic. Reduction pipelines are
lazy per runtime and precision; the unchanged selector and scratch are shared.
No new pass, buffer layout, transfer or limit is added. Fraction one delegates
to canonical ordinary moments for either choice. Transform/search and all-gated
fitness keep their f32 arithmetic, and CPU solving/composition/convergence remain.
The earlier large partially gated case now meets the original pose tolerance;
near-gate/tied/degenerate f32 matching can still change other trajectories. See
[trimming precision evidence](../notes/2026-10-13_gpu_icp_trimmed_precision.md).

GICP reuses this search through its own explicit GPU entry point, retaining
unbounded nearest matching before the existing host gate and ordered CPU
Hessian/gradient accumulation. Radius covariance, when configured, runs on the
caller-provided runtime and reports all input/grid payload and padded output
transfers; errors propagate. Without a radius, covariance remains CPU k-NN.
The legacy native shared-runtime covariance route keeps its fallback contract.

`GicpRegistration::align_with_gpu_reduction` under the same feature retains
source/target/tree and packed SPD covariances in `GpuGicpInput`. Five device
passes transform originals, search with the existing unbounded traversal,
apply the inclusive gate, prepare rotated covariance inverses, and reduce
Hessian/gradient/Euclidean error. After nearest matching, transformed query rows
are reused for centered position/count rows; inverse weights and residuals occupy
48 separate bytes per source. A complete local pair is prepared before each row
is stored once, including zero rows for rejected pairs. Private working storage
is 72*S + 240*min(ceil(S/64),256) bytes, down from 88*S plus the same partials.
The five passes, covariance/geometry checks and compensated sum order remain.
Centered coordinates, distance normalization, a target-inverse trace weight
bound and compensated sums limit dynamic range.
Owned 256-byte equations/status are explicitly downloaded; CPU solves the
normalized 6x6 system and converts the linearized step to the existing world
update/stopping convention. Post-update equations serve the next solve/final
fitness. Covariance preparation stays CPU k-NN or explicit radius-GPU readback
and re-upload with complete receipts. The existing CPU k-NN preparation for this
GPU entry point uses at most one scoped worker per 16,384 points and available
CPU parallelism; below 32,768 points, one worker and WASM stay serial. Each row
retains exact neighbor/accumulation/eigenvector order and packs six f32 components
directly, avoiding the intermediate 72*S f64 matrix vector. Ordinary CPU GICP
retains serial placement and the shared per-point math. Unsupported covariance
conditioning or nonfinite arithmetic invalidates all result payload. GPU f32 math and normalized
solving can change pose/stopping; ordinary GICP remains unchanged. See
[GICP reduction evidence](../notes/2026-10-13_gpu_gicp_reduction.md) and
[GICP working-set/preparation evidence](../notes/2026-10-13_gpu_gicp_reuse_transformed_queries.md).

Explicit `GpuGicpInput::upload` validates and packs covariance rows together
before any device allocation. Native batches use at most one scoped worker per
16,384 combined source/target covariance rows and available CPU parallelism;
fewer than 32,768 rows, a single worker and WASM stay serial. Each row keeps
the original f64 Cholesky/inverse-trace arithmetic and exact f32 components,
including signed zero and two zero padding words. Only positive finite target
inverse traces enter the maximum, so splitting that bound changes no rounding.
Device limits and row counts are checked before staging; covariance errors
return before GPU input allocation. Packed rows occupy the existing 32*(S+T)
host bytes and remain live through position upload. Device payload, five passes,
scratch, weight scale, result/status and receipts are unchanged. GPU-generated
covariance constructors continue through their existing resident paths. See
[covariance upload staging evidence](../notes/2026-10-14_parallel_gicp_upload_preparation.md).

`GicpRegistration::align_with_gpu_resident_covariances` adds explicit k-NN
preparation before that same resident update loop. `GpuGicpInput::upload_with_gpu_covariances`
uses original positions for both queries and covariance samples, reuses the
uploaded target tree, and temporarily uploads the source tree. Two bounded
nearest-k passes and covariance/validation passes compute plane regularization
on GPU. Centered/scaled compensated sample covariance and 32-step Jacobi
normal eigenvectors preserve the physical 1e-12 stopping convention. A
32-byte device status prefix makes any bad preparation invalidate all later
256-byte equations; covariance/neighborhood payloads never cross to CPU.
Upload is 32*(S+T) + 32*(source_nodes+target_nodes) bytes. CPU tree construction
and the existing normalized solve/world update remain. This optional k=3..=64,
epsilon=1e-4..=1 path requires covariance_radius=None; f32 covariance and
repeated eigenvalues can change poses/stopping relative to old raw products.
See [resident covariance evidence](../notes/2026-10-13_gpu_gicp_resident_covariances.md).

`align_with_gpu_resident_radius_covariances` and
`GpuGicpInput::upload_with_gpu_radius_covariances` extend this preparation
boundary to complete inclusive radius rows with an explicit positive capacity.
Self/duplicates count toward capacity. Overflow zeros BOTH covariance sets
and propagates equation status 16; no truncated neighborhood is consumed.
Sparse rows (fewer than three samples) retain epsilon*I. Other rows use
centered/scaled compensated central sums and 32 max-pivot Jacobi iterations
with physical 1e-20 stopping, so grid accumulation/sweeps and degenerate
eigenvectors may differ. Finite positive radius with normal f32 square and
epsilon=1e-4..=1 are required. Upload/download ownership and receipts match
resident k-NN. CPU trees/6x6 updates remain; legacy preparation is unchanged.
See [resident radius evidence](../notes/2026-10-13_gpu_gicp_resident_radius_covariances.md).

`align_with_gpu_resident_grid_covariances` / `upload_with_gpu_grid_covariances`
offer explicit dense-grid preparation. CPU counting-sort indices/offsets are
uploaded into private scratch leased through one submission. Two grid gathers
share the existing original query/sample buffers and write directly into owned
packed covariances; a final pass zeros both sets on invalid data/support/SPD.
The canonical grid arithmetic is reused, preserving raw f32 means, central sums
and 16 three-pivot sweeps. Support capacity includes self/duplicates but requires
no neighborhood allocation; overflow rejects all equations. Source KD-tree
construction/upload is removed. Upload is 20*S+36*T+32*target_nodes plus four
bytes per grid offset entry. CPU grid construction/target tree/6x6 solve remain;
dense-grid cell limits reject without automatically selecting another route.
See [resident grid evidence](../notes/2026-10-13_gpu_gicp_resident_grid_covariances.md).

The optional CPU search `parallel` feature exposes explicit KD-tree construction
with bounded scoped subtree workers. One exact-sized preorder node array and
the point permutation are split into disjoint ranges, with no unsafe, locks,
per-worker trees or node merge. Canonical axis/partition code preserves all
layout/query bits. Ordinary constructors stay serial; small/one-worker/WASM
parallel calls use them directly. Resident GPU GICP opts into this host
preparation without altering device staging or equations. No production
dependency or core model change is introduced. See
[parallel tree evidence](../notes/2026-10-13_parallel_kdtree_build.md).

`PointToPlaneIcp::align_with_gpu_reduction` under the optional
`register-icp-point-to-plane-gpu` feature retains original positions/tree and
raw target normals. Five passes transform originals, perform the existing
unbounded nearest search followed by the inclusive Euclidean gate, prepare
centered/scaled Jacobian rows and reduce the symmetric 6x6 Hessian, gradient and
raw-normal plane squared error. One common normal scale preserves relative
length weights and zero rows; there is no per-normal unit normalization.
After nearest matching completes, each invocation reuses its transformed query
row for the rotation Jacobian and residual. A separate 16-byte row keeps the
normal and contribution flag. Nonempty private scratch is
40*S + 240*min(ceil(S/64),256) bytes, down from 56*S plus the same partials;
the five-pass ordering and compensated arithmetic stay unchanged. Preparation
builds a complete local pair and stores each row once, including zeros for
rejected pairs, before reduction.
Owned 256-byte equations/status are downloaded explicitly and reused after each
update. CPU tree construction, normalized 6x6 solve, linearized world-step
conversion and existing finite pose composition/stopping remain. Receipts count
16*S + 48*T + 32*M upload bytes and 256 bytes per evaluation. Invalid numeric
scales, nonfinite arithmetic or mismatched runtimes reject without fallback;
f32 rounding can change matching, pose and stopping. See
[point-to-plane evidence](../notes/2026-10-13_gpu_point_to_plane_reduction.md) and
[query storage reuse evidence](../notes/2026-10-13_gpu_plane_reuse_transformed_queries.md).

Reusable finite `GpuQueryBatch` uploads and owned `GpuNeighborhoods` make the
next feature-compute boundary explicit. GPU KD-tree k-NN/radius search writes
fixed-capacity rows with stored/total counts and original index/distance pairs.
Nearest-k rows retain CPU stable ordering; radius rows retain traversal order
and expose overflow. Complete CPU readback rejects incomplete neighborhoods;
GPU consumers must inspect counts before using rows. Target/query payloads
remain resident between passes. This adds no core/search dependency and does
not change ordinary feature algorithm placement; explicit device consumers follow.

FPFH, ISS and MLS upload constructors now pass the CPU KD-tree's borrowed
original-order SoA coordinates directly to the existing private query packer.
FPFH samples zip those coordinates with the original unit normals. This removes
one 12*N-byte host position vector per upload; padded query/sample staging and
the caller's CPU tree/normal storage remain. Original coordinate/normal bits,
positive-zero padding, validation order, device records, transfers and kernels
are preserved. This is a CPU staging/allocation improvement at the explicit
upload boundary, not a new GPU algorithm or CPU parallel placement change. See
[direct feature staging evidence](../notes/2026-10-14_direct_soa_feature_upload.md).

The shared SoA query packer uses bounded native CPU workers for at least
524,288 points. Useful workers require 262,144 points each and are capped by
available CPUs and four workers. Each worker validates finite axes and fills
its disjoint band of the existing 16*N-byte padded query allocation; all bands
finish successfully before a device buffer is created. Original point bits,
order and positive-zero padding are retained, with only worker-sized dispatch
metadata and no point-sized join allocation. Small/one-worker/WASM packing
keeps the serial scan and collection. The explicit GPU upload boundary, limits,
receipts and consumer kernels remain the same. See
[query staging evidence](../notes/2026-10-14_parallel_soa_query_staging.md).

FPFH device computation now consumes this radius-row representation through
`GpuFpfhInput` and neighborhoods tied to that exact uploaded input. Explicit
input upload retains original positions/unit normals plus target/query buffers.
SPFH normalization and ordered distance-weighted aggregation run on GPU in two
passes sharing one submission. GPU status detects capacity overflow or malformed
rows before downstream feature use; invalid results are zeroed and explicit
download rejects them. No neighborhood/SPFH readback is needed. The optional
`register-fpfh-gpu` wrapper retains CPU validation/tree construction and exposes
complete transfer receipts; CPU descriptor matching and RANSAC are unchanged.

ISS device computation also consumes input-bound radius rows via `GpuIssInput`.
The uploaded query positions double as covariance samples. Two device passes
validate both graphs, compute compensated local covariance/Jacobi saliency and
select local maxima. Incomplete/malformed graphs or numerical underflow invalidate
all flags/scores. Owned result buffers outlive their inputs; GPU consumers must
inspect status. `feature-iss-gpu` adds explicit detection with transfer receipts,
retaining CPU tree construction, ordered flag compaction and attribute gathering.
GPU f32 selection can differ near thresholds/ties from CPU f64; ordinary CPU
behavior remains unchanged. Eigen diagnostics require separate explicit readback.

MLS uses the same input-bound radius representation through `GpuMlsInput`.
The uploaded query positions are shared with frame computation. Three device
passes in one submission validate complete neighborhoods, compute compensated
weighted frames/Jacobi eigenvectors, fit order-0/1/2 surfaces and finalize global
validity. Private leased scratch stores 64-byte frames and 192-byte fit systems
per point; owned original-order position/mode buffers survive input drops.
`filter-mls-gpu` keeps CPU tree construction and schema/attribute reconstruction
at explicit upload/download boundaries. Minimum support includes self, then
fitting excludes only the query index. Sparse points retain their coordinates;
insufficient polynomial support or rank failure uses a device plane fallback.
Normalized f32 Cholesky uses relative rank threshold 1e-6; CPU f64 Gaussian
elimination uses absolute 1e-12. Nearly singular fits can differ substantially;
per-point modes and counts expose outcomes, without a CPU equivalence promise.
Malformed/incomplete rows or evaluated local-extent underflow invalidate and
zero all output; consumers must inspect the first five status words. See
[MLS evidence](../notes/2026-10-13_gpu_mls_smoothing.md).

Standalone resident-input `gather_voxel_first_xyz_gpu_buffers` and
`reduce_voxel_centroids_xyz_gpu_buffers` decode their mapped XYZ planes directly
into three owned coordinate vectors. The former flat vector and Y/Z splits
requested 20*N host payload bytes across the returned vectors, with 8*N bytes
copied again and retained as excess X capacity. Direct decoding requests 12*N
bytes and performs one copy per plane. GPU output/staging, 12*N download bytes,
compute/shader math, uploads and public signatures are preserved. A private
unmap guard releases the view before unmapping on return or decoder unwind;
the existing float cast and map diagnostics remain. Empty calls retain their
resource-free return. These payload/capacity counts do not establish process peak memory
or application latency. See
[voxel XYZ readback evidence](../notes/2026-10-15_direct_mapped_voxel_xyz.md).

Standalone `gather_voxel_first_f32_multi_gpu` and
`reduce_voxel_average_f32_multi_gpu` copy mapped channel planes directly into
their owned outputs through the existing unmap guard. A K-channel readback
batch requests 4*K*N float payload bytes instead of 4*(2*K-1)*N, removing
4*(K-1)*N repeated-copy bytes and excess first-channel capacity. Gather keeps
its adapter-selected two/four-channel batches; reductions keep one aggregate
readback. Four channels at 1,048,576 cells remove 12 MiB for a four-channel
batch, or 8 MiB over two two-channel gather batches. These are requested
payload/capacity counts, not RSS/peak-memory or latency guarantees. Channel
ordering, empty/length validation, per-call uploads, dispatches, map/wait,
GPU buffers, shaders and public signatures remain. See
[voxel channel readback evidence](../notes/2026-10-15_direct_mapped_voxel_channels.md).

The four combined XYZ/float/u8 gather/average entry points also copy mapped
planes directly into their final owned vectors. With F = 3 + float attributes,
U byte attributes and N output cells, float payload capacity becomes 4*F*N
instead of 4*(2*F-1)*N. Byte results decode the low eight bits of each staged
u32 directly into each final byte channel, removing a 4*U*N raw host allocation
and, for U > 1, a U*N flat-byte allocation over construction. The guard drops
the mapped view before unmapping on return/unwind;
upload recycling follows successful final host decoding. Counts exclude vector
metadata and do not establish peak memory or latency. See
[combined voxel readback evidence](../notes/2026-10-15_direct_mapped_voxel_mixed.md).

Combined XYZ/attribute calls now use packed u8 gather/average outputs. All
256 lanes compute their original cell value in parallel, write a 1 KiB shared
array, then reach one uniform workgroup barrier. The first 64 lanes each write
four consecutive low-byte results as one u32; inactive tail lanes write zero.
Each byte channel's output/copy/staging stride becomes 4*ceil(N/4) bytes,
instead of 4*N. Float planes remain 4*F*N bytes. No compute pass, atomic,
submission, readback wait or input transfer is added. Per-cell gather and
sequential f32 average/round arithmetic remain identical to the original WGSL.
The existing public word-output pipelines/layouts retain their behavior; two
private packed variants initialize lazily per runtime using those same layouts.
On little-endian hosts the mapped decoder copies byte planes into final owned
vectors; big-endian hosts extract the corresponding u32 bit offsets. Neither
route creates an intermediate host array. For 1,048,576 cells and three byte channels,
byte downloads fall from 12 MiB to 3 MiB; byte device outputs and the byte part
of staging each fall by 9 MiB. Counts exclude driver allocation overhead and
per-workgroup scratch. Performance depends on workload/backend and barrier
cost; requested device limits and public API validation remain unchanged.
See [packed byte output evidence](../notes/2026-10-15_packed_voxel_u8_output.md).

`reduce_voxel_centroids_xyz_and_average_multi_gpu` uses a lazy fused float
pipeline for one-to-four float attributes when device limits permit. One lane
per cell loads each point index once and accumulates XYZ and active attributes
in their original order. XYZ retains sum-times-reciprocal arithmetic; float
attributes retain sum-divided-by-count arithmetic. A planar output replaces
the separate float outputs without changing their summed payload or download.
For four attributes, float passes/bind groups/uniforms fall from five to one,
and float output buffers/copies from seven to one. Byte passes and packed
downloads remain unchanged; there is still one submission and mapped readback.
The private planner requires ten storage bindings, eleven total bindings,
256-lane compute support, a valid dispatch count and a packed output that fits
storage-binding/buffer/u32-index limits. Zero/more-than-four attributes and
unsupported sizes/capabilities keep separate XYZ/attribute commands; the
attribute-average branch may use the bounded four-channel batch below. No
device feature/limit request, public API, input transfer or host packing is
added. Inactive read-only attribute bindings alias the first active input;
the shader skips them. See
[fused float average evidence](../notes/2026-10-15_fused_voxel_float_average.md).

`gather_voxel_first_xyz_and_average_multi_gpu` uses an independent lazy kernel
under the same unchanged fused-output planner. It copies first-point XYZ
without arithmetic while accumulating one-to-four attributes in the original
order with scalar division. Empty cells retain input point-zero XYZ and
positive-zero attributes. Four attributes need one float pass/output/copy
instead of five passes and seven outputs/copies. Byte commands, payloads,
mapped decoding and the separate fallback remain. Input uploads, device
requirements and public APIs are unchanged. See
[first-point average fusion evidence](../notes/2026-10-15_fused_voxel_first_average.md).

`reduce_voxel_centroids_xyz_and_gather_first_multi_gpu` uses another independent
lazy kernel with that same planner. One pass preserves original centroid
sum-times-reciprocal math and copies one-to-four first-point float attributes.
Empty XYZ stays positive zero; empty scalar attributes stay zero, while
multi-gather attributes retain input point-zero bits. On supported devices,
float passes fall from two to one and four-attribute copies from seven to one.
The old three/four-channel dummy outputs are removed on this route. Active
payloads, mapped output ownership and the original fallback remain; pooled
uploads and post-readback recycling keep their existing boundary. See
[centroid/first-attribute fusion evidence](../notes/2026-10-15_fused_voxel_centroid_first.md).

`gather_voxel_first_xyz_and_multi_gpu` keeps its original four-attribute,
no-byte fused branch and extends fusion to the remaining one-to-four-float
cases accepted by the shared planner. An independent lazy integer-word kernel
copies first-point XYZ/attributes without floating arithmetic. Scalar versus
multi-channel empty-cell rules, pooled uploads and byte commands remain.
Newly eligible calls use one float pass instead of two; four attributes mixed
with bytes need one float copy instead of seven. The old fast branch and split
fallback, active payloads, mapped decoding and recycling are preserved. See
[broader first-point gather evidence](../notes/2026-10-15_fused_voxel_first_gather.md).

The four combined voxel XYZ/attribute calls batch two-to-four u8 channels
into one packed planar output when seven storage/eight total bindings,
256 lanes, 4KiB workgroup scratch, dispatch and complete output limits fit.
Independent lazy first/average kernels share segment/index loads while
preserving sequential f32 byte sums, division/rounding and zero byte tails.
Byte passes/copies fall from channel count to one; host input padding,
payloads and owned decoding remain. Zero/one/five-or-more byte channels and
unsupported devices/sizes keep their original per-channel loops. No wait,
submission or hidden transfer is added; float fusion and pool boundaries
remain. See [batched byte evidence](../notes/2026-10-15_batched_voxel_u8_attributes.md).

Standalone multi-channel first-point voxel gathers coalesce at most two
existing two/four-channel batches per submit/map/wait when combined staging
fits device and host-slice limits. This submission policy preserves the
original pass count; output packing below reduces copies and inactive slots.
Unsupported calls retain the original loop. The live per-window
input/output/staging set can double,
bounded independently of total channel count. Input/output payloads and
owned capacities remain; no shader, cache or public API change. See
[coalesced gather evidence](../notes/2026-10-15_coalesced_voxel_gather_submit.md).

Standalone multi-channel voxel float averages share point-index traversal
in groups of up to four and write one planar f32 output per group. A lazy
seven-storage/eight-binding kernel preserves sequential per-channel sums,
scalar division and positive-zero empty cells. Passes/output copies fall
from C to ceil(C/4); a one-channel tail keeps the original scalar recorder.
The existing single submit/map/wait, active input/output/staging payloads,
ownership and unpooled uploads remain. Checked group output, whole staging,
host-slice and device limits retain the complete original fallback. See
[batched average evidence](../notes/2026-10-15_batched_voxel_float_average.md).

Both combined voxel average APIs also reuse the four-channel attribute
kernel in their separate-XYZ branches. Original one-to-four-attribute XYZ
fusion keeps priority; more-than-four and supported smaller-binding/size
fallbacks use ceil(C/4) attribute passes/copies instead of C. Seven float
attributes need three total float passes instead of eight. An append-only
recorder validates destination offset/extent/alignment and whole target
device/host limits before byte-identical uploads, with original scalar
tails and full fallback. It records into existing staging without another
submit/map/wait; XYZ, bytes, ownership and payloads remain. See
[combined attribute average evidence](../notes/2026-10-16_combined_voxel_attribute_averages.md).

Combined first-attribute voxel gathering packs each original two/four-channel
batch into one planar output and one staging copy. Keeping the original
runtime width preserves point-zero multi-channel empty rules versus positive-
zero scalar tails. Existing XYZ fusion keeps priority, uploads stay pooled,
and unsupported plans retain the original helper. A new lazy seven-storage/
eight-binding integer gather removes dummy outputs without adding submit,
readback, wait or payload. Seven attributes with width four keep three float
passes while reducing ten float copies to five and removing eight dummy
outputs. Checked destination/device/host limits precede packed resources. See
[packed first-attribute evidence](../notes/2026-10-16_packed_combined_voxel_gather.md).

Standalone first-point voxel gathering also reuses that existing integer
packed kernel, including every one-channel tail's point-zero empty-cell
rule. One planar output/copy per original two/four-wide batch replaces
separate active/dummy outputs, keeping bounded two-batch submission windows,
unpooled input uploads, active payloads, owned capacities and map/wait counts.
Seven channels at width four keep two passes/one submit while copies fall
from seven to two and one inactive output plane disappears. Checked packed
binding/index and original device/host limits retain the entire old fallback.
No shader/cache variant or requested feature is added. See
[standalone packed gather evidence](../notes/2026-10-16_packed_standalone_voxel_gather.md).

Resident-input standalone voxel XYZ gather and centroid calls use one
planar output/copy instead of three when the full binding/device/host limits
fit. First-point gathering borrows axes into the existing integer packed
kernel; centroids use an independent lazy kernel with the exact original
sequential sums, reciprocal multiplication and zero-empty rule. Original
eight-storage/nine-binding capability checks and complete old fallbacks
remain. One pass/submit/map/wait, 12*N useful output/download bytes and exact
owned XYZ capacities stay unchanged; output/staging data-buffer objects fall
from four to two, with the same summed 24*N logical bytes. See
[resident packed XYZ evidence](../notes/2026-10-16_packed_resident_voxel_xyz.md).

All four combined XYZ/attribute voxel APIs also reuse the existing packed
XYZ kernels in their separate-XYZ branches. Original fused plans retain
priority; otherwise one XYZ output/copy replaces three within the existing
encoder/staging, with no extra submit/map/wait. The append-only helper checks
aligned offset/extent and whole destination/device/host limits before its
output allocation, retaining the complete old XYZ block on rejection.
XYZ math, attributes/bytes, upload recycling, useful payloads and owned
capacities stay unchanged. Seven float attributes on a four-wide backend
keep three float passes while copies fall from five to three. No shader or
cache variant is added. See
[combined packed XYZ evidence](../notes/2026-10-16_packed_combined_voxel_xyz.md).

Combined first-XYZ/average-attributes and centroid/average-attributes voxel
calls with more than four float attributes reuse existing fused kernels for
XYZ plus the first four attributes, followed by original four-wide batches
and scalar tails. Checked seven-plane binding, full target/host and dispatch
guards precede helper uploads/recording; rejection retains the full previous
split path. Float passes/output copies each fall from three to two for
seven attributes, while math, upload payload, staging bytes, byte paths,
one submit/map/wait and exact owned capacities remain. No shader/cache is
added. See [wide voxel average evidence](../notes/2026-10-16_wide_voxel_average_fusion.md).

Combined first-XYZ/first-attributes and centroid/first-attributes voxel calls
with more than four float attributes reuse the existing fused XYZ/first-four
kernels, then the unchanged packed attribute gather recorder. Checked seven-
plane binding, original batch width and full destination/host/dispatch guards
precede helper uploads/recording; rejection keeps the complete previous path.
Original scalar empty-cell tails, raw bits and pooled uploads/recycling remain.
Seven float attributes at width four use two instead of three float passes/
copies; useful payload, bytes, one submit/map/wait and ownership stay unchanged.
See [wide first-attribute evidence](../notes/2026-10-16_wide_voxel_first_fusion.md).

Combined GPU voxel readback copies at least two float attribute planes with
at least 1,048,576 cells in bounded native CPU bands, capped at four workers
and available CPUs. Small/single-attribute/WASM paths retain serial copying;
small calls skip CPU-count queries. Scoped workers join while the original
mapped view/unmap guard remain alive. Original raw bits, owned capacities,
XYZ/byte decoding, input/error order, GPU work/transfers and one submit/map/
wait are unchanged. No point-sized scratch is added; worker stacks/metadata
add bounded overhead. See [parallel attribute readback evidence](../notes/2026-10-16_parallel_voxel_attribute_readback.md).

Standalone GPU voxel first-value and average channel readback reuses the
same bounded CPU plane copier per mapped window. At least 1,048,576 cells
and two channels enable at most four native workers; small/single-channel/
WASM paths stay serial. The original full mapped extent check, guard and
view enclose scoped workers. GPU math, packed/coalesced/scalar fallback
dispatch, two-batch gather windows, average batching, transfers and submit/
map/wait counts remain. Exact raw bits and N-element capacities persist.
Thread metadata/stacks add bounded overhead; no latency reduction is claimed
without isolated measurements. See [standalone parallel readback evidence](../notes/2026-10-16_parallel_standalone_voxel_readback.md).

Combined voxel byte-attribute uploads borrow the caller's slice when its
length is divisible by four, avoiding the temporary host padding Vec/copy.
Other lengths retain the exact original owned zero-padding path. All five
scalar/batched contents sites preserve DeviceExt snapshot semantics, GPU
payloads/labels, kernels, transfers, submits/maps/waits and output ownership.
Host pointer alignment is not a restriction on a byte slice. GPU/backend
uploads still copy the original padded bytes. See
[borrowed byte upload evidence](../notes/2026-10-16_borrowed_voxel_u8_uploads.md).

Standalone voxel XYZ readback uses the shared checked plane policy at
1,048,576 cells, capped at three native copy workers including the caller.
One child copies X/Y for two CPUs; two children copy X and Y for three CPUs,
while the caller copies Z. Small/single-CPU/WASM paths retain serial copies
without small/WASM CPU-count queries. All scoped work joins under the
original mapped view/guard. Exact XYZ bits/capacities and GPU layouts, math,
transfers, submits/maps/waits remain. No outer result Vec or point-sized
scratch is added; thread metadata/stacks add bounded overhead. See
[parallel XYZ readback evidence](../notes/2026-10-16_parallel_voxel_xyz_readback.md).

All four combined voxel APIs reuse the same bounded XYZ copier after
finishing their original float attribute and packed-byte copies. The phases
remain sequential, so attribute and XYZ worker groups never overlap within
a call. XYZ uses at most three native workers including the caller at
1,048,576 cells; small/single-CPU/WASM paths stay serial. All source bounds
are still checked before copying and the mapped view/guard enclose every
scope. GPU bodies, math/layouts/transfers, submit/map/wait count and exact
owned results remain; worker metadata/stacks add bounded overhead. See
[combined parallel XYZ evidence](../notes/2026-10-16_parallel_combined_voxel_xyz_readback.md).

Combined voxel byte readback uses scoped channel bands for at least
4,194,304 cells and two u8 channels, matching the existing four-MiB useful
bytes per plane. The original checked plane planner bounds packed source
bytes, available CPUs and at most four children. Absent/single/small/one-CPU/
WASM paths stay serial. Float, byte and XYZ phases remain sequential under
the original mapped view/guard. Endian-aware bytes, packed tail exclusion,
exact owned capacities and all GPU math/layouts/transfers remain. Only
worker metadata/stacks add bounded overhead; no latency or peak-memory gain
is assumed. See [parallel byte readback evidence](../notes/2026-10-16_parallel_voxel_u8_readback.md).

Unaligned voxel byte uploads reserve the complete padded host Vec capacity
before copying source bytes and appending zeros. This removes Vec growth
requests and excess tested payload capacity; aligned inputs remain borrowed.
Payload bytes, GPU math/layouts/transfers, public APIs and readback worker
policies remain. Summed capacity reductions across channel temporaries are
not peak-memory/RSS savings or latency evidence. See
[reserved byte padding evidence](../notes/2026-10-16_reserved_voxel_u8_padding.md).

The host XYZ centroid API shares indices/starts uploads across three
unchanged scalar reductions, copying each axis from one private scratch
output before reuse. One staging buffer and submit/map/wait replace three
complete scalar calls. Checked scalar capabilities, source sizes, combined
buffer/host and u32 limits select this batch; the original fallback and
error/empty behavior remain. Payload H2D saves two repeated metadata uploads,
D2H/math stay exact, and the existing owned XYZ reader/copy policy is reused.
All axes remain live until submission and staging is larger; fewer requests
do not imply peak-memory or latency savings. See
[host XYZ batching evidence](../notes/2026-10-16_coalesced_host_voxel_xyz.md).

Voxel attribute average recorders retain one private output sized for the
largest four-channel batch, copying each active result to staging before
reuse. Standalone average and both combined XYZ/average APIs share this
policy, while the wide fused prefix keeps its existing separate output.
Empty recorders remain no-ops. Existing planners, math, transfers, dispatch/
copy counts, fallback and owned readback policy remain. Fewer GPU buffer
requests and summed descriptor bytes are source accounting, without a
physical allocation, peak-memory or latency claim. See
[voxel average output reuse evidence](../notes/2026-10-16_reused_voxel_average_output.md).

Packed voxel first-attribute recorders reuse one private output within each
recording operation, preserving every active batch with its original staging
copy before overwrite. Standalone submission windows retain at most two
original batches; combined wide paths keep the existing fused prefix. Batch
widths, scalar-versus-packed empty-cell behavior, upload recycling, transfers,
math, fallback and owned CPU readers remain. Source descriptor reductions
carry no physical allocation, peak-memory or latency claim. See
[voxel first output reuse evidence](../notes/2026-10-16_reused_voxel_first_output.md).

Combined voxel APIs group more than four byte attributes into four-channel
GPU batches with the existing gather/average shaders, one private packed
output and original scalar single-channel tails. Checked capabilities,
packed output indexing and complete staging/host/offset bounds retain the
original fallback on unsupported calls. Small-byte routes, XYZ/float work,
input payloads, packed readback and caller waits remain; fewer byte passes
and allocation requests carry no physical-memory or latency claim. See
[wide byte batch evidence](../notes/2026-10-16_wide_voxel_u8_batches.md).

| Area | Decision |
| --- | --- |
| Core model | `SpatialTensor` + `SpatialIndex` + `SpatialAlgorithm` + `SpatialRuntime` |
| Storage | Hybrid Schema-SoA + chunked AoSoA views |
| GPU | wgpu/WebGPU-first + CUDA specialized backend |
| Robotics | ROS2 first-class, zero-copy oriented |
| AI | DLPack / ONNX / embedding-native point cloud |
| Repository | Mono repo / Cargo workspace |

## Initial workspace (Epic 0)

- `spatialrust` — meta crate
- `spatialrust-core` — schema, metadata, traits
- `spatialrust-math` — Vec/Mat/Pose math
- `spatialrust-io` — readers/writers (feature-gated formats)
- `spatialrust-gpu` — device buffers and GPU runtime
- `spatialrust-image` — typed CPU image buffers and strided zero-copy views
- `spatialrust-image-io` — bounded, feature-gated image codecs and source metadata
- `spatialrust-tensor` — runtime-independent tensor metadata, CPU ownership, and DLPack boundary
- `spatialrust-camera` — camera models, distortion, and RGB-D/point-cloud bridge
- `spatialrust-vision` — feature-gated CPU preprocessing, Feature2D, geometry/multiview, warps, detection, masks, and dense maps

## MVP scope

1. Load PCD/PLY/LAS/LAZ
2. Voxel downsample
3. Normal estimation
4. RANSAC plane segmentation
5. Euclidean clustering
6. ICP registration
7. Save output
8. Feature-gated wgpu execution with explicit host/device receipts

Post-MVP additions:

- Unified file IO via `read_point_cloud_file` / `write_point_cloud_file`
- Explicit `spatialrust_io::StorageRoots` for external input/output storage;
  `io-manifest` adds checksummed size receipts without adding storage policy to
  `spatialrust-core`
- `MvpPipelineConfig::*_policy` for feature-gated GPU MVP stages
- `MvpPipelineResult::receipt` for per-stage backend and transfer accounting

## Dependency direction

```
math -> core -> search/geometry/io/gpu -> algorithms -> integration
```

Forbidden: `core -> io`, `core -> gpu impl`, `core -> ros2`, `core -> ai`.

Image and camera dependency direction:

```
math -> image
math -> image -> image-io
math -> image -> vision
math -> image -> tensor -> ai
math + image + core -> camera -> vision::spatial/rgbd/odometry
```

`spatialrust-image` remains independent of `spatialrust-core`. GPU image storage
must use a dedicated backend and explicit upload/readback APIs; Epic 89 provides
`GpuImage` and image compute kernels in `spatialrust-gpu` behind `gpu-image`.
The v2 representation is a pooled `rgba8uint` texture with a retained logical
channel count. Kernels return device-resident images and append named receipt
stages; only `upload_u8` and `readback_u8` cross the host/device boundary.
Runtime adapter identity and synchronization are explicit (`adapter_info`,
`wait_idle`), and `recycle` returns textures to the runtime pool.
Image uploads validate positive u32 dimensions against the runtime's 2D texture
limit, checked RGBA row/storage bytes and host Vec capacity before texture or
packing allocation. Readback validates runtime ownership and logical channels,
then checks row alignment, padded staging bytes against `max_buffer_size`, and
host staging/output capacities before buffer allocation or copy submission.
Transfer planning uses no compute or storage-binding limit: uploads write a
texture and readback stages through an unmapped COPY_DST/MAP_READ buffer.
Receipts retain four physical bytes per pixel rather than staging-row padding;
rejections leave them unchanged. Internal image construction also checks u64
physical storage arithmetic. Plans do not impose a new retention/peak-memory
budget or change successful transfers, metadata, kernels or public signatures.
`ImageView::as_packed_slice` borrows exactly the logical scalar elements of
contiguous views, including empty and single-row views, without exposing
trailing storage or ROI neighbors. `GpuImage::upload_u8` borrows those bytes
for four-channel inputs through the existing queue write, removing one
4*width*height-byte framework host vector and its copy. Single-row ROIs qualify
regardless of stride. Backend staging and the explicit four-byte-per-pixel
upload remain; no device copy, pass, wait or readback is added. Strided multi-row
RGBA views retain complete-row host copies. See
[packed RGBA upload evidence](../notes/2026-10-15_borrow_packed_rgba_upload.md).
Large native 1..=3
channel images use scoped disjoint row bands for RGBA packing and logical-channel
unpacking from at least 1,048,576 pixels. Workers are capped by available CPUs,
262,144-pixel work chunks and image height; single-row/small images and WASM
retain caller execution. Extra scheduling metadata scales with workers and the
budget is per call, without a process-wide thread or stack limit. Unused upload
components stay zero, ROI/host padding is excluded, and readback staging-row
padding is skipped. Explicit device transfers and their receipts are unchanged.
Image readback unpacks directly from the mapped padded staging buffer into the
owned logical image. This removes one padded-row-times-height host vector and
its whole-buffer copy for every channel count. The existing scoped unpacking
workers finish while the mapping is live; the view drops before an unmap guard
on success or decoder unwinding. The private mapping helper's owned-byte wrapper
still returns a copy for native histogram and other owned-byte consumers;
this wrapper is compiled only on native targets. GPU staging,
row padding, final output allocation, mapping wait, shaders, metadata and
transfer receipts are preserved; the mapping now remains live during unpacking.
See [direct mapped readback evidence](../notes/2026-10-15_direct_mapped_image_readback.md).
RGB-to-gray retains the original 256-lane linear shader when its group count
fits one dispatch axis. Larger inputs balance those flat groups across bounded
dispatch rows. Padding groups return before pixel-index multiplication to prevent
u32 wrap; source dimensions, pixel count, compute/binding limits and one free
profiler stage are validated before work allocation or pipeline initialization.
Linear and multi-row pipelines are independently lazy within one runtime-owned
family and share the existing 16-byte uniform/layout. Integer BT.601 output,
gray metadata, one pass, four physical write bytes per pixel and 16 query bytes
on explicit collection remain unchanged. There is no new scratch or payload copy.
GPU AI CHW packing validates positive dimensions, logical channels, signed texture
coordinates, u32 pixel/element indices, storage-binding/output-buffer sizes,
16x16 dispatch limits and the 64-byte uniform before allocation or cache setup.
Profile ownership and one free stage are checked before work begins. Finite
scale/mean and finite nonzero standard deviations retain their existing contract;
the original f32 normalization and CHW shader, single pass, storage retention
and explicit payload/query collection remain unchanged. Experimental channel-lane
and fixed-channel kernels remain native test fixtures after completed-call measurements.
`GpuAiTensor::readback_f32` decodes the mapped CHW bytes directly into its owned
float vector through the same guarded mapping helper. This removes one
4*channels*width*height-byte host intermediate and copy; staging and the final
float allocation remain. Native-endian `f32::from_ne_bytes`, plane order,
device checks, copy/map/wait and transfer receipts are preserved, with receipt
recording still preceding float conversion. The mapping remains live through
conversion and releases before return or decoder unwind. Shader normalization,
packing and pool ownership are unchanged. See
[direct AI readback evidence](../notes/2026-10-15_direct_mapped_ai_readback.md).
Gray GPU box blur retains the integer full-box mean with one final half-up
rounding. Boxes with area at least 81 and at least 65,536 total pixel/sample
visits use horizontal unrounded u32 sums and a vertical reduction in one
submission; smaller workloads retain one-pass traversal.
For separable kernel widths at most 257, exact horizontal sums fit u16;
one invocation packs two adjacent pixels into each u32 with a zero odd tail.
The exclusive intermediate request becomes `4*ceil(pixels/2)` bytes, while
wider kernels keep four bytes per pixel. Original full-size capacity guards,
vertical arithmetic, final rounding and submitted scratch leases remain.
Work receipts count the actual horizontal request and unchanged RGBA output.
See [packed box-blur evidence](../notes/2026-10-16_packed_box_blur_scratch.md). Oversized separable kernels scan only valid coordinates and
weight replicated edge pixels analytically; constant-zero padding retains the
full kernel divisor. Pixel, coordinate, sum and device limits reject before
allocation. Both paths use bounded grid strides and independently lazy runtime
pipelines. GPU rectangular morphology retains odd kernels in `1..=31` and
replicate borders. With box area at least 81 and at least 1,048,576 pixel/sample
visits it reduces horizontal then vertical minima/maxima. Clipping valid
coordinates is exact because repeated edge values do not alter an extremum.
Both passes share one submission. Separable widths greater than one pack
four horizontal extrema into each RGBA8Uint texel, using an intermediate of
ceil(width/4) by height; row-tail bytes stay zero. Vertical loads extract
the matching byte before unchanged extrema. Widths divisible by four use
75% less logical scratch. One-column images keep the original intermediate.
Scratch returns to the bounded image pool with its actual extent only after
submission; horizontal receipts count its actual bytes. Two independent lazy
packed variants retain the old shaders, layouts, pass counts and transfers.
See [packed morphology evidence](../notes/2026-10-16_packed_morphology_scratch.md).
Small workloads keep the original full-box pass. Coordinate, texture, dispatch,
uniform and workgroup limits reject before output/scratch/cache allocation.
The separable pair is lazy within the existing spatial pipeline family.
The resident Vision chain records six passes for small workloads and one extra
for each separable blur/morphology path (up to eight), reserving all timestamp
capacity before recording.

`gaussian_blur_gpu` / `gaussian_blur_gpu_profiled` add an explicit resident
Gaussian filter under `gpu-image`. `GpuGaussianBlurOptions` admits 3/5/7 kernels,
finite positive sigmas, and replicate/reflect/reflect101/wrap/constant borders.
All 1..=4 logical channels retain their ordering and metadata. The host prepares
two small normalized integer kernels in binary64, matching specialized CPU u8
quantization: Q8 for 3/5 calls, Q15 for both axes when either axis uses 7.
Two GPU passes compute unrounded horizontal sums and the final weighted result;
Q15 sums accumulate two bounded base-65536 digits in u32, combine the carry
once and round half-up only at the final byte.
Q8 multi-channel calls store two exact 16-bit sums per u32, requesting
`4*ceil(channels/2)` private scratch bytes per pixel; single-channel/Q15 calls
keep `4*channels`. Each invocation owns its pixel words, including zero odd
high halves. Old planner limits remain before selecting the smaller scratch;
old/packed pipeline pairs initialize independently on demand. Work receipts
count actual horizontal bytes and the unchanged output texture. See
[packed Q8 scratch evidence](../notes/2026-10-16_packed_gaussian_q8_scratch.md).
Scratch stays leased through one submission. No image payload readback,
intermediate upload or host wait occurs.
The 128-byte uniform holds control/kernel coefficients and is excluded from
payload receipts; device writes count scratch plus the RGBA8 output texture.
Runtime/dimension/coordinate/index/buffer/binding/workgroup and optional two-stage
profiler limits are checked before resource allocation or lazy cache setup.
Oversized scratch requests reject; the caller controls placement. Returned images
compose with gray/Sobel/AI stages through the existing resident image API. See
[Gaussian execution evidence](../notes/2026-10-13_gpu_gaussian_blur.md).

`median_blur_gpu` under `gpu-image` applies exact per-component 3x3/5x5
median selection to resident `rgba8uint` images with 1..=4 logical channels.
Replicate and constant-zero borders match CPU median sampling; alpha is filtered
as an independent component and metadata is retained. The 3x3 path uses 19
compare-exchange pairs: sort three triples, then select the median of the
largest minimum, middle median and smallest maximum. A threshold argument and
all rank permutations validate exact order selection, including duplicates.
The 5x5 path keeps its generated, pruned bitonic network with maximum-byte
padding; its generated shader remains identical. Static integer min/max comparisons avoid data-dependent
sorting loops and dynamic private-array indexing. One device pass returns an
owned image with four physical output bytes per pixel; no image-sized scratch,
payload upload/readback or host completion wait is hidden. Dimension, signed
coordinate, dispatch, texture/binding and 16-byte uniform limits are checked
before output allocation or lazy cache creation. The two kernel pipelines are
independently lazy within one runtime-owned family. Caller-selected CPU median
placement and generic CPU kernel/border support remain unchanged. See
[median execution evidence](../notes/2026-10-14_gpu_median_filter.md). See also
[3x3 selection evidence](../notes/2026-10-15_gpu_median_nine_selection.md).

The CPU `median_blur` retains generic component conversions and `f64::total_cmp`
ordering. Windows larger than 3x3 use middle-element selection; tiny windows
retain the library's optimized full sort. Native calls with at least 262,144
logical components and multiple rows use contiguous output bands, bounded by
the current Rayon pool and height.
Each band reuses one kernel-sized f64 sample vector per channel; bands write
disjoint parts of one final output vector. Small, single-row, one-worker and
WASM calls run on the caller. Border modes, odd apertures, strided views, metadata
and explicit CPU/GPU placement are preserved. GPU measurements against this
improved CPU baseline are recorded in
[CPU median execution evidence](../notes/2026-10-14_parallel_cpu_median.md).

`GpuBilateralFilter::prepare` under `gpu-image` explicitly uploads immutable,
image-independent coefficients for 1..=4 u8 channels, diameters 1..=7 and
replicate/constant-zero borders. A circular radius `floor(diameter/2)` and the
squared joint L1 color distance, including alpha, match CPU bilateral sampling.
Host binary64 FMA/exp prepares shared distance rows; normal-or-zero weights and
finite factors are admitted. The runtime-bound plan is reusable across image
sizes; its coefficient transfer receipt is separate from image payload receipts.
`filter` performs color-distance calculation, ordered integer-word binary64
weighted sums and exact byte rounding in one device pass. Comparing against
half-byte division-rounding boundaries avoids native shader float64 and a full
software quotient. There is no image-sized scratch, payload transfer or host
completion wait. Ownership, channel and resource/coordinate/dispatch limits
reject before output/cache allocation. CPU bilateral calls use bounded native
row bands for large images and retain the original per-pixel math; small,
single-row, one-worker and WASM calls keep the original serial loop. See
[bilateral execution evidence](../notes/2026-10-14_gpu_bilateral_filter.md).

Bilateral final byte rounding now proposes a candidate using the normalized
high mantissa words and a bounded normal f32 scale. Exact binary64 division
midpoint comparisons verify the lower/upper byte boundaries before any result
returns; a rejected candidate narrows a bounded integer search. Float estimates
never decide output correctness. Zero and extreme exponent gaps use exact
guards. Accumulation, coefficients, preparation, passes, scratch, limits and
explicit transfer receipts retain their existing contracts. See
[byte-candidate evidence](../notes/2026-10-14_gpu_bilateral_byte_candidates.md).

`canny_gpu` / `canny_gpu_profiled` extend the resident grayscale image API with
complete device Canny. `GpuCannyOptions` accepts 3/5/7 Sobel apertures, finite
nonnegative binary64 thresholds and L1/squared-L2 comparison. CPU-compatible
replicate gradients and directional asymmetric NMS ties use integer arithmetic;
aperture 7 rounds scaled signed derivatives to even and scales thresholds by
1/16. A direct aperture-3 pass or two separable 5/7 passes precede classification.
Four forward neighbor edges enumerate the entire undirected eight-neighbor graph
of weak pixels. Atomic CAS hooks only a current higher root to a lower one;
path halving preserves this acyclic forest. After all unions, weak roots are
compressed and flagged when any member touches a strong pixel. Strong pixels
are retained directly, and only flagged weak components survive. This is
equivalent to full hysteresis while avoiding strong-only union work. There is no
weak-edge propagation cutoff or host convergence
readback. Five/six ordered passes share one submission and return an owned
binary `GpuImage`; blur/grayscale conversion remain explicit caller stages.
Scaled signed derivatives are bounded by 1020/12240/10200 for 3/5/7 apertures;
they occupy two exact signed 16-bit halves of a u32. Magnitude remains u32,
so each private gradient occupies eight bytes. Unrounded horizontal derivative/
smoothing pairs are bounded by 765/4080 (5) and 2550/16320 (7); they also use
two exact signed 16-bit halves, occupying four bytes per pixel. Vertical
accumulation reconstructs them before the unchanged final scale/rounding.
The horizontal pass temporarily stores those packed pairs in the later atomic
state/flag binding. Vertical gradients finish before classification overwrites
the rows, so disjoint lifetimes share storage without a copy or additional pass.
Classification and root-neighbor flags share one atomic u32 per pixel: low
two bits classify, bit 2 marks a weak root touching a strong pixel. Classification
overwrites all bits before connectivity; later readers mask the low bits while
markers atomically OR only bit 2. Full-width parent indices and the complete
union/root rules remain. Private scratch requires `16*N` bytes for every aperture:
gradients `8*N`, shared rows/states/flags `4*N`, and parents `4*N`. The separate
horizontal allocation and aperture-3 dummy are removed. The largest binding is
`8*N`; three storage buffers and a 96-byte control uniform
are required. Runtime/dimension/signed-coordinate/index/buffer/binding/workgroup
and profiler checks precede cache/output/scratch allocation. Receipts retain
exact image upload/readback bytes and count per-stage destination storage
footprints (32*N or 36*N), rather than repeated in-place atomic traffic.
Near-constant or sparse edge geometry affects union cost; device capacity or
CPU/GPU placement is not selected automatically. See
[Canny execution evidence](../notes/2026-10-13_gpu_canny.md) and
[compact-gradient evidence](../notes/2026-10-13_gpu_canny_compact_gradients.md) and
[horizontal-packing evidence](../notes/2026-10-13_gpu_canny_horizontal_packing.md) and
[shared-state evidence](../notes/2026-10-13_gpu_canny_shared_state_flags.md) and
[row/state lifetime evidence](../notes/2026-10-13_gpu_canny_reuse_horizontal_state.md).

GPU nearest resize and Sobel validate their runtime identity, positive source/
output dimensions, signed coordinates, texture/compute/binding limits and one
free profiler stage before output allocation or spatial-pipeline initialization.
Resize checks that each maximum output-coordinate/source-dimension product fits
u32 before division. Valid work retains the original shader and one output pass,
with four physical write bytes per pixel and 16 query bytes on explicit collection.
No global scratch, intermediate texture, pipeline or payload transfer is added;
Vision chains keep their six-to-eight-pass contract.
Optional `GpuStageProfiler` instrumentation is shared by image and point-cloud
chains in `spatialrust-gpu`. Timestamp-enabled runtime construction requests
only adapter-supported pass queries; default runtimes enable none. Profiled
entry points reserve bounded capacity before execution. Query collection is an
explicit native synchronization/readback boundary with a separate byte receipt.
GPU pass times exclude CPU setup, transfers and inter-pass gaps; completed wall
time remains a separate measurement. Failed recordings cannot be collected.
Standalone host-returning k-NN normals, grid radius normals and plane covariance
estimators share bounded execution. Each kind lazily initializes its own pipeline
on its owning runtime; shader math and neighborhood order stay unchanged.
Storage inputs/output use exclusive submitted scratch leases, with full input
uploads on every call and no content cache. Compute and result staging copy share
one submission before scratch is returned. Mapping remains the explicit
synchronous boundary of these host-returning APIs. Uniform and staging buffers
are still created per call; the existing scratch pool bounds retained storage.
Every active covariance invocation writes both vec4 output rows. With fewer
than three neighbors it writes the existing epsilon diagonal and zero cross
terms/padding explicitly, so recycled storage cannot influence the fallback.
The existing point/output bounds cover both row addresses; no global clear,
extra dispatch, allocation or transfer is needed.
Point, index, buffer, uniform and dispatch limits reject before device allocation.
Grid cell storage is checked before CPU counting-sort allocation. Resident AoSoA
frame execution and its timing/transfer receipts remain separate.
Shared dense-grid bounds validate equal XYZ lengths, finite coordinates and a
positive cell size with finite reciprocal before dimension planning. Finite
cell sizes retain the original f32 subtraction/multiply/floor mapping; nonfinite
spans reject instead of saturating into integer dimensions. Dimension increments
and cumulative cell products use checked u64 arithmetic, enforce the existing
cell cap before checked u32 narrowing, and never return zero dimensions.
Positive-infinite radius retains a one-cell grid for finite inputs, including
widths whose f32 subtraction would overflow. Valid empty bounds retain the
existing infinite origin and unit dimensions. `uniform_grid_fits` returns false
for invalid inputs. Radius normal/covariance calls reject before counting-sort,
scratch acquisition or pipeline initialization; existing empty-call shortcuts,
device/sentinel checks, transfers and accepted shader math remain unchanged.
Dense-grid radius normals and plane covariances merge the valid neighboring
x cells into one counting-sort interval per y/z row, for at most nine row
intervals per gather instead of 27 cell intervals. Both mean and covariance
gathers retain the original z/y/x/point order, distance test and floating
operations. Clamped x endpoints are reused across both gathers. The existing
positive i32 dimensions and bounded cell/sentinel storage also bound these
endpoints; no buffer, workgroup memory, barrier or transfer is added.
The sparse AoSoA radius kernel has a dedicated shader matching the previous
generated source. Its signed-key binary searches and per-cell traversal remain
independent of dense CSR row gathering.
Standalone Euclidean clustering retains uploaded XYZ and compacted sparse cells
for GPU component labeling. Each sorted point recovers its owning cell from
ascending starts, traverses signed adjacent-cell keys with overflow guards, and
tests original-order scalar squared distances. Each undirected edge is visited
once. Weak compare/exchange hooks only a current higher root to a lower root;
atomic path shortening preserves strictly decreasing parents. Initialization,
connection and final compression are separate device passes in one submission,
without host union/find or convergence polling. The component minimum index is
preserved independently of invocation order. Only roots and the existing scalar
cell count return to the host: `4 * points + 4` bytes for finite tolerance,
replacing cell-key/start/index metadata downloads. XYZ uploads remain `12 * points`.
Positive-infinite tolerance initializes zero roots on device, avoiding XYZ/grid
work. Invalid coordinates, reciprocal or buffer/compute sizes reject before
allocation. Host validation/origin reduction and high-level cluster-size
filtering/label-cloud assembly remain explicit. Placement and public signatures
are unchanged; receipts now name `gpu-component-labeling`. Parent scratch is
fully initialized and leased through submission; cached pipelines belong to the
runtime. Dense neighborhoods retain quadratic worst-case pair work.
Voxel boundary compaction uses a 256-lane shared-memory scan with recursively
scanned block totals and offset propagation. All levels stay device-resident
in one submission; the existing four-byte cell-count synchronization remains
explicit. Temporary hierarchy storage shrinks by a factor of 256 per level.
Radix, boundary flags, prefix levels and atomic count scratch use a private
runtime-owned buffer pool. A scratch batch owns every lease until its encoder,
including the scalar count copy, is submitted; abandoned batches are not
recycled. Submitted buffers can be reused through queue ordering without a new
wait or device copy. Caller-owned segment outputs stay outside this pool.
Large voxel radix sorts reuse offset scratch for per-workgroup earlier-digit
subtotals written by the histogram pass. The prefix pass reads only its own
digit slice, scans chunk totals with a work-efficient tree and overwrites offsets
after all lanes finish reading. This removes repeated global earlier-digit loads
without another scratch allocation or pass. Stable ranks, 24 digits/72 passes, bounded
dispatch, queue ordering and the explicit compaction count readback are retained.
Up to 2,048 input workgroups (524,288 points), legacy histogram/prefix kernels
remain specialized without the extra subtotal barrier. Larger inputs lazily cache
two optimized pipelines using a compile-time override; scatter is shared. This
internal GPU selection retains small-input work and adds no CPU/GPU placement.
The cached histogram's existing per-digit count loop also writes transposed
exclusive prefixes for eight 32-bit mask words. After the existing subtotal
barrier, each point reads its prefix and popcounts only preceding bits in its
own word. Local ranks stay stable and exact, including partial input groups.
All 128 prefix words are overwritten by sixteen digit lanes. This adds 512
bytes of histogram workgroup storage, keeping its 1,088-byte footprint below
the already-required 2,052-byte prefix kernel. No barrier, global allocation or
pass is added; legacy specialization, pipeline selection and scatter are retained.
RANSAC plane scoring retains its explicit XYZ/hypothesis uploads and score
readback. Below 1,024 points the scalar per-hypothesis loop is retained; larger
inputs assign a 256-lane workgroup to a hypothesis and tree-reduce integer
inlier counts. Plane construction and each point's f32 distance expression
retain their operation order. At most 256 groups stride over hypotheses, with
subtraction guards preventing u32 wrap and a barrier before model reuse.
Serial/parallel pipelines are independently lazy and cached per runtime. Input
indices and buffer/compute limits are checked before allocation. The score copy
and compute pass share one submission; no global partial-count scratch is added.
Upload/CHW buffers, point scratch and RGBA8 textures each retain at most 64 MiB
by default, independently, and at most eight resources per size. Full pools
reject incoming resources; a budget decrease below retention clears the cache,
and zero disables retention. Budgets cover recycled payload bytes, not active
outputs, uniforms, mapped staging, allocator/driver overhead or peak process
memory. `GpuResourcePoolStats` counts requested allocations, cache hits and
retained/discarded resources; clearing preserves these lifetime counters.
Image/tensor ownership uses a retained runtime identity token rather than a
runtime address, so moving a runtime preserves valid resources.
GPU channel histograms under `gpu-image` use exact integer RGBA8 loads and two
compute passes in one submission. At most 256 workgroups build private 256-bin
partials (at most 256 KiB), then independently reduce each bin into a live
1 KiB `GpuImageHistogram`. Every partial and output bin is overwritten, including
zero counts in reused storage. Scratch follows the submitted-lease protocol;
live histogram outputs stay outside the scratch pool until explicit recycling.
Channels are logical source components, with no implicit luma conversion.
Inputs above u32::MAX pixels and insufficient compute/buffer limits are rejected
before allocation. Counts and their sum therefore remain exactly representable.
Optional profiling records exactly two passes; native `readback_u32` explicitly
transfers 1,024 bytes and timestamp collection separately transfers 32 bytes.
Histogram construction has no payload readback or implicit host synchronization.
The public APIs are provisional; CPU references/Rayon are dev-only dependencies.
At 4,194,304..=8,388,608 pixels and at least 8,224 bytes of workgroup storage,
histogram partials use eight independent local bin replicas. One padding word
per replica separates equal-bin addresses; each lane accumulates into its
`lane & 7` replica and sums the eight counters after a workgroup barrier. All
used local bins are explicitly initialized. Integer mass stays bounded by the
original u32 pixel count. The extra partial pipeline is lazy per runtime;
inputs outside that range or lower shared-memory limits retain the original pipeline.
Group count, global partial/output storage, final reduction, two-pass profiling
and transfer receipts are unchanged. Native per-lane run compression remains
an experimental fixture after distribution-dependent completion measurements.
Resident histogram consumers include `histogram_cdf_gpu`, a one-workgroup
inclusive tree scan of the 256 bins. It uses 1 KiB local memory, a separate 1 KiB
output, and no global temporary buffer or uniform. Integer counts remain exact
through u32::MAX; the last cumulative count equals source pixels. `GpuImageCdf`
inherits source transfer history, adds one device-write stage, and requires
explicit native readback/recycling. The histogram may be recycled immediately
after the CDF call submits its reads; queue order protects later pool reuse.
CDF recording adds one profiler stage (16 query bytes upon explicit collection).
Grayscale `equalize_histogram_gpu` consumes the histogram/CDF inside a five-pass
resident operation: two histogram passes, one CDF pass, a 256-entry lookup pass
and bounded grid-stride image application. Lookup scratch is 1 KiB, leased until
both lookup/application passes are submitted; internal histogram/CDF outputs are
recycled after their dependent reads. The final texture remains caller-owned.
A u32-word arithmetic model reproduces the CPU's binary64 reciprocal/multiply
rounding at exact half-byte boundaries, without requiring float64 shaders or
changing CPU output. Non-half rational values are rounded exactly; constant
images retain their byte and all images retain metadata. Construction performs
no payload readback or host wait. Five profiled passes require five free stages;
explicit timestamp collection returns 80 query bytes separately from payloads.
Other channel counts and unsupported counters/device limits fail before work
allocation. Empty GPU images remain unsupported. Normalized public CDFs and
adaptive tile equalization remain separate future contracts.
Resident grayscale `clahe_gpu` runs three or four passes in one submission:
axis geometry, histogram/lookup, then interpolation/application. Lookup storage
is 1 KiB per tile; axis geometry is 16 bytes per source row/column, both private
scratch leased through application submission. Reflect101 padding and exact
clipped-count redistribution match the CPU. Per-tile cumulative counts use a
workgroup-wide tree scan of bounded counts and clipped excess. The residual
redistribution prefix is computed analytically from the CPU's bin positions,
avoiding a second scan and serial 256-bin clipping/CDF loops. Histogram bins and
scan storage use about 3 KiB of workgroup memory within the existing 4 KiB limit;
the minimum workgroup storage requirement remains 4 KiB. Direct histograms use
three passes. Tiles with at most 256 padded pixels assign one pixel per lane.
Larger direct tiles partition columns/rows with power-of-two strides, avoiding per-pixel flat-index
division. Short tiles use more columns to keep lanes occupied; narrow tiles use
more rows. Reflect101 computes a remainder only outside the source bounds, and
subtraction guards keep stride increments within the validated u32 dimensions.
For at most 16 tiles with at least 65,536 padded pixels each, a separate
histogram pass partitions each tile into `ceil(area/8192)` contiguous chunks,
bounded to 8–32 chunks. Up to 512 independent workgroups write 256 integer bins
per chunk; fewer available groups stride through all chunks. The lookup pass
merges these exact counts before the same clipping/scan/rounding. This adds at
most 512 KiB of private scratch and one pass; insufficient buffer/binding limits
select the original direct path. Both split pipelines initialize lazily in the
owning runtime. Partial buffers stay exclusively leased through application
submission and follow the existing byte-bounded scratch retention controls.
Receipts include `clahe_histogram_gpu` and the partial histogram write bytes.
Shared integer-word lookup rounding and a finite non-negative binary64 arithmetic
subset preserve the CPU's division, subtraction, multiplication/addition
operation order without shader float64.
Geometry ratios with positive power-of-two divisors scale the exponent of an
exact u32 value, returning identical binary64 words without fractional long
division. Other ratios retain the original division and rounding algorithm.
All axis entries are still regenerated on device per call using the same bounded
scratch; no geometry-retention cache or additional transfer is introduced.
Equal integer corners bypass interpolation only when the final byte is invariant
under CPU rounding. No payload readback, geometry upload or host wait is hidden.
The clip limit is a scalar host parameter; per-tile counts, padded coordinates,
buffer bindings and compute limits are checked before allocations. Caller-owned
output preserves metadata; constant bytes may change as in CPU CLAHE. Direct
work requires three free profile stages and collects 48 query bytes; split work
requires four stages and collects 64 bytes. Capacity is checked before work
allocation. Browser shader execution is still unverified; adaptive contrast APIs remain provisional.
CPU image analysis under `imgproc-analysis` uses optional Rayon row bands for
large thresholds, u8 histograms and histogram-equalization lookup application.
Tasks are bounded by image rows and the current pool; histograms merge exact
integer counts from 256-bin local arrays. CLAHE independently groups lookup
tiles and output rows into at most the current pool's worker count at one
million pixels. Tiles retain their original clipping and reflect-101 sampling;
parallel interpolation shares one horizontal geometry table proportional to
image width, without changing f64 operation order or ties-to-even rounding.
Strided/ROI views exclude storage padding. Small inputs and one-worker pools
keep scalar interpolation; a single output row cannot dispatch row bands, but
large CLAHE lookup grids can still dispatch independent tiles. CLAHE grid size
and lookup byte size are checked before allocation. The existing lookup matrix
and output buffer remain image/grid-sized; per-active-tile scratch is 256 usize
bins, and horizontal geometry is shared rather than duplicated per worker.
CPU point queries under `spatialrust-search/parallel` group contiguous original
`SpatialTensor` chunks into scoped workers, capped per call by available CPU
parallelism and chunk count. Worker bands differ by at most one chunk; callbacks
keep their original ranges and result order. Fewer than 4,096 points, one chunk
or one available worker run on the caller. Dispatch range/handle metadata scales
with workers instead of chunks; result buffers still scale with neighbor count.
This uses standard-library threads without an additional runtime dependency or
implicit device transfer; concurrent callers each have their own worker budget.
Shared CPU `build_grid` construction retains O(points + cells) counting-sort
work, using cumulative cell ends and descending source indices for stable
scattering. Each end is decremented to its final CSR start; the last sentinel
remains the point count. This removes the separate cell cursor clone and
`4 * (cells + 1)` bytes of live logical heap payload. Returned indices/starts,
per-point f32 coordinate mapping and increasing source order inside each cell
are identical. Native `parallel` builds scan cell-offset tables with at least
8,388,608 entries and at least 1,024 entries per point using scoped CPU workers,
capped by available parallelism and useful 262,144-entry chunks. Worker-local
inclusive scans join before ordered base computation; a second scope adds bases
in disjoint bands, skipping
zero bases. Metadata scales with workers, without another cell/point-sized
table. Small/denser tables, one-worker calls and WASM keep caller execution.
Mapping and stable reverse scattering remain serial. `gpu-wgpu` enables this existing
search feature for shared host preparation without another dependency. Cargo
feature unification also enables the existing native dense-root dispatcher;
WASM dense roots keep the serial path. Budgets are per call rather than a
process-wide CPU/stack limit. Standalone GPU radius normal/covariance preparation
retains its storage uploads, shaders, scratch pools and readback.
Dense-grid `euclidean_cluster_roots` merges the neighboring x cells into one
stable CSR interval for each valid y/z row. Clipped row bounds avoid walking
out-of-grid offsets, including single-row/axis grids. The z/y/x cell order,
within-cell source order, self exclusion and f32 radius predicate are preserved.
Both caller-thread and existing atomic union-find workers use the same iterator;
the 4,096-point dispatch threshold and scheduling remain unchanged. Iterators
hold constant stack state, without additional heap scratch, device transfers or
dependencies. Grid construction and sparse segment labeling retain their prior
execution. This applies to the public dense-grid search API; GPU sparse-grid
clustering and the segmentation extractor's KD-tree CPU path use other walkers.
`search-graph` together with `parallel` adds explicit `knn_graph_parallel` and
`radius_graph_parallel` APIs. They build one shared KD-tree, reuse that bounded
chunk dispatcher with 256-point chunks, and produce worker-local edge lists in
source-point order. Ordered joins preserve the serial neighbor/edge sequence,
including KD-tree ties and duplicate-coordinate selection. Single-worker and
below-threshold graph calls keep one caller range. No all-query neighbor table
or distance matrix is staged; per-query neighbors and live local/final edge
vectors still consume result-dependent memory (radius graphs may be quadratic).
Existing serial APIs keep serial execution. Graph point counts must fit u32;
large k requests are capped by cloud size with checked reservation arithmetic.
MLS smoothing under `spatialrust-filtering/filter-mls` shares one caller-built
KD-tree and writes independent contiguous output bands with scoped CPU workers.
It reuses the existing point-staging policy: below 4,096 points, one available
worker, or fewer than two useful 16,384-point chunks, execution stays serial.
Worker count is capped by available CPU parallelism and useful chunk count.
Each point retains unsorted KD-tree radius-neighbor order, leave-one-out
selection and the original weighted covariance/polynomial arithmetic. The final position
vector is allocated once; fitting scratch remains local to each active point.
No all-query neighborhood table, new dependency or implicit device transfer is
introduced. Filter reconstruction preserves field names, schema, all non-position
buffers and metadata. Concurrent calls have independent per-call worker budgets.
FPFH descriptors under `spatialrust-registration/register-fpfh` reuse the same
point-staging policy for independent SPFH rows and weighted FPFH rows. Each
phase fills disjoint slices directly in its final descriptor array, sharing one
immutable KD-tree. The SPFH scope joins completely before weighting reads its
rows. Per-point neighbor order, Darboux/bin/normalization arithmetic and output
order are preserved; descriptor matching and seeded RANSAC retain their existing
execution. Two 132-byte-per-point descriptor arrays are retained during weighting,
with one radius-neighbor vector per active worker and no all-query neighborhood
table. Worker caps are per call, not a process-wide CPU or scratch-byte budget.
Boundary detection under `spatialrust-features/feature-boundary` uses the same
bounded staging policy, sharing one immutable KD-tree while filling disjoint
contiguous slices directly in the final boundary mask. Each worker reuses one
angle vector across its points and holds only the current radius-query neighbors.
Neighbor order, tangent-plane math, angle sorting and threshold comparisons stay
unchanged; no floating-point reduction spans workers. The detector gathers
selected points in original index order and preserves all field types and
metadata. Small/one-worker calls remain serial. Neighbor/angle scratch scales
with radius support per active worker; the worker cap is not a scratch-byte or
process-wide CPU budget. No device copy or new dependency is introduced.
ISS keypoint detection under `spatialrust-features/feature-iss` uses the same
bounded staging policy in two scoped phases sharing one KD-tree. Workers fill
disjoint slices of the existing four-byte-per-point saliency array; the first
scope joins before local-maximum selection can read any row. Parallel selection
writes a temporary one-byte-per-point mask and scans it in original index order,
dropping it before schema/attribute gathering. Serial calls retain direct index
collection and allocate no selection mask. Unsorted neighborhood order, f64
centroid/covariance arithmetic, eigenvalue ratios and the existing >= tie rule
are preserved. Per-query neighbor vectors scale with radius support; worker
caps remain per call rather than a scratch-byte or process-wide CPU budget.
No floating-point reduction spans workers and no device transfer is introduced.
Cloud distances under `spatialrust-metrics/metrics-distance` build one KD-tree
per cloud and independently bound each direction's query workers with the same
point-staging policy. Parallel queries fill disjoint slices of a reusable window
of at most `workers * DEFAULT_SPATIAL_TENSOR_CHUNK_SIZE` optional f32 distances.
Each window joins before its values are converted to f64, added and square-rooted
in original point order; sums are never regrouped across workers or windows.
Missing neighbors remain distinct from returned NaNs. One-worker directions use
the unchanged serial loop and allocate no score window. The two directions run
sequentially, so their windows are not live together. Scratch scales with worker
budget rather than total query count; KD-tree storage remains input-sized, and
worker caps are per call rather than a process-wide CPU limit. Each active band
reuses one one-neighbor result vector with the same KD-tree traversal and fresh
per-query accumulator, avoiding a heap allocation for every parallel query.
KD-tree construction carries the selected axis's already-computed minimum into
median selection. A minimum or NaN pivot cannot have a strictly smaller element;
selection reproduces the original pivot swap directly instead of scanning the
range. Moving right past a finite pivot raises the active range's lower bound;
NaN pivots leave it unchanged. General partitions, axis choice, leaf size and
recursive node construction keep their original behavior. Point permutations
and split words remain identical, preserving query ties and unsorted neighbor
order for downstream floating-point algorithms. This removes repeated scans of
minimum buckets while retaining serial execution and existing storage. General
quickselect worst-case complexity and parallel tree construction are unchanged.
Unbounded KD-tree single-neighbor queries hold one optional neighbor on the
stack, without heap scratch. Traversal and distance arithmetic match the k=1
accumulator: a candidate is rejected with `>=`, including the original NaN
replacement behavior; far branches use strict distance pruning once a neighbor
exists. Empty trees return no neighbor. Query validation, nonfinite results,
ties and bounded/k-NN APIs retain their existing semantics. This path introduces
no device copy, allocation, dependency or worker scheduling.
`spatialrust-image-io` depends on storage, never the reverse; standard codecs
are additive, while TIFF and OpenEXR remain independently gated.
Calibration datasets, robust solver controls, and residual receipts live in
`spatialrust-camera`; they depend only on `spatialrust-math` small dense solves.
External nonlinear optimizers may be added only behind dedicated features.
`spatialrust-vision` keeps preprocessing, Feature2D, geometry/multiview (H/F/E,
PnP, sparse LK, stereo BM), warp, detection, dense-map, and spatial bridges in
separate additive features. Geometry depends on `spatialrust-camera` only and does
not pull Feature2D or dense-map types. CPU APIs never perform implicit device
copies; future GPU/CUDA implementations belong behind explicit backend features.
Video algorithms depend on dense/detection contracts, while timestamped pull
sources are isolated behind `video-adapters`; native codec/camera runtimes stay
in future dedicated adapter crates/features.
Visual and RGB-D odometry kernels remain in the additive `odometry` vision
feature. Their conversion into stamped trajectory motion is a one-way optional
bridge in `spatialrust-mapping`; monocular scale and invalid depth remain
explicit at that boundary.
Computational photography composes image/warp/geometry primitives behind the
additive `photography` feature. Panorama APIs expose canvas bounds and enforce a
pre-allocation pixel budget; codec and device execution remain separate.
The optional runtime execution graph owns operator scheduling only. Placement,
watermarks, and named transfers reuse `spatialrust-distribute`; fusion is
limited to linear fusable nodes on one device and cannot erase transfer edges.
Its `imgproc-*` features share one border extrapolation contract; `filter2d`
means correlation, while true convolution is an explicitly named operation.
`spatialrust-tensor` is distinct from the point-cloud chunk iterator named
`spatialrust-core::SpatialTensor`; it owns generic dtype/shape/stride/device
contracts and never performs implicit host/device transfers.
`spatialrust-ai` depends on `spatialrust-tensor`, never the reverse. Its default
build defines only backend/session and explicit-copy contracts, plus a
deterministic `MockInferenceBackend` for demos and tests. ONNX Runtime and
each hardware execution provider are additive features. Runtime-owned CPU
outputs cross back through the runtime-independent `HostTensorStorage` trait,
so their allocator lifetime can be retained without copying or adding an ONNX
dependency to the tensor crate.
`spatialrust-vision` `ai-adapters` bridge CPU images to contiguous host tensors
and decode depth/mask/detection tensors back into dense vision types without
depending on `spatialrust-ai`.

ROS 2 bag storage follows the same boundary. `spatialrust-runtime` owns only
runtime-independent PointCloud2 CDR and type-negotiation contracts;
`spatialrust-ros2` owns the optional read-only rosbag2 SQLite source behind
`rosbag2-sqlite`. Native `rclrs` executors, custom message bindings, and
compressed bag storage remain separate adapter features. The source emits
bounded XYZ/XYZI `SpatialRecordChunk` leases with frame/timestamp metadata and
protocol-independent `spatialrust-records::RecordProvenance`; its bounded
TF inventory preserves source frame edges without composing or applying them.
It never places ROS 2 or SQLite types in `spatialrust-core`.

Versioned record envelopes remain in `spatialrust-records`. Their concrete
point fields are described by `SchemaDescriptor`, capture time and coordinate
frame remain in `SpatialMetadata`, and source lineage is carried separately by
`RecordProvenance`. Schema migration and bounded record transforms must retain
source lineage; aggregations clear a single source sequence when it no longer
identifies one input record.

## Roadmap epics

| Year | Focus |
| --- | --- |
| 1 | Foundation / MVP |
| 2 | v1.0 stable geometry runtime |
| 3 | Robotics adoption (ROS2/Autoware/Nav2) |
| 4 | AI integration |
| 5 | Spatial computing platform |

The canonical post-foundation horizon is Epics 91–100 in `docs/ROADMAP.md`.
Epic 91 adds `spatialrust-records` and `spatialrust-arrow` while keeping Arrow
out of `spatialrust-core`. Epic 92 adds `spatialrust-sync` for clocked stamps,
frame graphs, and deterministic multimodal episode replay. Epic 93 adds
`spatialrust-mapping` for trajectories, relative motion traits, and pose graphs.
Epics 94–100 extend into scene reconstruction (`spatialrust-scene`), semantic
spatial data (`spatialrust-semantic`), embodied-AI episodes
(`spatialrust-episode`), robotics runtime contracts (`spatialrust-runtime`),
glTF/OpenUSD interchange (`spatialrust-interchange`), explicit distributed
execution (`spatialrust-distribute`), and platform stability
(`spatialrust-platform`). Heavy native bindings stay optional.

See the full master architecture document in project planning materials for trait-level design, ADRs, and Codex execution tasks (Epics 0–13).

## SpatialRust 1.2 bounded-streaming boundary

The canonical 1.2 program is Epics 121–126 in `docs/ROADMAP.md`. It adds
bounded point-cloud execution without expanding `spatialrust-core`:

```text
core PointCloud/SpatialTensor
        -> records stream contracts
        -> feature-gated io adapters
        -> chunk-safe algorithms/pipeline
        -> CLI/Python
```

`SpatialTensor` continues to describe borrowed chunks of one materialized
`PointCloud`; it is not an out-of-core source. Streaming memory limits account
for explicitly owned buffers and fail before exceeding their hard ceiling.
Format-specific caches and unavoidable temporary spool storage are named in
versioned receipts. GPU upload/readback remains caller-selected and measurable.
Leased record chunks keep their memory reservation alive for exactly the record
borrow lifetime. The additive `PointCloud::into_parts` inverse and mutable
column iteration are the only core ownership primitives required for
allocation-free source-owned buffer recycling; stream traits, threads, queues,
receipts, and policies remain outside core.

The 1.2 release boundary is machine-checked by
`Streaming12ReleaseGate`. Its dedicated Linux/Windows/macOS matrix exercises
records, all bounded formats, deterministic operations, the real CLI path, and
the canonical receipt example. The gate rejects missing or skipped evidence,
tracked-memory or spool overruns, retained reservations after finish,
unrecorded host copies, any CPU-workflow device transfer, deterministic
mismatches, and excessive run/file-handle fan-out. Concrete IO/pipeline/Python
adapters remain provisional; the bounded record traits, chunk lease, memory
budget, cancellation, options, and versioned receipt are the stable 1.2
foundation.

## Visual architecture boundary

The Visual program is Epics 133–140 in `docs/ROADMAP.md`. Visualization remains
an additive consumer of spatial data and never expands `spatialrust-core`:

```text
math + optional core capabilities -> viz contracts
viz contracts + gpu-wgpu          -> render-wgpu
render-wgpu + io/scene/mapping    -> native viewer
render-wgpu + bounded records     -> explicit streaming LOD
render-wgpu                        -> Web/WASM and Python/Jupyter adapters
```

`spatialrust-viz` owns backend-independent borrowed geometry, camera, style,
layer, residency, and transfer-receipt contracts. Borrowed host views do not
interleave or copy structure-of-arrays point data. GPU allocation, upload,
render passes, picking, and readback belong to `spatialrust-render-wgpu`; every
host/device crossing is caller-requested and recorded. Windowing and UI
dependencies stay in the dedicated `spatialrust-viewer` application crate.
That crate's default build contains only portable viewer state, input reduction,
attribute inspection, and owned debug-overlay adapters. Its `native` feature
adds winit window/event handling; it does not upload geometry. Applications
retain control of explicit renderer upload and presentation integration.
Scene, mapping, camera, and semantic adapters are separate additive viewer
features. Mesh storage is borrowed directly; adapters that must transpose AoS
data or generate lines return a byte-exact `AdapterReceipt`. Synchronized RGB-D
inspection borrows image/cloud payloads and validates dimensions, ordering, and
sensor skew before exposing a timeline frame.

Native windowing, Web/WASM bindings, and Python/Jupyter adapters are independently
feature-gated. Large-cloud display consumes the bounded records and persisted
index contracts from Epics 127–132, with explicit point, memory, upload, and
in-flight chunk budgets. Camera motion may cancel obsolete requests but must not
leak leases or GPU allocations. Headless rendering and deterministic image
comparison provide the portable correctness boundary; interactive frame-rate
numbers remain dated, adapter-specific receipts.

`spatialrust-lod` is the renderer-independent enforcement layer. It accepts a
validated hierarchy rather than opening data itself, uses separate enter/exit
screen-error thresholds, and selects resident ancestors while finer children
are unavailable. Optional records integration owns drop-scoped decoded-memory
leases; GPU admission records caller-declared upload/allocation bytes and
evicts only unprotected LRU nodes. The COPC adapter produces a bounded query but
never performs range IO implicitly.

`spatialrust-web` owns only the browser/WASM boundary. Its portable JSON
envelope embeds the same `ViewerState`, and WebGPU rendering delegates to
`spatialrust-render-wgpu` after caller-requested upload. Native process-global
wgpu sharing is unavailable on wasm32; browser callers construct a main-thread
runtime asynchronously. Remote byte access requires a bounded range plan before
fetch, an abort signal, `206` plus exact response-length evidence, and explicit
admission into the bounded WASM cache.

The standalone `spatialrust-py` wheel wraps that same Web viewer envelope
instead of defining another state model. NumPy viewer geometry either retains
three contiguous SoA owners without copying or enters owned Rust storage through
an explicitly named, byte-receipted copy. Native launch creates only the winit
state/input shell; renderer uploads remain separate. The independently packaged
`spatialrust-jupyter` AnyWidget validates state with the Rust binding and sends
it to `spatialrust-web` through a versioned iframe protocol with exact
source/origin checks. Neither adapter chooses a device, range source, upload, or
readback.

The aggregate boundary is machine-checked by `VisualReleaseGate`. Stable
backend-independent contracts and provisional adapters are registered
explicitly. The gate requires strict Linux/Windows/macOS headless image
evidence, native/browser/Python/Jupyter smoke evidence, transfer and LOD
receipts, documentation and unsafe audit, and acknowledgement of the
`visual-1` migration policy. It fails closed on missing, skipped, duplicate,
future-dated, older-than-30-day, over-budget, or experimentally unstable
evidence. The canonical decision and measurements are committed in
`docs/VISUAL_RELEASE_RECEIPT.md`.
