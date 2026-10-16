# Packed GPU morphology intermediate pixels

## Behavior and selection

The existing separable grayscale `morphology_gpu` path now packs four adjacent
horizontal extrema into one RGBA8Uint texel. The intermediate texture width
becomes ceil(width/4), with the original height. Each horizontal invocation
owns all four bytes of a texel; incomplete row tails write zero. The vertical
pass loads texel x/4 and selects byte x%4 before its original min/max reduction.
No atomic, barrier, cross-invocation update or floating arithmetic is needed.
The final physical output remains [value,0,0,0] per logical grayscale pixel.

Original odd kernels 1..=31, replicate borders, area threshold 81 and
pixel/sample threshold 1,048,576 remain. Small workloads keep the original
direct shader. Separable one-column images keep the original two shaders:
row packing would not reduce their storage. All original coordinate, texture,
uniform, workgroup and full dispatch limits still reject before resources.
Horizontal packed dispatch uses ceil(ceil(width/4)/16) groups in x;
vertical dispatch keeps ceil(width/16). Both keep ceil(height/16) in y.
The new dispatch is bounded by the original full-width checks.

The original 32-byte uniform and three-binding texture layout remain. There
is no storage-buffer binding, device feature/limit request, dependency,
unsafe code or public API addition. Two packed pipelines initialize together
on first packed use in an independent OnceLock. Original direct and separable
pipelines remain, and a packed-only call does not initialize the old pair.
Resize, Sobel, all other images and point-cloud kernels stay unchanged.

## Storage, accounting and ownership

For separable width W>1 and height H, logical scratch requests change from
4*W*H bytes to 4*ceil(W/4)*H bytes. Widths divisible by four use 75% less
scratch; widths two/three use 50%/two-thirds less, and odd rows retain up to
three zero tail bytes. Width one keeps its original request. At 1920x1080,
the request changes from 8,294,400 to 2,073,600 bytes; at 3840x2160 it changes
from 33,177,600 to 8,294,400 bytes. These are logical texture/request bytes;
backend row alignment, driver allocations and peak/RSS are unmeasured.

Both passes still share one submission. Scratch returns to the bounded
image pool only after submission, using its actual ceil(W/4)-by-H extent.
An abandoned encoder drops its texture instead of publishing it for reuse.
Queue ordering protects submitted scratch and input reuse while owned outputs
remain live. Different original widths sharing a packed width can reuse
the same scratch key. Every texel component is overwritten on every call.

The horizontal work receipt records the actual 4*ceil(W/4)*H bytes; the
output receipt remains 4*W*H. Stage names and two optional timestamps remain.
Vision-chain pass capacity stays unchanged. Metadata, explicit host upload
and download payloads and returned ownership remain. No extra pass, submit,
host wait, readback or hidden transfer is added. Lower scratch requests and
fewer horizontal texture stores alone do not establish lower latency.

## Correctness and source audit

The complete parent planner, spatial cache with original shaders, encoder
and public compute call are frozen from
acaaa71650374480f6cd1a843616e5d27751b067. Only test visibility/type names and
an explicitly prewarmed original cache parameter change. Source comparison
verifies the complete original guards/thresholds, shared resize/Sobel APIs
and resource/submit/pool/owned-result boundaries. The old separable shader
stays byte-for-byte unchanged. New per-pixel horizontal bounds/extrema are
the original horizontal specialization; the new vertical kernel is the
original vertical specialization with only packed texture extraction.

Pure tests compare 2,171 complete frozen planner outcomes,
uniform bytes, original pass thresholds and exact per-row requests. They
cover exact/one-short texture, coordinate, dispatch, workgroup, uniform and
binding limits, zero storage-buffer support and large coordinate/error cases.
Direct horizontal probes compare 19,203,084 original GPU,
independent CPU and packed bytes. Widths 1/2/3/4/5/7/8/9/17/33/257/1025,
heights through 1025, kernels 1/3/9/31 and both operations cover every row-tail
remainder and partial workgroups. Patterned, zero, hot and checker inputs
pass. Poisoned textures verify zero unused row bytes and untouched two suffix
columns plus one suffix row in both original/packed layouts.

Complete public calls compare 118,835,784 frozen GPU/independent
CPU/physical bytes. A separable CPU oracle is checked against the existing
brute rectangular oracle on small cases. Both operations, all eight selected
kernel shapes, four patterns, thin/odd/single-column and threshold-neighbor
fixtures plus million-pixel direct/separable controls pass. Every physical
output padding byte stays zero; metadata and actual work receipts match.

Cold concurrent direct/packed/original-column calls verify pipeline pointer
identity, timestamps/stage names and outputs surviving source/pool recycling.
Queued changed-input calls at widths 33/34/35/36 share nine-texel scratch
rows. Allocation counters and actual texture extents verify correct pool
keys; poisoned reuse, invalid calls leaving the cache cold and abandoned
packed encoders also pass. Existing strided-ROI, budget and Vision-chain
regressions remain, updating only documented receipt/cache expectations.

All 131 native image, 95 voxel
and 2 public facade regressions pass (228 distinct
tests), including five new morphology tests. All-feature/all-target Clippy
with warnings denied, CPU-only, minimal gpu-wgpu and CI
gpu-image,gpu-aoso-staging WASM builds with warnings denied, formatting and
diff checks pass. Audit verifies 420 unchanged GPU/image
sources and 427 source/executable hashes fixed after validation
and measurements. Tracked/new-file and PR-body privacy match counts are zero.

## Measurement method

The ignored release benchmark compares frozen/current completed resident
calls (public morphology plus wait) and full calls (upload, morphology and
owned download). Resource/command creation, pool/cache access, submission,
device completion and each route's explicit public transfers stay inside
timers. The frozen comparator uses an explicitly prewarmed original cache;
current runtime cache lookup stays inside the call. Fixture/oracle setup,
assertions, output recycling/destruction and resident result readback remain
outside timers. Three warmups precede thirty cyclically alternating samples
of all four variant/route combinations. Every output and receipt is checked.
Cold pipeline/device construction is excluded.

Eight (width,height,kernel_width,kernel_height,operation) conditions are
(16,17,3,3,Erode), (257,259,9,9,Dilate), (1920,1080,9,9,Erode),
(1920,1080,31,31,Dilate), (1,4097,31,31,Erode),
(257,259,1,31,Dilate), (3840,2160,9,9,Dilate), (3840,2160,31,31,Erode).
Small/direct and single-column cases control unchanged paths.

Two surveys each allow a one-minute quiet scheduling window with three
five-second quiet checks. Process censuses before/during/after reject
overlapping trials; only owned benchmark children may be cancelled.
Source/executable hashes are captured after validation and checked afterward.
Accepted paired log old/new ratios use 10000 bootstrap resamples with seed
2026101606 + 100*survey + 10*condition_index + JSON_field_index; sorted
indices 249/9749 define 95% intervals. Ratios above one favor packing,
intervals crossing one are inconclusive and zero samples suppress ratios.
All outcomes are retained. Local receipts use morphology-packed- under
C:/Users/rsasa/Workspace/SpatialRust/target/automation/.
Published evidence omits the local adapter model name.

Neither bounded one-minute quiet window completed an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed.

A separate non-isolated functional invocation passed all eight complete-call conditions and both routes, checking every output and actual work receipt. All its timing samples are excluded.

