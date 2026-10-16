# Borrow already padded voxel byte upload inputs

The four combined voxel XYZ/float/u8 APIs previously copied every byte
attribute into an owned padding vector before the existing device upload,
even when its length was already a multiple of four. Their scalar upload
sites and shared batched-byte upload controller now use
`u8_storage_upload_contents`, an internal borrowed-or-owned slice selector.

## Payload selection and lifetime

When point count P is divisible by four, the helper returns `Cow::Borrowed`
over exactly the caller's byte slice, including an empty slice. A host
pointer need not be four-byte aligned: the existing DeviceExt upload accepts
bytes and copies the same payload. The zero-padding requirement concerns
the storage byte length. Other lengths retain `Cow::Owned` from the exact
original `pad_u8_for_gpu_storage` function, preserving its allocation/growth
behavior and zero through three trailing zero bytes.

All five production contents sites change only their helper call. The
original BufferInitDescriptor labels, usage, device allocations and payloads
remain. DeviceExt consumes the bytes during buffer initialization; a borrowed
slice is not retained by the command encoder or returned from the public
API. Mutation/destruction of the host source after initialization cannot
change that uploaded snapshot. No unsafe code, dependency, synchronization,
new transfer, worker or public API is added. The original padding helper is
unchanged, including its use by historical test references.

First/average byte kernels, two-to-four-channel batched acceptance, scalar
fallback (including more than four channels), float/XYZ fusion, validation
order, uploaded buffer layouts and output capacities are unchanged. GPU
compute/copy commands, submits, maps/waits, mapped decoder order/lifetime,
ownership, and buffer-pool/recycling policies remain.

## Deterministic preparation work

For B nonempty byte channels of aligned length P, preparation formerly
copied P*B bytes into B temporary payload Vec allocations. The helper now
borrows all those bytes, removing those preparation copies and allocations.
An empty slice never required a nonempty payload allocation, so no saved
allocation is claimed for P=0. For other lengths, both routes still call
the same original owned padding function.

Device upload remains B*4*ceil(P/4) bytes; no host-to-device bytes disappear.
GPU buffers, backend upload copies/staging, float attribute uploads and
returned host output allocations stay unchanged. For example, 1,048,576
points and seven byte channels avoid 7,340,032 preparation copy bytes and
seven temporary payload allocations while uploading the same 7,340,032
bytes. With 1,048,577 points those channels retain original owned padding
and upload 7,340,060 bytes. Peak memory, RSS, total allocator/backend work
and latency are not inferred from these logical payload counts.

## Complete current-parent references and validation

The complete parents are frozen from main commit
45ab30d72c52889862d21ba857961c6e954f9830. All four public functions and
their original batched byte-upload controller are retained. Both the
controller and each scalar fallback use the unchanged original owned
padding helper. The old route therefore cannot borrow through a delegate.
GPU recording helpers and the current mapped decoder remain shared and
unchanged. Only test names, visibility and module qualifications change.

Replacing the five new contents helper call sites with the old helper and
removing the additive selector restores all original production source.
All 444 other original GPU/image Rust/WGSL files
are byte-identical, including every shader/cache and all float plane-copy
logic. The post-validation manifest pins 453 source/executable
hashes. Published evidence omits the local adapter model name.

Five new tests include 64,512,096 exact old/independent host payload
byte comparisons, 4,458,040 GPU upload byte checks and
198,220,200 complete four-mode current-parent/CPU element checks.
The host test covers lengths 0..=1025 plus 65535/65536/65537,
1048575/1048576/1048577 and 4194304, each with source offsets zero to three.
It verifies borrowed discriminants and exact source pointer identity only
for divisible lengths, old owned payloads for every other remainder, exact
bytes and zero tails. Across those probes, 21,760,000 preparation copy
bytes and 1,036 nonempty payload allocations are avoided by
the selector; these are logical case totals, not process allocation samples.

Direct GPU uploads cover all remainders through million-byte inputs from
odd host addresses. Both current and old uploaded buffers are copied into
guarded mapped staging after host source mutation/destruction before submit.
Exact bytes, padding and surrounding sentinel regions remain. Complete
public cases cover empty, one/two/three/four, 17/18/19/20, 257, 65537 and
1048576/1048577 cells; 0/1/4/7 float attributes; 1/3/7 byte attributes;
one/four points per cell; and all four modes. The 257-cell case uses sorted
GPU segments with only four bytes of construction readback. Results and
capacities compare exactly with the full current parent and CPU reference,
then survive input destruction and pool clearing.

Sparse repeated starts and reversed indices retain empty-cell byte zeroes,
first-value selection and rounded means. The independent integer byte oracle
accepts either adjacent integer at exact half ties, as the existing kernel
tests do, while complete current-parent GPU results remain bitwise exact.
First/middle/last byte mismatch and float-before-byte error order remain;
empty segments still return before input validation or pool work. Four
concurrent callers cover every point-count remainder, scalar/batched/many-
channel paths, changed host bytes and retained earlier outputs after drop
and pool clearing. CPU float references use occupancy one/four, so their
sum/division order remains bit-exact for these finite fixtures.

All 130 native voxel, 131
image, 18 voxel-filter and
2 facade tests pass (281 distinct tests).
All-feature/all-target Clippy denies warnings. CPU-only, minimal gpu-wgpu
WASM and CI gpu-image/gpu-aoso-staging WASM with warnings denied, formatting
and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_borrowed_u8_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-borrowed-u8-*.

## Complete-call timing

The benchmark times the full public API from resident XYZ/segments and host
attributes through validation, payload selection, uploads/encoding, GPU
submission/map/wait and owned decoding. Input/reference preparation, output/
capacity assertions and destruction are excluded. Three warmups precede 30
alternating paired current-parent/new calls. Conditions
(cells,float,byte,occupancy,mode) are (16,0,1,1,0), (17,0,1,1,0),
(65536,0,3,4,1), (65537,0,3,1,1), (1048576,7,3,1,2),
(1048577,7,3,1,2), (1048576,0,7,1,3), (1048577,0,7,1,3).
Modes 0/1 use first XYZ with first/average attributes; modes 2/3 use
centroid XYZ with average/first attributes. Alternating aligned/unaligned
cases retain small scalar, accepted batched, wide float and many-byte scalar
fallback paths. Odd point counts control the unchanged owned preparation.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Source/executable hashes are captured
after validation and checked afterward. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101613 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
