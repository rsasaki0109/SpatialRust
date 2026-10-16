# Reserve padded voxel u8 upload capacity once

`pad_u8_for_gpu_storage` previously copied N bytes into an N-capacity Vec,
then resized to P=4*ceil(N/4) bytes. Unaligned nonempty inputs needed capacity
growth when appending their one to three zero bytes. Reserve P bytes first,
then extend from the source and append the same zeros without growing the
Vec. No unsafe code, public API, dependency or thread policy is added.

The unchanged `u8_storage_upload_contents` selector continues borrowing
inputs whose length is divisible by four, including empty inputs. Only
unaligned inputs request an owned padded payload. All four combined voxel
XYZ/float/u8 APIs and their byte batch uploader share this preparation.

## Capacity and transfer accounting

For the tested native fixtures, the new owned Vec has length and capacity P.
It copies the original N bytes once and writes P-N zeros; no payload-sized
intermediate is added. An initially sufficient capacity removes the old
Vec growth request. Whether an allocator could grow the old allocation in
place is implementation-dependent, so this is not a claim that every old
call performed a second physical allocation or N-byte copy.

For N=1,048,577, the frozen old Vec capacity is 2,097,154 bytes; the new
capacity is 1,048,580 bytes. Across seven channel preparations, the sum of
those capacities falls from 14,680,078 to 7,340,060 bytes, a reduction of
7,340,018 requested payload-capacity bytes, with seven growth sites removed.
The channel temporaries need not coexist: these are sums across preparation
calls, not peak memory or RSS savings. Backend/allocator overhead is not
measured. The actual upload payload remains 7,340,060 bytes in both routes.

Source bytes, order, zero padding and H2D payload lengths are exact. Buffer
creation still snapshots inputs before source mutation and queue submission.
Fused/wide/scalar/batched GPU selection, shaders, math, limits, dispatch,
storage, copy commands, transfers, submit/map/wait count and recycling remain.
The original borrowed selector and all CPU readback planners/workers remain.
Only initial host Vec capacity changes; returned voxel values and their
ownership/capacities are preserved.

## Exact current parents and validation

The baseline is main commit ef7bc2331a0bcbf565296ad84f5c9e0fcac68bb0.
Tests freeze all four complete public parents, the original Cow selector,
original growing pad helper and complete batch upload helper. Both scalar
and batched old uploads call the frozen preparation; neither can pick up
new padding through a shared delegate. Unchanged batch planning/GPU
recording and current readback remain shared. Source normalization permits
only test names, visibility, module qualifications, whitespace and trailing
commas. Replacing the reserve/extend pair with the original initialization
restores the whole original readback file. The other 459 original GPU/
image Rust/WGSL files are byte-identical. The post-validation manifest pins
465 source/executable hashes before and after measurement.
Published evidence omits the local adapter model name.

Five new tests compare 181,952,768 host payload bytes,
4,458,040 captured GPU upload bytes and 340,826,888
complete four-mode current-parent/CPU elements. Host cases include every
length 0..=1025, 65535/65536/65537, all million-point remainders and all
4,194,304..=4,194,307 remainders, at source offsets zero through three.
Checks cover original payloads, independent source/zero expectations, exact
owned capacities, aligned/empty borrowed identity and the capacity example.

GPU captures cover all remainders and odd host addresses, source mutation/
destruction before submit and unchanged destination guards. Complete APIs
cover absent/single/three/seven byte channels, zero/one/four/seven float
attributes, all modes and point-count remainders, occupancy one/four and
sorted 257-cell segments whose construction reads back only four bytes.
All regular finite fields match frozen parents and CPU references exactly.
Sparse/empty-cell and byte average tie cases retain original GPU results
and independent integer bounds. Float-before-byte and first/middle/last
byte validation order, empty early returns, output capacities and retained
results after input destruction/pool clearing remain. Four concurrent calls
retain old results after source-byte mutation and later uploads.

All 150 native voxel, 131
image, 18 filter and 2
facade tests pass (301 distinct tests). All-feature/all-target Clippy
denies warnings. CPU-only, minimal gpu-wgpu WASM and CI gpu-image/
gpu-aoso-staging WASM with warnings denied, formatting and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/readback.rs
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_reserve_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-u8-reserve-*.

## Complete-call timing

The benchmark includes complete public calls from resident XYZ/segments and
host attributes through validation, host/device allocation and preparation,
uploads, GPU submit/map/wait and owned decoding. Input/reference preparation,
output/capacity assertions and destruction are outside the clock. Three
warmups precede thirty alternating paired current-parent/new calls.
Conditions (cells,float,byte,occupancy,mode) are (16,0,1,1,0), (17,0,1,1,0),
(65536,0,3,1,1), (65537,0,3,1,1), (1048576,7,3,1,2),
(1048577,7,3,1,2), (1048578,0,7,1,3), (1048579,0,7,1,3). The aligned
conditions control the unchanged borrowed path; every unaligned remainder
is covered. Modes 0/1 use first XYZ with first/average attributes; modes 2/3
use centroid XYZ with average/first attributes. Both routes retain the
current parallel readback policies.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Accepted paired log old/new ratios
use 10000 bootstrap resamples, seed 2026101617 + 100*survey +
10*condition_index + JSON_field_index, and sorted indices 249/9749 for 95%
intervals. All outcomes remain; ratios above one favor the new route,
intervals crossing one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs; its non-isolated timing samples are excluded.
