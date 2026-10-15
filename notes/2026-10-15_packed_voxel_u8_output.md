# Packed voxel u8 output

The four combined XYZ/float/u8 gather/average APIs previously stored each byte
result as one u32 on the GPU and downloaded all four bytes. Their byte output,
copy and staging regions now use four cells per word, with independent padded
planes for each byte attribute. Public signatures and final owned host channel
vectors remain the same.

All 256 lanes still calculate their original cell in parallel. A new packed
WGSL variant writes these results into a 256-word workgroup array, followed by
one uniform barrier. The first 64 lanes each assemble and store one word from
four neighboring cell results. Every lane initializes scratch, including
out-of-range tail lanes; tail bytes become zero. One writer owns each output
word. There are no atomics, output clears or additional compute passes.

Gather's first-index selection and byte extraction are unchanged. Reduction
retains its original sequential f32 accumulation, division, WGSL round and
low-byte mask per cell. The original word-output shaders are untouched. Their
public cached pipelines and binding layouts remain compatible; two private
packed pipelines initialize with OnceLock on first use and share those same
layouts. Cold pipeline creation is additional one-time work. Packed workgroups
use 1 KiB of shared memory; requested device features/limits remain unchanged.

For N cells, F = 3 + float attribute count and U byte channels:

| Payload | Previous word output | Packed output |
| --- | --- | --- |
| Byte output/copy stride per channel | 4*N | 4*ceil(N/4) |
| Byte output buffers, summed over channels | 4*N*U | 4*ceil(N/4)*U |
| Combined readback staging/download | 4*F*N + 4*N*U | 4*F*N + 4*ceil(N/4)*U |
| Returned float capacity | 4*F*N | 4*F*N |
| Returned byte capacity | N*U | N*U |

For 1,048,576 cells with three byte channels, byte download falls from 12 MiB
to 3 MiB: 9 MiB less, or 75% of the byte payload. Byte output buffers and the
byte part of staging each also shrink by 9 MiB, giving 18 MiB less requested
device buffer payload across those allocations. With four float attributes,
the combined float/byte download changes from 40 MiB to 31 MiB. Padding means
small nonmultiples of four save less; a one-cell channel still takes four
bytes. Workgroup scratch, allocator/driver overhead and other buffers are
additional. These are payload counts, not peak-memory or latency measurements.

The mapped decoder copies float planes directly into their final vectors.
On little-endian hosts it also copies each mapped byte plane directly into its
final vector; big-endian hosts extract each packed u32's byte at the cell's bit
offset. This preserves the existing word cast convention. Both routes ignore
padding between channels and create no intermediate host array. The big-endian
branch is source-audited; the available native and WASM checks are little-endian.
The existing guard releases mapped views before unmapping on return/unwind.
Input padding, resident XYZ/segment inputs, per-call attribute uploads, ordered
validation, empty returns, float kernels, four-float fusion eligibility,
command/pass/submission counts and wait/error behavior stay unchanged. Internal
word recorders/decoder are retained for native regression references.

## Correctness method

Four complete parent public calls are frozen from merge
`a884c9f383938fb5ef4be2d6b150be8380697e78`, with only names/test visibility
changed. Their shared word recorders/decoder, original binding layouts and
both word-output shaders must match that parent. The source audit bounds
all production differences to the packed variants, byte layout changes,
decoder and lazy cache additions. Every other GPU/image source must match.

Direct kernel fixtures compare legacy words with packed output and an
independent integer CPU reference for all non-tie averages. Half ties must
match the original GPU exactly; the CPU also bounds them to the two adjacent
integers. The fixtures cover all 256 byte codes, zero/one/two/three/four/seven/
seventeen points per cell, reversed point indices, three byte patterns and
0/1/2/3/4/5/63/64/65/255/256/257/1023/1024/1025/65,537 cells. Output starts
with nonzero sentinels. Entire packed words, zero tail bytes and eight trailing
guard words must match, including partial workgroups and empty cells.

Mapped fixtures cover 0/4 float and 0/1/3 byte attributes, arbitrary float bits
including NaNs/subnormals/signed zeros, packed channel strides, nonzero unused
padding, repeated maps, live-output mutation and dropped buffers. Float/byte
bounds panics must unmap and permit subsequent valid maps. Public calls compare
all four gather/average modes with both the complete parent and independent CPU
at byte/workgroup boundaries through 1,048,577 cells. Existing combined-call
regressions retain empty/length validation, updated host attributes, pool reuse
and lifetime coverage. Concurrent calls must share stable lazy pipeline handles
and preserve owned outputs after inputs are dropped and pools cleared.

## Measurement method

Measurements time the complete public resident-XYZ/host-attribute call. They
include per-call host attribute uploads, device/staging/uniform creation,
compute/copies, submit/map/wait, final host decoding and owned return. CPU
fixture/expected generation, attribute reference arrays, XYZ/segment uploads,
output assertions and output drop are outside timing. This is warm call
latency, not GPU kernel time or complete key-sort/downsampling latency. Both
variants share unchanged float paths and use direct owned host decoding.

Fixed (cells, float attributes, byte attributes) conditions are (16,0,1),
(17,4,3), (4096,0,3), (65537,4,3), (262144,0,1), (262144,4,3),
(1048576,0,3), (1048576,4,3), each with four scattered points per cell.
All four modes use three warmups and 30 alternating paired samples. Mode 0
gathers XYZ/attributes; mode 1 gathers XYZ and averages attributes; mode 2
averages all; mode 3 averages XYZ and gathers attributes. Every returned value
and capacity is checked outside timing, including all warmups and samples.

Two isolated surveys each have a bounded three-minute quiet scheduling window.
Competing-work checks run before, during and after each attempt; overlap
discards the entire attempt. Other work is left running. Source/executable
SHA-256 hashes are captured after validation and checked after attempts.
Paired log old/new ratios use 10,000 bootstrap samples, sorted indices 249/9749
for 95% intervals, with seed 2026101537 + 100*survey + 10*condition_index +
JSON_field_index. Ratios above one favor packing; intervals crossing one are
inconclusive. Zero samples suppress that row's ratio. Every favorable, adverse
and inconclusive row is retained; ratios are not ratios of reported medians.

Local receipts are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with `voxel-u8-`
prefix. Published environment/measurement evidence excludes adapter model names.

## Completed checks and measurements

All 49 native GPU voxel regressions and 18 voxel-filter regressions passed
(67 distinct local tests). Four new regular regressions cover 417,504
direct kernel cell results, including 59,640 half ties matched exactly to
the legacy GPU and all 256 byte values, 24,964,632 mapped float/byte
elements, and 124,868,464 complete parent/CPU exact elements. Existing
combined-call validation, ownership, updated-input and pool regressions remain
in that selection. Concurrent lazy initialization and caught panic/remap
checks passed. The final run includes the little-endian direct byte copy.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with
warnings denied, formatting and diff checks passed. Whole-source comparison
verified four complete frozen public parent functions, 354 unchanged GPU/image
sources including all 97 original WGSL files, and two new packed shaders with
identical per-cell arithmetic. All 365 source/executable hashes stayed fixed
after validation and measurement attempts. Privacy scan matches were zero.

Neither bounded quiet scheduling window completed an isolated trial because
competing work remained. No timing sample from those attempts is accepted or
published, and no complete-call latency improvement is claimed.


One additional functional invocation completed all eight benchmark conditions
and all four public modes with exact output/capacity assertions for three
warmups and 30 paired calls per mode. It is separate from the 67 regular tests;
its non-isolated timing samples are excluded from performance evidence.
