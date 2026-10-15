# Fused voxel XYZ and float attribute average

`reduce_voxel_centroids_xyz_and_average_multi_gpu` used one XYZ centroid pass
plus one scalar average pass per float attribute. Each pass reread the same
cell metadata and point-index sequence. A private capability/size planner now
selects a fused pass for one-to-four float attributes, including calls with
packed byte attributes. All other combined/standalone public APIs are unchanged.

One invocation still owns one cell and visits its points in the original order.
It accumulates XYZ and active float attributes together and writes channel-major
planes into one output buffer. XYZ retains `sum * (1.0 / f32(count))`; attributes
retain `sum / f32(count)`. These expressions are deliberately distinct. Empty
cells write positive zero to every active plane. No shared memory, barrier,
atomic or new arithmetic reassociation is introduced.

The pipeline and binding layout initialize lazily with OnceLock. Ten storage
bindings hold indices, starts, seven possible input planes and one output;
one more binding holds the unchanged-size 16-byte control payload. Inactive
read-only attribute bindings alias the first active channel and are never read
by the shader. The active attribute inputs are uploaded exactly as before;
there is no host concatenation or additional point-sized host copy. A bounded
vector of at most four GPU buffer handles retains those inputs while recording.

Before new resources/cache initialization, the planner checks a nonempty cell
count, one-to-four attributes, ten storage/eleven total bindings, 256-lane
workgroup support, dispatch extent and packed output size. Output words must
fit u32 indexing; bytes must fit both the storage-binding and whole-buffer
limits. Checked u64 arithmetic precedes allocation. Failure selects the old
separate float path. This matters when the aggregate packed output exceeds a
binding limit although each old scalar/XYZ output fits. Requested device
features/limits are unchanged. Zero/five-or-more attributes retain the old path.

For N cells, A float attributes (1 through 4) and U byte attributes:

| Execution/payload | Separate float path | Fused float path |
| --- | --- | --- |
| Float compute passes, bind groups and uniforms | 1+A | 1 |
| Float output buffers and copy commands | 3+A | 1 |
| Float output payload/download bytes | 4*(3+A)*N | 4*(3+A)*N |
| Byte compute/copy commands | U | U |
| Byte output/download bytes | 4*ceil(N/4)*U | 4*ceil(N/4)*U |
| Submissions and mapped readbacks | 1 | 1 |

For four float and three byte attributes, total compute passes change from
eight to four and output copies from ten to four. Transfer bytes, summed device
output payload, staging size and final owned host capacity do not shrink in
this slice. Fewer GPU objects do not establish a particular allocation-overhead
or peak-memory saving. Register pressure, conditionals and backend scheduling
can affect latency; pass-count reduction alone is not a speed measurement.

Public signatures, ordered length errors, empty returns, input slices/resident
XYZ/segments, byte padding/packing, float/u8 channel order, final decoding and
map/wait diagnostics remain. The common staging allocation now precedes the
fallback float outputs; validation still precedes all device resources. The
unsupported route's compute/copy body is unchanged. No dependency, unsafe code,
CPU fallback, hidden transfer or additional host wait is added.

## Verification method

The complete public parent call is frozen from merge
`2c73e575a096adbd49001fe16beba28612f75e99`, changing only names/test visibility.
Its shared XYZ/scalar/packed-byte recorders, decoder and all original shaders
must match the parent. Whole-source comparison bounds production differences
to the conditional float recording path, private planner/recorder, lazy cache
and shader. The original fallback commands, byte tail, submit/readback and
validation must match the parent exactly after normalization.

Planner fixtures cover all one-to-four channel sizes, zero/five/seven/usize-max
attributes, exact/one-byte-short storage and buffer bounds, binding counts,
workgroup size/invocation limits, exact/one-cell-over dispatch and u32 extrema.
A CPU discriminator checks that division and reciprocal multiplication can
produce different f32 results.

Direct kernel fixtures compare the fused output with original XYZ/scalar GPU
passes for all four attribute counts at 1/3/17/255/256/257/1025/65,537 cells,
with zero/one/three/four/seven/seventeen points per cell and reversed indices.
Signed zero, subnormal, finite, infinity and NaN inputs are included. Finite/
infinity words must match exactly; NaN classifications must match, without a
NaN payload promise for arithmetic. Eight nonzero trailing guard words must
remain untouched and empty cells must match the old zero result.

Public calls compare the complete parent and an independent CPU reference at
0/1/17/257/1025/65,537/1,048,577 cells; 0/1/2/3/4/5/7 attributes; zero/three
byte channels; and one/three/four/seven scattered points per cell. Byte cases
use one/four lanes with integral means. Finite parent GPU results must match
bitwise; the CPU float oracle allows absolute error at most
`2e-6 * max(abs(reference), 1)`. The 257-cell
cases use the GPU sorter. These exercise both supported fusion and channel-count
fallback, validation precedence, owned results after dropped inputs and pool
clears. Synthetic limits test capability/size fallback decisions; no unsupported
physical adapter is claimed. Existing combined-call regressions also cover
updated host inputs, live output mutation, empty validation and pool reuse.
Concurrent first calls must share a stable lazy pipeline and preserve outputs.

## Measurement method

The benchmark compares the complete public resident-XYZ/host-attribute call with
the frozen parent. Per-call host attribute uploads, allocations, record/compute/
copy/submit, map/wait, decoding and owned return are timed. CPU source/expected
generation, input reference arrays, XYZ/segment setup, output assertions and
output drop are outside timing. Three warmups exclude cold pipeline creation.
This is warm complete-call latency, not just GPU kernel or full voxel sorting.

Fixed (cells, float attributes, byte attributes, points per cell) conditions:
(16,1,0,4), (16,4,3,4), (4096,4,0,4), (65537,3,0,3),
(262144,1,0,4), (262144,4,3,4), (1048576,1,0,4), (1048576,4,3,4).
All eight use 30 alternating paired samples after three warmups. Every output
and capacity is checked for each invocation. The non-power-of-two condition
specifically retains scalar attribute division. The environment records packed
planner support at the largest four-attribute condition without adapter names.

Two surveys each use a bounded three-minute quiet scheduling window and
before/during/after competing-work checks. Overlap discards the entire trial;
other work is left running. Source/executable SHA-256 hashes are captured after
validation and checked after attempts. Paired log old/new ratios use 10,000
bootstrap samples, sorted indices 249/9749 for 95% intervals, seeded by
2026101551 + 100*survey + 10*condition_index + JSON_field_index. Ratios above
one favor fusion; intervals crossing one are inconclusive. Zero samples suppress
that ratio. All favorable, adverse and inconclusive rows are retained; paired
ratios are not ratios of the reported medians.

Receipts are local under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with `voxel-fused-`
prefix. Published evidence omits local adapter model names.

## Completed checks and measurements

All 53 native voxel-suite regressions and 18 voxel-filter regressions passed
(71 distinct local tests), including four new planner/kernel/public/concurrent
regressions. The native device satisfied the fusion planner. Direct GPU tests
compared 1,481,722 output elements against original XYZ/scalar passes:
finite/infinity words matched exactly, NaN classifications matched and empty
cells/guard words passed. Complete public tests checked 238,534,500
parent/CPU elements through 1,048,577 cells, including three/seven-point
means, one-to-four active attributes, zero/five/seven-attribute fallback,
ordered validation, mixed byte outputs and owned results. Planner cap/size
fixtures and concurrent lazy initialization passed.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with
warnings denied, formatting and diff checks passed. Whole-source comparison
verified the complete frozen parent call, unchanged fallback/byte/readback
bodies and 360 unchanged GPU/image sources including all 99 original shaders.
All 370 source/executable hashes stayed fixed after validation and measurement
attempts. Tracked/new-file and published-body privacy scan matches were zero.

Neither bounded quiet scheduling window completed an isolated trial because
competing work remained. No timing sample from those attempts is accepted or
published; no complete-call latency improvement is claimed. The deterministic
evidence establishes fewer passes/objects/copies with unchanged payloads.

One additional functional invocation passed all eight benchmark conditions,
including the largest four-attribute planner path, exact old/new outputs and
capacities for every warmup and paired call. It is separate from the 71 regular
tests; its non-isolated timing samples are excluded from performance evidence.
