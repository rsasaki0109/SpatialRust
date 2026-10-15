# Fused voxel first-point XYZ and attribute average

`gather_voxel_first_xyz_and_average_multi_gpu` previously recorded one XYZ
gather pass and one scalar average pass per float attribute. A private path
now records one combined pass for one-to-four float attributes when the shared
fused-output planner accepts the device limits and output size. Calls with
packed byte attributes can also use this path. Other public entry points and
the previous centroid/attribute-average fusion are unchanged.

One invocation owns one cell. It copies the first point's XYZ without float
arithmetic, then visits the cell's points in their original order and computes
each active attribute with the old scalar sum and `sum / f32(count)`.
Coordinate loads retain signed zeros and nonfinite representations; arithmetic
NaN payloads in attribute averages have no new guarantee. The existing empty
cell rule is deliberately preserved: XYZ comes from input point zero, while
attribute averages are positive zero. This rule differs from centroid XYZ.

The independent pipeline and layout initialize lazily in the gather cache.
The existing capability planner is reused unchanged: ten storage bindings,
eleven total bindings, 256-lane workgroup support, bounded dispatch extent,
u32 output indexing and checked output bytes fitting both the storage binding
and whole-buffer limits. Unsupported devices/sizes and zero/five-or-more
float attributes retain the original separate commands. Device requirements
requested at runtime construction do not change.

Active attribute buffers are uploaded as before; inactive read-only bindings
alias the first channel and are not read. A vector holds at most four input
buffer handles, beside a fixed eleven-entry bind group description. There is
no host concatenation, additional point-sized host copy, shared memory,
barrier, atomics or extra submission. The encoder is now allocated before
fallback float outputs; ordered validation still precedes all resources.
The fallback compute/copy body, byte tail and mapped readback are unchanged.

For N cells, A float attributes (one through four) and U byte attributes:

| Execution/payload | Separate float path | Fused float path |
| --- | --- | --- |
| Float compute passes, bind groups and uniforms | 1+A | 1 |
| Float output buffers and copy commands | 3+A | 1 |
| Float output payload/download bytes | 4*(3+A)*N | 4*(3+A)*N |
| Byte compute/copy commands | U | U |
| Byte output/download bytes | 4*ceil(N/4)*U | 4*ceil(N/4)*U |
| Submissions and mapped readbacks | 1 | 1 |

Four float and three byte attributes therefore use four total compute passes
instead of eight, and four output copies instead of ten. Transfers, summed
device output bytes, staging size and returned host capacity are unchanged.
These command counts do not establish a latency or peak-memory improvement;
register pressure, conditionals and backend scheduling can affect latency.

## Correctness method

The complete parent public call is frozen from merge
`db5210b648be628c9f5536ee7d034b1c56695b16`, with only test visibility/name
changes. Whole-source checks require the shared scalar/XYZ/byte recorders,
decoder, prior fused planner/recorder and every original shader to remain
unchanged. Source comparison separately verifies the complete fallback, byte
tail, submit/readback and ordered validation around the new branch.

Direct GPU fixtures compare combined outputs with original XYZ gather and
scalar averages at 1/3/17/255/256/257/1025/65537 cells and all one-to-four
attribute counts. Cells contain zero/one/three/four/seven/seventeen points,
with reversed indices, signed zero, subnormal, finite, infinity and NaN inputs.
XYZ words must match bitwise even for NaNs; finite/infinity attribute words
must match bitwise and arithmetic NaN classifications must agree. An all-empty
single-cell fixture uses negative-zero point-zero coordinates to distinguish
gather semantics from zero-filled centroid outputs. Eight trailing guard
words must remain untouched.

Complete public calls compare the parent and independent CPU values through
1,048,577 cells, using 0/1/2/3/4/5/7 float and zero/three byte attributes.
Non-power-of-two occupancies exercise original division, and 257-cell fixtures
use the GPU sorter. Parent outputs must match exactly; the finite CPU oracle
allows absolute error `2e-6 * max(abs(reference), 1)`. Byte fixtures use
one/four points with integral means. Checks include ordered length errors,
owned results after dropping inputs/clearing pools and concurrent first calls
sharing the same lazy pipeline. Existing planner boundary regressions cover
capability/size fallback; these do not claim tests on a physically limited
adapter. Existing public regressions cover empty returns, host input updates,
live result mutation and pool reuse.

## Measurement method

The ignored release benchmark times complete public calls with resident XYZ
and segments and borrowed host attributes: attribute upload, resources,
compute/copies/submit, map/wait, decoding and owned return. Setup, expected
values, output assertions and output destruction are outside the timer.
Three warmups precede 30 alternating paired samples per condition:
(cells, float attributes, byte attributes, points per cell) =
(16,1,0,4), (16,4,3,4), (4096,4,0,4), (65537,3,0,3),
(262144,1,0,4), (262144,4,3,4), (1048576,1,0,4), (1048576,4,3,4).
Every invocation checks exact parent output and final vector capacities.
This boundary excludes sorting and cold pipeline creation.

Two surveys each allow a bounded one-minute quiet scheduling window. Three
five-second quiet checks precede each attempt; process census checks during
the child run and after exit discard overlapping work. Only an owned child
can be cancelled; unrelated jobs continue. Source and executable SHA-256
hashes are captured after validation and verified after measurement.
For accepted surveys, 10,000 bootstrap resamples of paired log old/new ratios
use seed `2026101552 + 100*survey + 10*condition_index + JSON_field_index`;
sorted indices 249/9749 define 95% intervals. Ratios above one favor fusion;
intervals crossing one are inconclusive. Any zero sample suppresses its ratio.
All favorable, adverse and inconclusive conditions are retained.

Local receipts use the `voxel-first-avg-` prefix under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`.
Published evidence omits local adapter model names.

## Completion evidence

All 56 native voxel-suite regressions and 18 voxel-filter regressions passed
(74 distinct local tests), including three new kernel/public/concurrent
regressions. The native device satisfied the shared fusion planner. Direct
GPU fixtures compared 1,481,722 output elements: XYZ bits matched exactly
including NaNs and empty-cell point-zero coordinates; finite/infinity
attribute bits and arithmetic NaN classifications matched. Output guards
passed. Complete public tests checked 238,534,500 parent/CPU elements through
1,048,577 cells, including non-power-of-two means, mixed byte outputs,
zero/five/seven-attribute fallback, ordered errors and ownership. Existing
planner boundary fixtures and new concurrent lazy initialization passed.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with warnings
denied, formatting and diff checks passed. Whole-source comparison verified
the frozen complete public parent, unchanged fallback/byte/readback bodies,
the unchanged shared planner and 365 unchanged GPU/image sources including
all 100 original shaders. All 375 source/executable hashes stayed fixed
after validation and measurement attempts. Tracked/new-file and published-body
privacy scan matches were zero.

Neither one-minute quiet scheduling window completed an isolated trial
because competing work remained. No timing sample from those attempts is
accepted or published; no complete-call latency improvement is claimed.
Deterministic evidence establishes fewer commands/objects with unchanged
transfer payloads.

One additional functional invocation passed all eight benchmark conditions,
including the largest four-attribute planner path and exact outputs/capacities
for every warmup and paired call. It is separate from the regular tests; all
non-isolated timing samples are excluded from performance evidence.
