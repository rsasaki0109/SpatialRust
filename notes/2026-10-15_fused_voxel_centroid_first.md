# Fused voxel centroids and first-point attributes

`reduce_voxel_centroids_xyz_and_gather_first_multi_gpu` now has a private
capability/size-gated pass for centroid XYZ and one-to-four first-point float
attributes. Its old XYZ reducer and scalar/two/four-channel gather recorders
remain as the fallback. Other public entry points, previous fusion shaders,
runtime requirements, dependencies and public signatures are unchanged.

One invocation owns one cell. XYZ preserves the original point order, three
independent f32 sums and `sum * (1.0 / f32(count))`. Active float attributes
are copied from the first point without arithmetic. Empty cells retain
positive-zero XYZ and the existing channel-specific attribute behavior:
one attribute uses scalar gather's positive zero; two-to-four attributes use
multi-gather's input point-zero value. This distinction is intentionally
preserved, including signed zero and NaN bits in gathered values.

The independent pipeline/layout initialize lazily in the reduction cache.
The unchanged shared fused-output planner requires ten storage bindings,
eleven total bindings, 256-lane compute support, bounded dispatch, u32 output
indexing and checked output bytes fitting both binding and buffer limits.
Zero/five-or-more attributes and unsupported limits/sizes retain the original
commands. Inactive read-only bindings alias the first attribute input and
are skipped by the shader. No workgroup storage, barrier or atomic is added.

Active attribute inputs use the same `upload_f32_storage` pool, input label
and per-channel data. The existing upload-handle vector holds at most four
inputs on the fused route, and the unchanged post-readback tail recycles
them only after successful decoding. No host concatenation, additional
point-sized host copy, GPU submission or mapped wait is added. The encoder
and empty handle vector now precede fallback XYZ output allocation; ordered
validation still precedes device resources. The full fallback, packed byte
commands and direct mapped decoder are unchanged.

## Deterministic execution accounting

On a device accepted by the fusion planner, the old gather cache supports
four-channel batches. For N cells, A float attributes (one through four) and
U byte attributes:

| Execution/payload | Original capable-device path | Fused path |
| --- | --- | --- |
| Float compute passes, bind groups and uniforms | 2 | 1 |
| Active float output buffers/copy commands | 3+A | 1 |
| Dummy output buffers for A=3/4 | 4 | 0 |
| Dummy output bytes requested for A=3/4 | 16*N | 0 |
| Active float output/download payload bytes | 4*(3+A)*N | 4*(3+A)*N |
| Byte compute/copy commands | U | U |
| Byte output/download payload bytes | 4*ceil(N/4)*U | 4*ceil(N/4)*U |
| Submissions and mapped readbacks | 1 | 1 |

The old three/four-channel recorder creates all four dummy outputs even when
none is bound for four active channels. It also initializes a four-byte
placeholder input, which the fused route replaces with inactive aliases.
For four float and three byte attributes, total compute passes fall from
five to four and copies from ten to four. Float/u8 payload uploads, active
output payloads, staging and returned host capacity are unchanged. These
payload counts exclude uniforms and placeholders. Fewer objects and removed
dummy requests do not establish a particular peak-memory or latency saving.

## Correctness method

The complete public parent call is frozen from merge
`ef318dd6631d6703d85cd548dabbb1900856457a`, with only test name/visibility
changes. Whole-source comparison verifies all original GPU/image sources
except the bounded public branch, cache additions and module declarations.
In particular, the shared planner, scalar/multi gather recorders, original
shaders, pool upload/recycle implementation, fallback, byte tail, final map
and ordered validation must remain unchanged.

Direct GPU fixtures compare centroid output with the original XYZ reducer
and attributes with the actual original scalar/multi2/multi4 recorders.
They cover all one-to-four attribute counts, 1/3/17/255/256/257/1025/65537
cells, zero/one/three/four/seven/seventeen points per cell, reversed indices,
signed zero, subnormal, finite, infinity and NaN values. XYZ finite/infinity
bits and arithmetic NaN classifications must match; attribute bits must match
even for NaNs. A one-cell all-empty case uses negative-zero input point zero
to distinguish scalar zero from multi-gather. Eight trailing guard words must
remain untouched.

Public calls compare frozen-parent outputs and independent CPU values
through 1,048,577 cells at 0/1/2/3/4/5/7 float and zero/three byte attributes,
with non-power-of-two occupancy and GPU-sorted 257-cell inputs. Finite parent
results must match bitwise; CPU XYZ allows absolute error
`2e-6 * max(abs(reference), 1)`, while first attributes and bytes are exact.
Byte fixtures use one/four-point occupancy. Tests cover ordered length
errors, dropped inputs, cleared pools and concurrent first-use cache sharing.
Another regression warms the pool for every supported channel count,
changes host attributes to NaN/negative-zero/infinity/subnormal values and
byte inputs, then checks exact allocation/reuse counts, fresh outputs and
owned-result independence. Existing planner tests cover synthetic capability
and size boundaries, without claiming a physically limited-device test.

## Measurement method

The ignored release benchmark times complete public calls with resident XYZ
and segments and borrowed host attributes: pooled uploads, resources,
compute/copies/submit, map/wait, final decoding, recycling and owned return.
Setup, output assertions, expected values and output destruction are outside
the timer. Three warmups precede thirty alternating paired samples at:
(cells, float attributes, byte attributes, points per cell) =
(16,1,0,4), (16,4,3,4), (4096,4,0,4), (65537,3,0,3),
(262144,1,0,4), (262144,4,3,4), (1048576,1,0,4), (1048576,4,3,4).
Each invocation checks exact outputs/capacities. This measures warm completed
calls and excludes sorting and cold pipeline creation.

Two surveys each allow a one-minute quiet scheduling window. Three
five-second quiet checks precede each attempt; process censuses before,
during and after the owned benchmark discard overlaps. Only owned children
may be cancelled; unrelated work continues. Source and executable SHA-256
hashes are captured after validation and verified after measurement.
Accepted paired log old/new ratios use 10,000 bootstrap samples with seed
`2026101553 + 100*survey + 10*condition_index + JSON_field_index`; sorted
indices 249/9749 define 95% intervals. Ratios above one favor fusion and
intervals crossing one are inconclusive. Zero samples suppress their ratio.
All favorable, adverse and inconclusive rows are retained.

Local receipts use the `voxel-centroid-first-` prefix under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`.
Published evidence omits local adapter model names.

## Completion evidence

All 60 native voxel-suite regressions and 18 voxel-filter regressions passed
(78 distinct local tests), including four new kernel/public/concurrent/pool
regressions. The native device accepted fusion. Direct GPU fixtures compared
1,481,722 output elements against the original XYZ reducer and actual
scalar/multi2/multi4 gather recorders. Finite/infinity XYZ bits and arithmetic
NaN classifications matched, while attribute bits matched including NaNs,
signed zeros and the scalar/multi empty-cell distinction. Guards passed.
Complete public tests checked 238,534,500 parent/CPU elements through
1,048,577 cells, including non-power-of-two means, mixed bytes, zero/five/seven
attribute fallback, ordered errors and owned outputs. The pool regression
passed for every one-to-four attribute count with changed nonfinite host
inputs, no new warm input allocations, exact reuse increments and retained
outputs. Existing planner boundaries and concurrent lazy initialization passed.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with warnings
denied, formatting and diff checks passed. Whole-source audit verified the
complete frozen parent, fallback/byte/map/recycle tails and 370 unchanged
GPU/image sources including all 101 original shaders. The shared planner,
pool implementation and previous fusion kernels are unchanged. All 380
source/executable hashes stayed fixed after validation/measurement attempts.
Tracked/new-file and published-body privacy scan matches were zero.

Neither bounded one-minute window completed an isolated trial because
competing work remained. No timing samples from those attempts are accepted
or published, and no latency improvement is claimed. Deterministic command
and dummy-allocation accounting is reported separately from performance.

One additional functional invocation passed all eight benchmark conditions,
including the largest four-attribute path, exact outputs and capacities on
every warmup/paired call. It is separate from the regular tests; all of its
non-isolated timing samples are excluded from performance evidence.
