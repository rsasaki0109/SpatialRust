# Broader fused first-point voxel gathering

`gather_voxel_first_xyz_and_multi_gpu` already had a fused four-float-attribute,
no-byte-attribute path. That branch keeps its priority and original commands.
A new independently lazy path handles one-to-three float attributes and
one-to-four float attributes mixed with byte channels when the shared planner
accepts the device and output size. Other public entry points are unchanged.

One invocation copies first-point XYZ and active float attributes as u32
storage words into planar output. There is no floating-point arithmetic,
reduction, workgroup storage, barrier or atomic in this new shader. Signed
zero, subnormal, infinity and NaN payload words remain intact. A cell whose
end is at or before its start retains input point-zero XYZ; one scalar float
attribute returns positive zero, while multi-channel attributes copy input
point zero. These are the original scalar/multi gather rules. Packed byte
gathering remains separate and preserves its own empty-cell zero.

The unchanged shared fused-output planner requires one-to-four attributes,
ten storage/eleven total bindings, 256-lane support, bounded dispatch extent,
u32 output indexing, and checked output bytes fitting both storage-binding
and whole-buffer limits. Unsupported cases, zero/five-or-more attributes
retain the original split fallback. The pre-existing four-attribute/no-byte
branch remains governed by its original condition, even when the new planner
rejects a size; this slice does not add new guards to that older branch.

The existing pooled attribute uploads, input labels, handle vector and
post-readback recycling are reused. At most four active input handles are
retained on the new route. Inactive read-only attribute bindings alias the
first input and are skipped by the shader. There is no host concatenation,
extra point-sized host copy, submission or mapped wait. Ordered validation,
empty returns, byte commands, mapped decoding and owned result capacities
remain. No public API, feature/dependency or runtime requirement is added.

## Deterministic execution accounting

For newly eligible calls on a capable device, with N cells, A float
attributes (one through four) and U byte channels:

| Execution/payload | Old split path | New fused path |
| --- | --- | --- |
| Float compute passes, bind groups and uniforms | 2 | 1 |
| Active float output buffers and copy commands | 3+A | 1 |
| Dummy output buffers for A=3/4 | 4 | 0 |
| Dummy output bytes requested for A=3/4 | 16*N | 0 |
| Active float output/download payload bytes | 4*(3+A)*N | 4*(3+A)*N |
| Byte compute/copy commands | U | U |
| Byte output/download payload bytes | 4*ceil(N/4)*U | 4*ceil(N/4)*U |
| Submissions and mapped readbacks | 1 | 1 |

For four float and three byte attributes, total compute passes fall from
five to four and copies from ten to four. The old four-attribute/no-byte
route remains one float pass/output/copy. Old split three/four-channel calls
also initialized a four-byte placeholder input; inactive aliases remove it
on the new route. Active payload transfers, staging size and final host
capacity remain; counts exclude uniforms/placeholders. Fewer objects and
removed dummy requests do not establish peak-memory or latency improvements.

## Correctness method

The complete parent public call is frozen from merge
`a64efa64d76b9d4a71cf5988b58197a921c05d4e`, changing only test visibility/name.
Removing the new planner line and branch must restore that entire function,
including its original four-attribute branch, split fallback, validation,
byte tail, submission, mapped decoding and recycling. Whole-source comparison
also requires the shared planner, original shaders/recorders, pool code and
previous fusion paths to remain unchanged.

Direct GPU comparisons use original XYZ gather and actual scalar/multi2/
multi4 attribute recorders. All one-to-four counts are tested at
1/3/17/255/256/257/1025/65537 cells with zero/one/three/four/seven/seventeen
points per cell and reversed indices. Inputs include arbitrary u32 payload
patterns, signaling/quiet NaNs, signed zero and subnormals. Every output word
must match exactly; eight trailing guard words must remain untouched. The
all-empty one-cell case uses negative-zero point-zero input to distinguish
the scalar empty rule from multi-gather behavior.

Complete public calls compare frozen-parent and exact independent CPU outputs
through 1,048,577 cells at 0/1/2/3/4/5/7 float and zero/three byte attributes,
with one/three/four/seven-point occupancy and GPU-sorted 257-cell inputs.
The existing four-attribute path and zero/five/seven-attribute fallback are
included. A sparse public fixture checks raw XYZ/attribute words, empty and
descending start pairs, and byte-empty zeros against an independent per-cell
oracle for each supported count, both with and without byte channels.
Other tests cover ordered length errors, dropped inputs, cleared pools,
owned output mutation, concurrent first-use cache identity and warm input
reuse after host attributes change to nonfinite/subnormal values. Pool
allocation counts must stay fixed and reuse increments must equal the active
attribute count. Existing planner tests cover synthetic capability/size
boundaries, without claiming a physically limited-device run.

## Measurement method

The ignored release benchmark times complete resident-XYZ/segment,
host-attribute public calls: pooled uploads, resources, record/compute/copy/
submit, map/wait, decode/recycle and owned return. Setup, expected values,
assertions and output destruction stay outside the timer. Three warmups
precede thirty alternating paired samples at (cells, float, byte, occupancy):
(16,1,0,4), (16,4,3,4), (4096,4,0,4), (65537,3,0,3),
(262144,1,0,4), (262144,4,3,4), (1048576,1,0,4), (1048576,4,3,4).
The 4096/four-float/no-byte condition is the unchanged legacy-path control.
Every invocation checks exact outputs and capacities. Sorting and cold
pipeline creation are excluded.

Two surveys each allow a one-minute quiet scheduling window. Three
five-second quiet checks precede attempts, and process censuses before,
during and after the owned child discard overlapping trials. Only owned
children may be cancelled; unrelated jobs continue. Source/executable hashes
are captured after validation and verified after measurement. Accepted paired
log old/new ratios use 10,000 bootstrap resamples with seed
`2026101554 + 100*survey + 10*condition_index + JSON_field_index`; sorted
indices 249/9749 define 95% intervals. Ratios above one favor fusion;
intervals crossing one are inconclusive. Zero samples suppress their ratio.
All favorable, adverse and inconclusive rows are retained.

Local receipts use `voxel-first-gather-` under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`.
Published evidence omits local adapter model names.

## Completion evidence

All 65 native voxel regressions and 18 voxel-filter regressions passed
(83 distinct local tests), including five new kernel/public/sparse/pool/
concurrent regressions. The native device accepted fusion. Direct GPU
fixtures compared 1,481,722 words against original XYZ and scalar/multi2/
multi4 recorders, all bit-exact including signaling/quiet NaNs and signed
zeros. Empty-cell rules and trailing guards passed. Public matrix tests
checked 238,698,596 parent/CPU elements through 1,048,577 cells, covering
one-to-four new cases, legacy four-float/no-byte control, zero/five/seven
attribute fallback, mixed bytes, ordered errors and owned outputs. Sparse
public oracles passed empty/descending start pairs with raw payload words.
All one-to-four warm pool cases retained allocation counts, exact reuse
increments, fresh changed-host inputs and prior owned outputs. Concurrent
first-use cache identity and existing planner boundaries passed.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with warnings
denied, formatting and diff checks passed. Whole-source audit verified the
complete frozen public parent, unchanged legacy/fallback/byte/map/recycle
bodies and 375 unchanged GPU/image sources including all 102 original
shaders. The shared planner, pool code and previous fusion kernels are
unchanged. All 385 source/executable hashes stayed fixed after validation
and measurement attempts. Tracked/new-file and published-body privacy scan
matches were zero.

Neither bounded one-minute window completed an isolated trial because
competing work remained. No timing sample from those attempts is accepted
or published, and no latency improvement is claimed. Deterministic command
and dummy-output accounting remains separate from performance.

One additional functional invocation passed all eight benchmark conditions,
including the largest mixed-byte path and unchanged legacy control, with
exact outputs/capacities on every warmup/paired call. It is separate from the
regular tests; all non-isolated timing samples are excluded.
