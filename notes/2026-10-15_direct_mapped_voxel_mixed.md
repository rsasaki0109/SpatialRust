# Direct mapped combined voxel readback

All four combined XYZ/float/u8 APIs now decode mapped staging directly into
their final owned channel vectors:

- `gather_voxel_first_xyz_and_multi_gpu`
- `gather_voxel_first_xyz_and_average_multi_gpu`
- `reduce_voxel_centroids_xyz_and_average_multi_gpu`
- `reduce_voxel_centroids_xyz_and_gather_first_multi_gpu`

Previously the float region copied into one flat vector before splitting.
The returned X vector kept the entire flat allocation, while Y, Z and every
float attribute copied again. The byte region copied into a raw host byte
vector, converted u32 words into a flat byte vector, then copied each byte
channel again when more than one byte attribute was present.

The private decoder copies each float plane once and extracts each byte
plane's u32 low eight bits directly into its final vector. The existing unmap
guard is declared before the mapped view, so the view drops before unmapping
on return or unwind. No reference to mapped storage escapes. Upload recycling
in the two first-attribute gather routes follows successful final decoding;
the old byte-channel split happened after recycling. Map failures still return
before recycling. This host allocation ordering change adds no device operation.

For N output cells, F = 3 + float attribute count and U byte attributes:

| Payload accounting | Parent | Direct |
| --- | --- | --- |
| Returned float capacity bytes | 4*(2*F-1)*N | 4*F*N |
| Float host copy bytes | 4*(2*F-1)*N | 4*F*N |
| Byte payload allocations over construction, U = 0 | 0 | 0 |
| Byte payload allocations over construction, U = 1 | 5*N | N |
| Byte payload allocations over construction, U > 1 | 6*U*N | U*N |
| Returned byte capacity bytes | U*N | U*N |
| GPU float + byte output staging bytes | 4*(F+U)*N | 4*(F+U)*N |

Byte host memcpy is reduced by 4*U*N plus U*N when U > 1; low-byte decode
still writes U*N final elements. With 1,048,576 cells, four float attributes
and three byte attributes, float returned capacity and repeated-copy payload
fall by 24 MiB. Byte payload allocations removed over construction total
15 MiB, giving a combined 39 MiB reduction over construction. Only the float
24 MiB is a reduction in retained returned payload capacity. Counts exclude
vector metadata, allocator rounding, backend allocations and GPU resources;
they are not RSS or peak-memory measurements.

Public signatures, coordinate/attribute ordering, empty returns, ordered
length errors, host attribute uploads, device inputs/outputs, uniforms,
compute/copy passes, submissions, map/wait diagnostics and shaders are
preserved. The fused XYZ/four-float gather path remains eligible under the
same conditions. Byte input padding and four-byte staged byte outputs remain.
No dependency, unsafe code, hidden transfer or synchronization is added.

## Verification

The complete four public parent functions and five decoding helpers are
frozen from merge `dcdbe2ea3a9a49bec16e3e01387a53b9945df58b`; only names and
test visibility change. Shared recording helpers are used only after a
whole-source comparison verifies that the only production changes are the
decoder replacement, four readback tails/imports and a native test declaration.
All other GPU/image sources and all shaders must match the parent exactly.

Raw mapped fixtures use float/u8 attribute counts (0,0), (1,0), (4,0), (7,0),
(0,1), (0,3), (1,1), (4,3), with 0/1/17/65,537/262,145 cells and trailing
padding. Signed zeros, NaN payloads, infinities, subnormals, extrema and mixed
u32 patterns must preserve exact float bits and low bytes, ignoring upper
byte-output bits. Tests check parent/direct capacities, repeated mappings,
independently mutated outputs and dropped staging. Both float-region and
byte-region bounds panics must unmap and allow subsequent direct/parent maps.

Complete public calls cover all four modes, the same attribute combinations,
0/1/17/257/65,537 cells, and one/four scattered points per cell. Dyadic float
inputs and integral byte means have independent exact CPU oracles. The
257-cell fixtures use GPU key sorting; other fixtures upload valid grouped
metadata explicitly. Tests cover empty validation precedence, float-before-byte
length errors before resource use, changed host attributes, dropped inputs,
live output mutation, pool clears and concurrent calls. An additional regular
regression checks all four complete modes at 1,048,577 cells with four float
and three byte attributes against both the parent and CPU, including capacity.

Validation receipts and source/executable SHA-256 hashes are stored locally in
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with prefix
`voxel-mixed-`. Published evidence excludes local adapter model names.

This slice verifies output correctness and deterministic payload capacity.
No elapsed-time benchmark is reported; these counts do not establish a speed
improvement for complete calls or applications.

## Completed checks

All 45 native GPU voxel regressions and 18 voxel-filter regressions passed
(63 distinct local tests). The new raw matrix checked 48,171,900 exact
float/byte elements, and the public matrix checked 77,394,912
frozen/analytic/repeated elements. These counts exclude the additional
million-cell, changed-input, empty/error, concurrency and panic/remap checks.
The million-cell regression passed all four complete modes at 1,048,577
cells, with four float and three byte attributes, against both CPU and frozen
parent outputs and with exact returned capacity assertions.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with
warnings denied, formatting and diff checks passed. The source audit
verified nine complete frozen functions, 353 unchanged GPU/image sources
including all 97 WGSL files, and pinned 360 source/executable SHA-256 hashes.
The tracked/new-file and PR-body adapter privacy scan had zero matches.

An initial raw fixture incorrectly invoked the frozen byte decoder on an
empty byte vector, exposing its alignment precondition. The fixture now
matches the parent public API's empty/absent-byte guards. Frozen parent
functions and production behavior were not changed for this fixture repair;
the completed validation run used the repaired test source.
