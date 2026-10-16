# Coalesce host voxel XYZ centroid submissions

`reduce_voxel_centroids_xyz` previously called the complete scalar reduction
API once per axis. Each call uploaded the same point indices and cell starts,
submitted one scalar pass/copy and synchronously mapped/waited for its result.
The host wrapper now tries one private batch after its original shape check.
It uploads each XYZ input once and shares one indices/starts pair across
three unchanged scalar passes, one submission and one owned XYZ readback.
The GPU-resident XYZ APIs and their packed/vector kernels are unchanged.

## Scalar math, ordering and admission

The batch reuses `record_voxel_reduce_f32_pass`, whose uniform bytes,
bindings, pipeline and dispatch commands are source-equivalent to the old
scalar dispatcher. The scalar shader and point permutation are unchanged.
Each pass writes a single private C-float output buffer, then immediately
copies that axis into its own region of a 12*C-byte staging buffer before
the next pass overwrites the output. Encoder command order preserves these
copies; normal wgpu resource tracking supplies transitions. No output is
recycled or shared with another API call. There are still three scalar
compute dispatches and three GPU copies, without float reassociation.

The pure planner checks nonzero/u32 cell and index counts, original scalar
compute/binding/uniform capabilities, index/output binding bytes, combined
staging size, device buffer size and host capacity. Source value and actual
start-array byte lengths are checked before batch allocations. The batch
uses the original four storage bindings plus one uniform, without requiring
the wider resident XYZ capability envelope. Oversized combined staging,
host/index/capability rejections or unsupported source sizes return None and
reach the original three-call fallback. The old mismatch check still runs
first, including its existing found-y behavior for a z-only mismatch.
Empty segments still return three empty Vecs without device work; empty
values with occupied segments retain the original scalar error.

The API still allows unused source tails and does not add a source-length
equals-index-count check. Keys and cell_counts remain unused by GPU math.
The complete scalar fallback, resident APIs, multi-attribute APIs, shaders,
pipelines, caches, readback guards and all upload/readback helpers remain.
No unsafe code, public API, dependency or global pool is added.

## Source-verified transfer and allocation ledger

Let N be each axis length, P the number of point indices, S the actual
cell-start array length and C the number of cells. Payload H2D bytes change
from 12*N+12*(P+S) to 12*N+4*(P+S), eliminating 8*(P+S) duplicated metadata
bytes. The same three 16-byte uniforms are excluded from both formulas.
D2H payload remains 12*C bytes. Source counts, not a new public runtime
receipt or GPU timestamp, establish this ledger:

| Operation | Original | Coalesced |
| --- | ---: | ---: |
| XYZ input uploads | 3 | 3 |
| Indices / starts uploads | 3 / 3 | 1 / 1 |
| Scalar compute passes / result copies | 3 / 3 | 3 / 3 |
| Submit / readback-map / synchronous wait calls | 3 / 3 / 3 | 1 / 1 / 1 |
| Result scratch / staging buffer creations | 3 / 3 | 1 / 1 |
| Uniform buffers / bind groups | 3 / 3 | 3 / 3 |
| Total explicit buffer creations | 18 | 10 |

For N=P=1,048,576 and S=C=262,144, H2D payload falls from 28,311,552 to
17,825,792 bytes, removing 10,485,760 bytes (10 MiB), with unchanged
3,145,728-byte D2H payload. This is deterministic transfer accounting,
not a latency measurement or peak-memory claim.

The batch holds all three uploaded axes until submission and uses one
larger staging buffer. Thus fewer allocation requests do not imply lower
peak GPU memory, RSS or backend allocation work. The combined buffer must
fit the existing limits. Final outputs still own three exact-C-capacity
Vecs with no outer XYZ Vec or point-sized host scratch. The shared XYZ
reader keeps its mapped guard and existing native copy policy: at least
1,048,576 cells can use up to three copy workers including the caller;
small/one-CPU/WASM paths stay serial. This policy is reused unchanged,
whereas the original scalar fallback still copies each result serially.

## Current parent and validation

The baseline is main commit 6ce2bdddb51bdb3124cc7d36b50bc0a81f28d42f.
Tests freeze the complete old host XYZ wrapper. Its scalar upload,
dispatch and readback delegates remain source-identical, so old calls
still perform three independent complete calls. Audit removes only the
new selection block to restore the entire original reduction file and
checks the reused scalar uniform/bindings/dispatch against the original
dispatcher. The other 462 original GPU/image
Rust/WGSL files remain unchanged; the manifest pins 469
source/executable hashes after final validation and before/after timing.
Published evidence omits the local adapter model name.

Five new tests verify 252 planner cases,
19,272,306 complete current-parent/independent-f64 words and
11,565 sparse/numeric words. Planner cases cover exact buffer,
host, dispatch, scalar binding and u32 count boundaries, nonoverlapping
staging regions and the source ledger. A scalar-sized output can fit while
combined staging is rejected; original fallback source remains exact.

Complete cases cover 0/1/17/255/256/257/65537/1048575/1048576/1048577 cells,
occupancy one/four, unused input tails, sorted CPU-built segments, exact
capacities and outputs retained after source destruction/pool clearing.
All regular fields compare bitwise with complete parents and independent
f64 means of dyadic fixtures. Sparse/permuted cases include ignored
cell_counts, empty cells and occupancies 1/2/3/9/257; finite means compare
to independent f64 within four f32 epsilons times max(abs(mean),1), while
new/old GPU results remain bit-exact. Nonfinite words include NaN payloads,
infinities, subnormals, extreme magnitudes and signed zeros, preserving
the original GPU's bits. Validation/empty cases retain original errors;
four concurrent calls retain outputs after host-axis/index mutation.

All 155 native voxel, 131
image, 18 filter and 2
facade tests pass (306 distinct tests). All-feature/all-target Clippy
denies warnings; CPU-only, minimal gpu-wgpu WASM and CI gpu-image/
gpu-aoso-staging WASM with warnings denied, formatting and diff checks pass.

Sources: C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_host_xyz.rs,
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_reduce.rs,
and C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_host_xyz_tests.rs.
Receipts: C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-host-xyz-*.

## Complete-call timing

The benchmark times complete public calls from borrowed host XYZ and CPU
segments through allocation, uploads, three scalar kernels/copies, GPU
submit/map/wait and owned output collection. Fixture/reference preparation,
assertions and output destruction are outside the clock. Three warmups
precede thirty alternating paired parent/new calls. Conditions
(cells,points-per-cell) are (16,4), (257,4), (65537,1), (65537,4),
(262145,1), (1048575,1), (1048576,1), (1048577,1). Both old and new calls
upload their own inputs during the timed boundary. Large new calls also
exercise the unchanged shared XYZ copy policy; original calls remain serial.

Two one-minute scheduling windows require three five-second quiet checks
and process censuses before/during/after. Overlap discards trials; only owned
benchmark children may be cancelled. Accepted paired log old/new ratios use
10000 bootstrap resamples, seed 2026101618 + 100*survey + 10*condition_index
plus JSON_field_index, and sorted indices 249/9749 for 95% intervals. All
outcomes remain; ratios above one favor the new route, intervals crossing
one are inconclusive and zero samples suppress ratios.

Neither bounded one-minute quiet window provided an isolated trial because competing work remained. No timing samples are accepted or published, and no latency improvement is claimed. A separate complete-call functional invocation passed all eight conditions and exact owned outputs/capacities; its non-isolated timing samples are excluded.
