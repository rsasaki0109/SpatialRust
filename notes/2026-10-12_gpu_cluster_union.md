# GPU minimum-root Euclidean component labeling

## Device execution

The Euclidean GPU API previously constructed sparse cells on GPU, downloaded
cell keys/starts/point indices, and ran deterministic union/find on the CPU.
Component connection and final minimum-root compression now run as actual
WGSL compute kernels. The high-level extractor retains CPU cluster-size
filtering, label remapping and output-cloud assembly. This slice changes one
major algorithm phase; other CPU spatial algorithms are not GPU implementations.

Production sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/cluster_union.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/euclidean_cluster.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-segmentation/src/cluster.rs`

Uploaded XYZ buffers stay live through sparse key generation, stable GPU radix
sorting/compaction and component labeling. Each sorted point recovers its
actual owning cell through ascending starts, avoiding coordinate recomputation
and metadata downloads. Signed lexicographic searches find the 27 adjacent
cells; signed offset guards avoid overflow at i32 endpoints. Original scalar
f32 squared-distance order is retained rather than using a vector dot product.
Each undirected qualifying pair is visited once (`j > i`).

Three separately ordered device passes initialize parents, connect components,
and compress roots, followed by root-to-staging copy in the same submission.
The existing sparse-cell construction still synchronizes its four-byte cell
count beforehand. There is no CPU union/find, iterative host convergence check,
device lock or global workgroup barrier.

Atomic compare/exchange hooks a current higher root to a lower root, retrying
both contention and spurious weak-CAS failures. Atomic ancestor shortening
only decreases parents. Every parent remains an in-range index; strict decrease
away from a root prevents cycles. Successful hooks merge components and failed
hooks recompute both endpoints. Once the connection pass completes, compression
resolves each point to the component's minimum original index regardless of
invocation scheduling. Retry semantics follow the
[WGSL atomicCompareExchangeWeak specification](https://www.w3.org/TR/WGSL/#atomicCompareExchangeWeak-builtin).
Forward progress still depends on the device's atomic execution; dense
neighborhoods retain quadratic worst-case pair work, without a duration budget.

The runtime owns lazy cached layouts/pipelines. Parents occupy four bytes per
point in an exclusive submitted scratch lease, fully overwritten by initialization
before reads. Future reuse follows queue order. Staging and returned host roots
are call-owned. XYZ/key uploads are recycled after root completion; segment
outputs remain independently owned. No public signature, dependency or core
type is added, and Rust unsafe remains denied.

## Transfers and validation

For `N` nonempty points and `C` occupied sparse cells, the frozen GPU/CPU hybrid
downloads `4 + 20*C + 4*N` bytes. The new finite-tolerance path downloads
**`4 + 4*N` bytes**, eliminating cell metadata payloads. Both upload `12*N`
XYZ bytes. Kernel uniforms and device storage reads/writes are excluded from
the established transfer counters. Host input validation/origin reduction stay
explicit; APIs still synchronously return a CPU root vector.

Positive-infinite tolerance uses a GPU zero-root initialization only. Finite
coordinates imply one component under the existing infinite-radius distance
predicate. No XYZ upload, sparse sort or cell-count readback is required; only
`4*N` root bytes return. Empty calls perform no device work or transfers.

Point/index/host byte capacity, flat dispatch, workgroup, binding/uniform and
buffer limits are checked before allocation. Finite-tolerance preprocessing
also checks 16-byte key/entry storage, 32-byte uniforms and existing radix
shared-storage requirements. Nonfinite XYZ and nonfinite finite-tolerance
reciprocals reject before GPU work. Accepted sparse-key quantization retains
the existing GPU key implementation and its signed-i32 range.

Policy receipts now name `gpu-component-labeling` for nonempty GPU operations.
Only finite nonempty calls name `gpu-sparse-grid`; empty calls name no device
stage. Placement thresholds and explicit-policy/fallback contracts are retained.
High-level attribute columns, metadata and min/max cluster-size semantics remain.

## Correctness and baselines

Native tests compare **335,276 GPU root words** across independent brute-force
graph traversal, frozen GPU/CPU-hybrid output and analytic component minima.
Cases include singleton/partial dispatch, 255/256/257 boundaries, duplicate
cliques, lattices, isolated points, separated repeated groups, long forward
and reversed chains through 65,537 points, distance next-float boundaries,
infinite radius and extreme finite coordinates. Small graph cases repeat three
times to exercise invocation scheduling. Counts describe repeated comparisons,
not unique values. Dirty all-maximum parent storage and four concurrent callers
with two retained results each exercise initialization, ownership and cache reuse.
Pure limit probes cover exact byte boundaries and unsupported shader capacities;
invalid public inputs leave scratch/cache state unchanged.

The segmentation policy checks exact labeled clouds across six physical column
variants, schema and synthetic metadata, accepted/rejected cluster-size ranges,
and transfer/stage receipts. Infinite and valid empty clouds check actual stage
selection. A native facade smoke compares explicit GPU labels with unchanged
automatic CPU selection below a configured threshold. Adapter initialization
failures are not silently skipped.

Parent: `19b5d0b874bb6a1b8db121eeaba1af2eee702da6`.

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/cluster_host_previous.rs`
  freezes the prior public GPU-grid/CPU-union function without logic changes.
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/cluster_kdtree_cpu.rs`
  preserves the segmentation CPU KD-tree/BFS root function after parameter,
  visibility and result-routing substitutions. This is the CPU reference used
  for timing, rather than a different parallel dense-grid CPU algorithm.

## Measurement boundary

```powershell
cargo test -p spatialrust-gpu --release --features gpu-wgpu paired_gpu_hybrid_and_cpu_cluster_completion -- --ignored --nocapture --test-threads=1
```

Each condition uses three warmups per version and twenty cyclic-order triples:
frozen GPU/CPU hybrid, current GPU components, and the current CPU KD-tree/BFS
root path. All versions produce the same expected minimum roots. Inputs and
equality checking/returned-root destruction are outside timing. GPU timers
include XYZ upload, key construction, radix sorting, cell-count synchronization,
component work and final host return; the old version includes metadata
download and CPU union instead. CPU timing includes KD-tree construction and
BFS. High-level size filtering/label remapping/cloud assembly is excluded from
all versions; these are completed root-API measurements, not full extractor
or registration benchmarks. Upload pooling differences of the actual paths
are included; the comparison does not isolate shader cost alone.

Fixtures cover 257 / 4,096 / 16,384 / 65,537 points in lattice, reversed-chain
and separated-seven-point-group distributions. Correctness/build jobs finish
before isolated timing. Windows rustc 1.98.1; release thin LTO/one codegen unit;
12 available CPU workers; Vulkan/discrete native GPU. Adapter model names are
excluded. WASM validation establishes compilation, not browser execution.

Ratios use paired geometric means, with 95% paired log-ratio bootstrap intervals
from 10,000 resamples, seed 20261012. Greater than one favors current GPU work.
Old-hybrid/current-GPU and CPU/current-GPU comparisons are reported separately.
Intervals describe observed triples rather than systematic host/device bias.
Raw samples and private diagnostics are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-cluster-union-*`.

## Validation and two surveys

| Check | Result |
| --- | --- |
| Native GPU image/AoSoA and point regressions | 185 passed (158 unit + 27 integration), 0 failed, 20 ignored diagnostics |
| Final focused GPU component checks | 6 passed, 0 failed, 1 ignored diagnostic |
| All-feature segmentation | 29 passed, 0 failed |
| Explicit native facade smoke | 1 passed |
| All-feature/all-target GPU and segmentation Clippy | Passed with warnings denied |
| Minimal GPU crate | Compile check passed |
| GPU image/AoSoA WASM and CPU segmentation WASM | Compile checks passed |
| Frozen/source routing audit | Passed |
| Formatting/whitespace | Passed |

All twenty triples per condition are retained in each of two surveys. Shapes
are a half-unit lattice, a reversed three-quarter-unit chain, and separated
groups of seven coincident points. CPU reference times are single-threaded
KD-tree/BFS root work; the available-worker count describes the host and other
existing preprocessing rather than changing that CPU baseline.

| Survey | Points | Shape | Hybrid median ms | GPU median ms | CPU median ms | Hybrid/GPU [95% interval] | CPU/GPU [95% interval] |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | 257 | lattice | 4.919 | 3.042 | 0.180 | 1.502 [1.395, 1.593] | 0.058 [0.055, 0.061] |
| 1 | 257 | reversed chain | 5.012 | 3.148 | 0.065 | 1.648 [1.586, 1.718] | 0.021 [0.020, 0.022] |
| 1 | 257 | separated groups | 4.707 | 3.006 | 0.102 | 1.544 [1.484, 1.582] | 0.033 [0.032, 0.035] |
| 1 | 4,096 | lattice | 7.277 | 3.294 | 8.011 | 2.102 [1.955, 2.225] | 2.438 [2.301, 2.579] |
| 1 | 4,096 | reversed chain | 7.302 | 3.017 | 1.081 | 2.417 [2.360, 2.478] | 0.347 [0.338, 0.355] |
| 1 | 4,096 | separated groups | 5.116 | 2.951 | 1.512 | 1.740 [1.709, 1.771] | 0.521 [0.504, 0.544] |
| 1 | 16,384 | lattice | 18.599 | 7.949 | 38.048 | 2.639 [2.414, 2.914] | 5.873 [5.286, 6.538] |
| 1 | 16,384 | reversed chain | 21.480 | 8.611 | 4.742 | 2.552 [2.449, 2.666] | 0.566 [0.542, 0.592] |
| 1 | 16,384 | separated groups | 13.840 | 8.883 | 6.476 | 1.536 [1.502, 1.570] | 0.724 [0.706, 0.742] |
| 1 | 65,537 | lattice | 51.851 | 20.445 | 150.103 | 2.557 [2.512, 2.605] | 7.380 [7.233, 7.546] |
| 1 | 65,537 | reversed chain | 67.923 | 17.078 | 20.829 | 3.995 [3.864, 4.159] | 1.243 [1.209, 1.280] |
| 1 | 65,537 | separated groups | 27.579 | 16.365 | 30.966 | 1.786 [1.714, 1.865] | 2.009 [1.887, 2.157] |
| 2 | 257 | lattice | 5.343 | 3.336 | 0.198 | 1.596 [1.564, 1.630] | 0.061 [0.060, 0.062] |
| 2 | 257 | reversed chain | 5.009 | 3.053 | 0.059 | 1.638 [1.607, 1.670] | 0.020 [0.019, 0.020] |
| 2 | 257 | separated groups | 5.196 | 3.252 | 0.105 | 1.593 [1.552, 1.634] | 0.033 [0.031, 0.035] |
| 2 | 4,096 | lattice | 7.924 | 3.604 | 8.446 | 2.173 [2.073, 2.257] | 2.360 [2.243, 2.474] |
| 2 | 4,096 | reversed chain | 8.652 | 3.566 | 1.068 | 2.389 [2.292, 2.482] | 0.324 [0.296, 0.359] |
| 2 | 4,096 | separated groups | 6.241 | 3.392 | 1.515 | 1.767 [1.686, 1.859] | 0.509 [0.454, 0.577] |
| 2 | 16,384 | lattice | 19.946 | 7.906 | 44.214 | 2.803 [2.575, 3.080] | 6.190 [5.440, 7.110] |
| 2 | 16,384 | reversed chain | 23.411 | 8.227 | 5.501 | 3.026 [2.857, 3.212] | 0.705 [0.646, 0.771] |
| 2 | 16,384 | separated groups | 14.146 | 8.299 | 10.154 | 1.751 [1.681, 1.844] | 1.074 [0.941, 1.229] |
| 2 | 65,537 | lattice | 75.843 | 26.282 | 179.837 | 3.034 [2.714, 3.388] | 7.951 [7.138, 8.868] |
| 2 | 65,537 | reversed chain | 89.279 | 20.214 | 27.446 | 4.510 [4.266, 4.782] | 1.405 [1.285, 1.534] |
| 2 | 65,537 | separated groups | 31.623 | 18.955 | 36.764 | 1.718 [1.656, 1.785] | 1.989 [1.793, 2.211] |

Every frozen-hybrid/current-GPU interval is above one in both surveys. Ratios
range from 1.502 to 4.510 across the measured conditions, supporting improvement
over the prior public GPU-grid/CPU-union root path on this host.

Compared with CPU KD-tree/BFS, 65,537-point lattices measure 7.380 and 7.951
times; chains measure 1.243 and 1.405, and separated groups 2.009 and 1.989.
All corresponding intervals are above one. Lattices also favor GPU at 4,096
and 16,384 points. This supports actual GPU benefit for these workloads,
including transfers, rather than merely moving work between processors.

CPU remains substantially faster at 257 points (CPU/GPU ratios 0.020–0.061).
At 4,096 points it also wins on chains/groups; at 16,384 it wins on chains, and
the group comparison changes from 0.724 [0.706, 0.742] to 1.074 [0.941, 1.229].
These adverse/uncertain CPU comparisons are retained. Existing automatic
placement thresholds are unchanged, and these measurements do not establish
a universal new GPU crossover or full labeled-cloud-extractor speedup.

The results use synthetic fixtures and one native backend/device class. There
is no real-sensor, browser, cross-adapter or registration performance claim.
The full triple summary is
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-cluster-union-measurements.json`.
