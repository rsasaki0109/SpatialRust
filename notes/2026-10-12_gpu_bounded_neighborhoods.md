# Bounded resident GPU neighborhoods

Date: 2026-10-12. Named slice: reusable query uploads and resident k-NN/radius
results, building on the GPU KD-tree from PR #164 and preserving PR #165 APIs.

## Data and execution boundary

`GpuQueryBatch::upload(runtime, xyz)` validates finite XYZ, u32 counts, device
buffer/dispatch limits and host capacity, then explicitly uploads padded
16-byte query records. Empty batches allocate no query buffer. The owned,
immutable batch can outlive its host input and be reused across target trees
and passes on the same runtime. The target's CPU-built KD-tree upload remains
explicit and separately accounted, as before.

`GpuKdTree::nearest_k_resident` supports k=0..=64. One GPU invocation traverses
the balanced target tree with a 32-level deferred far-child stack, reproduces
the CPU accumulator's first-worst-slot replacement and strict tie rejection,
then stably sorts the retained slots by distance. This preserves the CPU
`nearest_k` membership AND ordered ties on the tested fixtures. Self and
duplicate points are included; feature consumers may request k+1 and apply
their own self-exclusion policy. Finite XYZ can still produce infinite squared
distances, which retain the CPU selection/pruning behavior.

`radius_neighbors_resident(runtime, queries, radius, capacity)` uses a radius,
not a squared gate. It preserves CPU near-first leaf traversal order, counts
ALL radius matches even after the row is full, and stores only the first
capacity matches. Capacity=0 performs count-only search. Negative/NaN radii
produce empty rows. Positive infinity and overflowing positive radius squares
are unbounded, as on CPU. A truncated row is a traversal prefix, not a closest-
point selection. Radius capacity is not restricted to 64, but buffers, shader
word offsets and decoding capacities must fit validated limits.

Search submits device work and returns owned `GpuNeighborhoods` without a
payload upload, readback or completion wait. Uniform controls are excluded from
payload receipts. Target/query buffers are reused; result buffers are separately
owned and never returned to scratch pools. Each row contains `2 + 2*capacity`
u32 words: stored count, total count, then original target index / f32 squared-
distance bit pairs. k-NN rows are sorted; radius rows retain traversal order.
Unused slots are zero, including fast-path rows. Empty results expose an unused
8-byte physical buffer, with zero logical rows/readback bytes.

`GpuNeighborhoods::buffer` lets subsequent GPU kernels consume that layout on
the same runtime. Consumers must respect stored counts and handle radius total
counts larger than capacity before relying on complete neighborhoods. Inputs
and outputs are immutable application data; custom kernels must not overwrite
them. Public search/download methods validate runtime ownership even for empty
or invalid-radius paths. The shader needs four storage bindings, one uniform
and 64 invocations per workgroup. Row size, total storage, u32 shader offsets,
CPU row/neighbor allocation bounds and dispatch are validated before allocation.

`download` explicitly waits for completion and copies the FULL padded row
buffer: `queries * (8 + 8*capacity)` bytes. It returns exact counts and stored
rows; malformed GPU counts reject before host neighbor allocation/indexing.
`complete_rows` and `into_complete_rows` reject the first overflowed radius row.
`truncated_rows` is the explicitly named opt-in access to bounded prefixes.
`is_complete` and `total_counts` let callers choose a larger capacity or bounded
query batches rather than silently lose neighbors.

Target upload is `16*T + 32*tree_nodes` H2D bytes. Query upload is `16*Q` H2D.
Resident search has zero payload transfer bytes. Download is `Q*(8+8*C)` D2H,
including padding. No global CPU nearest search or automatic CPU fallback is
used by these new GPU methods. Existing nearest-one ICP/GICP and feature APIs
are unchanged. FPFH/ISS/MLS remain CPU algorithms; their device compute and
feature-specific neighborhood integration are the next slices, not completed
by merely exposing this resident data model.

GPU f32 rounding/underflow may differ across backends near gate/tie boundaries;
exact parity below is for selected native fixtures, not every floating-point
input. Native synchronous download/consumer execution is verified. WASM build
success alone does not establish browser execution.

## Correctness

- 467,364 selected nearest neighbors match exact CPU membership/ordered ties
  and independent exhaustive sorted distances, across k=0/1/2/8/16/32/64 and
  target sizes 1/16/17/63/64/65/255/257/513. Fixtures include regular lattice,
  reversed line, coincident points and repeated symmetric coordinates.
- 199,846 stored radius entries match CPU traversal prefixes, with exact total
  counts also checked by exhaustive membership scans. Zero/one-ULP/negative/
  NaN/infinite radii, capacities 0/3/64/full-target and missing matches cover
  complete and overflow paths.
- Four concurrent callers twice each reuse a 4,097-query batch and target;
  524,416 selected neighbors remain exact. Earlier outputs survive later
  k-NN/radius calls and CPU tree, query and target drops.
- Two 65,537-query partial dispatches preserve all original self indices
  (131,074 neighbors). A raw readback confirms unused slots remain zero when
  target count is smaller than k. Empty/invalid/finite-overflow data and wrong
  target/query/output runtime ownership, unsupported counts/capacities and
  malformed result metadata reject as specified.
- A downstream native GPU test consumes radius counts and original indices
  directly from resident storage, before any neighborhood readback; it detects
  257 total / 3 stored neighbors per row. The queue retains resources even
  after the owning target/query/result handles are dropped. One native facade
  test checks reuse, complete/overflow access and padded transfer receipts.

Final validation: all GPU features passed 196 tests (23 ignored), registration
passed 35 unit tests (1 ignored) plus all six explicit GPU ICP/GICP regression
tests, and the explicit native facade test passed. GPU all-feature/all-target
clippy passed with warnings denied. Minimal GPU, GPU image/AoSoA WASM,
formatting and diff checks passed. Source comparison against parent 8b1226f
confirms the nearest-one shader/upload/search and search/feature/registration
algorithm files are unchanged; only the new neighborhood API/cache is added.

## Measurement method

Native wgpu Vulkan discrete-class adapter, release build, CPU baseline one
worker. No adapter model is recorded. Target CPU construction and target GPU
upload occur once before timing for BOTH reused-index comparisons. Query XYZ
are supplied by the caller and offset x by 1/16, y by -1/32. Fixtures are a
half-unit lattice and a reversed half-unit line, with equal target/query counts.
Nearest-k uses k=16; radius uses 0.75 and capacity=32 (all selected fixture rows
fit). Two isolated surveys each contain 20 cyclic-order CPU / completed GPU /
resident GPU triples for 16 conditions. A CPU reference and three checked GPU
warmups of each path precede timing. No other local builds/tests run during
measurements; output checks and result drop are outside measured intervals.

Completed GPU time includes query packing/upload, owned result allocation,
search, queue completion, padded download, metadata/overflow checking and CPU
row decoding. CPU time includes all scalar KD-tree queries and returned row
vectors. Resident time separately includes result allocation, submission and
queue completion with target/queries already uploaded, and ends before CPU
readback/decoding. Resident results are downloaded/checked AFTER their timer.
Resident columns therefore describe a different transfer boundary and are NOT
used to claim a CPU speedup or an end-to-end feature speedup. No GPU-only pass
timestamps are claimed. Target uploads and CPU construction are excluded from
these reused-index query timings and must be added for cold single-use calls.

Tables give medians in ms and the geometric mean paired CPU/completed-GPU ratio;
above one favors the completed GPU API. Brackets are separate 95% paired log-
ratio bootstrap intervals with 10,000 resamples and deterministic seed 20261012
plus record index. Geometry, query count, k/capacity, input reuse, CPU worker
count and transfer boundary matter. Synthetic fixtures do not establish a
universal crossover or performance on an entire feature/registration pipeline.

| Survey | Queries | Geometry | Query | CPU ms | Complete GPU ms | Resident completion ms | CPU/complete GPU [95% interval] |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 257 | lattice | k=16 | 0.192 | 1.129 | 0.218 | 0.167 [0.157, 0.176] |
| 1 | 257 | lattice | radius=0.75, C=32 | 0.103 | 1.096 | 0.165 | 0.096 [0.091, 0.102] |
| 1 | 257 | reversed line | k=16 | 0.146 | 1.167 | 0.200 | 0.128 [0.124, 0.133] |
| 1 | 257 | reversed line | radius=0.75, C=32 | 0.044 | 1.162 | 0.153 | 0.038 [0.037, 0.039] |
| 1 | 4,097 | lattice | k=16 | 4.731 | 1.833 | 0.563 | 2.598 [2.525, 2.674] |
| 1 | 4,097 | lattice | radius=0.75, C=32 | 2.721 | 1.577 | 0.354 | 1.739 [1.655, 1.816] |
| 1 | 4,097 | reversed line | k=16 | 2.421 | 1.538 | 0.339 | 1.548 [1.479, 1.613] |
| 1 | 4,097 | reversed line | radius=0.75, C=32 | 0.711 | 1.251 | 0.158 | 0.554 [0.531, 0.571] |
| 1 | 16,384 | lattice | k=16 | 25.935 | 4.124 | 1.653 | 6.236 [5.894, 6.574] |
| 1 | 16,384 | lattice | radius=0.75, C=32 | 17.572 | 5.656 | 1.769 | 3.162 [3.024, 3.303] |
| 1 | 16,384 | reversed line | k=16 | 9.875 | 5.373 | 2.449 | 1.857 [1.794, 1.920] |
| 1 | 16,384 | reversed line | radius=0.75, C=32 | 3.309 | 6.393 | 2.796 | 0.545 [0.520, 0.576] |
| 1 | 65,537 | lattice | k=16 | 125.638 | 34.424 | 8.370 | 4.124 [3.615, 4.693] |
| 1 | 65,537 | lattice | radius=0.75, C=32 | 75.407 | 31.469 | 12.887 | 2.574 [2.262, 2.925] |
| 1 | 65,537 | reversed line | k=16 | 43.134 | 28.232 | 14.003 | 1.316 [1.183, 1.449] |
| 1 | 65,537 | reversed line | radius=0.75, C=32 | 13.534 | 18.782 | 6.446 | 0.771 [0.733, 0.815] |
| 2 | 257 | lattice | k=16 | 0.220 | 1.087 | 0.214 | 0.196 [0.191, 0.200] |
| 2 | 257 | lattice | radius=0.75, C=32 | 0.102 | 1.047 | 0.146 | 0.098 [0.096, 0.099] |
| 2 | 257 | reversed line | k=16 | 0.153 | 1.162 | 0.203 | 0.130 [0.126, 0.135] |
| 2 | 257 | reversed line | radius=0.75, C=32 | 0.044 | 1.063 | 0.139 | 0.042 [0.041, 0.043] |
| 2 | 4,097 | lattice | k=16 | 4.537 | 1.659 | 0.481 | 2.740 [2.702, 2.780] |
| 2 | 4,097 | lattice | radius=0.75, C=32 | 2.704 | 1.402 | 0.279 | 1.898 [1.837, 1.957] |
| 2 | 4,097 | reversed line | k=16 | 2.367 | 1.387 | 0.259 | 1.648 [1.555, 1.741] |
| 2 | 4,097 | reversed line | radius=0.75, C=32 | 0.714 | 1.240 | 0.155 | 0.575 [0.565, 0.583] |
| 2 | 16,384 | lattice | k=16 | 26.155 | 4.102 | 1.648 | 6.409 [6.080, 6.760] |
| 2 | 16,384 | lattice | radius=0.75, C=32 | 16.635 | 5.458 | 1.775 | 2.848 [2.649, 3.037] |
| 2 | 16,384 | reversed line | k=16 | 9.965 | 5.896 | 2.597 | 1.418 [1.259, 1.594] |
| 2 | 16,384 | reversed line | radius=0.75, C=32 | 3.331 | 8.049 | 3.498 | 0.425 [0.412, 0.439] |
| 2 | 65,537 | lattice | k=16 | 144.535 | 23.909 | 8.979 | 5.041 [4.044, 6.106] |
| 2 | 65,537 | lattice | radius=0.75, C=32 | 101.913 | 34.310 | 14.801 | 3.046 [2.889, 3.209] |
| 2 | 65,537 | reversed line | k=16 | 58.495 | 42.144 | 15.069 | 1.449 [1.265, 1.666] |
| 2 | 65,537 | reversed line | radius=0.75, C=32 | 22.245 | 25.248 | 9.141 | 0.829 [0.749, 0.908] |

At 65,537 lattice queries, completed k-NN improves by 4.124x / 5.041x and radius
by 2.574x / 3.046x across the two surveys. Both retain transfer-inclusive gains.
257-query cases favor CPU. Reversed-line radius queries favor CPU at every size
in both surveys, even though the resident-only boundary is cheaper. Larger
line k-NN cases have smaller and more variable gains than the lattice. All
adverse cases and separate intervals are retained. These differences justify
explicit GPU APIs without changing any automatic feature placement thresholds.

Reproduce each survey alone:

```powershell
cargo test -p spatialrust-gpu --features gpu-wgpu --release --test neighborhood_benchmark -- --ignored --nocapture
```

Local raw logs/statistical records stay under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation`. Benchmark output prints
backend, device class and CPU worker count only; published tables contain no
adapter model or private sensor/customer data.
