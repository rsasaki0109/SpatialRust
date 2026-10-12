# Explicit GPU ICP correspondences

Date: 2026-10-12. Named slice: reusable nearest-point target and ICP integration.

## Execution contract

`KdTree::layout()` borrows original XYZ, original-index leaf order and neutral
branch/leaf nodes without copying and without GPU dependencies in search/core.
`GpuKdTree::upload(&runtime, &tree)` validates finite targets and device limits,
then explicitly packs/uploads 16 bytes per point and 32 per tree node. Target
partitioning remains CPU; the owned device target can outlive the CPU tree and
is reused across all query batches. Empty targets allocate no device buffers.

`nearest_one_within_download(&runtime, &queries, squared_gate)` explicitly
uploads 16 bytes per query and synchronously downloads 8 bytes per query.
Uniform controls are excluded from data-transfer accounting. Nonfinite queries
and negative/NaN gates produce None; positive infinity is unbounded. Empty
queries/targets and invalid gates copy no query/result data. Device identity and
buffer/dispatch limits are checked even for these fast paths.

The device traverses the same balanced KD-tree, with a private 32-level deferred
far-child stack per query and 64 invocations per workgroup. Far-child pruning is
evaluated after the near subtree, with inclusive gate comparison before any
match and strict incumbent comparison afterwards. Leaf iteration retains the
CPU first-encountered tie policy rather than imposing minimum original index.
The result preserves query input order. Four storage bindings and one uniform
binding are required. Storage size, host capacity, u32 indices, dispatch size,
workgroup dimensions and binding/uniform limits reject before GPU allocation.
The 32-level stack is bounded by u32 point counts and the CPU tree's balanced
partitioning; caller-provided arbitrary trees are not accepted.

The optional `register-icp-gpu` feature exposes
`IcpRegistration::align_with_gpu_correspondences`. Target upload occurs once per
alignment. Each update, post-update fitness pass and final fitness evaluation
uses the reused GPU target. The shared ICP loop retains CPU transform
application, source-order pairing, trimming, Kabsch, pose composition, ordered
f64 fitness accumulation and convergence. Ordinary CPU alignment still visits
matches directly without allocating a new batch result vector. The explicit
GPU result includes the ordinary trace, transfer receipt and `2 * iterations +
1` completed query batches. GPU errors propagate; there is no automatic CPU
fallback and no automatic placement/crossover change.

This completes ICP correspondence computation, not GICP correspondences or GPU
pose solving. GICP still has its existing optional GPU covariance path and CPU
matching. Feature neighborhoods and FPFH/ISS/MLS device computation remain next
dependencies. Native synchronous execution is verified. WASM compilation alone
does not establish browser execution; GPU f32 rounding/underflow may differ
across backends near gate or tie boundaries. Exact parity below describes the
selected fixtures on the tested native backend, not every floating-point input.

## Correctness and integration

- 68,420 ordered nearest results match CPU KD-tree traversal exactly, including
  coincident/tied/reversed targets, sizes 1/16/17/63/64/65/255/256/257/4,097,
  inclusive zero and one-ULP gate boundaries, infinity and nonfinite queries.
  An independent exhaustive point scan also confirms minimum gated distances.
- A 65,537-point target survives CPU-tree drop and serves four concurrent
  callers twice each (524,296 exact results), plus subsequent partial/empty
  reuse. Output scratch is exclusively leased until submission; target buffers
  and host results never enter the scratch pool. Same-queue submission order
  protects reused buffers after submission.
- Empty/invalid inputs, nonfinite target rejection, finite-coordinate distance
  overflow, wrong-device ownership and synthetic arithmetic/device/dispatch
  boundaries are covered. Payload receipts are exact.
- ICP GPU/CPU pose, fitness, stop reason and full trace match for 17/65/257/4,097
  points with translations and optional trimming, a rotated nonidentity initial
  guess, disabled convergence thresholds and iteration exhaustion. Invalid
  configurations and insufficient gated/trimmed correspondences retain CPU
  errors. Native GPU integration tests are explicitly ignored by default and
  run locally with `--include-ignored`.

Final validation: search 55 tests passed (5 ignored), all GPU features 189 tests
passed (21 ignored), registration 35 unit tests passed (1 ignored), three
explicit native GPU ICP integration tests passed, and one explicit facade test
passed. Clippy on GPU/search/registration all features and targets passed with
warnings denied. Minimal GPU and CPU ICP builds, GPU image/AoSoA WASM and
GPU-ICP WASM builds, formatting and diff checks passed. The early blanket
`--include-ignored` registration command also selected a pre-existing long
FPFH benchmark; it was stopped and replaced by the completed standard library
suite plus explicit new GPU ICP tests. No measurement from that interrupted
command is used here.

## Completed-call measurements

Native wgpu Vulkan discrete-class adapter, release build, CPU baseline one
worker. No local adapter model is recorded. Queries and registration use the
production public APIs; timing ends after readback and queue completion. No
other local builds/tests run during measurements. One CPU reference execution
and three checked GPU warmups precede 20 alternating-order pairs per condition,
with two isolated surveys (32 condition records). Preparation, output checks
and result drop are outside query timings. The retained-target query comparison
excludes one-time target construction/upload; the ICP comparison includes CPU
KD-tree construction, target upload, all query packing/uploads/readbacks, CPU
solver, transforms and full diagnostics. No GPU-only pass-time claim is made.

Nearest fixtures have target/query counts equal: a regular lattice, reversed
half-unit line and a repeated 7x3 planar pattern. Queries offset x by 1/8 and y
by -1/16 with squared gate 1/4. Repeated targets are an adversarial CPU-pruning
case; their gains must not be generalized to ordinary sensor clouds. ICP uses a
half-unit lattice, dyadic translation, distance gate 1/4 and exactly three
updates with convergence disabled, keeping both CPU/GPU iteration counts equal.

Tables report CPU/GPU completion medians in milliseconds and the geometric
mean paired CPU/GPU ratio. Ratios above one favor GPU. Brackets are separate
95% paired log-ratio bootstrap intervals (10,000 resamples; deterministic seed
20261012 plus record index). Input distribution, point count, CPU baseline,
target reuse and transfer boundary all matter; these are synthetic fixtures,
not a universal crossover or a throughput claim for an entire application.

### Reused-target nearest queries

| Survey | Points | Distribution | CPU ms | GPU ms | CPU/GPU [95% interval] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 257 | lattice | 0.039 | 1.171 | 0.033 [0.032, 0.034] |
| 1 | 257 | reversed line | 0.039 | 1.347 | 0.029 [0.028, 0.030] |
| 1 | 257 | repeated plane | 0.078 | 1.328 | 0.059 [0.052, 0.065] |
| 1 | 4,097 | lattice | 0.794 | 1.148 | 0.701 [0.685, 0.717] |
| 1 | 4,097 | reversed line | 0.533 | 1.100 | 0.481 [0.463, 0.499] |
| 1 | 4,097 | repeated plane | 4.529 | 1.669 | 2.816 [2.639, 2.995] |
| 1 | 16,384 | lattice | 6.605 | 2.317 | 2.732 [2.467, 3.001] |
| 1 | 16,384 | reversed line | 3.950 | 1.933 | 1.943 [1.772, 2.117] |
| 1 | 16,384 | repeated plane | 115.337 | 4.437 | 20.278 [16.260, 24.975] |
| 1 | 65,537 | lattice | 23.271 | 6.158 | 3.511 [3.122, 3.909] |
| 1 | 65,537 | reversed line | 20.168 | 4.616 | 4.056 [3.778, 4.327] |
| 1 | 65,537 | repeated plane | 2259.988 | 28.603 | 44.926 [30.222, 64.997] |
| 2 | 257 | lattice | 0.047 | 1.893 | 0.027 [0.025, 0.030] |
| 2 | 257 | reversed line | 0.037 | 1.777 | 0.022 [0.021, 0.023] |
| 2 | 257 | repeated plane | 0.095 | 1.652 | 0.056 [0.053, 0.059] |
| 2 | 4,097 | lattice | 1.037 | 1.884 | 0.572 [0.536, 0.616] |
| 2 | 4,097 | reversed line | 1.163 | 2.274 | 0.438 [0.404, 0.474] |
| 2 | 4,097 | repeated plane | 5.245 | 1.988 | 2.860 [2.632, 3.147] |
| 2 | 16,384 | lattice | 5.280 | 1.931 | 2.745 [2.619, 2.875] |
| 2 | 16,384 | reversed line | 3.206 | 1.884 | 1.695 [1.611, 1.793] |
| 2 | 16,384 | repeated plane | 94.688 | 4.241 | 23.161 [21.562, 24.989] |
| 2 | 65,537 | lattice | 28.106 | 4.414 | 6.708 [6.154, 7.388] |
| 2 | 65,537 | reversed line | 17.414 | 3.415 | 5.138 [4.635, 5.699] |
| 2 | 65,537 | repeated plane | 2677.610 | 46.044 | 40.366 [28.126, 57.306] |

### Full three-update ICP

| Survey | Points | CPU ms | GPU ms | CPU/GPU [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| 1 | 257 | 0.300 | 12.840 | 0.026 [0.024, 0.028] |
| 1 | 4,097 | 7.216 | 13.904 | 0.558 [0.506, 0.621] |
| 1 | 16,384 | 32.441 | 22.416 | 1.546 [1.444, 1.653] |
| 1 | 65,537 | 198.355 | 79.610 | 2.599 [2.412, 2.799] |
| 2 | 257 | 0.351 | 10.451 | 0.034 [0.032, 0.036] |
| 2 | 4,097 | 14.471 | 22.468 | 0.523 [0.439, 0.614] |
| 2 | 16,384 | 53.154 | 27.622 | 1.949 [1.784, 2.142] |
| 2 | 65,537 | 237.259 | 105.149 | 2.140 [1.950, 2.354] |

Small 257-point batches and ICP calls clearly favor CPU. Lattice/line queries at
4,097 points and full ICP at that size also favor CPU in both surveys. The
16,384/65,537-point lattice and line query cases, and the larger full ICP cases,
show transfer-inclusive gains in both surveys. Repeated-plane queries show
larger but distribution-specific gains. CPU timing and adapter state vary
between surveys, so all adverse cases and separate intervals are retained.

Reproduce each survey separately, without simultaneous builds/tests:

```powershell
cargo test -p spatialrust-gpu --features gpu-wgpu --release --test nearest_benchmark -- --ignored --nocapture
cargo test -p spatialrust-registration --features register-icp-gpu --release --test gpu_icp_benchmark -- --ignored --nocapture
```

The benchmark sources print backend and device class only. Ignored local raw
logs/statistical records stay under the local directory
`C:/Users/rsasa/Workspace/SpatialRust/target/automation`; published tables above
contain no adapter model, private sensor inputs or customer data.
