# Explicit GPU tree point and shared query gathering

Date: 2026-10-14. Parent: `b2ae11896fdf843b0ced0cb77d5c3e838d011e7e` (merged PR 198).

## API and execution boundary

`GpuKdTree::upload_with_gpu_gather` explicitly sends the CPU tree's borrowed
original-order SoA axes and original-index permutation to GPU, then gathers
the existing 16-byte XYZ/index records in leaf traversal order. Integer loads
and stores preserve signed zero, subnormals and all finite coordinate bits.
CPU partitioning, finite validation and 32-byte node staging remain.

`GpuKdTree::upload_and_queries_with_gpu_gather` also produces padded 16-byte
queries for every point in original input order, in that same pass. The shared
upload reduces point/query host payload from 32*N to 16*N bytes. Node payload
remains 32*M bytes. The target reports 16*N + 32*M host bytes and 16*N device
output bytes; generated queries report zero host bytes and 16*N device output
bytes. Neither constructor downloads data or waits for queue completion.
Uniform control bytes are excluded, as in existing transfer receipts.

The two transient device inputs occupy 12*N bytes for coordinates and 4*N
for indices. Owned outputs occupy 16*N for tree points and optionally another
16*N for queries, plus unchanged nodes. This removes host point/query packing
vectors (16*N or 32*N bytes respectively), with additional temporary GPU memory
of 16*N bytes. Direct mapped SoA writes require no host concatenation vector.
Backend upload staging and allocator overhead are not included in these logical
buffer sizes. The submitted queue retains temporary resources, and subsequent
searches follow queue order. Returned trees and queries own their outputs;
the CPU tree and its input storage may be dropped immediately.

Finite scans use the existing bounded native policy: at least 524,288 points,
useful 131,072-point grains and at most four available CPU workers. They allocate
only worker-sized metadata. Small/one-worker/WASM scans stay serial. Every
coordinate is checked before device allocation. Existing buffer/compute limits
are checked before scans; the paired API additionally checks the complete
query dispatch size. A bounded grid-stride gather supports larger tree-only
uploads beyond one dispatch wave. Empty inputs allocate no buffers or pipeline.
Tree-only and paired compute pipelines initialize independently and lazily on
each runtime, including concurrent first callers.

Default tree/query upload bodies, point/node layouts, CPU staging and search
kernels are preserved. Feature and registration placement is unchanged. The
new methods expose an explicit transfer/memory tradeoff; no automatic size
selection or complete feature/registration speedup is claimed. No dependency,
unsafe block or core change is introduced.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/nearest.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_gpu_gather.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/tree_point_gather.wgsl`

## Correctness and integration

Thirty nearest-family native release tests passed. New tree-only probes compare
74,720,236 point/node words against independent coordinate/index/node oracles
and the default upload. Cases cover small/partial/worker-threshold inputs,
signed zeros, signed subnormals and seeded arbitrary finite words through
1,048,577 points. A 4,194,241-point case exercises a second iteration beyond
the capped 65,535*64 grid. Paired probes verify 26,221,648 original-order query
words against independent bit/padding expectations and the existing SoA upload,
and compare both tree buffers against the default path.

Tree-only neighborhood probes compare 3,810 result words; shared-query probes
compare 195,834 words. k=1/8/64 and radius=0/0.125/MAX include ties, traversal
order, total counts, truncation/overflow handling and exact transfer receipts.
CPU trees are dropped before searches and GPU inputs before result readback.
Foreign runtime/query identities reject. Nine large NaN/infinity inputs match
default errors for both APIs before pipeline creation, inside a clean device
validation scope. Three concurrent cold paired uploads retain independent
point/query buffers and initialize only the paired pipeline variant.

Scan-only CPU tests compare existing packing errors and success for empty,
small and threshold-adjacent inputs with injected 0/1/2/3/4/12 worker budgets,
including per-axis nonfinite final elements and mismatched lengths. Existing
CPU/GPU staging, neighborhood, feature and registration regressions remain.

All 251 ordinary GPU/image release tests passed (65 explicit tests/benchmarks
remain opt-in). Related native regression groups cover three feature-input,
three FPFH, four ISS, four MLS and six registration geometry/SoA tests. Ten public facade tests passed across
FPFH/ISS/MLS, neighborhood search, ICP/trimmed ICP, GICP and point-to-plane
equations/convergence. All-feature/all-target GPU Clippy passed with warnings
denied; CPU-only, WASM, formatting and diff checks passed. A source receipt
checks default tree/query upload bodies, CPU point preparation, point/node
layouts and nearest search against the parent. Seven consumer/query sources
and all 95 preexisting shaders remain unchanged; 112 measured source/shader/
executable hashes are retained locally.

Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --features gpu-image --release --lib kernels::nearest -- --include-ignored --skip benchmark --skip survey_bounded_host_packing --skip survey_bounded_host_tree_packing --skip completed_cpu_and_gpu_tree_uploads --nocapture --test-threads=1
cargo test -p spatialrust-gpu --features gpu-image --release --lib -- --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --no-default-features
cargo check -p spatialrust --no-default-features --features register-fpfh-gpu,feature-iss-gpu,filter-mls-gpu,register-icp-gpu,register-gicp-gpu,register-icp-point-to-plane-gpu --target wasm32-unknown-unknown
```

Local logs and measured source/executable hashes:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/tree-gpu-*`.

## Completed-upload measurement

The ignored `completed_cpu_and_gpu_tree_uploads` test compares four routes:
default tree upload, explicit tree-only GPU gather, default tree plus existing
SoA query upload, and shared GPU tree/query gather. Eight size/profile cases
rotate the four routes over three warmup and 30 recorded rounds. Fixtures,
CPU tree construction and runtime setup are outside timing. Each timed call
includes device/finite validation, node staging, applicable point/query packing,
all device buffer creation/submission, `wait_idle`, receipt capture and output
drop. Assertions follow timing. Search and complete algorithms are outside the
measurement boundary. Warm pipeline results exclude first compilation latency.

Profiles: 0 bounded varied finite coordinates, 1 alternating signed zeros,
2 signed subnormals. Two sequential isolated surveys finish before analysis;
the controller requires a quiet window, monitors build/test/benchmark overlap,
discards a contaminated attempt and stops only its own child if needed.
Raw paired sample logs, including adverse cases, remain in the local receipts.

Ratios are paired geometric means of old/new duration, with values above one
meaning shorter new calls. Intervals use 10,000 paired log-ratio bootstrap
resamples, seed `2026101425 + 100*survey + 10*case + field_index` (JSON record
field index), sorted entries 249/9749. Every row retains 30 samples. Any zero
sample suppresses the ratio and retains medians/zero counts.


Final environment: native release, `rustc 1.98.1 (48a229cea 2026-09-01)`,
Vulkan discrete-GPU backend; 12 available CPU workers, at most four used for
large finite scans. The gather uses 64 lanes, one uniform and at most four
storage bindings; actual successful tree-only coverage reaches 4,194,241 points.
Both accepted controller attempts were number 1, with no overlap and empty
stderr; survey durations were 11.22 and 11.48 seconds. Sources/executable
matched the same 112-file SHA-256 receipt before and after both surveys.

For 1,048,577 bounded varied finite points, paired tree/query uploads:

- survey 1: 17.499750 to 15.249950 ms; ratio 1.179271 [1.089093, 1.283314].
- survey 2: 17.511950 to 15.266850 ms; ratio 1.185964 [1.084424, 1.293262].

Both median reductions are about 13%, with paired intervals above one. This
boundary has repeatable gains for the measured varied-coordinate cases at
262,145, 524,289 and 1,048,577 points; it does not establish a universal input
threshold. Signed-zero/subnormal shared cases also improve, but synthetic
profiles are not evidence for typical application-wide acceleration. Search,
feature computation and registration solving remain outside these timings.

Tree-only gathering is slower for every measured bounded varied-coordinate
size in both surveys, including 1,048,577 points (ratios 0.918066 and 0.906606,
both intervals below one). Its benefit is removing host point staging, at the
documented temporary GPU memory and latency cost. Paired uploads are slower
at 1 and 4,097 points in both surveys. At 65,537 points the paired result is
slower in survey 1 and inconclusive in survey 2. No default path was switched.
All 17 comparisons whose complete interval is below one and all inconclusive
comparisons are retained below. There were no zero timer samples.

An exploratory earlier tree-only candidate uploaded three separate axis
buffers and required five storage bindings. Its local pilot motivated the
final contiguous SoA buffer. That candidate's performance is excluded from
the final table, which uses only the verified final source/executable.

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | tree_gpu | 0.018000 | 0.118300 | 0.140739 [0.128342, 0.153571] |
| 1 | 1 | 0 | pair_gpu | 0.020500 | 0.120350 | 0.159822 [0.147859, 0.172011] |
| 1 | 4097 | 0 | tree_gpu | 0.037800 | 0.146800 | 0.242003 [0.226031, 0.259404] |
| 1 | 4097 | 0 | pair_gpu | 0.053600 | 0.156350 | 0.334102 [0.317525, 0.351125] |
| 1 | 65537 | 0 | tree_gpu | 0.696100 | 0.846300 | 0.817577 [0.769692, 0.870087] |
| 1 | 65537 | 0 | pair_gpu | 1.170900 | 1.281900 | 0.872377 [0.767231, 0.994632] |
| 1 | 262145 | 0 | tree_gpu | 3.415750 | 4.074550 | 0.813000 [0.731489, 0.898059] |
| 1 | 262145 | 0 | pair_gpu | 5.663750 | 5.257650 | 1.130724 [1.074722, 1.192563] |
| 1 | 524289 | 0 | tree_gpu | 5.904800 | 7.337500 | 0.868090 [0.778970, 0.972867] |
| 1 | 524289 | 0 | pair_gpu | 10.139950 | 8.749700 | 1.131714 [1.065301, 1.201667] |
| 1 | 1048577 | 0 | tree_gpu | 11.150900 | 12.237050 | 0.918066 [0.846862, 0.992888] |
| 1 | 1048577 | 0 | pair_gpu | 17.499750 | 15.249950 | 1.179271 [1.089093, 1.283314] |
| 1 | 1048577 | 1 | tree_gpu | 12.041500 | 11.925550 | 0.961880 [0.887032, 1.037762] |
| 1 | 1048577 | 1 | pair_gpu | 18.173850 | 15.588950 | 1.201965 [1.108686, 1.305460] |
| 1 | 1048577 | 2 | tree_gpu | 16.868800 | 13.700250 | 1.235033 [1.128405, 1.355973] |
| 1 | 1048577 | 2 | pair_gpu | 23.985950 | 16.950900 | 1.420624 [1.304914, 1.534803] |
| 2 | 1 | 0 | tree_gpu | 0.018050 | 0.115850 | 0.136486 [0.124503, 0.148673] |
| 2 | 1 | 0 | pair_gpu | 0.020500 | 0.118800 | 0.157340 [0.145166, 0.169671] |
| 2 | 4097 | 0 | tree_gpu | 0.038100 | 0.146450 | 0.244088 [0.228858, 0.260542] |
| 2 | 4097 | 0 | pair_gpu | 0.054100 | 0.155300 | 0.345220 [0.329178, 0.363126] |
| 2 | 65537 | 0 | tree_gpu | 0.699150 | 0.955850 | 0.750086 [0.691489, 0.811645] |
| 2 | 65537 | 0 | pair_gpu | 1.167350 | 1.309050 | 0.891566 [0.767820, 1.033193] |
| 2 | 262145 | 0 | tree_gpu | 3.450300 | 4.539950 | 0.784263 [0.719945, 0.855376] |
| 2 | 262145 | 0 | pair_gpu | 5.783000 | 5.436150 | 1.107684 [1.046865, 1.176046] |
| 2 | 524289 | 0 | tree_gpu | 5.996800 | 7.249100 | 0.847116 [0.760160, 0.950088] |
| 2 | 524289 | 0 | pair_gpu | 9.936850 | 8.967700 | 1.138599 [1.062250, 1.221816] |
| 2 | 1048577 | 0 | tree_gpu | 11.359250 | 12.239300 | 0.906606 [0.847508, 0.967607] |
| 2 | 1048577 | 0 | pair_gpu | 17.511950 | 15.266850 | 1.185964 [1.084424, 1.293262] |
| 2 | 1048577 | 1 | tree_gpu | 11.829700 | 11.764950 | 1.002734 [0.938091, 1.071280] |
| 2 | 1048577 | 1 | pair_gpu | 18.361150 | 15.208250 | 1.299978 [1.216918, 1.392514] |
| 2 | 1048577 | 2 | tree_gpu | 17.327550 | 13.799750 | 1.245081 [1.145724, 1.354746] |
| 2 | 1048577 | 2 | pair_gpu | 24.214200 | 16.733700 | 1.417500 [1.270541, 1.568413] |
