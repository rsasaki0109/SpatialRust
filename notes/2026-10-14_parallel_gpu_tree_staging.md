# Bounded native GPU KD-tree point staging

Date: 2026-10-14. Parent: `d6b75580626d649533497c1c0aae8a83dbebb8a2` (merged PR 197).

## Change and boundary

`GpuKdTree::upload` validates every original XYZ coordinate and packs points in
the CPU tree's leaf traversal order. Native inputs with at least 524,288 points
use useful 131,072-point grains, available CPU parallelism and at most four
scoped workers. Each worker validates its original-coordinate band, then fills
a disjoint output band using the tree's original-index permutation. Validation
covers all original coordinates independently of gather-band membership.

Small, one-worker and WASM inputs retain the parent's finite scans and map/
collection loops; small inputs skip the CPU-count query. The source comparison
checks that serial fallback exactly matches the frozen parent loops after
formatting/comment normalization. The private helper checks matching axis/order
lengths; original indices come from the immutable CPU KD-tree layout and retain
ordinary safe Rust bounds checks. No new public API, dependency or unsafe block
is introduced.

The existing 16*N-byte point staging capacity remains exact in every tested
worker path. Parallel output is initialized once and workers write directly;
there is no point-sized join vector. Node staging remains serial and 32*M bytes.
Only worker-sized handles and native thread resources are added. Initialization,
dispatch and join costs are included in measurements. Point packing now occurs
before host node packing. Large invalid inputs can allocate bounded host point
storage before all finite checks finish, while device buffer/dispatch limits
still precede packing and every worker succeeds before device-buffer creation.

Root/node records, original XYZ bits (including zero signs/subnormals), original
indices, leaf/tie traversal order, device-buffer creation order and labels,
owned target lifetime and transfer receipts are preserved. Upload remains
16*N + 32*M host-to-device bytes and zero device-to-host bytes. The nearest
module is compared to the parent with only the intended staging substitution
and private module registration allowed. Eight consumer/query sources and all
95 GPU shaders are unchanged.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/nearest.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_staging.rs`

## Correctness and integration

Six new release tests passed in 13.56 seconds. CPU comparisons cover
346,032,096 point words against frozen packing and independent coordinate-bit/
original-index expectations. Cases include empty/small/threshold/partial
inputs through 1,048,577 points, four finite profiles, reverse/rotated original
indices crossing worker bands, injected 0/1/2/3/6/12 CPU budgets and the actual
dispatcher. All returned vectors retain exact point capacity. Planner tests
include unbounded CPU budgets and usize maximum arithmetic.

There are 252 exact frozen nonfinite errors at original-coordinate worker-band
boundaries, plus private length mismatch and unreferenced-nonfinite checks.
Concurrent callers and actual CPU-tree leaf permutations remain exact. Native
GPU probes compare 68,157,840 point/node words in 40 size/profile cases,
including both sides of 524,288 points, signed zero, subnormal and large finite
bits. Root, empty allocation, receipts and ownership after CPU tree/storage
drop are checked. Nine large public nonfinite inputs match the frozen error
inside a clean device-validation scope.

Large 524,289-point target tests compare 108 k-NN/radius output rows, including
stable distance ties, stored traversal prefixes, complete total counts,
overflow rejection, zero/unbounded radii and foreign-runtime denial. Query
buffers are reused and CPU target storage is dropped before these searches.

Related native regressions: 37 successes (17 nearest/neighborhood/shared-query,
three previous feature-input comparisons, three FPFH, four ISS, four MLS and six
registration geometry/SoA input tests). Public facade: eight successes across
FPFH/ISS/MLS, neighborhood search, ICP/trimmed ICP, GICP and point-to-plane
equations/convergence. All-feature/all-target GPU Clippy passed with warnings
denied. GPU-disabled, WASM, formatting and diff checks passed. Frozen complete
upload, finite validation and point gathering match the parent after naming/
format normalization.

Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --features gpu-image --release --lib tree_staging -- --include-ignored --skip survey_bounded_host_tree_packing --skip completed_host_and_tree_uploads --nocapture --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --no-default-features
cargo check -p spatialrust --no-default-features --features register-fpfh-gpu,feature-iss-gpu,filter-mls-gpu,register-icp-gpu,register-gicp-gpu,register-icp-point-to-plane-gpu --target wasm32-unknown-unknown
```

Local logs/receipts: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/tree-staging-*`.

## Candidate rejection and final measurement

An earlier candidate used a manual serial push loop and began parallel work at
262,144 points. Two isolated measurements found serial overhead and inconclusive
parallel gains around that threshold. At 1,048,577 varied finite points, its
forced-one-worker ratios were 0.903938 [0.833096, 0.985563] and 0.881761
[0.829603, 0.937738]. It was rejected. The final implementation restores the
parent collector and starts parallel work at 524,288 points. Candidate sources,
executable, full comparisons and receipts are archived at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/tree-staging-candidate1/` and are excluded
from the final ratios below. Final measurements use the corrected source and
executable receipt, with both accepted controller attempts numbered 2.

Final environment: `rustc 1.98.1 (48a229cea 2026-09-01)`; native release; Vulkan discrete-GPU backend;
12 available CPU workers, at most four used. No local adapter model is recorded.
Two sequential accepted surveys ran without overlapping build/test/benchmark
processes and with empty stderr, lasting 21.91 and 20.11 seconds.
The controller checks a quiet window, monitors overlap and would discard a
whole attempt and stop only its own benchmark child if overlap appeared. Both
surveys finished before statistical analysis began.

For each of 11 size/profile cases, five routes rotate cyclically for three
warmup and 30 recorded rounds: frozen host validation/point gathering, current
forced-one-worker host preparation, current automatic host preparation, frozen
whole tree upload and current whole tree upload. Fixtures, CPU tree construction
and runtime setup are outside timing. Host timing includes validation,
allocation, gathering, observation and drop. Upload timing includes limits,
validation, point/node staging, both device-buffer creations, `wait_idle`,
receipt capture and target drop. Assertions follow timing. This covers the
explicit tree-input boundary; search, feature/registration kernels and complete
algorithm latency are outside the measured interval. Profiles: 0 bounded varied
finite coordinates, 1 alternating zero signs, 2 signed subnormals.

All 66 final route comparisons are retained. Ratios are paired geometric means
of old/new times; values above one mean shorter current calls. Confidence
intervals use 10,000 paired log-ratio bootstrap resamples with seed
`2026101421 + 100*survey + 10*case + field_index` (JSON record field index) and
sorted entries 249/9749. Every row has 30 samples. Comparisons containing any
zero timer sample receive no ratio estimate and retain medians/zero counts.
One-point host durations are near clock resolution, so no practical speed claim
is drawn from them, even when a nonzero ratio can be computed.

At 1,048,577 varied finite points:

- survey 1: 10.226200 to 7.460750 ms; paired ratio 1.366061 [1.299549, 1.434738] (host_new).
- survey 1: 15.263950 to 12.925250 ms; paired ratio 1.157607 [0.989507, 1.312627] (upload_new).
- survey 2: 10.499400 to 7.862100 ms; paired ratio 1.334534 [1.285835, 1.389209] (host_new).
- survey 2: 15.379800 to 13.592800 ms; paired ratio 1.081492 [0.952184, 1.191971] (upload_new).

The automatic host medians at this finite-coordinate case are about 25–27%
shorter, with both paired intervals above one. Corresponding whole-upload
intervals cross one in both surveys, so a repeatable complete-upload gain is
not established for that case. Synthetic subnormal-coordinate uploads have
intervals above one in both surveys; this does not establish an application-wide
gain for typical coordinate distributions.

Rows whose full interval is below one are retained explicitly:

No measured row has its entire interval below one; intervals crossing one remain in the table.

All other inconclusive and variable upload results remain in the table. The
measurements do not establish universal upload or whole-algorithm acceleration.

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | host_one | 0.000100 | 0.000100 | 1.137736 [1.000000, 1.294442] |
| 1 | 1 | 0 | host_new | 0.000100 | 0.000100 | N/A: zero samples old=0, new=2 |
| 1 | 1 | 0 | upload_new | 0.010100 | 0.009550 | 1.033316 [0.943667, 1.122119] |
| 1 | 4097 | 0 | host_one | 0.012950 | 0.012250 | 1.109806 [1.012086, 1.221901] |
| 1 | 4097 | 0 | host_new | 0.012950 | 0.012100 | 1.154142 [1.077101, 1.254539] |
| 1 | 4097 | 0 | upload_new | 0.070350 | 0.070450 | 0.903247 [0.683480, 1.070262] |
| 1 | 65537 | 0 | host_one | 0.391100 | 0.383000 | 1.016295 [0.997657, 1.035075] |
| 1 | 65537 | 0 | host_new | 0.391100 | 0.379400 | 1.032755 [1.008233, 1.058785] |
| 1 | 65537 | 0 | upload_new | 0.601700 | 0.595650 | 1.046413 [0.955102, 1.146616] |
| 1 | 131073 | 0 | host_one | 0.814950 | 0.800450 | 1.058532 [1.015743, 1.104972] |
| 1 | 131073 | 0 | host_new | 0.814950 | 0.786300 | 1.089559 [1.037694, 1.150663] |
| 1 | 131073 | 0 | upload_new | 1.392200 | 1.329400 | 0.990438 [0.823209, 1.120483] |
| 1 | 262145 | 0 | host_one | 2.223000 | 2.049000 | 1.096850 [1.040271, 1.157405] |
| 1 | 262145 | 0 | host_new | 2.223000 | 1.948050 | 1.116390 [1.052320, 1.185684] |
| 1 | 262145 | 0 | upload_new | 3.429950 | 3.458850 | 1.056921 [0.934982, 1.288143] |
| 1 | 524287 | 0 | host_one | 4.726650 | 4.759950 | 1.035109 [0.996565, 1.072491] |
| 1 | 524287 | 0 | host_new | 4.726650 | 4.799900 | 1.040046 [0.951570, 1.142190] |
| 1 | 524287 | 0 | upload_new | 7.169800 | 7.171300 | 1.047526 [0.964536, 1.166501] |
| 1 | 524288 | 0 | host_one | 4.772200 | 4.546550 | 1.016349 [0.941571, 1.088644] |
| 1 | 524288 | 0 | host_new | 4.772200 | 3.978150 | 1.236478 [1.174425, 1.304368] |
| 1 | 524288 | 0 | upload_new | 6.895200 | 6.547000 | 1.026205 [0.902417, 1.137706] |
| 1 | 524289 | 0 | host_one | 4.887700 | 4.591500 | 1.070954 [1.019481, 1.123866] |
| 1 | 524289 | 0 | host_new | 4.887700 | 3.998000 | 1.274759 [1.208197, 1.351320] |
| 1 | 524289 | 0 | upload_new | 7.305800 | 6.657400 | 1.068285 [0.947971, 1.158580] |
| 1 | 1048577 | 0 | host_one | 10.226200 | 10.244550 | 1.024863 [0.983588, 1.068455] |
| 1 | 1048577 | 0 | host_new | 10.226200 | 7.460750 | 1.366061 [1.299549, 1.434738] |
| 1 | 1048577 | 0 | upload_new | 15.263950 | 12.925250 | 1.157607 [0.989507, 1.312627] |
| 1 | 1048577 | 1 | host_one | 14.103950 | 12.233500 | 1.088787 [1.035025, 1.146509] |
| 1 | 1048577 | 1 | host_new | 14.103950 | 9.204350 | 1.446284 [1.360498, 1.531459] |
| 1 | 1048577 | 1 | upload_new | 23.965100 | 19.422850 | 1.108906 [0.949180, 1.261031] |
| 1 | 1048577 | 2 | host_one | 36.804900 | 30.360900 | 1.064808 [0.977380, 1.160145] |
| 1 | 1048577 | 2 | host_new | 36.804900 | 17.393650 | 1.977581 [1.790541, 2.174621] |
| 1 | 1048577 | 2 | upload_new | 48.665600 | 33.475550 | 1.386478 [1.259702, 1.508242] |
| 2 | 1 | 0 | host_one | 0.000100 | 0.000100 | 1.071773 [0.933033, 1.231144] |
| 2 | 1 | 0 | host_new | 0.000100 | 0.000100 | 1.096825 [0.911722, 1.319508] |
| 2 | 1 | 0 | upload_new | 0.012250 | 0.011650 | 1.089372 [1.001064, 1.199330] |
| 2 | 4097 | 0 | host_one | 0.013600 | 0.013100 | 1.045670 [1.035222, 1.059936] |
| 2 | 4097 | 0 | host_new | 0.013600 | 0.013000 | 1.052791 [1.027162, 1.075411] |
| 2 | 4097 | 0 | upload_new | 0.075400 | 0.076750 | 0.875137 [0.614647, 1.087227] |
| 2 | 65537 | 0 | host_one | 0.536300 | 0.528250 | 1.047600 [1.001028, 1.097416] |
| 2 | 65537 | 0 | host_new | 0.536300 | 0.515450 | 1.067797 [1.016223, 1.121874] |
| 2 | 65537 | 0 | upload_new | 0.774600 | 0.819850 | 1.003906 [0.929576, 1.083580] |
| 2 | 131073 | 0 | host_one | 1.066550 | 1.003550 | 1.085648 [1.042626, 1.128650] |
| 2 | 131073 | 0 | host_new | 1.066550 | 0.998000 | 1.087205 [1.024801, 1.151226] |
| 2 | 131073 | 0 | upload_new | 1.660850 | 1.618900 | 0.979070 [0.819308, 1.094061] |
| 2 | 262145 | 0 | host_one | 2.894700 | 2.831350 | 1.030809 [0.970676, 1.089163] |
| 2 | 262145 | 0 | host_new | 2.894700 | 2.771100 | 1.061495 [0.973420, 1.153825] |
| 2 | 262145 | 0 | upload_new | 4.355300 | 4.433050 | 1.067923 [0.942923, 1.280333] |
| 2 | 524287 | 0 | host_one | 5.709250 | 5.386150 | 1.006876 [0.960092, 1.054232] |
| 2 | 524287 | 0 | host_new | 5.709250 | 5.585350 | 1.025188 [0.968400, 1.081573] |
| 2 | 524287 | 0 | upload_new | 8.269350 | 8.447700 | 1.037550 [0.965417, 1.154974] |
| 2 | 524288 | 0 | host_one | 5.417950 | 5.590850 | 0.972144 [0.925060, 1.022006] |
| 2 | 524288 | 0 | host_new | 5.417950 | 4.639400 | 1.206087 [1.146464, 1.272397] |
| 2 | 524288 | 0 | upload_new | 8.534250 | 7.219450 | 1.116944 [1.000493, 1.209644] |
| 2 | 524289 | 0 | host_one | 5.783600 | 5.646250 | 1.024269 [0.981275, 1.071764] |
| 2 | 524289 | 0 | host_new | 5.783600 | 4.638950 | 1.275569 [1.215303, 1.341785] |
| 2 | 524289 | 0 | upload_new | 8.262300 | 7.202800 | 1.079294 [0.967147, 1.170393] |
| 2 | 1048577 | 0 | host_one | 10.499400 | 9.889600 | 1.053840 [1.020493, 1.092004] |
| 2 | 1048577 | 0 | host_new | 10.499400 | 7.862100 | 1.334534 [1.285835, 1.389209] |
| 2 | 1048577 | 0 | upload_new | 15.379800 | 13.592800 | 1.081492 [0.952184, 1.191971] |
| 2 | 1048577 | 1 | host_one | 12.576700 | 12.104650 | 1.030754 [0.974761, 1.087097] |
| 2 | 1048577 | 1 | host_new | 12.576700 | 9.410350 | 1.348693 [1.277098, 1.431054] |
| 2 | 1048577 | 1 | upload_new | 25.008900 | 21.012750 | 1.156208 [1.049314, 1.264104] |
| 2 | 1048577 | 2 | host_one | 33.999700 | 29.739800 | 1.134504 [1.061598, 1.219385] |
| 2 | 1048577 | 2 | host_new | 33.999700 | 15.516500 | 2.193030 [2.030568, 2.370166] |
| 2 | 1048577 | 2 | upload_new | 40.774800 | 29.899200 | 1.357210 [1.233487, 1.487243] |

## Source and executable receipts

109 measured source/executable hashes remain unchanged after both final surveys:
13 Rust sources, 95 shader sources and the executable. The complete receipt is
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/tree-staging-provenance.json`. The parent
identity verifies the 103 unchanged consumer/query/shader sources. Changed
packer/test sources and the executable are listed below.

| Absolute path | SHA-256 |
| --- | --- |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/nearest.rs` | `36611d13b8779d99a742e180362ddbf77e61fc222e77fc7a35337c1a669061bc` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_staging.rs` | `fda7e6df2f6cc13def30d63ce06c1581db94e58d5666fc3d894a1ef0bc60927b` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_staging_tests.rs` | `f01aa7cac267749ec41ea469bae2013bdfc9ec96af2184979168266cf100020f` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_staging_gpu_tests.rs` | `2824526c8767ed92ba638a1792f0f047c93792a3f74ef3e465ae4362f5a5e181` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_staging_benchmark.rs` | `5760d0edf1d1ec61ae4b259cea3183b9d8026715a6cec1cb8224162427a2faa0` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-62f845495446a7e6.exe` | `c306da11e641a8bd1f626d1a2aae3f15c664a8492c018b05256dd35228cc1c84` |

Raw accepted final logs:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/tree-staging-benchmark-1.log`: SHA-256 `be7b54bd80264dfd736edc3f7f36320ce07baeb64df32b39154d4db253296fda`.
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/tree-staging-benchmark-2.log`: SHA-256 `5e7719cfcb507912326e863a15feec6da68b15b3624435470daef9ef441186c5`.
