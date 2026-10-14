# Bounded native SoA GPU query staging

Date: 2026-10-14. Parent: `3345396ffdd79e734461c2b065033b33b7ad53a4` (merged PR 196).

## Change and boundary

The shared private SoA query packer serves FPFH, ISS, MLS and the registration
target used by ICP, GICP and point-to-plane inputs. Native calls with at least
524,288 points use available CPUs and useful 262,144-point grains, capped at
four scoped workers. Smaller, one-worker and WASM calls keep the original
serial finite scans and collection; small calls skip the CPU-count query.

Workers validate their borrowed axes and write disjoint original-order bands
of one initialized 16*N-byte output allocation. This is the same output payload
capacity as the previous packer, with worker-sized handles and no point-sized
join vector. Initialization is included in all measurements. Large invalid
inputs can allocate this host output before finite validation finishes; shape,
device-buffer and dispatch checks precede packing, and every worker must succeed
before device-buffer creation. Invalid-input errors remain exact.

Original f32 coordinate bits, negative zero, subnormals, record order and
positive-zero padding are retained. GPU query uploads remain explicit, record
16*N host-to-device bytes and zero device-to-host bytes, and own their input
buffer after CPU storage is dropped. Public AoS uploads and the SoA upload
constructor body match the parent. Seven consumer sources and all 95 GPU shader
sources also match the parent. There are no new dependencies or unsafe blocks.

Production paths in this checkout:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/query_staging.rs`

## Correctness and integration

Four new release tests passed. The CPU tests compare 257,951,616 packed words
against the frozen parent and independent coordinate-bit/padding expectations,
including empty/small/threshold/partial and 1,048,577-point inputs, four finite
profiles, injected 0/1/2/6/12/unbounded CPU budgets and the actual dispatcher.
The worker planner separately checks threshold/grain/maximum arithmetic.

Invalid axes and NaN/positive-infinity/negative-infinity at each worker band's
first/last positions return the frozen error. Four concurrent callers retain
exact coordinate bits. The native GPU test probes 41,943,344 uploaded words in
28 size/profile cases, including both sides of the threshold, every zero sign,
subnormals and large finite bit patterns. The integer storage probe uses bounded
grid-stride dispatch; it adds no production buffer usage or readback. CPU axes
are dropped before probing old/new owned buffers. Empty allocation, count,
transfer receipts and large invalid-input rejection within a validation scope
are checked.

Related native regressions: 33 successes (13 nearest/neighborhood, three prior
feature-input equivalence tests, three FPFH, four ISS, four MLS and six shared
registration geometry/SoA input tests). Public facade tests: eight successes
across FPFH/ISS/MLS, neighborhood search, ICP/trimmed ICP, GICP and point-to-plane
equations/convergence. GPU all-feature/all-target Clippy passed with warnings
denied. GPU-disabled and WASM builds passed; formatting and diff checks passed.
Frozen serial pack/upload functions were compared to the parent after function
names, concrete constructor types and formatting normalization.

Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --features gpu-image --release --lib query_staging -- --include-ignored --skip survey_bounded_host_packing --skip completed_host_and_query_uploads --nocapture --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --no-default-features
cargo check -p spatialrust --no-default-features --features register-fpfh-gpu,feature-iss-gpu,filter-mls-gpu,register-icp-gpu,register-gicp-gpu,register-icp-point-to-plane-gpu --target wasm32-unknown-unknown
```

Local verification logs and measurement receipts are in
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/query-staging-*`.

## Completed-work measurement

Environment: `rustc 1.98.1 (48a229cea 2026-09-01)`; release profile; Vulkan discrete-GPU backend;
12 available CPU workers, at most four used by the new packer. No adapter model
is recorded. Two sequential accepted surveys ran without other build/test/
benchmark processes; each passed on attempt 1 with empty stderr. A controller
observed a quiet window, monitored process overlap and would discard a complete
attempt and cancel only its own benchmark child on overlap. Analysis started
after both surveys finished. Durations were 19.21 and 13.89 seconds.

Each of 11 size/profile cases rotates five routes cyclically for three warmup
and 30 recorded rounds: frozen parent host pack, current forced-one-worker host
pack, current automatic host pack, frozen parent query upload and current query
upload. Fixtures and runtime setup are outside timing. Host timing includes
validation, allocation, packing, observation and drop. Query upload timing
includes limits/shape validation, host packing, device-buffer creation,
`wait_idle`, count/receipt capture and buffer drop. Assertions follow timing.
This measures the shared query-input boundary; it does not time tree upload,
feature sample packing, neighborhood search, registration/feature kernels or
whole-algorithm latency. Profiles: 0 bounded varied finite coordinates,
1 alternating signed zeros, 2 signed subnormals.

The table retains all 66 route comparisons. Ratios are paired geometric means
of old/new sample times. Values above one indicate shorter current times.
Intervals use 10,000 paired log-ratio bootstrap resamples, seed
`2026101417 + 100*survey + 10*case + field_index` (field index in the JSON record),
and sorted entries 249/9749. Each row has 30 samples. One host-one comparison at
one point contains a zero clock sample and has no ratio estimate; its medians
and zero counts remain. All one-point host durations are near timer resolution
and support no practical speed claim, including nonzero ratios.

At 1,048,577 varied finite points, host ratios are 1.535112 and 1.534383, with
both intervals above one. Query-upload medians fall from 9.200900 to 6.430850 ms
and 9.216250 to 6.365300 ms (about 30% less time); paired ratios are 1.603077
[1.372690, 1.973805] and 1.485314 [1.327054, 1.714036]. At the exact 524,288-point
threshold, both host and upload intervals also lie above one.

Limits and adverse cases remain visible. At 2,097,153 points, the first upload
ratio is 0.899216 [0.681008, 1.136414], while the second is 1.242692
[1.100791, 1.375775]; no repeatable transfer-boundary gain is claimed there.
At 65,537 points (serial in both implementations), the second upload ratio is
0.890169 [0.815600, 0.972269]. The first interval crosses one. Signed-zero and
subnormal upload timings have substantial variability; the second subnormal
interval crosses one. There is no claim of universal input or end-to-end
algorithm acceleration.

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | host_one | 0.000200 | 0.000100 | N/A: zero samples old=0, new=1 |
| 1 | 1 | 0 | host_new | 0.000200 | 0.000100 | 1.481098 [1.319508, 1.662476] |
| 1 | 1 | 0 | upload_new | 0.008250 | 0.007800 | 1.120361 [1.052076, 1.210610] |
| 1 | 4097 | 0 | host_one | 0.011500 | 0.011100 | 1.020510 [0.953026, 1.073099] |
| 1 | 4097 | 0 | host_new | 0.011500 | 0.010950 | 1.057840 [1.037388, 1.091433] |
| 1 | 4097 | 0 | upload_new | 0.059100 | 0.059850 | 1.139100 [0.944545, 1.605055] |
| 1 | 65537 | 0 | host_one | 0.446100 | 0.443950 | 1.012689 [0.963376, 1.064329] |
| 1 | 65537 | 0 | host_new | 0.446100 | 0.437950 | 1.026383 [0.955360, 1.107039] |
| 1 | 65537 | 0 | upload_new | 0.639500 | 0.733550 | 0.897617 [0.807432, 1.005674] |
| 1 | 262145 | 0 | host_one | 1.421350 | 1.472000 | 0.975594 [0.943029, 1.008677] |
| 1 | 262145 | 0 | host_new | 1.421350 | 1.439900 | 0.991408 [0.952906, 1.035730] |
| 1 | 262145 | 0 | upload_new | 2.223850 | 2.032800 | 1.140457 [1.033813, 1.322831] |
| 1 | 524287 | 0 | host_one | 3.209250 | 3.210950 | 0.997772 [0.967046, 1.030153] |
| 1 | 524287 | 0 | host_new | 3.209250 | 3.343300 | 0.967064 [0.932775, 1.002407] |
| 1 | 524287 | 0 | upload_new | 4.662350 | 4.209250 | 1.179856 [1.042408, 1.433131] |
| 1 | 524288 | 0 | host_one | 2.990800 | 3.015250 | 1.001471 [0.966559, 1.046289] |
| 1 | 524288 | 0 | host_new | 2.990800 | 2.478100 | 1.208493 [1.145654, 1.272881] |
| 1 | 524288 | 0 | upload_new | 4.199950 | 3.427400 | 1.194309 [1.126849, 1.256877] |
| 1 | 524289 | 0 | host_one | 3.442350 | 3.161650 | 0.995231 [0.954117, 1.041380] |
| 1 | 524289 | 0 | host_new | 3.442350 | 2.769950 | 1.231865 [1.171032, 1.296721] |
| 1 | 524289 | 0 | upload_new | 4.508750 | 4.097350 | 1.292716 [1.130152, 1.603044] |
| 1 | 1048577 | 0 | host_one | 6.719350 | 6.543500 | 0.973744 [0.945541, 1.003099] |
| 1 | 1048577 | 0 | host_new | 6.719350 | 4.146500 | 1.535112 [1.469921, 1.597774] |
| 1 | 1048577 | 0 | upload_new | 9.200900 | 6.430850 | 1.603077 [1.372690, 1.973805] |
| 1 | 2097153 | 0 | host_one | 16.282400 | 14.783900 | 1.102799 [1.043917, 1.170672] |
| 1 | 2097153 | 0 | host_new | 16.282400 | 8.410450 | 1.969066 [1.838486, 2.114094] |
| 1 | 2097153 | 0 | upload_new | 20.028000 | 16.320400 | 0.899216 [0.681008, 1.136414] |
| 1 | 1048577 | 1 | host_one | 11.128550 | 11.810600 | 0.960074 [0.921669, 1.000997] |
| 1 | 1048577 | 1 | host_new | 11.128550 | 6.726650 | 1.626239 [1.553282, 1.700782] |
| 1 | 1048577 | 1 | upload_new | 50.287750 | 11.780750 | 2.438249 [1.522741, 3.647675] |
| 1 | 1048577 | 2 | host_one | 11.691100 | 12.138500 | 0.965077 [0.935251, 0.995848] |
| 1 | 1048577 | 2 | host_new | 11.691100 | 6.921550 | 1.627279 [1.541403, 1.719431] |
| 1 | 1048577 | 2 | upload_new | 45.580900 | 12.625650 | 2.243779 [1.446855, 3.444329] |
| 2 | 1 | 0 | host_one | 0.000200 | 0.000100 | 1.219395 [1.096825, 1.368724] |
| 2 | 1 | 0 | host_new | 0.000200 | 0.000100 | 1.191544 [1.003934, 1.400717] |
| 2 | 1 | 0 | upload_new | 0.010200 | 0.009100 | 1.155025 [1.002768, 1.323070] |
| 2 | 4097 | 0 | host_one | 0.011700 | 0.011200 | 1.040526 [0.976726, 1.089902] |
| 2 | 4097 | 0 | host_new | 0.011700 | 0.011100 | 1.061871 [1.002666, 1.112770] |
| 2 | 4097 | 0 | upload_new | 0.062100 | 0.062300 | 1.150581 [0.977643, 1.545438] |
| 2 | 65537 | 0 | host_one | 0.430950 | 0.421800 | 1.037059 [1.003084, 1.077947] |
| 2 | 65537 | 0 | host_new | 0.430950 | 0.423800 | 1.004717 [0.930343, 1.070693] |
| 2 | 65537 | 0 | upload_new | 0.587450 | 0.662750 | 0.890169 [0.815600, 0.972269] |
| 2 | 262145 | 0 | host_one | 1.701000 | 1.606050 | 1.046767 [1.008634, 1.087721] |
| 2 | 262145 | 0 | host_new | 1.701000 | 1.642200 | 1.034658 [0.996145, 1.078246] |
| 2 | 262145 | 0 | upload_new | 2.459600 | 2.271400 | 1.169385 [1.065843, 1.345294] |
| 2 | 524287 | 0 | host_one | 3.519550 | 3.584250 | 0.969154 [0.936299, 1.001375] |
| 2 | 524287 | 0 | host_new | 3.519550 | 3.600350 | 0.986342 [0.949311, 1.023052] |
| 2 | 524287 | 0 | upload_new | 4.825600 | 4.797000 | 1.080159 [0.974312, 1.268257] |
| 2 | 524288 | 0 | host_one | 3.498600 | 3.429650 | 0.996423 [0.964652, 1.027857] |
| 2 | 524288 | 0 | host_new | 3.498600 | 2.800050 | 1.242682 [1.189287, 1.301313] |
| 2 | 524288 | 0 | upload_new | 4.849800 | 4.291800 | 1.137286 [1.085986, 1.186100] |
| 2 | 524289 | 0 | host_one | 3.381250 | 3.439350 | 0.978991 [0.944621, 1.011643] |
| 2 | 524289 | 0 | host_new | 3.381250 | 2.846250 | 1.181677 [1.138618, 1.227297] |
| 2 | 524289 | 0 | upload_new | 4.776700 | 4.016300 | 1.248771 [1.131286, 1.453461] |
| 2 | 1048577 | 0 | host_one | 6.331650 | 6.261750 | 0.988167 [0.952843, 1.023577] |
| 2 | 1048577 | 0 | host_new | 6.331650 | 4.196000 | 1.534383 [1.484776, 1.589667] |
| 2 | 1048577 | 0 | upload_new | 9.216250 | 6.365300 | 1.485314 [1.327054, 1.714036] |
| 2 | 2097153 | 0 | host_one | 13.939100 | 13.300850 | 1.011440 [0.978779, 1.045003] |
| 2 | 2097153 | 0 | host_new | 13.939100 | 8.068800 | 1.740216 [1.688367, 1.794697] |
| 2 | 2097153 | 0 | upload_new | 18.223250 | 13.044450 | 1.242692 [1.100791, 1.375775] |
| 2 | 1048577 | 1 | host_one | 14.725050 | 8.754750 | 1.401111 [1.263959, 1.544630] |
| 2 | 1048577 | 1 | host_new | 14.725050 | 5.173250 | 2.485854 [2.243669, 2.733129] |
| 2 | 1048577 | 1 | upload_new | 13.464750 | 10.934300 | 1.211161 [1.049200, 1.388144] |
| 2 | 1048577 | 2 | host_one | 14.939600 | 8.944550 | 1.403319 [1.259936, 1.561091] |
| 2 | 1048577 | 2 | host_new | 14.939600 | 5.440350 | 2.453874 [2.209484, 2.707575] |
| 2 | 1048577 | 2 | upload_new | 14.370800 | 11.363100 | 1.194101 [0.998262, 1.409722] |

## Source and executable receipts

The 108 file hashes captured before measurement still match after both surveys:
12 Rust sources, 95 unchanged shader sources and the measured executable.
Parent source identity verifies the 102 unchanged consumer/shader files.
The complete local receipt is
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/query-staging-provenance.json`.
The changed packer/test sources and executable are listed below.

| Absolute path | SHA-256 |
| --- | --- |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs` | `ff6eab2a048481dc74167ca71648861742d7fb8db7da166f8ad7b5390b78ef2d` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/query_staging.rs` | `3285eba39557c0174f04b2fb9b9d7e380cac16f27c5daf5b6cde1a5b4f982cd7` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/query_staging_tests.rs` | `23a164df579f9ea0579ce3d41dcb710dc53be0ba1bd6bce1a548ec01861f790d` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/query_staging_gpu_tests.rs` | `eaf7a58c12f38c238319249057f9cc5b95bd8df0d62a68a41cc28c16d4c7999a` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/query_staging_benchmark.rs` | `9780adce9e3ca4c871616c5c2c5e74811b3fcb9ca6d0b80c0fc2159cbe035169` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-62f845495446a7e6.exe` | `02cd857f698a1244f68c5d2d5f8ec2eec1098b61de662729ce6698d0794b5f8e` |

Raw accepted survey logs:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/query-staging-benchmark-1.log`: SHA-256 `27350ae0355abb2349711875e9b950c9b479936616521b71132bde88eb211134`.
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/query-staging-benchmark-2.log`: SHA-256 `2405e364f40fa9896ff28d01b92efa909b5684928699daa2fd91cebfe732b5e5`.
