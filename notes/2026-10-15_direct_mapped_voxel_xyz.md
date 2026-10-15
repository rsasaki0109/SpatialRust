# Direct mapped voxel XYZ readback

The resident-input public XYZ gather and centroid functions previously read
12*N bytes into one flat float vector, then split off the Y and Z planes.
The returned X vector retained capacity for all three planes; Y and Z each
allocated another 4*N bytes and copied their values a second time.

Both functions now use one private mapped-plane decoder to copy each plane
straight into its final owned vector. The old split helper is removed.
Requested output payload/capacity falls from 20*N to 12*N bytes; 8*N bytes of
repeat host copies and excess retained capacity are removed. Allocation count
is still three, and GPU output/staging allocations and 12*N download bytes
remain. For 1,048,576 output cells, this removes 8,388,608 bytes (8 MiB) of
capacity and repeat copying. This is not an RSS or process peak-memory claim.

`gather_voxel_first_xyz_gpu_buffers` and
`reduce_voxel_centroids_xyz_gpu_buffers` preserve their signatures, validation,
empty returns, runtime pipelines, buffer layouts, copy commands, map/wait and
diagnostics. `bytemuck::cast_slice` and float copies retain all source bits.
Shader arithmetic and cell/point order do not change. Combined attribute APIs,
single-channel reads and their decoding remain as before. No new GPU pass,
transfer, synchronization, unsafe code or dependency is introduced.

The view remains mapped while the three vectors are populated. A private RAII
guard is declared before the view and unmaps after the view drops on return
or unwinding. No mapped reference escapes in a returned coordinate vector.

## Verification method

The complete parent gather entry point, centroid entry point/dispatch, float
mapper and XYZ splitter are frozen from merge
`c4b3155f9d63faed8b1a60499601bc3ee0fddc51`. Only test visibility and names
change. Production changes are limited to the new decoder, removal of the
old split helper and two call sites/imports. Other GPU sources and every WGSL
file must match that parent before measurements.

Raw staged words cover 0, 1, 17, 256, 65,537 and 262,145 cells plus trailing
padding, using signed zeros, NaN payloads, infinities, subnormals, extrema and
mixed u32 encodings. Direct and frozen decoding, lengths/capacities, repeat
mapping, independent live outputs and source buffer destruction are checked.
A deliberate out-of-bounds decoder panic must release the mapped view and
allow subsequent direct and original maps of the same staging buffer.

Complete public calls use 0, 1, 17, 257 and 65,537 cells with one, two and
four scattered source points per cell. Finite dyadic coordinates allow an
independent exact CPU gather/mean oracle. The 257-cell fixtures additionally
build segments through the public GPU sorter; the other fixtures explicitly
upload valid CPU group indices/starts. Original host sources are dropped
after upload. Frozen parity, empty behavior, retained capacities, repeated
calls, pool clears, dropped GPU inputs and concurrent callers are checked.

## Measurement method

The benchmark times complete public resident-input gather/centroid calls:
output/staging/uniform setup, compute/copy submission, map/wait, host decoding
and owned return. Upload, segment construction, CPU expected values, output
assertions and output drop are outside the timer. It is not a measurement of
complete downsampling with key construction/sort/upload, or GPU pass time.

Eight fixed input conditions use 16, 4096, 262,144 and 1,048,576 output cells
with one/four source points per cell. Both routes retain three warmups and
30 alternating paired samples against the frozen complete parent API. Every
returned word is checked against the independent CPU oracle. All conditions
and favorable/adverse/inconclusive results must be retained.

Paired log old/new ratios use 10,000 bootstrap samples, sorted indices
249/9749 for 95% intervals, with seed
2026101517 + 100*survey + 10*condition_index + JSON_field_index. Ratios above
one favor direct decoding; intervals crossing one are inconclusive. A zero
sample suppresses that row's ratio. These ratios are geometric means of paired
ratios, rather than ratios of the reported medians.

Two isolated surveys are attempted with three-minute quiet scheduling windows
and competing-work checks before, during and after each benchmark process.
Overlapping/incomplete trials are discarded in full; other projects' jobs are
left running. If scheduling remains unavailable, timing evidence is reported
as unavailable or unreplicated. Source/capacity evidence is kept separate from
latency evidence. Raw validation/measurement logs and hash receipts remain local
under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with the
`voxel-xyz-` prefix. Published environment descriptions omit the device model.

## Completed checks and measurements

All 36 native GPU voxel regressions and 18 voxel-filter regressions passed
(54 distinct local tests). This includes 2,951,604 exact raw float-word
comparisons and 3,553,848 frozen/analytic/repeated public-output comparisons,
plus retained outputs, padding, capacities, empty inputs, pool clears,
concurrent calls and panic/remap checks. The deliberate decoder panic is
caught by its test and is not a failed test.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds and
two warnings-denied WASM GPU configurations (minimal `gpu-wgpu` and CI's
`gpu-image,gpu-aoso-staging`) passed, as did formatting and diff checks.
The GPU feature by itself did not select voxel filter tests; the actual
18-test filter run enables both `filter-voxel` and `filter-voxel-gpu`.

355 source/executable hashes were pinned after validation and verified after
measurement attempts. All five frozen parent functions match the original.
347 other existing GPU/image sources and all 97 WGSL files are unchanged.
The new mapper preserves the original map/wait/error path. Privacy scans
found no forbidden adapter-name matches.

2 isolated survey(s), 32 rows: 16 favorable, 2 adverse, 14 inconclusive.

Survey 1 accepted attempt 100 across 2 bounded scheduling window(s), after 0 discarded overlapping trial(s). The accepted benchmark exited zero with one passing test and empty stderr; competing work was not detected before, during or after the process.

Survey 2 accepted attempt 1 across 1 bounded scheduling window(s), after 0 discarded overlapping trial(s). The accepted benchmark exited zero with one passing test and empty stderr; competing work was not detected before, during or after the process.

Accepted-run environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 boundary=complete_public_resident_input_call output_drop_asserts_excluded=true`.

| Survey | Cells / points per cell | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 16 / 1 | gather | 0.891950 | 0.898200 | 0.993519 [0.963264, 1.021294] |
| 1 | 16 / 1 | reduce | 0.944150 | 0.946550 | 0.995692 [0.982868, 1.008573] |
| 1 | 16 / 4 | gather | 0.937850 | 0.936950 | 0.998645 [0.979979, 1.017674] |
| 1 | 16 / 4 | reduce | 0.992300 | 0.982600 | 0.997956 [0.942214, 1.038455] |
| 1 | 4096 / 1 | gather | 0.946300 | 0.945300 | 1.005908 [0.987292, 1.026361] |
| 1 | 4096 / 1 | reduce | 0.959700 | 0.948900 | 1.004277 [0.982136, 1.026876] |
| 1 | 4096 / 4 | gather | 0.990550 | 1.008050 | 0.983373 [0.957947, 1.009629] |
| 1 | 4096 / 4 | reduce | 1.096200 | 1.066000 | 1.018905 [0.972144, 1.065273] |
| 1 | 262144 / 1 | gather | 2.816350 | 2.454300 | 1.165862 [1.134930, 1.196844] |
| 1 | 262144 / 1 | reduce | 2.806200 | 2.467550 | 1.159456 [1.123966, 1.195297] |
| 1 | 262144 / 4 | gather | 3.518900 | 3.103950 | 1.135911 [1.104160, 1.169754] |
| 1 | 262144 / 4 | reduce | 3.488950 | 3.076200 | 1.111592 [1.073872, 1.146098] |
| 1 | 1048576 / 1 | gather | 10.221200 | 8.778850 | 1.154193 [1.115812, 1.192983] |
| 1 | 1048576 / 1 | reduce | 14.095850 | 12.393700 | 1.129541 [1.057646, 1.210197] |
| 1 | 1048576 / 4 | gather | 11.702350 | 9.936500 | 1.167552 [1.149747, 1.186874] |
| 1 | 1048576 / 4 | reduce | 12.565700 | 11.087700 | 1.126180 [1.088983, 1.151227] |
| 2 | 16 / 1 | gather | 0.974800 | 0.963450 | 1.009654 [0.972514, 1.046072] |
| 2 | 16 / 1 | reduce | 0.937500 | 0.963550 | 0.978094 [0.957647, 0.998829] |
| 2 | 16 / 4 | gather | 0.926150 | 0.946700 | 0.967926 [0.946432, 0.988641] |
| 2 | 16 / 4 | reduce | 0.966650 | 0.950650 | 1.017166 [0.995721, 1.041218] |
| 2 | 4096 / 1 | gather | 1.010400 | 1.006550 | 0.995080 [0.968772, 1.020235] |
| 2 | 4096 / 1 | reduce | 1.051700 | 1.053600 | 0.993559 [0.968961, 1.020379] |
| 2 | 4096 / 4 | gather | 1.044700 | 1.047400 | 0.992829 [0.964543, 1.019811] |
| 2 | 4096 / 4 | reduce | 1.018950 | 1.010100 | 1.016229 [0.990027, 1.042298] |
| 2 | 262144 / 1 | gather | 2.818800 | 2.434450 | 1.168154 [1.141261, 1.195458] |
| 2 | 262144 / 1 | reduce | 2.703850 | 2.277250 | 1.176949 [1.148806, 1.206162] |
| 2 | 262144 / 4 | gather | 3.540700 | 3.130400 | 1.137187 [1.107922, 1.167220] |
| 2 | 262144 / 4 | reduce | 3.428600 | 3.084600 | 1.100408 [1.068151, 1.132058] |
| 2 | 1048576 / 1 | gather | 10.141750 | 8.781800 | 1.192765 [1.151801, 1.248302] |
| 2 | 1048576 / 1 | reduce | 13.952150 | 12.178700 | 1.105772 [1.023517, 1.196498] |
| 2 | 1048576 / 4 | gather | 11.541200 | 9.892450 | 1.154047 [1.112137, 1.183983] |
| 2 | 1048576 / 4 | reduce | 12.595800 | 11.341500 | 1.086962 [1.037040, 1.128911] |

Survey 1's initial quiet window expired before a benchmark started. Survey 2
then completed; one additional bounded window completed survey 1 (attempt 100).
Both accepted surveys use the original fixed matrix and executable, with no
discarded partial results contributing timing samples.

All eight large-input rows in each survey (262,144 and 1,048,576 cells) favor
direct decoding by their paired intervals. With 1,048,576 cells and four
points per cell, gather medians are 11.702350 to 9.936500 ms and 11.541200 to
9.892450 ms (15.1% / 14.3% lower). Centroid medians are 12.565700 to
11.087700 ms and 12.595800 to 11.341500 ms (11.8% / 10.0% lower).

The two adverse intervals occur only in survey 2 at 16 cells: one-point
centroids rise from 0.937500 to 0.963550 ms, and four-point gather rises
from 0.926150 to 0.946700 ms. Both are inconclusive in survey 1. All 4096-cell
rows are inconclusive. The change consistently reduces requested output
capacity; it is not a universal latency improvement for small calls. These
are local resident-input completed-call observations, not portable or complete
application latency guarantees or process peak-memory measurements.
