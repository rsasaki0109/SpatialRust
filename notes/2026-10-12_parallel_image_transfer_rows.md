# Bounded host rows for explicit GPU image transfers

## Execution and ownership

GPU image upload previously copied each logical pixel into a zero-filled RGBA
buffer. Readback appended each pixel's logical channels to a capacity-reserved
vector. The revised host transforms copy complete rows for four channels and
write directly into disjoint output rows for one, two and three channels.
Unused upload components remain zero; source ROI/row padding and readback
alignment padding are excluded.

Native 1..=3-channel images from 1,048,576 pixels use scoped row bands. Worker
count is the minimum of available CPU parallelism, useful 262,144-pixel chunks
and image height. Small images, single rows, four-channel row copies and WASM
keep caller execution. Scopes join before returning output. A final partial
band is supported, and zero requested workers normalize to one in test helpers.
Mutable output bands are disjoint; input views and staging bytes are borrowed.

The budget is per call, not a global CPU/thread/stack limit. Concurrent callers
have independent budgets. Scoped scheduling metadata scales with workers. The
existing RGBA upload vector, mapped staging-to-RGBA host copy and logical
readback output remain; no extra image-sized transform scratch is introduced.
Logical output now uses a safely initialized vector with fixed row destinations.
No unsafe code, dependency, public signature, shader or device copy is added.

Production sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/transfer_packing.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/gpu_image.rs`

All device work remains inside caller-selected upload/readback APIs. Existing
validation precedes texture/buffer/packing allocation. Runtime ownership,
metadata, pool lifetime and four-physical-byte receipts retain their contracts.
Readback staging padding is not counted as logical transferred texels. No host
synchronization is added to resident compute chains or production uploads.

## Correctness and baseline audit

CPU tests compare **838,120,764 bytes** across independent coordinate/channel
oracles, frozen loops, forced 0/1/2/6 workers and public worker selection. Cases
include all four channel counts, packed/strided/ROI layouts, dirty host padding,
alignment boundaries, one-row/tall strips, threshold neighbors, partial row
bands and empty internal transforms. Large fixtures use ROI layout; small
fixtures cover all three layouts. Four concurrent large transforms retain
previous results after subsequent calls. This count denotes comparisons,
not distinct pixel values.

Native GPU tests compare **69,574,296 physical/logical bytes** through small and
large strided ROI uploads, two successive readbacks, all channel counts,
limited/BGR/BGRA/premultiplied metadata, full physical RGBA views and exact
receipt equality. Four concurrent callers retain two large live textures each
and recycle them only after output checks. Existing rejection, transfer-limit,
pool, image-chain and point-cloud regressions remain in the native suite.
The facade adds an explicitly ignored native large-ROI smoke test; adapter
initialization errors are not treated as success.

The parent is `fd740e9a5db2a25e5967556d50ddb6e5621cb5f9`. Frozen
upload/readback methods and packing loop preserve its source after visibility,
method routing and formatting substitutions. The frozen unpack helper retains
the original append loop with plan fields replaced by equivalent arguments.
Both API versions use the same current transfer plans, runtime texture pool,
queue writes/copies and staging-map helper. Source comparison verifies that
dimension/capacity validation and staging are unchanged.

Test and fixture sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/transfer_packing_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/transfer_packing_gpu_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/image_transfer_serial_packing.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/image_transfer_serial_api.rs`

## Validation

| Check | Result |
| --- | --- |
| Native GPU image/AoSoA and point-cloud regressions | 179 passed (152 unit + 27 integration), 0 failed, 19 ignored diagnostics |
| Native facade image transfer smoke | 2 passed, including the explicitly enabled large-ROI test |
| All-feature/all-target GPU Clippy | Passed with `-D warnings` |
| Minimal GPU crate | Compile check passed |
| GPU image/AoSoA WASM | Compile check passed |
| Source baseline audit | Frozen loops/APIs and unchanged validation/staging verified |
| Formatting/whitespace | Passed |

## Measurement boundary

```powershell
cargo test -p spatialrust-gpu --release --features gpu-image paired_serial_current_transfer_rows -- --ignored --nocapture --test-threads=1
cargo test -p spatialrust-gpu --release --features gpu-image paired_serial_current_gpu_image_transfers -- --ignored --nocapture --test-threads=1
```

Each condition uses three warmups per version and twenty alternating-order
pairs. CPU tests time completed transforms with allocation included. Returned
vector destruction, equality checking and input generation are outside timing.
GPU tests time each completed public transfer: uploads include an explicit
empty queue submission to flush pending writes and `wait_idle`; readbacks
include staging allocation/copy, mapping wait, host staging copy, logical
unpacking and image construction. Upload verification readback is outside its
timer. Production uploads retain asynchronous queue-write behavior.

CPU input rows are strided; GPU inputs are strided ROIs. Sizes are 640x480,
1920x1080 and 3840x2160 with each logical channel count. The comparison shares
current validation/staging/pools and uses no GPU pass timestamps. Correctness
and build jobs finish before isolated release timing. Windows rustc 1.98.1,
thin LTO, one codegen unit and 12 available CPU workers describe the host;
native GPU completion uses Vulkan and a discrete adapter without model names.
WASM checks establish compilation only.

Reported ratios use geometric means of paired previous/current times, with
95% paired log-ratio bootstrap intervals from 10,000 resamples, seed 20261012.
Ratios above one favor the revision. Separate marginal medians need not have
the same ratio. Intervals describe observed pairs, not systematic host bias or
unmeasured distributions/devices. Scalar/four-channel controls are distinguished
from threaded cases; improvements there cannot be credited to parallel work.

Private raw diagnostics and full sample summaries are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/image-transfer-rows-*`.
Device model identity is excluded from tracked files and diagnostic output.

## Two isolated surveys

All twenty pairs per condition are retained. Each survey contains 24 CPU
transform and 24 completed GPU-transfer conditions, for 96 condition records
over two surveys. `threaded` denotes native row bands; `scalar` includes small
images and four-channel row copies.

### Completed host transforms

| Survey | Operation | Channels | Size | Path | Previous median ms | Current median ms | Paired ratio [95% interval] |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | pack | 1 | 640x480 | scalar | 0.346 | 0.350 | 0.991 [0.968, 1.014] |
| 1 | unpack | 1 | 640x480 | scalar | 0.204 | 0.163 | 1.212 [1.176, 1.241] |
| 1 | pack | 1 | 1920x1080 | threaded | 1.723 | 1.120 | 1.509 [1.474, 1.545] |
| 1 | unpack | 1 | 1920x1080 | threaded | 2.017 | 1.161 | 1.718 [1.652, 1.779] |
| 1 | pack | 1 | 3840x2160 | threaded | 7.759 | 3.345 | 2.356 [2.210, 2.530] |
| 1 | unpack | 1 | 3840x2160 | threaded | 8.301 | 2.639 | 3.269 [3.078, 3.483] |
| 1 | pack | 2 | 640x480 | scalar | 0.375 | 0.381 | 1.002 [0.977, 1.032] |
| 1 | unpack | 2 | 640x480 | scalar | 0.411 | 0.244 | 1.677 [1.650, 1.694] |
| 1 | pack | 2 | 1920x1080 | threaded | 2.439 | 1.565 | 1.564 [1.502, 1.634] |
| 1 | unpack | 2 | 1920x1080 | threaded | 2.647 | 1.180 | 2.287 [2.200, 2.374] |
| 1 | pack | 2 | 3840x2160 | threaded | 8.066 | 3.497 | 2.280 [2.118, 2.430] |
| 1 | unpack | 2 | 3840x2160 | threaded | 9.897 | 2.723 | 3.692 [3.487, 3.931] |
| 1 | pack | 3 | 640x480 | scalar | 0.440 | 0.444 | 1.010 [0.992, 1.030] |
| 1 | unpack | 3 | 640x480 | scalar | 0.586 | 0.484 | 1.197 [1.171, 1.220] |
| 1 | pack | 3 | 1920x1080 | threaded | 2.387 | 1.244 | 1.866 [1.774, 1.960] |
| 1 | unpack | 3 | 1920x1080 | threaded | 3.034 | 1.229 | 2.522 [2.288, 2.732] |
| 1 | pack | 3 | 3840x2160 | threaded | 9.376 | 4.093 | 2.327 [2.167, 2.469] |
| 1 | unpack | 3 | 3840x2160 | threaded | 15.750 | 3.794 | 3.828 [3.553, 4.122] |
| 1 | pack | 4 | 640x480 | scalar | 0.312 | 0.302 | 1.004 [0.954, 1.046] |
| 1 | unpack | 4 | 640x480 | scalar | 0.466 | 0.297 | 1.538 [1.490, 1.584] |
| 1 | pack | 4 | 1920x1080 | scalar | 1.960 | 2.003 | 0.974 [0.900, 1.033] |
| 1 | unpack | 4 | 1920x1080 | scalar | 2.730 | 1.849 | 1.563 [1.451, 1.682] |
| 1 | pack | 4 | 3840x2160 | scalar | 7.575 | 7.510 | 1.048 [0.923, 1.189] |
| 1 | unpack | 4 | 3840x2160 | scalar | 10.998 | 6.792 | 1.634 [1.454, 1.807] |
| 2 | pack | 1 | 640x480 | scalar | 0.265 | 0.268 | 1.016 [0.995, 1.039] |
| 2 | unpack | 1 | 640x480 | scalar | 0.244 | 0.225 | 1.099 [1.064, 1.129] |
| 2 | pack | 1 | 1920x1080 | threaded | 1.659 | 1.193 | 1.356 [1.258, 1.460] |
| 2 | unpack | 1 | 1920x1080 | threaded | 1.731 | 0.973 | 1.770 [1.735, 1.803] |
| 2 | pack | 1 | 3840x2160 | threaded | 6.571 | 3.226 | 2.088 [1.921, 2.255] |
| 2 | unpack | 1 | 3840x2160 | threaded | 6.825 | 2.386 | 2.979 [2.845, 3.118] |
| 2 | pack | 2 | 640x480 | scalar | 0.261 | 0.261 | 0.967 [0.924, 1.006] |
| 2 | unpack | 2 | 640x480 | scalar | 0.210 | 0.079 | 2.738 [2.640, 2.887] |
| 2 | pack | 2 | 1920x1080 | threaded | 1.730 | 1.350 | 1.349 [1.275, 1.427] |
| 2 | unpack | 2 | 1920x1080 | threaded | 2.186 | 1.037 | 2.079 [1.985, 2.166] |
| 2 | pack | 2 | 3840x2160 | threaded | 6.781 | 3.420 | 2.102 [1.957, 2.317] |
| 2 | unpack | 2 | 3840x2160 | threaded | 8.316 | 2.812 | 2.957 [2.847, 3.098] |
| 2 | pack | 3 | 640x480 | scalar | 0.354 | 0.348 | 1.026 [0.990, 1.065] |
| 2 | unpack | 3 | 640x480 | scalar | 0.420 | 0.339 | 1.238 [1.213, 1.262] |
| 2 | pack | 3 | 1920x1080 | threaded | 2.307 | 1.509 | 1.553 [1.519, 1.590] |
| 2 | unpack | 3 | 1920x1080 | threaded | 3.026 | 1.161 | 2.611 [2.549, 2.678] |
| 2 | pack | 3 | 3840x2160 | threaded | 9.318 | 3.609 | 2.611 [2.549, 2.674] |
| 2 | unpack | 3 | 3840x2160 | threaded | 11.954 | 3.442 | 3.506 [3.405, 3.612] |
| 2 | pack | 4 | 640x480 | scalar | 0.411 | 0.427 | 1.104 [1.015, 1.220] |
| 2 | unpack | 4 | 640x480 | scalar | 0.406 | 0.256 | 1.620 [1.551, 1.694] |
| 2 | pack | 4 | 1920x1080 | scalar | 1.703 | 1.804 | 0.964 [0.940, 0.989] |
| 2 | unpack | 4 | 1920x1080 | scalar | 2.559 | 1.724 | 1.481 [1.400, 1.555] |
| 2 | pack | 4 | 3840x2160 | scalar | 6.843 | 6.548 | 1.050 [1.012, 1.095] |
| 2 | unpack | 4 | 3840x2160 | scalar | 10.247 | 6.440 | 1.584 [1.551, 1.612] |

### Completed public GPU transfers

| Survey | Operation | Channels | Size | Path | Previous median ms | Current median ms | Paired ratio [95% interval] |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | upload | 1 | 640x480 | scalar | 0.738 | 0.729 | 1.011 [0.992, 1.029] |
| 1 | readback | 1 | 640x480 | scalar | 1.660 | 1.643 | 1.012 [0.995, 1.030] |
| 1 | upload | 1 | 1920x1080 | threaded | 3.629 | 3.314 | 1.102 [1.080, 1.125] |
| 1 | readback | 1 | 1920x1080 | threaded | 7.054 | 6.041 | 1.154 [1.100, 1.199] |
| 1 | upload | 1 | 3840x2160 | threaded | 14.846 | 10.512 | 1.313 [1.182, 1.450] |
| 1 | readback | 1 | 3840x2160 | threaded | 28.225 | 21.392 | 1.295 [1.242, 1.341] |
| 1 | upload | 2 | 640x480 | scalar | 1.643 | 1.710 | 0.979 [0.945, 1.014] |
| 1 | readback | 2 | 640x480 | scalar | 7.378 | 7.002 | 1.048 [1.013, 1.083] |
| 1 | upload | 2 | 1920x1080 | threaded | 6.631 | 5.957 | 1.101 [1.070, 1.134] |
| 1 | readback | 2 | 1920x1080 | threaded | 13.276 | 11.613 | 1.146 [1.116, 1.181] |
| 1 | upload | 2 | 3840x2160 | threaded | 23.315 | 18.954 | 1.202 [1.134, 1.273] |
| 1 | readback | 2 | 3840x2160 | threaded | 36.813 | 27.209 | 1.319 [1.285, 1.353] |
| 1 | upload | 3 | 640x480 | scalar | 1.860 | 1.869 | 0.984 [0.930, 1.037] |
| 1 | readback | 3 | 640x480 | scalar | 7.359 | 7.215 | 1.030 [1.000, 1.057] |
| 1 | upload | 3 | 1920x1080 | threaded | 7.391 | 6.128 | 1.211 [1.171, 1.251] |
| 1 | readback | 3 | 1920x1080 | threaded | 14.428 | 12.132 | 1.210 [1.174, 1.247] |
| 1 | upload | 3 | 3840x2160 | threaded | 25.945 | 19.214 | 1.270 [1.203, 1.333] |
| 1 | readback | 3 | 3840x2160 | threaded | 40.594 | 29.263 | 1.409 [1.364, 1.454] |
| 1 | upload | 4 | 640x480 | scalar | 1.805 | 1.828 | 0.988 [0.948, 1.028] |
| 1 | readback | 4 | 640x480 | scalar | 8.299 | 8.077 | 1.019 [0.867, 1.166] |
| 1 | upload | 4 | 1920x1080 | scalar | 6.285 | 6.498 | 0.977 [0.913, 1.049] |
| 1 | readback | 4 | 1920x1080 | scalar | 14.945 | 12.191 | 1.206 [1.044, 1.392] |
| 1 | upload | 4 | 3840x2160 | scalar | 21.728 | 21.802 | 1.035 [0.994, 1.087] |
| 1 | readback | 4 | 3840x2160 | scalar | 37.483 | 31.054 | 1.152 [1.061, 1.249] |
| 2 | upload | 1 | 640x480 | scalar | 0.652 | 0.655 | 1.013 [0.991, 1.039] |
| 2 | readback | 1 | 640x480 | scalar | 1.581 | 1.567 | 1.020 [1.002, 1.038] |
| 2 | upload | 1 | 1920x1080 | threaded | 3.530 | 3.271 | 1.209 [1.112, 1.345] |
| 2 | readback | 1 | 1920x1080 | threaded | 5.975 | 5.617 | 1.045 [0.925, 1.188] |
| 2 | upload | 1 | 3840x2160 | threaded | 13.609 | 10.533 | 1.376 [1.277, 1.509] |
| 2 | readback | 1 | 3840x2160 | threaded | 25.286 | 20.857 | 1.146 [1.020, 1.280] |
| 2 | upload | 2 | 640x480 | scalar | 1.750 | 1.708 | 1.019 [0.975, 1.062] |
| 2 | readback | 2 | 640x480 | scalar | 6.864 | 6.596 | 1.067 [0.995, 1.160] |
| 2 | upload | 2 | 1920x1080 | threaded | 6.466 | 6.028 | 1.074 [0.984, 1.169] |
| 2 | readback | 2 | 1920x1080 | threaded | 12.546 | 11.864 | 1.136 [0.963, 1.345] |
| 2 | upload | 2 | 3840x2160 | threaded | 21.676 | 18.264 | 1.182 [1.113, 1.265] |
| 2 | readback | 2 | 3840x2160 | threaded | 26.261 | 20.952 | 1.173 [1.068, 1.289] |
| 2 | upload | 3 | 640x480 | scalar | 1.390 | 1.358 | 0.996 [0.926, 1.052] |
| 2 | readback | 3 | 640x480 | scalar | 6.211 | 6.125 | 1.036 [1.015, 1.059] |
| 2 | upload | 3 | 1920x1080 | threaded | 6.984 | 6.281 | 1.066 [0.962, 1.170] |
| 2 | readback | 3 | 1920x1080 | threaded | 13.132 | 13.218 | 0.982 [0.850, 1.123] |
| 2 | upload | 3 | 3840x2160 | threaded | 24.526 | 18.903 | 1.341 [1.294, 1.418] |
| 2 | readback | 3 | 3840x2160 | threaded | 36.445 | 26.897 | 1.321 [1.268, 1.359] |
| 2 | upload | 4 | 640x480 | scalar | 1.714 | 1.751 | 1.016 [0.963, 1.076] |
| 2 | readback | 4 | 640x480 | scalar | 6.992 | 6.732 | 1.048 [0.974, 1.123] |
| 2 | upload | 4 | 1920x1080 | scalar | 6.485 | 6.593 | 1.013 [0.972, 1.067] |
| 2 | readback | 4 | 1920x1080 | scalar | 13.897 | 12.760 | 1.131 [1.076, 1.197] |
| 2 | upload | 4 | 3840x2160 | scalar | 22.452 | 22.054 | 1.042 [0.993, 1.112] |
| 2 | readback | 4 | 3840x2160 | scalar | 36.376 | 31.822 | 1.171 [1.124, 1.228] |

For 3840x2160 with 1..=3 channels, paired completed GPU-transfer ratios
range from 1.146 to 1.409 across uploads/readbacks and both surveys (about
13–29% less time from paired ratios). Every corresponding interval is above
one. Their host-transform ratios range from 2.088 to 3.828. These observations
support this measured large strided/ROI workload family on the measured host.
They do not establish universal transfer throughput or GPU-kernel acceleration.

At 1920x1080, every selected CPU transform interval is above one, but several
completed GPU intervals cross one in the second survey; the three-channel
readback ratio is 0.982 [0.850, 1.123]. No completed-transfer improvement is
claimed for all conditions at that size. Small upload ratios also remain
uncertain. Four-channel unpacking uses scalar bulk row copies, not worker
acceleration. Four-channel 1920x1080 CPU packing has an adverse second-survey
ratio of 0.964 [0.940, 0.989]; the corresponding completed upload ratios have
intervals crossing one in both surveys. This control is retained rather than
hidden or credited as a gain. No completed GPU-transfer interval is entirely
below one in either survey.

The full sample summary is
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/image-transfer-rows-measurements.json`.
All scalar controls, noisy cases and adverse host samples are retained there
and in the tables. Neither transfer direction receives a blanket speedup claim.
