# Direct SoA feature and MLS input staging

Base: merged PR #195, `7a11731181ec5c08605a43a219d24dcf7fc7e132`.

`GpuFpfhInput::upload`, `GpuIssInput::upload` and `GpuMlsInput::upload` now
pack original-order queries directly from the CPU KD-tree's borrowed SoA
coordinates with the existing private `GpuQueryBatch::upload_soa` boundary.
FPFH sample records zip those same coordinates with the original unit normals.
The intermediate `Vec<[f32;3]>` is removed in each constructor.

This removes one allocation/requested payload and copy of 12*N host bytes per
upload: 3,145,740 bytes (about 3 MiB) at 262,145 points. It is a statement about
the eliminated vector, not measured process RSS, allocator metadata or driver
peak memory. The caller's CPU tree/normals, padded 16*N query staging, 32*N FPFH
sample staging and tree-order point/node staging remain. Empty inputs allocate
no device payload. No host position vector is retained by the new constructors.

Finite XYZ, unit normals, normal-count checks, f64 axis extent arithmetic,
capability/count limits and validation preceding device allocation retain their
existing order. The SoA query packer already checks shape/finite coordinates;
the validated tree supplies equal-length immutable axes. Packing only copies
f32 components: signed zeros and subnormal bits stay exact, and both query and
sample padding remain positive zero. No normalization, arithmetic conversion
or attribute gathering is added at this boundary.

Public APIs, runtime/input identities, radius graph ownership, GPU shaders,
status handling, source ordering, payload bytes and explicit readbacks are
unchanged. ISS/MLS continue to share query positions with covariance/fitting;
FPFH retains separate sample/query buffers. Nonempty uploads remain
64*N + 32*M bytes for FPFH and 32*N + 32*M for ISS/MLS (M tree nodes).
There is no new GPU pass, payload copy, host wait, CPU fallback or implicit
CPU/GPU placement change. Staging remains serial; this slice removes a copy
rather than adding CPU workers. No dependency, unsafe boundary or core change
is introduced. The GPU algorithms were already implemented before this slice.

## Source and frozen reference

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/iss.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mls.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh_soa_upload_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/iss_soa_upload_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mls_soa_upload_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/feature_soa_upload_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/feature_soa_upload_benchmark.rs`

All three frozen constructor bodies retain the merged parent logic after
renaming `Self`/the function and normalizing formatting/optional trailing
commas. Capability plans, validation math and all six feature/MLS shader files
are unchanged. References and probes are native test-only; no reference public
API, COPY_SRC input usage or additional production pipeline is introduced.

## Correctness and compatibility

Three new native tests pass in 5.24 s:

- 24 input conditions through 262,145 points compare **26,236,560 actual GPU
  32-bit words** with frozen inputs and independently expected original-order
  coordinates/normals/padding. Fixtures include grid/noise, signed zeros,
  positive/negative subnormal coordinates, large finite coincident points and
  empty inputs. A read-only integer compute probe observes STORAGE-only inputs;
  its explicit readback is test code and is not added to production transfers.
- 60 FPFH/ISS/MLS evaluations compare **51,432 output/status words** with the
  frozen path: descriptor bins, flags/eigenvalue scores, three MLS orders/modes,
  valid and overflow/numerical-invalid outcomes. Uploaded inputs survive CPU
  tree drops, and outputs survive input drops. Foreign runtimes and
  different-input neighborhoods reject.
  Explicit downloads, errors and transfer receipts agree.
- 35 rejected upload cases match the frozen error strings: nonfinite values
  in each axis, extent overflow, wrong normal count and invalid unit normals.

Existing native FPFH/ISS/MLS/nearest subsets pass **24 tests** (3/4/4/13),
including plan/device boundaries, malformed/overflow rows, analytic feature and
fit checks and SoA neighborhood ordering. GPU all-feature/all-target Clippy
passes with `-D warnings` (5.88 s). All three public-facade GPU smoke targets
pass; GPU-disabled compilation and the WASM facade with
`register-fpfh-gpu,feature-iss-gpu,filter-mls-gpu` compile. Formatting/diff and
publication privacy checks are completed before committing the slice.

## Measurement boundaries

Rust 1.98.1 release executable, Vulkan, discrete device type and 12 available
CPU workers. The constructors themselves remain serial. CPU trees, deterministic
positions and normals, runtime construction and fixtures are outside timing.
Eight conditions each use three warm rounds and twenty accepted observations
per route. Six routes rotate cyclically: frozen/current FPFH, frozen/current ISS,
frozen/current MLS. Upload construction, native `wait_idle`, count/receipt capture
and input destruction return are timed; assertions occur after timers.

These are completed **input-construction** API observations. Neighborhood
search, descriptor/keypoint/fitting passes, output readback, CPU tree building,
attribute reconstruction and whole application frame latency are excluded.
Driver-internal deferred cleanup is not separately measured. Raw uploaded bits
and algorithm consumption are verified in correctness tests, not read back in
the measured constructors. Receipt/count assertions run for every sample.
This is not a GPU-kernel or complete-algorithm speedup claim.

Two surveys ran sequentially (24.91 s and 16.82 s); each accepted attempt 1 with
zero observed named build/test/benchmark overlap and empty native stderr.
The controller requires three quiet observations five seconds apart and checks
other named work between 500 ms child waits. If overlap appears it rejects the
whole attempt and may stop only its own child. This bounded process guard does
not prove complete machine idleness. Statistical analysis starts after both
surveys finish. Source/executable hashes are captured before measurement and
rechecked afterward; production and measurement sources do not change.

Profiles 0/1/2 are grid/noise, alternating signed-zero coordinates, and signed
subnormal coordinates. The same CPU tree and unit-normal arrays feed both paths.
All 16 condition records / 48 paired ratios are retained below. Ratios are
old/new geometric means, with percentile 95% intervals from 10,000 resamples of
twenty paired log ratios (endpoints 249/9749). Deterministic seed:
`2026101408 + 24*(survey-1) + 3*condition_index + input_index`, where FPFH/ISS/MLS
indices are 0/1/2. Intervals are per condition, not a deployment guarantee.

## Interpretation and adverse cases

At 262,145 grid/noise points ISS upload old/new ratios are
1.125148 [1.062351, 1.188314] and 1.121293 [1.073928, 1.170662], about 11% less
input-construction time. At 131,073 grid/noise points FPFH ratios are
1.077358/1.093399 and MLS 1.122007/1.108451; their intervals exceed one in both
surveys. These selected improvements reproduce, but many other intervals cross
one and overall times vary substantially between surveys/routes.

Survey 1 FPFH subnormal-coordinate ratio is adverse:
0.731383 [0.505725, 0.992838]. Its survey 2 interval crosses one. Several small
and duplicate-coordinate results also have ratios below one with inconclusive
intervals. Their full samples/results are retained, and the cause of the
variation is not established. No universal speedup, small-input threshold or
whole FPFH/ISS/MLS latency improvement is asserted. The removed 12*N allocation
is the deterministic benefit across admitted inputs; device payload is unchanged.
Medians below describe separate distributions and are not the paired estimator.

## Complete measurements

| Survey | Points / profile | Input | Previous ms | Direct SoA ms | Paired ratio [95%] | Removed temporary bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 / 0 | fpfh | 0.024 | 0.023 | 0.977x (0.795-1.210) | 12 |
| 1 | 1 / 0 | iss | 0.020 | 0.020 | 1.009x (0.899-1.141) | 12 |
| 1 | 1 / 0 | mls | 0.022 | 0.019 | 1.162x (0.982-1.385) | 12 |
| 1 | 257 / 0 | fpfh | 0.037 | 0.037 | 0.924x (0.789-1.057) | 3084 |
| 1 | 257 / 0 | iss | 0.027 | 0.027 | 0.866x (0.726-1.015) | 3084 |
| 1 | 257 / 0 | mls | 0.028 | 0.027 | 0.874x (0.723-1.039) | 3084 |
| 1 | 4097 / 0 | fpfh | 0.126 | 0.115 | 1.020x (0.857-1.194) | 49164 |
| 1 | 4097 / 0 | iss | 0.089 | 0.078 | 1.145x (1.030-1.325) | 49164 |
| 1 | 4097 / 0 | mls | 0.080 | 0.086 | 0.940x (0.756-1.169) | 49164 |
| 1 | 65537 / 0 | fpfh | 3.016 | 2.835 | 1.172x (1.027-1.482) | 786444 |
| 1 | 65537 / 0 | iss | 1.862 | 1.799 | 1.019x (0.950-1.086) | 786444 |
| 1 | 65537 / 0 | mls | 1.825 | 1.732 | 0.902x (0.639-1.097) | 786444 |
| 1 | 131073 / 0 | fpfh | 6.376 | 6.086 | 1.077x (1.032-1.126) | 1572876 |
| 1 | 131073 / 0 | iss | 4.141 | 3.580 | 0.964x (0.636-1.215) | 1572876 |
| 1 | 131073 / 0 | mls | 4.106 | 3.679 | 1.122x (1.058-1.188) | 1572876 |
| 1 | 262145 / 0 | fpfh | 13.314 | 11.932 | 1.030x (0.828-1.183) | 3145740 |
| 1 | 262145 / 0 | iss | 8.271 | 7.487 | 1.125x (1.062-1.188) | 3145740 |
| 1 | 262145 / 0 | mls | 8.337 | 7.627 | 0.950x (0.647-1.174) | 3145740 |
| 1 | 262145 / 1 | fpfh | 60.634 | 57.882 | 0.820x (0.615-1.038) | 3145740 |
| 1 | 262145 / 1 | iss | 49.771 | 48.597 | 1.093x (0.730-1.589) | 3145740 |
| 1 | 262145 / 1 | mls | 45.664 | 47.347 | 0.955x (0.614-1.469) | 3145740 |
| 1 | 262145 / 2 | fpfh | 58.765 | 59.033 | 0.731x (0.506-0.993) | 3145740 |
| 1 | 262145 / 2 | iss | 49.810 | 45.881 | 1.220x (0.913-1.642) | 3145740 |
| 1 | 262145 / 2 | mls | 46.772 | 48.255 | 0.917x (0.603-1.347) | 3145740 |
| 2 | 1 / 0 | fpfh | 0.018 | 0.015 | 1.189x (0.904-1.539) | 12 |
| 2 | 1 / 0 | iss | 0.013 | 0.013 | 0.914x (0.790-1.047) | 12 |
| 2 | 1 / 0 | mls | 0.012 | 0.012 | 1.078x (0.871-1.297) | 12 |
| 2 | 257 / 0 | fpfh | 0.033 | 0.033 | 0.943x (0.808-1.070) | 3084 |
| 2 | 257 / 0 | iss | 0.023 | 0.025 | 0.949x (0.850-1.043) | 3084 |
| 2 | 257 / 0 | mls | 0.025 | 0.025 | 0.834x (0.674-1.016) | 3084 |
| 2 | 4097 / 0 | fpfh | 0.108 | 0.105 | 1.008x (0.867-1.167) | 49164 |
| 2 | 4097 / 0 | iss | 0.071 | 0.068 | 1.076x (0.978-1.244) | 49164 |
| 2 | 4097 / 0 | mls | 0.074 | 0.069 | 1.013x (0.787-1.269) | 49164 |
| 2 | 65537 / 0 | fpfh | 3.414 | 3.153 | 1.151x (0.991-1.438) | 786444 |
| 2 | 65537 / 0 | iss | 1.932 | 2.016 | 0.982x (0.894-1.074) | 786444 |
| 2 | 65537 / 0 | mls | 2.015 | 1.759 | 0.919x (0.675-1.119) | 786444 |
| 2 | 131073 / 0 | fpfh | 6.867 | 6.327 | 1.093x (1.040-1.154) | 1572876 |
| 2 | 131073 / 0 | iss | 4.237 | 3.664 | 1.031x (0.768-1.235) | 1572876 |
| 2 | 131073 / 0 | mls | 4.084 | 3.781 | 1.108x (1.059-1.161) | 1572876 |
| 2 | 262145 / 0 | fpfh | 13.942 | 12.943 | 0.990x (0.832-1.106) | 3145740 |
| 2 | 262145 / 0 | iss | 8.816 | 7.934 | 1.121x (1.074-1.171) | 3145740 |
| 2 | 262145 / 0 | mls | 9.131 | 8.176 | 1.024x (0.812-1.172) | 3145740 |
| 2 | 262145 / 1 | fpfh | 36.045 | 34.114 | 1.003x (0.876-1.131) | 3145740 |
| 2 | 262145 / 1 | iss | 26.539 | 23.003 | 1.032x (0.845-1.191) | 3145740 |
| 2 | 262145 / 1 | mls | 24.922 | 22.854 | 0.957x (0.755-1.125) | 3145740 |
| 2 | 262145 / 2 | fpfh | 38.983 | 37.690 | 0.946x (0.789-1.086) | 3145740 |
| 2 | 262145 / 2 | iss | 28.070 | 24.945 | 1.118x (1.046-1.190) | 3145740 |
| 2 | 262145 / 2 | mls | 28.831 | 24.425 | 1.054x (0.903-1.179) | 3145740 |

## Execution receipts

| File | SHA-256 |
| --- | --- |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh.rs` | `4a9eae01db66b5af37440bcb7e7a7fdab388b396c7b7dad2bf2849ab6936aa26` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/iss.rs` | `a28fd46234247725f6fcce9d875d80cab6a57d66f4105919dcc46b15f73e2dd3` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mls.rs` | `59e052e1ea56e4130609f8ca9b643ca505fa85feb98e8ac319a8567930259232` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mod.rs` | `f5c7398adf869eb2ab055fe5a9a461f224f9f4e75206f9f2cb6ef03d32e30ff7` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs` | `641dc65660280a161ed23fdd108ec65901088fb39e1c5cb0a71181f1ff9f4780` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/feature_soa_upload_tests.rs` | `e306005e8ec6803946c7d255576627b471d81886af7a8c6060b9cc575c878d8c` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/feature_soa_upload_benchmark.rs` | `d96794e13087ebf9b360541397b56eb3f1f35d24abee54267c1da642816b32eb` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh_soa_upload_reference.rs` | `c6988bbe26c82578e9362fbffd2869afb2fef181c47590231642c6af9f16b252` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/iss_soa_upload_reference.rs` | `93da8fc50cf4f269446062599a0895843ffb02ef15b2e752ce02309519fb5821` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mls_soa_upload_reference.rs` | `39dd555e514c9db06b62d8ca13cc421c09d9225ed1d8e02b003cf197685fd266` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-62f845495446a7e6.exe` | `ddc71080ac6a5e2345bb94f87338439fe8a891928d0654f45c94f5c04e3c4ca1` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/mls_frames.wgsl` | `0dc609162947c6f9b7561885008faf7d245f5ca7ac5de1997a89af3d35409a2f` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/mls_fit.wgsl` | `6c6171c47989461d12e0cdb5dd18880851bfc3d8167eb2d303b3851ece4edaf7` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/mls_constant.wgsl` | `346c981eba627ec854ef8819fc626e7cc1391c939e21f3ef7ee1f0a64517126c` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/mls_common.wgsl` | `c7d07d7e92ba82f520475c030c59685c5d78558619c7cf38bccdfbf5ca688d51` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/iss.wgsl` | `ee2994ded65c53900371cda1fbfee291e249e9a31e6983201b38385d5ceeeed1` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/fpfh.wgsl` | `299d5f6f4bead33dfa7d6a8ce196883482ee028b777c93c55da650a9fb3fe672` |

These are worktree-byte hashes. Local raw records, controllers, complete logs
and bootstrap results remain under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with the
`feature-soa-upload` prefix.
