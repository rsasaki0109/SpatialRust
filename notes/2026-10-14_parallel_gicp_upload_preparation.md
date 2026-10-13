# Bounded GICP covariance upload preparation (2026-10-14)

## Result and scope

Explicit `GpuGicpInput::upload` now validates and packs host-provided covariance
rows together. Native large batches use bounded scoped CPU workers. This is CPU
preparation for an explicit GPU upload; GPU shaders and algorithms are unchanged.

On the selected large fixtures, completed upload plus one GPU equation evaluation
improved in both isolated surveys: 65,537 source/target rows by 1.254x / 1.277x,
262,145 sources with one target by 1.414x / 1.466x, and 1,048,577 sources with one
target by 1.519x / 1.498x. For the last condition, full medians changed from
83.182 / 83.254 ms to 55.280 / 55.300 ms. These are synthetic, provided-covariance
calls with a prebuilt target tree, not full registration speedups.

The actual three-update public registration API did **not** establish a repeated
speedup in any tested condition. Both paired confidence intervals include one
for the 65,537-point CPU k-NN route; its second survey varies substantially.
An adverse first-survey GPU-grid ratio, 0.993x [0.987, 0.997], remains in the
results; its second survey includes one. GPU-generated covariance constructors
bypass the changed staging helper. Do not attribute their mixed timings to a
new device kernel or claim a general registration improvement.

## Implementation and contracts

Production sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/gicp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/gicp_upload_preparation.rs`

Before staging, the existing upload checks matching source/target covariance
counts and device/byte limits. Source and target rows retain the original
`inverse_trace` function: exact packed-f32 values promoted to f64, Cholesky
pivots and inverse-trace arithmetic, positive finite trace rejection, and the
same final finite normal f32 weight-scale requirement. Only target traces
contribute to the bound. Their positive finite maximum is exact across chunk
boundaries; no floating sum or changed per-row arithmetic is introduced.

Packing keeps source rows before target rows and the original six f32 component
bits, followed by two positive-zero padding words. Native worker count is
`min(available_parallelism, max(1, floor((S+T)/16_384)))`, using the existing
search staging policy. Fewer than 32,768 combined rows, one available worker
and WASM use serial preparation. Parallel output slices are disjoint and scoped;
all workers finish before any GPU input allocation, including on validation
failure. Existing errors are uniform and unchanged. Public APIs remain safe;
no unsafe code, dependency or public configuration is added.

The serial path uses a capacity-reserved vector and pushes validated rows. The
parallel path safely zero-initializes the same 32*(S+T)-byte vector, then fills
its disjoint slices. The packed vector now remains live through position/tree
upload, so it can overlap other host staging temporaries and increase peak host
memory compared with the previous allocation order. Invalid geometry can also
allocate this host vector before the existing position validation fails. Device
limits still bound its size. This change does not claim lower peak host memory.

The H2D receipt remains `48*S + 64*T + 32*M` bytes (M: target tree nodes).
Immutable source/target/tree, five equation passes, working scratch, numeric
status, fixed owned 256-byte equations and explicit readbacks remain unchanged.
CPU solving, pose composition and stopping remain unchanged. Ordinary CPU GICP
and GPU-generated k-NN/radius/grid covariance entry points retain their placement.

## Correctness and build validation

Test sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/gicp_upload_preparation_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/gicp_upload_preparation_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_gicp_reuse_query_benchmark.rs`

The frozen serial validation/packing and upload references come from parent
`b56246c6721dcdb18fa72668168415b09f00e510` (merged PR #188). `inverse_trace`, ICP
position preparation and GPU shader math are unchanged and shared with that
reference; existing independent f64 equation and spectral tests guard that math.

- 9,193,125 packed row comparisons plus 393,224 rows from four concurrent
  preparations match old f32 words, padding and weight-scale bits exactly.
  Sizes include 0, 1, 257, 16,383/16,384, 32,767/32,768/32,769, 65,537 and
  262,145; targets include one, 257 and equal counts; requested workers 0/1/2/6
  and public staging are checked. Covariances vary diagonal/non-diagonal SPD,
  signed zero and scales from 1e-8 through 1e8. Pure worker-cap boundaries
  include `usize::MAX` without allocation.
- Analytic diagonal inverse trace 1.75 and exact signed-zero/padding bits;
  NaN/infinity in every component, zero and indefinite matrices in source and
  target at chunk boundaries; empty target and subnormal/extreme scale outcomes
  match the previous errors.
- 42 uploaded equation conditions match all 2,688 raw u32 words, decoded fields,
  weight bits and full transfer receipts. Queued results survive input drop.
  Includes empty/single input, worker boundaries, 65,537/262,145 sources,
  one/257 targets and three inclusive distance gates.
- 24 native GICP kernel/covariance/status tests passed, including the previous
  747 frozen equation conditions, 90 resident covariance/status conditions,
  independent f64 equations, limits, concurrency and ownership. Maximum
  normalized independent f64 equation error remains 1.0000217020915932e-6.
- Four registration integration targets passed: 16 test executions (shared
  reference tests repeat) cover 594 unique accuracy conditions and 544,104
  points. Maximum normalized CPU pose-coordinate error remains 9.536743e-7;
  existing tolerances were not widened.
- GPU all-feature/all-target Clippy, registration GPU/CPU-only Clippy, GPU
  CPU-only check, native minimal facade, GPU WASM and register-gicp-gpu facade
  WASM checks, rustfmt and diff checks passed.

Exact native test command, from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --no-default-features --features gpu-wgpu --release --lib gicp_reduction -- --include-ignored --nocapture --test-threads=1 --skip benchmark::
cargo test -p spatialrust-registration --no-default-features --features register-gicp-gpu --release --test gpu_gicp_reduction --test gpu_gicp_resident_covariances --test gpu_gicp_resident_radius_covariances --test gpu_gicp_resident_grid_covariances -- --include-ignored --nocapture --test-threads=1
```

## Measurement conditions and provenance

Native release tests, Vulkan discrete GPU backend, 12 available CPU workers,
default 64 MiB private scratch budget. Completed wall-clock timings use explicit
result download/wait; no process startup, runtime initialization or simultaneous
old/new computation enters samples. The local adapter name is omitted.

A process monitor rejects overlap with foreign Cargo, compiler, formatter,
linker and named benchmark/test executables. Three quiet observations precede
measurement. No build or statistical analysis ran during accepted measurements;
foreign work was never stopped. This monitor does not prove the whole machine
idle or eliminate operating-system/background variability.

For primitive calls, both entire surveys were accepted on their first attempt
(test durations 8.33 / 8.37 seconds). Seven fixtures use the same varied,
well-conditioned SPD rows and deterministic coincident/repeated XYZ support.
Each condition uses three warmups and 20 paired samples for four interleaved
routes: old/new host preparation and old/new completed upload/evaluation.
Host timing ends when the helper returns its owned packed vector; comparison
and owner drop follow outside that timer. Full timing includes upload validation,
position/tree/covariance packing and transfer, equation submission, download/wait
and input/output owner drop. Caller points/covariances and CPU target-tree
construction are outside the full timer. Old upload preserves its original
order (validate, position upload, serial packing). GPU outputs and receipts are
asserted equal for every call. Source row profiles cover 257, 4,097, the serial
32,767 combined-row boundary, a two-worker batch, and three large batches.

For full registration, the identical pre-existing stdin/stdout benchmark was
built at the parent before production edits and saved, then rebuilt after them.
Two processes stay alive; the controller sends exactly one computation command
at a time. Each condition clears scratch, builds caller fixtures outside timing,
then checks the entire debug representation of pose, fitness, stopping, history,
covariance batches and transfer receipts across processes. Each timed actual API
call includes CPU target/source trees and covariance preparation, validation,
packing, transfer, four device evaluations/readbacks, three solves/updates and
owned cleanup. Config: k=6, distance=0.25, radius=0.8 for radius/grid routes,
three forced updates, zero transformation/fitness stopping thresholds, and
translation (0.03125, -0.015625, 0.0078125). Duplicate profiles have 17 extra
target points. All 640 accepted timed calls match complete diagnostics exactly.

The public controller independently monitors preparation, warmups and all 20
pairs for each condition. Five contaminated conditions were discarded in full:
survey 1's 257-point CPU route once; survey 2's 4,097-point GPU k-NN route once,
65,537-point CPU route once and 65,537-point GPU k-NN route twice. Each was
retried after a quiet window; accepted conditions report no monitored overlap.
Other accepted conditions were their first attempts. Discarded records remain
in the local logs and are excluded from the tables below.

Reported times are medians, whereas ratios are geometric means of **paired**
old/new times. Confidence intervals use 10,000 bootstrap resamples of the 20
paired log ratios, sorted endpoints 249/9749. Primitive seeds:
`20261013 + 14*(survey-1) + 2*condition_index + stage_index` (host=0/full=1);
public seeds: `20261013 + 8*(survey-1) + condition_index`. Intervals are unadjusted
for multiple conditions; medians need not divide to the paired ratio. Full phase
components add to each individual sample; component medians need not add to
the full median. Two surveys support only the listed workload conclusions.

Local executable provenance (SHA-256):

- Parent public benchmark: `8F9278D201A30849807ED79B6CB1FCCB7CA77015B2340F07FBE0E93E6A988E72`
- Identical public benchmark source: `34343FCC04D26E903FFA5B764BDF5C27EAEE8470D9CCB7416AD2539389F6C0D8`
- Current public benchmark: `4F342DB3284DC5BBA71E0A2D37D7F80DA87D214486F9A272A2D13A9E0830D14E`
- Native primitive/frozen benchmark: `F141FBF451927FC642D1B3F5B6DC7CA5AAB1CF03D98FB1C5D61494E7A21D7259`

Local raw arrays, stderr/stdout, overlap/retry receipts, controller and analysis
sources live under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:

- `parallel-gicp-upload-preparation-baseline.json`
- `run-parallel-gicp-upload-preparation-survey.ps1`
- `parallel-gicp-upload-preparation-benchmark-1.log` / `-2.log`
- `parallel-gicp-upload-preparation-measurements.json`
- `run-parallel-gicp-upload-registration-survey.ps1`
- `parallel-gicp-upload-registration-benchmark-1.json` / `-2.json`
- `parallel-gicp-upload-registration-measurements.json`
- `analyze-parallel-gicp-upload-preparation.py`
- `spatialrust_gicp_upload_previous.exe`
- `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-fafe8a8488d734a8.exe`
- `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/gpu_gicp_reuse_query_benchmark-2e751f82c032c4ce.exe`

These local executables/logs are untracked; the frozen Rust fixtures and complete
result tables remain in the repository. The benchmark source is ignored by the
ordinary test suite and requires the explicit native release test/controller.

## All primitive conditions

| Survey | Sources / targets | Workers | Stage | Old ms | New ms | Old/new paired ratio [95%] |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 257 / 257 | 1 | host | 0.016 | 0.015 | 1.104x (1.045-1.190) |
| 1 | 257 / 257 | 1 | full | 1.185 | 1.227 | 0.998x (0.965-1.029) |
| 1 | 4,097 / 4,097 | 1 | host | 0.257 | 0.251 | 1.026x (0.992-1.073) |
| 1 | 4,097 / 4,097 | 1 | full | 1.727 | 1.702 | 0.999x (0.959-1.044) |
| 1 | 16,383 / 16,384 | 1 | host | 1.104 | 1.050 | 1.027x (0.997-1.051) |
| 1 | 16,383 / 16,384 | 1 | full | 3.291 | 3.210 | 1.024x (0.995-1.051) |
| 1 | 32,768 / 257 | 2 | host | 1.065 | 1.001 | 1.069x (1.028-1.110) |
| 1 | 32,768 / 257 | 2 | full | 3.326 | 3.257 | 1.023x (0.996-1.049) |
| 1 | 65,537 / 65,537 | 8 | host | 4.473 | 2.100 | 2.083x (2.000-2.179) |
| 1 | 65,537 / 65,537 | 8 | full | 10.944 | 8.805 | 1.254x (1.231-1.286) |
| 1 | 262,145 / 1 | 12 | host | 8.963 | 2.971 | 3.024x (2.734-3.417) |
| 1 | 262,145 / 1 | 12 | full | 18.357 | 12.794 | 1.414x (1.352-1.468) |
| 1 | 1,048,577 / 1 | 12 | host | 35.568 | 7.721 | 4.544x (4.352-4.747) |
| 1 | 1,048,577 / 1 | 12 | full | 83.182 | 55.280 | 1.519x (1.477-1.558) |
| 2 | 257 / 257 | 1 | host | 0.015 | 0.014 | 1.041x (1.030-1.050) |
| 2 | 257 / 257 | 1 | full | 1.171 | 1.170 | 0.986x (0.957-1.014) |
| 2 | 4,097 / 4,097 | 1 | host | 0.241 | 0.232 | 1.019x (0.999-1.038) |
| 2 | 4,097 / 4,097 | 1 | full | 1.662 | 1.659 | 0.988x (0.941-1.032) |
| 2 | 16,383 / 16,384 | 1 | host | 1.256 | 1.208 | 1.018x (0.969-1.057) |
| 2 | 16,383 / 16,384 | 1 | full | 3.613 | 3.615 | 1.007x (0.966-1.054) |
| 2 | 32,768 / 257 | 2 | host | 1.252 | 1.116 | 1.111x (1.080-1.143) |
| 2 | 32,768 / 257 | 2 | full | 3.586 | 3.500 | 1.039x (1.007-1.071) |
| 2 | 65,537 / 65,537 | 8 | host | 4.732 | 2.116 | 2.023x (1.801-2.207) |
| 2 | 65,537 / 65,537 | 8 | full | 11.773 | 9.039 | 1.277x (1.219-1.327) |
| 2 | 262,145 / 1 | 12 | host | 8.519 | 2.848 | 2.942x (2.785-3.088) |
| 2 | 262,145 / 1 | 12 | full | 17.779 | 12.125 | 1.466x (1.436-1.498) |
| 2 | 1,048,577 / 1 | 12 | host | 37.504 | 8.301 | 4.524x (4.304-4.754) |
| 2 | 1,048,577 / 1 | 12 | full | 83.254 | 55.300 | 1.498x (1.462-1.535) |

## All full registration conditions

| Survey | Sources | Profile | Covariance route | Old ms | New ms | Old/new paired ratio [95%] |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 257 | lattice | CPU k-NN | 7.269 | 7.220 | 1.007x (0.983-1.033) |
| 1 | 4,097 | lattice | CPU k-NN | 18.777 | 18.742 | 1.002x (0.974-1.034) |
| 1 | 4,097 | duplicate/unequal | GPU k-NN | 15.857 | 15.827 | 1.003x (0.976-1.031) |
| 1 | 16,384 | lattice | GPU grid | 36.483 | 36.252 | 0.998x (0.985-1.010) |
| 1 | 65,537 | lattice | CPU k-NN | 139.600 | 139.038 | 0.995x (0.959-1.027) |
| 1 | 65,537 | lattice | GPU k-NN | 57.403 | 56.561 | 1.044x (0.994-1.123) |
| 1 | 65,537 | lattice | GPU radius | 76.245 | 77.335 | 1.014x (0.907-1.135) |
| 1 | 65,537 | duplicate/unequal | GPU grid | 84.061 | 84.569 | 0.993x (0.987-0.997) |
| 2 | 257 | lattice | CPU k-NN | 8.283 | 8.642 | 0.963x (0.842-1.087) |
| 2 | 4,097 | lattice | CPU k-NN | 19.341 | 19.296 | 1.004x (0.961-1.056) |
| 2 | 4,097 | duplicate/unequal | GPU k-NN | 17.979 | 17.039 | 1.050x (0.855-1.325) |
| 2 | 16,384 | lattice | GPU grid | 29.659 | 30.432 | 1.003x (0.749-1.282) |
| 2 | 65,537 | lattice | CPU k-NN | 435.719 | 295.483 | 1.042x (0.623-1.728) |
| 2 | 65,537 | lattice | GPU k-NN | 58.998 | 60.704 | 1.004x (0.893-1.129) |
| 2 | 65,537 | lattice | GPU radius | 74.354 | 73.670 | 0.946x (0.862-1.012) |
| 2 | 65,537 | duplicate/unequal | GPU grid | 84.698 | 84.607 | 1.000x (0.997-1.004) |

## Completed primitive phases

These CPU-visible phases place device execution and copies in the download/wait
phase; they do not separately establish shader, allocator or driver timing.
At 1,048,577 sources, upload medians decrease from 58.883 / 58.291 ms to
29.687 / 31.207 ms, while download/wait remains about 22 ms.

| Survey | Sources / targets | Variant | Upload ms | Submit ms | Download/wait ms | Drop ms | Full ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 257 / 257 | old | 0.053 | 0.133 | 0.996 | 0.004 | 1.185 |
| 1 | 257 / 257 | new | 0.053 | 0.139 | 1.022 | 0.004 | 1.227 |
| 1 | 4,097 / 4,097 | old | 0.418 | 0.171 | 1.116 | 0.005 | 1.727 |
| 1 | 4,097 / 4,097 | new | 0.431 | 0.161 | 1.104 | 0.006 | 1.702 |
| 1 | 16,383 / 16,384 | old | 1.667 | 0.181 | 1.375 | 0.005 | 3.291 |
| 1 | 16,383 / 16,384 | new | 1.653 | 0.166 | 1.396 | 0.004 | 3.210 |
| 1 | 32,768 / 257 | old | 1.595 | 0.182 | 1.563 | 0.004 | 3.326 |
| 1 | 32,768 / 257 | new | 1.501 | 0.168 | 1.544 | 0.005 | 3.257 |
| 1 | 65,537 / 65,537 | old | 7.803 | 0.293 | 2.910 | 0.006 | 10.944 |
| 1 | 65,537 / 65,537 | new | 5.638 | 0.258 | 2.899 | 0.006 | 8.805 |
| 1 | 262,145 / 1 | old | 14.063 | 0.239 | 4.130 | 0.007 | 18.357 |
| 1 | 262,145 / 1 | new | 8.444 | 0.219 | 4.124 | 0.007 | 12.794 |
| 1 | 1,048,577 / 1 | old | 58.883 | 0.518 | 21.724 | 1.081 | 83.182 |
| 1 | 1,048,577 / 1 | new | 29.687 | 0.500 | 21.991 | 1.108 | 55.280 |
| 2 | 257 / 257 | old | 0.050 | 0.130 | 0.992 | 0.004 | 1.171 |
| 2 | 257 / 257 | new | 0.049 | 0.124 | 0.994 | 0.004 | 1.170 |
| 2 | 4,097 / 4,097 | old | 0.400 | 0.157 | 1.092 | 0.005 | 1.662 |
| 2 | 4,097 / 4,097 | new | 0.408 | 0.150 | 1.111 | 0.005 | 1.659 |
| 2 | 16,383 / 16,384 | old | 1.969 | 0.198 | 1.387 | 0.005 | 3.613 |
| 2 | 16,383 / 16,384 | new | 1.990 | 0.221 | 1.384 | 0.006 | 3.615 |
| 2 | 32,768 / 257 | old | 1.800 | 0.199 | 1.584 | 0.006 | 3.586 |
| 2 | 32,768 / 257 | new | 1.721 | 0.188 | 1.586 | 0.005 | 3.500 |
| 2 | 65,537 / 65,537 | old | 8.615 | 0.278 | 2.884 | 0.007 | 11.773 |
| 2 | 65,537 / 65,537 | new | 5.888 | 0.276 | 2.910 | 0.007 | 9.039 |
| 2 | 262,145 / 1 | old | 13.377 | 0.243 | 4.131 | 0.007 | 17.779 |
| 2 | 262,145 / 1 | new | 7.864 | 0.196 | 4.129 | 0.006 | 12.125 |
| 2 | 1,048,577 / 1 | old | 58.291 | 0.493 | 22.570 | 1.106 | 83.254 |
| 2 | 1,048,577 / 1 | new | 31.207 | 0.493 | 22.728 | 1.085 | 55.300 |
