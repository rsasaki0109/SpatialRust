# Explicit GPU GICP correspondences

Date: 2026-10-12. Named slice: GICP correspondence reuse and complete covariance
transfer receipts. Builds on the reusable GPU KD-tree from PR #164.

## Execution and compatibility

`GicpRegistration::align_with_gpu_correspondences` under `register-gicp-gpu`
accepts the caller's runtime, uploads the CPU-built target KD-tree once and
reuses it for all update/fitness queries. Target construction, source transforms,
ordered source-index Hessian/gradient accumulation, normal-equation solve,
pose composition and Euclidean f64 fitness sums remain CPU.

GICP keeps its original unbounded `nearest_one` matching followed by the host
`distance_squared > squared_gate` rejection. The GPU uses the existing inclusive
nearest kernel with positive-infinity bound; for finite target/query coordinates
this retains the same near-first first-encountered ties, including infinite
squared distances. Host gate behavior is retained even for zero/negative/NaN/
infinite configured distances. Config validation and ordinary CPU APIs are
unchanged. The explicit GPU path additionally rejects nonfinite input or
transformed XYZ instead of using CPU nearest_one's nonfinite behavior. GPU f32
rounding/underflow can differ near boundaries on other backends; tested parity
is not a proof for every floating-point input.

Covariance remains CPU k-NN when `covariance_radius` is None. When configured,
the explicit path runs radius-GPU covariance on the provided runtime and
propagates errors, with no hidden shared runtime or automatic CPU fallback.
This composes the already-existing device covariance computation with the new
correspondence route; it adds no new covariance shader. Legacy native alignment
retains its shared-runtime covariance fallback. That legacy shared-runtime
helper is native-only; WASM legacy alignment uses CPU covariance. The explicit
GPU API compiles for WASM, but browser execution is not verified.

`GpuGicpDiagnostics` returns the ordinary registration result, explicit payload
transfers, completed correspondence batch count and covariance batch count
(0 or 2). The number of nearest batches depends on the stopping path: an update
stopping on transform magnitude skips its per-iteration fitness query; final
fitness is always evaluated, including for zero iterations. Source indices and
result order survive omitted/gated correspondences and unequal point counts.

The new `estimate_plane_covariances_grid_gpu_with_receipt` counts every XYZ,
sorted-index and grid-offset input upload, plus the complete padded output
readback. Existing covariance/normal APIs retain their values and signatures;
the shared executor now computes the receipt from actual input/output lengths.
Counting-sort grid construction is still CPU. For N points and C cells,
H2D = `16*N + 4*(C+1)`, D2H = `32*N` for covariance. The latter includes the
padding in two vec4 rows per point, although the returned covariance contains
only six f32 values. Uniform control data is excluded, as in other GPU payload
receipts. Nearest matching adds the one-time `16*T + 32*tree_nodes` target upload,
`16*S` query upload and `8*S` result readback per batch (source S, target T).
These are actual repeated transfers; targets are not assumed shared between
the KD-tree and the covariance SoA buffers.

## Validation

Three explicit native GICP tests compare exact pose/fitness/iterations/
convergence with legacy alignment for rotated and translated box corners,
nonidentity initial guesses, three stopping paths, CPU and radius-GPU
covariance, unequal clouds and gated outliers. Zero iterations and unusual
host gates retain behavior. Empty inputs and invalid neighbor/minimum counts
retain errors; nonfinite XYZ/transforms and invalid explicit GPU radii reject.
An explicit covariance receipt test checks exact values against the original
API, independently known cell counts, empty/mismatch behavior and padded
readback bytes. A facade test checks composed GPU covariance/correspondence
receipts. These adapter-dependent integration tests are ignored by default and
run locally with `--include-ignored`.

Final validation: 35 registration unit tests passed (1 ignored), all GPU
features 189 tests passed (22 ignored), the three new explicit GICP tests,
covariance receipt test, facade test and three GPU ICP regression tests passed.
All-feature/all-target GPU/registration clippy and CPU-only GICP clippy passed
with warnings denied. Minimal GPU and GPU-GICP WASM builds, formatting and diff
checks passed. A source audit against parent f1e70db confirms the unchanged
covariance shader, 14 math/covariance helpers and ordered Hessian accumulation;
only matching execution and receipt boundaries are changed.

## Completed GICP measurements

Native wgpu Vulkan discrete-class adapter, release build, one CPU worker.
No adapter model is recorded. Two isolated surveys each contain 20 alternating
legacy/explicit-GPU pairs per condition, after one legacy reference and three
checked explicit-GPU warmups. No simultaneous local builds/tests ran during
measurement. Output checks and result drop are outside timing. Each timed API
includes CPU tree construction, target upload, both covariance computations,
all correspondence/fitness batches, query uploads/readbacks, CPU reductions,
pose solves, transforms and returned diagnostics.

Both paths align a half-unit lattice with dyadic translation, 3 forced updates,
6 covariance neighbors and correspondence distance 1/4. For radius None the
legacy baseline is fully CPU and the explicit path changes matching only.
For radius 0.625 BOTH paths already use radius-GPU covariance; the baseline
keeps CPU matching while the explicit path adds GPU matching. This is not a
comparison of CPU k-NN covariance against GPU radius covariance. Those methods
use different neighborhoods and numerical precision, so their timings cannot
be combined into one speedup claim.

The table reports completion medians in ms and geometric mean paired
legacy/explicit-GPU ratios; above one favors the explicit GPU path. Brackets
are separate 95% paired log-ratio bootstrap intervals, 10,000 resamples, seed
20261012 plus record index. These synthetic geometry/count/iteration conditions
do not establish an automatic crossover or performance on real sensor scans.

| Survey | Points | Covariance in BOTH paths | Legacy ms | Explicit GPU ms | Legacy/GPU [95% interval] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 257 | CPU k-NN | 0.575 | 6.846 | 0.083 [0.082, 0.084] |
| 1 | 257 | radius GPU (0.625) | 2.340 | 9.006 | 0.260 [0.254, 0.267] |
| 1 | 4,097 | CPU k-NN | 11.756 | 15.297 | 0.765 [0.753, 0.778] |
| 1 | 4,097 | radius GPU (0.625) | 8.567 | 11.410 | 0.750 [0.720, 0.779] |
| 1 | 16,384 | CPU k-NN | 59.519 | 53.629 | 1.122 [1.063, 1.184] |
| 1 | 16,384 | radius GPU (0.625) | 36.138 | 35.449 | 1.028 [1.003, 1.056] |
| 1 | 65,537 | CPU k-NN | 282.361 | 225.452 | 1.235 [1.193, 1.275] |
| 1 | 65,537 | radius GPU (0.625) | 182.970 | 120.170 | 1.588 [1.476, 1.708] |
| 2 | 257 | CPU k-NN | 0.615 | 7.375 | 0.083 [0.081, 0.085] |
| 2 | 257 | radius GPU (0.625) | 2.449 | 9.281 | 0.265 [0.259, 0.271] |
| 2 | 4,097 | CPU k-NN | 13.577 | 16.596 | 0.801 [0.758, 0.847] |
| 2 | 4,097 | radius GPU (0.625) | 9.410 | 12.436 | 0.765 [0.740, 0.792] |
| 2 | 16,384 | CPU k-NN | 85.404 | 67.599 | 1.192 [1.070, 1.341] |
| 2 | 16,384 | radius GPU (0.625) | 43.675 | 37.402 | 1.196 [1.132, 1.269] |
| 2 | 65,537 | CPU k-NN | 312.784 | 258.591 | 1.198 [1.129, 1.270] |
| 2 | 65,537 | radius GPU (0.625) | 189.886 | 100.458 | 1.878 [1.774, 1.993] |

Small 257/4,097-point conditions favor legacy matching in both surveys.
The 65,537-point cases improve in both covariance modes and both surveys;
CPU covariance and host solving limit the full-loop gain when matching is the
only device phase. The 16,384-point radius-GPU case has only a small first-survey gain close to
unity; it is retained alongside the larger second-survey gain. All adverse
cases remain visible, and timings vary between surveys.

Reproduce each survey separately:

```powershell
cargo test -p spatialrust-registration --features register-gicp-gpu --release --test gpu_gicp_benchmark -- --ignored --nocapture
```

Local raw measurements/statistics stay under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation`. The public benchmark
prints backend/device class only. ICP and GICP matching are now GPU-capable;
feature neighborhoods, FPFH/ISS/MLS device computation and device registration
reductions/pose updates remain major follow-up phases.
