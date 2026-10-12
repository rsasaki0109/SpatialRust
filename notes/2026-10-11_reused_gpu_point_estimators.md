# Reused standalone GPU normal and covariance estimation

Workspace: `C:/Users/rsasa/Workspace/SpatialRust`.
Parent main: `eeab49f65fa7ca94f25a8deac328de100aeef7fb` (PR #138 merged).
Slice: reduce repeated host/device setup in the existing host-returning point
estimators, preserving kernel math and explicit uploads/readback. No dependency,
public signature or automatic placement changes.

## Execution and validation

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/point_estimators.rs`
shares execution for k-NN normals, grid radius normals and grid plane
covariances. Three independent `OnceLock` entries initialize their owning
runtime's pipelines lazily. No point data, neighborhoods or results are cached.
Every call uploads all of its current storage inputs. Uniform and result
staging buffers still allocate per call; grid counting sort remains on the CPU.

Storage inputs and output acquire private leases from the existing bounded
scratch pool. All inputs are overwritten and the unchanged shaders write every
result row. Compute and the dependent output-to-staging copy share one encoder
and one submission, replacing the previous two submissions. Scratch returns to
the pool only after both uses are submitted, so later uploads/dispatches observe
queue ordering. An unsubmitted lease is dropped without recycling, following
the existing `ScratchBuffers` contract. Result staging is never pooled as
storage. Returned host vectors retain their own data independently of subsequent
scratch acquisitions. Retained storage uses the existing 64 MiB default and
eight-resources-per-size bounds; live allocations are outside the retention
budget. Budget pressure can cause later storage allocations.

The pure planner bounds point/output row counts and neighbor indices to u32,
checks byte arithmetic, storage binding/buffer sizes, 16/48-byte uniform limits,
bindings and the original 256-thread one-dimensional dispatch. Covariance's two
vec4 blocks also have bounded u32 indexing. Unsupported sizes return errors
before device allocation/cache initialization. Grid point/output limits are
checked before bounds processing; cell/sentinel storage is checked before CPU
counting-sort allocation. Grid dimensions must be positive signed coordinates.
The shared CPU grid cap remains in effect.

k-NN neighbor length multiplication is checked, and every index must reference
an input point. Invalid indices now return `InvalidArgument` rather than relying
on robust out-of-range device reads. z-length errors now report z's actual
length. Empty calls retain their previous return before kernel/radius validation,
with no pool or pipeline initialization. Radius and epsilon arithmetic are
unchanged, including the previously accepted positive-infinite radius.

The source shaders in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/normals.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/normals_grid.rs`
and
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/covariances_grid.rs`
remain literal matches to the parent (normalized line endings). Neighborhood
order, sums, Jacobi rotations, normal sign/curvature and covariance regularization
are preserved. Resident AoSoA/frame execution, transfers and profiler capacity
remain separate.

For N points, K k-NN entries per row and C grid cells, storage-plus-uniform host
uploads remain `12*N + 4*N*K + 16` bytes for k-NN or
`16*N + 4*(C+1) + 48` for either grid estimator. Result copies remain `16*N`
bytes for normals and `32*N` for covariances. These host-returning APIs still
synchronize/map their result on every nonempty call; this slice adds no new
public transfer receipt.

## Correctness and ownership

Parent host implementations, with their original test blocks omitted, are
frozen under:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/point_normals_previous.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/point_radius_normals_previous.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/point_covariances_previous.rs`

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/point_estimators_tests.rs`
compares 416 old/current cases and all 464,896 returned f32 words, with no
tolerance. Sizes include 1/2/3-point fallbacks, 17, 255/256/257 and 1,025 points.
Patterns include planar, curved, duplicate and sparse points. k values 1/2/7/33
exercise repeated/short neighborhoods. Radius values 0.25/0.75/infinity and
epsilon values 0.001/1 cover radius selection and regularization.

Separate tests cover twelve device-limit failures per kind, index/byte/count
overflow, dispatch and storage boundaries, malformed/empty calls with no
allocation, six concurrent cold calls, independent runtime caches, moving the
runtime, retained host results, warmed storage reuse and shrinking/zero budgets.
Existing submitted/abandoned scratch tests remain active. The initial additional
tests had an incorrect reference-cloud selection and assumed default rather
than actual adapter limits; these test assumptions were corrected without
changing production arithmetic or weakening the planner checks.

## Measurement method

Native release measurements use Rust/Cargo 1.98.1, wgpu 24.0.5 and the local
Vulkan discrete adapter. The adapter model is omitted.

The ignored diagnostic in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/point_estimators_benchmark.rs`
compares the frozen host APIs against current APIs at 17/257/4,096/65,536 points
for each kind, with k=16, radius=0.25 and epsilon=0.001. Fixtures and indices are
prepared outside timing. Each condition gets three warmups and twenty old/new
pairs, alternating order. The timer spans API entry through result return:
validation, CPU grid work, pipeline objects, allocation/uploads, GPU compute,
result copy, map wait and API result decoding. Word conversion, comparison and
logging occur after the timer. Both paths finish synchronously and every round
compares all output words. This measures completed full-API latency, not isolated
GPU shader time or resident frame throughput. First-use runtime/pipeline cost
is excluded by warmup; the old API continues creating objects on later calls.
Diagnostic GPU work does not overlap this task's regression tests. Existing
user jobs remain untouched, so timings describe local observed conditions.

The table records timing medians and median paired old/current ratios. The
latter can differ from a ratio of medians. Confidence intervals bootstrap paired
ratios 10,000 times with seed 20261011 (2.5/97.5 percentiles). Pool counters cover
storage allocations/reuses only, excluding uniform/staging and host allocations.
Cross-device speedups and universal latency improvement are not claimed.

All twelve measured paired ratios favored the current warmed API, from
1.161 [1.047,1.192] to 2.246 [2.165,2.348]. At 65,536 points the ratios are
1.200 / 1.161 / 1.172 for k-NN normals / radius normals / plane covariances.
The largest case retains substantial CPU grid and transfer work. Warm storage
allocations are zero in every timed condition; twenty calls reused 100 k-NN
or 120 grid-estimator storage buffers. Uniform/staging and host allocations
remain outside those pool counters. These numbers describe this local dataset
and warmed full-API calls, not first-call latency or faster shader arithmetic.

| Points | Estimator | Previous/current full API median (ms) | Paired ratio [95% CI] | Timed storage allocations/reuses |
|---:|---|---:|---:|---:|
| 17 | KnnNormals | 1.885/0.986 | 1.843 [1.831,1.875] | 0/100 |
| 17 | RadiusNormals | 2.492/1.188 | 2.146 [2.065,2.203] | 0/120 |
| 17 | PlaneCovariances | 2.482/1.111 | 2.246 [2.165,2.348] | 0/120 |
| 257 | KnnNormals | 2.187/1.236 | 1.872 [1.711,1.980] | 0/100 |
| 257 | RadiusNormals | 2.266/1.133 | 2.041 [1.978,2.072] | 0/120 |
| 257 | PlaneCovariances | 2.444/1.113 | 2.162 [2.106,2.222] | 0/120 |
| 4,096 | KnnNormals | 2.167/1.265 | 1.728 [1.661,1.796] | 0/100 |
| 4,096 | RadiusNormals | 2.624/1.361 | 1.938 [1.797,1.980] | 0/120 |
| 4,096 | PlaneCovariances | 3.137/1.495 | 2.060 [1.853,2.144] | 0/120 |
| 65,536 | KnnNormals | 4.166/3.522 | 1.200 [1.160,1.288] | 0/100 |
| 65,536 | RadiusNormals | 13.249/11.909 | 1.161 [1.047,1.192] | 0/120 |
| 65,536 | PlaneCovariances | 14.464/12.112 | 1.172 [1.121,1.216] | 0/120 |

## Completed validation

- 132 native tests passed: 106 unit, 4 CDF, 5 CLAHE, 5 equalization,
  3 histogram and 9 stage-timing tests. Seven explicit diagnostics are ignored
  by the normal suite; the new point-estimator diagnostic ran separately.
- Native all-feature/all-target and default/all-target Clippy passed with
  `-D warnings`. Minimal wgpu and combined image/AoSoA WASM library Clippy passed.
- The exact minimal-wgpu planner command passed without adapter access.
- All twelve paired release conditions completed, with every round matching
  all returned f32 words and twenty positive timing pairs per condition.
- Formatting, diff whitespace, parent fixtures (trailing whitespace normalized),
  all three shader literals and publication scans passed.

Local ignored receipts:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-regressions.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-default-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-wasm-minimal.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-wasm-combined.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-ci-pure-plan.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-paired.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-estimates.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-point-estimators-table.md`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-gpu-point-estimators.py`

The estimate script validates sample counts, positivity and zero timed storage
allocations, retaining individual ratios and timing ranges with the full table.


## Reproduction

Run from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1
cargo test -p spatialrust-gpu --features gpu-wgpu point_estimator_plan_bounds_storage_uniform_indices_and_dispatch --lib
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-wgpu --target wasm32-unknown-unknown --lib -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown --lib -- -D warnings
cargo test -p spatialrust-gpu --features gpu-wgpu --release --lib paired_previous_current_standalone_point_estimators -- --ignored --nocapture --test-threads=1
cargo fmt --all -- --check
git diff --check
```

CI's existing GPU-image job executes the new adapter-free planner test after
compiling the GPU tests. The required job inventory remains 114. Native GPU
regressions/diagnostics need an adapter; the planner does not. WASM checks
compile only; browser execution of these synchronous APIs is unverified.
