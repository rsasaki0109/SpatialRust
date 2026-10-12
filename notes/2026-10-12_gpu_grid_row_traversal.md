# Ordered GPU dense-grid row traversal

## Implementation and ordering

Dense-grid radius normal and plane-covariance shaders previously visited up to
27 adjacent cells twice: once for the mean and once for covariance. Counting
sort stores adjacent x cells contiguously. Each gather now visits up to nine
y/z rows, reading the interval from the first valid x cell's start to the
sentinel immediately after the last valid x cell. The two clamped x endpoints
are computed once per query point and reused in both gathers.

For each fixed y/z row, concatenating the old x-cell intervals gives exactly
the new interval, including empty cells. Rows retain z/y order; points retain
cell order and original index order within each cell. Distance tests, mean/count
updates, centered covariance updates, eigensolver/regularization and output
decoding retain every operation and operand. This changes index traversal,
not floating reduction order or neighborhood membership.

Existing validation bounds positive grid dimensions by i32::MAX and checks
cell/sentinel storage before allocation. With `0 <= cx < dimx`, both `cx-1` and
`cx+1` fit i32. Clamping gives `0 <= first_x < after_x <= dimx`; the largest
row-plus-endpoint is the legal final cell sentinel. The bounded total cell
count also bounds u32 row arithmetic. No public signature, feature, dependency,
unsafe code, pipeline, workgroup storage, barrier or global buffer is added.
Host counting-sort grid construction, full input uploads, pooled leases, one
compute/result-copy submission and synchronous result mapping remain explicit.
Standalone storage input payload remains `16*n + 4*(cells+1)` bytes, plus
the existing 48-byte uniform. Result staging retains `16*n` normal output or
`32*n` covariance output bytes.

Production dense sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/normals_grid.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/covariances_grid.rs`

## Independent sparse radius source

The sparse AoSoA path previously constructed its shader by replacing pieces of
the dense shader's text. Dense row traversal is inappropriate for sparse keys:
their binary search and x/y/z key order differ from dense CSR layout. Native
regression tests exposed this text dependency during the trial.

The sparse shader is now stored directly at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/sparse_radius_normals.wgsl`.
Its contents match the parent's generated shader after line-ending and trailing
whitespace normalization.
Its signed cell lookup, neighborhood traversal, floating operations, bindings,
workgroup size and execution remain unchanged. The source is borrowed directly;
per-call source replacement allocations are removed. Dense source re-export
is removed because the sparse path no longer needs it. This does not claim a
measured sparse execution speedup or change AoSoA transfer/profiling boundaries.

A native source test reconstructs all sixteen original replacements using the
frozen dense shader and checks the materialized result. It normalizes CRLF
because Rust normalizes inline source literals while `include_str!` preserves
file bytes; blank indentation left by removed guards is trimmed. Existing
sparse-normal and frame tests exercise the compiled shader,
including negative chunked positions, live capabilities and different runtimes.

## Correctness and builds

The frozen dense sources match parent source
`4cdcbf77dfb2ff880a128d55cadf1627589edd75`:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/grid_radius_previous.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/grid_covariance_previous.wgsl`

Two new native GPU tests check:

- **1,702,944 exact f32 output words across 486 cases**, comparing public normal
  and covariance outputs against frozen shaders through 4,097 points. Shapes
  include 3D volume, sheets, single-x/single-y/single-z dimensions, coincident
  points and curved coordinates; signed coordinates, lane/group boundaries,
  insufficient support, finite radii and infinite radius are covered. Two
  covariance regularization values are checked. Existing frozen estimator tests
  add 416 cases and 464,896 output words, with reuse, ownership and validation.
- **295,520 independent ordered-neighbor fingerprint/count words** across
  frozen/current normal and covariance traversal. Probes keep the actual shader's
  coordinate, interval and distance code and replace mean/count accumulation with
  two order-sensitive integer fingerprints, an index checksum and a count.
  A CPU oracle independently sorts by z/y/x cell and original index, then
  enumerates membership without production CSR offsets. Binary coordinates and
  radii make distance decisions exact. Fingerprints supplement frozen full-output
  comparisons and the contiguous-interval argument; they are not full index lists.

The native reference API wrappers and executor copy the parent code, changing
only names/uniform type names and injecting prepared pipelines instead of the
cache lookup. These specific substitutions were verified with whitespace
normalization. Production host wrappers and shared executor are unchanged;
all four gather accumulation bodies preserve their operations/operands.
Native tests and measurement sources are:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_rows_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_rows_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/aoso_staging/sparse_radius_shader_tests.rs`

Native Windows, Rust/Cargo 1.98.1 and wgpu 24: **168 GPU tests pass** (141
unit plus 27 integration), zero failures and fifteen ignored diagnostics.
Minimal GPU traversal tests, the explicit facade smoke, two feature-level GPU
normal tests and one GICP GPU covariance alignment test pass. Strict default/
all-feature all-target GPU Clippy and facade Clippy pass, including the native
resident diagnostic after its addition. Three WASM builds pass: minimal GPU,
GPU image/AoSoA, and minimal facade GPU. This establishes compilation, not
browser execution/performance. Formatting and diff checks pass.

The facade smoke requires a native adapter and is ignored by default on
GPU-less feature CI. It is executed explicitly with `--include-ignored` locally,
preserving hard adapter/output assertions. Its source is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_grid_row_smoke.rs`.

```text
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo test -p spatialrust-gpu --no-default-features --features gpu-wgpu --lib grid_rows -- --test-threads=1 --nocapture
cargo test -p spatialrust --no-default-features --features gpu-wgpu --test gpu_grid_row_smoke -- --include-ignored --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features gpu-wgpu --test gpu_grid_row_smoke -- -D warnings
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu,gpu-image,gpu-aoso-staging
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu
```

## Paired completed API measurements

Native release measurements use the reported backend/device capabilities without
publishing an adapter model. Each condition has three warmup pairs followed by
twenty alternating frozen/current pairs. Three point counts and three shapes
(volume, tilted sheet, separated eight-point clusters) cover both normal and
covariance APIs: eighteen conditions per survey. The radius is 0.25; covariance
epsilon is 0.001. Own test/build processes finish before performance measurement.

The timer covers validation, CPU grid bounds/counting sort, storage acquisition,
input uploads, uniform/bind-group/staging creation, encoding/submission, GPU
compute/result copy, mapping wait and public output decoding. Frozen pipelines
are prepared beforehand; current execution uses the actual warmed public API
cache. Conversion to comparison words, comparisons and result drop are outside
the timer. Every warmup and measured output pair must match exactly. After
warmup the scratch allocation count must remain unchanged; storage reuse is
recorded. These measurements include the standalone host/device boundary and
CPU grid work, not an isolated GPU stage or resident AoSoA frame.

Paired ratios are medians of twenty old/new API completion ratios; above one
favors row traversal. The 95% intervals use 10,000 bootstrap resamples of that
median, seed 20261012. Intervals describe within-survey variation, not between-run
uncertainty or portable performance. Separate wall medians can have a different
quotient from the paired median ratio.

First complete-API survey:

| Points | Pattern | Kind | Old/new API ms | Paired ratio (95% CI) |
| ---: | --- | --- | ---: | ---: |
| 257 | volume | RadiusNormals | 1.847/1.811 | 0.982 (0.903-1.021) |
| 257 | volume | PlaneCovariances | 3.259/3.077 | 0.923 (0.510-1.271) |
| 257 | tilted sheet | RadiusNormals | 1.791/1.815 | 1.009 (0.964-1.070) |
| 257 | tilted sheet | PlaneCovariances | 1.832/1.774 | 1.043 (0.985-1.103) |
| 257 | eight-point clusters | RadiusNormals | 1.777/1.759 | 1.000 (0.937-1.056) |
| 257 | eight-point clusters | PlaneCovariances | 1.836/1.950 | 1.020 (0.920-1.193) |
| 4,096 | volume | RadiusNormals | 2.618/2.544 | 1.050 (0.949-1.229) |
| 4,096 | volume | PlaneCovariances | 2.023/1.910 | 1.037 (1.013-1.122) |
| 4,096 | tilted sheet | RadiusNormals | 2.139/2.068 | 1.025 (0.966-1.046) |
| 4,096 | tilted sheet | PlaneCovariances | 4.227/3.732 | 1.000 (0.657-1.532) |
| 4,096 | eight-point clusters | RadiusNormals | 2.178/2.127 | 1.009 (0.959-1.082) |
| 4,096 | eight-point clusters | PlaneCovariances | 2.434/2.511 | 0.975 (0.898-1.039) |
| 65,536 | volume | RadiusNormals | 10.000/11.488 | 1.010 (0.822-1.475) |
| 65,536 | volume | PlaneCovariances | 11.068/10.754 | 1.043 (0.959-1.208) |
| 65,536 | tilted sheet | RadiusNormals | 12.927/13.172 | 0.990 (0.891-1.059) |
| 65,536 | tilted sheet | PlaneCovariances | 12.922/12.809 | 1.019 (0.894-1.156) |
| 65,536 | eight-point clusters | RadiusNormals | 40.240/36.818 | 1.087 (0.948-1.198) |
| 65,536 | eight-point clusters | PlaneCovariances | 21.043/19.931 | 1.046 (0.992-1.087) |

Immediate complete-API repeat:

| Points | Pattern | Kind | Old/new API ms | Paired ratio (95% CI) |
| ---: | --- | --- | ---: | ---: |
| 257 | volume | RadiusNormals | 1.684/1.630 | 1.029 (0.985-1.110) |
| 257 | volume | PlaneCovariances | 1.515/1.561 | 0.977 (0.937-1.017) |
| 257 | tilted sheet | RadiusNormals | 1.476/1.446 | 0.993 (0.942-1.031) |
| 257 | tilted sheet | PlaneCovariances | 1.428/1.435 | 1.000 (0.975-1.016) |
| 257 | eight-point clusters | RadiusNormals | 1.382/1.423 | 0.963 (0.935-1.050) |
| 257 | eight-point clusters | PlaneCovariances | 1.525/1.580 | 1.019 (0.947-1.048) |
| 4,096 | volume | RadiusNormals | 1.721/1.703 | 1.015 (0.978-1.043) |
| 4,096 | volume | PlaneCovariances | 1.504/1.487 | 1.012 (0.988-1.083) |
| 4,096 | tilted sheet | RadiusNormals | 1.939/1.983 | 1.014 (0.880-1.089) |
| 4,096 | tilted sheet | PlaneCovariances | 1.582/1.693 | 1.001 (0.910-1.015) |
| 4,096 | eight-point clusters | RadiusNormals | 1.959/1.947 | 1.023 (0.896-1.070) |
| 4,096 | eight-point clusters | PlaneCovariances | 2.152/2.210 | 1.007 (0.886-1.082) |
| 65,536 | volume | RadiusNormals | 8.362/8.946 | 0.958 (0.876-1.102) |
| 65,536 | volume | PlaneCovariances | 6.110/6.186 | 1.008 (0.946-1.104) |
| 65,536 | tilted sheet | RadiusNormals | 13.093/10.603 | 1.044 (0.956-1.188) |
| 65,536 | tilted sheet | PlaneCovariances | 12.851/12.767 | 1.013 (0.923-1.045) |
| 65,536 | eight-point clusters | RadiusNormals | 14.630/13.447 | 1.042 (0.962-1.173) |
| 65,536 | eight-point clusters | PlaneCovariances | 10.710/10.312 | 1.047 (0.975-1.122) |

Prepared resident diagnostics use Vulkan with pass timestamps. Grid construction, input/output initialization, uniform/bind-group creation and profiler setup are outside the completion timer. That timer covers command encoding, one compute pass, submission and device wait. Query collection (one stage, 16 bytes) and complete raw output comparison/readback are outside both timers. Every warmup and measured output pair agrees, including padding. These are native diagnostic buffers and do not change production transfer APIs.

First resident survey:

| Points | Pattern | Kind | Old/new GPU us | GPU ratio (95% CI) | Prepared completion ratio (95% CI) |
| ---: | --- | --- | ---: | ---: | ---: |
| 257 | volume | RadiusNormals | 11.68/9.86 | 1.182 (1.163-1.198) | 1.039 (0.977-1.062) |
| 257 | volume | PlaneCovariances | 12.91/9.79 | 1.313 (1.292-1.338) | 1.008 (1.001-1.024) |
| 257 | tilted sheet | RadiusNormals | 27.38/17.31 | 1.578 (1.565-1.584) | 1.083 (1.041-1.094) |
| 257 | tilted sheet | PlaneCovariances | 25.34/17.02 | 1.489 (1.484-1.499) | 1.081 (1.068-1.088) |
| 257 | eight-point clusters | RadiusNormals | 8.80/7.82 | 1.122 (1.088-1.170) | 1.014 (0.989-1.066) |
| 257 | eight-point clusters | PlaneCovariances | 8.67/9.09 | 0.976 (0.944-1.047) | 1.003 (0.990-1.018) |
| 4,096 | volume | RadiusNormals | 31.14/27.60 | 1.130 (1.119-1.136) | 0.983 (0.966-0.999) |
| 4,096 | volume | PlaneCovariances | 39.44/26.75 | 1.464 (1.445-1.490) | 1.033 (0.996-1.108) |
| 4,096 | tilted sheet | RadiusNormals | 32.37/21.44 | 1.492 (1.479-1.506) | 1.023 (1.002-1.044) |
| 4,096 | tilted sheet | PlaneCovariances | 29.60/21.46 | 1.371 (1.359-1.385) | 1.026 (0.985-1.045) |
| 4,096 | eight-point clusters | RadiusNormals | 11.84/11.20 | 1.051 (1.043-1.076) | 0.993 (0.986-1.015) |
| 4,096 | eight-point clusters | PlaneCovariances | 8.86/8.03 | 1.063 (0.980-1.104) | 0.985 (0.977-1.010) |
| 65,536 | volume | RadiusNormals | 316.67/303.18 | 1.044 (1.037-1.054) | 1.019 (0.866-1.111) |
| 65,536 | volume | PlaneCovariances | 331.09/302.40 | 1.098 (1.090-1.103) | 1.139 (1.025-1.298) |
| 65,536 | tilted sheet | RadiusNormals | 156.48/109.74 | 1.426 (1.421-1.432) | 1.207 (1.074-1.477) |
| 65,536 | tilted sheet | PlaneCovariances | 137.60/108.90 | 1.263 (1.261-1.265) | 1.114 (0.909-1.258) |
| 65,536 | eight-point clusters | RadiusNormals | 54.32/36.42 | 1.490 (1.477-1.499) | 0.789 (0.662-1.452) |
| 65,536 | eight-point clusters | PlaneCovariances | 60.91/36.32 | 1.673 (1.646-1.692) | 1.030 (0.902-1.231) |

Immediate resident repeat:

| Points | Pattern | Kind | Old/new GPU us | GPU ratio (95% CI) | Prepared completion ratio (95% CI) |
| ---: | --- | --- | ---: | ---: | ---: |
| 257 | volume | RadiusNormals | 11.81/9.86 | 1.200 (1.190-1.216) | 1.022 (1.007-1.046) |
| 257 | volume | PlaneCovariances | 12.70/9.63 | 1.315 (1.271-1.329) | 1.013 (0.996-1.068) |
| 257 | tilted sheet | RadiusNormals | 27.47/17.46 | 1.575 (1.548-1.616) | 1.279 (1.185-1.430) |
| 257 | tilted sheet | PlaneCovariances | 25.52/17.23 | 1.476 (1.459-1.486) | 1.077 (1.022-1.144) |
| 257 | eight-point clusters | RadiusNormals | 8.93/7.90 | 1.142 (1.123-1.187) | 1.015 (0.963-1.062) |
| 257 | eight-point clusters | PlaneCovariances | 8.38/7.66 | 1.057 (1.008-1.103) | 0.977 (0.957-1.003) |
| 4,096 | volume | RadiusNormals | 30.56/26.61 | 1.153 (1.130-1.166) | 1.001 (0.970-1.036) |
| 4,096 | volume | PlaneCovariances | 39.58/27.28 | 1.447 (1.442-1.458) | 1.021 (0.992-1.057) |
| 4,096 | tilted sheet | RadiusNormals | 31.98/21.04 | 1.536 (1.516-1.551) | 1.030 (1.005-1.070) |
| 4,096 | tilted sheet | PlaneCovariances | 29.46/20.69 | 1.421 (1.415-1.459) | 1.039 (0.978-1.072) |
| 4,096 | eight-point clusters | RadiusNormals | 9.60/8.29 | 1.156 (1.142-1.170) | 1.017 (0.915-1.150) |
| 4,096 | eight-point clusters | PlaneCovariances | 10.34/10.54 | 0.972 (0.954-0.981) | 0.982 (0.956-1.044) |
| 65,536 | volume | RadiusNormals | 317.86/305.22 | 1.040 (1.037-1.045) | 0.951 (0.730-1.329) |
| 65,536 | volume | PlaneCovariances | 333.38/304.19 | 1.095 (1.088-1.108) | 1.120 (0.920-1.205) |
| 65,536 | tilted sheet | RadiusNormals | 158.24/111.12 | 1.423 (1.416-1.432) | 0.998 (0.911-1.115) |
| 65,536 | tilted sheet | PlaneCovariances | 138.86/110.11 | 1.261 (1.257-1.267) | 1.076 (0.888-1.236) |
| 65,536 | eight-point clusters | RadiusNormals | 54.72/36.61 | 1.503 (1.489-1.516) | 0.921 (0.643-1.584) |
| 65,536 | eight-point clusters | PlaneCovariances | 61.44/36.62 | 1.673 (1.665-1.684) | 0.956 (0.745-1.123) |

At 65,536 points, volume GPU ratios are 1.040-1.044 for normals and
1.095-1.098 for covariance. Tilted-sheet ratios are 1.423-1.426 and 1.261-1.263;
separated-cluster ratios are 1.490-1.503 and 1.673 in both surveys. Each of these
GPU-stage intervals is above one. Traversal savings remain visible while CPU
construction, transfers and synchronization dominate the complete API time.

The API surveys do not establish a repeatable overall speedup: all repeat API
intervals include one, including 0.958 (0.876-1.102) for 65,536 volume normals.
The first 4,096 volume covariance API ratio is 1.037 (1.013-1.122), but the
repeat is 1.012 (0.988-1.083). Prepared completion intervals are also noisy;
adverse observations such as 0.789 (0.662-1.452) for first 65,536 cluster normals
and 0.983 (0.966-0.999) for first 4,096 volume normals are retained above.

Small/cluster GPU conditions are not universally faster. The repeated 4,096
cluster covariance stage ratio is 0.972 (0.954-0.981), with old/new medians
10.34/10.54 us; the first survey is 1.063 (0.980-1.104). The first 257 cluster
covariance stage is 0.976 (0.944-1.047), followed by 1.057 (1.008-1.103).
No distribution-sensitive specialization, placement crossover, portable gain
or end-to-end pipeline acceleration is claimed. The adopted change is ordered
row traversal with measured GPU-stage savings and preserved complete-API
behavior; total API improvements remain unestablished.

Native reproduction: unset `SPATIALRUST_GRID_ROW_COMPARE` for complete-API
measurements; set it to `resident` only for the prepared GPU diagnostic.

```text
cargo test -p spatialrust-gpu --release --no-default-features --features gpu-wgpu --lib paired_previous_current_grid_rows -- --ignored --nocapture --test-threads=1
```

All validation logs and complete paired samples remain private under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-grid-row-ranges-*`.
The existing 114 named CI checks and exact source/dependency merge guard gate
delivery; failed, pending, missing or skipped required checks cannot merge.
