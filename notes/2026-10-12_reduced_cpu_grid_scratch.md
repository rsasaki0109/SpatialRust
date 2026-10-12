# Reduced CPU dense-grid scratch

## Change and stable-order proof

`spatialrust-search::build_grid` previously retained cell counts/starts plus a
cloned cell-sized cursor while scattering points into stable CSR order. The
revised builder accumulates inclusive cell ends in the original count vector,
then visits source points in descending index order and writes each point at
its cell's decremented end. The resulting vector is already the final CSR
start table; no cursor clone or additional start-reconstruction pass is needed.

For each cell c, its cumulative end E_c is decremented exactly count_c times,
leaving E_c - count_c = start_c. Descending source indices fill descending
slots, preserving ascending source order within each cell. Cells without
points retain their correct start; the extra sentinel is never scattered into
and remains the total point count. For valid dimensions and u32 point counts,
all cumulative ends and scatter addresses stay within the original bounds.
Coordinate conversion, clamping and per-point f32 arithmetic remain unchanged.

Production source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid.rs`.

Work remains O(points + cells), with two coordinate-mapping passes. Construction
stays serial under every feature set. There is no new point-ID cache, worker,
feature, dependency, unsafe code or implicit device transfer. The public return
values and signatures remain unchanged. CPU dense-grid clustering and standalone
GPU radius normal/plane covariance preparation share this builder. The normal
API documentation now states O(n + cells) rather than O(n).

## Memory boundary

Excluding XYZ inputs, Vec headers, allocator overhead and all downstream work,
live logical builder payload changes from `4*n + 8*(C+1)` to
`4*n + 4*(C+1)` bytes. The **removed temporary payload is 4*(C+1) bytes**, where
n is points and C is cells. The two returned vectors retain `4*n + 4*(C+1)`
bytes in either implementation. At 1,048,576 cells the removed allocation is
4 MiB plus four bytes; at 8,388,608 cells it is 32 MiB plus four bytes. At the
existing CPU grid cap of 64,000,000 cells it would be 256,000,004 bytes.
GPU device/storage guards may reject such large grids before construction.

This is an allocation/payload derivation from the two implementations, not an
RSS or whole-process peak-memory measurement. Allocator retention, GPU storage,
readback staging and independently retained results are outside this boundary.
GPU uploads still contain exactly the same indices/offsets; shaders, GPU pools,
copy counts, mapping synchronization and result decoding are unchanged.

## Correctness and comparison source

CPU regression source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid_build_tests.rs`.

- **2,316,018 exact CSR words across 180 cases** match the previously frozen
  builder and an independent integer-key comparison-sort/partition oracle.
  Six grid shapes include singleton axes and 27,559 cells, ten point counts
  include empty inputs, 255/256/257 and 4,095/4,096/4,097 boundaries plus 65,537
  points. Three binary-exact cell sizes, interleaved cells, duplicates, exact
  cell boundaries and many empty cells preserve stable order and sentinel mass.
- Separate frozen comparisons cover clamping, signed zero, subnormals, arbitrary
  radii, extreme finite values, infinity and NaN on prescribed valid grids.
  They do not expand the existing grid-bounds validation contract. An
  independently specified clamped-cell result verifies stable indices and
  empty-cell offsets directly.
- Four concurrent callers retain two independently owned grid results each;
  both match the frozen output after later calls. Existing row traversal and
  brute-force/minimum-component regressions also pass with the new builder.
- The facade adds an independently expected stable permutation/start/sentinel
  case and retains its interleaved partial-group root regression.

The existing CPU frozen production prefix is at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/tests/fixtures/uniform_grid_previous.rs`.
Its build function was verified identical to parent
`a249b1df2cdf37507b3ebe4c2cd04b7ae2f87b84`; it predates the prior row-iterator
change, which did not change construction. A standalone exact copy of that
parent build function supports GPU replay at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/grid_build_previous.rs`.

GPU regression/replay source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_build_tests.rs`.

**520,344 exact output words across 160 full-API cases** compare current and
frozen-builder radius normals / covariances. Cases cover empty, singleton,
pair/triple, partial groups, compact and sparse grids, two radii and three
epsilon values including negative zero. The replay's wrappers and preparation
were checked against the pinned parent after only function/import routing and
formatting substitutions. Both modes use the same current runtime-owned
pipeline cache, shaders, bounded scratch executor and native readback. In
particular, both retain the corrected covariance fallback writes. No older
shader difference is included in these builder timings or output comparisons.
The replay compiles only into native test builds.

Validation on Windows with rustc 1.98.1:

| Check | Result |
| --- | --- |
| Search all-feature tests | 46 passed, 0 failed, 4 ignored timing diagnostics |
| Search minimal tests | 14 passed, 0 failed, 2 ignored timing diagnostics |
| GPU image + AoSoA native tests | 172 passed (145 unit + 27 integration), 0 failed, 16 ignored |
| CPU facade minimal and `search-parallel` | 2 passed each |
| Native GPU facade grid/fallback smoke | 1 passed each with explicit ignored-test execution |
| Search/GPU all-feature all-target Clippy | `-D warnings` passed |
| CPU facade Clippy | `-D warnings` passed |
| Minimal search / GPU image + AoSoA WASM | Compile checks passed |
| Formatting and diff whitespace | Passed |

WASM checks establish compilation only. All own correctness/build jobs finish
before the respective timing runs; diagnostics execute alone, rather than
alongside another cargo test/build workload. Facade minimal/Clippy checks were
also completed before the repeat survey. The new native GPU regression fails on an adapter
initialization error rather than returning successful omission.

## Completed-call measurements

The release diagnostics are:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid_build_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_build_benchmark.rs`

```powershell
cargo test -p spatialrust-search --release --no-default-features paired_previous_current_grid_build -- --ignored --nocapture --test-threads=1
cargo test -p spatialrust-gpu --release --features gpu-wgpu paired_previous_current_cpu_grid_scratch -- --ignored --nocapture --test-threads=1
```

Each condition warms both modes three times and records 30 alternating-order
pairs. CPU timers include completed `build_grid` allocation, histogram, prefix
and scatter, with input generation, equality checking and returned-vector drop
outside timing. Prescribed shapes vary cell/point density, including one-cell
and very sparse grids. GPU timers include public input/device validation,
bounds/grid preparation, uploads, binding/encoding/submission, mapping wait
and normal/covariance decoding. Word conversion/checks/drop are outside the
GPU timer. The same current shader/executor serves both modes, and timestamp
queries are not requested. Native measurements use Vulkan and a discrete
adapter, with model identity omitted. Release uses thin LTO and one codegen unit.

Two surveys cover 12 CPU and 12 completed GPU conditions each. Tables give
marginal median milliseconds and the geometric mean of paired previous/current
ratios, with a 95% paired log-ratio bootstrap interval (10,000 resamples,
seed 20261012). A ratio above one favors the revision. Ratios need not equal
ratios of marginal medians. These intervals describe the observed paired
samples, not unmeasured hardware or systematic host/scheduling bias.

The fixed 1,048,576-cell sparse GPU fixtures reproduce favorable completed-call
ratios in all six size/operation cases: about **1.11–1.47** across both surveys
(approximately **10–32%** shorter paired completed calls). Normal/covariance
ratios at 65,536 points span 1.109–1.301; smaller sparse calls benefit more
because grid work dominates. Several compact GPU cases have intervals that
include one; a general GPU throughput gain is not established for those cases.
Sparse fixtures are synthetic density controls, not a claim about typical
sensor workloads or GPU kernel speed.

For CPU construction, 257-point grids with at least 1,048,576 cells and
65,536-point sparse grids reproduce gains. The 1,048,579-point conditions are
less conclusive across surveys, including a one-cell repeat ratio of 0.935
[0.839, 1.041] and a million-cell initial ratio of 0.970 [0.834, 1.138]. Overall
CPU benchmark wall time varies substantially between surveys (57.90s and
30.55s); absolute timings should not be treated as stable. The memory reduction
is independent of these performance uncertainties, and no uniform CPU speedup
is claimed.

### Survey A: CPU construction

| Points | Cells | Saved temporary bytes | Previous ms | Current ms | Paired ratio [95% interval] |
| ---: | ---: | ---: | ---: | ---: | --- |
| 257 | 1 | 8 | 0.0344 | 0.0326 | 1.156 [1.046, 1.378] |
| 257 | 4,096 | 16,388 | 0.0428 | 0.0403 | 1.073 [1.055, 1.099] |
| 257 | 1,048,576 | 4,194,308 | 4.8459 | 2.9455 | 1.728 [1.596, 1.883] |
| 257 | 8,388,608 | 33,554,436 | 30.8744 | 18.9969 | 1.566 [1.420, 1.717] |
| 65,536 | 1 | 8 | 3.8649 | 3.1722 | 1.175 [1.008, 1.353] |
| 65,536 | 4,096 | 16,388 | 3.4972 | 3.4753 | 0.994 [0.858, 1.150] |
| 65,536 | 1,048,576 | 4,194,308 | 16.7655 | 18.0517 | 1.268 [1.085, 1.509] |
| 65,536 | 8,388,608 | 33,554,436 | 51.3751 | 44.3312 | 1.319 [1.201, 1.449] |
| 1,048,579 | 1 | 8 | 37.8789 | 35.8030 | 1.088 [0.996, 1.190] |
| 1,048,579 | 4,096 | 16,388 | 43.0685 | 41.2620 | 1.056 [0.962, 1.159] |
| 1,048,579 | 1,048,576 | 4,194,308 | 201.4888 | 268.2736 | 0.970 [0.834, 1.138] |
| 1,048,579 | 8,388,608 | 33,554,436 | 254.9820 | 247.2818 | 1.041 [0.983, 1.108] |

### Survey A: completed GPU public API

| Points | Cells | Operation | Previous ms | Current ms | Paired ratio [95% interval] |
| ---: | ---: | --- | ---: | ---: | --- |
| 257 | 256 | normal | 1.3324 | 1.3143 | 1.023 [0.987, 1.060] |
| 257 | 256 | covariance | 1.1840 | 1.1494 | 1.014 [0.987, 1.042] |
| 257 | 1,048,576 | normal | 5.7044 | 3.9480 | 1.432 [1.393, 1.472] |
| 257 | 1,048,576 | covariance | 8.7975 | 6.2248 | 1.455 [1.369, 1.543] |
| 4,096 | 4,096 | normal | 2.4112 | 2.2599 | 1.070 [1.000, 1.137] |
| 4,096 | 4,096 | covariance | 2.0306 | 2.2569 | 0.947 [0.890, 1.007] |
| 4,096 | 1,048,576 | normal | 8.1075 | 6.1702 | 1.353 [1.274, 1.436] |
| 4,096 | 1,048,576 | covariance | 7.3718 | 5.4771 | 1.352 [1.282, 1.430] |
| 65,536 | 65,536 | normal | 6.2636 | 5.8031 | 1.032 [0.938, 1.132] |
| 65,536 | 65,536 | covariance | 13.8271 | 12.1314 | 1.130 [1.019, 1.278] |
| 65,536 | 1,048,576 | normal | 22.6026 | 21.3004 | 1.109 [1.048, 1.175] |
| 65,536 | 1,048,576 | covariance | 25.7116 | 23.4083 | 1.138 [1.080, 1.201] |

### Survey B: CPU construction

| Points | Cells | Saved temporary bytes | Previous ms | Current ms | Paired ratio [95% interval] |
| ---: | ---: | ---: | ---: | ---: | --- |
| 257 | 1 | 8 | 0.0065 | 0.0058 | 1.130 [1.111, 1.150] |
| 257 | 4,096 | 16,388 | 0.0093 | 0.0083 | 1.108 [1.056, 1.141] |
| 257 | 1,048,576 | 4,194,308 | 2.5917 | 1.6401 | 1.588 [1.556, 1.623] |
| 257 | 8,388,608 | 33,554,436 | 20.8004 | 12.2047 | 1.732 [1.666, 1.798] |
| 65,536 | 1 | 8 | 1.7840 | 1.6326 | 1.104 [1.056, 1.159] |
| 65,536 | 4,096 | 16,388 | 1.8882 | 1.7815 | 1.072 [1.013, 1.140] |
| 65,536 | 1,048,576 | 4,194,308 | 8.5855 | 6.6692 | 1.299 [1.259, 1.342] |
| 65,536 | 8,388,608 | 33,554,436 | 40.4203 | 30.1946 | 1.323 [1.258, 1.387] |
| 1,048,579 | 1 | 8 | 30.3811 | 30.0620 | 0.935 [0.839, 1.041] |
| 1,048,579 | 4,096 | 16,388 | 34.5334 | 32.4613 | 1.058 [0.967, 1.155] |
| 1,048,579 | 1,048,576 | 4,194,308 | 88.1649 | 76.9250 | 1.094 [1.017, 1.173] |
| 1,048,579 | 8,388,608 | 33,554,436 | 207.3324 | 200.1424 | 1.050 [1.014, 1.087] |

### Survey B: completed GPU public API

| Points | Cells | Operation | Previous ms | Current ms | Paired ratio [95% interval] |
| ---: | ---: | --- | ---: | ---: | --- |
| 257 | 256 | normal | 1.4779 | 1.4766 | 1.002 [0.957, 1.052] |
| 257 | 256 | covariance | 1.2357 | 1.2372 | 0.983 [0.947, 1.019] |
| 257 | 1,048,576 | normal | 6.1274 | 4.2176 | 1.418 [1.379, 1.458] |
| 257 | 1,048,576 | covariance | 5.3326 | 3.6787 | 1.472 [1.435, 1.512] |
| 4,096 | 4,096 | normal | 1.7399 | 1.7024 | 0.993 [0.931, 1.058] |
| 4,096 | 4,096 | covariance | 1.5065 | 1.5228 | 0.995 [0.943, 1.061] |
| 4,096 | 1,048,576 | normal | 8.5372 | 6.4351 | 1.325 [1.249, 1.402] |
| 4,096 | 1,048,576 | covariance | 8.4480 | 6.0237 | 1.435 [1.387, 1.485] |
| 65,536 | 65,536 | normal | 7.5113 | 6.9364 | 1.101 [1.005, 1.211] |
| 65,536 | 65,536 | covariance | 6.2691 | 6.1709 | 1.088 [1.012, 1.170] |
| 65,536 | 1,048,576 | normal | 15.9413 | 12.6811 | 1.228 [1.128, 1.336] |
| 65,536 | 1,048,576 | covariance | 15.8498 | 11.4982 | 1.301 [1.207, 1.398] |

Raw diagnostics and computed samples remain ignored beneath
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`, using the
`reduced-grid-` prefix. This change does not address nonfinite/extreme grid-bounds
validation, sparse segment lookup complexity, CPU parallel grid construction
or automatic device placement; those require separate focused slices.
