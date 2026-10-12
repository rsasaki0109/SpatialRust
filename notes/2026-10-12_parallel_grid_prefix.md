# Bounded native CPU grid prefixes

## Selected execution policy

Dense-grid construction keeps its histogram and stable reverse scatter. The
inclusive cell-count prefix can now use scoped native CPU workers under the
existing `spatialrust-search/parallel` feature. It is selected only for tables
with at least **8,388,608 entries (including the CSR sentinel)** and at least
**1,024 entries per point**. Smaller/denser tables and single-worker calls retain
the scalar loop. The scalar dispatcher/scan are inline; the threaded body is
outlined to avoid enlarging the ordinary path.

Workers are capped by available CPU parallelism and useful 262,144-entry chunks,
using the existing staging policy. Each disjoint band computes a local inclusive
prefix and returns its total. All first-phase workers join before the caller
computes bases in band order. A second scope adds bases to disjoint bands,
skipping zero bases, and joins before reverse scatter. No barriers can strand
workers if a first-phase join propagates a panic.

For valid u32 total mass, local prefixes, bases and adjusted prefixes stay within
that mass. The final prefix is the original inclusive sum at every entry;
integer associativity introduces no floating reduction or neighbor reordering.
The stable scatter therefore produces the same indices, CSR starts and sentinel,
and the GPU gathers see identical ordered inputs.

Production source:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/grid_prefix.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid.rs`

The original point/cell-sized arrays remain unchanged. Additional totals/handles
and scoped scheduling metadata scale with active workers, without another
cell-sized cursor or point-ID cache. Thread stacks/runtime overhead still exist;
the budget is per call, not a process-wide CPU or memory limit. Concurrent callers
have independent budgets and owned outputs. Mapping/scattering and checked bounds
remain serial. No dependency, public signature, unsafe code or device copy is added.

`gpu-wgpu` now activates the existing dependency-free search `parallel` feature
for native host grid preparation. Cargo feature unification also activates the
existing native dense-root dispatcher, retaining its minimum-root semantics.
WASM prefix and dense-root dispatch explicitly retain serial execution even with
that feature enabled. Other explicit CPU parallel-query APIs retain their own
existing contracts. This change does not establish browser thread performance.
GPU storage, shaders, pass/submission counts, transfers and readback stay unchanged.

## Correctness and source replay

CPU tests:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/grid_prefix_tests.rs`.

**672,414,960 word comparisons** cover independent u64 running sums, zero/dense/
sparse counts and maximum u32 mass, empty/partial bands, scalar/2/6-worker paths
and public dispatch around the cutoff. Requested zero workers normalize to a
scalar path. Large 4,194,304/8,388,608-cell grid construction matches frozen stable
indices and offsets. Four concurrent callers retain two results each after later
builds. These tests cover repeated worker variants; the word count is comparisons,
not unique input values.

Native GPU tests/replay:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_prefix_tests.rs`.

**2,237,056 exact output words** cover 17 / 257 / 4,097 / 65,537 points on both
large grid shapes, radius normals and two covariance epsilon values including
negative zero. Cases include the selected parallel path and scalar controls.
Both modes use current shaders, pipeline cache, scratch executor and readback.

The serial reverse-scatter builder is frozen verbatim from parent
`4ad7c4d9f8a15f8c62532edf29c24fb41a347f16` in each crate's local
`tests/fixtures/grid_prefix_serial.rs`. GPU replay wrappers/preparation preserve
the parent's code after only import/function routing and formatting substitutions.
The serial comparison also retains the prior cursor removal and checked bounds;
no earlier algorithm optimization or shader difference is mixed into this timing.

## Validation

Windows, rustc 1.98.1; model identity is omitted from adapter diagnostics.

| Check | Result |
| --- | --- |
| Search all-feature tests | 55 passed, 0 failed, 5 ignored timing diagnostics |
| Search minimal tests | 22 passed, 0 failed, 3 ignored timing diagnostics |
| Native GPU image/AoSoA tests | 174 passed (147 unit + 27 integration), 0 failed, 17 ignored |
| Existing CPU facade root/bounds smoke | 2 + 2 passed |
| Native GPU facade bounds/rows/fallback smoke | 1 each, explicitly run |
| Search/GPU all-feature all-target Clippy | `-D warnings` passed |
| All-feature search / GPU image/AoSoA WASM | Compile checks passed |
| Formatting and whitespace | Passed |

WASM checks establish compilation only. Native adapter failures are not silently
skipped. Correctness/build jobs finish before timing; diagnostics run alone.

## Measurement boundary

The ignored release diagnostics are:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/grid_prefix_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_prefix_benchmark.rs`

```powershell
cargo test -p spatialrust-search --release --no-default-features --features parallel paired_serial_current_grid_prefix -- --ignored --nocapture --test-threads=1
cargo test -p spatialrust-gpu --release --features gpu-wgpu paired_serial_current_cpu_grid_prefix -- --ignored --nocapture --test-threads=1
```

Each condition uses three warmups per version and twenty alternating-order pairs.
CPU timers include completed histogram/prefix/scatter/allocation, with input
preparation, equality checking and returned-vector destruction outside. GPU timers
include validation, checked bounds, CPU grid, uploads, compute, mapping wait and
normal/covariance decoding; word conversion/checks/drop are outside. Both versions
use the same current shader/executor. No GPU pass timestamps are requested.
The host reports 12 available CPU workers. Native GPU completion uses Vulkan and
a discrete adapter. Release uses thin LTO and one codegen unit.

Tables use median milliseconds and geometric mean of paired previous/current
ratios, with 95% paired log-ratio bootstrap intervals (10,000 resamples, seed
20261012). Ratios greater than one favor the revision; they need not equal ratios
of marginal medians. These intervals describe observed pairs, not systematic
host bias, unmeasured hardware or all possible data distributions.

Early prototypes selected from 4,194,304 entries, then from 8,388,608 entries
without a point-density gate. Repeated completed-API measurements found adverse
4-million-cell cases and an 8-million-cell 65,536-point covariance case (ratio
0.939 [0.902, 0.975] in one survey). These observations motivated the conservative
size/density selection and outlined threaded path. Those policies are not shipped;
private diagnostics retain their adverse samples. Scalar control timings can
also change through compiler/host effects and must not be credited to parallel
work. Only the selected policy is reported in the final tables below.


## Selected policy: two isolated surveys

Each cell count below excludes the extra CSR sentinel. `parallel` means the
selected policy used scoped workers; `scalar` is a control outside that policy.
All twenty pairs are retained for each of 42 conditions.

### Completed CPU grid construction

| Survey | Operation | Points | Cells | Path | Previous median ms | Current median ms | Paired ratio [95% interval] |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | grid build | 257 | 1,048,576 | scalar | 1.721 | 1.594 | 1.103 [1.080, 1.130] |
| 1 | grid build | 257 | 4,194,304 | scalar | 6.110 | 6.085 | 1.043 [0.993, 1.092] |
| 1 | grid build | 257 | 8,388,608 | parallel | 13.870 | 6.709 | 2.052 [1.961, 2.142] |
| 1 | grid build | 65,536 | 1,048,576 | scalar | 6.130 | 6.128 | 0.951 [0.868, 1.034] |
| 1 | grid build | 65,536 | 4,194,304 | scalar | 17.220 | 16.288 | 1.059 [1.022, 1.097] |
| 1 | grid build | 65,536 | 8,388,608 | scalar | 28.523 | 28.125 | 1.019 [0.974, 1.066] |
| 1 | grid build | 1,048,579 | 1,048,576 | scalar | 71.327 | 67.612 | 1.045 [0.980, 1.116] |
| 1 | grid build | 1,048,579 | 4,194,304 | scalar | 183.720 | 184.061 | 0.957 [0.892, 1.020] |
| 1 | grid build | 1,048,579 | 8,388,608 | scalar | 191.692 | 196.705 | 0.979 [0.922, 1.035] |
| 2 | grid build | 257 | 1,048,576 | scalar | 1.500 | 1.350 | 1.077 [0.999, 1.153] |
| 2 | grid build | 257 | 4,194,304 | scalar | 5.911 | 5.404 | 1.104 [1.080, 1.134] |
| 2 | grid build | 257 | 8,388,608 | parallel | 11.189 | 6.014 | 1.887 [1.828, 1.954] |
| 2 | grid build | 65,536 | 1,048,576 | scalar | 6.143 | 5.853 | 1.059 [1.019, 1.108] |
| 2 | grid build | 65,536 | 4,194,304 | scalar | 16.901 | 16.196 | 1.020 [0.998, 1.042] |
| 2 | grid build | 65,536 | 8,388,608 | scalar | 28.949 | 28.047 | 1.049 [1.005, 1.094] |
| 2 | grid build | 1,048,579 | 1,048,576 | scalar | 91.225 | 91.402 | 1.015 [0.963, 1.073] |
| 2 | grid build | 1,048,579 | 4,194,304 | scalar | 177.864 | 183.814 | 0.996 [0.957, 1.042] |
| 2 | grid build | 1,048,579 | 8,388,608 | scalar | 201.790 | 202.608 | 1.000 [0.995, 1.006] |

### Completed public GPU normal/covariance APIs

| Survey | Operation | Points | Cells | Path | Previous median ms | Current median ms | Paired ratio [95% interval] |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | normal | 257 | 4,194,304 | scalar | 10.478 | 10.514 | 0.998 [0.980, 1.015] |
| 1 | covariance | 257 | 4,194,304 | scalar | 10.512 | 10.645 | 0.983 [0.958, 1.003] |
| 1 | normal | 257 | 8,388,608 | parallel | 19.250 | 15.014 | 1.310 [1.286, 1.337] |
| 1 | covariance | 257 | 8,388,608 | parallel | 21.395 | 17.646 | 1.241 [1.214, 1.266] |
| 1 | normal | 4,096 | 4,194,304 | scalar | 15.618 | 15.629 | 0.974 [0.928, 1.025] |
| 1 | covariance | 4,096 | 4,194,304 | scalar | 16.406 | 16.262 | 0.978 [0.943, 1.014] |
| 1 | normal | 4,096 | 8,388,608 | parallel | 34.496 | 30.710 | 1.163 [1.130, 1.208] |
| 1 | covariance | 4,096 | 8,388,608 | parallel | 37.922 | 32.480 | 1.153 [1.076, 1.226] |
| 1 | normal | 65,536 | 4,194,304 | scalar | 32.429 | 32.798 | 0.988 [0.960, 1.016] |
| 1 | covariance | 65,536 | 4,194,304 | scalar | 33.087 | 33.154 | 1.018 [0.989, 1.054] |
| 1 | normal | 65,536 | 8,388,608 | scalar | 51.890 | 50.509 | 0.982 [0.934, 1.032] |
| 1 | covariance | 65,536 | 8,388,608 | scalar | 51.503 | 50.250 | 1.025 [0.989, 1.065] |
| 2 | normal | 257 | 4,194,304 | scalar | 10.560 | 10.547 | 1.004 [0.986, 1.022] |
| 2 | covariance | 257 | 4,194,304 | scalar | 10.591 | 10.569 | 0.996 [0.984, 1.010] |
| 2 | normal | 257 | 8,388,608 | parallel | 19.036 | 15.054 | 1.266 [1.245, 1.286] |
| 2 | covariance | 257 | 8,388,608 | parallel | 19.297 | 15.410 | 1.264 [1.233, 1.297] |
| 2 | normal | 4,096 | 4,194,304 | scalar | 15.323 | 15.313 | 1.000 [0.987, 1.012] |
| 2 | covariance | 4,096 | 4,194,304 | scalar | 16.125 | 15.654 | 0.992 [0.945, 1.031] |
| 2 | normal | 4,096 | 8,388,608 | parallel | 34.396 | 30.455 | 1.118 [1.097, 1.137] |
| 2 | covariance | 4,096 | 8,388,608 | parallel | 34.246 | 30.644 | 1.125 [1.103, 1.149] |
| 2 | normal | 65,536 | 4,194,304 | scalar | 32.092 | 31.766 | 1.009 [0.991, 1.028] |
| 2 | covariance | 65,536 | 4,194,304 | scalar | 32.760 | 32.667 | 0.994 [0.985, 1.004] |
| 2 | normal | 65,536 | 8,388,608 | scalar | 48.773 | 50.345 | 1.017 [0.978, 1.060] |
| 2 | covariance | 65,536 | 8,388,608 | scalar | 49.878 | 49.994 | 0.997 [0.965, 1.030] |

At 257 points / 8,388,608 cells, complete CPU construction ratios are
2.052 [1.961, 2.142] and 1.887 [1.828, 1.954] across the two surveys. Selected
GPU conditions at 257 / 4,096 points and the same cell count have ratios
1.118–1.310 (about 11–24% less completed-call time from paired ratios), with
all corresponding intervals above one. Those results support only this sparse,
large-table workload family on the measured host. They do not establish general
point-cloud, dense-table, browser or GPU-kernel speedups.

Scalar controls have no interval entirely below one in either selected-policy
survey. Some CPU control intervals are above one despite retaining scalar work;
these are compiler/host observations rather than a parallel acceleration claim.
Dense 65,536 / 1,048,579-point cases retain scalar prefixes and make no performance
promise. The conservative gate avoids the adverse prototype cases without
changing neighbor order or adding hidden device transfers.

Private full-pair and validation logs are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/parallel-grid-prefix-*`.
The selected summary is `parallel-grid-prefix-selected-measurements.json` there.
Prototype samples and compiler/adapter diagnostics remain private; no model
identity is included in tracked notes or benchmark output.
