# Checked dense-grid bounds

## Reproduced failures

The shared `grid_bounds` planner previously converted a floored f32 span to i64,
added one without checking, and narrowed to u32 before testing the cell cap.
The product of three narrowed dimensions was also unchecked. Invalid arrays,
cell scales and coordinates were not validated by this public helper.

A finite x width of 2^32 at cell size one produced `[1, 1, 1]` through u32
wrapping, so the fit predicate incorrectly accepted a huge grid. The new
rejection regression failed against the parent. The native normal API likewise
incorrectly succeeded for that span, so the new native rejection test failed.
For `[f32::MIN, f32::MAX]`, f32 subtraction overflowed to infinity; the saturated
i64 conversion plus one panicked in the debug build instead of returning an
error. The separate overflow regression reproduced that panic before the fix.

Private reproduction logs:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/grid-bounds-wrap-repro.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/grid-bounds-overflow-repro.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/grid-bounds-gpu-repro.log`

These three failing regressions now pass. The rejection suite also passes in
release mode, where overflow checks cannot be relied upon for safety.

## Validation and compatibility

Production source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid.rs`.

The planner now:

1. Checks equal XYZ lengths and a positive, non-NaN cell size whose reciprocal
   is finite. Tiny positive sizes that overflow the reciprocal return errors.
2. Checks each coordinate's finiteness in the existing min/max scan, avoiding
   silent NaN omission and undefined grid placement of infinities.
3. Preserves the valid empty result: infinite origin and unit dimensions.
   A zero reciprocal for positive-infinite cell size returns a one-cell grid
   after coordinate validation, preserving the existing unbounded-radius path.
4. Retains the original f32 subtraction/multiplication/floor for finite cell
   sizes. Nonfinite scaled spans reject rather than reaching integer casts.
5. Adds one to each floored span and accumulates cell products with checked u64
   arithmetic, enforces the unchanged 64,000,000-cell cap before checked u32
   narrowing, and returns `SpatialError::InvalidArgument` on invalid planning.

Every accepted nonempty finite-scale span is nonnegative. Its checked dimension
is at least one; checked partial products remain within the cap, so accepted
u32 dimensions are positive and fit i32 under the current cap. The fit predicate
returns false on every planning error. No dimension can wrap into a smaller
accepted grid, and array-length errors no longer panic through indexing.

Bounds remain O(points) with constant stack state and no grid allocation.
Errors may allocate their diagnostic strings. No new worker, dependency,
feature, unsafe code, device copy, queue submission or placement policy is
introduced. Invalid inputs are now explicitly rejected; they are not a promise
to preserve prior NaN/overflow behavior. Empty normal/covariance API shortcuts
still return their prior empty results before radius checking.

Standalone GPU radius normals and plane covariances already call this helper
before counting-sort and before the shared executor. They therefore reject
invalid planning without acquiring GPU scratch or initializing a point pipeline.
Their existing point/output/device checks and later cell/sentinel binding checks
remain in place. No GPU shader, gather order, solver, output layout, transfer or
readback behavior changes for admitted inputs. Positive-infinite radius remains
supported for finite coordinates, including extreme widths that would overflow
f32 subtraction; two-point native/facade fallback results are checked explicitly.
This does not extend finite-result guarantees for extreme-coordinate covariance
math, change KD-tree handling, or validate raw prebuilt-grid construction APIs.

## Regression coverage

CPU regressions:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-search/src/uniform_grid_bounds_tests.rs`.

- Five tests cover the explicit overflow reproduction; wrapped u32 dimensions;
  f32 subtraction/scaling overflow; huge finite u64 casts; cross-axis u64 product
  overflow; malformed axis lengths; NaN/infinite coordinates on every axis;
  nonpositive/NaN/infinite-negative cell sizes and nonfinite reciprocals.
- **300 frozen cases** retain all three origin words and dimensions for ten
  sizes, five finite volume/sheet/cluster/chain/coincident fixtures and six cell
  sizes including 0.3 and positive infinity. Signed-zero origins are checked.
  The previously frozen planner is unchanged; its source matches parent
  `716f95469eea05320693e5e90883f8a64057f4b6` before this correction.
- Independent dimension expectations check the exact `400^3` cap and first
  excess product without constructing those grids, the last representable
  single-axis f32 span below the cap, and reciprocal bit patterns 0x00200000 /
  0x00200001 at the overflow boundary. Valid empty and unbounded-radius results
  retain their original dimensions/origin contract.
- Dense-grid component labeling also returns errors for the extreme span cases
  before its sort/union phase. Existing stable CSR and component regressions
  remain green.

Native GPU regression:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/grid_bounds_tests.rs`.

Twenty-three malformed grid cases are passed to both public radius normal and
covariance APIs, on both a cold runtime and a runtime with those pipelines
warmed: **92 rejected calls**. Every call returns InvalidArgument and retains
all scratch-pool statistics; the cold cache stays uninitialized and wgpu
validation scopes remain empty. Adapter initialization errors fail the test.

Facade regressions:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/grid_bounds_smoke.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_grid_bounds_smoke.rs`

CPU facade checks run under minimal and parallel features. The native GPU facade
check is ignored by default with an explicit adapter requirement, then run
locally with `--include-ignored`. It checks rejection without scratch changes
and admitted extreme-width, positive-infinite-radius two-point fallback outputs.

## Validation receipt

Windows host with rustc 1.98.1; local adapter model identity is omitted.

| Check | Result |
| --- | --- |
| Search all-feature tests | 51 passed, 0 failed, 4 ignored timing diagnostics |
| Search minimal tests | 19 passed, 0 failed, 2 ignored timing diagnostics |
| Minimal release bounds regressions | 5 passed |
| Native GPU image + AoSoA tests | 173 passed (146 unit + 27 integration), 0 failed, 16 ignored |
| CPU facade minimal / parallel | 2 passed each |
| Native GPU facade bounds smoke | 1 passed, explicitly run |
| Search/GPU all-feature all-target Clippy | `-D warnings` passed |
| New CPU/GPU facade Clippy | `-D warnings` passed |
| Minimal search / GPU image + AoSoA WASM | Compile checks passed |
| Formatting and diff whitespace | Passed |

WASM validation establishes compilation only. No new throughput benchmark or
performance gain is claimed for this input-validation correction. Raw local
validation diagnostics remain ignored beneath
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with the
`grid-bounds-` prefix.
