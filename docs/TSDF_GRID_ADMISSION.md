# Fallible dense TSDF grid construction

Slice 146AF rejects invalid dense-grid geometry and unrepresentable field
capacities through the existing `TsdfVolume::try_new` result. Previously a
non-finite origin or overflowing voxel center could enter the volume; excessive
dimensions reached the infallible Vec constructor and panicked on capacity.

The [prospective contract](../benchmarks/tsdf-grid-plan.json) was committed
before native execution. The [receipt](receipts/2026-10-21_tsdf_grid.json) binds
the original failure log, tested source, final checks and preserved artifacts.

## Admission and initialization

Retain positive finite voxel size/truncation and nonzero dimensions. Require
every origin component finite. Multiply the three dimensions with checked
arithmetic and require each f32 field to fit Rust's `isize::MAX` byte limit.
These checks occur before any field allocation and also keep signed grid-index
conversion and flat offsets representable.

For each axis, evaluate its last voxel center using the same f32 expression as
`index_to_world`: `origin + ((dimension - 1) as f32 + 0.5) * voxel_size`. Reject
non-finite results, including an overflowing intermediate offset that a f64
world-coordinate-only check could miss. Positive offsets are monotonic, so
finite origins and last centers bound all cell centers. Rounded or coincident
finite centers, including subnormal scales, keep the existing acceptance policy.
This does not establish a physical grid resolution or general numerical accuracy.

Reserve both distance and weight fields with `try_reserve_exact` before
initializing either. A reported reservation failure returns the existing
`SceneError::InvalidConfiguration`; previously reserved storage is dropped.
Successful construction keeps exact truncation-distance and positive-zero
weight values, dimensions and x-fastest layout. No arbitrary volume-size cap
or public error variant is added.

Fallible reservation does not guarantee recovery from operating-system
memory overcommit or an allocator abort. Representable requests can still be
too large for an application's memory budget. Tests never attempt such a large
allocation, exhaust physical memory or claim an observed allocator-failure run.

## Verification and compatibility

With production unchanged, 17 new synthetic controls reproduce **9 failures**;
**36 tests pass** in the all-feature baseline. Its failure log is retained.
Invalid cases cover each origin axis, first/last center overflow, an overflowing
f32 offset despite a finite exact world coordinate, dimension products and
field byte limits. Capacity cases are mathematically impossible Vec requests,
so the original implementation raises capacity panics before allocator access.

After the fix, scene default/all-feature suites pass **43/45 tests**. Existing
invalid scalar/zero-dimension handling and accepted extreme, subnormal,
non-cubic and exact-bit initial fields are also checked. Formatting preparation
is recorded separately; baseline/final test semantics are unchanged.

All non-constructor TSDF methods, existing support tests, other scene production
files, every other crate, public fields/signatures/error variants, dependencies
and device ownership remain unchanged. Integration and meshing mathematics,
submitted-cloud counts, canonical source/calibration admission and fusion
ledger/readback retain their contracts. Related full-bag/E2E, ROS2, mapping,
viewer, calibration controls, all-target Clippy and formatting provide regression
coverage. Existing three-OS scene tests and all 131 named checks gate merging.
These code results remain separate from physical reconstruction acceptance.

## Storage and remaining qualification

Use `E:/SpatialRustBuild/tsdf-grid-v1` and the existing shared Cargo cache, with
128MiB owned logs/fixtures, 2GiB positive cache growth, 2.125GiB combined growth,
four build/Rayon/test workers, 1,800 seconds per command and at least 20GiB free.
The receipt records measured usage. All 493 prior scientific JSON, 42 frozen
native artifacts, pinned references and completed room outputs remain
hash-identical. No real sensor payload read/download/copy, real-motion fit or
frozen binary rebuild occurs.

Native complete-room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, matching canonical measured clock uncertainty and
physical frame accuracy, and complete mapping/semantic/Viewer/interchange/release
acceptance remain open. Constructor admission does not qualify operational 80%.
