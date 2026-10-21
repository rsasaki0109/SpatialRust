# TSDF support admission before clamping

Slice 146AE fixes boundary contamination by samples whose discretized TSDF
support is entirely outside the volume. The old implementation clamped both
support endpoints before checking intersection. A far-away sample could then
update a boundary plane, edge or corner and even create a spurious mesh.

The [prospective contract](../benchmarks/tsdf-support-plan.json) was committed
before native execution. The [receipt](receipts/2026-10-21_tsdf_support.json)
binds original failures, final source, logs and preservation checks.

## Discrete support contract

The existing support is the cube `point +/- truncation`. For each axis the
original f32 subtraction/division/floor and saturating signed-index conversion
produce an inclusive raw range. Intersect that range with `[0, dims-1]` before
clamping either endpoint. If any axis is disjoint, leave all weights/distances
unchanged. Otherwise use the same clipped endpoints and original update loop.

For an X grid with origin 0, voxel size 1, four cells and truncation 0.5:

| Sample X | Raw X cells | Valid X cells | Behavior |
| --- | --- | --- | --- |
| 10.5 | 10 through 11 | none | Skip; previously updated cell 3 |
| -10.5 | -11 through -10 | none | Skip; previously updated cell 0 |
| 4.25 | 3 through 4 | 3 | Retain partial overlap |
| -0.25 | -1 through 0 | 0 | Retain partial overlap |
| 4.5 | 4 through 5 | none | Skip |
| -0.5 | -1 through 0 | 0 | Retain inclusive upper floor endpoint |

This preserves the existing floor quantization, including its endpoint
asymmetry. It does not introduce voxel-center membership, spherical support,
projective visibility or a ray-marching algorithm. The public method comment
now describes the discretized cube accurately.

## Preserved computation and APIs

Overlapping samples keep the same ray normalization, signed-distance equation
and clamp, z/y/x visitation order, weighted update and meshing. Submitted XYZ,
columnar cloud and explicitly posed cloud routes all delegate to the corrected
point integration. Poses still transform both points and the sensor origin.

Cloud methods still return `cloud.len()`, including skipped samples. Those
numbers and the full-bag ledger describe submissions, not accepted points or
voxel contributions. No new public field, method signature, dependency, thread,
unsafe code, point payload copy or GPU transfer is introduced. The check adds
only fixed stack arrays and O(1) work per sample. Constructor/allocation policy
and broader extreme-value numerical hardening remain separate.

## Verification

Production stayed unchanged while 21 support cases and the frozen parent
integrator were committed. The all-feature baseline reports **10 passes and
18 failures**, including all six disjoint faces and the spurious-mesh case.
Its failure log is retained. Three new compatibility controls and the seven
original tests already pass in that baseline.

After the fix, scene default/all-feature suites pass **26/28 tests**. Independent
full-grid enumeration checks **729 dyadic sample positions**, using exact f64
support bounds rather than production clamping helpers. Original f32 field
math remains the comparison target. A frozen copy of the parent integrator
checks distance/weight words after **343 overlapping repeated updates**.
Hand-checked partial-face masks and endpoints, edge/corner rejection, finite
signed-index saturation, translated/scaled grids, invalid/degenerate depths,
mixed XYZ/cloud/posed routes, submission counters and mesh rejection also pass.

Full-bag/E2E examples, ROS2, mapping, viewer-all, calibration controls, scene and
ROS2 all-target Clippy, and formatting provide related regression coverage.
The existing Linux/Windows/macOS mapping job now explicitly tests scene all
features; all 131 named PR checks are required without changing check names.
These local results are separate from the complete CI merge gate.

All crate trees outside scene, other scene production files, public APIs,
source/calibration/clock admission, scan transport, fusion ledger/readback,
core/GPU/Python, dependencies and physical acceptance gates remain unchanged.

## Storage and remaining qualification

Use the fresh `E:/SpatialRustBuild/tsdf-support-v1` root and shared Cargo cache,
with a 128MiB owned log/fixture cap, 2GiB positive cache growth, 2.125GiB combined
growth, four build/Rayon/test workers, a 1,800-second command deadline and at
least 20GiB free. Actual sizes and terminal command records are in the receipt.
All 493 prior scientific JSON, 42 frozen native artifacts, pinned references
and completed room outputs remain hash-identical. No real sensor read/download/
copy, real-motion fit or frozen binary rebuild occurs.

Native complete-room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, matching canonical measured clock uncertainty and
physical frame accuracy, and complete mapping/semantic/Viewer/interchange/release
acceptance remain open. Synthetic support correction does not qualify physical
accuracy, unique voxel/scene coverage or operational 80%.
