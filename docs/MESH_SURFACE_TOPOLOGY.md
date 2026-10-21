# Marching-tetrahedra surface topology

Slice 146AG fixes the ordering of cut vertices in six quadrilateral cases and
the winding of the three-inside `1011` case. Previously two triangles could
share a polygon boundary instead of an interior diagonal, producing overlap
and missing area. One triangular case faced decreasing field values.

The [prospective contract](../benchmarks/mesh-topology-plan.json) predates native
execution. The [receipt](receipts/2026-10-21_mesh_topology.json) binds the original
regressions, final source, every retained failure and measured storage.

## Surface contract

The existing six positively oriented tetrahedra still cover each cube. A corner
is inside exactly when its value is below the isolevel. In each two-inside case,
order the four crossing edges around the polygon boundary before forming the
triangle fan. The shared diagonal joins disjoint tetrahedron edges; every
surface boundary segment lies on a tetrahedron face and joins crossing edges
with a common corner. Winding points toward increasing field values for the
existing positive-oriented tetrahedra. The other seven nonempty case rows stay
unchanged.

Only seven static case rows change. Interpolation, its existing endpoint/epsilon
snapping, cube decomposition, fixed four-vertex scratch, triangle append/index
layout and output types remain. No vertex welding, new normal buffer, degenerate
filtering, heap scratch, dependency, unsafe code or device transfer is added.
TSDF construction, integration/support, weights, cloud submission counts and
source/calibration/clock admission retain their contracts.

This fixes local case topology; it does not establish a globally watertight
mesh for arbitrary fields. Existing epsilon snapping and extreme/non-finite
interpolation behavior remain separate. Affine controls use nondegenerate
finite fixtures away from the existing snapping thresholds.

## Independent checks

Twenty committed controls run against unchanged production first. The genuine
all-feature baseline reports **48 passes and 17 failures**, including all six
quadrilateral masks, `1011` winding and full-volume plane checks. Tests remain
byte-identical from that baseline through the final implementation.

After the fix, scene default/all-feature suites pass **63/65 tests**. Independent
edge enumeration, f64 affine gradients, boundary/diagonal incidence and polygon
area cover all 16 masks, unequal magnitudes, all six cube tetrahedra, translated/
scaled coordinates and shifted isolevels. Complementary signs reverse normals.
There are **499 case checks**, including empty masks. Append offsets and original
interpolation endpoints/epsilon behavior also pass.

Twelve tiny analytic plane volumes check all axes, translated/scaled grids and
reversed fields. Extracted triangles have the expected area, direction and
outer perimeter; each interior edge occurs twice and each boundary edge once.
These synthetic checks exercise complete extraction across neighboring cells.
They do not measure real scene accuracy, sensor coverage or timing.

The initial fixture scalar-operator compilation failure and the first correction's
remaining `0011` orientation failures are retained with their exact sources.
The accepted regression baseline follows the fixture-only correction. Final
source-bound checks supersede earlier format/default-scene attempts without
overwriting them. Related full-bag/E2E, ROS2, mapping, viewer, calibration controls,
scene/ROS2 all-target Clippy and formatting also pass. Existing three-OS scene
coverage and all 131 named CI checks gate merge.

## Storage and remaining qualification

Use `E:/SpatialRustBuild/mesh-topology-v1` with the shared Cargo cache, 128MiB
owned logs/fixtures, 2GiB positive cache growth, 2.125GiB combined growth, four
workers, 1,800 seconds per command and at least 20GiB free. All 493 prior
scientific JSON, 42 frozen native artifacts, pinned references and completed
room outputs remain hash-identical. No real sensor payload read/download/copy,
real-motion fit or frozen binary rebuild occurs.

Native complete-room accuracy/unobserved motion, both qualifying raw Linux
hardware classes, matching canonical measured clock uncertainty and physical
frame accuracy, and complete mapping/semantic/Viewer/interchange/release
acceptance remain open. These geometry checks do not establish operational 80%.
