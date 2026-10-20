# Mapping sensor transport

Slice 146Y corrects scan-to-TSDF coordinates in the full-bag mapping and bounded
E2E examples. Stored scan odometry maps the first front frame into the current
front frame; the old consumer used that direction directly on current points.
It also applied front-derived motion independently in the rear sensor frame,
so rotation failed to carry the rear sensor's lever arm with the body.

The [prospective contract](../benchmarks/mapping-sensor-transport-plan.json) was
committed at `3eb45f0fc2535138bbccc546b3c0116f48b4c20a` before tests or implementation. The
[receipt](receipts/2026-10-20_mapping_sensor_transport.json) binds tested source
`ebfb18c577a94ab935ed20027e2f2624a2df8a53`, exact commands, hashes and retained failures.

## Coordinates and admission

`A_T_B` maps B points into A. Calibration provides `body_T_front` and
`body_T_sensor`. The fixed volume gauge is the first body frame:

```text
volume_T_current_front = body_T_front * inverse(current_front_T_first_front)
volume_T_current_body  = volume_T_current_front * inverse(body_T_front)
volume_T_current_sensor = volume_T_current_body * body_T_sensor
```

A private shared example helper applies this constant-work transport to both
sensors. The bounded E2E smoke uses identity extrinsics and the first front
frame gauge. Inputs and all intermediate/final transforms must be finite before
TSDF insertion; proper rigid inputs still come from calibration and matching.
No new quaternion-normalization policy is introduced.

Source path/size/SHA identity, registered clock/frame readiness, actual source
frame IDs, timestamp matching and bounded IO remain enforced. Rear scans use
the existing matched front timestamp, without motion interpolation; a finite
sync window does not prove simultaneous capture. Synthetic transforms do not
supply measured physical calibration or clock uncertainty.

Executing the existing E2E tests on Windows also exposed rooted output paths
accepted by `is_absolute()`. A separate [prospective path contract](../benchmarks/mapping-sensor-transport-path-contract.json)
at `faff6233eebe2b26756ad4f4b30e43d440f69f03` preceded the additional regression
and fix. Output directories now require nonempty native Normal/CurDir components;
RootDir, Prefix and ParentDir reject. This is lexical admission and does not
resolve filesystem links. Storage/free-space and source/calibration gates remain.

## Independent tests

Twelve shared coordinate/finite cases include signed-axis observations of
known static world points, translating/rotating bodies, nonidentity Rx/Ry sensor
extrinsics, sensor origins, first pose and a counterexample to inverse-only rear
transport. The same suite checks non-finite inputs and finite-input overflow.
The shared cases execute in both example harnesses. One further test checks
output escape components, including native drive-relative/UNC forms on Windows.

The original consumer algebra was extracted without changing its behavior:
**5 passed, 10 failed** before the fix. Reversing the extraction reproduces all
other original consumer code. The first compile attempt used the wrong vector
length method in tests; that log is retained separately. The first E2E run also
retains its existing Windows path-test failure (15 pass/1 fail), followed by a
prospective path baseline (15 pass/2 fail). A first all-target Clippy attempt
found two existing needless borrows in the health example; the behavior-preserving cleanup and failed log are also retained.
No failure is discarded or relabeled.

| Local check | Result |
| --- | --- |
| Full-bag example | 15 passed |
| Bounded E2E example, including checkpoint metadata/lineage | 17 passed |
| ROS2 default / all features | 0 / 4 passed |
| Mapping all features and scan direction bridge | 24 passed |
| Existing calibration tests | 6 passed |
| ROS2 all-feature/all-target Clippy | Pass, warnings denied |
| Workspace formatting | Pass |

Three dedicated CI jobs execute both example harnesses, ROS2 all-feature tests
and selected-example Clippy on Linux, Windows and macOS. Local counts above
are separate from CI and do not represent real-data acceptance runs.

## Storage and qualification

All 493 prior scientific JSON, 42 frozen native artifacts, pinned reference
sources and completed room-audit outputs remain hash-identical. Existing
core/math/mapping/GPU/vision/Python/sync/Viewer and dependencies are unchanged.
There are no sensor downloads, payload copies, real-motion fits or frozen builds.

Owned logs/metadata occupy **373,034 bytes** at
`E:/SpatialRustBuild/mapping-sensor-transport-v1`. The reused Cargo cache grows
by **1,763,436,631 bytes**, below the fixed 3GiB increase
limit. Four workers, a 128MiB owned-log cap, combined 3.125GiB growth cap and at
least 20GiB free are monitored. Failed logs and old evidence remain preserved.

Native room accuracy and genuinely unobserved motion qualification, both raw
Linux hardware classes, matching canonical measured clock uncertainty/physical
frame accuracy and complete canonical mapping/semantic/Viewer/interchange/release
acceptance remain open. These coordinate and path fixes do not establish
operational 80% qualification.
