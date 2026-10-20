# Bounded fusion submission coverage

Slice 146AC adds `full-bag-mapping.coverage.json` to the private full-bag
mapping example. The existing mapping-state schema remains version 1. No
library API, dependency, matching policy, point geometry or GPU transfer changes.
The [prospective contract](../benchmarks/fusion-coverage-plan.json) was committed
before native execution. The [receipt](receipts/2026-10-21_fusion_coverage.json)
binds the tested source, command logs and preservation checks.

## Identity and meaning

Each descriptor names its immutable sorted `MemoryEpisode` index, front/rear
role, raw and corrected nanoseconds, source point count, successful call count
and submitted point count. Indices identify retained episode records, not
physical bag messages. No addresses or copied point payloads enter the ledger.

The context binds verified input and calibration file SHA-256/size, selected
topics, sensor/root frames, corrected clock ID, registered offset/drift and
uncertainty, the collection anchor, chunk size and sync window. The calibration
file checksum binds the registered transform graph as well as its frame names.
Descriptions are checked against the original episode and an independently
supplied expected context. Unknown JSON fields reject; decoding alone does not
validate evidence. Production publication accepts only a privately completed,
validated ledger, and no general-purpose file reader is introduced.

The unchanged `DeterministicReplayer` chooses the nearest eligible rear record
inside inclusive time/quality windows. A distance tie picks the first stable
record. Rear records remain reusable. Completion replays this same policy to
check every association index and absolute corrected-time delta. This adds a
verification traversal; it is not a throughput improvement claim.

Every front record is submitted once, including unmatched fronts. Each chosen
rear association adds a rear call, including repeated use of one rear record.
A visit is recorded only after `integrate_cloud_with_pose` succeeds. Its current
return value is `cloud.len()`: submitted points, not points that changed a voxel.
The ledger rejects omitted/duplicate calls, wrong returned point counts, integer
overflow and contradictions with the existing TSDF and source totals.

| Synthetic retained records, four points each | Successful calls / points | Distinct visited / unvisited records | Rear reuse |
| --- | --- | --- | --- |
| Front 10/20ns, rear 11/100ns, 2ns window | 3 / 12 | 3 / 1 | 0 |
| Front 10/20/30ns, rear 20ns, 100ns window | 6 / 24 | 4 / 0 | 2 extra calls |

Both rows run through actual synthetic SQLite collection, ICP, TSDF and artifact
publication. Synthetic identity calibration is explicitly labeled; it provides
no measured sensor synchronization, extrinsic accuracy or physical coverage.

## Bounds and publication

The example now rejects more than **65,536 retained records before odometry**.
Descriptors, associations, the ledger's validation counters and conservative
bounded headers/writer storage have an **8MiB** budget checked before requested
allocation and again against actual vector capacities. Each identity label is
at most 1,024 bytes, and total context string capacity is bounded. These limits
are separate from source/episode budgets, existing replay/mapping allocations,
allocator overhead and process RSS.

JSON streams through a 16KiB buffer with a **32MiB** byte cap including its final
newline. A `create_new` temporary file is synchronized, then a same-directory
hard link atomically creates the destination without replacing an existing file.
Filesystems without hard-link support fail closed. Only the exact temporary
file created by that attempt is removed; an existing temporary file is left
untouched. No complete output JSON string is allocated in production.

Publication and checksumming precede the parent success state. The new
`fusion-coverage` artifact role and existing `DatasetManifest` carry the file's
actual SHA-256 and size. Blocked calibration/input/execution keeps existing
diagnostics and omits this coverage file. Publication errors prevent a parent
success receipt. Existing output-directory and storage preflight rules remain.

## Verification and limits

Thirteen characterization cases passed against the original replayer before
implementation: ties, inclusive/outside windows, uncertainty, reuse, unmatched
and empty streams, stable indices and duplicate-topic admission. This is an
additive feature, so no failing legacy ledger API is invented.

Thirty-three ledger cases cover independent bindings, round trips/forgery,
missing/double calls, arithmetic/capacity limits, no-clobber bounded IO, actual
synthetic SQLite publication and blocked admission. Together with the 54
existing controls, the full-bag example passes **100 tests**. Initial compile
and synthetic fixture-preflight failures are retained, along with corrections;
these are development failures, not original behavioral regressions.

Related E2E, ROS2, records, sync, mapping, viewer feature variants, calibration,
all-target Clippy and formatting use the existing shared Cargo cache. Existing
Linux/Windows/macOS mapping jobs execute the expanded example suite; the full
131-check PR merge gate is verified separately from local checks.

The fixed resource policy uses `E:/SpatialRustBuild/fusion-coverage-v1`, at most
128MiB owned logs/fixtures, 2GiB positive cache growth, 2.125GiB combined growth,
four build/Rayon/test workers, a 1,800-second command deadline and at least
20GiB free. All 493 prior scientific JSON, 42 frozen native artifacts, pinned
sources and completed room outputs are hash-protected. No real sensor payload
is read, downloaded or copied, no real-motion fit runs, and no frozen binary is
rebuilt. Actual sizes and every terminal command are recorded in the receipt.

Native complete-room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, matching canonical measured clock uncertainty and
physical frame accuracy, and complete mapping/semantic/Viewer/interchange/release
acceptance remain open. Submission traceability does not establish operational
80% or physical voxel/scene coverage.
