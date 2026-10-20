# Rooted pose-graph propagation

Slice 146X fixes `PoseGraph::localize_from_root`: the original implementation
ignored pure rotations and translations below 1e-4, updated disconnected
components, and could propagate a temporarily moved root. It also returned
success after bounded relaxation of ambiguous or non-finite constraints.

The [prospective contract](../benchmarks/pose-graph-rooted-localization-plan.json)
was committed at `353648589a4ef9a903eb8a53215cd709a03cf1ff` before regression execution or
implementation. The [complete evidence](receipts/2026-10-20_pose_graph_rooted_localization.json)
binds source `61308f759d427b3a380f2973b5a800616c2317d4`, command logs, original failures,
current tests and resource/preservation checks.

## Frame and state contract

The existing algebra is retained:

```text
to_T_anchor = to_T_from * from_T_anchor
anchor_T_node = inverse(node_T_anchor)
```

Localized graph isometries map the common anchor into a node frame. Scan
odometry retains `current_T_first_scan`; callers explicitly invert it when
transforming a current cloud back into the first frame. This change does not
insert a hidden inversion or alter edge direction.

Only directed non-loop edges reachable from the chosen root participate. The
reachable component must form a tree. Cycles, duplicate constraints and multiple
incoming paths now return `MappingError::Graph`; disconnected components and
stored loop-closure constraints stay untouched. A missing root still returns
`MappingError::Missing`. The original root, every target timestamp and all edge
records remain fixed. Complete rotations and translations propagate without a
translation-only convergence threshold.

Adjacency indexing and an iterative breadth-first queue replace repeated
relaxation. Each non-loop edge is indexed once and each reachable edge is
examined once. Hash-map operations give expected O(V+E) work and O(V+E) scratch,
bounded by the supplied graph; there is no recursion or new thread/dependency.
All reachable topology and finite transforms/composed results validate before
the first node write, so every returned error leaves the graph unchanged.

## Independent correctness evidence

Sixteen public-API tests ran against the unchanged production implementation
first: **4 passed, 12 failed**. Their exact source commit and failing log remain.
The first test-only compilation failed because a long-chain helper passed a
String where `PoseNodeId` was required; that log is retained and the typed helper
was fixed before original-behavior reproduction. Production code had not changed.

All 16 tests now pass. Hand-computed transformed basis/origin points check pure
rotation and `Ry(90) * Rx(90) * Rz(90)` with a nonidentity root, in both edge
orders. Other cases check tiny translation, unchanged-parent reachability,
disconnected components/incoming edges/cycles, root pinning, timestamps,
idempotence, loop exclusion, missing root, atomic cycle/duplicate/multi-parent
failure, non-finite root/edge, finite-input overflow and a reversed 4,096-edge
chain. One additional scan integration case checks the current-from-first
rotation direction and timestamps. These are analytical/synthetic tests, not
real-motion accuracy evidence or a throughput benchmark.

| Local check | Result |
| --- | --- |
| Mapping default features | 22 passed |
| Mapping all features, including scan ICP and vision bridge | 24 passed |
| Existing north-star cross-crate pipeline | 1 passed |
| All-feature/all-target Clippy | Pass, warnings denied |
| All-feature WASM check | Pass |
| Workspace formatting check | Pass |

Three dedicated CI jobs run default/all-feature tests and Clippy on Linux,
Windows and macOS; the existing Web/WASM job also checks all mapping features.
The CI matrix is separate from the local results above.

## Storage and operational limits

All 493 previous scientific JSON, 42 frozen native artifacts, pinned reference
sources and completed room-audit outputs remain hash-identical. Core, math,
GPU, vision, Python crate/bridge/stub, dependencies and existing sensor payloads
are unchanged. No sensor acquisition/copy, frozen-wheel rebuild or new
real-motion fitting occurs.

Execution metadata/logs occupy **105,134 bytes**
at `E:/SpatialRustBuild/pose-graph-rooted-v1`. Reused Cargo cache grows by
**913,241,902 bytes**, under the fixed 2GiB increase
limit; combined growth stays below 2.125GiB. Four build workers, a 128MiB owned
root limit and at least 20GiB free are monitored. Failed attempts are preserved;
no old evidence or shared cache is deleted.

This supplies a correct propagation substrate. Loop-closure optimization and
room drift correction remain unimplemented here. Native room accuracy and
genuinely unobserved motion, both qualifying raw Linux hardware classes,
matching canonical measured clock uncertainty/physical frame accuracy and
complete canonical mapping/semantic/Viewer/interchange/release acceptance
remain open. No operational 80% qualification follows from these tests.
