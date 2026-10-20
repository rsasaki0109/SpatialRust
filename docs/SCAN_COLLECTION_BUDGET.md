# Incremental scan collection admission

Slice 146Z corrects the full-bag mapping example's late episode admission.
Previously, both complete topic vectors were retained before the bounded
episode builder checked records, points and allocated column capacity. A long
same-header scan group also grew before any episode limit was checked.

The [prospective contract](../benchmarks/scan-collection-budget-plan.json) at
`1bb898601901bb23c625004481c0e9b88304fd4d` precedes regression execution and implementation.
The [receipt](receipts/2026-10-21_scan_collection_budget.json) binds tested source
`bf91ea884b07b1e717b6cbb894cb68baf39a8350`, all commands and retained failures.

## Ownership and limits

One shared `MemoryEpisodeBuilder` now receives completed scans while front and
rear topics are read. Finished scans plus pending owned chunks must fit all
three episode limits. Pending counts as one scan, not one record per chunk.
Checked arithmetic rejects count/point/capacity overflow. Input column capacity
is checked while the source chunk remains borrowed, before its explicit CPU
clone; the clone's actual capacity is checked again before retention. No record
is detached from its source reservation. The next chunk is never requested
after a collection, source, cancellation or clock error.

At a header/frame boundary or EOF, a single pending chunk moves into the scan;
multiple chunks request exact-size scalar arrays and concatenate fields in
source order. The assembly request is at most the pending logical payload,
separate from retained episode capacity. The existing builder rechecks final
actual allocated capacities before admission. Allocation over-reservation can
therefore reject an output; this is not a guarantee about allocator internals,
container/metadata overhead or process RSS. Source decoding leases retain their
own independent budget. No process-memory or throughput benchmark is claimed.

The CLI reports peak admitted finished-plus-pending scalar capacity and maximum
assembly request separately. A passing synthetic 12-point XYZ case asserts
144 bytes retained/pending, a 96-byte assembly request and a 48-byte source-lease
peak. These are fixture values, not a canonical bag measurement.

Scalar values, all six buffer dtypes, schema, raw cloud metadata and first
provenance with sequence removed retain their contracts. Only `StampedTime` is
clock-corrected. Episode ordering stays deterministic across topics. Source
identity, registered calibration/clock evidence, actual frame checks, sensor
transport, odometry/TSDF/interchange and disk gates are unchanged. An overall
collection error does not admit a partial map.

Consecutive identical header timestamps and frames still form one scan group.
Distinct physical messages with identical headers can therefore merge; message
identity is a separate unresolved contract. Complete collection is not proof
of physical calibration, synchronized motion or canonical acceptance.

## Reproduction and verification

The baseline extracts the original per-topic callback loop and pending assembly
without incremental admission, then applies the existing builder after that
topic. This changes whole-program scheduling from the original two-vector
collection; it is not described as an unchanged whole-program baseline. Source
comparison confirms the original drain loop and pending assembly are preserved.
Of 32 baseline cases, **23 passed and 9 failed**. The initial test compile used
a private runtime module; the test-only public re-export fix and log remain.

Twenty-two new IO/capacity cases cover source-spy early stops for within-topic
and shared-topic record/point/capacity limits, oversized input capacity versus
payload, pending growth, empty chunks, exact limits, ordering/metadata/lineage,
all scalar variants, checked overflow, source/schema/cancellation/clock errors,
prior committed records and existing duplicate-header grouping. A tiny synthetic
SQLite bag has a malformed tail: an earlier point limit now returns its typed
episode error before the tail can be decoded, instead of returning a CDR error.

| Local check | Result |
| --- | --- |
| Full-bag example | 37 passed |
| Bounded E2E example | 17 passed |
| ROS2 default / all features | Empty default harness builds / 4 passed |
| Records / sync all features | 29 / 8 passed |
| Mapping all features / calibration cases | 24 / 6 passed |
| ROS2 all-feature/all-target Clippy | Pass, warnings denied |
| Workspace formatting | Pass |

The existing three OS example CI jobs also execute records/sync all-feature
tests. Local counts are separate from the full 131-check PR CI gate.

## Storage and operational qualification

All 493 prior scientific JSON, 42 frozen native artifacts, pinned reference
sources and completed room outputs remain hash-identical. All production library
trees, core/math/GPU, Python, dependencies and existing real sensor payloads
remain unchanged. No sensor download/copy, real-motion fit or frozen build occurs;
the SQLite fixtures are synthetic and use the owned external temp directory.

Logs/metadata occupy **248,720 bytes** at
`E:/SpatialRustBuild/scan-collection-budget-v1`. Reused cache growth is
**218,414,385 bytes**, below the prospective 2GiB cap.
Four build/Rayon/test workers, a 128MiB owned-root cap, 2.125GiB combined increase
and at least 20GiB free are monitored. Failed logs and exact source commits are
preserved; old evidence and the shared cache are not deleted.

Native complete room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, measured canonical clock uncertainty/physical frame
accuracy and complete mapping/semantic/Viewer/interchange/release acceptance
remain open. These synthetic IO checks do not establish operational 80%.
