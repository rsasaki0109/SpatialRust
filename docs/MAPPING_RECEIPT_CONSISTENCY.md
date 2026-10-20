# Portable mapping receipt consistency

Slice 146AB rejects contradictory full-bag mapping receipts. Previously, each
stage could validate locally while its scan count, topic, source frame, root
or corrected clock referred to different input. Completion flags alone could
then admit that combined state.

The [prospective contract](../benchmarks/mapping-receipt-consistency-plan.json)
at `21273842edf02935dbf1ccab8ad6166a362588f4` predates native execution.
The [receipt](receipts/2026-10-21_mapping_receipt_consistency.json) binds tested
source `f2004f298d84a8d4d9ba65c0bab8a9aa418700d4` and retained command logs.

## Source and stage bindings

A completed source requires message/retained-scan counts to agree for each
selected topic, at least one point per retained scan in the aggregate, and both
corrected timestamp bounds. Existing checked totals, ordered bounds, positive
storage and truncation checks remain. These are necessary consistency checks;
aggregate counts do not prove the identity or content of every physical scan.

Every present odometry receipt must name the selected front topic and cannot
contain more scans than retained front records. A complete front trajectory
must cover exactly the front records and front bag messages. A consistent
partial prefix remains a valid blocked diagnostic. A complete front trajectory
also remains valid when rear ingest alone is incomplete; it does not make the
overall mapping admissible.

When calibration is present, odometry's source frame must match its registered
front role, the trajectory root must match the calibration root, and the
corrected clock ID must match the registered target domain. The descriptive
`clock_domain` label remains free text. TSDF's frame must match the calibration
root and any present odometry root, including in blocked diagnostics.

All checks borrow existing state without mutation or data transfers. Existing
registration, primitive graph/volume checks, derived readiness flags and blocker
rules remain. Missing calibration still produces a valid blocked state when
the available stages are internally consistent.

## Integration visits and compatibility

`integrated_record_count` and `integrated_point_count` count visits. A nearest
rear record may be reused; another rear record may remain unmatched. Equality
with unique retained source totals would reject legitimate current-policy
receipts. Passing synthetic examples explicitly retain both cases:

| Front/rear retained scans | Unique source scans/points | Integration visits/points |
| --- | --- | --- |
| 2 / 2, one rear match | 4 / 12 | 3 / 9 |
| 3 / 1, rear reused three times | 4 / 12 | 6 / 18 |

These demonstrate receipt semantics, not physical accuracy or unique fusion
coverage. No new matching policy, physical message ID, count-equality rule or
producer-specific visit bound is introduced. Unique fusion coverage remains
unproven by the current fields.

Schema version **1**, serialized fields, public method signatures and
dependencies are unchanged. Contradictory receipts previously accepted by
`validate()` now reject. Valid completed and consistent partial states retain
their representation. Serde decoding does not run validation automatically:
call `validate()` before admitting decoded state downstream. Tests cover a
healthy round trip, forged decoded counts/clocks and unknown-field rejection.

## Reproduction and verification

The original completed synthetic fixture is extracted as a shared test helper
without changing its values. All production validators before the test module
compare equal to the parent for behavioral reproduction. The mapping subset
reports **14 passes and 15 failures across 29 cases**, with 52 unrelated cases
filtered out. That failure log is retained.

Twenty-six new cases cover source cardinality/points/bounds, complete/partial
front counts, topic/frame/root/clock bindings, volume roots, decoded forgery,
constructor rejection, checked overflow, partial diagnostics, descriptive clock
labels and visits above/below unique source totals. They provide schema and
validation evidence; they are not real mapping or calibration experiments.

| Local check | Result |
| --- | --- |
| Viewer default / serde / all features | 68 / 73 / 81 passed |
| Full-bag / bounded E2E examples | 54 / 17 passed |
| ROS2 default / all features | Empty default harness builds / 4 passed |
| Records / sync / mapping all features | 29 / 8 / 24 passed |
| Calibration name filter | 8 passed: 6 old controls plus 2 overlapping new cases |
| Viewer and ROS2 all-feature/all-target Clippy | Pass, warnings denied |
| Workspace formatting | Pass |

Existing Linux/Windows/macOS visual jobs now test viewer default and serde in
addition to all features. These local results are separate from the complete
131-check PR merge gate.

## Storage and remaining qualification

All 493 prior scientific JSON, 42 frozen native artifacts, pinned sources and
completed room outputs remain hash-identical. All production trees outside the
viewer validator remain unchanged, including source/calibration registration,
collection, matching, transport, mapping computation, core/GPU/Python and
dependencies. Other viewer production files remain unchanged as well.

Logs/metadata occupy **339,629 bytes** at
`E:/SpatialRustBuild/mapping-receipt-consistency-v1`. Shared cache growth is
**932,814,073 bytes**, below the prospective 2GiB cap. Four build/Rayon/test
workers, a 128MiB owned-root cap, 2.125GiB combined increase, 1,800-second command
deadline and at least 20GiB free are monitored. Failed logs and earlier evidence
are preserved. No real sensor read/download/copy, real-motion fit or frozen
build occurs. Storage bounds are separate from process RSS or throughput.

Native complete room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, measured canonical clock uncertainty/physical frame
accuracy and complete mapping/semantic/Viewer/interchange/release acceptance
remain open. Portable receipt consistency does not establish operational 80%.
