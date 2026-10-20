# Full-bag scan identity admission

Slice 146AA prevents ambiguous grouping and time-to-pose association from
producing a successful full-bag map. The collector groups consecutive chunks
by raw header time and frame; separate physical messages can share those
headers. The consumer also keyed front poses only by corrected nanoseconds,
silently overwriting duplicates, and used a trajectory zip that could truncate.

The [prospective contract](../benchmarks/scan-identity-admission-plan.json) at
`7ee12ebf4e471706a8a258ffda7c48eff8834251` predates native execution.
The [receipt](receipts/2026-10-21_scan_identity_admission.json) binds tested
implementation `33032bcb2a4d9a89386e466b46d064dfb04afa83` and all retained logs.

## Admission order

After successfully draining each selected SQLite topic, require collected scan
groups to equal its physical message count. A single message split into many
chunks still produces one scan. Identical-header messages that merge, and empty
messages skipped by the source, now reject completeness. This check happens
after collection; existing record/point/capacity admission still stops decoding
on the first earlier error. A failed run may retain private builder state but
cannot return a successful partial map.

Before odometry, select each topic from the sorted episode and require strictly
increasing corrected nanoseconds within that topic. Nonconsecutive repeated
headers and clock-correction rounding collisions reject. Raw metadata stays
intact; the runner neither deduplicates scans nor invents replacement times.
Equal times across front and rear topics remain valid.

Before allocating or integrating TSDF, require exactly one trajectory sample
per front scan. The complete `StampedTime` must match, including clock identity,
domain, offset, uncertainty and estimated status. Mixed topics/frames and
duplicate lookup keys reject. Valid distinct poses retain their original
association, and both sensor transport formulas remain unchanged. The guarded
front integration zip now has equal lengths and matching stamps by construction.

These are necessary admission conditions. They do not add physical message IDs
to the public source/chunk APIs or prove physical synchronization. Header/frame
chunk assembly, nearest-neighbor tie/reuse policy and source clock correction
remain unchanged. Deterministic matching is separate from calibrated timing
accuracy. Public physical-message identity remains an open contract.

The subsequent [146AB portable receipt checks](MAPPING_RECEIPT_CONSISTENCY.md)
also bind source and stage summaries when saved state is explicitly validated.
They preserve schema 1 and distinguish integration visits from unique coverage;
the source-bound 146AA results below retain their original scope.

## Reproduction and verification

The baseline extracts the existing topic filter and timestamp-to-pose zip into
private fallible helpers without adding rejection behavior. Source comparison
confirms the parent collection function and pose zip are preserved. It reports
**41 passes and 13 failures across 54 cases**. The initial test-only quaternion
type ambiguity and its compiler log are also retained; that attempt is excluded
from behavioral reproduction.

Seventeen new synthetic cases cover SQLite single-message chunking, distinct
multi-chunk messages, identical-header physical messages, skipped empties,
nonconsecutive repeated times, corrected-time collisions, valid cross-topic
ties, short/long trajectories, duplicate pose keys, timestamp/clock/domain/all
quality fields, mixed topics/frames and exact matched-front pose selection.
The existing collection-capacity, source-error and coordinate cases still pass.

| Local check | Result |
| --- | --- |
| Full-bag example | 54 passed |
| Bounded E2E example | 17 passed |
| ROS2 default / all features | Empty default harness builds / 4 passed |
| Records / sync all features | 29 / 8 passed |
| Mapping all features / calibration cases | 24 / 6 passed |
| ROS2 all-feature/all-target Clippy | Pass, warnings denied |
| Workspace formatting | Pass |

The existing Linux/Windows/macOS example jobs execute the new cases. Local
results are separate from the complete 131-check PR merge gate.

## Storage and remaining qualification

All 493 prior scientific JSON, 42 frozen native artifacts, pinned sources and
completed room outputs remain hash-identical. Library/core/math/GPU/Python,
dependencies, incremental collection and sensor transport trees are unchanged.
There is no real sensor read/download/copy, new real-motion fit or frozen build.
Tiny SQLite fixtures use the owned external temporary directory.

Logs/metadata occupy **267,918 bytes** at
`E:/SpatialRustBuild/scan-identity-admission-v1`. Shared cache growth is
**15,069,060 bytes**, below the prospectively fixed 2GiB cap. Four build/Rayon/test
workers, a 128MiB owned-root cap, 2.125GiB combined increase, 1,800-second command
deadline and at least 20GiB free are monitored. Failed attempts are preserved;
no old evidence or shared cache is deleted. These are storage bounds, not RSS
or throughput measurements.

Native complete room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, measured canonical clock uncertainty/physical frame
accuracy and complete mapping/semantic/Viewer/interchange/release acceptance
remain open. Synthetic admission correctness does not establish operational 80%.
