# Bounded canonical fusion readback

Slice 146AD checks the persisted coverage artifact against the independently
completed in-memory ledger before the full-bag example writes a parent success
state. Previously the file was checksummed without this producer readback gate.
The [prospective contract](../benchmarks/fusion-readback-plan.json) and roadmap
reservation precede native execution. The [receipt](receipts/2026-10-21_fusion_readback.json)
binds source, logs, storage and preserved evidence.

## Expected bytes and trust

The expected value is the sealed `CompletedCoverage` already validated against
the original episode, verified source/calibration context and actual TSDF
submission counts. Verification never reads expected fields from the saved
file. It regenerates the same compact serde JSON and required final newline
into a comparator, while streaming the saved bytes from a regular file.

This deliberately checks the exact producer representation. Different values,
unknown fields, changed schema, missing bytes/newline, extra whitespace, trailing
data and equivalent pretty-printing all reject. It does not accept arbitrary
JSON formatting or introduce an untrusted JSON deserializer. There is no full
output string, array of decoded records or allocation sized by a saved label.

The gate returns the matched byte count, then uses the existing `FileReceipt`
checksum pass and requires its observed size to agree. The verified receipt is
linked with the existing `fusion-coverage` artifact role. An error propagates
before the parent state is written. Existing dataset-manifest hashing remains.
This operates on producer-owned files; comparison and checksum are separate
passes, not an authenticated or atomic transaction against concurrent writers.

## Bounded IO

The underlying reader is capped before buffering at the configured JSON limit
plus one byte. The final byte probe distinguishes exact EOF from appended data,
including a file that grows beyond its initial size. The JSON limit remains at
most 32MiB, including the final newline. Oversized/non-file paths reject early.

Read buffering, comparison scratch and expected-output buffering each use a
fixed 16KiB buffer. Their conservative 48KiB budget is accounted inside the
existing 8MiB ledger metadata limit, alongside actual record/association vector
capacities and validation scratch. Existing source/mapping/replay allocations,
checksum/manifest IO and process RSS remain separate. The 65,536-record limit,
1,024-byte labels, header capacity and byte-size guards are unchanged.

Short reads and `Interrupted` are handled normally. IO errors, truncation and
mismatch poison the comparator; a retry or buffered-writer drop cannot read
more source bytes. Verification does not replace, repair or delete the artifact
or an existing temporary file. Atomic no-clobber publication remains unchanged.

## Verification

The unchanged producer first passes all **100 original full-bag tests**. This
is an additive readback gate, so no failing legacy verifier is invented.
Twenty-eight new controls cover exact canonical receipt/checksum identity,
source/calibration/frame/clock/record/association/summary forgery, schema/unknown
fields, truncation/newline/trailing bytes/reformatting, missing/directory paths,
read-only failure behavior, short/interrupted/error reads, exact byte limits and
one-byte probes, overlong untrusted input, multi-buffer reports and poisoning.

All **128 full-bag tests** and all-target ROS2 Clippy pass. The existing synthetic
SQLite collection-to-ICP/TSDF publication and blocked input/calibration cases
exercise the new successful-run gate. Related E2E, ROS2 default/all, records,
sync, mapping, viewer-all, calibration controls and formatting use the existing
shared cache. Existing Linux/Windows/macOS mapping jobs run the expanded suite;
all 131 named PR checks are required separately from local results.

All core/math/records/sync/mapping/scene/GPU/vision/Python/viewer and ROS2 library
sources, public APIs and dependencies stay unchanged. Coverage/state schemas,
canonical publication, original matching/tie/reuse, source/calibration admission,
scan transport and odometry/TSDF math retain their contracts.

## Storage and remaining qualification

The fresh owned root is `E:/SpatialRustBuild/fusion-readback-v1`, with a 128MiB
log/fixture cap, 2GiB positive shared-cache growth, 2.125GiB combined growth,
four build/Rayon/test workers, a 1,800-second command deadline and at least
20GiB free. The receipt records actual sizes and every command's terminal state.
All 493 prior scientific JSON, 42 frozen native artifacts, pinned references and
completed room outputs remain hash-protected. No real sensor read/download/copy,
real-motion fit or frozen binary rebuild occurs.

Native complete-room accuracy and genuinely unobserved motion, both qualifying
raw Linux hardware classes, matching canonical measured clock uncertainty and
physical frame accuracy, and complete mapping/semantic/Viewer/interchange/release
acceptance remain open. Canonical readback does not qualify physical messages,
voxel contribution, sensor calibration or operational 80%.
