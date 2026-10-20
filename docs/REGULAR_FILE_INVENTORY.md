# Complete regular-file capacity inventory

Operational slice 146R fixes an observed capacity-accounting error before more
real-data experiments. The unpublished first 146Q snapshot used recursive glob
iteration under restricted permissions and omitted a protected failed-test
folder. Its corrected complete snapshot and both metadata versions remain.

Commit this contract before implementation or real inventory probes. Use a
small standard-library-only script helper returning immutable counts only after
a complete traversal. Existing absolute directory roots and their ancestors
must be regular directories. Reject symlinks/reparse points and unsupported
nodes; propagate permission, vanished-entry and stat/enumeration failures.
Filename filtering never bypasses a directory. Logical regular-file bytes are
charged once per pathname, including each hard-link pathname. This is a complete
observation, not an atomic snapshot or physical allocated-block measurement.

Both fixed-source acquisition helpers must use the complete archive inventory
before any network request, preserving every old prefix and all original source,
resource and native/accuracy controls. Add synthetic IO tests for complete counts,
filtered denied subtrees and both callers, and exercise the real restricted
failed-test directory without changing ACLs. A separate privileged read-only
inventory must reproduce the fixed prior input/study/build totals in
`benchmarks/regular-file-inventory-plan.json` and an independent strict walker.
No estimator run, new payload, native rebuild or wheel installation is needed.

Use only `E:/SpatialRustBuild/fail-closed-inventory-v1/` and
`E:/SpatialRustData/maturity-80/fail-closed-inventory-v1/` for new scratch/results.
Keep >=20GiB free, build/test scratch <=1GiB, results <=16MiB and combined new
owned bytes <=2GiB, with four CPU workers and <=one hour per command. Preserve
all old scientific JSON, archives, native artifacts and failed scratch. Reclaim
only exact completed passing pytest basetemps after XML, absolute ancestry and
full no-reparse checks; retain every log and failure. Deliver one focused PR
with actual IO, hash-preservation, resource and CI evidence.

This storage correction does not satisfy motion, canonical clock/physical frame,
Linux hardware-class, full mapping/semantic or release acceptance. Overall 80%
remains unproven. Numerical precision/pose-update investigation follows this
capacity prerequisite, using existing observed development inputs.

## Verified outcomes

The fixed protocol is commit `9d3cc6f`; implementation and actual observations
use `681d6a7`. The restricted real process reproduced the original failure:
recursive glob accepted only 110,769 bytes for the old build tree, omitting
81,317,869 bytes. The new inventory raises PermissionError without returning
a valid total. No ACL was changed and no protected file was deleted.

With complete directory access, the new inventory and independent strict
`os.walk(onerror=raise)` agree on every count and all three fixed prior totals:

| Prior root | Logical bytes | Regular files | Directories |
| --- | ---: | ---: | ---: |
| prior-input | 3,061,264,520 | 5016 | 9 |
| prior-study | 481,564,539 | 122 | 11 |
| prior-build | 81,428,638 | 120 | 71 |

The combined old trees remain 3,624,257,697 bytes. Filtering archives still
inspects all nine input directories and 5,016 regular files; only two archive
paths contribute the original 1,478,001,585 bytes. Hard links are charged per
pathname. All 388 prior scientific JSON files and 36 frozen native artifacts
retain their hashes; previous failed scratch remains in the complete count.

Nineteen new IO cases cover nested/filter/hard-link counts, missing/non-directory
roots, denied/vanished/stat failures, links/reparse/unsupported entries, root
ancestry and both acquisition callers. Both callers refuse before a network
request and preserve an existing range prefix. All 77 focused cases pass;
full Python has 1,248 passes and three existing optional-dependency skips.
Python 3.8 syntax passes. No Rust/bridge/stub, native binary, source selection,
model/score control, device transfer or production dependency changed.

Reclaim only the two XML-verified passing pytest basetemps: 641,589,145 logical
bytes. Retain their logs/XML and every old failure. New retained metadata is
102,276 build bytes plus 4,004 result bytes, within the fixed 2GiB combined
ceiling; E free is 672,495,636,480 bytes and the 20GiB floor holds throughout.
No sensor payload was downloaded/copied, no native package was built/installed
and no new trajectory or score was generated. Motion/calibration/hardware and
full operational 80% acceptance remain open.

Source, command, restricted/full proof, counts, preservation and resource hashes:
[public receipt](receipts/2026-10-20_regular_file_inventory.json).
