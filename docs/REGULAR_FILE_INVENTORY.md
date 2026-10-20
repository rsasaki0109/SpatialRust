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
