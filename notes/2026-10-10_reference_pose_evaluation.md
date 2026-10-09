# Separate pose accuracy evaluation from fitting

Distance-gated support and low residuals cannot establish a physically correct
pose. Before adding reference-pose datasets, this slice makes the evaluation
contract explicit and binds saved results to the exact input pair.

Ordinary, multiscale, global and caller-candidate file workflows now stream
SHA-256 fingerprints before and after point-cloud reading and reject changed
inputs. Reports gain `input_file_sha256.source` and `.target`. Algorithm updates,
candidate selection, retained fields and convergence behavior are unchanged.
In-memory workflows do not invent file fingerprints. Hashing adds two streaming
file passes to file loading; it is not part of the native registration kernel.

`/workspace/SpatialRust-python-delivery/scripts/evaluate_pose_reference.py`
accepts already saved reports and a separate `spatialrust.pose-reference.v1`
manifest. Both reference and estimate must map source to target, be finite and
proper rigid matrices, and have identical ordered input byte hashes. References
declare units and synthetic or dataset provenance; dataset references require
an HTTPS source URL. Missing hashes in old reports are rejected rather than
retrospectively assigning files to a past run.

Rotation error is the angle of the relative rotation, computed with atan2 to
handle small angles and 180 degrees. Translation error is Euclidean distance
between translations in the common target frame. Streaming JSON/HTML evaluation
retains reference/report hashes, input hashes, declared provenance and per-report
errors, with fixed-scale angle bars. The reference is never passed to a solver
or used to select candidate poses. The evaluator cannot certify the authenticity
of a caller's reference, its units, or how earlier initialization was obtained.

19 new tests cover known analytic angles, translation distances, target-frame
invariance, invalid references/rotations/units/direction, swapped or missing
hashes, changing files and all four real-binding file workflows through the
evaluation CLI. All 440 ONNX-enabled Python tests pass. Existing candidate versus
independent-run comparisons still match; loaded-versus-file tests distinguish
file-only provenance while retaining equality of every algorithmic field.
The separate default-wheel environment passes 437 tests with exactly 3 expected
ONNX skips; installed native/type bytes and ICP/support/PCD checks still pass.

Local artifact:
`/workspace/SpatialRust-python-delivery/target/pose-reference-validation/evaluation/report.html`.
A 600-point known-transform fixture uses a 60-degree rotation and translation
(3,-2,.5), runs the no-caller-pose global workflow, and constructs the reference
only after candidate selection. Its selected pose has rotation error
1.3060e-6 degrees and translation error 1.5136e-7 m. This is synthetic validation,
not a public sensor accuracy result.

The provisional maturity assessment remains 70%. The next evidence gap is
independently verified public reference transforms and multiple real pairs,
including dataset direction, units, pair identities and provenance checks.
