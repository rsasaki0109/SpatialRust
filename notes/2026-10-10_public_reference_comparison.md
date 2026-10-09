# Supplied public reference and partial-cloud refinement

See `/workspace/SpatialRust-python-delivery/docs/REFERENCE_COMPARISON.md` for
commands, pinned public repository, input/reference convention, controls and
limitations. No external repository code, pretrained model or pickle is executed;
only numeric arrays loaded with `allow_pickle=False` are used.

The supplied reference is not precisely rigid. Its near-rotation correction is
explicit and retains original SHA-256 fingerprints, matrix, singular values and
SVD correction magnitude. This preserves strict evaluator behavior and avoids
silently accepting scaled poses. It does not certify sensor lineage or ground truth.

Ten global registrations and twenty fixed-prior refinements run on the same
15,953/18,977-point demo pair. The untrimmed native median reference translation
error is 0.206075 m; 0.6 trim gives 0.050511 m. Lower correspondence residuals or
more proximity support do not necessarily imply the correct pose. At seed 8,
aggressive 0.4 trim worsens rotation error from 0.87° at 0.6 to 3.77°. No trim is
selected by reference, and defaults are unchanged. All five untrimmed replay
poses are byte-equivalent numeric arrays to their original comparison poses.

Local artifacts:

- Prepared hashes/PCD/reference: `/workspace/SpatialRust-python-delivery/target/public-geotransformer-prepared/`
- Final comparison: `/workspace/SpatialRust-python-delivery/target/public-geotransformer-comparison-final/`
- Final replay: `/workspace/SpatialRust-python-delivery/target/public-geotransformer-trim-final/`

450 ONNX-enabled Python tests and 20 optional comparison/report tests pass,
including safe array validation, explicit-only correction, reflection/large
distortion rejection, original matrix preservation, PCD roundtrip, both native
and Open3D recovery on an independent known-transform fixture and replay input
binding. Provisional maturity reaches 72%; multiple independently verified sensor
datasets, platform runtimes, MSRV and operational stability remain requirements
toward the 90% goal.
The isolated default wheel also passes 447 tests with exactly 3 expected ONNX
skips; installed native/type bytes and ICP/support/PCD checks match the archive.
