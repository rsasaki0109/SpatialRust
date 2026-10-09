# Reference direction and publisher-compatible saved-log evaluation

After environment publication, the 3DMatch project page is accessible. Its
official fragment/evaluation archives are hosted on a separate domain,
`3dvision.princeton.edu`, whose proxy connection still returns 403. A confirmed
configuration draft preserves `3dmatch.cs.princeton.edu` and adds only that
actual download domain. `requires_publish` is true; saving does not apply it.

Independent work completed before waiting for access:

- Read the official toolbox at pinned commit
  `4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36`. Its source establishes that GT
  headers i,j,N map fragment j to i, via inverse(cam1-to-world)*cam2-to-world.
- Strict pose/information parsing, original-matrix preservation, duplicate and
  count validation, PSD/symmetry checks and explicit near-rotation handling.
- Pair importer binds original PLY/PCD hashes, source/target IDs, direction,
  optional inversion, GT hash and correction details. No sensor data is rewritten.
- Saved prediction evaluator implements normalized information error squared
  threshold 0.04 with consecutive-pair exclusion and explicit false-positive/
  undefined counts. Original logged matrices are scored. Empty predictions have
  undefined precision. Pairwise complementarity includes an explicitly labeled
  non-deployable oracle upper bound, with no truth-based algorithm selection.

Local metadata receipt validates 2,563 GT/information pairs across 12 scenes
(eight real and four synthetic). Supplied livingroom2 logs show 55 correct pairs
shared by FPFH and 3DMatch, 4 FPFH-only and 11 3DMatch-only, giving an oracle union
of 70 versus individual 59/66. PCL-modified's supplied log has higher recall
but lower precision. These are historical publisher results, not current native
library measurements or a new live-cloud study.

20 new tests verify direction/inversion analytically, malformed records,
original-matrix retention, near-rigid opt-in, large finite information, negative
or asymmetric information rejection, analytic quaternion error with cross terms,
recall/precision denominators, false positives, half-turn singularity, original
PLY file binding and saved-log CLI complementarity. All 470 ONNX-enabled Python
tests pass; the new 20 also pass against the isolated default wheel. Native
library/packaging code is unchanged from the preceding verified release slice.

Artifacts:

- `/workspace/SpatialRust-python-delivery/target/3dmatch-reference-metadata-validation/receipt.json`
- `/workspace/SpatialRust-python-delivery/target/published-redwood-log-validation-final/evaluation.json`
- `/workspace/SpatialRust-python-delivery/target/published-redwood-log-validation-final/report.html`

Usage and scoring limits are documented in
`/workspace/SpatialRust-python-delivery/docs/3DMATCH_REFERENCE.md`.
Provisional maturity is 75%; reaching 90% still needs live multi-scene accuracy,
platform/GPU conformance and operational evidence. Network publication is the
concrete next prerequisite for the official fragment accuracy gate.
