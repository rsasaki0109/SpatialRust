# Redwood/3DMatch reference import and saved-log evaluation

The official [3DMatch benchmark page](https://3dmatch.cs.princeton.edu/#geometric-registration-benchmark)
links fragment archives on **3dvision.princeton.edu**, a separate download host.
Its fragment archives are tens of megabytes, distinct from the multi-gigabyte
RGB-D datasets. Obtain fragment and evaluation archives for the same scene.

## Bind a selected pair to original files

The publisher's
[`getGtInfoLog.m`](https://github.com/andyzeng/3dmatch-toolbox/blob/4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36/evaluation/geometric-registration/getGtInfoLog.m)
computes `inverse(fragment1_cam_to_world) * fragment2_cam_to_world`.
Thus header `i j N` stores a matrix mapping **fragment j to fragment i**.
IDs are zero-based. Do not interpret that matrix as i-to-j.

```bash
python scripts/prepare_3dmatch_reference.py \
  --fragments-dir /path/to/7-scenes-redkitchen \
  --gt-log /path/to/7-scenes-redkitchen-evaluation/gt.log \
  --source-id 2 --target-id 0 --project-reference-rotation \
  --provenance-url https://3dmatch.cs.princeton.edu/ \
  --description 'Official redkitchen fragment/evaluation archives' \
  --output-dir target/redkitchen-pair-2-to-0
```

The importer binds hashes of original `cloud_bin_2.ply` and `cloud_bin_0.ply`,
without rewriting XYZ, colors or fields. It verifies finite points and retains
point counts, absolute paths, selected IDs, fragment count, GT hash and correction
details. Reversing source/target explicitly inverts the matrix and records that
fact. Duplicate/missing pairs, malformed/truncated rows, changing fragment counts,
reflections and invalid rotations are rejected. Near-rotation projection is
opt-in with the existing `1e-3` singular-value limit. Units are metres.

Run the file alignment workflow on original PLY files, then use
`scripts/evaluate_pose_reference.py` for rotation/translation diagnostics.
Those metrics are not the publisher's benchmark recall criterion. Adjacent pairs
can be imported for diagnostics but are excluded from saved-log benchmark recall.

## Evaluate completed prediction logs

```bash
python scripts/evaluate_redwood_logs.py \
  --gt-log /path/to/scene-evaluation/gt.log \
  --gt-info /path/to/scene-evaluation/gt.info \
  --predictions /path/to/first.log /path/to/second.log \
  --provenance-url https://github.com/andyzeng/3dmatch-toolbox/tree/4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36/data/fragments \
  --allow-near-rigid-records --output-dir target/saved-log-evaluation
```

The metric follows the publisher's
[`mrEvaluateRegistration.m`](https://github.com/andyzeng/3dmatch-toolbox/blob/4c6b2f613adb8bdcc9a62cb04134b7e1379b1a36/evaluation/geometric-registration/external/ElasticReconstruction/mrEvaluateRegistration.m):
relative transform `GT^-1 * prediction`, error vector containing translation and
the quaternion vector part, normalized information form `e^T I e / I[0,0]`,
correct when squared score is at most **0.04**. This is not a universal 0.2 m
translation gate. Nonconsecutive pairs (`j-i > 1`) only enter recall/precision.
Information must match GT pairs and be finite, symmetric and positive semidefinite
with positive normalization.

The near-rigid flag validates small deviations but **scores original logged
matrices**, preserving the publisher formula rather than silently scoring
projected predictions. Normalized matrices are retained for reference import.
The publisher quaternion formula is singular near a half-turn; those scores
remain explicitly undefined and count as incorrect, rather than producing NaN
JSON or disappearing from denominators. Empty prediction logs give zero recall
and undefined precision. Duplicate predictions are rejected.

JSON/HTML retain scored pairs, false positives, undefined scores, denominators,
input/code hashes, versions and pairwise overlap of correct results. GT and
information records are joined by pair IDs rather than assuming row order.
An oracle union uses reference labels: it is a complementarity upper bound,
**not a deployable selector**. No registration algorithm runs in this evaluator.

## Verified metadata and a complementarity example

The pinned official toolbox contains GT/information for eight real and four
synthetic scenes. All **2,563** paired records parse, match keys/fragment counts,
and satisfy matrix checks with explicit near-rigid validation. This does not
execute registration on the underlying clouds.

On supplied `iclnuim-livingroom2` logs (153 eligible GT pairs):

| Supplied log label | Correct pairs | Recall | Precision |
| --- | --- | --- | --- |
| fpfh.log | 59 | 0.385621 | 0.188498 |
| 3dmatch.log | 66 | 0.431373 | 0.197015 |
| pcl_modified.log | 76 | 0.496732 | 0.170404 |

FPFH and 3DMatch share 55 correct pairs, with 4 FPFH-only and 11 3DMatch-only.
Their oracle union is 70; neither correct-pair set contains the other. The
highest-recall log also has lower precision, illustrating an application tradeoff.
These are historical supplied log labels, not current SpatialRust, Open3D or PCL
measurements. No fallback or confidence-based selector is established here.
