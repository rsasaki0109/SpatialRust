# Causal previous-motion dense RGB-D validation

Operational slice 146H. The preceding frozen dense study accepts fr1/xyz at
798/798 poses, ATE 0.090314m and 1s translation RPE 0.025987m. Its rpy run
retains 40/722 poses: the next returned hypothesis has 95,855/228,114 projective
correspondences (0.4202065633849742), below the unchanged 0.5 final gate. The
previous goal turn made concrete progress through merged PR #295 and its
independently verified full-scene result; the operational 80% goal remains open.

## Data model and execution contract

Use the existing explicit native source/target grayscale and metric-depth API
and frozen final-overlap numerical extension. A fresh per-run Python adapter
retains one proper 4x4 source-to-target relative pose and a successful-pair
counter. The first pair uses identity. Each later pair uses the exact preceding
returned increment, without timestamp rescaling or a new motion-fit parameter.
This is a previous-step warm start, not a measured constant-velocity model.

The native call receives an owned seed copy. Advance retained state only after
the native policy, proper pose and original final-projective support checks
pass. The existing generator admits each returned pose through unchanged
full-cloud support, world composition and typed IO checks. Any failure stops
the chain permanently, so a rejected full-cloud/IO candidate never seeds a
later call. An adapter error leaves its previous state unchanged. Separate
instances share no state; no image or unbounded pose history is retained.

Record the actual initial source-to-target pose and the preceding successful
pair index on every returned row. Independent NumPy auditing verifies identity
on pair one and exact equality to the prior recorded relative pose thereafter,
alongside all existing final projective statistics and world-chain checks.
Tests must cover direction, ownership, separate instances, rejection without
state advancement, unchanged numeric controls and no call after tracking loss.

No Rust/core/GPU API, shader, solver objective, iteration schedule or numerical
binary changes. Existing identity-initialized adapters and their plans remain
available. This adds no SciPy/OpenCV/Open3D production dependency or device copy.

## Fixed full-scene experiment

Commit both `tum-fr1-{xyz,rpy}-causal-dense-plan.json` plans before any native
call with the new initializer. Reuse the parent registered PNG/index files
under `E:/SpatialRustData/maturity-80/` using checked hard links in fresh prepared
directories. Preserve all original archives, failed/successful results, wheels
and native/GPU executables. No dataset download or additional image payload copy.

Run all 798 xyz and 722 rpy rows with the same camera/depth units, grayscale,
20/10/5 iterations, 0.03m projective depth gate, 0.968 depth weight, minimum 30
correspondences and 0.5 final fraction. Keep original full-cloud 0.5 support at
0.1m, projection/typed IO and timestamp checks, the sole identity world anchor
and loss without resets. No retries, alternative seed selection, future-frame
lookahead, reference initialization or post-score criterion changes.

Freeze both complete estimate timelines before either new reference score.
Require 100% pose generation, >=95% reference coverage, first-pose-aligned ATE
RMSE <=0.10m and one-second translation RPE RMSE <=0.05m. Retain all original
reference gaps. Independently audit projective counts exactly, fractions to
1e-12, photometric/depth RMSE to 1e-10, seed/world matrices to 1e-12/1e-9 and
SciPy scores to 1e-8m/1e-6degree. A surviving prefix cannot establish full
motion accuracy. Both scenes are already observed development data.

Use the existing immutable E-drive package environment and numerical binary,
with code/plan hashes and fresh output directories. Confine data, caches and
temporary outputs to E; keep a 20GiB free floor, 1GiB study-output cap, four
native workers and one hour per bounded command. Only owned passing pytest
scratch may be cleaned after XML, absolute-path and no-reparse verification.
Publish actual outcomes and source/native/storage receipts. Canonical measured
clock/physical calibration, hardware portability and full release/mapping
acceptance remain separate open requirements for the original 80% objective.
