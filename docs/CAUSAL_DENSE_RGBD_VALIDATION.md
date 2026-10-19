# Causal previous-motion dense RGB-D validation

Operational slice 146H. The preceding frozen dense study accepts fr1/xyz at
798/798 poses, ATE 0.090314m and 1s translation RPE 0.025987m. Its rpy run
retains 40/722 poses: the next returned hypothesis has 95,855/228,114 projective
correspondences (0.4202065633849742), below the unchanged 0.5 final gate. The
parent study is preserved in [the dense protocol](DENSE_RGBD_ODOMETRY_VALIDATION.md).
The operational 80% goal remains open.

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

## Frozen outcomes

Implementation commit `19bfd0e8a61b1a0a2ef6dfeb8512ec65939bd9b6` precedes both
full runs. Both estimates were frozen at `2026-10-19T15:47:16.169384+00:00`
before either new reference score. The numerical extension remains SHA-256
`f2112078a97484461fd3f2e653204a8394839784f833f7060abaf85c94a70d88`,
from the original native source at `1e56c9352f6b91aaccec97e946b10b5feec651c0`.

| Fixed method / scene | Generated / planned | Reference evaluated | ATE RMSE, m | 1s translation RPE RMSE, m | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| Parent identity / xyz | 798/798 | 795 | 0.090314 | 0.025987 | Pass development scene |
| Parent identity / rpy | 40/722 | 40 | Prefix only | Prefix only | Fail: incomplete |
| Causal increment / xyz | 798/798 | 795 | 0.089993 | 0.026591 | Pass development scene |
| Causal increment / rpy | 722/722 | 716 | 0.114793 | 0.048145 | Fail: ATE >0.10m |

The new rpy run solves the observed tracking loss and passes its one-second
translation RPE gate; it does not pass the original absolute-trajectory limit.
No threshold, initializer policy, frame subset or score alignment is revised
after these results. Retain all 1,520 planned rows, three xyz/six rpy reference
gaps and 762/768 xyz plus 680/692 rpy one-second pairs. Both development scenes
remain previously observed data; this is not held-out generalization evidence.

Independent NumPy audits pass all 797 xyz and 721 rpy adjacent pairs, including
every seed's direction/preceding-pair identity, complete source counts, final
photometric/depth statistics and causal world composition. Both independent
SciPy score audits pass their original tolerances. No reference pose enters
generation. Seed arrays, returned inverse transforms and report lists do not
alias retained state, and a native failure does not advance it.

Local validation passes 963 Python tests with three existing optional ONNX
skips. Twenty-one new cases cover causal state, owned outputs, exact plan and
audit failures, first-pair identity and no later call after full-cloud rejection.
All new Python parses with the Python 3.8 grammar. Vision, Python native source,
core and GPU trees match the parent; no native rebuild or new wheel is required.

The [compact receipt](receipts/2026-10-20_causal_dense_rgbd.json) binds source,
plans, archives, manifests, numerical binary, estimates, audits, tests and storage.
Complete original logs and execution/freeze records remain under
`E:/SpatialRustData/maturity-80/causal-dense-study-v1/`. Per-scene prepared,
run, evaluation and audit directories use the `causal-dense-v1` suffix under
`E:/SpatialRustData/maturity-80/tum-fr1-{xyz,rpy}/`.

The study preserves 166 prior JSON files and five wheels, plus all frozen
native/GPU executables. It shares 1,598 xyz and 1,447 rpy PNG/index files with
zero additional image payload copies or downloads. Results retain 312,905,171
logical bytes, below the fixed 1GiB cap. Removing only two validated passing
pytest scratch directories reclaims 356,900,639 logical bytes; XML and logs
remain. The retained float64 seed payload is 128 bytes; native 640x480 working
payload and Python input-copy receipts retain 12,913,440/3,072,000 bytes with
at most four workers. These counts exclude temporary Python/allocator/thread
overhead and do not measure peak process or GPU memory.

The hardware-scope audit also corrects an overly broad remaining-work label
in the parent dense report: Windows already passes both IntegratedGpu and
DiscreteGpu classes on the fixed public operation cases. The remaining hardware
gap is Linux GPU operation conformance; its accepted Cpu cases are software
execution. The [original multi-adapter receipt](receipts/2026-10-19_public_multi_adapter_rgbd.json)
and all underlying outcomes are unchanged. Canonical measured clock/physical
calibration, full mapping/semantic/release acceptance and rpy ATE remain open;
this slice does not prove the operational 80% objective.
