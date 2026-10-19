# Fixed wide-capture RGB-D validation

Epic 146E tests the unresolved motion boundary after robust-PnP consensus
preservation. The previous fixed native/wheel revision completes both xyz
frontends but misses 0.10m ATE; rpy stops at row 2 under the unchanged inlier
support gate. All old estimates, failures, scores and installed extensions stay
byte-identical. See [the numerical fix and outcomes](PNP_CONSENSUS_VALIDATION.md).

## Decision before fitting

The preceding post-freeze prefix diagnostic has already observed more valid
tracks with larger LK windows/deeper pyramids. Radius 10 and four levels is one
explicit setting selected from that observed diagnostic. It is fixed for both
stateless/adaptive frontends and both xyz/rpy development sequences before any
new full trajectory. It is not a holdout, a default change, or a settings search
that reports only the best case. No candidate fitting/scoring is performed
before implementation, tests and these four plans are committed:

- `benchmarks/tum-fr1-xyz-rgbd-wide-capture-plan.json`
- `benchmarks/tum-fr1-rpy-rgbd-wide-capture-plan.json`
- `benchmarks/tum-fr1-xyz-adaptive-wide-capture-plan.json`
- `benchmarks/tum-fr1-rpy-adaptive-wide-capture-plan.json`

Only `window_radius` 5 -> 10 and `pyramid_levels` 3 -> 4 change. Each plan binds
its exact consensus-preserving parent; JSON type identity is strict. Calibration,
image margin 40, candidate/retained feature bounds, local-depth medians, causal
keyframe promotion, metric PnP settings, .75px forward/backward and 2px inlier
tests, 20 tracks/15 inliers/50% inlier fraction, geometric support and complete
motion accuracy thresholds retain their original values. No ground-truth
initializer, scale fit, selection of future frames or lost-chain recovery.

## Implementation and resource boundary

The new executable `scripts/run_tum_wide_rgbd.py` selects the explicit estimator
from the validated plan and delegates to shared original stateless/adaptive
implementations. A keyframe tracking-settings method keeps existing callers at
the original settings; the wide subclass supplies the separately fixed settings.
There is no mutation of original global controls. Both identities enter the
existing image generation/ownership/provenance path, including full failure rows.
All code/native/parent-plan bindings are recorded before publishing estimates.

No native API, shader, core dependency, camera model or transfer behavior changes.
Use the existing separate candidate environment:
`E:\SpatialRustBuild\pnp-consensus-v1\venv\Scripts\python.exe`.
The loaded extension remains SHA-256
`3a4cc031d171590acd760357397ff1029ff22b9f554600b8bb2d42e97ccad600`, built from native
source `cf457fd84d241714f005502a920cad5c4ef2c1d8`. Retain its wheel and the older
extension; do not rebuild or install into either frozen environment.

A window now samples 21x21 instead of 11x11 pixels and one more pyramid level;
extra tracking work is explicit. At most 300 retained tracks and 900 adaptive
detector candidates remain. Timing on a shared host is observation only, not a
throughput ranking or GPU acceleration claim.

## Execution and acceptance

1. Preflight at least 20GiB on `E:\SpatialRustData\maturity-80\`. Keep source
   images and old result hashes; create fresh checked hard links and metadata
   with `scripts/reuse_tum_prepared.py --capture-revision`. The explicit mode
   requires the exact prepared parent-plan hash and data-only validation that
   only the two declared settings change. Original reuse callers still reject
   changed controls. Zero new archive/image payload bytes.
2. Run each of the four fixed frontend/scene cases exactly once into fresh
   external result directories. Keep every 798 xyz / 722 rpy planned row and
   every original image/hash/geometry/support/typed-IO check. No prefix result
   can replace a complete trajectory or silently restart after loss.
3. Freeze all four estimates before interpreting any new reference score. Use
   unchanged `scripts/evaluate_tum_sequence.py` and the independent SciPy
   `scripts/audit_tum_trajectory.py`. Retain unmatched reference rows and uncovered
   one-second pairs. Numerical tolerances remain 1e-8m / 1e-6degree.
4. A candidate must generate 100%, evaluate >=95%, reach first-pose-aligned ATE
   RMSE <=0.10m and one-second translation RPE RMSE <=0.05m. Report all four
   outcomes and old complete/prefix comparisons even when every candidate fails.
   Incomplete-prefix scores cannot establish improved full-motion accuracy.
5. Independently verify source landmark IDs/depths, PnP masks, causal keyframe
   ordinals and pose/world composition. Verify all old hashes and hard-link
   identities. Record the loaded native extension and every survivor's role,
   bytes, hash and storage boundary without committing sensor payloads.
6. Run installed-wheel regression/full Python tests, Python 3.8 syntax and all
   required hosted checks. Native code/binaries remain source-bound and unchanged.
   Delete only completed dedicated pytest basetemp directories after resolved
   path, parent and reparse checks; preserve XML/logs, input data and old/new
   frozen results. Keep build/temp/cache writes on E.

## Results

Implementation and four fixed plans were committed at
`a6acccc80e97aed46055a6cfede9898720ddc04f` before any candidate fit. All four
estimates were frozen before new reference scoring. See
[the compact complete record](receipts/2026-10-19_wide_rgbd_capture.json).

| Method / scene | Generated | Evaluated | ATE RMSE (m) | 1s RPE RMSE (m) | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen parent xyz-rgbd | 798/798 | 795 | 0.122263 | 0.035838 | No |
| Wide xyz-rgbd | 798/798 | 795 | 0.114557 | 0.038457 | No |
| Frozen parent xyz-adaptive | 798/798 | 795 | 0.118821 | 0.028585 | No |
| Wide xyz-adaptive | 798/798 | 795 | 0.144424 | 0.039448 | No |
| Frozen parent rpy-rgbd | 2/722 | 2 | 0.011220 | unavailable | No |
| Wide rpy-rgbd | 2/722 | 2 | 0.015445 | unavailable | No |
| Frozen parent rpy-adaptive | 2/722 | 2 | 0.005037 | unavailable | No |
| Wide rpy-adaptive | 59/722 | 59 | 0.084487 | 0.124613 | No |

0 of four new cases meet every original gate. Incomplete-case ATE/RPE
covers only the accepted prefix and cannot establish complete-motion improvement.
Every one of the 3,040 planned rows, reference gaps and uncovered one-second pairs
is retained. All 16 older estimate/score hashes remain unchanged. The only
settings change is the committed radius-10/four-level LK capture configuration.

All four independent SciPy score audits pass unchanged 1e-8m/1e-6degree limits.
Source-depth/landmark IDs, returned-pose PnP masks and causal world/keyframe
composition independently match across 318,435 landmark rows. All 32 typed IO
audits are exact. Full installed-wheel Python passes 868 cases with three optional
ONNX skips, and all 71 focused cases pass. Changed source parses as Python 3.8.
Native source, loaded extension, old wheel and old extension remain unchanged;
no native rebuild or dependency installation is needed. Hosted checks remain
required before merge.

Checked hard links reuse 1,836,299,360 logical input bytes, with zero new archive or
image payload copies. New run/evaluation/audit files use 410,826,962 logical bytes.
Five successful dedicated pytest basetemps reclaim 773,231,900 logical bytes;
XML/logs, input data, native wheel/cache and every old/new frozen result remain.
These are logical sizes, not allocated-cluster/free-space deltas. The minimum
output free space remains 20GiB and all data/temp/cache writes use E.

## Post-freeze solver diagnosis

After all full estimates/scores are frozen, one exact failed rpy interval
(rows 1 -> 2) is replayed independently. Original registered PNG bytes, controls,
frame associations and the loaded native extension are checked. Captured RGB-D
correspondences reproduce the same native direct-PnP pose, mask and admission
error. All diagnostic sensor arrays remain external and are hash-bound.

The same 126 supplied float64 rows, camera and 2px criterion are passed once to
the existing OpenCV 4.13.0 EPnP RANSAC (seed 0, confidence .99, 2,000 maximum
iterations). All returned models are independently rescored on original float64
rows. Native has 56 inliers; OpenCV rescoring has 56. This
is numerical fitting evidence for one failed interval, not a motion/calibration
accuracy result, equivalent-algorithm test or replacement full trajectory.

OpenCV uses a five-point EPnP RANSAC model and internally converts inputs to
float32 in its pinned legacy implementation; native uses six-point hypotheses.
Sampling, initialization and fitting differ. These limits are explicit in the
[versioned primary implementation](https://raw.githubusercontent.com/opencv/opencv/4.13.0/modules/calib3d/src/solvepnp.cpp)
and [official pose documentation](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html).
OpenCV is an already installed private diagnostic dependency; this change adds
no production dependency or solver fallback.

The receipt also retains the native cold fit attempted on the independently
rescored OpenCV inlier subset. No reference motion or search over alternative
capture settings is used in this diagnostic. It informs subsequent native solver
work without rewriting any failed result above.

An additional post-freeze LK comparison uses the same two RGB images and all
203 initial native detector points. Native accepts 126 forward/backward tracks;
OpenCV accepts 128. On jointly accepted points the median target difference is
0.022385px; only two differ by more than 2px. Feeding the OpenCV tracks into the
same unchanged native PnP yields 58/128 inliers, still below 50%. Different
pyramids, gradients, precision, border handling and eigen-admission semantics
remain explicit. Neither this solver replacement nor this tracking replacement
accepts the interval; the diagnostic does not establish a physical cause. Depth,
registration/time consistency and wider real-motion robustness remain open.
These are two retained numerical diagnostics, not new full-trajectory candidates.

Canonical measured clock/physical calibration, Linux hardware-GPU execution and
the public-release boundary remain separate requirements. This study does not
prove the operational 80% goal.
