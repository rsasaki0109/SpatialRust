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

Before public fitting: the final installed-wheel suite passes 868 cases with
three optional ONNX skips. All 71 focused cases pass, including 31 new cases
for strict plans, explicit parent-bound reuse, native forwarding/row ownership,
noncommuting pose direction and complete lost timelines. Changed Python files
parse as Python 3.8. Native source and both frozen extensions are unchanged.
Public candidate fitting and all reference scores remain pending.
No partial result establishes the operational 80% goal. Canonical measured
clock/physical calibration, Linux hardware-GPU execution and the public-release
boundary remain separate unresolved requirements.