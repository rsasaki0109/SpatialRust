# Persistent RGB-D keyframe landmark validation

## Fixed study scope

This Epic 146C candidate keeps metric depth landmarks from one keyframe while
tracking their pixels through successive images. It targets accumulated RGB-D
motion drift; it changes no SDK API, native solver, GPU kernel or production
default. The existing stateless image and depth-only failures remain evidence.

Two already observed development sequences, `freiburg1_xyz` and `freiburg1_rpy`,
are fixed before new fitting. Neither is held out. The implementation and all
three new plans are committed before rpy reacquisition or any candidate fitting:

- `benchmarks/tum-fr1-xyz-keyframe-rgbd-pnp-plan.json`
- `benchmarks/tum-fr1-rpy-keyframe-rgbd-pnp-plan.json`
- `benchmarks/tum-fr1-rpy-rgbd-pnp-plan.json` (new stateless image comparison)

The parent image protocol supplies identical controls, intrinsics, depth units,
full-cloud support, timeline, IO and accuracy limits to both scenes. The rpy
plans intentionally inherit that shared image protocol, rather than a new
calibration. The [official download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
provides the sequences, and [publisher formats](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
recommend default intrinsics for registered depth. This approximation is retained;
no depth rescaling, extra extrinsic, undistortion or fitted scale is added.

| Archive | Bytes | Previously recorded official-HTTPS SHA-256 |
| --- | ---: | --- |
| xyz | 448204271 | `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98` |
| rpy | 410268381 | `78103722a25873dbbb4de027eaa8c810a6382100691f02b6cf95c3adc91c4ac1` |

These locally recorded hashes are not publisher checksums. The new copy must
match them. The xyz archive/images already exist on the external data drive;
only the missing rpy archive is acquired. Credit: Sturm et al., RGB-D SLAM
benchmark, IROS 2012. Raw/derived sensor payloads are not redistributed here.

## Fixed estimator and frame convention

Native Shi-Tomasi and forward/backward pyramidal LK retain all parent settings:
300 maximum features, 40px margin, 0.75px roundtrip bound, 20 minimum tracks,
2px PnP threshold, 15 minimum inliers and 0.5 minimum inlier fraction. The
existing full-cloud 0.5 support requirement at 0.1m remains outside the adapter.

The first previous frame seeds features with valid .1..5m source depth. One
owned float32 metric depth image and the original source pixels are retained.
On each subsequent call, LK starts from the last accepted pixels; its valid
mask and the metric PnP inlier mask subset the original landmark IDs together.
PnP reuses keyframe depth at the original pixels, not the next noisy depth.
Only geometric inliers survive to the next call.

PnP returns `current_from_keyframe`. The last accepted pose supplies
`previous_from_keyframe`. Column-vector composition is:

```text
previous_from_current = previous_from_keyframe * inverse(current_from_keyframe)
world_from_current = world_from_previous * previous_from_current
```

This cancels the previous estimate within a keyframe; simply inverting every
keyframe pose and multiplying it into the world chain would apply it twice.
Synthetic noncommuting rotations/translations and actual native PnP check the
metric direction independently of real trajectory fitting.

The keyframe interval is exactly **30 successful pose steps**. At the next call
after that interval, the previous accepted frame supplies its depth and fresh
features. Thus anchors are ordinals 0,30,60,...; no future depth/pose is needed.
The world pose is never reset. Insufficient features/tracks/inliers, rejected
depth, nonrigid poses or wrong previous-image identity lose the chain permanently.
Every later timeline row remains an explicit failure.

## Storage and input reuse

`scripts/reuse_tum_prepared.py` verifies a frozen source manifest, unchanged
calibration/limits/original controls and every source PNG/index identity. It
requires absolute external data paths, a new sibling output and a 20 GiB
filesystem floor before opening input metadata, rechecked with 64 MiB headroom
before linking. It creates hard links, never a payload-copy fallback. Only plan,
calibration and manifest metadata is newly written. Source directories stay
unchanged; source/target identities are verified. A link failure preserves a
partial survivor inventory and no accepted manifest.

One rpy extraction serves both estimators through hard links. All source and
result paths are below `E:\SpatialRustData\maturity-80\`; build/test scratch
stays below `E:\SpatialRustBuild\`. Existing native wheel bytes are reused.
Each run preflights 20 GiB and records storage before/after. Size totals distinguish
shared logical input bytes from newly written metadata and result files.

Example xyz preparation:

```powershell
python C:\Users\rsasa\Workspace\SpatialRust\scripts\reuse_tum_prepared.py `
  --prepared-dir E:\SpatialRustData\maturity-80\tum-fr1-xyz\prepared-rgbd-v1 `
  --manifest-sha256 e7e98da77bed799ac281168bd6d81d737692cf7aa25b1c8443fb0485b0e87434 `
  --plan C:\Users\rsasa\Workspace\SpatialRust\benchmarks\tum-fr1-xyz-keyframe-rgbd-pnp-plan.json `
  --output-dir E:\SpatialRustData\maturity-80\tum-fr1-xyz\prepared-keyframe-rgbd-v1
```

Then use `scripts/run_tum_keyframe_rgbd.py` with the prepared directory,
its explicit manifest SHA-256 and a new output directory. The new rpy baseline
uses the unchanged `scripts/run_tum_rgbd_odometry.py` adapter.

## Fixed execution and acceptance

1. Verify/record the committed protocol, all code/plan/native hashes and the
   existing xyz baseline's frozen estimates/evaluation/audit. Reacquire rpy via
   verified official HTTPS, enforce its exact size/hash and retain source receipt.
2. Prepare every indexed image, withholding groundtruth and accelerometer
   contents. Retain all 798 xyz and 722 rpy depth-frame timeline rows, integer
   timestamps and existing RGB associations; never sample away lost frames.
3. Run xyz candidate, rpy stateless baseline and rpy candidate once each with
   fixed settings, freezing estimates before any new reference scoring. The
   older xyz stateless result is retained verbatim as the paired parent result.
   No fitting-stage reference pose, tuning, variable keyframe promotion or reset.
4. Evaluate all frozen cases with the unchanged scorer and independently audit
   every new score using SciPy at unchanged 1e-8m/1e-6degree tolerances. Preserve
   unmatched reference rows, uncovered RPE pairs, all failures and initial
   results. Recheck old xyz evaluation hashes and audit without rewriting them.
5. Accept a case only with 100% generated poses, at least 95% evaluated poses,
   first-pose-aligned ATE RMSE <=0.10m and one-second translation RPE RMSE <=0.05m.
   Both candidates must pass for a two-scene accepted study. Keep each individual
   outcome visible even if the other fails; do not relax thresholds.
6. Record typed IO audits, full-cloud support, keyframe ordinals, input/estimate/
   evaluation hashes, installed wheel hash, storage and any concurrent local work.
   Timing observations are not isolated throughput or speed rankings.

Local focused/full Python regressions and all hosted required checks must pass
before merge. Synthetic success alone cannot establish real accuracy.

## Results

Fixed-policy protocol/source commit: `2df466ecdd149f5aaa95b756696972c0060644c6`.
The official rpy archive matches its exact prior size/hash. Source imagery is
shared through hard links; no native rebuild is performed. See
[the complete compact record](receipts/2026-10-19_tum_keyframe_rgbd.json).

| Method / scene | Generated | Evaluated | Prefix ATE RMSE (m) | Prefix 1s RPE RMSE (m) | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen parent stateless xyz | 798/798 | 795 | 0.120235 | 0.033505 | No: ATE |
| Fixed keyframe xyz | 90/798 | 90 | 0.018775 | 0.021180 | No: completeness/coverage |
| New stateless rpy | 2/722 | 2 | 0.010364 | unavailable | No: completeness/coverage/RPE |
| Fixed keyframe rpy | 2/722 | 2 | 0.010364 | unavailable | No: completeness/coverage/RPE |

These tiny prefixes cannot establish improved full-sequence accuracy or a
method ranking. All **2,242 new planned rows** are present; once tracking is
lost, later rows explicitly remain errors. Missing reference poses (three xyz,
six rpy) and uncovered one-second pairs stay in the scorer's denominators.
All three new independent audits and a new audit of the frozen parent pass
unchanged 1e-8m/1e-6degree tolerances. Maximum new score differences are below
8.80e-16m and 8.53e-14degree. Four generated typed IO audits pass exactly.

XYZ loses tracking at zero-based row 90: the last accepted frame retains 18
inliers, below the next call's unchanged 20-track minimum. The age-30 promotion
would occur one call later. In rpy, the stateless row 2 PnP has 25/60 inliers,
below the unchanged 0.5 fraction. Persistent rpy lacks 20 forward/backward
tracks at the same row. Both failures show that a fixed interval alone is not
a sufficient frontend. No threshold or default is changed to accept them.

The xyz run starts independently while rpy downloads, and its completed frozen
result is resumed without rerunning it. The orchestration's parent-source check
initially assumes LF bytes for two older Windows files. Reconstructing their
CRLF checkout bytes from the exact parent Git blobs matches the original full
hashes; both identities and reconstruction are recorded. Data, plans and frozen
results are not normalized or rewritten.

Local validation passes 817 full-suite cases with three optional ONNX skips,
plus one additional unique partial-mask case; all 17 new focused cases pass.
The new files parse as Python 3.8. Hosted checks remain required before merge.

The fixed failures above are preserved for the separate input-quality revision
below. Canonical Autoware measured clock/physical calibration and full mapping
remain unresolved, so these studies do not establish the 80% operational goal.

## Input-quality revision

Not fitted yet. A count-triggered early promotion was proposed from xyz input
starvation before the new rpy image fits/reference scoring, with a fixed 60-track
capacity margin (three times the existing minimum). The completed fixed study
also exposes low rpy inlier support. The next separately committed protocol will
combine that input-only promotion with local source-depth quality selection;
retain every failed fixed result and the same completeness/support/accuracy gates.

### Separate input-quality protocol, fixed before adaptive fitting

The fixed study and all four independent audits are preserved in commit
`8843aea`. This second candidate is a development revision after observing those
failures. It is not a held-out result. Its implementation, tests and both
`benchmarks/tum-fr1-*-adaptive-rgbd-pnp-plan.json` files are committed before its
first fit. The same source archives and prepared full timelines are reused.

The new explicit `spatialrust_adaptive_keyframe_rgbd_pnp_cpu` method changes:

1. Detect at most **900** candidates with the original Shi-Tomasi quality,
   distance and margin, then retain at most the first **300** depth-qualified
   points in the detector's order. This increases the bounded detection work;
   the retained landmark/PnP budget remains 300.
2. At each candidate's original rounded pixel, require all nine 3x3 depth counts
   in the closed interval `[500,25000]`. With integer median `M` and maximum
   absolute deviation `D`, require `50*D <= max(5000,M)`. This is an exact
   integer-count comparison, equivalent to a max(0.02m,2% median depth) local
   spread rule under the fixed 1/5000m convention. The integer median replaces
   only the selected sample in the owned metric estimator-depth image. Original
   PNGs, raw frame projection, typed IO and full-cloud support remain unchanged.
3. In addition to the fixed maximum age 30, promote the previous accepted frame
   before alignment when fewer than **60** geometric-inlier landmarks remain
   (three times the unchanged 20-track minimum). This does not recover a lost
   chain or reset the world pose. Failed promotion preserves the prior cached
   depth/state and leaves tracking lost. Age-30 promotion remains the parent path.

These are explicit frontend algorithm changes, not sensor calibration or a new
noise model. A depth-qualification failure never falls back to unqualified
samples. Forward/backward 0.75px, PnP 2px, 20 tracks, 15 inliers/50%, full-cloud
50% support and all completeness/coverage/ATE/RPE criteria remain fixed.

The exact integer depth limits, both absolute/relative consensus boundaries,
holes, feature order/cap, threshold-60 behavior, noncommuting world composition
and failed state updates have new synthetic regressions. Parent fixed-keyframe
and stateless adapters are retained verbatim. The only common-runner change is
admission of the new explicit estimator name.

Execution order and admission:

- Create new `prepared-adaptive-rgbd-v1` directories through the same checked
  hard-link reuse helper; retain all source files and prior manifests.
- Run xyz and rpy once each to `run-adaptive-rgbd-v1`. Bind all estimator,
  parent/source/scoring/plan and existing native extension bytes. Freeze both
  complete timelines before new reference scoring.
- Use unchanged scoring and independent SciPy audits. Preserve every failure,
  reference gap and uncovered RPE pair, with no reruns selected by score.
- Both cases need full generation, >=95% evaluation, ATE <=0.10m and one-second
  translation RPE <=0.05m for a two-scene accepted revision. Record all outcomes
  even if only one case passes. A local passing prefix or synthetic test does
  not satisfy these requirements.
- Preflight the same 20 GiB disk floor and reuse existing native binaries and
  source image payloads. Keep the added detector work and any shared-host timing
  observations visible; make no performance ranking.

### Input-quality revision results

Algorithm commit `56dc4ed`; complete source/binding commit
`10780e3a6fded310c5cfed9b334b2dc466383be2`, before either adaptive fit. No new
archive, extraction or native build is required; both new prepared directories
share verified source payloads. All earlier fixed results are retained verbatim.

| Adaptive case | Generated | Evaluated | Prefix ATE RMSE (m) | Prefix 1s RPE RMSE (m) | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| xyz | 240/798 | 237 | 0.056469 | 0.030103 | No: completeness/coverage |
| rpy | 2/722 | 2 | 0.007455 | unavailable | No: completeness/coverage/RPE |

XYZ now reaches row 240, but the unchanged PnP inlier/depth admission stops it.
Rpy still stops at row 2. These are incomplete prefixes, not accepted full-motion
accuracy or a ranking against the complete parent trajectory. Every planned row
and all failed/unmatched counts remain. Both new independent audits pass, with
maximum differences below 8.82e-16m and 1.43e-13degree.

A separate trace/source audit checks **33,203** source landmark rows directly
against their original PNG 3x3 integer counts; all meet the declared consensus
rule. It checks persistent landmark IDs, causal keyframe ordinals, all pose
compositions, ten early xyz promotions, original support/roundtrip/inlier gates,
all hard-link identities and five new exact typed-IO file pairs. Thus the
remaining motion failures are not counted as acceptance or hidden by resets.
Together the fixed and adaptive studies preserve **3,762 new timeline rows**,
nine generated typed IO audits, five independent new-score audits and a separate
reaudit of the original xyz parent.

The final local suite has 834 passes/three optional ONNX skips, plus one extra
unique native detector-to-depth case; all 17 adaptive and 17 fixed new focused
cases pass. Both revised source files parse as Python 3.8. Hosted required
checks remain a merge prerequisite.

At the recorded storage snapshot, new study paths contain 961,778,948 unique
logical payload bytes (about 917 MiB), after excluding existing xyz inputs and
deduplicating hard links by file identity. Shared/repeated input payload totals
1,836,299,360 bytes. These are payload sizes, not NTFS allocated-cluster or
free-space deltas. All files remain external; the 20 GiB floor is retained.

### Post-freeze numerical diagnosis and next work

Diagnostic prefix replays occur only after both adaptive estimates and scores
are frozen. They neither replace a trajectory nor satisfy a study gate. The
original rpy second interval reproduces 22/58 inliers both before and after the
final refit, so refit regression is not its observed cause. Reference-reprojection
residuals remain large; image matching and calibration/time consistency remain
open beyond this short diagnostic.

An explicit four-setting LK prefix probe retains the same acceptance thresholds
and original feature margin. Valid first-interval track counts are 53 (r=5,
levels=3), 93 (5,4), 107 (10,3), and 125 (10,4). Several larger-motion settings
expose a native proper-rotation error. These exploratory prefixes are preserved
in the compact receipt; none is a new accepted trajectory or chosen full result.

For r=5/levels=4, a public-solver replay of the exact six-point sample sequence
finds a **54/93-inlier** hypothesis after 165 iterations. Its rotation determinant
is 1.0 and maximum orthogonality error 2.23e-16. The final fresh-DLT/inlier refit
then raises `absolute pose rotation must be a proper orthonormal matrix`, matching
the native failure. The existing native implementation discards the selected
hypothesis before that fallible refit. This establishes a concrete numerical
failure to address; it does not establish physical correctness of that hypothesis
or predict full-trajectory accuracy.

A separate follow-up will reproduce this with synthetic geometry, retain the
best valid robust-PnP hypothesis when optional refinement errors or worsens its
original consensus criterion, and verify all native/complete-data outcomes under
a new fixed protocol. Native solver/default behavior is unchanged in this PR.
Canonical measured Autoware calibration, full mapping and the operational 80%
goal remain unproven. The operational maturity estimate stays 63%.
