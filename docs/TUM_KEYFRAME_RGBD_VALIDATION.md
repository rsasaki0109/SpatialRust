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
