# Dense projective RGB-D motion validation

Operational slice 146G. This protocol is committed before native dense fitting
or full comparator trajectories on the two already observed development scenes.
The preceding single-interval feasibility probe used existing Open3D 0.19.0:
hybrid projective RGB-D returned a proper pose and 0.996324 full-cloud support
where sparse PnP retained 56/126 rows. This is neither a native result nor a full
trajectory. Seeded PnP LM retained 56/126 and is not adopted as a motion fix.

## Data and execution contract

Add a dedicated `odometry-dense` feature to `spatialrust-vision` and an explicit
facade feature. A checked borrowed frame contains one u8 grayscale view and one
f32 metric-depth view of equal geometry. An estimation call explicitly consumes
source and target frames, calibrated pinhole intrinsics, a proper source-to-target
initial pose and bounded options. Return the pose, correspondence/iteration/error
and logical working-storage receipts. No core dependency, unsafe code or implicit
device transfer. Python owns all four input copies before releasing the GIL.
Original sparse APIs, tracking settings, native wheels and results remain intact.

The CPU implementation jointly minimizes normalized photometric and projective
depth residuals using a coarse-to-fine pyramid and six-parameter left pose updates.
Analytic image/depth warp derivatives are checked against independent finite
differences. Target sampling is bilinear with four valid depth corners. Pyramid
samples use a centered 3x3 [1,2,1] Gaussian at even coordinates; depth averages
only valid samples with at least half of kernel weight. Camera focal/principal
coordinates scale by two per level. This is an independent implementation, not
bitwise Open3D equivalence. Fixed depth weight 0.968 and iteration schedule
20/10/5 follow the published hybrid family; see
[the primary implementation](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/odometry/RGBDOdometryJacobian.cpp)
and [the official RGB-D tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html).

Fixed row tiles reduce in index order. At most four private native workers may
evaluate independent tiles; WASM uses the same serial reduction. Logical owned
pyramid/reduction payloads are preflighted against 128MiB before allocation.
Input Python copies and thread/allocator overhead are recorded or excluded
explicitly; these counters do not measure peak process or GPU memory.

## Fixed full-sequence experiment: initial strict-iteration policy

Use existing registered TUM Freiburg 1 xyz and rpy frames under
`E:/SpatialRustData/maturity-80/`, retaining all 798/722 planned rows. Bind exact
source PNG hashes, nearest RGB timestamps, manifests and original camera/depth
units. Reuse checked hard links in fresh prepared directories with zero new
dataset downloads or image copies. Default intrinsics remain 525/525/319.5/239.5;
depth range remains 0.1..5m. No new distortion, scale or extrinsic correction.

Freeze the two scene plans before either full run. The explicit dense adapter
uses both adjacent registered depth and grayscale frames and identity initial
motion for every pair. Use 20/10/5 coarse-to-fine iterations, 0.03m depth gate,
0.968 depth weight, 1e-7 update convergence, at least 30 projective pairs and
0.5 support among valid source pixels. These are dense-model admission rules,
not a relaxation of the unchanged sparse PnP gate. Preserve the original
independent full-cloud 0.5 support at 0.1m, all image/projection/typed IO checks,
the single world anchor and loss without resets. No future frame, reference
initializer or ground-truth fitting is allowed.

Freeze both complete estimate files before reading new reference scores. Keep
100% generation, >=95% reference coverage, first-pose-aligned ATE RMSE <=0.10m
and one-second translation RPE RMSE <=0.05m. Independent SciPy score auditing
retains 1e-8m/1e-6degree tolerances and all reference gaps. Preserve both results
and every earlier failure, including incomplete trajectories. No setting search,
post-score gate revision, timing ranking or held-out generalization claim.

Independent NumPy reprojection auditing checks every successful adjacent pair's
complete valid-source count, four-corner target depth mask, final correspondence
count and fraction, photometric RMSE and metric depth RMSE. Counts are exact;
fraction tolerance is 1e-12 and RMSE tolerances are 1e-10. The inherited sparse
control dictionary remains only for checked source-plan reuse; the dense model
does not consume those detector, LK or PnP settings.

Build/install only into a new E-drive environment. Preserve both frozen wheels
and extensions, original GPU binaries, source archives and prior results. Data,
build, cache and temporary outputs stay on E with a 20GiB free floor; cap new
study outputs at 1GiB. Publish hashes, bounded storage and all actual outcomes.
Canonical measured clock/physical calibration, Linux hardware operations and
release acceptance remain separate open requirements for the operational 80% goal.

## Explicit final-acceptance solver revision

The first source/wheel at `ece33c25513c757958f5b7b1780bdd087125cade` preserves
xyz 75/798 and rpy 1/722; both fail. All 1,520 planned rows, scores and audits
are frozen before diagnosis. Independent identity-warp checks at the first
failed intervals find 0.517172 initial fraction for xyz and 0.331943 for rpy.
A separately rendered 0.05-radian planar scene reproduces rejection from a
below-50% initial hypothesis although the correct pose has high overlap.

The revised native solver requires >=30 correspondences, finite residuals and
a nonsingular normal system while fitting. The unchanged >=0.5 fraction is
checked on the returned full-resolution pose. This separates a numerical
hypothesis from an accepted estimate; no final confidence, geometric support
or accuracy limit changes. All numeric options, objective, pyramid, iterations,
identity initialization and datasets remain fixed. Richer support errors report
level/iteration/counts. Native receipts declare `final_support_only`; adapters
reject a mismatching solver policy rather than silently relabeling old plans.

The two separately named `tum-fr1-{xyz,rpy}-dense-final-overlap-plan.json` plans
are committed before revised native/full-data execution. Keep the original
strict-policy wheel/extension and results; build a new E-drive wheel/environment
and use fresh `prepared/run/evaluation/audit` directories. Freeze both revised
estimates before new scores, and rerun independent projective/trajectory audits.
The original source and plans remain reproducible with their frozen old binary.

## Frozen full-scene outcomes

Both revised estimates were frozen at `2026-10-19T14:51:43.464678+00:00`
before either revised reference score. Source commit
`1e56c9352f6b91aaccec97e946b10b5feec651c0` and the loaded extension SHA-256
`f2112078a97484461fd3f2e653204a8394839784f833f7060abaf85c94a70d88`
bind the actual numerical implementation. The earlier strict variant remains
bound to `ece33c25513c757958f5b7b1780bdd087125cade` and extension
`f3a8717ec97a04e91145b3c1dd01eebfe5abdddc28924736acb42b36614014f4`.

| Fixed method / scene | Generated / planned | Reference evaluated | ATE RMSE, m | 1s RPE translation RMSE, m | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| Strict iteration / xyz | 75/798 | 75 | 0.023091* | 0.016479* | Fail: incomplete |
| Strict iteration / rpy | 1/722 | 1 | Anchor only | No pairs | Fail: incomplete |
| Final-pose overlap / xyz | 798/798 | 795 | 0.090314 | 0.025987 | Pass this development scene |
| Final-pose overlap / rpy | 40/722 | 40 | 0.055276* | 0.115722* | Fail: incomplete and RPE |

`*` denotes a surviving prefix, not a full-sequence accuracy result. The initial
rpy numerical ATE of zero is only its sole identity anchor. Original reference
gaps remain three xyz and six rpy rows. Revised xyz evaluates 762/768 requested
one-second pairs; rpy evaluates 10/692. No failed frame is dropped from either
1,520-row experiment, and no trajectory restarts after loss.

At revised rpy index 40 (zero based), the returned full-resolution hypothesis
has 95,855/228,114 correspondences, fraction 0.4202065633849742. The unchanged
0.5 acceptance gate rejects it; all 682 failed rows remain. The protocol does
not justify further gate changes or a physical diagnosis of this loss.

All four independent final-projective audits pass: 74/0 successful adjacent
pairs in the original xyz/rpy runs and 797/39 in the revised runs. They check
all source rows and the causal world composition. All four SciPy trajectory
audits pass their fixed score tolerances. Passing a score audit establishes
correct arithmetic and coverage accounting; rpy remains unaccepted.

The native implementation passes 175 tests (two existing ignored tests), 13
vision property tests, Clippy with warnings denied, the explicit facade feature
check and `wasm32-unknown-unknown` compilation. The low-initial-overlap rendered
regression fails on the strict parent and passes after the solver revision;
a separate exact-residual/low-overlap regression verifies rejection at return.
Python tests cover owned and strided inputs, concurrent calls, typed failures,
policy mismatches, independent residuals and inverse/world composition.

The distribution check also reproduces and corrects four pre-existing missing
public names in the stub `__all__`. Commit
`43ad0ddfa94b02d2575c4862caf7cfa9d5be6e9d` changes only typing metadata and
its contract test after scientific results are frozen. A fresh wheel contains
the identical numerical extension above; every wheel RECORD digest and all 135
public export names are verified. Original experiment environments are preserved.
The final wheel passes 942 Python tests with three existing optional ONNX skips
and the repository's runtime/stub comparison command.

## Storage, reproducibility and remaining scope

The compact [machine receipt](receipts/2026-10-19_dense_rgbd_odometry.json)
records exact plans, manifests, estimates, audits, wheels, tests and storage
counts. Original evidence remains under the absolute directories
`E:/SpatialRustData/maturity-80/dense-rgbd-study-v1/` and
`E:/SpatialRustData/maturity-80/dense-final-overlap-study-v1/`; each contains
`study.json`, preparation, freeze and process logs. Use the original source
commit and wheel for the strict-policy CLI; the revised CLI explicitly requires
the native `final_support_only` receipt. New estimates and both audit CLIs require
fresh output directories and the exact prepared-manifest/estimate hashes.

Each preparation reuses 1,598 xyz and 1,447 rpy existing PNG/index files as checked
hard links. Dataset downloads and additional image payload copies are zero. The
original study retains 32,570,302 logical result bytes and the revised study
169,843,273 bytes, each below its 1GiB cap. At 640x480 the solver reports
12,913,440 logical pyramid/reduction bytes plus 3,072,000 Python input-copy
bytes per call, with at most four native CPU workers. Thread stacks, allocator
overhead and peak process memory are outside these counters.

Only completed, passing, explicitly owned pytest scratch directories are
eligible for cleanup after absolute-path and no-reparse checks. Their XML/log
evidence, all old results, source archives, wheels and native/GPU binaries remain.
Builds, caches, data and temporary files use E with the fixed 20GiB free floor.

This adds an explicit CPU model and one accepted public development sequence.
There is no GPU implementation or speed comparison for this dense solver yet.
The failed rpy sequence, independent measured canonical clock/physical
calibration, second hardware-class operation coverage and full release/mapping
acceptance remain open. These two previously observed development scenes do
not establish held-out robustness, competitor parity or operational 80% maturity.
