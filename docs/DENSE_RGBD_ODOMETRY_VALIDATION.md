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
