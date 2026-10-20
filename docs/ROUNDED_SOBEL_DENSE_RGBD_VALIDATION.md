# Native rounded-pixel Sobel dense RGB-D validation

Operational slice 146P, fixed before implementation or new fitting. The source-bound
operator audit confirms grid/lookup differences but does not establish their cause
in native rpy error. Implement one complete opt-in fitting model, keeping every
prior selector, result, failure, default and operational 80% requirement.

## Fixed model and API

Add safe non-exhaustive `DenseRgbdFittingModel::{Bilinear,RoundedSobel}` and
`estimate_dense_rgbd_odometry_with_model`, alongside all existing entry points.
Bilinear delegates to the unchanged preprocessing path. RoundedSobel requires the
legacy preprocessing selector Raw and bundles the complete model in the committed
`dense_fitting_model` plan. Python adds `fitting_model="bilinear"` as a keyword;
only the selected new model adds its model tag to the report. Reject mixed policy
requests before input copying/allocation. Old reports remain identical.

Initial gray/metric depth receive replicated-edge Gaussian3 [1,2,1]/4 on each axis;
nonfinite depth taps propagate. Initial rounded correspondences define streaming
row-major gray means and positive finite gains to mean 0.5, applied once without
clipping. Check count and both gains before mutation. At lower levels, color gets
Gaussian3 then 2x2 mean; depth gets only 2x2 mean, propagating nonfinite values.
Dimensions round down; all scheduled levels must be at least 3x3. Halve all four
pinhole intrinsics each level.

Project source points with positive finite transformed z. Compute target indices
as trunc(projected_coordinate+0.5), with finite/signed bounds checked before
integer conversion, including the explicitly tested negative fringe. Use one
finite selected target depth and the original depth-difference gate, without
claiming target uniqueness. Replicated-edge Sobel3 derivatives use scale 1/8;
nonfinite depth derivatives become zero. These are local surrogate Jacobians,
not exact derivatives of the discontinuous rounded objective. Keep native f64
arithmetic, existing proper left Rodrigues updates and the original solver;
execute all scheduled iterations. No bitwise or complete Open3D equivalence claim.

Final support, unweighted residual statistics and acceptance use the original raw
full-resolution bilinear path. Keep causal previous-increment seeds, world chain,
complete rows and independent full-cloud/score/IO gates. The original ATE 0.10m,
1s RPE 0.05m, generated 100% and evaluated 95% limits stay fixed.

## Ownership, storage and execution

Borrow native inputs; own all four Python inputs before releasing the GIL.
Preflight logical working bytes before allocation: 32 times total floor-pyramid
pixels, plus 64 times base pixels, plus ceil(height/16) ordered Normal records.
This charges both fitting pyramids, original raw levels and the greater of four
current-level target gradient planes or two temporary smoothing planes. Inputs,
allocator metadata and stacks remain explicit exclusions. Cap at 128MiB by default,
use at most four private CPU workers with deterministic ordered row reductions;
WASM is serial. Add no core/heavy dependency, unsafe code or hidden device copy.

Reuse `E:/SpatialRustBuild/maturity-80-target/` offline. New owned build/test root
is `E:/SpatialRustBuild/native-dense-rounded-sobel-v1/`; cap it plus incremental
cache growth at 2GiB. Reuse source archives and PNG preparation without copying
payload or downloading sensor data. Study output stays under
`E:/SpatialRustData/maturity-80/rounded-sobel-dense-study-v1/`, including indexed
case result directories, capped collectively at 1GiB. Keep >=20GiB free throughout.
Limit owned build/test commands to one hour and scene/audit commands to two hours;
kill only the owned process tree on a breached limit. Record before/after bytes,
commands, all source/executable/module/plan hashes and original failure logs.
Preserve all prior scientific outputs, JSON, wheels, executables and libraries.
Only exact passing pytest scratch can be reclaimed after XML, absolute parent/
target and full no-reparse verification.

## Required verification and delivery

Check analytical gray/gradient/Jacobian bases, rounding/border/NaN/odd dimensions,
support and gain admission, exact preflight boundaries, proper poses, complete
iteration counts, deterministic serial/parallel reductions, Python input ownership
and schema/stub behavior. Compare all four old selectors on the same five synthetic
cases against the frozen previous wheel, requiring exact old report/pose equality.
Run native and full Python tests, bridge/native Clippy, WASM compile and Rustfmt
before full fitting. Freeze the new package and both complete xyz/rpy timelines
before reading either new score. Independently audit every causal/raw/world pair
and both full-scene scores. Preserve all failure rows and original score gaps.

Candidate selection precedes any future held-out recording score. Canonical measured
clock/physical frame calibration, both raw Linux hardware classes and complete
mapping/semantic/release acceptance remain required. Green tests or this development
comparison alone cannot prove operational 80%. Deliver as one named PR and merge.
