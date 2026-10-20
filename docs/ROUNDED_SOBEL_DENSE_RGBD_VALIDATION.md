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
pixels, plus 64 times base pixels, plus the greater of ceil(height/16) ordered
Normal records or 48 f64 values for nonconcurrent 6x6 solver matrix/RHS/solution.
This charges both fitting pyramids, original raw levels and the greater of four
current-level target gradient planes or two temporary smoothing planes. Inputs,
allocator metadata and stacks remain explicit exclusions. A pre-execution source
review corrected the tiny-image solver minimum before any numerical test or fit;
the original prospective protocol commit remains preserved. Cap at 128MiB by default,
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

## Observed outcome

The initial prospective protocol is `381b89a`; the pre-execution solver-scratch
review is `607415e`. Numerical/package/workflow source is `0b345e4`. The new
validated ABI3 wheel and 28-test executable were frozen before fitting; all
four older selectors retain exact old-wheel poses and reports in 20 synthetic
cases. Existing native parent lines remain byte-equivalent after line-ending
normalization, with only additive model declarations/dispatch.

Both complete development timelines were frozen before either new score.

| Scene | Generated | Reference matched | First-pose ATE RMSE | 1s translation RPE RMSE | Original acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| fr1/xyz | 798/798 | 795/798 | 0.05025800681626491m | 0.023050793987003902m | Pass |
| fr1/rpy | 722/722 | 716/722 | 0.10032699404221075m | 0.048503244138082525m | Fail: ATE > 0.10m |

Rpy exceeds the fixed ATE limit by 0.00032699404221075m; do not round that into
a pass or adjust the threshold. Its RPE passes, but full motion accuracy remains
unaccepted. Prior raw causal ATE was 0.089993/0.114793m; the external CPU control
was 0.051174/0.099684m. These are complete variant comparisons on observed
development recordings, not evidence of a component's causal effect or held-out
robustness. Every earlier variant and failure remains preserved.

Independent raw-image/causal/world auditing passes all 1,518 pairs, and both
independent trajectory score audits pass. Counts/fractions match exactly;
maximum photometric/depth RMSE differences are 2.78e-16/8.67e-18m. Maximum world
composition difference is 2.11e-14; independent score differences are at most
1.17e-15m and 3.84e-13 degrees. Publisher reference gaps remain 3/6; original
1s RPE denominators stay 762/768 and 680/692. Final raw correspondence fractions
stay at least 0.784046/0.637377, and independent full-cloud support at least
0.978885/0.960685. Each successful pair records 35 planned updates, 32,574,240
logical working bytes, 3,072,000 input-copy bytes and four CPU workers.

Validation before fitting: native 28, focused Python 143 and full Python 1,171
passes/3 existing optional ONNX skips. Native/bridge Clippy, WASM compile,
Rustfmt, Python 3.8 syntax and installed-wheel byte/runtime checks pass. The
initial test camera-type reference and WASM parallel-only variable compile
failures are preserved; constructor/scope fixes introduce no warning suppression
or model change. The [measurement receipt](receipts/2026-10-20_rounded_sobel_dense_rgbd.json)
binds all plans, source, packages, commands, logs, XML and independent results.

Scientific results use 322,171,414 bytes under
`E:/SpatialRustData/maturity-80/rounded-sobel-dense-study-v1/`. Retained new build
plus incremental cache growth is 46,392,418 bytes; maximum successful-command
sampled growth was 685,537,082 bytes, below 2GiB. Verified passing pytest scratch
cleanup reclaimed 641,419,300 logical bytes. All 339 prior JSON outputs and 36
frozen native artifacts (including this model's package/test executable, nine
wheels in total) are hash-verified. No sensor payload was downloaded or copied;
PNG inputs share checked existing hard links. E free space after measurement was
676,128,202,752 bytes, above the original 20GiB floor.

The opt-in model improves the observed ATE versus the raw causal control while
retaining full tracking, but rpy's accuracy gate remains closed. Further work
must fix a new complete contract before another fit, retain this result and all
original raw/full-timeline/accuracy gates, and choose any future held-out candidate
before reading held-out scores. Operational 80% remains unproven.
