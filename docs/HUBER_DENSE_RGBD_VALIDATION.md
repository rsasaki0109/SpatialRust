# Fixed Huber dense RGB-D fitting comparison

Slice 146V adds opt-in `DenseRgbdFittingModel::ReferencePrecisionHuber` /
Python `fitting_model="reference-precision-huber"`. The
[prospective protocol](../benchmarks/huber-dense-plan.json) was committed as
`51f6b99eeb71415dfe9b912586452d652ed4527e` before implementation, build and new fitting.
The [public receipt](receipts/2026-10-20_huber_dense_rgbd.json) binds the numerical
source `214bfd7e17ce52b8644197eddc33e3c6290a5b55`, separately frozen wheel, extension,
test executable, complete evidence and every earlier artifact.

The single fixed candidate passes the original gates on **xyz**. These
four recordings were already observed development data. The fixed thresholds
are engineering priors; the previous room drift audit establishes neither
outliers nor an error cause. No parameters, scene selection or accuracy gates
changed after observing any new score.

This fixed Huber candidate regresses the previously accepted rpy/plant
recordings. It remains an explicit experimental option; these results do not
support promoting it as the preferred model.

## Numerical and ownership contract

Retain the reference-precision pipeline: f32-intermediate separable Gaussian
and Sobel operations, once-only pair mean 0.5 normalization, floor pyramids,
signed rounded matching on native f64 projections, separately f32-quantized
fitting XYZ, a checked f64 normal solve and proper left `Rz*Ry*Rx` increments.
All 35 scheduled updates remain. The selector requires `preprocessing="raw"`;
all other fitting preprocessing combinations reject before fitting.

For each scalar residual `r`, use Huber loss `0.5*r*r` when `abs(r)<=delta`,
otherwise `delta*(abs(r)-0.5*delta)`. Its IRLS factor is one at zero/inside the
threshold, otherwise `delta/abs(r)`. Photo delta is 0.02 in gray units after
pair normalization; depth delta is `max_depth_difference/3` in metres (0.01m
with the original 0.03m gate). Scale each Jacobian and residual by
`sqrt(original scalar weight) * sqrt(Huber factor)`, in that f64 order.
Invalid residuals/thresholds or a factor underflowing to zero propagate NaN to
the finite-system rejection. The derived depth delta must be finite and
strictly positive during preflight; rows are never silently removed or
assigned zero weight to mask invalid arithmetic.

Fitting SSE/counts and all final original raw bilinear measurements stay
unweighted. Minimum support 0.5, metric depth/count/finite admission, causal
identity then preceding returned increment, world composition, full-cloud
generation and typed IO retain their original contracts. No reference enters
generation, and no retry, reset, trimming, rescaling or per-scene model switch
is allowed. All four complete timelines were frozen before the first score.

All six old selectors/defaults/reports remain. The old `rounded_sobel.rs` and
`reference_precision.rs` are byte-identical to the parent. The new model is
isolated; its owned f64 buffers hold explicitly f32-quantized values and do
not claim f32 memory savings. Checked working bytes remain
`32*sum(floor-pyramid pixels) + 64*base_pixels + max(tile partials, 48*f64)`:
32,574,240 bytes at 640x480 with the original schedule. Caller storage and
Python-owned copies are charged separately. Native ordered tile reductions
use at most four CPU workers; WASM stays serial. Core/math/GPU trees,
dependencies, safe API boundaries and explicit CPU/GPU ownership do not change.

## Complete observed development results

Original gates: generated fraction 1.0, reference match fraction at least 0.95,
first-pose-anchored ATE RMSE at most 0.10m and 1s translation RPE RMSE at most
0.05m. No trajectory-wide fit is substituted for the first-pose gauge.

| Scene | Generated / planned | Reference matches | Previous ATE / RPE (m) | Huber ATE / RPE (m) | Original gates |
| --- | ---: | ---: | ---: | ---: | --- |
| xyz | 798/798 | 795 | 0.050582 / 0.023075 | 0.055052 / 0.022474 | pass |
| rpy | 722/722 | 716 | 0.099679 / 0.048503 | 0.267810 / 0.058021 | fail |
| plant | 1139/1139 | 1130 | 0.078227 / 0.037715 | 0.215782 / 0.041347 | fail |
| room | 1360/1360 | 1359 | 0.378653 / 0.058375 | 0.376725 / 0.053511 | fail |

The complete 4,019 generated poses and 4,015 causal/raw pair checks,
all gaps, original failures and all previous candidates remain. Each score has
a separate SciPy SLERP/rotation audit; raw pixel/support/depth diagnostics,
proper poses, seeds, world direction and complete timelines are independently
checked. Development passes alone do not prove independent robustness.

## Verification and storage

The 42 native dense tests include seven new checks of the independent Huber
loss derivative/weight, exact unclipped scale, nonfinite/underflow admission,
robust scalar objective, analytic clipped normal equations/unweighted SSE,
ordered serial/parallel reduction, budget boundaries and original identity
admission. Python has 190 focused and 1,337 complete passing tests plus three
existing skips. Both Clippy targets, WASM compilation, installed wheel and
stubs pass. Thirty cases across all six old choices match the preceding
frozen extension exactly. Six frozen synthetic inputs agree with an
independent NumPy H/J/Huber solve to maximum pose component error
6.5052130349130266e-17, below 1e-10;
the stronger case exercises both clipped photo and depth residuals. No new
external reference fit or observed external runtime Jacobian is claimed.

The initial native test compilation failed for a missing test-only `Image`
import. That source, compiler log and command receipt remain; the import fix
changed no production weighting. No scientific score was retried.

| Owned root | Logical regular-file bytes |
| --- | ---: |
| `E:/SpatialRustBuild/native-dense-huber-v1` | 23,723,990 |
| `E:/SpatialRustData/maturity-80/huber-dense-input-v1` | 2,502,843,067 |
| `E:/SpatialRustData/maturity-80/huber-dense-study-v1` | 817,643,860 |

Positive shared Cargo-cache growth is 27,267,880
bytes; combined new logical trees/cache are 3,371,478,797 bytes against the fixed
8GiB cap. PNG/index views use hard links: sensor downloads and new image
payload copies are zero. Complete fail-closed inventories charge every linked
pathname logically; physical free space remains above 20GiB. Exactly
722,930,025 bytes of successful pytest scratch were removed after terminal,
XML, absolute-path and no-reparse checks. All 439 prior scientific JSON and
39 old frozen native artifacts remain byte-identical.

Independent genuinely unobserved motion qualification, both raw Linux hardware
classes, matching canonical measured clock uncertainty and physical frame
accuracy, and complete canonical mapping/semantic/Viewer/interchange/release
acceptance remain required. CI and development passes do not establish 80%.
