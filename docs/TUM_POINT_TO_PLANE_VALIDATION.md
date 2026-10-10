# Fixed native point-to-plane trajectory validation

This study adds an explicit native CPU point-to-plane adapter; default
point-to-point ICP, SDK APIs and runtime dependencies are unchanged. The known
`freiburg1_xyz` sequence is exploratory. `freiburg1_rpy` was fixed as an untouched
validation sequence before acquisition, fitting or reference interpretation.
Neither sequence uses reference poses for initialization, fitting or tuning.

## Protocol fixed before execution

Protocol/implementation commit: `a973dcb3afc811058670b96758328a173213788a`.
The three committed plans are `benchmarks/tum-fr1-xyz-plane-plan.json`,
`benchmarks/tum-fr1-rpy-plane-plan.json` and `benchmarks/tum-fr1-rpy-plan.json`.
They bind the original native plan's SHA-256 and retain its registered-depth
calibration, common CPU voxel points, depth limits, two-stage geometry/iteration
budgets, support test, tracking-gap limits, integer nanosecond times and typed IO.

Point-to-plane uses the existing native solver with 20-neighbor target normals,
20/20 iteration caps, its default 1e-8 transform and 1e-6 squared plane-fitness
stopping thresholds, and six minimum correspondences. This is a different
objective/stopping rule from point-to-point, not a speed or equivalent-stopping
comparison. Actual update counts and convergence flags are retained; the API
does not expose the stopping cause, so it is not invented. Singular motion is
an error; there is no fallback to a pose that the geometry cannot determine.

The acceptance conditions were committed before this method ran: every planned
frame generated, at least 95% of planned poses independently reference-matched,
first-pose-only SE(3) ATE RMSE at most 0.10 m, and one-second translation RPE RMSE
at most 0.05 m. These are engineering study limits, not publisher guarantees.
They are not relaxed after evaluation. No scale or best-fit trajectory gauge is
optimized, and no tracking reset or reference relocalization is allowed.

## Exploratory xyz outcome

All 798 frames generate, with 795 reference matches and 762/768 RPE pairs. ATE is
**0.125197 m**, versus 0.658033 m for the original point-to-point result; RPE is
**0.043704 m**, versus 0.175945 m. This is an observed improvement on already
examined data. **The fixed 0.10 m ATE condition fails**, despite the other three
conditions passing. No broad accuracy or new blind validation is claimed here.
The original three unmatched reference poses and six unevaluated RPE pairs are
retained. Independent SciPy score/coverage recomputation passes its previously
fixed numerical comparison tolerances; those tolerances are not sensor accuracy.

Frozen estimates:
`ab4f906cda9d104153d7659832dd50b5e6f11bc4b540bda69247eb4a6c7435dd`.
The standalone common-axis comparison, including the original Open3D diagnostic,
with corrected SLERP scoring is
`/workspace/SpatialRust/target/tum-reference/plane-xyz-report-v2/report.html`.

## Untouched rpy acquisition and validation

The official verified-HTTPS URL redirects to the already authorized
`webshare.cvg.cit.tum.de` host and returns HTTP 200. The complete archive is
410,268,381 bytes, SHA-256
`78103722a25873dbbb4de027eaa8c810a6382100691f02b6cf95c3adc91c4ac1`.
This is a locally recorded hash of official HTTPS bytes, not a publisher checksum.
Both preparation runs retain all 722 depth images and 723 RGB images, with zero
RGB-unmatched depth images, 1,452 archive members and 440,635,931 expanded bytes.
The depth timeline covers 24.061021 seconds. Groundtruth and accelerometer
contents are withheld from preparation and generation. Native point-to-point and
point-to-plane generation use separately frozen plans and manifests.

Both methods generate **722/722** frames and independently evaluate **716**
poses. Each projects **170,661,420** points and passes **14** exact typed IO
audits. Six reference-unmatched poses and twelve unevaluated one-second RPE
pairs remain in the denominators.

| Corrected rpy result | Native point-to-point | Native point-to-plane |
| --- | ---: | ---: |
| First-pose translation ATE RMSE | 0.726446 m | 0.529330 m |
| Translation RPE RMSE | 0.264944 m | 0.138976 m |
| Rotation RPE RMSE | 15.655371° | 7.329991° |
| Generated / planned poses | 722 / 722 | 722 / 722 |
| Evaluated / planned poses | 716 / 722 | 716 / 722 |
| Evaluated / planned RPE pairs | 680 / 692 | 680 / 692 |
| Fixed ATE and RPE limits | Both fail | Both fail |

The held-out sequence corroborates lower error under this fixed alternative,
but **neither method meets the accuracy acceptance conditions**. Successful
processing and over 99% reference coverage do not establish accurate motion.
No thresholds or estimator settings are changed after these results.
Unmatched indices are 458, 459, 460, 461, 463 and 464; exact timestamps,
enclosing reference gaps and all twelve missing RPE pairs are in the compact
receipt. The standalone figure/report is
`/workspace/SpatialRust/target/tum-reference/plane-rpy-report-v2/report.html`.

## Independent audit exposed a scoring defect

Both initial rpy audits reject their first-pose gauge. The old evaluator uses
normalized linear quaternion interpolation when the endpoint dot product exceeds
0.9995. Away from the midpoint this approximates, rather than equals,
constant-angular-speed SLERP. Against SciPy, the largest reference rotation
difference is 6.1354e-6°, and the first gauge has a 7.6503e-9 matrix-element
difference. Original v1 scores and the rejected audit diagnosis are retained.

Commit `7a5751cb275ca323692506fcebb9b71965479703` uses stable sinc weights for
exact SLERP, including zero/tiny-angle and quaternion-sign cases. Six analytic
regressions test quarter-interval constant angular speed. No frozen trajectory
is regenerated, and its historical generation-helper hashes are preserved.
Both rpy methods and all three xyz methods are rescored into separate v2
directories. **All five independent audits pass unchanged tolerances**. The
largest aggregate ATE correction is 2.655e-10 m: this is a numerical evaluator
fix, not a physical odometry improvement. The previous accuracy conclusions
and fixed acceptance outcomes remain unchanged.

## Verification and limits

Both Python 3.8 / PyArrow 17 and Python 3.12 / PyArrow 26 pass **696** applicable
tests, with three expected ONNX skips each. The 43 TUM regressions include known
current-to-previous motion recovery, rejection of an unobservable single plane,
method identity, frozen normal/control/calibration settings and exact sequence
archive-root isolation. The real generation paths continue independent numerical
projection and exact typed xyz/raw-depth/pixel-index IO checks.

Reviewable data is in
[`receipts/2026-10-10_tum_point_to_plane.json`](receipts/2026-10-10_tum_point_to_plane.json).
The first hosted Windows wheel run rejected the original-plan hash because
Git's CRLF checkout conversion changed its raw bytes. Repository attributes
now require LF checkout for benchmark JSON protocols. A simulated
`core.autocrlf=true` checkout retains the exact precommitted SHA-256; input hash
checks are not normalized or relaxed. The repaired head requires its own full
hosted checks before merge.
The provisional maturity remains **87%**: the bounded improvement and numerical
scoring repair do not close accurate real motion or canonical calibration gates.

The publisher-recommended intrinsics remain an approximation. This study does
not measure clock synchronization or certify the original Autoware bag's
front/rear mounting. Groundtruth is used only after estimates freeze. Inputs,
plans, native/helper bytes, frozen estimates, references, scores and numerical
audits are hash-bound; original frozen files are retained.

Data attribution: TUM RGB-D Benchmark, Jürgen Sturm, Nikolas Engelhard,
Felix Endres, Wolfram Burgard and Daniel Cremers, *A Benchmark for the Evaluation
of RGB-D SLAM Systems* (IROS 2012). The official overview specifies CC BY 4.0 for
the data. Archive/images/point-cloud dumps/native binaries are local validation
artifacts and are not committed. Source-bound measured clock/mounting calibration
and broader real-motion evidence remain material for the 90% target.
