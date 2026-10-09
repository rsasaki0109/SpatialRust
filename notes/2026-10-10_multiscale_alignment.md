# Explicit multiscale file alignment

Added an example workflow with 1–16 validated stages, non-increasing voxel sizes
and distance gates, and a mandatory full-resolution final stage. Per-stage trim
and convergence settings are explicit. Corrections compose in target frame and
each stage transforms the original source; repeated target voxel resolutions
are cached. Outputs preserve scalar attributes and rotate normals.

Verification: 314 Python tests passed, including independent legacy two-stage
equivalence, three-stage composition, target-cache reuse, rotated/translated
prior, typed scalar attributes, normals, input immutability, validation before
IO, exclusive output and cleanup after a partial write.

Public-pair artifacts:
`/workspace/SpatialRust-python-delivery/target/public-multiscale-validation/`.
Schedule: voxel .1/gate .2/30 updates, voxel .05/gate .1/30 updates,
full resolution/gate .02/30 updates with translation/rotation thresholds 1e-4
and disabled absolute-fitness stopping. The supplied prior comes from the
previous public candidate run. Actual updates: 30, 30, 28; the final stage stops
on transform thresholds. Forward support is 123,517/198,835 at .02 metres,
gated RMSE .006567129 m. Reverse support is 129,117/137,833, RMSE .006370175 m.
The full-source saved PCD roundtrips XYZ exactly. JSON receipts fingerprint
inputs, pipeline and native extension; HTML shows all three stage histories.

Single local run took 4.90 seconds, including support evaluation. This is not a
controlled speed comparison. There is no verified ground-truth pose for this
pair; proximity and convergence do not certify physical alignment. Open3D,
PCL and OpenCV comparative superiority remains unestablished. Maturity 57% is
a provisional engineering estimate; the long-term goal remains 90%.
