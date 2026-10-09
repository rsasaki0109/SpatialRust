# Separate coarse and fine correspondence gates

The file and candidate Python examples accept `--fine-distance` (or the
`fine_distance` keyword). The voxel ICP stage uses max-distance and the
full-resolution stage uses fine-distance, defaulting to max-distance for
backward-compatible behavior. Evaluation distance remains independent and
still defaults to max-distance. Both stage gates and the resolved fine gate
are recorded in JSON; the standalone HTML displays them. Numeric validation
rejects nonpositive, nonfinite or unusable f32-square distances before file IO.

Tests verify native call arguments, per-stage metadata, default versus explicit
equal-gate equivalence, independent evaluation gates, early invalid-input failure,
candidate CLI output and fine-gate HTML. All 219 Python tests passed.

The public-pair harness also accepts the flag. The controlled comparison tool
supports `--vary fine-distance`; it normalizes omitted old fine gates to
max-distance and still requires identical inputs, supplied poses and every
other setting. The existing iteration-only comparison remains supported.

Reproduce the real-pair experiment:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/validate_public_icp_candidates.py \
 --iterations 30 --fine-distance .02 \
 --output-dir /workspace/SpatialRust-python-delivery/target/public-icp-fine-02

python3 /workspace/SpatialRust-python-delivery/scripts/report_icp_iterations.py \
 /workspace/SpatialRust-python-delivery/target/public-icp-iterations-30 \
 /workspace/SpatialRust-python-delivery/target/public-icp-fine-02 \
 --vary fine-distance \
 --output-dir /workspace/SpatialRust-python-delivery/target/public-icp-fine-comparison
```

Both runs use .1 m coarse gate, .02 m evaluation gate, .05 m leaf and 30
iterations per stage. Every new candidate again matches independent alignment,
with exact aligned-XYZ PCD roundtrip.

| Candidate | Fine .1 m: forward support / RMSE | Fine .02 m: forward support / RMSE |
| --- | --- | --- |
| Identity | 12.47% / 12.560 mm | 12.95% / 7.952 mm |
| Projected tutorial prior | 62.28% / 7.453 mm | 62.12% / 6.565 mm |
| Perturbed tutorial prior | 62.28% / 7.451 mm | 62.11% / 6.564 mm |

The selected index changes from 2 to 1 because the projected prior now supports
two more original source points, despite slightly higher gated RMSE than the
perturbed prior. Selected reverse support changes from 93.87% to 93.67%.
Single observed candidate-search time changes from 34.71 to 12.45 seconds.
All candidates remain nonconverged; the selected run uses all 30 iterations in
each stage. No independent true pose is available.

Tightening only the fine gate changes accepted correspondences, yielding lower
measured residual with slightly reduced support for the nearby priors. It also
changes bounded-search work, so this is a parameter tradeoff rather than an
output-preserving optimization. The observation does not prove improved pose
accuracy or universally better settings. The identity prior remains weak despite
the narrower gate. Initial pose, coarse alignment and fine correspondence choice
must be examined together; one public pair and single timings limit generality.
