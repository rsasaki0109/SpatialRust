# More ICP iterations change support but do not guarantee convergence

The public-pair harness now accepts `--iterations`, default five, validated as
positive before archive access or output creation. Progress messages identify
candidate search and each independent comparison. A 30-iteration run completed
after bounded search made this validation practical, and every candidate again
matched independent alignment with an exact aligned-XYZ PCD roundtrip.

Reproduce the new run and offline comparison:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/validate_public_icp_candidates.py \
 --iterations 30 --output-dir /workspace/SpatialRust-python-delivery/target/public-icp-iterations-30

python3 /workspace/SpatialRust-python-delivery/scripts/report_icp_iterations.py \
 /workspace/SpatialRust-python-delivery/target/bounded-support-review/after \
 /workspace/SpatialRust-python-delivery/target/public-icp-iterations-30 \
 --output-dir /workspace/SpatialRust-python-delivery/target/public-icp-iteration-comparison
```

The comparison requires identical input hashes, initial poses and settings
except iteration cap, plus successful equivalence/roundtrip receipts. It saves
all candidate records in JSON and an offline HTML support table with bars.
These runs use the same bounded-ICP/bounded-support implementation, .1 m search
gate, .02 m evaluation gate and .05 m voxel leaf. Caps apply to each stage,
not to total iterations or wall-clock time.

| Candidate | Cap 5 forward support / RMSE | Cap 30 forward support / RMSE |
| --- | --- | --- |
| Identity | 9.64% / 11.715 mm | 12.47% / 12.560 mm |
| Projected tutorial prior | 61.58% / 7.831 mm | 62.28% / 7.453 mm |
| Perturbed tutorial prior | 36.81% / 9.431 mm | 62.28% / 7.451 mm |

No candidate's full-resolution stage converged under either cap. The selected
index changes from 1 to 2: at cap 30 both supplied nearby priors support exactly
the same number of source points, and the perturbed prior wins by slightly
lower gated RMSE. Selected reverse support changes from 93.24% to 93.87%.
Candidate-search wall time changes from 8.31 to 34.71 seconds, single observations
on a shared host rather than repeated performance estimates.

More iterations substantially change the perturbed prior's proximity result,
while the identity candidate remains weak and its gated RMSE rises as supported
membership changes. Convergence flags alone miss these distinctions, and more
supported points are not automatically lower gated RMSE. There is no independent
true pose, so these observations do not certify pose improvement or establish
that still more iterations would recover it.

A concrete next experiment is separating coarse and fine correspondence gates:
the .1 m gate remains active during full-resolution refinement even though the
diagnostic gate is .02 m. Whether tightening only the fine gate helps requires
paired measurement; this observation alone does not identify a failure cause.
The full Python regression suite passed 216 tests, and a zero iteration cap was
manually verified to fail before archive IO or output creation.
