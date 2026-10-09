# Apply bounded nearest-neighbor search to support evaluation

The shared helper behind `distance_gated_support` and
`DistanceSupportIndex.support` now calls `nearest_one_within` using the existing
squared support gate. The previous path found an unrestricted nearest point
into a reusable vector and then rejected it if outside the gate. The bounded
primitive preserves accepted-neighbor traversal/ties and f32 distances while
pruning regions that cannot contribute. F64 accumulation order, source/target
validation, index ownership and released-GIL behavior are unchanged.

All 216 Python tests passed with the rebuilt ONNX-enabled release extension,
including the brute-force support oracle, exact gate/tie cases, no-support
behavior, invalid inputs, target lifetime, concurrent queries and GIL release.
Strict ONNX Clippy passed. The bounded primitive's 5,000-condition differential
tests and search/registration tests passed in the previous slice; its code is
unchanged here.

The public ICP pair was rerun with identical data, supplied poses, five iterations
per stage and support/search settings:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/validate_public_icp_candidates.py \
 --output-dir /workspace/SpatialRust-python-delivery/target/bounded-support-review/after
```

Previous receipts/reports are retained as `before-receipt.json` and
`before-alignment.json` under `/workspace/SpatialRust-python-delivery/target/bounded-support-review/`.
Input hashes match. Entire alignment dictionaries match exactly after excluding
the two directory-dependent file labels, including selected index, every
candidate's pose/support, RMSE, stages and convergence. Aligned XYZ also preserve
an exact PCD roundtrip.

| Single local observation | Bounded ICP only | Bounded ICP and support |
| --- | --- | --- |
| Optimized candidate workflow | 30.75 s | 8.31 s |
| Three independent file alignments | 39.84 s | 8.91 s |

The original unbounded workflow observation was 105.40 seconds on this pair.
Thus both changes address runtime waste without changing recorded outputs;
the new support bound removes substantial remaining cost in this condition.
These are sequential single observations across builds on a shared host, not
randomized repeated benchmarks or portable speedup guarantees. No convergence
gain is claimed: all candidates still stop nonconverged at five iterations.
No independent true pose is available. Broader scenes, higher-iteration runs
and repeated timings remain needed before interpreting overall robustness.
