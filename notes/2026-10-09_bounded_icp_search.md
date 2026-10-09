# Bound ICP nearest-neighbor search by the correspondence distance

`KdTree::nearest_one_within(x, y, z, max_distance_squared)` finds the nearest
point within an inclusive squared-distance bound without a temporary vector.
It visits near branches first, prunes far branches by the bound until finding a
point, then uses the existing strict distance pruning and first-encountered tie
policy. Boundary equality is visited when no accepted point has been found.
Negative/NaN bounds or nonfinite query coordinates return None; zero accepts
exact matches and positive infinity permits unbounded search.

Point-to-point ICP uses this method for both correspondence collection and
updated-transform fitness evaluation. Previously both found an unrestricted
nearest neighbor before applying the same distance gate. This leaves the
accepted neighbor order and distance arithmetic unchanged while removing work
in regions that cannot supply an accepted correspondence. Other registration
algorithms and Python support-index queries are unchanged by this slice.

Validation: 23 Rust tests passed across search and registration, 216 Python tests
passed with the rebuilt ONNX-enabled release extension, and strict search and
registration Clippy passed. New tests compare full Neighbor values, including
indices, against unbounded search followed by gating for 1,000 deterministic
queries times five bounds. Separate checks cover ties, duplicates, exact gates,
zero, infinity, empty indexes and invalid queries. Code uses APIs compatible
with the declared Rust 1.75 minimum; this run used the installed newer toolchain.

The public ICP sample validation was rerun with the same two files, three
supplied poses, five iterations per stage and evaluation settings:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/validate_public_icp_candidates.py \
 --output-dir /workspace/SpatialRust-python-delivery/target/bounded-icp-review/after
```

The baseline at `ce35c64` is retained as `before-alignment.json` and
`before-receipt.json` under `/workspace/SpatialRust-python-delivery/target/bounded-icp-review/`.
Input-file hashes match. Entire alignment dictionaries match exactly after
excluding the two artifact-directory-dependent file labels, including all
candidate transforms/support, selected index, stage results and convergence.
Aligned XYZ also round-trip exactly through PCD in the new run.

| Single local observation | Before | After |
| --- | --- | --- |
| Optimized candidate workflow | 105.40 s | 30.75 s |
| Three independent file alignments | 115.33 s | 39.84 s |

The candidate-workflow observation is about 3.4 times faster, but this is one
public pair and sequential runs across release builds, not randomized repeated
benchmarking. Short tests and compilation checks overlapped part of the new
run; shared-host load adds uncertainty. No convergence or pose accuracy gain is
claimed: all candidates remain nonconverged within the five-iteration cap and
no independent true pose is available. The unchanged final diagnostics are
evidence of equivalent behavior on this pair, not universal numerical identity.

This validates a concrete reason geometry/gates affect runtime: the old search
still solved nearest-neighbor problems for points that the gate would reject.
The new bound can prune those searches early. Support queries still use
unrestricted nearest-neighbor search followed by a support gate, making them a
separate next optimization candidate; real component timings remain needed.
