# Reuse the nearest-neighbor output buffer in support evaluation

The native Python support function previously called `KdTree::nearest_one` for
every query point. That method delegates to `nearest_k`, allocating a vector
per call. It now calls the existing `nearest_k_into(..., 1, &mut buffer)` with
one reusable vector per support evaluation. The search algorithm, ordering,
f32 distance gate, f64 RMSE accumulation and released-GIL behavior stay the same.
No new public API or shared mutable state is introduced. Each concurrent query
owns its own output buffer. Reference tree construction remains per call.

The full Python suite passed 186 tests after rebuilding the ONNX-enabled release
extension; strict ONNX-enabled Clippy also passed. A new brute-force oracle test
checks random source/reference clouds, exact gate inclusion, duplicate/tied
references, outside-gate queries and a following exact match. Existing support
tests cover invalid inputs, asymmetric support and simultaneous read-only calls.

Before and after separate release builds, run:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/benchmark_support_query.py \
 --output /workspace/SpatialRust-python-delivery/target/support-query-buffer/before.json
```

After rebuilding, use `after.json` as the output. Each condition has 15 samples
of ten sequential support calls on the same deterministic translated uniform
XYZ cloud. JSON records every per-query sample, platform and native-extension
hash. The hashes differ across builds, while count, fraction and RMSE agree
exactly for both point counts.

| Points | Before median per query | After | Reduction |
| --- | --- | --- | --- |
| 5,000 | 1.962 ms | 1.685 ms | 14.1% |
| 20,000 | 8.962 ms | 8.685 ms | 3.1% |

These wall times include reference-tree construction and source/target finite
validation. Measurements are sequential before/after builds on a shared host,
not randomized build alternation; host variance can affect small differences.
One synthetic scene seed, warmed execution, XYZ-only data and one support gate
do not establish universal gains. The result does not quantify whole candidate
search acceleration. Further tree reuse needs an ownership-safe API and separate
measurement; this change only removes repeated output-vector allocations.
