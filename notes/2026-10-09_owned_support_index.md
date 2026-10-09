# Owned immutable reference index for repeated support queries

`DistanceSupportIndex(target)` explicitly copies reference XYZ into its CPU
KD-tree. `index.support(source, max_distance)` shares the existing support
implementation and returns the same count, fraction and optional gated RMSE.
The object is frozen; each call owns its temporary query buffer. Construction
and query loops release the GIL. No reference to Python-owned target memory is
retained, so releasing the target does not invalidate the index. A different
reference requires a new index; this is explicit allocation, not a hidden cache.

The existing standalone function remains available and uses the same helper.
Tests check result equality, target lifetime independence, concurrent queries,
duplicate references, exact gate inclusion, empty/nonfinite XYZ, invalid gates
and actual GIL release during construction and query. The full Python suite
passed 199 tests; runtime/stub conformance and strict ONNX Clippy also passed.

Reproduce direct and reused-index timings with the same installed build:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/benchmark_support_query.py \
 --output /workspace/SpatialRust-python-delivery/target/support-index/direct.json

LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/benchmark_support_query.py \
 --reuse-index --output /workspace/SpatialRust-python-delivery/target/support-index/reused.json
```

Each timed sample performs ten support queries. Reused mode constructs one
index inside each timed batch, including construction in the reported amortized
per-query time. Fifteen samples per point count retain raw wall times, reference
result and native-extension hash. All query results match exactly.

| Points | Rebuild tree every query | Reuse across ten queries | Reduction |
| --- | --- | --- | --- |
| 5,000 | 1.680 ms/query | 1.109 ms/query | 34.0% |
| 20,000 | 8.541 ms/query | 5.936 ms/query | 30.5% |

These are sequential mode measurements on one synthetic translated uniform
cloud per size, warm execution and a shared host. They do not include independent
scene seeds, randomized mode order, memory profiling or real-scene comparisons.
The index is not yet integrated into candidate alignment; these measurements
establish local repeated-query benefit, not end-to-end candidate acceleration.
Forward queries share a fixed target; reverse support has a different reference
for each aligned candidate and needs separate treatment.
