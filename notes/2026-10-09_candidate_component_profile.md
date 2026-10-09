# Candidate timing components identify the next optimization

Run with the installed SpatialRust extension:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/profile_pose_candidates.py
```

JSON samples and standalone stacked-bar HTML are saved under
`/workspace/SpatialRust-python-delivery/target/candidate-component-profile/`.
The script instruments Python-to-native call boundaries in the existing paired
benchmark workload: 1,000 or 20,000 uniform points, eight nearby translation
priors, nine repetitions per mode, rotating mode order and untimed warmups.
Every diagnostic dictionary matches an uninstrumented independent-file run.
Assertions check read, voxel, two-stage ICP and four support-query call counts.

Each displayed stack is one actual median-total sample, so its components add
to its total. These are elapsed wall times, including wrapper overhead, rather
than native CPU samples. Functions do not nest through these Python wrappers.
The remainder includes XYZ exports, NumPy validation, pose composition and
Python report assembly. No selection bookkeeping or output writes are timed.

For the fully shared target-voxel mode, the local observations were:

| Component | 1,000 points (20.69 ms total) | 20,000 points (571.35 ms total) |
| --- | --- | --- |
| Reads | 0.6% | 0.5% |
| Transforms | 1.2% | 0.6% |
| Voxelization | 1.6% | 1.1% |
| Coarse ICP | 28.6% | 16.1% |
| Full-resolution ICP | 24.5% | 28.2% |
| Before support | 10.7% | 13.5% |
| Initial support | 9.3% | 14.3% |
| Final forward support | 7.3% | 11.4% |
| Final reverse support | 7.9% | 12.8% |
| Remainder | 8.2% | 1.5% |

Support queries together account for about 52% on the larger synthetic cloud,
versus about 44% for both ICP stages. This explains why removing repeated
voxelization has little scope to accelerate this particular workload. The
unchanged before-support query is repeated eight times; sharing it is a
concrete next candidate, while the other support queries depend on each pose.
Reusable reference indexes may also help, but these boundary timings do not
separate tree construction from queries and cannot yet establish that benefit.

Call categorization assumes the existing successful two-stage pipeline and its
four support calls in their documented order. It must be updated if that pipeline
changes; assertions detect count drift. All candidates succeed in this study.
One scene seed, warm caches, shared-host variance and instrumentation overhead
limit generalization. Results are not cross-library benchmarks or real-scene
performance claims; the uninstrumented benchmark remains the total-time check.
