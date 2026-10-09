# Candidate workflow sharing: measure end-to-end effects

Reproduce with the installed SpatialRust extension and NumPy:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/benchmark_pose_candidates.py \
 --sizes 1000 5000 20000 --repeats 15
```

Artifacts are overwritten under
`/workspace/SpatialRust-python-delivery/target/candidate-performance/`:
`timings.json` records every sample, settings, Python/platform and source hashes;
`report.html` displays median times, observed ranges and offline timing bars.

All modes execute the same candidate poses, source/target files, two-stage ICP
and support diagnostics. `read_each` reads both files per candidate; `read_once`
reads once but voxelizes the target each time; `shared_target_voxel` additionally
shares that voxel cloud. Every candidate's entire diagnostic dictionary is
checked for exact equality against independent file alignment, outside timing.
Warmup runs precede measurement and mode order rotates between repetitions.

Local final-run medians in milliseconds:

| Points | Candidates | Repeated reads | Read once | Shared target voxel |
| --- | --- | --- | --- | --- |
| 1,000 | 1 | 2.59 | 2.50 | 2.51 |
| 1,000 | 8 | 20.10 | 18.72 | 18.61 |
| 5,000 | 1 | 15.45 | 15.45 | 15.33 |
| 5,000 | 8 | 127.09 | 123.58 | 124.36 |
| 20,000 | 1 | 74.83 | 74.15 | 72.06 |
| 20,000 | 8 | 545.23 | 548.38 | 547.13 |

Eight candidates on 1,000 points took about 7.4% less time with both forms of
sharing, but on 20,000 points the final modes were effectively tied. An earlier
15-repeat run found 560.46 vs 541.19 ms for that largest case, illustrating
run-to-run variation. Target voxel sharing alone does not show a consistent
additional improvement. A one-candidate control has no repeated work to save;
its small timing differences also illustrate measurement noise.

These timings include reads, validation, transforms, registration and support;
they exclude file generation, selection bookkeeping, output serialization and
HTML. Geometry is uniform synthetic XYZ with small translation and nearby
translation priors; all candidates succeed. Filesystem caches are warm and the
host is shared. There are no cold-read, real-scene, failed-candidate, memory or
cross-library measurements, confidence intervals or independent scene seeds.
The observations suggest profiling registration/support next, but do not prove
which subroutine dominates without component timing. This is evidence against
claiming universal acceleration from removing redundant preprocessing.
