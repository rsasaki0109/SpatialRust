# Reuse the owned target support index in candidate alignment

Candidate searches now lazily create one `DistanceSupportIndex` for the fixed
target and reuse it for before, initial and final forward support. Target XYZ
are explicitly copied into the owned immutable CPU tree. The cache belongs to
one search; changes to files or evaluation distance cannot leak between searches.
Reverse support still uses each aligned source as its own reference and rebuilds
its index. Independent single-file alignment remains the equivalence reference.

The full Python suite passed 200 tests. Candidate tests check full selected-report
equality, per-candidate transforms/support/convergence, stable selection, original
input preservation and one index construction even with a failed middle candidate.
Separate gates and report-mutation isolation remain covered.

Reproduce the five-mode paired comparison:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/benchmark_pose_candidates.py \
 --sizes 1000 20000 --candidates 1 8 --repeats 9 \
 --output-dir /workspace/SpatialRust-python-delivery/target/candidate-support-index-performance
```

Raw samples, source hashes, settings and offline HTML bars are saved there.
Each mode is warmed up, execution order rotates between repetitions and every
complete candidate diagnostic dictionary matches independent file alignment.
Both tree construction and queries are inside the measurement.

Local median milliseconds:

| Points | Candidates | Repeated reads | Read once | Shared voxel | Shared before support (previous) | Shared index (new) |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 1 | 2.39 | 2.30 | 2.32 | 2.36 | 2.22 |
| 1,000 | 8 | 20.15 | 18.55 | 18.27 | 17.09 | 16.16 |
| 20,000 | 1 | 65.04 | 65.12 | 68.37 | 68.35 | 57.91 |
| 20,000 | 8 | 519.68 | 509.02 | 504.02 | 440.13 | 394.64 |

The larger eight-candidate condition improved about 10.3% relative to the previous
pipeline, and 24.1% relative to repeated reads. A single candidate also reuses
the target tree across its three forward-support queries. These timings cover
candidate alignment and diagnostics, excluding selection bookkeeping and output
writes. Synthetic uniform geometry, one scene seed, warm caches and shared-host
variance limit generalization; real-scene and memory profiling remain outstanding.

The component profiler now separately times index construction and indexed
queries, maintaining disjoint boundary timings and call-count assertions.
Three-repeat 20,000-point validation artifacts are under
`/workspace/SpatialRust-python-delivery/target/candidate-support-index-profile/`.
Indexed mode must build exactly one owned tree and run one before, eight initial,
eight final and eight reverse support calls, with unchanged two-stage ICP counts.
