# Share unchanged before-support across pose candidates

The candidate example now computes the original source-to-target support query
once per search. Its source, target and evaluation distance are unchanged by
candidate poses; initial, final forward and final reverse support remain per
candidate. The private cache is created inside each search and populated only
after a candidate completes registration. Reports receive separate dictionary
copies. Failed registrations neither consume nor corrupt cached support.

Tests cover independent-run diagnostic equality, stable selection with a failed
first candidate, seven support calls for two successful candidates instead of
eight, report mutation isolation and separate searches with different gates.
The full Python suite passed: 185 tests.

Reproduce the timing comparison:

```bash
LD_LIBRARY_PATH=/workspace/SpatialRust-python-delivery/target/onnxruntime \
 /workspace/.spatialrust-env/comparison-venv/bin/python \
 /workspace/SpatialRust-python-delivery/scripts/benchmark_pose_candidates.py \
 --sizes 1000 20000 --candidates 1 8 --repeats 9 \
 --output-dir /workspace/SpatialRust-python-delivery/target/shared-before-support-performance
```

The fourth mode, `shared_before_support`, adds this optimization to existing
read-once and shared-target-voxel behavior. All complete candidate diagnostics
match independent file runs exactly. Every timing is retained in JSON, with
source hashes and settings; the standalone HTML shows medians and ranges.

Local median milliseconds:

| Points | Candidates | Repeated reads | Read once | Shared voxel (previous) | Shared before support (new) |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 1 | 2.48 | 2.44 | 2.57 | 2.50 |
| 1,000 | 8 | 19.11 | 17.87 | 16.78 | 16.46 |
| 20,000 | 1 | 72.56 | 67.82 | 70.85 | 70.86 |
| 20,000 | 8 | 557.48 | 538.46 | 537.86 | 470.92 |

The larger eight-candidate case improved about 12.4% relative to the previous
implementation, and 15.5% relative to repeated reads. One candidate does not
benefit from sharing this query, consistent with the control timings. The
smaller-case incremental difference is small and may reflect host variance.
These are warm-cache synthetic XYZ results on a shared host, not real-scene
or cross-library guarantees; selection bookkeeping and output writes are
excluded. Timing equality checks occur outside measured intervals.

The component profiler was also updated for four modes and variable before
query counts. Its three-repeat, 20,000-point validation artifacts under
`/workspace/SpatialRust-python-delivery/target/shared-before-support-profile/`
verify one before query versus eight previously and unchanged eight calls for
each other support phase. All instrumented diagnostics match the reference.
Historical component-profile notes describe the previous four-query-per-pose
pipeline; the updated profiler models the shared before query explicitly.
