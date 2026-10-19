# Full-scene Open3D RGB-D comparison

Operational slice 146K. Commit this contract and both
`tum-fr1-{xyz,rpy}-open3d-dense-plan.json` plans before any new comparator
estimation. Native causal controls remain frozen at 798/798 xyz and 722/722
rpy poses. XYZ passes; rpy ATE 0.114793m exceeds the existing 0.10m limit even
though its 1s RPE 0.048145m passes. This study seeks a full external comparison
before attributing the gap to a particular preprocessing difference.

## Data and execution contract

Use installed Open3D 0.19.0 CPU module SHA-256
`ab1b2f4a1126a11886a2a9f3eeab5ceaeac0bab1b53ee319eb363df7ef629d88`.
The existing 151,434,752-byte module stays in place; download or copy no package.
Keep the frozen SpatialRust numerical extension for shared input projection,
grayscale, geometric support and typed IO. Add no Rust, core, GPU or production
dependency, native rebuild, implicit device transfer or solver change.

The comparator adapter owns one float64 proper 4x4 previous source-to-target
increment (128 bytes) and a successful-pair count. First seed is identity; later
seeds are the exact preceding returned increment. Pass an owned seed and owned
float32 grayscale/metric-depth CPU images to the legacy CPU hybrid API. Retain
no image or unbounded pose history. Record the seed, preceding pair index,
proper returned/inverse transforms, package/model identity and diagnostic 6x6
information matrix. Advance state only after successful finite/proper output
and unchanged raw final projective validation. A later generator rejection
permanently loses the chain and makes no further estimator call.

Use 20/10/5 iterations, 0.03m depth difference, 0.1..5m depth and the same
525/525/319.5/239.5 registered-depth intrinsics. Original grayscale and depth
units are unchanged. The pinned implementation internally filters full-resolution
gray/depth with Gaussian3 and normalizes gray means over initial correspondences
to 0.5 before its own pyramid. Sampling, derivatives and stopping differ from
the native bilinear estimator. This compares implementations; it does not claim
equivalence or establish that one preprocessing step caused the observed drift.
See the [pinned source](https://raw.githubusercontent.com/isl-org/Open3D/v0.19.0/cpp/open3d/pipelines/odometry/Odometry.cpp)
and [versioned API](https://www.open3d.org/docs/0.19.0/python_api/open3d.pipelines.odometry.compute_rgbd_odometry.html).

## Fixed full-scene gates

Reuse the already checked PNG/index payloads under
`E:/SpatialRustData/maturity-80/tum-fr1-{xyz,rpy}/prepared-causal-dense-v1/`
through checked hard links to new prepared directories. Never fall back to image
copies. Preserve original archives, manifests, all native/GPU binaries, wheels,
failed/successful estimates, metrics and audits. Read no reference pose during
preparation or estimation; freeze both complete timelines before either score.

Process all 798 xyz and 722 rpy rows with no retry, reset, dropped failure,
future-frame selection, scale fitting or reference initializer. Use unchanged
common raw full-resolution final checks: at least 30 correspondences and 0.5
fraction, with independent 0.5 full-cloud support at 0.1m. Keep camera/world
composition, timestamp association and typed IO exactly as the native controls.
Producer validation uses bounded NumPy row tiles; a separate auditor recomputes
full-frame projective counts/residuals and seed/world lineage. These raw metrics
are common admission measurements, not Open3D's internal filtered objective.

Keep 100% pose generation, >=95% reference coverage, first-pose SE3-aligned ATE
RMSE <=0.10m and one-second translation RPE RMSE <=0.05m. Retain three xyz/six
rpy reference gaps and all unmatched RPE pairs. Audit statistics, transformations
and scores at the existing tolerances. An estimator success flag or surviving
prefix does not satisfy these unchanged full-scene gates. Never use the returned
information matrix as sensor calibration or clock uncertainty evidence.

## Resource and scope boundary

Write new results only under the E drive, using an
`open3d-dense-v1` suffix and study root
`E:/SpatialRustData/maturity-80/open3d-dense-study-v1/`. Keep at least 20GiB free,
cap combined new study output at 1GiB and each owned command at one hour. Use
at most four CPU workers through scoped thread controls. Freeze package/native,
code, plan, input and output identities. Open3D internal allocator/pyramid memory
is not a measured or equivalent native working-memory bound. No speed claim.

Only exact owned passing pytest scratch can be removed after successful XML,
absolute-parent and full no-reparse checks. Preserve all scientific outcomes and
logs, including failures. Both datasets are observed development data. Canonical
measured clock/physical calibration, both raw Linux hardware classes and full
mapping/semantic/release acceptance remain separate open 80% requirements.
