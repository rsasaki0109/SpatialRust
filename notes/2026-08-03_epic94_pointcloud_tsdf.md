# Epic 94 PointCloud-to-TSDF bridge — 2026-08-03

## Scope

`spatialrust-scene::TsdfVolume` now accepts a `PointCloud` directly through
`integrate_cloud` and `integrate_cloud_with_pose`. Both methods read the
`HasPositions3` capability columns, use the caller-provided sensor origin, and
do not materialize an interleaved XYZ vector. The pose variant applies an
explicit sensor-to-volume isometry to positions and origin in the integration
loop. Unsupported position dtypes and schema errors cross the existing
`SceneError` boundary; invalid metric samples retain the established ignore
behavior.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo fmt --all
cargo test -p spatialrust-scene
cargo clippy -p spatialrust-scene --all-targets -- -D warnings
```

The direct-column test compares the resulting TSDF volume via derived value
equality against the existing interleaved `integrate_xyz` path on the same
two-point cloud, and a second test exercises explicit sensor-pose admission.
