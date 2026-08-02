# SpatialRecord provenance receipt — 2026-08-03

## Scope

`spatialrust-records` now carries a validated, protocol-independent
`RecordProvenance` envelope alongside the versioned schema and point cloud.
It records a source identity, optional source URI, logical stream, and source
sequence. The contract remains outside `spatialrust-core` and has no ROS,
SQLite, Arrow, or model-runtime dependency.

Provenance is preserved by schema migration, `MemoryChunkSource`/`Sink`,
bounded crop/transform, and external voxel aggregation. Voxel aggregation
retains source identity/URI/stream but clears a single sequence because one
output may combine multiple source chunks. The rosbag2 source attaches the
input DB3 path, topic, and deterministic chunk sequence.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo test -p spatialrust-records --all-features
cargo test -p spatialrust-ros2 --features rosbag2-sqlite
cargo test -p spatialrust-pipeline --features pipeline-streaming
cargo clippy -p spatialrust-records --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-pipeline --features pipeline-streaming --all-targets -- -D warnings
cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --all-targets -- -D warnings
```

All checks passed.

The external rosbag2 batch run used:

`/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`

and wrote only to:

`/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-provenance`

Results were 2 converted PointCloud2 topics, 11 skipped unsupported topics,
and 0 failures. Front/rear retained the XYZI schema and produced the same
LAS SHA-256 values as the previous batch run:

- Front: `0c7fff23b82bdd94b5a9af2ad858190269509df3973b5ed95f2b99fa00121700`.
- Rear: `b77be6f9fa959a877f3c0a2738d18d72855259c50dce97949738d4e96c4c7189`.
