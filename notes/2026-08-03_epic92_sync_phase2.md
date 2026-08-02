# Epic 92 sync phase 2 receipt — 2026-08-03

## Scope

The `spatialrust-sync` boundary now has an explicit bounded episode collector
and record-level frame transformation:

- `MemoryEpisodeBuilder` rejects a next record before retention when record,
  point, or allocated column-storage limits would be exceeded.
- `FrameGraph::transform_record_to` resolves a calibrated rigid path, transforms
  positions, complete normal triplets, and `sensor_origin`, updates the target
  frame, and preserves every other field plus `RecordProvenance`.
- `rosbag2_sync_preview` retains only a caller-bounded front/rear prefix,
  indexes it deterministically, and reports nearest-topic match counts without
  writing sensor payloads.

The preview uses the PointCloud2 header timestamp as an external ROS clock for
the bounded comparison. This is an explicit analysis assumption, not a clock
calibration result, and the preview does not invent a front/rear extrinsic
transform.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo fmt --all
cargo test -p spatialrust-sync --all-features
cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_sync_preview
```

The external preview input was:

`/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`

Its count-only receipt was written outside the repository to:

`/media/sasaki/aiueo/spatialrust-results/rosbag2-sync-preview.receipt.json`
