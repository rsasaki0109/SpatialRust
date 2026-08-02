# rosbag2 batch ingest receipt — 2026-08-03

## Scope

The `rosbag2_ingest` example adds a metadata-only topic inventory and bounded
batch conversion for CDR `sensor_msgs/msg/PointCloud2` topics. Unsupported
topics are recorded as skipped during an all-topic ingest. An explicitly
requested unsupported topic is recorded as failed and makes the command exit
non-zero.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_ingest
cargo test -p spatialrust-ros2 --features rosbag2-sqlite
cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --examples -- -D warnings
```

All tests and clippy checks passed.

The external input was:

`/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`

The inventory at:

`/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-inventory.json`

found `/lidar_front/points_raw` with 768 messages and
`/lidar_rear/points_raw` with 767 messages. The batch output directory was:

`/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/batch-ingest`

Results:

- 2 topics converted, 0 skipped, 0 failed.
- Front: 1,536 chunks, 22,266,624 points, XYZI schema, peak tracked bytes 993,608.
- Rear: 1,534 chunks, 22,237,631 points, XYZI schema, peak tracked bytes 993,608.
- Front LAS SHA-256: `0c7fff23b82bdd94b5a9af2ad858190269509df3973b5ed95f2b99fa00121700`.
- Rear LAS SHA-256: `b77be6f9fa959a877f3c0a2738d18d72855259c50dce97949738d4e96c4c7189`.
- Input SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c`.

The batch LAS hashes match the earlier single-topic intensity-preserving
conversion, confirming deterministic output for this dataset.
