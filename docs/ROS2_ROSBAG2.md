# ROS 2 and rosbag2 input

`spatialrust-ros2` keeps native ROS 2 executors separate from the core
workspace. The `rosbag2-sqlite` feature adds a read-only SQLite source for
rosbag2 bags that store `sensor_msgs/msg/PointCloud2` messages serialized as
CDR.

The source selects one topic, orders messages by `(timestamp, id)`, decodes
XYZ `float32` fields plus an optional scalar `intensity` `float32` field, and
emits bounded `SpatialRecordChunk` leases. A single PointCloud2 message may be
split across multiple chunks. SQLite payload, decoded message storage, and
emitted columns are included in the same hard memory budget; no bag copy or
full-bag materialization is performed. The selected topic's schema is fixed at
open time as either XYZ or XYZI, and a later change in intensity presence fails
closed.

Native `rclrs` executors, custom ROS message definitions, compressed
`.db3.zstd` storage, and PointCloud2 attributes other than float32 intensity
are outside this adapter slice. The adapter rejects unsupported topic types,
serialization formats, and intensity layouts instead of silently dropping
them.

## Bounded conversion

The runnable example writes one selected topic to an open-ended LAS sink and
creates both a workflow receipt and a checksummed input/output manifest:

```bash
cargo run -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_to_las -- \
  /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3 \
  /lidar_front/points_raw \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-front.las \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-front.receipt.json \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-front.manifest.json
```

The path is explicit at the application boundary, so input and output can
remain on separate storage roots. Sensor data is not a repository fixture.
