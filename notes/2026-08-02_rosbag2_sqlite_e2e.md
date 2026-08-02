# rosbag2 SQLite PointCloud2 E2E — 2026-08-02

## Scope

This receipt validates the first `spatialrust-ros2` read-only rosbag2 SQLite
source against an external Autoware bag. The source selected one
`sensor_msgs/msg/PointCloud2` CDR topic, decoded XYZ `float32` fields, split each
message into bounded records, and wrote an open-ended LAS stream. No sensor
data was added to the repository.

## Command

```bash
cargo run --release -p spatialrust-ros2 --features rosbag2-sqlite \
  --example rosbag2_to_las -- \
  /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3 \
  /lidar_front/points_raw \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front.las \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front.receipt.json \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front.manifest.json
```

## Results

- Input: 768 messages, 713,670,656 bytes.
- Output: 445,332,707 bytes, LAS 1.2 point format 0.
- Decoded output: 22,266,624 points in 1,536 bounded chunks.
- Maximum raw CDR payload: 464,036 bytes.
- Maximum declared source working set: 1,190,216 bytes.
- Peak tracked memory: 993,608 bytes.
- `/media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front.manifest.json`
  SHA-256 and size entries matched both files.
- `laspy` readback reported 22,266,624 points and finite XYZ bounds.

The input bag, LAS output, receipt, and manifest remain outside Git on the
external SSD. The committed implementation intentionally does not support
compressed `.db3.zstd` storage or custom ROS message definitions yet.
