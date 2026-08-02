# rosbag2 SQLite PointCloud2 E2E — 2026-08-02

## Scope

This receipt validates the `spatialrust-ros2` read-only rosbag2 SQLite source
against an external Autoware bag. The source selected both
`sensor_msgs/msg/PointCloud2` CDR topics, decoded XYZ plus float32 intensity,
split each message into bounded records, and wrote open-ended LAS streams. No
sensor data was added to the repository.

## Command

```bash
cargo run --release -p spatialrust-ros2 --features rosbag2-sqlite \
  --example rosbag2_to_las -- \
  /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3 \
  /lidar_front/points_raw \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front-intensity.las \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front-intensity.receipt.json \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/front-intensity.manifest.json

cargo run --release -p spatialrust-ros2 --features rosbag2-sqlite \
  --example rosbag2_to_las -- \
  /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3 \
  /lidar_rear/points_raw \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/rear-intensity.las \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/rear-intensity.receipt.json \
  /media/sasaki/aiueo/spatialrust-results/rosbag2-e2e/rear-intensity.manifest.json
```

## Results

- Input: 1,535 messages, 713,670,656 bytes.
- Front output: 445,332,707 bytes, LAS 1.2 point format 0, 22,266,624 points
  in 1,536 bounded chunks.
- Rear output: 444,752,847 bytes, LAS 1.2 point format 0, 22,237,631 points
  in 1,534 bounded chunks.
- LAS intensity readback was nonzero for 22,258,867 front points and
  22,058,243 rear points, with the observed range 0–255.
- Maximum raw CDR payload: 464,036 bytes.
- Maximum declared source working set: 1,255,752 bytes.
- Peak tracked memory: 993,608 bytes.
- Both manifests' SHA-256 and size entries matched the input and their LAS
  outputs. `laspy` readback reported the expected point counts, finite XYZ
  bounds, and LAS point format 0.

The input bag, LAS output, receipt, and manifest remain outside Git on the
external SSD. The committed implementation intentionally does not support
compressed `.db3.zstd` storage, custom ROS message definitions, or PointCloud2
attributes other than float32 intensity yet.
