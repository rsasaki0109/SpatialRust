# Source-bound rosbag2 TF inventory

## Scope

143B now has a bounded CDR decoder for `tf2_msgs/msg/TFMessage`, a read-only
SQLite accessor, and a source-bound inventory receipt. The implementation keeps
TF edges as decoded data; it does not compose transforms, infer a root, or
claim calibration for another input.

## Implementation

- `spatialrust-runtime::decode_tf_message` validates little-endian CDR,
  timestamps, UTF-8/non-empty frame identifiers, nested float64 alignment, and
  trailing bytes.
- `spatialrust-ros2::list_tf_messages` validates the topic type and CDR
  serialization, orders rows by `(timestamp, id)`, and applies a message bound.
- `rosbag2_tf_inventory` requires absolute paths and the expected input
  SHA-256, writes an atomic external receipt, and fails closed on identity,
  topic, decode, truncation, or required-frame blockers.

## Evidence

The positive fixture is
`/media/sasaki/aiueo/datasets/migrated/autoware_data/all-sensors-bag1/all-sensors-bag1_compressed_0.db3`
with size `2506907648` bytes and SHA-256
`74e5915719a7b7b4820b5339207eeade0c656deaa38b8e5b5e8d18787a58ac22`.
The receipt is
`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/all-sensors-bag1.tf-static.inventory.json`.
It contains one `/tf_static` message, 14 transforms, and all required
`sensor_kit_base_link`, `velodyne_front`, `velodyne_left`, and `velodyne_right`
frames with `passed: true`.

The negative source-binding check is
`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/canonical-wrong-source.tf-static.inventory.json`.
It uses the canonical input
`/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`,
observes SHA-256
`b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`, and
correctly exits non-zero for the mismatched expected fixture SHA. The separate
2022 fixture is diagnostic parser evidence only; its frames must not be
registered as calibration for the canonical 2020 bag.

## Validation

- `cargo test -p spatialrust-runtime --features ros2`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --all-targets`
- Positive and negative external-storage CLI runs above
