# 145H ROS 2 Live Publish Bridge

Date: 2026-08-03

## Scope

This slice adds a portable, source-bound ROS 2 PointCloud2 publish bridge. The
viewer state records explicit source-to-publish topic names, topic-specific
frame identity, deterministic packet order, CDR payload sizes, queue policy,
round-trip equality, and host/device transfer counters. `publish_ready` is
separate from `mapping_admitted`: the CPU loopback adapter can prove transport
readiness without claiming calibrated-world geometry.

`rosbag2_live_publish` uses the existing read-only rosbag2 SQLite source,
bounded record episode, PointCloud2 CDR encoder/decoder, and
`in-process-loopback` adapter. Native `rclrs` execution remains a separate
integration boundary. No clock correction, TF composition, frame transform, or
front/rear fusion is applied.

## External evidence

Canonical input:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

Positive run:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145h-live-publish-v2/live-publish.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145h-live-publish-v2/live-publish.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145h-live-publish-v2/live-publish.manifest.json`

Observed positive state:

- `publish_ready=true`, `mapping_admitted=false`
- four packets and four exact CDR loopback round trips
- 115,972 points, 1,856,132 host encode bytes, 1,856,132 host decode bytes
- front `lidar_front` and rear `lidar_rear` topic/frame identities preserved
- zero device upload/readback bytes and zero backpressure events

Negative source-binding probe:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145h-live-publish-validation-probe/live-publish.json`
- wrong expected SHA withheld all packet generation
- `publish_ready=false`, `mapping_admitted=false`, CLI exit status 2

## Verification

Focused checks passed:

- `cargo fmt --all -- --check`
- `cargo test -p spatialrust-viewer --features serde`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_live_publish`
- `cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_live_publish -- -D warnings`
- positive and wrong-source external runs with manifest re-hashing

The bridge therefore proves bounded CPU transport and exact CDR round-trip
behavior while keeping calibration and mapping admission fail-closed.
