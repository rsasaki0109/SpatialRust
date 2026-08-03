# One-command Replay Demo 145C — 2026-08-03

## Scope

The replay slice adds a renderer-independent `ReplayDemoState` and a
feature-gated `rosbag2_replay_demo` example. The example reads a canonical
rosbag2 SQLite input read-only, retains a small bounded front/rear prefix,
verifies deterministic replay order, and writes a portable JSON trace, static
HTML dashboard, and checksummed manifest. Replay readiness is intentionally
separate from calibrated mapping admission.

Implementation paths:

- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-viewer/src/replay.rs`
- `/home/sasaki/workspace/SpatialRust/crates/spatialrust-ros2/examples/rosbag2_replay_demo.rs`

## Command

```bash
cargo run -p spatialrust-ros2 --features rosbag2-sqlite \
  --example rosbag2_replay_demo -- \
  /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3 \
  --output-dir /media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo \
  --expected-input-sha256 b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8
```

The input and output paths are absolute. The output directory must not exist;
the example refuses to overwrite a prior run and performs an external-SSD
free-space preflight before opening the source.

## Evidence

Canonical input:

- path: `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- size: `713670656` bytes
- expected and observed SHA-256:
  `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`
- topics: `/lidar_front/points_raw` (768 messages) and
  `/lidar_rear/points_raw` (767 messages)

External outputs:

- JSON state: `/media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo/replay-demo.json`
- HTML dashboard: `/media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo/replay-demo.html`
- manifest: `/media/sasaki/aiueo/spatialrust-results/v1-3/145c-one-command-replay-demo/replay-demo.manifest.json`

The bounded episode retained four records and 115,972 points, matched two
front/rear bundles, and observed a maximum matched delta of 92,435,944 ns
within the 100,000,000 ns window. The second deterministic walk passed; the
state reports `replay_ready:true` and `mapping_admitted:false`. The timestamp
basis is `PointCloud2 header stamp; ros2-external domain; no clock calibration
applied`, and the dashboard lists missing clock calibration, unapplied
TF/frame composition, and the source-bound calibration requirement as
blockers. The manifest checked the input, JSON, and HTML entries (three files,
713,683,963 bytes total).

No sensor payload or generated dashboard artifact is committed to the
repository.

## Verification

```text
cargo fmt --all -- --check
git diff --check
cargo test -p spatialrust-viewer --features serde
cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_replay_demo
cargo clippy -p spatialrust-viewer --features serde --all-targets -- -D warnings
cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_replay_demo -- -D warnings
cargo test --workspace --all-features --no-run
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-features --lib -- -D warnings
```

All focused and workspace checks passed after generating the external receipt.
