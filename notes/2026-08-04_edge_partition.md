# 145I Edge Partition Execution Receipt

Date: 2026-08-04

## Scope

This slice connects the source-bound 145H live-publish receipt to an explicit
edge-to-host execution contract. `EdgePartitionState` is portable JSON/HTML
state: it records named partitions, packet transfers, queue signals, explicit
copy bytes, deterministic order, upstream publish readiness, and separate
calibration/mapping admission.

`rosbag2_edge_partition` consumes `live-publish.json` rather than opening
SQLite again. It builds a deterministic `PartitionGraph` with `edge -> host`,
validates a `TransferPlan`, and uses `BoundedTransferQueue` with a two-slot
capacity. Packets are admitted in two-packet batches, producing observable
soft-watermark receipts without exceeding the hard limit. No implicit network
transport, GPU transfer, frame transform, or sensor fusion is claimed.

## External evidence

Canonical source identity is inherited from:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

Positive run:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145i-edge-partition-v2/edge-partition.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145i-edge-partition-v2/edge-partition.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145i-edge-partition-v2/edge-partition.manifest.json`

Observed positive state:

- `partition_ready=true`, `mapping_admitted=false`
- four source packets, four completed transfers, deterministic order verified
- queue depth 2, soft-limit trips 2, hard rejects 0
- explicit-copy bytes 1,856,132

Negative source-binding probe:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145i-edge-partition-validation-probe-v2/edge-partition.json`
- wrong expected SHA withheld all transfers
- `partition_ready=false`, `mapping_admitted=false`, CLI exit status 2

## Verification

Focused and workspace checks passed:

- `cargo fmt --all -- --check`
- `cargo test -p spatialrust-viewer --features serde edge_partition`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_edge_partition`
- `cargo clippy -p spatialrust-viewer --features serde --all-targets -- -D warnings`
- `cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_edge_partition -- -D warnings`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --workspace --all-features --lib -- -D warnings`
- `cargo test --workspace --all-features --no-run`
