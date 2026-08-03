# TF / Calibration Observatory 145B — 2026-08-03

## Scope

The observatory adds a portable state for calibration artifact receipts, clock
diagnostics, source-bound TF edges, graph topology, composition status, and
fail-closed admission. It deliberately records evidence rather than solving or
applying calibration.

The implementation is in:

- /home/sasaki/workspace/SpatialRust/crates/spatialrust-viewer/src/observatory.rs
- /home/sasaki/workspace/SpatialRust/crates/spatialrust-ros2/examples/rosbag2_calibration_observatory.rs

## Evidence

The canonical run uses:

- input: /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3
- readiness: /media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-survey/canonical.calibration.readiness.json
- diagnostic TF inventory: /media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/canonical-wrong-source.tf-static.inventory.json

The external SSD output is:

- JSON state: /media/sasaki/aiueo/spatialrust-results/v1-3/145b-tf-calibration-observatory/calibration-observatory.json
- HTML dashboard: /media/sasaki/aiueo/spatialrust-results/v1-3/145b-tf-calibration-observatory/calibration-observatory.html
- output manifest: /media/sasaki/aiueo/spatialrust-results/v1-3/145b-tf-calibration-observatory/calibration-observatory.manifest.json

The observed input SHA-256 matches the expected canonical identity. The state
shows both calibration artifacts as not_registered, the clock model as not
applied, zero accepted TF edges, no composed graph, and
calibration_admitted:false. The diagnostic TF receipt's different expected
input SHA is preserved as a blocker; its frames never enter the graph.

## Verification

    cargo fmt --all -- --check
    cargo test -p spatialrust-viewer --features serde
    cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_calibration_observatory
    cargo clippy -p spatialrust-viewer --features serde --all-targets -- -D warnings
    cargo clippy -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_calibration_observatory -- -D warnings
