# Spatial Studio 145A — 2026-08-03

## Scope

The first Spatial Studio vertical slice is a portable, renderer-independent
state plus a zero-dependency static dashboard. It combines source identity,
receipt-backed point-cloud/derived layers, a rosbag2 timeline, calibration and
TF admission panels, and explicit pipeline memory/transfer metrics without
adding ROS or SQLite dependencies to spatialrust-viewer.

The implementation is in:

- /home/sasaki/workspace/SpatialRust/crates/spatialrust-viewer/src/studio.rs
- /home/sasaki/workspace/SpatialRust/crates/spatialrust-ros2/examples/rosbag2_studio.rs

## Evidence

The demo was built from the canonical input:

- /media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3
- expected and observed SHA-256: b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8
- readiness receipt: /media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-survey/canonical.calibration.readiness.json
- diagnostic TF receipt: /media/sasaki/aiueo/spatialrust-results/v1-3/143b-tf-inventory/canonical-wrong-source.tf-static.inventory.json
- source-bound E2E/performance receipt: /media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh-3/rosbag2.e2e.receipt.json
- E2E input manifest: /media/sasaki/aiueo/spatialrust-results/v1-3/144-performance-fresh-3/rosbag2.e2e.manifest.json

The generated evidence remains on the external SSD:

- JSON state: /media/sasaki/aiueo/spatialrust-results/v1-3/145a-spatial-studio-v2/spatial-studio.json
- HTML dashboard: /media/sasaki/aiueo/spatialrust-results/v1-3/145a-spatial-studio-v2/spatial-studio.html
- output manifest: /media/sasaki/aiueo/spatialrust-results/v1-3/145a-spatial-studio-v2/spatial-studio.manifest.json

The dashboard records four renderable receipt-backed layers, 1,535 source
samples, the header-stamp time basis with no clock calibration, a 66.7-second
pipeline receipt, and zero hidden device copies. It explicitly shows
mapping_admitted:false: the canonical clock/frame artifacts are
not_registered, and the supplied TF inventory is source-mismatched and is
therefore not accepted into the frame graph.

## Verification

The focused checks are:

    cargo test -p spatialrust-viewer --features serde
    cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_studio
    cargo fmt --all -- --check

The output state and dashboard are generated only from absolute paths. The
example refuses existing output paths, requires an exact expected SHA-256, and
requires an E2E manifest input entry before it marks receipt-derived geometry
renderable.
