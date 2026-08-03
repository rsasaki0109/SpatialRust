# 145G Dataset Health Dashboard

Date: 2026-08-03

## Scope

This slice adds a source-bound Dataset Health Dashboard for the canonical
rosbag2 acceptance run. `DatasetHealthState` keeps dataset readiness,
calibration readiness, and mapping admission separate. It aggregates topic
inventory, canonical mesh/readiness checks, storage receipts, and the six
visual slices from 145A through 145F without accepting an alternate source,
implicit transform, or unapproved fusion.

`rosbag2_dataset_health` reads the canonical rosbag2 SQLite database, the
calibration readiness receipt, and each prior stage's receipt-backed state. It
supports the existing legacy 145A/145B manifests as explicit file receipts,
re-hashes accepted artifacts, and writes JSON, HTML, and a checksummed manifest
to the external SSD.

## External evidence

Canonical input:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

Positive run:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145g-dataset-health-v2/dataset-health.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145g-dataset-health-v2/dataset-health.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145g-dataset-health-v2/dataset-health.manifest.json`

Observed positive state:

- `dataset_ready=true`, `mapping_admitted=false`
- two topics, 1,535 source messages, four retained records, 115,972 points
- six validated stages: 145A/C/D/E/F pass; 145B warning
- 25 artifacts and 18 checks: 13 pass, two warning, three non-critical blocked
- no `/clock`, `/tf`, `/tf_static`, or `/odom`; readiness receipt
  `registration_ready=false`

Negative source-binding probe:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145g-dataset-health-validation-probe/dataset-health.json`
- wrong expected SHA withheld all stage aggregation
- `dataset_ready=false`, `mapping_admitted=false`, CLI exit status 2

## Verification

Focused checks passed:

- `cargo fmt --all -- --check`
- `cargo test -p spatialrust-viewer dataset_health --features serde`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_dataset_health`
- positive and wrong-source negative external runs with manifest re-hashing

The canonical data is therefore healthy enough for bounded replay and visual
inspection, but not admitted as calibrated-world mapping evidence. The negative
probe confirms that a source identity failure is visible and fail-closed.
