# 145J-B Calibration Evidence Gate

Date: 2026-08-04

## Direction

The canonical rosbag2 snapshot has front/rear PointCloud2 topics but no
`/clock`, `/tf`, `/tf_static`, `/odom`, clock model, or extrinsic artifact.
145J-B therefore implements the registration contract and audit surface without
manufacturing calibration values.

## Implementation

`spatialrust-viewer::CalibrationEvidenceState` is a small serde-compatible
contract with:

- exact source identity inherited from `StudioSource`;
- separate clock/frame artifact receipts;
- explicit clock source/target domains, method, sample count, p95 offset, and
  uncertainty;
- finite source-bound `FrameTransform` edges;
- an acyclic graph with a root-to-`front` and root-to-`rear` path check;
- derived `registration_ready` and non-empty fail-closed blockers.

`rosbag2_calibration_evidence` reads two strict JSON document shapes:

- `spatialrust.calibration.clock-evidence` version 1;
- `spatialrust.calibration.frame-evidence` version 1.

The embedded source binding must match the canonical input path, byte size, and
SHA-256. The command writes JSON, HTML, and a re-hashed manifest even when the
gate is blocked, then exits 2. It never applies a clock correction or TF edge.

`rosbag2_mission_cockpit` accepts the optional
`--calibration-evidence` receipt. When supplied, it validates the state,
includes its checksum in the cockpit artifact list and manifest, and surfaces
its blockers while leaving publish/partition inspection available.

## External evidence

Canonical input:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

Canonical no-artifact run:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-calibration-evidence-v2/calibration-evidence.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-calibration-evidence-v2/calibration-evidence.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-calibration-evidence-v2/calibration-evidence.manifest.json`

It reports `identity_matches:true`, both artifact statuses
`not_registered`, an empty frame graph, `registration_ready:false`, and exits
2. The wrong-source probe is:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-calibration-evidence-validation-probe-v1/calibration-evidence.json`

It reports `identity_matches:false`, `registration_ready:false`, and exits 2.

Mission Cockpit integration:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-145jb-v1/mission-cockpit.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-145jb-v1/mission-cockpit.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145j-mission-cockpit-145jb-v1/mission-cockpit.manifest.json`

The cockpit retained four packet frames and 768 bounded samples,
`publish_ready:true`, `partition_ready:true`, and
`mapping_admitted:false`. Its manifest checked seven files, including the
calibration evidence receipt.

## Validation

- `cargo fmt --all -- --check`
- `cargo test -p spatialrust-viewer --features serde`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_calibration_evidence`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_mission_cockpit`
- targeted clippy for viewer, calibration evidence, and Mission Cockpit with
  `-D warnings`
- Node browser-script syntax checks for both evidence dashboards
