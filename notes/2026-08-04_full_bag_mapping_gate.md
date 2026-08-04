# 145K-A bounded full-bag mapping gate

Date: 2026-08-04

## Direction

The 145J-B registration contract is now available, but the canonical bag has
no matching clock or front/rear extrinsic artifacts. 145K-A therefore adds the
real mapping execution path without weakening that prerequisite: a mapping
state can be emitted as blocked, while calibrated-world admission requires all
source, clock, frame, full-bag, odometry, and TSDF gates.

## Implementation

`spatialrust-viewer::FullBagMappingState` records:

- exact input identity and optional parsed `CalibrationEvidenceState`;
- bag message, bounded source chunk, retained record/point/byte, and peak
  source-memory totals for both selected topics;
- corrected timeline bounds and ICP/pose-graph identity/counts;
- root-frame TSDF configuration, integrated records/points, and mesh totals;
- explicit `clock_applied`, `frame_graph_applied`, `full_bag_processed`,
  `odometry_complete`, `tsdf_complete`, and `mapping_admitted` decisions; and
- checksummed input, calibration, and derived mesh artifacts with blockers.

`rosbag2_full_bag_mapping` is feature-gated behind `rosbag2-sqlite`. It uses a
large but explicit source chunk bound, reassembles chunks with the same
PointCloud2 timestamp/frame, collects a bounded deterministic episode, applies
the registered clock offset plus an anchor-relative drift term, resolves the
accepted root-to-front/rear graph, runs ICP over all retained front records,
and integrates front plus synchronized rear records into a root-frame TSDF.
No calibration solver or implicit transform is introduced.

`rosbag2_mission_cockpit` accepts `--mapping-state`, includes its checksum in
the artifact list/manifest, and forces cockpit mapping admission false when
the full-bag state is blocked or bound to a different input.

## Canonical evidence

Canonical input:

- `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- SHA-256: `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`

No-artifact mapping run:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-full-bag-mapping-v3/full-bag-mapping.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-full-bag-mapping-v3/full-bag-mapping.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-full-bag-mapping-v3/full-bag-mapping.manifest.json`

The input identity matched, but `calibration` was null and
`mapping_admitted:false`. The command exited 2 after writing the state,
dashboard, and manifest; the manifest checked the input plus those two local
outputs. No mapping pipeline was run and no mesh was produced.

Mission Cockpit propagation:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-mission-cockpit-v3/mission-cockpit.json`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-mission-cockpit-v3/mission-cockpit.html`
- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-mission-cockpit-v3/mission-cockpit.manifest.json`

The cockpit remains `mapping_admitted:false` and lists the full-bag mapping
blockers while retaining upstream packet inspection behavior.

Wrong-input probe:

- `/media/sasaki/aiueo/spatialrust-results/v1-3/145k-full-bag-mapping-validation-probe-v1/full-bag-mapping.json`

It observes the canonical SHA against an all-`f` expected SHA, records
`identity_matches:false`, exits 2, and does not enter source ingest or mapping.

## Validation

- `cargo fmt --all`
- `cargo test -p spatialrust-viewer --features serde`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_full_bag_mapping`
- `cargo test -p spatialrust-ros2 --features rosbag2-sqlite --example rosbag2_mission_cockpit`
- targeted `cargo clippy` for the viewer, mapping runner, and cockpit with
  `-D warnings`
- canonical mapping runner negative execution: exit status 2,
  `mapping_admitted:false`, checked-files manifest
- Node syntax check for the generated full-bag mapping dashboard
