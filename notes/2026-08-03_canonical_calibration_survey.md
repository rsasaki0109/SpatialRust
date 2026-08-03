# Canonical calibration artifact survey

## Scope

The next 143B action was a read-only search for clock calibration and
front/rear frame calibration matching the canonical rosbag2 input:

- input:
  `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2_2020_09_23-15_58_07/rosbag2_2020_09_23-15_58_07.db3`
- size: `713670656` bytes
- SHA-256:
  `b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8`
- capture start: `1600901887524769241` ns since epoch
- sensor topics: `/lidar_front/points_raw` (768),
  `/lidar_rear/points_raw` (767)

## Read-only search evidence

- The canonical directory contains the DB3, `metadata.yaml`, and SQLite
  `-wal`/`-shm` sidecars only. No clock, TF, extrinsic, or frame-calibration
  file is present.
- The companion archive
  `/media/sasaki/aiueo/datasets/migrated/autoware_data/rosbag2-astuff-1-lidar-only.tar.gz`
  lists only the canonical bag directory, its metadata, and DB3 payload.
- The migration manifest
  `/media/sasaki/aiueo/datasets/migrated/autoware_data-migration-20260802.sha256`
  lists the canonical bag, metadata, sidecars, and unrelated migrated fixtures;
  it has no calibration artifact entry.
- A filename/content search across `/media/sasaki/aiueo` for calibration,
  extrinsic, clock, TF, transform, sensor-kit, canonical bag-name, start-time,
  and canonical front/rear topic identifiers found no additional source
  candidate outside existing SpatialRust receipts.
- The nearby
  `/media/sasaki/aiueo/datasets/migrated/autoware_data/all-sensors-bag1/all-sensors-bag1_compressed_0.db3`
  remains rejected: SHA-256
  `74e5915719a7b7b4820b5339207eeade0c656deaa38b8e5b5e8d18787a58ac22`, 2022
  capture, and `velodyne_*`/concatenated point-cloud naming.

## Gate result

The refreshed readiness receipt is
`/media/sasaki/aiueo/spatialrust-results/v1-3/143b-calibration-survey/canonical.calibration.readiness.json`.
It confirms the canonical sensor topics and absence of `/clock`, `/tf`,
`/tf_static`, and `/odom`, records both artifacts as `not_registered`, and
exits non-zero. No candidate was registered, copied, interpreted, or used for
mapping. The next external dependency is a real clock calibration and explicit
root-to-`lidar_front`/`lidar_rear` frame artifact matching the canonical input.

## Validation

- Read-only `rg --files` filename search and text search on the external SSD
- Read-only `find` inventory of the canonical directory and `tar -tzf` archive
  listing
- `rosbag2_calibration_readiness` against the canonical DB3; expected exit code
  `1` with an atomic blocker receipt
