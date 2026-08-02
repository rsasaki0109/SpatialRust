# Epic 91 spatial records / Arrow completion record

Date: 2026-07-15 (Asia/Tokyo)

## Delivered contracts

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-records`
  owns `SpatialRecord`, schema id/version/compatibility, migration, and
  in-memory chunked record sources/sinks (Arrow-free).
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-arrow`
  owns Arrow C Data export/import, C Stream over `SpatialRecordSource`, and
  CPU Arrow C Device arrays behind `arrow-c-data` / `arrow-c-stream` /
  `arrow-c-device`.
- Facade flags: `records`, `arrow-c-data`, `arrow-c-stream`, `arrow-c-device`,
  `arrow-full`.
- Long-horizon agent rule:
  `C:\Users\rsasa\Workspace\SpatialRust\.cursor\rules\long-horizon.mdc`
- ROADMAP Epics 92–100 activated as Planned with first-slice outlines.

## Verification

- `cargo test -p spatialrust-records --lib`
- `cargo test -p spatialrust-arrow --features full --lib`

## Notes

MCAP, ROS 2, and non-CPU Arrow devices remain deferred to Epics 92 / 97 / later
device-copy work. Out-of-core backends beyond in-memory chunking are follow-ons
on the `SpatialRecordSource` trait.
