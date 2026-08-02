# Epic 93 scan odometry bridge — 2026-08-03

## Scope

`spatialrust-mapping` now connects the bounded synchronized episode substrate
to localization contracts:

- `ScanOdometry` selects at most `ScanOdometryConfig::max_scans` records from
  one topic, validates a common frame and clock domain, and rejects scans
  below the configured point minimum.
- `ScanMatcher` is the algorithm boundary. Each result is a
  `current_T_previous` motion, inserted as both a `DeltaMotion` and sequential
  `PoseGraphEdge`; the output trajectory is reconstructed in deterministic
  topic/timestamp order.
- `mapping-scan-icp` provides `IcpScanMatcher` without making ICP part of the
  default mapping build. The facade equivalent is `mapping-scan-icp`.
- Prefix truncation is explicit in `ScanOdometryResult::truncated`; no episode
  or cloud is expanded implicitly.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo fmt --all
cargo test -p spatialrust-mapping --all-features
cargo check -p spatialrust --features mapping-scan-icp
```

The tests cover deterministic trajectory/pose-graph construction, bounded
prefix reporting, mixed-frame rejection, and synthetic point-to-point ICP
motion recovery.
