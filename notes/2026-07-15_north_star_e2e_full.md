# North-star E2E full portable path

Date: 2026-07-15 (Asia/Tokyo)

## Path

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust\tests\north_star_pipeline.rs`

1. RGB → mock depth → XYZ cloud
2. `SpatialRecord` / `MemoryEpisode` / provenance
3. MCAP XYZ write/read round-trip
4. ROS 2 CDR PointCloud2 encode → loopback → decode
5. Trajectory / pose-graph substrate (unchanged)
6. TSDF mesh → glTF JSON + USDA ASCII import check
7. Gaussian CPU soft-splat framebuffer
8. Semantic / runtime / distribute / conformance markers

## Feature

`north-star-e2e` = `north-star` + `ai-vision-pipeline` + `sync-mcap` + `runtime-ros2`
(also pulls `scene-gaussian` / `interchange-openusd` via `north-star`)

## Commands

```text
cargo test -p spatialrust --features north-star-e2e --test north_star_pipeline
cargo run -p spatialrust --example north_star_demo --features north-star-e2e
```
