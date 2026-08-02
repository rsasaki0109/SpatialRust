# North-star E2E pipeline

Date: 2026-07-15 (Asia/Tokyo)

## Path

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust\tests\north_star_pipeline.rs`

1. RGB letterbox → `MockProfile::SyntheticDepth`
2. Depth unproject → `SpatialRecord` / `MemoryEpisode` / `Episode`
3. `Trajectory` + `PoseGraph` localization substrate
4. `TsdfVolume` integrate → `TriangleMesh` → glTF JSON
5. Semantic search + bounded runtime + partition graph + conformance report

## Commands

```text
cargo test -p spatialrust --features north-star-e2e --test north_star_pipeline
cargo run -p spatialrust --example north_star_demo --features north-star-e2e
```
