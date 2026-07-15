# Deepen MCAP IO and TSDF meshing

Date: 2026-07-15 (Asia/Tokyo)

## Sync MCAP

Path: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-sync\src\mcap_io.rs`

- Feature `mcap` / facade `sync-mcap`
- `write_memory_episode_mcap` / `read_memory_episode_mcap`
- Encoding: `application/x-spatialrust-xyz-v1` (XYZ-only stamped clouds)
- Reads via `std::fs::read` + `MessageStream` (no mmap / no `unsafe`)

```text
cargo test -p spatialrust-sync --features mcap --lib
```

## Scene meshing

Path: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-scene\src\tsdf.rs`

- Truncation-band point integration (view-ray SDF)
- Marching tetrahedra zero isolevel extraction

```text
cargo test -p spatialrust-scene --lib
cargo test -p spatialrust --features north-star-e2e --test north_star_pipeline
```
