# SpatialTensor chunked views (2026-07-03)

## Goal

Introduce the architecture-level **`SpatialTensor`** name as a provisional chunked
view over existing schema-SoA [`PointCloud`] storage — the first step toward
Hybrid Schema-SoA + chunked AoSoA described in `docs/ARCHITECTURE.md`.

## API (provisional)

| Symbol | Role |
| --- | --- |
| `SpatialTensor` | Borrowed view over a `PointCloud` with fixed chunk size |
| `SpatialTensorChunk` | Half-open index range + field slice helpers |
| `PointCloud::spatial_tensor()` | Default chunk size `16_384` |
| `PointCloud::spatial_tensor_chunks(n)` | Explicit chunk size |

Stability: **Provisional** (`docs/API_STABILITY.md`). No device migration or GPU
views yet.

## Design constraints

- Zero-copy: chunks are `&[f32]` slices into existing columns
- No IO / GPU in `spatialrust-core`
- Chunk size is a hint for parallel/GPU staging; algorithms may choose their own

## Next steps

- GPU upload helper for `AoSoAXyzChunk` (wgpu staging)
- Wire parallel chunk queries into feature estimation hot paths
- DLPack export of chunk views for AI integration

[`PointCloud`]: ../../crates/spatialrust-core/src/pointcloud.rs
