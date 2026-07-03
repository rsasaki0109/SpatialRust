# Epic 72 — AoSoA chunk packing (2026-07-03)

## Goal

Add optional interleaved `[x,y,z, …]` packing per [`SpatialTensorChunk`] for SIMD/GPU
staging while keeping schema-SoA storage in [`PointCloud`].

## API (`spatialrust-core/tensor-aoso`)

| Symbol | Role |
| --- | --- |
| `AoSoAXyzChunk` | Owned interleaved buffer for one chunk |
| `SpatialTensorChunk::pack_xyz` | Build owned chunk buffer |
| `SpatialTensorChunk::pack_xyz_into` | Reuse caller `Vec<f32>` scratch |

Meta crate feature: `tensor-aoso`.

Stability: **Provisional**.

## Design constraints

- On-demand copy from SoA columns (no layout change to `PointCloud`)
- XYZ only for v1; other fields can follow the same pattern
- Zero `unsafe` in core

## Tests

```bash
cargo test -p spatialrust-core --features tensor-aoso
```

## Next steps

- Kernel dispatch consuming `GpuAoSoXyzChunk` buffers (Epic 74 upload done)
- Normals / intensity interleaved packers
- DLPack export of chunk views

[`SpatialTensorChunk`]: ../../crates/spatialrust-core/src/tensor.rs
[`PointCloud`]: ../../crates/spatialrust-core/src/pointcloud.rs
