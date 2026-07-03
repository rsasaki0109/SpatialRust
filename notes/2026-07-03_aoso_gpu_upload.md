# Epic 74 — AoSoA XYZ GPU upload helper (2026-07-03)

## Goal

Bridge [`AoSoAXyzChunk`] (Epic 72) to wgpu storage buffers via the existing
[`GpuBufferPool`] upload path.

## API (`gpu-aoso-staging`)

| Symbol | Role |
| --- | --- |
| `GpuAoSoXyzChunk` | Pooled storage buffer + point count |
| `GpuAoSoXyzChunk::upload` | Upload packed interleaved slice |
| `GpuAoSoXyzChunk::pack_and_upload` | Pack one `SpatialTensorChunk` then upload |
| `GpuAoSoXyzChunk::recycle` | Return buffer to upload pool |
| `upload_spatial_tensor_xyz_chunks` | Upload all chunks in a `SpatialTensor` |

Meta feature: `gpu-aoso-staging` (= `tensor-aoso` + `gpu-wgpu`).

## Example

```rust
use spatialrust::core::{PointCloud, SpatialTensor};
use spatialrust::gpu::{GpuAoSoXyzChunk, WgpuRuntime, upload_spatial_tensor_xyz_chunks};

let runtime = WgpuRuntime::shared()?;
let tensor = cloud.spatial_tensor()?;
let gpu_chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor)?;
for chunk in gpu_chunks {
    // dispatch kernel using chunk.buffer(), chunk.point_count()
    chunk.recycle(&runtime);
}
```

## Tests

```bash
cargo test -p spatialrust-gpu --features gpu-aoso-staging
```

## Next steps

- Kernel dispatch consuming interleaved XYZ chunks (normals / voxel staging)
- Normals/intensity interleaved packers
- DLPack export

[`AoSoAXyzChunk`]: ../../crates/spatialrust-core/src/tensor_aoso.rs
[`GpuBufferPool`]: ../../crates/spatialrust-gpu/src/upload_cache.rs
