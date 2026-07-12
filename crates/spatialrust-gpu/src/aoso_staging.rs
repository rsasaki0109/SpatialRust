//! Upload helpers for interleaved AoSoA XYZ chunks (`tensor-aoso` + `gpu-aoso-staging`).

use spatialrust_core::{
    AoSoAXyzChunk, PointCloud, SpatialResult, SpatialTensor, SpatialTensorChunk,
};
use wgpu::Buffer;

use crate::runtime::WgpuRuntime;

/// GPU storage buffer holding one interleaved XYZ chunk.
pub struct GpuAoSoXyzChunk {
    buffer: Buffer,
    point_count: usize,
}

impl GpuAoSoXyzChunk {
    /// Uploads a packed [`AoSoAXyzChunk`] into a pooled storage buffer.
    pub fn upload(
        runtime: &WgpuRuntime,
        label: &'static str,
        chunk: &AoSoAXyzChunk,
    ) -> SpatialResult<Self> {
        let buffer = runtime.upload_f32_storage(label, chunk.as_slice())?;
        Ok(Self { buffer, point_count: chunk.len() })
    }

    /// Packs and uploads one [`SpatialTensorChunk`] from `cloud`.
    pub fn pack_and_upload(
        runtime: &WgpuRuntime,
        label: &'static str,
        chunk: &SpatialTensorChunk,
        cloud: &PointCloud,
    ) -> SpatialResult<Self> {
        Self::upload(runtime, label, &chunk.pack_xyz(cloud)?)
    }

    /// Returns the wgpu storage buffer.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the number of points in this chunk.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns the storage buffer size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.buffer.size()
    }

    /// Returns this buffer to the runtime upload pool.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        runtime.recycle_storage(self.buffer.size(), self.buffer);
    }
}

/// Packs and uploads every chunk in a [`SpatialTensor`].
pub fn upload_spatial_tensor_xyz_chunks(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
) -> SpatialResult<Vec<GpuAoSoXyzChunk>> {
    let cloud = tensor.cloud();
    tensor
        .chunks()
        .map(|chunk| GpuAoSoXyzChunk::pack_and_upload(runtime, "aoso-xyz-chunk", &chunk, cloud))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{upload_spatial_tensor_xyz_chunks, GpuAoSoXyzChunk};
    use crate::runtime::WgpuRuntime;
    use spatialrust_core::{PointCloudBuilder, SpatialTensor};

    #[test]
    fn upload_matches_interleaved_byte_length() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0]).unwrap();
        let cloud = builder.build().unwrap();
        let packed = cloud
            .spatial_tensor_chunks(4)
            .unwrap()
            .chunks()
            .next()
            .unwrap()
            .pack_xyz(&cloud)
            .unwrap();

        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let gpu_chunk = GpuAoSoXyzChunk::upload(&runtime, "aoso-upload-test", &packed).unwrap();
        assert_eq!(gpu_chunk.point_count(), 2);
        assert_eq!(gpu_chunk.byte_len(), 24);
        gpu_chunk.recycle(&runtime);
    }

    #[test]
    fn uploads_all_tensor_chunks() {
        let mut builder = PointCloudBuilder::xyz();
        for index in 0..5 {
            builder.push_point([index as f32, 0.0, 0.0]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();

        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let gpu_chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
        assert_eq!(gpu_chunks.len(), 3);
        assert_eq!(gpu_chunks[0].point_count(), 2);
        assert_eq!(gpu_chunks[2].point_count(), 1);
        for chunk in gpu_chunks {
            chunk.recycle(&runtime);
        }
    }
}
