//! Upload helpers for interleaved AoSoA XYZ chunks (`tensor-aoso` + `gpu-aoso-staging`).

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{
    AoSoAXyzChunk, PointCloud, SpatialError, SpatialResult, SpatialTensor, SpatialTensorChunk,
};
use wgpu::util::DeviceExt;
use wgpu::Buffer;

use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VoxelKeyUniform {
    origin: [f32; 4],
    inv_leaf: f32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VoxelKeyOutput {
    ix: i32,
    iy: i32,
    iz: i32,
    _pad: i32,
}

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

/// Computes voxel keys directly from uploaded interleaved XYZ chunks.
///
/// The returned outer vector preserves chunk boundaries and order. Positions
/// stay in their existing GPU buffers; only the computed keys are read back.
/// All chunks must use the same `origin` and `inv_leaf` so keys are globally
/// comparable across chunk boundaries.
pub fn compute_voxel_keys_aoso_chunks(
    runtime: &WgpuRuntime,
    chunks: &[GpuAoSoXyzChunk],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<Vec<Vec<(i64, i64, i64)>>> {
    if !inv_leaf.is_finite() || inv_leaf <= 0.0 {
        return Err(SpatialError::InvalidArgument(
            "inverse voxel leaf size must be finite and positive".to_owned(),
        ));
    }

    chunks
        .iter()
        .map(|chunk| compute_voxel_keys_aoso_chunk(runtime, chunk, origin, inv_leaf))
        .collect()
}

fn compute_voxel_keys_aoso_chunk(
    runtime: &WgpuRuntime,
    chunk: &GpuAoSoXyzChunk,
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<Vec<(i64, i64, i64)>> {
    if chunk.point_count == 0 {
        return Ok(Vec::new());
    }
    let point_count = u32::try_from(chunk.point_count).map_err(|_| {
        SpatialError::InvalidArgument("AoSoA chunk exceeds the GPU point limit".to_owned())
    })?;
    let device = runtime.device();
    let queue = runtime.queue();
    let uniform = VoxelKeyUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        inv_leaf,
        point_count,
        _pad0: 0,
        _pad1: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-key-aoso-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_len = chunk.point_count * std::mem::size_of::<VoxelKeyOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-key-aoso-output"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-key-aoso-staging"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pipelines = runtime.pipelines();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-key-aoso-bind-group"),
        layout: &pipelines.voxel_keys_aoso.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: chunk.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
        ],
    });
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("voxel-key-aoso") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-key-aoso-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_keys_aoso.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(point_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_len as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;
    let data = slice.get_mapped_range();
    let keys = bytemuck::cast_slice::<u8, VoxelKeyOutput>(&data)
        .iter()
        .map(|key| (i64::from(key.ix), i64::from(key.iy), i64::from(key.iz)))
        .collect();
    drop(data);
    staging_buffer.unmap();
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::{
        compute_voxel_keys_aoso_chunks, upload_spatial_tensor_xyz_chunks, GpuAoSoXyzChunk,
    };
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

    #[test]
    fn chunk_dispatch_matches_global_voxel_keys() {
        let mut builder = PointCloudBuilder::xyz();
        for point in [
            [-0.6, 0.0, 1.2],
            [-0.1, 0.4, 0.9],
            [0.0, 0.5, 0.0],
            [0.6, 1.1, -0.2],
            [1.4, -0.7, 0.3],
        ] {
            builder.push_point(point).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();

        let origin = [-1.0, -1.0, -1.0];
        let inv_leaf = 2.0;
        let actual = compute_voxel_keys_aoso_chunks(&runtime, &chunks, origin, inv_leaf)
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let expected = [
            (-0.6, 0.0, 1.2),
            (-0.1, 0.4, 0.9),
            (0.0, 0.5, 0.0),
            (0.6, 1.1, -0.2),
            (1.4, -0.7, 0.3),
        ]
        .map(|(x, y, z)| {
            (
                ((x - origin[0]) * inv_leaf).floor() as i64,
                ((y - origin[1]) * inv_leaf).floor() as i64,
                ((z - origin[2]) * inv_leaf).floor() as i64,
            )
        });
        assert_eq!(actual, expected);
        for chunk in chunks {
            chunk.recycle(&runtime);
        }
    }
}
