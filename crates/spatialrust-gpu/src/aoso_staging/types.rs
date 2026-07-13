use super::*;

/// GPU-resident `(nx, ny, nz, curvature)` output for AoSoA positions.
pub struct GpuAoSoNormals {
    pub(super) buffer: Buffer,
    pub(super) point_count: usize,
    pub(super) device_key: usize,
}

/// Sparse GPU-resident uniform grid built directly from AoSoA positions.
pub struct GpuAoSoRadiusGrid {
    pub(super) radius: f32,
    pub(super) segments: GpuVoxelSegments,
}

impl GpuAoSoRadiusGrid {
    /// Returns the grid cell size and intended neighbor radius.
    #[must_use]
    pub const fn radius(&self) -> f32 {
        self.radius
    }

    /// Returns sorted sparse cell keys, starts, and point indices on the GPU.
    #[must_use]
    pub const fn segments(&self) -> &GpuVoxelSegments {
        &self.segments
    }

    /// Consumes the grid and returns its GPU segment buffers.
    #[must_use]
    pub fn into_segments(self) -> GpuVoxelSegments {
        self.segments
    }
}

impl GpuAoSoNormals {
    pub(crate) const fn device_key(&self) -> usize {
        self.device_key
    }

    /// Returns the normal output storage buffer.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the number of normal records.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns this output buffer to the runtime storage pool.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        runtime.recycle_storage(self.buffer.size(), self.buffer);
    }

    /// Reads normal and curvature records back to the CPU.
    pub fn readback(&self, runtime: &WgpuRuntime) -> SpatialResult<Vec<GpuNormal>> {
        if self.point_count == 0 {
            return Ok(Vec::new());
        }
        let device = runtime.device();
        let output_len = self.point_count * std::mem::size_of::<[f32; 4]>();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normals-aoso-staging"),
            size: output_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("normals-aoso-readback"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, output_len as u64);
        runtime.queue().submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| {
                SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned())
            })?
            .map_err(|error| {
                SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
            })?;
        let data = slice.get_mapped_range();
        let normals = bytemuck::cast_slice::<u8, [f32; 4]>(&data)
            .iter()
            .map(|value| GpuNormal { normal: [value[0], value[1], value[2]], curvature: value[3] })
            .collect();
        drop(data);
        staging.unmap();
        Ok(normals)
    }
}

/// Aggregation policy for interleaved AoSoA voxel attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AoSoAAttributeAggregation {
    /// Average every `f32` field within each voxel.
    Average,
    /// Select every field from the first point in each voxel.
    First,
}

/// CPU-visible interleaved records produced by GPU voxel aggregation.
#[derive(Clone, Debug, PartialEq)]
pub struct AoSoAAttributeReduction {
    pub(super) data: Vec<f32>,
    pub(super) point_count: usize,
    pub(super) layout: AoSoAAttributeLayout,
}

impl AoSoAAttributeReduction {
    /// Returns the number of occupied-voxel records.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.point_count
    }

    /// Returns whether no records were produced.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.point_count == 0
    }

    /// Returns the record layout.
    #[must_use]
    pub const fn layout(&self) -> AoSoAAttributeLayout {
        self.layout
    }

    /// Returns packed records in voxel-key order.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Centroid result from the GPU-resident AoSoA voxel pipeline.
pub struct AoSoAVoxelCentroidResult {
    /// Averaged x coordinates per occupied voxel.
    pub out_x: Vec<f32>,
    /// Averaged y coordinates per occupied voxel.
    pub out_y: Vec<f32>,
    /// Averaged z coordinates per occupied voxel.
    pub out_z: Vec<f32>,
    /// Global segment metadata retained on the GPU.
    pub segments: GpuVoxelSegments,
    /// Combined interleaved source positions retained for downstream kernels.
    pub positions: GpuAoSoXyzBuffer,
}

impl AoSoAVoxelCentroidResult {
    /// Returns the number of occupied voxels in the result.
    #[must_use]
    pub fn len(&self) -> usize {
        self.out_x.len()
    }

    /// Returns whether the result has no occupied voxels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.out_x.is_empty()
    }

    /// Recycles the retained position buffer and drops result metadata.
    pub fn recycle(self, runtime: &WgpuRuntime) {
        self.positions.recycle(runtime);
    }
}

/// GPU storage buffer containing contiguous interleaved XYZ positions.
pub struct GpuAoSoXyzBuffer {
    pub(super) buffer: Buffer,
    pub(super) point_count: usize,
    pub(super) device_key: usize,
}

impl GpuAoSoXyzBuffer {
    pub(crate) const fn device_key(&self) -> usize {
        self.device_key
    }

    /// Returns the underlying wgpu storage buffer for downstream kernels.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the number of interleaved XYZ points.
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

/// GPU storage buffer holding one interleaved XYZ chunk.
pub struct GpuAoSoXyzChunk {
    pub(super) buffer: Buffer,
    pub(super) point_count: usize,
}

/// GPU storage buffer for one interleaved attribute chunk and its layout.
pub struct GpuAoSoAttributeChunk {
    pub(super) buffer: Buffer,
    pub(super) point_count: usize,
    pub(super) layout: AoSoAAttributeLayout,
    pub(super) device_key: usize,
}

impl GpuAoSoAttributeChunk {
    /// Uploads a packed attribute chunk into pooled GPU storage.
    pub fn upload(
        runtime: &WgpuRuntime,
        label: &'static str,
        chunk: &AoSoAAttributeChunk,
    ) -> SpatialResult<Self> {
        let buffer = runtime.upload_f32_storage(label, chunk.as_slice())?;
        Ok(Self {
            buffer,
            point_count: chunk.len(),
            layout: chunk.layout(),
            device_key: runtime_device_key(runtime),
        })
    }

    pub(crate) const fn device_key(&self) -> usize {
        self.device_key
    }

    /// Returns the underlying wgpu storage buffer.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the number of packed points.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns the explicit stride and field-offset metadata.
    #[must_use]
    pub const fn layout(&self) -> AoSoAAttributeLayout {
        self.layout
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
