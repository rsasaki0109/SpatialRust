//! Upload helpers for interleaved AoSoA XYZ chunks (`tensor-aoso` + `gpu-aoso-staging`).

use bytemuck::{Pod, Zeroable};
use spatialrust_core::{
    AoSoAAttributeChunk, AoSoAAttributeLayout, AoSoAXyzChunk, PointCloud, SpatialError,
    SpatialResult, SpatialTensor, SpatialTensorChunk,
};
use wgpu::util::DeviceExt;
use wgpu::Buffer;

use crate::runtime::WgpuRuntime;
use crate::{build_voxel_segments_gpu_from_keys_buffer, GpuNormal, GpuVoxelSegments};

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VoxelReduceUniform {
    cell_count: u32,
    point_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AttributeReduceUniform {
    cell_count: u32,
    point_count: u32,
    stride: u32,
    first_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AoSoANormalsUniform {
    point_count: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SparseGridNormalsUniform {
    origin: [f32; 4],
    dims: [u32; 4],
    inv_cell: f32,
    radius_sq: f32,
    _pad0: f32,
    _pad1: f32,
}

/// GPU-resident `(nx, ny, nz, curvature)` output for AoSoA positions.
pub struct GpuAoSoNormals {
    buffer: Buffer,
    point_count: usize,
    device_key: usize,
}

/// Sparse GPU-resident uniform grid built directly from AoSoA positions.
pub struct GpuAoSoRadiusGrid {
    radius: f32,
    segments: GpuVoxelSegments,
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
    data: Vec<f32>,
    point_count: usize,
    layout: AoSoAAttributeLayout,
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
    buffer: Buffer,
    point_count: usize,
    device_key: usize,
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
    buffer: Buffer,
    point_count: usize,
}

/// GPU storage buffer for one interleaved attribute chunk and its layout.
pub struct GpuAoSoAttributeChunk {
    buffer: Buffer,
    point_count: usize,
    layout: AoSoAAttributeLayout,
    device_key: usize,
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

/// Packs and uploads every tensor chunk with an explicit attribute layout.
pub fn upload_spatial_tensor_attribute_chunks(
    runtime: &WgpuRuntime,
    tensor: &SpatialTensor<'_>,
    layout: AoSoAAttributeLayout,
) -> SpatialResult<Vec<GpuAoSoAttributeChunk>> {
    let cloud = tensor.cloud();
    tensor
        .chunks()
        .map(|chunk| {
            let packed = if layout == AoSoAAttributeLayout::XYZ_INTENSITY {
                chunk.pack_xyz_intensity(cloud)?
            } else if layout == AoSoAAttributeLayout::XYZ_NORMALS {
                chunk.pack_xyz_normals(cloud)?
            } else if layout == AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS {
                chunk.pack_xyz_intensity_normals(cloud)?
            } else {
                return Err(SpatialError::InvalidArgument(
                    "unsupported AoSoA attribute layout".to_owned(),
                ));
            };
            GpuAoSoAttributeChunk::upload(runtime, "aoso-attribute-chunk", &packed)
        })
        .collect()
}

/// Aggregates uploaded interleaved records using global GPU voxel segments.
pub fn reduce_voxel_attributes_aoso_chunks(
    runtime: &WgpuRuntime,
    chunks: &[GpuAoSoAttributeChunk],
    segments: &GpuVoxelSegments,
    aggregation: AoSoAAttributeAggregation,
) -> SpatialResult<AoSoAAttributeReduction> {
    let first = chunks.first().ok_or_else(|| {
        SpatialError::InvalidArgument("attribute chunks must not be empty".to_owned())
    })?;
    let layout = first.layout;
    if chunks.iter().any(|chunk| chunk.layout != layout) {
        return Err(SpatialError::InvalidArgument(
            "attribute chunks must use one AoSoA layout".to_owned(),
        ));
    }
    let point_count: usize = chunks.iter().map(|chunk| chunk.point_count).sum();
    if point_count != segments.point_count() as usize {
        return Err(SpatialError::BufferLengthMismatch {
            expected: segments.point_count() as usize,
            found: point_count,
        });
    }
    let cell_count = segments.cell_count();
    if cell_count == 0 {
        return Ok(AoSoAAttributeReduction { data: Vec::new(), point_count: 0, layout });
    }

    let stride = layout.stride_f32();
    let device = runtime.device();
    let combined_len = point_count * stride * std::mem::size_of::<f32>();
    let combined = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aoso-attributes-combined"),
        size: combined_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_count = cell_count as usize * stride;
    let output_len = output_count * std::mem::size_of::<f32>();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-attributes-aoso-output"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-attributes-aoso-staging"),
        size: output_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform = AttributeReduceUniform {
        cell_count,
        point_count: segments.point_count(),
        stride: stride as u32,
        first_mode: u32::from(aggregation == AoSoAAttributeAggregation::First),
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-attributes-aoso-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipelines = runtime.pipelines();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-attributes-aoso-bind-group"),
        layout: &pipelines.voxel_reduce_attributes_aoso.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: combined.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-attributes-aoso-encoder"),
    });
    let mut offset = 0;
    for chunk in chunks {
        let bytes = chunk.buffer.size();
        encoder.copy_buffer_to_buffer(&chunk.buffer, 0, &combined, offset, bytes);
        offset += bytes;
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-reduce-attributes-aoso-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_reduce_attributes_aoso.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((output_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_len as u64);
    runtime.queue().submit(Some(encoder.finish()));

    let slice = staging.slice(..);
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
    let mapped = slice.get_mapped_range();
    let data = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(AoSoAAttributeReduction { data, point_count: cell_count as usize, layout })
}

/// Estimates normals directly from a retained interleaved XYZ GPU buffer.
///
/// `neighbors` is a global flattened `point_count * k` index array. Position
/// data remains GPU-resident; only neighbor indices are uploaded. The returned
/// normal buffer stays on the GPU until [`GpuAoSoNormals::readback`] is called.
pub fn estimate_normals_aoso_gpu(
    runtime: &WgpuRuntime,
    positions: &GpuAoSoXyzBuffer,
    neighbors: &[u32],
    k: u32,
) -> SpatialResult<GpuAoSoNormals> {
    let point_count = positions.point_count;
    if k == 0 || neighbors.len() != point_count * k as usize {
        return Err(SpatialError::InvalidArgument(format!(
            "neighbors must have point_count*k = {} entries, got {}",
            point_count * k as usize,
            neighbors.len()
        )));
    }
    let device = runtime.device();
    if point_count == 0 {
        return Ok(GpuAoSoNormals {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("normals-aoso-empty"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            point_count: 0,
            device_key: runtime_device_key(runtime),
        });
    }
    let point_count_u32 = u32::try_from(point_count).map_err(|_| {
        SpatialError::InvalidArgument("AoSoA positions exceed the GPU point limit".to_owned())
    })?;
    let neighbor_buffer = runtime.upload_u32_storage("normals-aoso-neighbors", neighbors)?;
    let uniform = AoSoANormalsUniform { point_count: point_count_u32, k, _pad0: 0, _pad1: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-aoso-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normals-aoso-output"),
        size: (point_count * std::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader_source = crate::kernels::NORMALS_WGSL
        .replace(
            "@group(0) @binding(1) var<storage, read> xs: array<f32>;\n@group(0) @binding(2) var<storage, read> ys: array<f32>;\n@group(0) @binding(3) var<storage, read> zs: array<f32>;",
            "@group(0) @binding(1) var<storage, read> positions: array<f32>;",
        )
        .replace("xs[idx]", "positions[idx * 3u]")
        .replace("ys[idx]", "positions[idx * 3u + 1u]")
        .replace("zs[idx]", "positions[idx * 3u + 2u]");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("normals-aoso-shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normals-aoso-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normals-aoso-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: neighbor_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("normals-aoso-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normals-aoso-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(point_count_u32.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    Ok(GpuAoSoNormals { buffer: output, point_count, device_key: runtime_device_key(runtime) })
}

/// Builds a sparse uniform radius grid directly from retained AoSoA positions.
///
/// Cell keys, sorting, and segment compaction stay on the GPU. Grid keys use a
/// zero origin and `floor(position / radius)`; negative coordinates therefore
/// remain valid without requiring CPU-side bounds discovery.
pub fn build_radius_grid_aoso_gpu(
    runtime: &WgpuRuntime,
    positions: &GpuAoSoXyzBuffer,
    radius: f32,
) -> SpatialResult<GpuAoSoRadiusGrid> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(SpatialError::InvalidArgument(
            "grid radius must be finite and positive".to_owned(),
        ));
    }
    let point_count = u32::try_from(positions.point_count).map_err(|_| {
        SpatialError::InvalidArgument("AoSoA positions exceed the GPU point limit".to_owned())
    })?;
    let empty = empty_storage_buffer(runtime);
    let segments = if point_count == 0 {
        build_voxel_segments_gpu_from_keys_buffer(runtime, &empty, 0, 1)?
    } else {
        let keys = dispatch_voxel_keys_aoso(
            runtime,
            positions.buffer(),
            point_count,
            [0.0; 3],
            1.0 / radius,
        )?;
        build_voxel_segments_gpu_from_keys_buffer(
            runtime,
            &keys,
            point_count,
            point_count.next_power_of_two(),
        )?
    };
    Ok(GpuAoSoRadiusGrid { radius, segments })
}

/// Estimates radius normals directly from a GPU-resident sparse AoSoA grid.
pub fn estimate_normals_radius_grid_aoso_gpu(
    runtime: &WgpuRuntime,
    positions: &GpuAoSoXyzBuffer,
    grid: &GpuAoSoRadiusGrid,
) -> SpatialResult<GpuAoSoNormals> {
    let point_count = u32::try_from(positions.point_count).map_err(|_| {
        SpatialError::InvalidArgument("AoSoA positions exceed the GPU point limit".to_owned())
    })?;
    if grid.segments.point_count() != point_count {
        return Err(SpatialError::BufferLengthMismatch {
            expected: point_count as usize,
            found: grid.segments.point_count() as usize,
        });
    }
    if point_count == 0 {
        return estimate_normals_aoso_gpu(runtime, positions, &[], 1);
    }
    let device = runtime.device();
    let uniform = SparseGridNormalsUniform {
        origin: [0.0; 4],
        dims: [grid.segments.cell_count(), 0, 0, point_count],
        inv_cell: 1.0 / grid.radius,
        radius_sq: grid.radius * grid.radius,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normals-radius-aoso-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normals-radius-aoso-output"),
        size: u64::from(point_count) * std::mem::size_of::<[f32; 4]>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("normals-radius-aoso-shader"),
        source: wgpu::ShaderSource::Wgsl(sparse_grid_normals_shader().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("normals-radius-aoso-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normals-radius-aoso-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: grid.segments.keys_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grid.segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: grid.segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 5, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("normals-radius-aoso-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("normals-radius-aoso-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(point_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    runtime.queue().submit(Some(encoder.finish()));
    Ok(GpuAoSoNormals {
        buffer: output,
        point_count: point_count as usize,
        device_key: runtime_device_key(runtime),
    })
}

fn sparse_grid_normals_shader() -> String {
    let declarations = "@group(0) @binding(1) var<storage, read> xs: array<f32>;\n@group(0) @binding(2) var<storage, read> ys: array<f32>;\n@group(0) @binding(3) var<storage, read> zs: array<f32>;\n@group(0) @binding(4) var<storage, read> sorted: array<u32>;\n@group(0) @binding(5) var<storage, read> cell_start: array<u32>;\n@group(0) @binding(6) var<storage, read_write> out_normals: array<vec4<f32>>;";
    let sparse_declarations = r#"struct SparseKey { ix: i32, iy: i32, iz: i32, pad: i32, }
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<SparseKey>;
@group(0) @binding(3) var<storage, read> sorted: array<u32>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_normals: array<vec4<f32>>;"#;
    let cell_coord = r#"fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 {
    let c = i32(floor((value - origin) * inv_cell));
    return clamp(c, 0, i32(dim) - 1);
}"#;
    let lookup = r#"fn cell_coord(value: f32, origin: f32, inv_cell: f32, dim: u32) -> i32 { return i32(floor((value - origin) * inv_cell)); }
fn key_less(key: SparseKey, x: i32, y: i32, z: i32) -> bool {
    if (key.ix != x) { return key.ix < x; }
    if (key.iy != y) { return key.iy < y; }
    return key.iz < z;
}
fn find_cell(x: i32, y: i32, z: i32, count: u32) -> i32 {
    var low = 0u; var high = count;
    loop { if (low >= high) { break; } let mid = low + (high-low)/2u; if (key_less(keys[mid],x,y,z)) { low=mid+1u; } else { high=mid; } }
    if (low < count) { let key=keys[low]; if (key.ix==x && key.iy==y && key.iz==z) { return i32(low); } }
    return -1;
}"#;
    let dense = "let cid = (u32(nz) * dimy + u32(ny)) * dimx + u32(nx);\n                let begin = cell_start[cid];\n                let end = cell_start[cid + 1u];";
    let sparse = "let found = find_cell(nx, ny, nz, cell_count);\n                if (found < 0) { continue; }\n                let cid = u32(found);\n                let begin = cell_start[cid];\n                let end = select(params.dims.w, cell_start[cid + 1u], cid + 1u < cell_count);";
    crate::kernels::NORMALS_GRID_WGSL
        .replace(declarations, sparse_declarations)
        .replace(cell_coord, lookup)
        .replace("let px = xs[i];", "let px = positions[i * 3u];")
        .replace("let py = ys[i];", "let py = positions[i * 3u + 1u];")
        .replace("let pz = zs[i];", "let pz = positions[i * 3u + 2u];")
        .replace("let dimx = params.dims.x;\n    let dimy = params.dims.y;\n    let dimz = params.dims.z;", "let cell_count = params.dims.x;")
        .replace(", dimx);", ", cell_count);")
        .replace(", dimy);", ", cell_count);")
        .replace(", dimz);", ", cell_count);")
        .replace("if (nz < 0 || nz >= i32(dimz)) { continue; }", "")
        .replace("if (ny < 0 || ny >= i32(dimy)) { continue; }", "")
        .replace("if (nx < 0 || nx >= i32(dimx)) { continue; }", "")
        .replace(dense, sparse)
        .replace("xs[j]", "positions[j * 3u]")
        .replace("ys[j]", "positions[j * 3u + 1u]")
        .replace("zs[j]", "positions[j * 3u + 2u]")
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

/// Runs global voxel centroid downsampling from uploaded AoSoA chunks.
///
/// Chunk buffers are concatenated with GPU-to-GPU copies. Key generation,
/// sorting, segmentation, and centroid reduction then operate on that combined
/// GPU buffer, so voxels spanning chunk boundaries are merged correctly. Only
/// the final centroids are read back.
pub fn downsample_voxel_centroid_aoso_chunks(
    runtime: &WgpuRuntime,
    chunks: &[GpuAoSoXyzChunk],
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<AoSoAVoxelCentroidResult> {
    if !inv_leaf.is_finite() || inv_leaf <= 0.0 {
        return Err(SpatialError::InvalidArgument(
            "inverse voxel leaf size must be finite and positive".to_owned(),
        ));
    }
    let total_points = chunks.iter().try_fold(0usize, |total, chunk| {
        total
            .checked_add(chunk.point_count)
            .ok_or_else(|| SpatialError::InvalidArgument("AoSoA point count overflow".to_owned()))
    })?;
    if total_points == 0 {
        return Ok(AoSoAVoxelCentroidResult {
            out_x: Vec::new(),
            out_y: Vec::new(),
            out_z: Vec::new(),
            segments: build_voxel_segments_gpu_from_keys_buffer(
                runtime,
                &empty_storage_buffer(runtime),
                0,
                1,
            )?,
            positions: empty_aoso_buffer(runtime),
        });
    }
    let point_count = u32::try_from(total_points).map_err(|_| {
        SpatialError::InvalidArgument("AoSoA chunks exceed the GPU point limit".to_owned())
    })?;
    let positions = combine_aoso_chunks(runtime, chunks, total_points);
    let keys =
        dispatch_voxel_keys_aoso(runtime, positions.buffer(), point_count, origin, inv_leaf)?;
    let segments = build_voxel_segments_gpu_from_keys_buffer(
        runtime,
        &keys,
        point_count,
        point_count.next_power_of_two(),
    )?;
    let (out_x, out_y, out_z) =
        reduce_voxel_centroids_aoso(runtime, positions.buffer(), &segments)?;
    Ok(AoSoAVoxelCentroidResult { out_x, out_y, out_z, segments, positions })
}

fn empty_aoso_buffer(runtime: &WgpuRuntime) -> GpuAoSoXyzBuffer {
    GpuAoSoXyzBuffer {
        buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("aoso-empty-positions"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        point_count: 0,
        device_key: runtime_device_key(runtime),
    }
}

fn empty_storage_buffer(runtime: &WgpuRuntime) -> Buffer {
    runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("aoso-empty-storage"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn combine_aoso_chunks(
    runtime: &WgpuRuntime,
    chunks: &[GpuAoSoXyzChunk],
    total_points: usize,
) -> GpuAoSoXyzBuffer {
    let device = runtime.device();
    let combined = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aoso-xyz-combined"),
        size: (total_points * 3 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("aoso-combine-encoder"),
    });
    let mut offset = 0;
    for chunk in chunks {
        let bytes = (chunk.point_count * 3 * std::mem::size_of::<f32>()) as u64;
        if bytes > 0 {
            encoder.copy_buffer_to_buffer(&chunk.buffer, 0, &combined, offset, bytes);
            offset += bytes;
        }
    }
    runtime.queue().submit(Some(encoder.finish()));
    GpuAoSoXyzBuffer {
        buffer: combined,
        point_count: total_points,
        device_key: runtime_device_key(runtime),
    }
}

pub(crate) fn runtime_device_key(runtime: &WgpuRuntime) -> usize {
    runtime.device() as *const wgpu::Device as usize
}

fn dispatch_voxel_keys_aoso(
    runtime: &WgpuRuntime,
    positions: &Buffer,
    point_count: u32,
    origin: [f32; 3],
    inv_leaf: f32,
) -> SpatialResult<Buffer> {
    let device = runtime.device();
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
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-key-aoso-output"),
        size: u64::from(point_count) * std::mem::size_of::<VoxelKeyOutput>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let pipelines = runtime.pipelines();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-key-aoso-bind-group"),
        layout: &pipelines.voxel_keys_aoso.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
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
    runtime.queue().submit(Some(encoder.finish()));
    Ok(output)
}

fn reduce_voxel_centroids_aoso(
    runtime: &WgpuRuntime,
    positions: &Buffer,
    segments: &GpuVoxelSegments,
) -> SpatialResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let cell_count = segments.cell_count();
    if cell_count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let device = runtime.device();
    let output_len = u64::from(cell_count) * std::mem::size_of::<[f32; 4]>() as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-aoso-output"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("voxel-reduce-aoso-staging"),
        size: output_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let uniform =
        VoxelReduceUniform { cell_count, point_count: segments.point_count(), _pad0: 0, _pad1: 0 };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel-reduce-aoso-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipelines = runtime.pipelines();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel-reduce-aoso-bind-group"),
        layout: &pipelines.voxel_reduce_aoso.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: segments.point_indices_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: segments.cell_starts_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 3, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-reduce-aoso-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel-reduce-aoso-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.voxel_reduce_aoso.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_len);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
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
    let centroids: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let mut out_x = Vec::with_capacity(centroids.len());
    let mut out_y = Vec::with_capacity(centroids.len());
    let mut out_z = Vec::with_capacity(centroids.len());
    for centroid in centroids {
        out_x.push(centroid[0]);
        out_y.push(centroid[1]);
        out_z.push(centroid[2]);
    }
    drop(data);
    staging.unmap();
    Ok((out_x, out_y, out_z))
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
        build_radius_grid_aoso_gpu, compute_voxel_keys_aoso_chunks,
        downsample_voxel_centroid_aoso_chunks, estimate_normals_aoso_gpu,
        estimate_normals_radius_grid_aoso_gpu, reduce_voxel_attributes_aoso_chunks,
        upload_spatial_tensor_attribute_chunks, upload_spatial_tensor_xyz_chunks,
        AoSoAAttributeAggregation, GpuAoSoXyzChunk,
    };
    use crate::runtime::WgpuRuntime;
    use spatialrust_core::{
        AoSoAAttributeLayout, PointCloudBuilder, SpatialTensor, StandardSchemas,
    };

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

    #[test]
    fn centroid_pipeline_merges_voxels_across_chunks() {
        let mut builder = PointCloudBuilder::xyz();
        for point in
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [1.3, 0.0, 0.0], [2.1, 0.0, 0.0], [2.5, 0.0, 0.0]]
        {
            builder.push_point(point).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();

        let result =
            downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [0.0; 3], 1.0).unwrap();
        assert_eq!(result.positions.point_count(), 5);
        assert_eq!(result.positions.byte_len(), 5 * 3 * 4);
        let segments = result.segments.to_voxel_segments(&runtime).unwrap();
        assert_eq!(segments.keys, vec![(0, 0, 0), (1, 0, 0), (2, 0, 0)]);
        assert_eq!(result.out_x.len(), 3);
        assert!((result.out_x[0] - 0.1).abs() < 1e-6);
        assert!((result.out_x[1] - 1.2).abs() < 1e-6);
        assert!((result.out_x[2] - 2.3).abs() < 1e-6);
        assert_eq!(result.out_y, vec![0.0; 3]);
        assert_eq!(result.out_z, vec![0.0; 3]);
        result.recycle(&runtime);
        for chunk in chunks {
            chunk.recycle(&runtime);
        }
    }

    #[test]
    fn empty_centroid_pipeline_has_recyclable_positions() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let chunks = Vec::new();

        let result =
            downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [0.0; 3], 1.0).unwrap();
        assert!(result.is_empty());
        assert_eq!(result.positions.point_count(), 0);
        result.recycle(&runtime);
    }

    #[test]
    fn uploads_composite_attribute_chunks_with_layout() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        for index in 0..5 {
            builder.push_point([index as f32, 1.0, 2.0, 0.5, 0.0, 0.0, 1.0]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");

        let chunks = upload_spatial_tensor_attribute_chunks(
            &runtime,
            &tensor,
            AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS,
        )
        .unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].layout().stride_f32(), 7);
        assert_eq!(chunks[0].byte_len(), 2 * 7 * 4);
        assert_eq!(chunks[2].point_count(), 1);
        assert_eq!(chunks[2].byte_len(), 7 * 4);
        for chunk in chunks {
            chunk.recycle(&runtime);
        }
    }

    #[test]
    fn reduces_attributes_across_chunk_boundaries() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        for point in [
            [0.1, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0],
            [1.3, 0.0, 0.0, 8.0, 0.0, 0.0, 1.0],
            [2.1, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0],
        ] {
            builder.push_point(point).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let xyz_chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
        let attribute_chunks = upload_spatial_tensor_attribute_chunks(
            &runtime,
            &tensor,
            AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS,
        )
        .unwrap();
        let voxel =
            downsample_voxel_centroid_aoso_chunks(&runtime, &xyz_chunks, [0.0; 3], 1.0).unwrap();

        let average = reduce_voxel_attributes_aoso_chunks(
            &runtime,
            &attribute_chunks,
            &voxel.segments,
            AoSoAAttributeAggregation::Average,
        )
        .unwrap();
        assert_eq!(average.len(), 3);
        assert_eq!(average.layout(), AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS);
        assert_eq!(
            average.as_slice(),
            &[
                0.1, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.2, 0.0, 0.0, 6.0, 0.0, 0.5, 0.5, 2.1, 0.0,
                0.0, 6.0, 1.0, 0.0, 0.0,
            ]
        );

        let first = reduce_voxel_attributes_aoso_chunks(
            &runtime,
            &attribute_chunks,
            &voxel.segments,
            AoSoAAttributeAggregation::First,
        )
        .unwrap();
        assert_eq!(&first.as_slice()[7..14], &[1.1, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0]);

        voxel.recycle(&runtime);
        for chunk in xyz_chunks {
            chunk.recycle(&runtime);
        }
        for chunk in attribute_chunks {
            chunk.recycle(&runtime);
        }
    }

    #[test]
    fn retained_aoso_positions_match_existing_gpu_normals() {
        let mut builder = PointCloudBuilder::xyz();
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for row in 0..3 {
            for column in 0..3 {
                let point = [column as f32 * 0.1, row as f32 * 0.1, 0.0];
                builder.push_point(point).unwrap();
                x.push(point[0]);
                y.push(point[1]);
                z.push(point[2]);
            }
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 4).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
        let voxel =
            downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [0.0; 3], 10.0).unwrap();
        let neighbors = (0..9_u32).cycle().take(9 * 9).collect::<Vec<_>>();

        let retained =
            estimate_normals_aoso_gpu(&runtime, &voxel.positions, &neighbors, 9).unwrap();
        let actual = retained.readback(&runtime).unwrap();
        let expected = crate::estimate_normals_gpu(&runtime, &x, &y, &z, &neighbors, 9).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.normal[2].abs() - expected.normal[2].abs()).abs() < 1e-6);
            assert!((actual.curvature - expected.curvature).abs() < 1e-6);
        }

        retained.recycle(&runtime);
        voxel.recycle(&runtime);
        for chunk in chunks {
            chunk.recycle(&runtime);
        }
    }

    #[test]
    fn builds_sparse_radius_grid_for_negative_chunked_positions() {
        let mut builder = PointCloudBuilder::xyz();
        for point in
            [[-0.6, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [1.1, 0.0, 0.0]]
        {
            builder.push_point(point).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 2).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
        let voxel =
            downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [-1.0; 3], 10.0).unwrap();

        let grid = build_radius_grid_aoso_gpu(&runtime, &voxel.positions, 0.5).unwrap();
        assert_eq!(grid.radius(), 0.5);
        let segments = grid.segments().to_voxel_segments(&runtime).unwrap();
        assert_eq!(segments.keys, vec![(-2, 0, 0), (-1, 0, 0), (0, 0, 0), (2, 0, 0)]);
        assert_eq!(segments.cell_counts, vec![1, 1, 2, 1]);
        assert_eq!(segments.point_indices, vec![0, 1, 2, 3, 4]);

        voxel.recycle(&runtime);
        for chunk in chunks {
            chunk.recycle(&runtime);
        }
    }

    #[test]
    fn sparse_radius_normals_match_existing_grid_path() {
        let mut builder = PointCloudBuilder::xyz();
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for row in 0..12 {
            for column in 0..12 {
                let point = [column as f32 * 0.1 - 0.55, row as f32 * 0.1 - 0.55, 0.0];
                builder.push_point(point).unwrap();
                x.push(point[0]);
                y.push(point[1]);
                z.push(point[2]);
            }
        }
        let cloud = builder.build().unwrap();
        let tensor = SpatialTensor::new(&cloud, 31).unwrap();
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
        let voxel =
            downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [-1.0; 3], 100.0).unwrap();
        let grid = build_radius_grid_aoso_gpu(&runtime, &voxel.positions, 0.25).unwrap();
        let retained =
            estimate_normals_radius_grid_aoso_gpu(&runtime, &voxel.positions, &grid).unwrap();
        let actual = retained.readback(&runtime).unwrap();
        let expected = crate::estimate_normals_grid_gpu(&runtime, &x, &y, &z, 0.25).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.normal[2].abs() - expected.normal[2].abs()).abs() < 1e-5);
            assert!((actual.curvature - expected.curvature).abs() < 1e-5);
        }
        retained.recycle(&runtime);
        voxel.recycle(&runtime);
        for chunk in chunks {
            chunk.recycle(&runtime);
        }
    }
}
