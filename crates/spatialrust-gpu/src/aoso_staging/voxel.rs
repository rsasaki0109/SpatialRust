use super::*;

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

pub(super) fn empty_storage_buffer(runtime: &WgpuRuntime) -> Buffer {
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

pub(super) fn dispatch_voxel_keys_aoso(
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
