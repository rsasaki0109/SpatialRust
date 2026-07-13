use super::*;

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
