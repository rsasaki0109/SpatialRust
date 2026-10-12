use super::voxel::empty_storage_buffer;
use super::*;

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
    build_radius_grid_aoso_gpu_with_profiler(runtime, positions, radius, None)
}

pub(crate) fn build_radius_grid_aoso_gpu_with_profiler(
    runtime: &WgpuRuntime,
    positions: &GpuAoSoXyzBuffer,
    radius: f32,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
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
        let keys = super::voxel::dispatch_voxel_keys_aoso_with_profiler(
            runtime,
            positions.buffer(),
            point_count,
            [0.0; 3],
            1.0 / radius,
            profiler.as_deref_mut(),
        )?;
        crate::kernels::build_voxel_segments_with_profiler(
            runtime,
            &keys,
            point_count,
            point_count.next_power_of_two(),
            profiler,
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
    estimate_normals_radius_grid_aoso_gpu_with_profiler(runtime, positions, grid, None)
}

pub(crate) fn estimate_normals_radius_grid_aoso_gpu_with_profiler(
    runtime: &WgpuRuntime,
    positions: &GpuAoSoXyzBuffer,
    grid: &GpuAoSoRadiusGrid,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
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
            timestamp_writes: crate::profiling::timestamp_writes(&mut profiler, "radius-normals"),
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

// Sparse keys use a different ordering and lookup than dense CSR cell rows.
// Keep this shader independent of dense traversal changes.
fn sparse_grid_normals_shader() -> &'static str {
    include_str!("sparse_radius_normals.wgsl")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "sparse_radius_shader_tests.rs"]
mod shader_tests;
