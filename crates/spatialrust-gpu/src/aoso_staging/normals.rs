use super::voxel::{dispatch_voxel_keys_aoso, empty_storage_buffer};
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
