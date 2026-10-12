use super::*;
use crate::kernels::{point_estimators, GpuCovariance};
use bytemuck::{Pod, Zeroable};

fn pipeline(runtime: &WgpuRuntime, source: &str) -> wgpu::ComputePipeline {
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grid-row-reference"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grid-row-reference"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn row_fixture(n: usize, pattern: usize) -> [Vec<f32>; 3] {
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        let j = (i * 73 + 19) % 512;
        let x = (j % 8) as f32 * 0.5 - 2.0;
        let y = (j / 8 % 8) as f32 * 0.5 - 2.0;
        let z = (j / 64) as f32 * 0.5 - 2.0;
        let p = match pattern {
            0 => [x, y, z],
            1 => [x, y, 0.0],
            2 => [0.0, y, z],
            3 => [x, 0.0, 0.0],
            4 => [0.0, 0.0, 0.0],
            _ => [x, y, z + x * 0.137 - y * 0.019],
        };
        for axis in 0..3 {
            xyz[axis].push(p[axis]);
        }
    }
    xyz
}

#[test]
fn merged_grid_rows_preserve_all_frozen_normal_and_covariance_words() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let old_normal =
        pipeline(&runtime, include_str!("../../tests/fixtures/grid_radius_previous.wgsl"));
    let old_cov =
        pipeline(&runtime, include_str!("../../tests/fixtures/grid_covariance_previous.wgsl"));
    let mut cases = 0;
    let mut words = 0;
    for n in [1, 2, 3, 17, 255, 256, 257, 1025, 4097] {
        for pattern in 0..6 {
            let xyz = row_fixture(n, pattern);
            for radius in [0.5, 1.0, f32::INFINITY] {
                let actual = current(&runtime, Kind::RadiusNormals, &xyz, &[], 0, radius, 0.001);
                let expected = normal_bits(
                    &previous_normals_grid(
                        &runtime,
                        &old_normal,
                        &xyz[0],
                        &xyz[1],
                        &xyz[2],
                        radius,
                    )
                    .unwrap(),
                );
                assert_eq!(actual, expected, "normal n={n}, pattern={pattern}, radius={radius}");
                cases += 1;
                words += actual.len();
                for epsilon in [0.001, 1.0] {
                    let actual =
                        current(&runtime, Kind::PlaneCovariances, &xyz, &[], 0, radius, epsilon);
                    let expected: Vec<_> = previous_covariances_grid(
                        &runtime, &old_cov, &xyz[0], &xyz[1], &xyz[2], radius, epsilon,
                    )
                    .unwrap()
                    .into_iter()
                    .flat_map(|v| v.map(f32::to_bits))
                    .collect();
                    assert_eq!(
                        actual, expected,
                        "covariance n={n}, pattern={pattern}, radius={radius}, epsilon={epsilon}"
                    );
                    cases += 1;
                    words += actual.len();
                }
            }
        }
    }
    eprintln!("GRID_ROWS verified {words} exact frozen f32 words across {cases} cases.");
}

fn trace_source(source: &str, covariance: bool) -> String {
    // Retain the actual shader's coordinate/range/distance traversal. Replace
    // only accumulation with integer order fingerprints and return before math.
    let prefix = source.split("    if (count < 3.0)").next().unwrap();
    assert!(prefix.len() < source.len());
    let output = if covariance { "out_cov" } else { "out_normals" };
    let prefix = prefix
        .replace("var mean = vec3<f32>(0.0, 0.0, 0.0);", "var mean = vec3<u32>(2166136261u, 2654435769u, 0u);")
        .replace("var count = 0.0;", "var count = 0u;")
        .replace("mean = mean + vec3<f32>(xs[j], ys[j], zs[j]);", "mean = vec3<u32>((mean.x ^ j) * 16777619u, ((mean.y << 5u) | (mean.y >> 27u)) ^ j, mean.z + j * 73u + 1u);")
        .replace("count = count + 1.0;", "count = count + 1u;");
    format!("{prefix}    {output}[i] = bitcast<vec4<f32>>(vec4<u32>(mean, count));\n}}\n")
}

#[test]
fn merged_grid_rows_match_independent_ordered_neighbor_fingerprints() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let sources = [
        (Kind::RadiusNormals, include_str!("../../tests/fixtures/grid_radius_previous.wgsl")),
        (Kind::RadiusNormals, crate::kernels::normals_grid::NORMALS_GRID_WGSL),
        (
            Kind::PlaneCovariances,
            include_str!("../../tests/fixtures/grid_covariance_previous.wgsl"),
        ),
        (Kind::PlaneCovariances, crate::kernels::covariances_grid::COV_GRID_WGSL),
    ];
    let probes = sources.map(|(kind, source)| {
        pipeline(&runtime, &trace_source(source, matches!(kind, Kind::PlaneCovariances)))
    });
    let mut words = 0;
    for n in [1, 2, 3, 17, 31, 255, 256, 257, 1025] {
        // Binary coordinates/radii make every distance comparison exact.
        for pattern in 0..5 {
            let xyz = row_fixture(n, pattern);
            for radius in [0.5f32, 1.0] {
                let (origin, dims, sorted, starts) =
                    prepare_grid(&runtime, Kind::RadiusNormals, &xyz[0], &xyz[1], &xyz[2], radius)
                        .unwrap();
                let cells: Vec<[i32; 3]> = (0..n)
                    .map(|i| {
                        std::array::from_fn(|axis| {
                            ((xyz[axis][i] - origin[axis]) / radius).floor() as i32
                        })
                    })
                    .collect();
                // Independent comparison sorting, without production CSR offsets.
                let mut order: Vec<_> = (0..n).collect();
                order.sort_by_key(|&j| (cells[j][2], cells[j][1], cells[j][0], j));
                let expected: Vec<[u32; 4]> = (0..n)
                    .map(|i| {
                        let mut hash = [2166136261u32, 2654435769, 0, 0];
                        for &j in &order {
                            if (0..3).any(|axis| (cells[j][axis] - cells[i][axis]).abs() > 1) {
                                continue;
                            }
                            let dx = xyz[0][j] - xyz[0][i];
                            let dy = xyz[1][j] - xyz[1][i];
                            let dz = xyz[2][j] - xyz[2][i];
                            if dx * dx + dy * dy + dz * dz > radius * radius {
                                continue;
                            }
                            hash[0] = (hash[0] ^ j as u32).wrapping_mul(16777619);
                            hash[1] = hash[1].rotate_left(5) ^ j as u32;
                            hash[2] =
                                hash[2].wrapping_add((j as u32).wrapping_mul(73).wrapping_add(1));
                            hash[3] += 1;
                        }
                        hash
                    })
                    .collect();
                let params = PreviousCovUniform {
                    origin: [origin[0], origin[1], origin[2], 0.0],
                    dims: [dims[0], dims[1], dims[2], n as u32],
                    inv_cell: 1.0 / radius,
                    radius_sq: radius * radius,
                    epsilon: 0.0,
                    _pad: 0.0,
                };
                for (mode, (kind, _)) in sources.iter().enumerate() {
                    let actual = previous_run(
                        &runtime,
                        *kind,
                        n,
                        bytemuck::bytes_of(&params),
                        &[
                            bytemuck::cast_slice(&xyz[0]),
                            bytemuck::cast_slice(&xyz[1]),
                            bytemuck::cast_slice(&xyz[2]),
                            bytemuck::cast_slice(&sorted),
                            bytemuck::cast_slice(&starts),
                        ],
                        &probes[mode],
                        |rows| rows[..n].iter().map(|r| r.map(f32::to_bits)).collect::<Vec<_>>(),
                    )
                    .unwrap();
                    assert_eq!(
                        actual, expected,
                        "trace n={n}, pattern={pattern}, radius={radius}, mode={mode}"
                    );
                    words += n * 4;
                }
            }
        }
    }
    eprintln!("GRID_ROWS verified {words} independent ordered-neighbor fingerprint words.");
}

#[path = "grid_rows_benchmark.rs"]
mod benchmark;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PreviousGridUniform {
    origin: [f32; 4],
    dims: [u32; 4], // dimx, dimy, dimz, point_count
    inv_cell: f32,
    radius_sq: f32,
    _pad0: f32,
    _pad1: f32,
}

fn previous_normals_grid(
    runtime: &WgpuRuntime,
    pipeline: &wgpu::ComputePipeline,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
) -> SpatialResult<Vec<GpuNormal>> {
    let point_count = x.len();
    if y.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: y.len() });
    }
    if z.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: z.len() });
    }
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if radius <= 0.0 || radius.is_nan() {
        return Err(SpatialError::InvalidArgument("grid radius must be positive".to_owned()));
    }
    let (origin, dims, sorted, cell_start) =
        point_estimators::prepare_grid(runtime, Kind::RadiusNormals, x, y, z, radius)?;
    let params = PreviousGridUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / radius,
        radius_sq: radius * radius,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    previous_run(
        runtime,
        Kind::RadiusNormals,
        point_count,
        bytemuck::bytes_of(&params),
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            bytemuck::cast_slice(z),
            bytemuck::cast_slice(&sorted),
            bytemuck::cast_slice(&cell_start),
        ],
        pipeline,
        |rows| {
            rows.iter().map(|r| GpuNormal { normal: [r[0], r[1], r[2]], curvature: r[3] }).collect()
        },
    )
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PreviousCovUniform {
    origin: [f32; 4],
    dims: [u32; 4], // dimx, dimy, dimz, point_count
    inv_cell: f32,
    radius_sq: f32,
    epsilon: f32,
    _pad: f32,
}

fn previous_covariances_grid(
    runtime: &WgpuRuntime,
    pipeline: &wgpu::ComputePipeline,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    radius: f32,
    epsilon: f32,
) -> SpatialResult<Vec<GpuCovariance>> {
    let point_count = x.len();
    if y.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: y.len() });
    }
    if z.len() != point_count {
        return Err(SpatialError::BufferLengthMismatch { expected: point_count, found: z.len() });
    }
    if point_count == 0 {
        return Ok(Vec::new());
    }
    if radius <= 0.0 || radius.is_nan() {
        return Err(SpatialError::InvalidArgument("grid radius must be positive".to_owned()));
    }
    let (origin, dims, sorted, cell_start) =
        point_estimators::prepare_grid(runtime, Kind::PlaneCovariances, x, y, z, radius)?;
    let params = PreviousCovUniform {
        origin: [origin[0], origin[1], origin[2], 0.0],
        dims: [dims[0], dims[1], dims[2], point_count as u32],
        inv_cell: 1.0 / radius,
        radius_sq: radius * radius,
        epsilon,
        _pad: 0.0,
    };
    previous_run(
        runtime,
        Kind::PlaneCovariances,
        point_count,
        bytemuck::bytes_of(&params),
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            bytemuck::cast_slice(z),
            bytemuck::cast_slice(&sorted),
            bytemuck::cast_slice(&cell_start),
        ],
        pipeline,
        |rows| {
            (0..point_count)
                .map(|i| {
                    let a = rows[i];
                    let b = rows[point_count + i];
                    [a[0], a[1], a[2], a[3], b[0], b[1]]
                })
                .collect()
        },
    )
}

fn previous_run<T>(
    runtime: &WgpuRuntime,
    kind: Kind,
    points: usize,
    uniform: &[u8],
    inputs: &[&[u8]],
    pipeline: &wgpu::ComputePipeline,
    decode: impl FnOnce(&[[f32; 4]]) -> Vec<T>,
) -> SpatialResult<Vec<T>> {
    let auxiliary: Vec<_> = inputs[3..].iter().map(|bytes| (bytes.len() / 4) as u64).collect();
    let p = plan(kind, points as u64, &auxiliary, &runtime.device().limits())?;
    debug_assert_eq!(uniform.len() as u64, kind.uniform_bytes());
    debug_assert!(inputs[..3].iter().all(|bytes| bytes.len() == points * 4));
    let mut scratch = ScratchBuffers::new(runtime);
    let buffers: Vec<_> = inputs
        .iter()
        .map(|bytes| {
            let buffer = scratch.acquire("point-estimator-input", bytes.len() as u64);
            runtime.queue().write_buffer(&buffer, 0, bytes);
            buffer
        })
        .collect();
    let output = scratch.acquire("point-estimator-output", p.output_bytes);
    let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("point-estimator-params"),
        contents: uniform,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("point-estimator-readback"),
        size: p.output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut entries =
        vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
    for (i, buffer) in buffers.iter().chain(std::iter::once(&output)).enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: buffer.as_entire_binding(),
        });
    }
    let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("point-estimator-bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("point-estimator"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(p.groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, p.output_bytes);
    scratch.submit_and_recycle(encoder);
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|e| SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {e}")))?;
    let mapped = slice.get_mapped_range();
    let result = decode(bytemuck::cast_slice(&mapped));
    drop(mapped);
    staging.unmap();
    Ok(result)
}
