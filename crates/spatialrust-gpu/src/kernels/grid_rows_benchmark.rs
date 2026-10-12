use super::*;

fn benchmark_fixture(n: usize, pattern: usize) -> [Vec<f32>; 3] {
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        let p = match pattern {
            0 => [(i % 32) as f32 * 0.125, (i / 32 % 32) as f32 * 0.125, (i / 1024) as f32 * 0.125],
            1 => {
                let x = (i % 32) as f32 * 0.125;
                let y = (i / 32) as f32 * 0.125;
                [x, y, x * 0.137 - y * 0.019]
            }
            _ => {
                let j = i / 8;
                [(j % 32) as f32 * 0.75, (j / 32 % 32) as f32 * 0.75, (j / 1024) as f32 * 0.75]
            }
        };
        for axis in 0..3 {
            xyz[axis].push(p[axis]);
        }
    }
    xyz
}

#[test]
#[ignore = "explicit native paired full-API grid-row comparison"]
fn paired_previous_current_grid_rows() {
    if std::env::var("SPATIALRUST_GRID_ROW_COMPARE").as_deref() == Ok("resident") {
        return resident_grid_rows();
    }
    let runtime = WgpuRuntime::new_headless().unwrap();
    let old_normal =
        pipeline(&runtime, include_str!("../../tests/fixtures/grid_radius_previous.wgsl"));
    let old_cov =
        pipeline(&runtime, include_str!("../../tests/fixtures/grid_covariance_previous.wgsl"));
    eprintln!("backend={}; device_type={}; 3 warmups + 20 alternating pairs; full API includes validation, CPU grid, storage uploads, binding/encoding/submission, mapping wait and result decode; word conversion/checks/drop outside timer", runtime.adapter_info().backend, runtime.adapter_info().device_type);
    for n in [257, 4096, 65536] {
        for pattern in 0..3 {
            let xyz = benchmark_fixture(n, pattern);
            for kind in [Kind::RadiusNormals, Kind::PlaneCovariances] {
                let mut timings: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                let mut warmed = runtime.scratch_buffer_pool_stats();
                for round in 0..23 {
                    let mut outputs: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
                    for offset in 0..2 {
                        let mode = (round + offset) % 2;
                        let start = std::time::Instant::now();
                        let (elapsed, words) = match kind {
                            Kind::RadiusNormals => {
                                let result = if mode == 0 {
                                    previous_normals_grid(
                                        &runtime,
                                        &old_normal,
                                        &xyz[0],
                                        &xyz[1],
                                        &xyz[2],
                                        0.25,
                                    )
                                    .unwrap()
                                } else {
                                    estimate_normals_grid_gpu(
                                        &runtime, &xyz[0], &xyz[1], &xyz[2], 0.25,
                                    )
                                    .unwrap()
                                };
                                (start.elapsed().as_secs_f64() * 1000.0, normal_bits(&result))
                            }
                            Kind::PlaneCovariances => {
                                let result = if mode == 0 {
                                    previous_covariances_grid(
                                        &runtime, &old_cov, &xyz[0], &xyz[1], &xyz[2], 0.25, 0.001,
                                    )
                                    .unwrap()
                                } else {
                                    estimate_plane_covariances_grid_gpu(
                                        &runtime, &xyz[0], &xyz[1], &xyz[2], 0.25, 0.001,
                                    )
                                    .unwrap()
                                };
                                (
                                    start.elapsed().as_secs_f64() * 1000.0,
                                    result.into_iter().flat_map(|v| v.map(f32::to_bits)).collect(),
                                )
                            }
                            Kind::KnnNormals => unreachable!(),
                        };
                        outputs[mode] = words;
                        if round >= 3 {
                            timings[mode].push(elapsed);
                        }
                    }
                    assert_eq!(
                        outputs[0], outputs[1],
                        "{kind:?}, n={n}, pattern={pattern}, round={round}"
                    );
                    if round == 2 {
                        warmed = runtime.scratch_buffer_pool_stats();
                    }
                }
                let after = runtime.scratch_buffer_pool_stats();
                assert_eq!(after.allocation_count, warmed.allocation_count);
                eprintln!("GRID_PAIR {{\"points\":{n},\"pattern\":{pattern},\"kind\":\"{kind:?}\",\"previous_ms\":{:?},\"current_ms\":{:?},\"timed_storage_allocations\":{},\"timed_storage_reuses\":{}}}", timings[0], timings[1], after.allocation_count-warmed.allocation_count, after.reuse_count-warmed.reuse_count);
            }
        }
    }
}

fn resident_grid_rows() {
    let runtime =
        WgpuRuntime::new_headless_with_timestamps(crate::WgpuPowerPreference::HighPerformance)
            .unwrap();
    assert!(runtime.timestamp_queries_enabled());
    eprintln!("backend={}; device_type={}; prepared resident one-pass GPU comparison; 3 warmups + 20 alternating pairs; completion excludes grid/upload/bindings/profiler/query/output readback", runtime.adapter_info().backend, runtime.adapter_info().device_type);
    for n in [257usize, 4096, 65536] {
        for pattern in 0..3 {
            let xyz = benchmark_fixture(n, pattern);
            let (origin, dims, sorted, starts) =
                prepare_grid(&runtime, Kind::RadiusNormals, &xyz[0], &xyz[1], &xyz[2], 0.25)
                    .unwrap();
            let params = PreviousCovUniform {
                origin: [origin[0], origin[1], origin[2], 0.0],
                dims: [dims[0], dims[1], dims[2], n as u32],
                inv_cell: 4.0,
                radius_sq: 0.0625,
                epsilon: 0.001,
                _pad: 0.0,
            };
            let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grid-row-native-uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let inputs: Vec<_> = [
                bytemuck::cast_slice(&xyz[0]),
                bytemuck::cast_slice(&xyz[1]),
                bytemuck::cast_slice(&xyz[2]),
                bytemuck::cast_slice(&sorted),
                bytemuck::cast_slice(&starts),
            ]
            .map(|contents| {
                runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("grid-row-native-input"),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .into_iter()
            .collect();
            for kind in [Kind::RadiusNormals, Kind::PlaneCovariances] {
                let previous = if matches!(kind, Kind::RadiusNormals) {
                    include_str!("../../tests/fixtures/grid_radius_previous.wgsl")
                } else {
                    include_str!("../../tests/fixtures/grid_covariance_previous.wgsl")
                };
                let pipelines = [pipeline(&runtime, previous), pipeline(&runtime, kind.source())];
                let bytes = n * if matches!(kind, Kind::RadiusNormals) { 16 } else { 32 };
                let outputs = [0, 1].map(|_| {
                    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("grid-row-native-output"),
                        contents: &vec![0; bytes],
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    })
                });
                let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("grid-row-native-staging"),
                    size: bytes as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let groups = [0, 1].map(|mode| {
                    let mut entries = vec![wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }];
                    for (i, buffer) in
                        inputs.iter().chain(std::iter::once(&outputs[mode])).enumerate()
                    {
                        entries.push(wgpu::BindGroupEntry {
                            binding: i as u32 + 1,
                            resource: buffer.as_entire_binding(),
                        });
                    }
                    runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("grid-row-native-bindings"),
                        layout: &pipelines[mode].get_bind_group_layout(0),
                        entries: &entries,
                    })
                });
                runtime.wait_idle();
                let mut stage: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                let mut wall: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..23 {
                    let mut words: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
                    for offset in 0..2 {
                        let mode = (round + offset) % 2;
                        let mut profiler = crate::GpuStageProfiler::new(&runtime, 1).unwrap();
                        let mut profiling = Some(&mut profiler);
                        let begin = std::time::Instant::now();
                        let mut encoder =
                            runtime.device().create_command_encoder(&Default::default());
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("grid-row-native-pass"),
                                    timestamp_writes: crate::profiling::timestamp_writes(
                                        &mut profiling,
                                        "grid-row",
                                    ),
                                });
                            pass.set_pipeline(&pipelines[mode]);
                            pass.set_bind_group(0, &groups[mode], &[]);
                            pass.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
                        }
                        runtime.queue().submit(Some(encoder.finish()));
                        runtime.wait_idle();
                        let ms = begin.elapsed().as_secs_f64() * 1000.0;
                        let report = profiler.collect(&runtime).unwrap();
                        assert_eq!(report.stages.len(), 1);
                        assert_eq!(report.device_to_host_bytes, 16);
                        if round >= 3 {
                            stage[mode].push(report.stages[0].elapsed_ns / 1000.0);
                            wall[mode].push(ms);
                        }
                        let mut encoder =
                            runtime.device().create_command_encoder(&Default::default());
                        encoder.copy_buffer_to_buffer(&outputs[mode], 0, &staging, 0, bytes as u64);
                        runtime.queue().submit(Some(encoder.finish()));
                        words[mode] = crate::readback::read_staging_f32(
                            runtime.device(),
                            &staging,
                            bytes / 4,
                        )
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect();
                    }
                    assert_eq!(
                        words[0], words[1],
                        "resident {kind:?}, n={n}, pattern={pattern}, round={round}"
                    );
                }
                eprintln!("GPU_ROW {{\"points\":{n},\"pattern\":{pattern},\"kind\":\"{kind:?}\",\"previous_us\":{:?},\"current_us\":{:?},\"previous_ms\":{:?},\"current_ms\":{:?}}}", stage[0], stage[1], wall[0], wall[1]);
            }
        }
    }
}
