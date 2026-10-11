use super::*;

const PREVIOUS: &str = include_str!("../../tests/fixtures/ransac_plane_previous.wgsl");

fn fixture(points: u32, hypotheses: u32) -> ([Vec<f32>; 3], Vec<[u32; 3]>) {
    let mut axes: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    let seeds = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 1.],
        [2., 2., 0.],
        [4., 4., 0.],
    ];
    for i in 0..points {
        let point = if i < 7 {
            seeds[i as usize]
        } else {
            [
                (i % 19) as f32 / 8. - 1.,
                (i % 23) as f32 / 8. - 1.,
                [
                    0.,
                    0.25,
                    f32::from_bits(0.25_f32.to_bits() - 1),
                    f32::from_bits(0.25_f32.to_bits() + 1),
                    -0.25,
                    -0.5,
                    1.,
                ][i as usize % 7],
            ]
        };
        for a in 0..3 {
            axes[a].push(point[a]);
        }
    }
    let triangles = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3], [0, 1, 1], [0, 5, 6], [2, 1, 0]];
    let hypotheses = (0..hypotheses)
        .map(|h| {
            if h >= 7 && h % 7 == 3 {
                [
                    h.wrapping_mul(1543).wrapping_add(7) % points,
                    h.wrapping_mul(2371).wrapping_add(11) % points,
                    h.wrapping_mul(3571).wrapping_add(13) % points,
                ]
            } else {
                triangles[h as usize % 7].map(|i| i % points)
            }
        })
        .collect();
    (axes, hypotheses)
}

fn previous_pipeline(runtime: &WgpuRuntime, cache: &RansacPlanePipelines) -> wgpu::ComputePipeline {
    let module = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ransac-previous"),
        source: wgpu::ShaderSource::Wgsl(PREVIOUS.into()),
    });
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&cache.layout],
        push_constant_ranges: &[],
    });
    runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ransac-previous"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bindings(
    runtime: &WgpuRuntime,
    cache: &RansacPlanePipelines,
    p: &Plan,
    axes: &[Vec<f32>; 3],
    hypotheses: &[[u32; 3]],
    threshold: f32,
) -> ([wgpu::Buffer; 2], [wgpu::BindGroup; 2]) {
    let device = runtime.device();
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&RansacPlaneUniform {
            point_count: p.point_count,
            hypothesis_count: p.hypothesis_count,
            distance_threshold: threshold,
            groups: p.groups,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let axis_buffers: [wgpu::Buffer; 3] = std::array::from_fn(|a| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&axes[a]),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    let pods: Vec<_> = hypotheses
        .iter()
        .map(|h| RansacHypothesisPod { i0: h[0], i1: h[1], i2: h[2], _pad: 0 })
        .collect();
    let hypothesis_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pods),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let dirty = vec![u32::MAX; (p.output_bytes / 4) as usize];
    let outputs: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&dirty),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    });
    let groups = std::array::from_fn(|i| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cache.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: axis_buffers[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: axis_buffers[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: axis_buffers[2].as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hypothesis_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 5, resource: outputs[i].as_entire_binding() },
            ],
        })
    });
    (outputs, groups)
}

fn read_scores(runtime: &WgpuRuntime, output: &wgpu::Buffer, bytes: u64) -> Vec<GpuPlaneScore> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, bytes);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    runtime.device().poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let scores = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    scores
}

fn check_axes(axes: &[Vec<f32>; 3], threshold: f32, scores: &[GpuPlaneScore]) {
    for (h, score) in scores.iter().enumerate() {
        let axis = match h % 7 {
            0 | 6 => 2,
            1 => 0,
            2 => 1,
            4 | 5 => {
                assert_eq!(score.inlier_count, 0);
                assert_eq!(score.normal, [0.; 3]);
                continue;
            }
            _ => continue,
        };
        let expected = axes[axis].iter().filter(|&&value| value.abs() <= threshold).count() as u32;
        assert_eq!(
            score.inlier_count, expected,
            "axis {axis}, hypothesis {h}, threshold {threshold}"
        );
    }
}

#[test]
fn parallel_scores_match_frozen_words_and_cpu_threshold_boundaries() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cache = RansacPlanePipelines::new(runtime.device());
    let previous = previous_pipeline(&runtime, &cache);
    let mut compared = 0;
    for (n, h, cap) in [
        (1, 7, 1),
        (3, 7, 1),
        (255, 513, 1),
        (256, 513, 1),
        (257, 513, 1),
        (1023, 513, 256),
        (1024, 513, 256),
        (1025, 513, 256),
        (8193, 513, 256),
        (65537, 33, 256),
        (1000003, 7, 256),
    ] {
        let (axes, hypotheses) = fixture(n, h);
        for parallel in [false, true] {
            let mut p = plan(n as usize, h as usize, &runtime.device().limits()).unwrap();
            p.parallel = parallel;
            p.groups = (if parallel { h } else { h.div_ceil(256) }).min(cap);
            let current = cache.pipeline(runtime.device(), parallel);
            for threshold in [0., 0.25, f32::from_bits(0.25_f32.to_bits() + 1), -1., f32::NAN] {
                let (outputs, groups) =
                    bindings(&runtime, &cache, &p, &axes, &hypotheses, threshold);
                let mut encoder = runtime.device().create_command_encoder(&Default::default());
                for (i, pipeline, dispatch) in
                    [(0, &previous, h.div_ceil(256)), (1, current, p.groups)]
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &groups[i], &[]);
                    pass.dispatch_workgroups(dispatch, 1, 1);
                }
                runtime.queue().submit(Some(encoder.finish()));
                let a = read_scores(&runtime, &outputs[0], p.output_bytes);
                let b = read_scores(&runtime, &outputs[1], p.output_bytes);
                assert_eq!(bytemuck::cast_slice::<_,u32>(&a), bytemuck::cast_slice::<_,u32>(&b),
                    "{n} points, {h} hypotheses, parallel{parallel}, cap{cap}, threshold{threshold}");
                if n >= 7 {
                    check_axes(&axes, threshold, &b);
                }
                compared += b.len();
            }
        }
    }
    eprintln!("Compared {compared} complete scores / {} words to frozen GPU; CPU axis counts include exact/adjacent threshold values", compared * 5);
}

#[test]
fn plans_bound_dispatch_and_reject_unsupported_sizes_before_work() {
    let limits = wgpu::Limits::default();
    assert!(!plan(1023, 513, &limits).unwrap().parallel);
    assert!(plan(1024, 513, &limits).unwrap().parallel);
    let mut one_group = limits.clone();
    one_group.max_compute_workgroups_per_dimension = 1;
    assert_eq!(plan(1023, 513, &one_group).unwrap().groups, 1);
    assert_eq!(plan(1024, 513, &one_group).unwrap().groups, 1);
    for (n, h) in [
        (0, 1),
        (1, 0),
        (u32::MAX as usize, 1),
        (1, u32::MAX as usize),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        assert!(plan(n, h, &limits).is_err());
    }
    for field in 0..8 {
        let mut small = limits.clone();
        match field {
            0 => small.max_buffer_size = 19,
            1 => small.max_storage_buffer_binding_size = 19,
            2 => small.max_compute_invocations_per_workgroup = 255,
            3 => small.max_compute_workgroup_size_x = 255,
            4 => small.max_compute_workgroups_per_dimension = 0,
            5 => small.max_storage_buffers_per_shader_stage = 4,
            6 => small.max_uniform_buffer_binding_size = 15,
            _ => small.max_uniform_buffers_per_shader_stage = 0,
        }
        assert!(plan(1, 1, &small).is_err(), "limit {field}");
    }
    let mut small = limits;
    small.max_compute_workgroup_storage_size = 2047;
    assert!(plan(1024, 1, &small).is_err());
    assert!(plan(1023, 1, &small).is_ok());
}

#[test]
fn invalid_inputs_allocate_nothing_and_empty_hypotheses_remain_a_noop() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let before = runtime.buffer_pool().stats();
    for (x, y, z, h) in [
        (&[][..], &[][..], &[][..], &[[0, 0, 0]][..]),
        (&[0.][..], &[][..], &[0.][..], &[[0, 0, 0]][..]),
        (&[0.][..], &[0.][..], &[0.][..], &[[0, u32::MAX, 0]][..]),
    ] {
        assert!(score_ransac_plane_hypotheses_gpu(&runtime, x, y, z, h, 0.25).is_err());
    }
    assert!(score_ransac_plane_hypotheses_gpu(&runtime, &[], &[], &[], &[], f32::NAN)
        .unwrap()
        .is_empty());
    assert_eq!(runtime.buffer_pool().stats(), before);
    assert!(runtime.ransac_plane_pipelines.get().is_none());
}

#[test]
fn cold_pipeline_cache_concurrency_runtime_moves_and_recycled_xyz_are_independent() {
    let boxed = Box::new(WgpuRuntime::new_headless().unwrap());
    let (axes, hypotheses) = fixture(4099, 17);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let scores = score_ransac_plane_hypotheses_gpu(
                        &boxed,
                        &axes[0],
                        &axes[1],
                        &axes[2],
                        &hypotheses,
                        0.25,
                    )
                    .unwrap();
                    let pipeline =
                        boxed.ransac_plane_pipelines.get().unwrap().pipeline(boxed.device(), true);
                    (scores, pipeline as *const wgpu::ComputePipeline as usize)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    assert!(results.iter().all(|(_, p)| *p == results[0].1));
    let runtime = *boxed;
    let retained = &results[0].0;
    for (scores, _) in &results {
        check_axes(&axes, 0.25, scores);
    }
    let before = runtime.buffer_pool().stats();
    for threshold in [f32::NAN, -1., 0., 0.25] {
        let output = score_ransac_plane_hypotheses_gpu(
            &runtime,
            &axes[0],
            &axes[1],
            &axes[2],
            &hypotheses,
            threshold,
        )
        .unwrap();
        check_axes(&axes, threshold, &output);
        check_axes(&axes, 0.25, retained);
    }
    assert_eq!(runtime.buffer_pool().stats().allocation_count, before.allocation_count);
    assert!(runtime.ransac_plane_pipelines.get().unwrap().pipelines[0].get().is_none());
}

#[path = "ransac_plane_benchmark.rs"]
mod benchmark;
