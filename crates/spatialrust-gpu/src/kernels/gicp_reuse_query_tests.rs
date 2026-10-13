//! Frozen pre-reuse host planning/submission and WGSL from e7cabc2 (PR #187).
//! Host changes only rename methods and inject the frozen pipeline cache.
use super::*;

fn previous_pipelines(runtime: &WgpuRuntime) -> GicpReductionPipelines {
    let current = GicpReductionPipelines::new(runtime.device());
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("previous-gicp-pipeline-layout"),
        bind_group_layouts: &[&current.layout],
        push_constant_ranges: &[],
    });
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("previous-gicp-reduction"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../tests/fixtures/gicp_reuse_queries_previous.wgsl").into(),
        ),
    });
    let pipelines =
        ["transform_source", "prepare_pairs", "reduce_partials", "finalize"].map(|entry| {
            runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("previous-gicp-pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
    GicpReductionPipelines { layout: current.layout, pipelines }
}

fn previous_plan(source: u64, covariance_count: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let groups = source.div_ceil(64).min(256);
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    for (count, stride) in [(source, 64), (covariance_count, 32), (groups, 240)] {
        if count > u64::from(u32::MAX)
            || count
                .checked_mul(stride)
                .map_or(true, |bytes| bytes > bound || bytes > isize::MAX as u64)
        {
            return Err(invalid());
        }
    }
    if bound < RESULT_BYTES
        || limits.max_compute_workgroups_per_dimension < 28
        || source.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_storage_size < 1024
        || limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_bindings_per_bind_group < 9
        || limits.max_bind_groups == 0
        || limits.max_uniform_buffers_per_shader_stage == 0
        || limits.max_uniform_buffer_binding_size < 96
    {
        return Err(invalid());
    }
    Ok(groups as u32)
}

impl GpuGicpInput {
    fn previous_equations_resident(
        &self,
        cache: &GicpReductionPipelines,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
    ) -> SpatialResult<GpuGicpNormalEquations> {
        let q = transform.rotation();
        let t = transform.translation();
        let norm = f64::from(q.x).hypot(f64::from(q.y)).hypot(f64::from(q.z)).hypot(f64::from(q.w));
        let bound = max_distance * max_distance;
        if !Arc::ptr_eq(&self.positions.identity, &runtime.profiling_identity())
            || !norm.is_finite()
            || (norm - 1.0).abs() > 1e-4
            || !t.x.is_finite()
            || !t.y.is_finite()
            || !t.z.is_finite()
            || !max_distance.is_finite()
            || max_distance <= 0.0
            || !bound.is_finite()
            || bound == 0.0
        {
            return Err(invalid());
        }
        let count = self.source_count();
        let groups = previous_plan(
            u64::from(count),
            u64::from(count) + u64::from(self.positions.target.query_count()),
            &runtime.device().limits(),
        )?;
        let output = GpuGicpNormalEquations {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("gicp-equations"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            source_count: count,
            coordinate_scale: self.positions.scale,
            max_distance,
            weight_scale: self.weight_scale,
            identity: Arc::clone(&self.positions.identity),
        };
        if count == 0 {
            return Ok(output);
        }
        let r = q.to_mat3();
        let params = Params {
            rows: [
                [r.m[0][0], r.m[0][1], r.m[0][2], t.x],
                [r.m[1][0], r.m[1][1], r.m[1][2], t.y],
                [r.m[2][0], r.m[2][1], r.m[2][2], t.z],
            ],
            anchor: self.positions.source_anchor,
            count,
            targets: self.positions.target.query_count(),
            groups,
            unused: self.covariance_offset,
            bound,
            max_distance,
            scale: self.positions.scale,
            weight_scale: self.weight_scale,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gicp-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut scratch = ScratchBuffers::new(runtime);
        let queries = scratch.acquire("gicp-transformed", u64::from(count) * 16);
        let matches = scratch.acquire("gicp-matches", u64::from(count) * 8);
        let pairs = scratch.acquire("gicp-pairs", u64::from(count) * 64);
        let partials = scratch.acquire("gicp-partials", u64::from(groups) * 240);
        let buffers = [
            &uniform,
            self.positions.source.buffer().unwrap(),
            self.positions.target.buffer().unwrap(),
            &queries,
            &matches,
            &self.covariances,
            &pairs,
            &partials,
            &output.buffer,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gicp-bindings"),
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        dispatch(&mut encoder, &cache.pipelines[0], &group, count.div_ceil(64), 1);
        self.positions.tree.encode_nearest(
            runtime,
            &mut encoder,
            &queries,
            count,
            f32::INFINITY,
            &matches,
        )?;
        for (index, x, y) in [(1, count.div_ceil(64), 1), (2, groups, 28), (3, 1, 1)] {
            dispatch(&mut encoder, &cache.pipelines[index], &group, x, y);
        }
        scratch.submit_and_recycle(encoder);
        Ok(output)
    }
}

#[path = "gicp_reuse_query_benchmark.rs"]
mod benchmark;

type Fixture = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 6]>, Vec<[f32; 6]>);

fn fixture(n: usize, profile: u32, covariance: u32, scale: f32) -> Fixture {
    let source: Vec<_> = (0..n)
        .map(|i| {
            let j = if profile == 2 { i / 3 } else { i };
            [
                4.0 + (j % 37) as f32 * 0.125,
                -3.0 + (j / 37 % 37) as f32 * 0.125,
                2.0 + (j / 1369) as f32 * 0.125,
            ]
        })
        .collect();
    let target = if profile == 0 {
        vec![[4.0, -3.0, 2.0]]
    } else {
        let mut points: Vec<_> =
            source.iter().map(|p| [p[0] + 0.03125, p[1] - 0.015625, p[2] + 0.0078125]).collect();
        points.extend((0..17).map(|i| [30.0 + i as f32, 30.0, 30.0]));
        points
    };
    let row = |i: usize, source: bool| {
        match covariance {
            0 => [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            1 => {
                if source {
                    [0.25 + (i % 5) as f32 * 0.01, 0.5, 0.8, 0.015, -0.01, 0.04]
                } else {
                    [1.0 + (i % 7) as f32 * 0.01, 0.7, 0.5, 0.05, 0.03, -0.02]
                }
            }
            2 => [1e-8, 1.0, 1.0, 0.0, 0.0, 0.0],
            _ => panic!("unsupported covariance profile"),
        }
        .map(|v| v * scale)
    };
    let sc = (0..n).map(|i| row(i, true)).collect();
    let tc = (0..target.len()).map(|i| row(i, false)).collect();
    (source, target, sc, tc)
}

fn pose() -> Isometry3<f32> {
    Isometry3::new(
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.007),
        Vec3::new(0.01, -0.005, 0.002),
    )
}

fn poison_scratch(runtime: &WgpuRuntime, count: u32, previous: bool) {
    runtime.clear_scratch_buffer_pool();
    let groups = count.div_ceil(64).min(256);
    let mut scratch = ScratchBuffers::new(runtime);
    let buffers = if previous {
        [
            scratch.acquire("previous-queries", u64::from(count) * 16),
            scratch.acquire("previous-matches", u64::from(count) * 8),
            scratch.acquire("previous-pairs", u64::from(count) * 64),
            scratch.acquire("previous-partials", u64::from(groups) * 240),
        ]
    } else {
        acquire_working_buffers(&mut scratch, count, groups)
    };
    for b in buffers {
        runtime.queue().write_buffer(&b, 0, &vec![0xff; b.size() as usize]);
    }
    scratch.submit_and_recycle(runtime.device().create_command_encoder(&Default::default()));
}

#[test]
fn reused_queries_preserve_frozen_words_for_spd_and_conditioning_errors() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let mut conditions = 0;
    let mut words_checked = 0;
    let mut rejected = 0;
    for n in [0, 1, 3, 63, 64, 65, 257, 4097, 65537] {
        for profile in 0..3 {
            for covariance in 0..3 {
                for scale in [1e-8, 1.0, 1e8] {
                    let (source, target, sc, tc) = fixture(n, profile, covariance, scale);
                    let input =
                        GpuGicpInput::upload(&runtime, &source, &sc, &tree(&target), &tc).unwrap();
                    for distance in [0.001, 0.25, 100.0] {
                        let mut words = [vec![], vec![]];
                        for (old, result_words) in [true, false].into_iter().zip(&mut words) {
                            if n != 0 {
                                poison_scratch(&runtime, n as u32, old);
                            }
                            let before = runtime.scratch_buffer_pool_stats();
                            let result = if old {
                                input.previous_equations_resident(
                                    &previous,
                                    &runtime,
                                    pose(),
                                    distance,
                                )
                            } else {
                                input.equations_resident(&runtime, pose(), distance)
                            }
                            .unwrap();
                            *result_words = read_words(&runtime, result.buffer());
                            assert_eq!(result.download(&runtime).is_err(), result_words[0] != 0);
                            let after = runtime.scratch_buffer_pool_stats();
                            if n != 0 {
                                assert_eq!(after.reuse_count - before.reuse_count, 4);
                                assert_eq!(after.allocation_count, before.allocation_count);
                            }
                        }
                        assert_eq!(words[0],words[1],"n={n} profile={profile} covariance={covariance} scale={scale} distance={distance}");
                        if words[0][0] != 0 {
                            assert_eq!(words[0][0], 8);
                            assert!(words[0][1..].iter().all(|&v| v == 0));
                            rejected += 1;
                        }
                        conditions += 1;
                        words_checked += words[0].len();
                    }
                }
            }
        }
    }
    // Rotating a very anisotropic source can make the combined covariance
    // well-conditioned. Identity poses independently exercise rejection rather
    // than assuming every individually anisotropic row must fail after rotation.
    for n in [1, 65, 65537] {
        for scale in [1e-8, 1.0, 1e8] {
            let (source, target, sc, tc) = fixture(n, 1, 2, scale);
            let input = GpuGicpInput::upload(&runtime, &source, &sc, &tree(&target), &tc).unwrap();
            for distance in [0.25, 100.0] {
                poison_scratch(&runtime, n as u32, true);
                let old = input
                    .previous_equations_resident(
                        &previous,
                        &runtime,
                        Isometry3::identity(),
                        distance,
                    )
                    .unwrap();
                let reference = read_words(&runtime, old.buffer());
                poison_scratch(&runtime, n as u32, false);
                let new =
                    input.equations_resident(&runtime, Isometry3::identity(), distance).unwrap();
                let words = read_words(&runtime, new.buffer());
                assert_eq!(words, reference);
                assert_eq!(words[0], 8);
                assert!(words[1..].iter().all(|&v| v == 0));
                assert!(old.download(&runtime).is_err());
                assert!(new.download(&runtime).is_err());
                rejected += 1;
                conditions += 1;
                words_checked += words.len();
            }
        }
    }
    assert_eq!(rejected, 18);
    eprintln!("GPU_GICP_REUSE_EXACT conditions={conditions} equation_words={words_checked} rejected={rejected}");
}

#[test]
fn reused_query_leases_and_combined_covariance_limits_are_exact() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    // Allows all four large working leases to remain cached for size inspection.
    // Production still uses the unchanged default 64-MiB retention policy.
    runtime.set_scratch_buffer_pool_budget(128 * 1024 * 1024);
    for count in [1, 65, 65537, 1048577] {
        runtime.clear_scratch_buffer_pool();
        let before = runtime.scratch_buffer_pool_stats();
        let groups = plan(count, count + 1, &runtime.device().limits()).unwrap();
        let mut scratch = ScratchBuffers::new(&runtime);
        let buffers = acquire_working_buffers(&mut scratch, count as u32, groups);
        assert_eq!(
            buffers.each_ref().map(|b| b.size()),
            [16 * count, 8 * count, 48 * count, u64::from(groups) * 240]
        );
        for a in 0..4 {
            for b in a + 1..4 {
                assert_ne!(buffers[a], buffers[b]);
            }
        }
        assert_eq!(
            runtime.scratch_buffer_pool_stats().allocation_count - before.allocation_count,
            4
        );
        scratch.submit_and_recycle(runtime.device().create_command_encoder(&Default::default()));
        assert_eq!(
            runtime.scratch_buffer_pool_stats().cached_bytes,
            72 * count + u64::from(groups) * 240
        );
    }
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 49152,
        max_buffer_size: 49152,
        ..wgpu::Limits::default()
    };
    assert_eq!(plan(1024, 1025, &limits).unwrap(), 16);
    assert!(previous_plan(1024, 1025, &limits).is_err());
    assert!(plan(1025, 1026, &limits).is_err());
    assert!(plan(1, 1537, &limits).is_err());
    let limits =
        wgpu::Limits { max_compute_workgroups_per_dimension: 28, ..wgpu::Limits::default() };
    assert!(plan(28 * 64 + 1, 1, &limits).is_err());
    assert!(plan(u64::from(u32::MAX) + 1, 1, &wgpu::Limits::default()).is_err());
    // Equal-sized source/target covariances still occupy 64 bytes per source
    // together; shrinking the pair binding does not remove that separate bound.
    assert!(plan(1024, 2048, &limits_with_small_binding()).is_err());
}

fn limits_with_small_binding() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: 49152,
        max_buffer_size: 49152,
        ..wgpu::Limits::default()
    }
}

#[test]
fn reused_queries_keep_owned_results_across_queued_and_concurrent_poses() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let (source, target, sc, tc) = fixture(4097, 2, 1, 1.0);
    let input = GpuGicpInput::upload(&runtime, &source, &sc, &tree(&target), &tc).unwrap();
    let conditions = [
        (Isometry3::identity(), 0.001),
        (Isometry3::identity(), 0.25),
        (pose(), 0.25),
        (pose(), 100.0),
    ];
    let expected: Vec<_> = conditions
        .iter()
        .map(|&(pose, distance)| {
            let old =
                input.previous_equations_resident(&previous, &runtime, pose, distance).unwrap();
            (read_words(&runtime, old.buffer()), old.download(&runtime).unwrap())
        })
        .collect();
    let queued: Vec<_> = conditions
        .iter()
        .cycle()
        .take(12)
        .map(|&(pose, distance)| input.equations_resident(&runtime, pose, distance).unwrap())
        .collect();
    let concurrent = std::thread::scope(|scope| {
        let handles: Vec<_> = conditions
            .iter()
            .cycle()
            .take(8)
            .map(|&(pose, distance)| {
                let runtime = &runtime;
                let input = &input;
                scope.spawn(move || input.equations_resident(runtime, pose, distance).unwrap())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });
    drop(input);
    for (i, result) in queued.into_iter().chain(concurrent).enumerate() {
        let expected = &expected[i % conditions.len()];
        assert_eq!(read_words(&runtime, result.buffer()), expected.0);
        assert_eq!(result.download(&runtime).unwrap(), expected.1);
    }
}

#[test]
fn reused_queries_preserve_all_resident_covariance_paths_and_overflow_status() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let mut conditions = 0;
    for n in [65, 257, 4097] {
        for duplicates in [false, true] {
            let source: Vec<_> = (0..n)
                .map(|i| {
                    let j = if duplicates { i / 2 } else { i };
                    [(j % 17) as f32 * 0.25, (j / 17 % 17) as f32 * 0.25, (j / 289) as f32 * 0.25]
                })
                .collect();
            let target: Vec<_> = source
                .iter()
                .map(|p| [p[0] + 0.03125, p[1] - 0.015625, p[2] + 0.0078125])
                .collect();
            let st = tree(&source);
            let tt = tree(&target);
            let (x, y, z) = st.layout().positions();
            for mode in 0..5 {
                let input = match mode {
                    0 => GpuGicpInput::upload_with_gpu_covariances(&runtime, &st, &tt, 6, 0.001),
                    1 | 3 => GpuGicpInput::upload_with_gpu_radius_covariances(
                        &runtime,
                        &st,
                        &tt,
                        0.375,
                        if mode == 3 { 1 } else { 64 },
                        0.001,
                    ),
                    _ => GpuGicpInput::upload_with_gpu_grid_covariances(
                        &runtime,
                        (x, y, z),
                        &tt,
                        0.375,
                        if mode == 4 { 1 } else { 64 },
                        0.001,
                    ),
                }
                .unwrap();
                for distance in [0.001, 0.25, 100.0] {
                    let old = input
                        .previous_equations_resident(&previous, &runtime, pose(), distance)
                        .unwrap();
                    let reference = read_words(&runtime, old.buffer());
                    poison_scratch(&runtime, n as u32, false);
                    let new = input.equations_resident(&runtime, pose(), distance).unwrap();
                    let words = read_words(&runtime, new.buffer());
                    assert_eq!(
                        words, reference,
                        "n={n} duplicates={duplicates} mode={mode} distance={distance}"
                    );
                    if mode >= 3 {
                        assert_eq!(words[0], 16);
                        assert!(words[1..].iter().all(|&v| v == 0));
                        assert!(new.download(&runtime).is_err());
                    } else {
                        assert_eq!(words[0], 0);
                        new.download(&runtime).unwrap();
                    }
                    conditions += 1;
                }
            }
        }
    }
    eprintln!("GPU_GICP_REUSE_RESIDENT conditions={conditions}");
}
