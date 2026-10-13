//! Frozen pre-reuse host planning/submission and WGSL from 9543296 (PR #186).
//! Changes to the host reference are limited to naming and cache injection.
use super::*;

fn previous_pipelines(runtime: &WgpuRuntime) -> PointToPlaneReductionPipelines {
    let current = PointToPlaneReductionPipelines::new(runtime.device());
    let layout = runtime.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("previous-plane-pipeline-layout"),
        bind_group_layouts: &[&current.layout],
        push_constant_ranges: &[],
    });
    let shader = runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("previous-plane-reduction"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../tests/fixtures/point_to_plane_reuse_queries_previous.wgsl").into(),
        ),
    });
    let pipelines =
        ["transform_source", "prepare_pairs", "reduce_partials", "finalize"].map(|entry| {
            runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("previous-plane-pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
    PointToPlaneReductionPipelines { layout: current.layout, pipelines }
}

fn previous_plan(source: u64, normal_count: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let groups = source.div_ceil(64).min(256);
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    for (count, stride) in [(source, 32), (normal_count, 16), (groups, 240)] {
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

impl GpuPointToPlaneInput {
    fn previous_equations_resident(
        &self,
        cache: &PointToPlaneReductionPipelines,
        runtime: &WgpuRuntime,
        transform: Isometry3<f32>,
        max_distance: f32,
    ) -> SpatialResult<GpuPointToPlaneNormalEquations> {
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
            u64::from(self.positions.target.query_count()),
            &runtime.device().limits(),
        )?;
        let output = GpuPointToPlaneNormalEquations {
            buffer: runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("point_to_plane-equations"),
                size: RESULT_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            source_count: count,
            coordinate_scale: self.positions.scale,
            max_distance,
            normal_scale: self.normal_scale,
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
            unused: 0,
            bound,
            max_distance,
            scale: self.positions.scale,
            normal_scale: self.normal_scale,
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_to_plane-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut scratch = ScratchBuffers::new(runtime);
        let queries = scratch.acquire("point_to_plane-transformed", u64::from(count) * 16);
        let matches = scratch.acquire("point_to_plane-matches", u64::from(count) * 8);
        let pairs = scratch.acquire("point_to_plane-pairs", u64::from(count) * 32);
        let partials = scratch.acquire("point_to_plane-partials", u64::from(groups) * 240);
        let buffers = [
            &uniform,
            self.positions.source.buffer().unwrap(),
            self.positions.target.buffer().unwrap(),
            &queries,
            &matches,
            &self.normals,
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
            label: Some("point_to_plane-bindings"),
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

#[path = "point_to_plane_reuse_query_benchmark.rs"]
mod benchmark;

type Fixture = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>);

fn fixture(n: usize, profile: u32) -> Fixture {
    let source: Vec<_> = (0..n)
        .map(|i| {
            let i = if profile == 2 { i / 3 } else { i };
            [
                4.0 + (i % 37) as f32 * 0.125,
                -3.0 + (i / 37 % 37) as f32 * 0.125,
                2.0 + (i / 1369) as f32 * 0.125,
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
    let normals = (0..target.len())
        .map(|i| {
            if profile == 3 || (profile == 2 && i % 11 == 0) {
                [0.0; 3]
            } else {
                [1.0 + (i % 5) as f32 * 0.25, -0.5, (i % 3) as f32 * 0.125]
            }
        })
        .collect();
    (source, target, normals)
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
            scratch.acquire("previous-query", u64::from(count) * 16),
            scratch.acquire("previous-match", u64::from(count) * 8),
            scratch.acquire("previous-pair", u64::from(count) * 32),
            scratch.acquire("previous-partial", u64::from(groups) * 240),
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
fn reused_queries_preserve_frozen_equation_words_with_poisoned_scratch() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let mut conditions = 0;
    let mut words_checked = 0;
    for n in [0, 1, 3, 63, 64, 65, 257, 4097, 65537] {
        for profile in 0..4 {
            let (source, target, normals) = fixture(n, profile);
            let tree = tree(&target);
            for scale in [1e-8_f32, 1.0, 1e8] {
                let normals: Vec<_> = normals.iter().map(|row| row.map(|v| v * scale)).collect();
                let input =
                    GpuPointToPlaneInput::upload(&runtime, &source, &tree, &normals).unwrap();
                for distance in [0.001, 0.25, 100.0] {
                    let mut words = [vec![], vec![]];
                    for (old, result_words) in [true, false].into_iter().zip(&mut words) {
                        if n != 0 {
                            poison_scratch(&runtime, n as u32, old);
                        }
                        let before = runtime.scratch_buffer_pool_stats();
                        let result = if old {
                            input.previous_equations_resident(&previous, &runtime, pose(), distance)
                        } else {
                            input.equations_resident(&runtime, pose(), distance)
                        }
                        .unwrap();
                        *result_words = read_words(&runtime, result.buffer());
                        result.download(&runtime).unwrap();
                        let after = runtime.scratch_buffer_pool_stats();
                        if n != 0 {
                            assert_eq!(after.reuse_count - before.reuse_count, 4);
                            assert_eq!(after.allocation_count, before.allocation_count);
                        }
                    }
                    assert_eq!(
                        words[0], words[1],
                        "n={n} profile={profile} scale={scale} distance={distance}"
                    );
                    conditions += 1;
                    words_checked += words[0].len();
                }
            }
        }
    }
    eprintln!("GPU_PLANE_REUSE_EXACT conditions={conditions} equation_words={words_checked}");
}

#[test]
fn reused_queries_use_exact_distinct_leases_and_keep_limit_guards() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for count in [1, 65, 65537, 1048577] {
        runtime.clear_scratch_buffer_pool();
        let before = runtime.scratch_buffer_pool_stats();
        let groups = plan(count, 1, &runtime.device().limits()).unwrap();
        let mut scratch = ScratchBuffers::new(&runtime);
        let buffers = acquire_working_buffers(&mut scratch, count as u32, groups);
        assert_eq!(
            buffers.each_ref().map(|b| b.size()),
            [16 * count, 8 * count, 16 * count, u64::from(groups) * 240]
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
            40 * count + u64::from(groups) * 240
        );
    }
    // The largest per-source binding is now 16 bytes. Partial/result limits,
    // normal bindings and dispatch limits still apply independently.
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: 65536,
        max_buffer_size: 65536,
        ..wgpu::Limits::default()
    };
    assert_eq!(plan(4096, 4096, &limits).unwrap(), 64);
    assert!(previous_plan(4096, 4096, &limits).is_err());
    assert!(plan(4097, 1, &limits).is_err());
    assert!(plan(1, 4097, &limits).is_err());
    let mut limits =
        wgpu::Limits { max_compute_workgroups_per_dimension: 28, ..wgpu::Limits::default() };
    assert!(plan(28 * 64 + 1, 1, &limits).is_err());
    limits.max_compute_workgroups_per_dimension = u32::MAX;
    assert!(plan(u64::from(u32::MAX) + 1, 1, &limits).is_err());
}

#[test]
fn reused_queries_keep_owned_results_across_queued_and_concurrent_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let previous = previous_pipelines(&runtime);
    let (source, target, normals) = fixture(4097, 2);
    let input = GpuPointToPlaneInput::upload(&runtime, &source, &tree(&target), &normals).unwrap();
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
    for (i, output) in queued.into_iter().chain(concurrent).enumerate() {
        let reference = &expected[i % conditions.len()];
        assert_eq!(read_words(&runtime, output.buffer()), reference.0);
        assert_eq!(output.download(&runtime).unwrap(), reference.1);
    }
}
