use super::*;
use spatialrust_math::{Isometry3, Quat, Vec3};

fn axes(points: &[[f32; 3]]) -> [Vec<f32>; 3] {
    std::array::from_fn(|a| points.iter().map(|p| p[a]).collect())
}
fn tree(points: &[[f32; 3]]) -> KdTree {
    let c = axes(points);
    KdTree::from_slices(&c[0], &c[1], &c[2])
}
fn words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    runtime.wait_idle();
    rx.recv().unwrap().unwrap();
    let map = slice.get_mapped_range();
    let data = bytemuck::cast_slice(&map).to_vec();
    drop(map);
    staging.unmap();
    data
}
fn upload(
    runtime: &WgpuRuntime,
    source: &[[f32; 3]],
    target: &KdTree,
    radius: f32,
    capacity: u32,
) -> GpuGicpInput {
    let c = axes(source);
    GpuGicpInput::upload_with_gpu_grid_covariances(
        runtime,
        (&c[0], &c[1], &c[2]),
        target,
        radius,
        capacity,
        0.001,
    )
    .unwrap()
}

#[test]
fn grid_plans_and_shader_adapter_bound_counts_uniforms_dispatch_and_capacity() {
    assert_eq!(std::mem::size_of::<Params>(), 64);
    assert_eq!(
        grid_plan(65537, 65554, 0.25, 128, 0.001, &wgpu::Limits::default()).unwrap(),
        (65537 + 65554 + 1) * 32
    );
    for (s, t) in
        [(0, 1), (1, 0), (u64::MAX, 1), (1, u64::MAX), ((1 << 24) + 1, 1), (1, (1 << 24) + 1)]
    {
        assert!(grid_plan(s, t, 0.25, 128, 0.001, &wgpu::Limits::default()).is_err());
    }
    for r in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e20, 1e-22] {
        assert!(grid_plan(1, 1, r, 128, 0.001, &wgpu::Limits::default()).is_err());
    }
    for e in [0.0, 1e-5, 1.1, f32::NAN] {
        assert!(grid_plan(1, 1, 0.25, 128, e, &wgpu::Limits::default()).is_err());
    }
    assert!(grid_plan(1, 1, 0.25, 0, 0.001, &wgpu::Limits::default()).is_err());
    for case in 0..5 {
        let mut l = wgpu::Limits::default();
        match case {
            0 => l.max_compute_invocations_per_workgroup = 255,
            1 => l.max_compute_workgroup_size_x = 255,
            2 => l.max_compute_workgroups_per_dimension = 27,
            3 => l.max_storage_buffer_binding_size = 255,
            _ => l.max_uniform_buffer_binding_size = 95,
        };
        assert!(grid_plan(1, 1, 0.25, 128, 0.001, &l).is_err());
    }
    let shader = shader_source();
    let old = crate::kernels::covariances_grid::COV_GRID_WGSL;
    let begin = old.find("fn rotate(").unwrap();
    let end = old.find("fn cell_coord(").unwrap();
    assert!(shader.contains(&old[begin..end]));
    assert!(!shader.contains("out_cov"));
    assert!(!shader.contains("xs["));
}

#[test]
fn resident_grid_covariances_preserve_all_legacy_f32_words_and_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut checked = 0;
    for n in [1, 2, 3, 63, 64, 65, 257, 1025] {
        for scale in [1e-4_f32, 1.0, 1e4] {
            for radius in [0.075, 0.25, 0.5] {
                let points: Vec<_> = (0..n)
                    .map(|i| {
                        let j = i / 2;
                        [
                            (j % 17) as f32 * 0.125 * scale,
                            (j / 17) as f32 * 0.125 * scale,
                            (0.3 * (j % 17) as f32 * 0.125 - 0.2 * (j / 17) as f32 * 0.125 + 2.0)
                                * scale,
                        ]
                    })
                    .collect();
                let target: Vec<_> = points.iter().skip(n / 3).copied().collect();
                let t = tree(&target);
                let input = upload(&runtime, &points, &t, radius * scale, n as u32);
                let actual = words(&runtime, &input.covariances);
                assert!(actual[..8].iter().all(|v| *v == 0));
                let mut cells = 0;
                let mut expected = vec![];
                for cloud in [&points, &target] {
                    let c = axes(cloud);
                    let (old, _) = crate::estimate_plane_covariances_grid_gpu_with_receipt(
                        &runtime,
                        &c[0],
                        &c[1],
                        &c[2],
                        radius * scale,
                        0.001,
                    )
                    .unwrap();
                    let dims = spatialrust_search::grid_bounds(&c[0], &c[1], &c[2], radius * scale)
                        .unwrap()
                        .1;
                    cells += dims.into_iter().map(u64::from).product::<u64>();
                    expected.extend(old);
                }
                assert_eq!(
                    input.upload_transfers().host_to_device_bytes(),
                    20 * points.len() as u64
                        + 36 * target.len() as u64
                        + 32 * t.layout().nodes().len() as u64
                        + 4 * (cells + 2)
                );
                assert_eq!(input.upload_transfers().device_to_host_bytes(), 0);
                for (i, c) in expected.iter().enumerate() {
                    let row = &actual[8 + 8 * i..16 + 8 * i];
                    assert_eq!(
                        &row[..6],
                        &c.map(f32::to_bits),
                        "n={n} scale={scale} radius={radius} i={i}"
                    );
                    assert_eq!(&row[6..], &[0, 0]);
                    checked += 1;
                }
            }
        }
    }
    eprintln!(
        "GPU_RESIDENT_GRID_COVARIANCE matched {checked} legacy rows ({} covariance words) exactly",
        checked * 6
    );
}

#[test]
fn grid_capacity_and_numeric_errors_zero_both_sets_and_all_equations() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let duplicate = vec![[0.0; 3]; 65];
    let single = vec![[0.0; 3]];
    for second in [false, true] {
        let (s, t) = if second { (&single, &duplicate) } else { (&duplicate, &single) };
        let t = tree(t);
        let bad = upload(&runtime, s, &t, 0.5, 64);
        let w = words(&runtime, &bad.covariances);
        assert_eq!(w[0], 8);
        assert!(w[1..].iter().all(|v| *v == 0));
        let out = bad.equations_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
        let w = words(&runtime, out.buffer());
        assert_eq!(w[0], 16);
        assert!(w[1..].iter().all(|v| *v == 0));
        assert!(out.download(&runtime).is_err());
        let good = upload(&runtime, s, &t, 0.5, 65);
        assert!(words(&runtime, &good.covariances)[..8].iter().all(|v| *v == 0));
    }
    let huge = vec![[2e38_f32; 3]; 3];
    let bad = upload(&runtime, &huge, &tree(&huge), 0.5, 3);
    let w = words(&runtime, &bad.covariances);
    assert_eq!(w[0], 4);
    assert!(w[1..].iter().all(|v| *v == 0));
    let t = tree(&single);
    assert!(GpuGicpInput::upload_with_gpu_grid_covariances(
        &runtime,
        (&[0.0], &[], &[0.0]),
        &t,
        0.5,
        1,
        0.001
    )
    .is_err());
    assert!(GpuGicpInput::upload_with_gpu_grid_covariances(
        &runtime,
        (&[], &[], &[]),
        &t,
        0.5,
        1,
        0.001
    )
    .is_err());
    assert!(GpuGicpInput::upload_with_gpu_grid_covariances(
        &runtime,
        (&[f32::NAN], &[0.0], &[0.0]),
        &t,
        0.5,
        1,
        0.001
    )
    .is_err());
    let sparse = [[0.0; 3], [10000.0; 3]];
    let c = axes(&sparse);
    assert!(GpuGicpInput::upload_with_gpu_grid_covariances(
        &runtime,
        (&c[0], &c[1], &c[2]),
        &t,
        0.01,
        2,
        0.001
    )
    .is_err());
}

#[test]
fn grid_inclusive_boundary_sparse_epsilon_and_malformed_indices_are_checked() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let points = [
        [0.0; 3],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [f32::from_bits(0.5_f32.to_bits() + 1), 0.0, 0.0],
    ];
    let t = tree(&points);
    let good = upload(&runtime, &points, &t, 0.5, 3);
    let w = words(&runtime, &good.covariances);
    assert!(w[..8].iter().all(|v| *v == 0));
    // Legacy sweep/rebuild retains f32 orthogonality rounding even on an axial plane.
    for (word, expected) in w[8..14].iter().zip([1.0_f32, 1.0, 0.001, 0.0, 0.0, 0.0]) {
        assert!((f32::from_bits(*word) - expected).abs() < 1e-6);
    }
    let bad = upload(&runtime, &points, &t, 0.5, 2);
    assert_eq!(words(&runtime, &bad.covariances)[0], 8);
    let t = tree(&[[0.0; 3]]);
    for epsilon in [1e-4, 0.1, 1.0] {
        let good = GpuGicpInput::upload_with_gpu_grid_covariances(
            &runtime,
            (&[0.0], &[0.0], &[0.0]),
            &t,
            0.5,
            u32::MAX,
            epsilon,
        )
        .unwrap();
        let w = words(&runtime, &good.covariances);
        assert_eq!(&w[8..14], &[epsilon, epsilon, epsilon, 0.0, 0.0, 0.0].map(f32::to_bits));
    }
    let points = [[0.0; 3], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]];
    let t = tree(&points);
    for case in 0..2 {
        let positions = GpuIcpInput::upload(&runtime, &points, &t).unwrap();
        let params = Params {
            origin: [0.0; 4],
            dims: [1, 1, 1, 3],
            inv_cell: 1.0,
            radius_sq: 1.0,
            epsilon: 0.001,
            capacity: 3,
            offset: 0,
            total: 6,
            padding: [0; 2],
        };
        let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sorted = if case == 0 { [u32::MAX, 1, 2] } else { [0, 1, 2] };
        let starts = if case == 0 { [0u32, 3] } else { [0u32, 4] };
        let aux = [bytemuck::cast_slice(&sorted), bytemuck::cast_slice(&starts)].map(|data| {
            runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        });
        let covariances = runtime.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 7 * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cache = runtime
            .gicp_grid_covariance_pipelines
            .get_or_init(|| GridPipelines::new(runtime.device()));
        let entries: Vec<_> =
            [&uniform, positions.source.buffer().unwrap(), &aux[0], &aux[1], &covariances]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
        let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cache.layout,
            entries: &entries,
        });
        let mut encoder = runtime.device().create_command_encoder(&Default::default());
        for pipeline in [&cache.build, &cache.finalize] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        runtime.queue().submit(Some(encoder.finish()));
        let w = words(&runtime, &covariances);
        assert_eq!(w[0], 1);
        assert!(w[1..].iter().all(|v| *v == 0));
        let transfers = positions.upload_transfers();
        let input = GpuGicpInput {
            positions,
            covariances,
            weight_scale: 6000.0,
            covariance_offset: 1,
            transfers,
        };
        let output = input.equations_resident(&runtime, Isometry3::identity(), 1.0).unwrap();
        let w = words(&runtime, output.buffer());
        assert_eq!(w[0], 16);
        assert!(w[1..].iter().all(|v| *v == 0));
        assert!(output.download(&runtime).is_err());
    }
}

#[test]
fn grid_equations_preserve_physical_weights_owned_outputs_and_concurrent_leases() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let points: Vec<_> = (0..65)
        .map(|i| [(i % 5) as f32 * 0.125, (i / 5 % 5) as f32 * 0.125, (i / 25) as f32 * 0.125])
        .collect();
    let t = tree(&points);
    let input = upload(&runtime, &points, &t, 0.25, 65);
    let c = words(&runtime, &input.covariances);
    let c: Vec<[f32; 6]> =
        c[8..].chunks_exact(8).map(|r| std::array::from_fn(|i| f32::from_bits(r[i]))).collect();
    let old = GpuGicpInput::upload(&runtime, &points, &c[..65], &t, &c[65..]).unwrap();
    let pose = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.01, -0.005, 0.002));
    let result = input.equations_resident(&runtime, pose, 0.25).unwrap();
    let actual = result.download(&runtime).unwrap();
    let expected =
        old.equations_resident(&runtime, pose, 0.25).unwrap().download(&runtime).unwrap();
    assert_eq!(actual.squared_error_sum, expected.squared_error_sum);
    for a in 0..6 {
        assert!(
            (actual.gradient[a] * actual.weight_scale
                - expected.gradient[a] * expected.weight_scale)
                .abs()
                < 2e-4 * (1.0 + (expected.gradient[a] * expected.weight_scale).abs())
        );
        for b in 0..6 {
            assert!(
                (actual.hessian[a][b] * actual.weight_scale
                    - expected.hessian[a][b] * expected.weight_scale)
                    .abs()
                    < 2e-4 * (1.0 + (expected.hessian[a][b] * expected.weight_scale).abs())
            );
        }
    }
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..3 {
                        let next = upload(&runtime, &points, &t, 0.25, 65);
                        assert_eq!(
                            next.equations_resident(&runtime, pose, 0.25)
                                .unwrap()
                                .download(&runtime)
                                .unwrap(),
                            actual
                        );
                    }
                })
            })
            .collect();
        for job in jobs {
            job.join().unwrap();
        }
    });
    assert!(input.equations_resident(&other, pose, 0.25).is_err());
    drop(input);
    assert_eq!(result.download(&runtime).unwrap(), actual);
    assert!(result.download(&other).is_err());
}
