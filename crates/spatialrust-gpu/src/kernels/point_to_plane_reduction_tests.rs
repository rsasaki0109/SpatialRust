#![allow(clippy::needless_range_loop)]
use super::*;
use spatialrust_math::{solve_linear_system, LeastSquaresResult, Quat, TransformPoint, Vec3};
use spatialrust_search::NearestNeighborIndex;
fn tree(points: &[[f32; 3]]) -> KdTree {
    let c: [Vec<_>; 3] = std::array::from_fn(|i| points.iter().map(|p| p[i]).collect());
    KdTree::from_slices(&c[0], &c[1], &c[2])
}
fn read_words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 256);
    runtime.queue().submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    runtime.device().poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let words = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    words
}

#[test]
fn plane_layout_and_limits_are_checked() {
    assert_eq!(std::mem::size_of::<Params>(), 96);
    let limits = wgpu::Limits::default();
    assert_eq!(plan(65537, 65554, &limits).unwrap(), 256);
    assert!(plan(u64::MAX, 1, &limits).is_err());
    assert!(plan(1, u64::MAX, &limits).is_err());
    for i in 0..7 {
        let mut l = limits.clone();
        match i {
            0 => l.max_storage_buffers_per_shader_stage = 7,
            1 => l.max_compute_workgroup_storage_size = 1023,
            2 => l.max_uniform_buffer_binding_size = 95,
            3 => l.max_bindings_per_bind_group = 8,
            4 => l.max_compute_workgroups_per_dimension = 27,
            5 => l.max_compute_invocations_per_workgroup = 63,
            _ => l.max_buffer_size = 255,
        }
        assert!(plan(1, 1, &l).is_err());
    }
}

#[test]
fn plane_device_equations_match_independent_raw_normal_f64_math() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut maximum = 0.0_f64;
    for n in [1, 3, 63, 64, 65, 257, 4097] {
        for normal_scale in [1e-8_f32, 1.0, 1e8] {
            let source: Vec<_> = (0..n)
                .map(|i| {
                    [
                        4.0 + (i % 17) as f32 * 0.25,
                        -3.0 + (i / 17 % 17) as f32 * 0.25,
                        2.0 + (i / 289) as f32 * 0.25,
                    ]
                })
                .collect();
            let target: Vec<_> = source
                .iter()
                .map(|p| [p[0] + 0.03125, p[1] - 0.015625, p[2] + 0.0078125])
                .collect();
            let normals: Vec<_> = (0..n)
                .map(|i| {
                    if i % 11 == 0 && n > 1 {
                        [0.0; 3]
                    } else {
                        [1.0 + (i % 5) as f32 * 0.25, -0.5, (i % 3) as f32 * 0.125]
                            .map(|v| v * normal_scale)
                    }
                })
                .collect();
            let tree = tree(&target);
            let input = GpuPointToPlaneInput::upload(&runtime, &source, &tree, &normals).unwrap();
            let pose = Isometry3::new(
                Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.007),
                Vec3::new(0.01, -0.005, 0.002),
            );
            let actual =
                input.equations_resident(&runtime, pose, 0.25).unwrap().download(&runtime).unwrap();
            assert_eq!(actual.correspondences as usize, n);
            assert_eq!(actual.fitness_correspondences as usize, n);
            assert_eq!(actual.transfers.device_to_host_bytes(), 256);
            assert_eq!(
                input.upload_transfers().host_to_device_bytes(),
                (16 * n + 48 * n + 32 * tree.layout().nodes().len()) as u64
            );
            let scale = 0.25 / actual.rotation_step_scale;
            let mut h = [[0.0; 6]; 6];
            let mut g = [0.0; 6];
            let mut ah = [[0.0; 6]; 6];
            let mut ag = [0.0; 6];
            let mut error = 0.0;
            for original in &source {
                let p = pose.transform_point(Vec3::new(original[0], original[1], original[2]));
                let j = tree.nearest_one(p.x, p.y, p.z).unwrap().index;
                let p = [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
                let d: [f64; 3] = std::array::from_fn(|k| (p[k] - actual.anchor[k]) / scale);
                let normal = normals[j].map(|v| f64::from(v) / actual.normal_scale);
                let row = [
                    d[1] * normal[2] - d[2] * normal[1],
                    d[2] * normal[0] - d[0] * normal[2],
                    d[0] * normal[1] - d[1] * normal[0],
                    normal[0],
                    normal[1],
                    normal[2],
                ];
                let residual = (0..3)
                    .map(|k| (p[k] - f64::from(target[j][k])) / 0.25 * normal[k])
                    .sum::<f64>();
                for a in 0..6 {
                    g[a] += row[a] * residual;
                    ag[a] += (row[a] * residual).abs();
                    for b in 0..6 {
                        h[a][b] += row[a] * row[b];
                        ah[a][b] += (row[a] * row[b]).abs();
                    }
                }
                error += residual * residual;
            }
            for a in 0..6 {
                let e = (actual.gradient[a] - g[a]).abs() / (1e-4 + ag[a]);
                maximum = maximum.max(e);
                assert!(e <= 5e-5, "n={n} gradient={a}: {e}");
                for b in 0..6 {
                    let e = (actual.hessian[a][b] - h[a][b]).abs() / (1e-4 + ah[a][b]);
                    maximum = maximum.max(e);
                    assert!(e <= 5e-5, "n={n} H={a},{b}: {e}");
                }
            }
            let error = error * 0.0625 * actual.normal_scale.powi(2);
            assert!(
                (actual.squared_error_sum - error).abs()
                    <= 5e-5 * (1e-10 * actual.normal_scale.powi(2) + error)
            );
        }
    }
    eprintln!("GPU_PLANE_MATH maximum normalized component error={maximum}");
}

#[test]
fn plane_normalized_step_converts_to_original_world_linearized_basis() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut source = vec![];
    let mut normals = vec![];
    for i in 0..97 {
        let a = (i % 11) as f32 * 0.125;
        let b = (i / 11) as f32 * 0.125;
        source.extend([
            [4.0 + a, -3.0 + b, 2.0],
            [4.0 + a, -3.0, 2.0 + b],
            [4.0, -3.0 + a, 2.0 + b],
        ]);
        normals.extend([[0.0, 0.0, 2.0], [0.0, -3.0, 0.0], [0.5, 0.0, 0.0]]);
    }
    // A nonzero rotation at an off-origin anchor exercises the cross-anchor
    // translation conversion; pure translation would not detect that error.
    let known = [0.0005_f64, -0.0003, 0.0007, 0.005, -0.007, 0.009];
    let target: Vec<_> = source
        .iter()
        .map(|p| {
            let [x, y, z] = p.map(f64::from);
            [
                (x + known[1] * z - known[2] * y + known[3]) as f32,
                (y + known[2] * x - known[0] * z + known[4]) as f32,
                (z + known[0] * y - known[1] * x + known[5]) as f32,
            ]
        })
        .collect();
    let tree = tree(&target);
    let input = GpuPointToPlaneInput::upload(&runtime, &source, &tree, &normals).unwrap();
    let eq = input
        .equations_resident(&runtime, Isometry3::identity(), 0.1)
        .unwrap()
        .download(&runtime)
        .unwrap();
    let LeastSquaresResult::Solved(u) = solve_linear_system(
        eq.hessian.iter().map(|r| r.to_vec()).collect(),
        eq.gradient.iter().map(|g| -g).collect(),
    ) else {
        panic!("normalized singular")
    };
    let actual = eq.world_step(u.try_into().unwrap()).unwrap();
    assert!(eq.world_step([f64::NAN; 6]).is_err());
    let mut h = [[0.0; 6]; 6];
    let mut g = [0.0; 6];
    for i in 0..source.len() {
        let p = source[i].map(f64::from);
        let index = tree.nearest_one(source[i][0], source[i][1], source[i][2]).unwrap().index;
        let n = normals[index].map(f64::from);
        let j = [
            p[1] * n[2] - p[2] * n[1],
            p[2] * n[0] - p[0] * n[2],
            p[0] * n[1] - p[1] * n[0],
            n[0],
            n[1],
            n[2],
        ];
        let e = (0..3).map(|a| (p[a] - f64::from(target[index][a])) * n[a]).sum::<f64>();
        for a in 0..6 {
            g[a] -= j[a] * e;
            for b in 0..6 {
                h[a][b] += j[a] * j[b];
            }
        }
    }
    let LeastSquaresResult::Solved(expected) =
        solve_linear_system(h.iter().map(|r| r.to_vec()).collect(), g.to_vec())
    else {
        panic!("world singular")
    };
    for i in 0..6 {
        assert!((actual[i] - expected[i]).abs() < 2e-6, "component={i}");
        assert!((expected[i] - known[i]).abs() < 2e-6, "known component={i}");
    }
}

#[test]
fn plane_gates_zero_normals_status_ownership_and_runtime_validation() {
    let runtime = std::sync::Arc::new(WgpuRuntime::new_headless().unwrap());
    let source = [[1.0, 0.0, 0.0], [1.0001, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let target = tree(&[[0.0; 3]]);
    let input =
        GpuPointToPlaneInput::upload(&runtime, &source, &target, &[[2.0, 0.0, 0.0]]).unwrap();
    let a = input.equations_resident(&runtime, Isometry3::identity(), 1.0).unwrap();
    let b = input.equations_resident(&runtime, Isometry3::identity(), 0.5).unwrap();
    drop(input);
    assert_eq!(a.download(&runtime).unwrap().correspondences, 2);
    assert_eq!(a.download(&runtime).unwrap().squared_error_sum, 4.0);
    assert_eq!(b.download(&runtime).unwrap().correspondences, 1);
    let input = GpuPointToPlaneInput::upload(&runtime, &source, &target, &[[0.0; 3]]).unwrap();
    let zero = input
        .equations_resident(&runtime, Isometry3::identity(), 2.0)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(zero.correspondences, 3);
    assert_eq!(zero.hessian, [[0.0; 6]; 6]);
    assert_eq!(zero.squared_error_sum, 0.0);
    for normals in [
        vec![],
        vec![[f32::NAN, 0.0, 0.0]],
        vec![[f32::from_bits(1), 0.0, 0.0]],
        vec![[f32::MAX, 1e-10, 0.0]],
    ] {
        assert!(GpuPointToPlaneInput::upload(&runtime, &source, &target, &normals).is_err());
    }
    for bound in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(input.equations_resident(&runtime, Isometry3::identity(), bound).is_err());
    }
    assert!(input
        .equations_resident(
            &runtime,
            Isometry3::new(Quat::new(0.0, 0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 0.0)),
            1.0
        )
        .is_err());
    let other = WgpuRuntime::new_headless().unwrap();
    assert!(input.equations_resident(&other, Isometry3::identity(), 1.0).is_err());
    assert!(a.download(&other).is_err());
    let empty = GpuPointToPlaneInput::upload(&runtime, &[], &target, &[[1.0, 0.0, 0.0]]).unwrap();
    let output = empty.equations_resident(&runtime, Isometry3::identity(), 1.0).unwrap();
    assert_eq!(output.readback_bytes(), 0);
    assert_eq!(read_words(&runtime, output.buffer()), vec![0; 64]);
    assert_eq!(output.download(&runtime).unwrap().transfers.device_to_host_bytes(), 0);
    let huge = GpuPointToPlaneInput::upload(
        &runtime,
        &[[f32::MAX, 0.0, 0.0]],
        &target,
        &[[1.0, 0.0, 0.0]],
    )
    .unwrap();
    let output = huge
        .equations_resident(
            &runtime,
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(f32::MAX, 0.0, 0.0)),
            1.0,
        )
        .unwrap();
    let words = read_words(&runtime, output.buffer());
    assert_ne!(words[0] & 1, 0);
    assert!(words[1..].iter().all(|&v| v == 0));
    assert!(output.download(&runtime).is_err());
    let input = std::sync::Arc::new(input);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let r = runtime.clone();
            let input = input.clone();
            let zero = zero.clone();
            std::thread::spawn(move || {
                for _ in 0..2 {
                    assert_eq!(
                        input
                            .equations_resident(&r, Isometry3::identity(), 2.0)
                            .unwrap()
                            .download(&r)
                            .unwrap(),
                        zero
                    );
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
