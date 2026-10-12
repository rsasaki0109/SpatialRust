#![allow(clippy::needless_range_loop)]
use super::*;
use spatialrust_math::{solve_linear_system, LeastSquaresResult, Quat, TransformPoint, Vec3};
use spatialrust_search::NearestNeighborIndex;

fn tree(points: &[[f32; 3]]) -> KdTree {
    let columns: [Vec<_>; 3] = std::array::from_fn(|i| points.iter().map(|p| p[i]).collect());
    KdTree::from_slices(&columns[0], &columns[1], &columns[2])
}
fn matrix(c: [f32; 6]) -> [[f64; 3]; 3] {
    let [a, b, c, d, e, f] = c.map(f64::from);
    [[a, d, e], [d, b, f], [e, f, c]]
}
fn inverse(c: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for j in 0..3 {
        let b = (0..3).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
        let LeastSquaresResult::Solved(column) =
            solve_linear_system(c.iter().map(|r| r.to_vec()).collect(), b)
        else {
            panic!("reference covariance singular")
        };
        for i in 0..3 {
            result[i][j] = column[i];
        }
    }
    result
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
fn gicp_limits_covariance_spd_and_layout_are_checked() {
    assert_eq!(std::mem::size_of::<Params>(), 96);
    let limits = wgpu::Limits::default();
    assert_eq!(plan(65537, 131074, &limits).unwrap(), 256);
    assert!(plan(u64::MAX, 1, &limits).is_err());
    assert!(plan(1, u64::MAX, &limits).is_err());
    for case in 0..6 {
        let mut limited = limits.clone();
        match case {
            0 => limited.max_storage_buffers_per_shader_stage = 7,
            1 => limited.max_compute_workgroup_storage_size = 1023,
            2 => limited.max_uniform_buffer_binding_size = 95,
            3 => limited.max_bindings_per_bind_group = 8,
            4 => limited.max_compute_workgroups_per_dimension = 27,
            _ => limited.max_buffer_size = 255,
        }
        assert!(plan(1, 2, &limited).is_err());
    }
    assert!((inverse_trace(&[1.0, 2.0, 4.0, 0.0, 0.0, 0.0]).unwrap() - 1.75).abs() < 1e-12);
    for covariance in
        [[0.0; 6], [1.0, 1.0, 1.0, 2.0, 0.0, 0.0], [f32::NAN, 1.0, 1.0, 0.0, 0.0, 0.0]]
    {
        assert!(inverse_trace(&covariance).is_err());
    }
}

#[test]
fn device_normal_equations_match_independent_f64_inverse_jacobian_and_residuals() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut maximum = 0.0_f64;
    for n in [1, 3, 63, 64, 65, 257, 4097] {
        let source: Vec<_> = (0..n)
            .map(|i| [(i % 17) as f32 * 0.25, (i / 17 % 17) as f32 * 0.25, (i / 289) as f32 * 0.25])
            .collect();
        let target: Vec<_> =
            source.iter().map(|p| [p[0] + 0.03125, p[1] - 0.015625, p[2] + 0.0078125]).collect();
        let sc: Vec<_> =
            (0..n).map(|i| [0.25 + (i % 5) as f32 * 0.01, 0.5, 0.8, 0.015, -0.01, 0.04]).collect();
        let tc: Vec<_> =
            (0..n).map(|i| [1.0 + (i % 7) as f32 * 0.01, 0.7, 0.5, 0.05, 0.03, -0.02]).collect();
        let tree = tree(&target);
        let input = GpuGicpInput::upload(&runtime, &source, &sc, &tree, &tc).unwrap();
        let pose = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.007),
            Vec3::new(0.01, -0.005, 0.002),
        );
        let actual =
            input.equations_resident(&runtime, pose, 0.25).unwrap().download(&runtime).unwrap();
        assert_eq!(actual.correspondences as usize, n);
        assert_eq!(actual.fitness_correspondences as usize, n);
        let scale = 0.25 / actual.rotation_step_scale;
        let r = pose.rotation().to_mat3().m.map(|row| row.map(f64::from));
        let mut h = [[0.0; 6]; 6];
        let mut g = [0.0; 6];
        let mut ah = [[0.0; 6]; 6];
        let mut ag = [0.0; 6];
        let mut fitness = 0.0;
        for i in 0..n {
            let p = pose.transform_point(Vec3::new(source[i][0], source[i][1], source[i][2]));
            let j = tree.nearest_one(p.x, p.y, p.z).unwrap().index;
            let p = [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
            let d: [f64; 3] = std::array::from_fn(|a| (p[a] - actual.anchor[a]) / scale);
            let e: [f64; 3] = std::array::from_fn(|a| (p[a] - f64::from(target[j][a])) / 0.25);
            let s = matrix(sc[i]);
            let mut c = matrix(tc[j]);
            for a in 0..3 {
                for b in 0..3 {
                    for x in 0..3 {
                        for y in 0..3 {
                            c[a][b] += r[a][x] * s[x][y] * r[b][y];
                        }
                    }
                }
            }
            let m = inverse(c).map(|row| row.map(|v| v / actual.weight_scale));
            let jac = [
                [0.0, d[2], -d[1], 1.0, 0.0, 0.0],
                [-d[2], 0.0, d[0], 0.0, 1.0, 0.0],
                [d[1], -d[0], 0.0, 0.0, 0.0, 1.0],
            ];
            for a in 0..6 {
                for x in 0..3 {
                    for y in 0..3 {
                        let contribution = jac[x][a] * m[x][y] * e[y];
                        g[a] += contribution;
                        ag[a] += contribution.abs();
                        for b in 0..6 {
                            let contribution = jac[x][a] * m[x][y] * jac[y][b];
                            h[a][b] += contribution;
                            ah[a][b] += contribution.abs();
                        }
                    }
                }
            }
            fitness += e.iter().map(|v| v * v * 0.0625).sum::<f64>();
        }
        for a in 0..6 {
            maximum = maximum.max((actual.gradient[a] - g[a]).abs() / (1e-4 + ag[a]));
            assert!(
                (actual.gradient[a] - g[a]).abs() <= 2e-5 * (1e-4 + ag[a]),
                "n={n} gradient {a}"
            );
            for b in 0..6 {
                maximum = maximum.max((actual.hessian[a][b] - h[a][b]).abs() / (1e-4 + ah[a][b]));
                assert!(
                    (actual.hessian[a][b] - h[a][b]).abs() <= 2e-5 * (1e-4 + ah[a][b]),
                    "n={n} Hessian {a},{b}"
                );
            }
        }
        assert!((actual.squared_error_sum - fitness).abs() < 1e-6 * (1.0 + fitness));
        assert_eq!(actual.transfers.device_to_host_bytes(), 256);
        assert_eq!(
            input.upload_transfers().host_to_device_bytes(),
            (n * 112 + tree.layout().nodes().len() * 32) as u64
        );
        let u = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let step = actual.world_step(u).unwrap();
        let p = [1.0, 2.0, 3.0];
        let physical = [
            step[1] * p[2] - step[2] * p[1] + step[3],
            step[2] * p[0] - step[0] * p[2] + step[4],
            step[0] * p[1] - step[1] * p[0] + step[5],
        ];
        let d: [f64; 3] = std::array::from_fn(|i| (p[i] - actual.anchor[i]) / scale);
        let normalized = [
            u[1] * d[2] - u[2] * d[1] + u[3],
            u[2] * d[0] - u[0] * d[2] + u[4],
            u[0] * d[1] - u[1] * d[0] + u[5],
        ];
        for i in 0..3 {
            assert!((physical[i] - normalized[i] * 0.25).abs() < 1e-12);
        }
    }
    eprintln!("GPU_GICP_REDUCTION maximum normalized f64 equation error {maximum}");
}

#[test]
fn invalid_covariances_empty_gates_and_conditioning_errors_zero_payload() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = tree(&[[0.0; 3]]);
    let identity = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    assert!(GpuGicpInput::upload(&runtime, &[[0.0; 3]], &[], &target, &[identity]).is_err());
    assert!(GpuGicpInput::upload(
        &runtime,
        &[[0.0; 3]],
        &[[1.0, 1.0, 1.0, 2.0, 0.0, 0.0]],
        &target,
        &[identity]
    )
    .is_err());
    let empty = GpuGicpInput::upload(&runtime, &[], &[], &target, &[identity]).unwrap();
    assert_eq!(
        empty
            .equations_resident(&runtime, Isometry3::identity(), 0.25)
            .unwrap()
            .download(&runtime)
            .unwrap()
            .transfers
            .device_to_host_bytes(),
        0
    );
    let input =
        GpuGicpInput::upload(&runtime, &[[100.0; 3]], &[identity], &target, &[identity]).unwrap();
    for distance in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(input.equations_resident(&runtime, Isometry3::identity(), distance).is_err());
    }
    let none = input.equations_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
    assert!(read_words(&runtime, none.buffer()).iter().all(|&v| v == 0));
    assert_eq!(none.download(&runtime).unwrap().correspondences, 0);
    let bad = [1e-8, 1.0, 1.0, 0.0, 0.0, 0.0];
    let input = GpuGicpInput::upload(&runtime, &[[0.0; 3]], &[bad], &target, &[bad]).unwrap();
    let bad = input.equations_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
    let words = read_words(&runtime, bad.buffer());
    assert_eq!(words[0], 8);
    assert!(words[1..].iter().all(|&v| v == 0));
    assert!(bad.download(&runtime).is_err());
    let input =
        GpuGicpInput::upload(&runtime, &[[f32::MAX, 0.0, 0.0]], &[identity], &target, &[identity])
            .unwrap();
    let bad = input
        .equations_resident(
            &runtime,
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(f32::MAX, 0.0, 0.0)),
            0.25,
        )
        .unwrap();
    let words = read_words(&runtime, bad.buffer());
    assert_ne!(words[0] & 1, 0);
    assert!(words[1..].iter().all(|&v| v == 0));
    assert!(bad.download(&runtime).is_err());
    let edge =
        GpuGicpInput::upload(&runtime, &[[0.25, 0.0, 0.0]], &[identity], &target, &[identity])
            .unwrap()
            .equations_resident(&runtime, Isometry3::identity(), 0.25)
            .unwrap()
            .download(&runtime)
            .unwrap();
    assert_eq!(edge.correspondences, 1);
    assert_eq!(edge.squared_error_sum, 0.0625);
}

#[test]
fn owned_equations_survive_input_drop_and_concurrent_scratch_reuse() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let points: Vec<_> = (0..65)
        .map(|i| [(i % 5) as f32 * 0.25, (i / 5 % 5) as f32 * 0.25, (i / 25) as f32 * 0.25])
        .collect();
    let cov = vec![[0.001, 1.0, 1.0, 0.0, 0.0, 0.0]; 65];
    let input = GpuGicpInput::upload(&runtime, &points, &cov, &tree(&points), &cov).unwrap();
    assert!(input.equations_resident(&other, Isometry3::identity(), 0.25).is_err());
    let retained = input.equations_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
    let expected = retained.download(&runtime).unwrap();
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..2 {
                        assert_eq!(
                            input
                                .equations_resident(&runtime, Isometry3::identity(), 0.25)
                                .unwrap()
                                .download(&runtime)
                                .unwrap(),
                            expected
                        );
                    }
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }
    });
    drop(input);
    assert_eq!(retained.download(&runtime).unwrap(), expected);
    assert!(retained.download(&other).is_err());
}
