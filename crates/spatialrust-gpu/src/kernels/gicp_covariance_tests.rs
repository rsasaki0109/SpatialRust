#![allow(clippy::needless_range_loop)]
use super::*;
use spatialrust_math::{symmetric_eigen3, Isometry3, Mat3, Quat, Vec3};
use spatialrust_search::NearestNeighborIndex;

fn tree(points: &[[f32; 3]]) -> KdTree {
    let c: [Vec<_>; 3] = std::array::from_fn(|a| points.iter().map(|p| p[a]).collect());
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
    runtime.device().poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}
fn covariances(runtime: &WgpuRuntime, input: &GpuGicpInput) -> Vec<[f32; 6]> {
    let w = words(runtime, &input.covariances);
    assert!(w[..8].iter().all(|&v| v == 0));
    w[8..].chunks_exact(8).map(|row| std::array::from_fn(|i| f32::from_bits(row[i]))).collect()
}
fn reference(
    points: &[[f32; 3]],
    tree: &KdTree,
    i: usize,
    k: usize,
    epsilon: f64,
) -> ([f64; 6], Mat3<f64>) {
    let neighbors = tree.nearest_k(points[i][0], points[i][1], points[i][2], k);
    if neighbors.len() < 2 {
        return ([epsilon, 1.0, 1.0, 0.0, 0.0, 0.0], Mat3::from_rows([0.0; 3], [0.0; 3], [0.0; 3]));
    }
    let anchor = points[i].map(f64::from);
    let mean: [f64; 3] = std::array::from_fn(|a| {
        neighbors.iter().map(|p| f64::from(points[p.index][a]) - anchor[a]).sum::<f64>()
            / neighbors.len() as f64
    });
    let mut c = [[0.0_f64; 3]; 3];
    for neighbor in &neighbors {
        let d: [f64; 3] =
            std::array::from_fn(|a| f64::from(points[neighbor.index][a]) - anchor[a] - mean[a]);
        for a in 0..3 {
            for b in 0..3 {
                c[a][b] += d[a] * d[b] / (neighbors.len() - 1) as f64;
            }
        }
    }
    let raw = Mat3::from_rows(c[0], c[1], c[2]);
    let eigen = symmetric_eigen3(raw);
    let normal: [f64; 3] = std::array::from_fn(|a| eigen.eigenvectors.m[a][0]);
    let reg = std::array::from_fn::<_, 3, _>(|a| {
        std::array::from_fn::<_, 3, _>(|b| {
            if a == b {
                1.0 + (epsilon - 1.0) * normal[a] * normal[b]
            } else {
                (epsilon - 1.0) * normal[a] * normal[b]
            }
        })
    });
    ([reg[0][0], reg[1][1], reg[2][2], reg[0][1], reg[0][2], reg[1][2]], raw)
}
fn hash(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    (x & 0xffffff) as f32 / 2097152.0
}

fn matrix(c: [f32; 6]) -> Mat3<f64> {
    let [xx, yy, zz, xy, xz, yz] = c.map(f64::from);
    Mat3::from_rows([xx, xy, xz], [xy, yy, yz], [xz, yz, zz])
}

#[test]
fn resident_covariance_plans_bound_counts_arity_epsilon_and_dispatch() {
    assert_eq!(std::mem::size_of::<Params>(), 16);
    let limits = wgpu::Limits::default();
    assert_eq!(
        covariance_plan(65537, 65554, 20, 0.001, &limits).unwrap(),
        (65537 + 65554 + 1) * 32
    );
    for (s, t) in [(0, 1), (1, 0), (u64::MAX, 1), (1, u64::MAX)] {
        assert!(covariance_plan(s, t, 20, 0.001, &limits).is_err());
    }
    for k in [0, 1, 2, 65, u32::MAX] {
        assert!(covariance_plan(1, 1, k, 0.001, &limits).is_err());
    }
    for e in [0.0, 1e-5, -0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(covariance_plan(1, 1, 3, e, &limits).is_err());
    }
    for case in 0..5 {
        let mut l = limits.clone();
        match case {
            0 => l.max_buffer_size = 255,
            1 => l.max_compute_workgroups_per_dimension = 27,
            2 => l.max_storage_buffers_per_shader_stage = 7,
            3 => l.max_uniform_buffer_binding_size = 95,
            _ => l.max_storage_buffer_binding_size = 255,
        };
        assert!(covariance_plan(1, 1, 3, 0.001, &l).is_err());
    }
    let mut l = limits;
    l.max_compute_workgroups_per_dimension = 64;
    assert!(covariance_plan(2049, 2049, 3, 0.001, &l).is_err());
}

#[test]
fn resident_knn_covariances_match_centered_f64_planes_and_spectral_invariants() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut maximum = 0.0_f64;
    let mut checked = 0;
    for n in [1, 2, 3, 63, 64, 65, 257, 4097] {
        for scale in [1e-4_f32, 1.0, 1e4] {
            let points: Vec<_> = (0..n)
                .map(|i| {
                    let a = hash(i as u32 + 1);
                    let b = hash(i as u32 + 0x10203);
                    [(a + 4.0) * scale, (b - 3.0) * scale, (0.3 * a - 0.2 * b + 2.0) * scale]
                })
                .collect();
            let tree = tree(&points);
            for k in [3, 20, 64] {
                let input =
                    GpuGicpInput::upload_with_gpu_covariances(&runtime, &tree, &tree, k, 0.001)
                        .unwrap();
                let cov = covariances(&runtime, &input);
                assert_eq!(cov.len(), 2 * n);
                assert_eq!(
                    input.upload_transfers().host_to_device_bytes(),
                    (64 * n + 64 * tree.layout().nodes().len()) as u64
                );
                assert_eq!(input.upload_transfers().device_to_host_bytes(), 0);
                for i in 0..n {
                    assert_eq!(cov[i], cov[n + i]);
                    let eigen = symmetric_eigen3(matrix(cov[i]));
                    for (actual, expected) in eigen.eigenvalues.iter().zip([0.001, 1.0, 1.0]) {
                        assert!((actual - expected).abs() < 2e-6, "n={n} scale={scale}");
                    }
                    let (expected, raw) =
                        reference(&points, &tree, i, k as usize, f64::from(0.001_f32));
                    let spectrum = symmetric_eigen3(raw).eigenvalues;
                    let unique = spectrum[1] - spectrum[0] > (spectrum[2].abs() * 1e-4).max(1e-11);
                    // A repeated/small spectrum has no stable unique normal.
                    // Check its Rayleigh energy and regularized SPD spectrum,
                    // rather than demanding one arbitrary eigenvector's matrix.
                    let normal: [f64; 3] = std::array::from_fn(|a| eigen.eigenvectors.m[a][0]);
                    let mut energy = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            energy += normal[a] * raw.m[a][b] * normal[b];
                        }
                    }
                    assert!(
                        energy <= spectrum[0] + 1e-11 + 3e-5 * spectrum[2].abs(),
                        "Rayleigh n={n} scale={scale} k={k} point={i}"
                    );
                    if unique || n == 1 {
                        for a in 0..6 {
                            let error = (f64::from(cov[i][a]) - expected[a]).abs();
                            maximum = maximum.max(error);
                            assert!(
                                error < 3e-4,
                                "n={n} scale={scale} k={k} point={i} component={a} error={error}"
                            );
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    eprintln!("GPU_RESIDENT_COVARIANCE checked={checked}; maximum f64 covariance-component error={maximum}");
}

#[test]
fn resident_covariances_feed_equations_with_no_payload_readback_and_keep_owned_outputs() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    let points: Vec<_> = (0..65)
        .map(|i| [(i % 5) as f32 * 0.25, (i / 5 % 5) as f32 * 0.25, (i / 25) as f32 * 0.25])
        .collect();
    let target = tree(&points);
    let input =
        GpuGicpInput::upload_with_gpu_covariances(&runtime, &target, &target, 20, 0.001).unwrap();
    let cov = covariances(&runtime, &input);
    let uploaded =
        GpuGicpInput::upload(&runtime, &points, &cov[..65], &target, &cov[65..]).unwrap();
    let pose = Isometry3::new(
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.005),
        Vec3::new(0.01, -0.005, 0.002),
    );
    let retained = input.equations_resident(&runtime, pose, 0.25).unwrap();
    let actual = retained.download(&runtime).unwrap();
    let expected =
        uploaded.equations_resident(&runtime, pose, 0.25).unwrap().download(&runtime).unwrap();
    assert_eq!(actual.squared_error_sum, expected.squared_error_sum);
    assert_eq!(actual.correspondences, 65);
    assert_eq!(actual.transfers.device_to_host_bytes(), 256);
    for a in 0..6 {
        assert!(
            (actual.gradient[a] * actual.weight_scale
                - expected.gradient[a] * expected.weight_scale)
                .abs()
                <= 2e-4 * (1.0 + (expected.gradient[a] * expected.weight_scale).abs())
        );
        for b in 0..6 {
            assert!(
                (actual.hessian[a][b] * actual.weight_scale
                    - expected.hessian[a][b] * expected.weight_scale)
                    .abs()
                    <= 2e-4 * (1.0 + (expected.hessian[a][b] * expected.weight_scale).abs())
            );
        }
    }
    assert!(input.equations_resident(&other, pose, 0.25).is_err());
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..2 {
                        assert_eq!(
                            input
                                .equations_resident(&runtime, pose, 0.25)
                                .unwrap()
                                .download(&runtime)
                                .unwrap(),
                            actual
                        );
                    }
                })
            })
            .collect();
        for c in callers {
            c.join().unwrap();
        }
    });
    drop(input);
    assert_eq!(retained.download(&runtime).unwrap(), actual);
    assert!(retained.download(&other).is_err());
}

#[test]
fn resident_covariance_errors_zero_all_payload_and_propagate_through_equations() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let origin = tree(&[[0.0; 3]]);
    let target = tree(&[[0.0; 3], [1.0, 0.0, 0.0]]);
    let tiny = tree(&[[0.0; 3], [1e-22, 0.0, 0.0]]);
    let input =
        GpuGicpInput::upload_with_gpu_covariances(&runtime, &tiny, &target, 3, 0.001).unwrap();
    let cov = words(&runtime, &input.covariances);
    assert_eq!(cov[0], 2);
    assert!(cov[1..].iter().all(|&v| v == 0));
    let result = input.equations_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
    let out = words(&runtime, result.buffer());
    assert_eq!(out[0], 16);
    assert!(out[1..].iter().all(|&v| v == 0));
    assert!(result.download(&runtime).is_err());
    for epsilon in [1e-4, 0.1, 1.0] {
        let input =
            GpuGicpInput::upload_with_gpu_covariances(&runtime, &origin, &target, 3, epsilon)
                .unwrap();
        let c = covariances(&runtime, &input);
        assert_eq!(c[0], [epsilon, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }
    let duplicates = tree(&[[0.0; 3]; 65]);
    let input =
        GpuGicpInput::upload_with_gpu_covariances(&runtime, &duplicates, &target, 64, 0.001)
            .unwrap();
    assert_eq!(covariances(&runtime, &input)[0], [0.001, 1.0, 1.0, 0.0, 0.0, 0.0]);
    assert!(
        GpuGicpInput::upload_with_gpu_covariances(&runtime, &tree(&[]), &target, 3, 0.001).is_err()
    );
    let nan = tree(&[[f32::NAN, 0.0, 0.0]]);
    assert!(GpuGicpInput::upload_with_gpu_covariances(&runtime, &nan, &target, 3, 0.001).is_err());
}

#[test]
fn malformed_resident_neighbor_rows_invalidate_both_covariance_sets() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let source = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let st = tree(&source);
    let tt = tree(&source[..2]);
    for second in [false, true] {
        for case in 0..3 {
            let positions = GpuIcpInput::upload(&runtime, &source, &tt).unwrap();
            let source_tree = GpuKdTree::upload(&runtime, &st).unwrap();
            let sn = source_tree.nearest_k_resident(&runtime, &positions.source, 3).unwrap();
            let tn = positions.tree.nearest_k_resident(&runtime, &positions.target, 3).unwrap();
            let statement = match case {
                0 => "data[0]=0u;",
                1 => "data[2]=0xffffffffu;",
                _ => "data[3]=0x7f800000u;",
            };
            let shader=runtime.device().create_shader_module(wgpu::ShaderModuleDescriptor{label:None,source:wgpu::ShaderSource::Wgsl(format!("@group(0) @binding(0) var<storage,read_write> data:array<u32>; @compute @workgroup_size(1) fn main() {{{statement}}}").into())});
            let pipeline =
                runtime.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let buffer = if second { tn.buffer() } else { sn.buffer() };
            let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            let mut encoder = runtime.device().create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            runtime.queue().submit(Some(encoder.finish()));
            let covariances = runtime.device().create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 6 * 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            encode_covariances(
                &runtime,
                &positions,
                [&sn, &tn],
                &covariances,
                Params { sources: 3, targets: 2, neighbors: 3, epsilon: 0.001 },
            );
            let cov = words(&runtime, &covariances);
            assert_eq!(cov[0], 1);
            assert!(cov[1..].iter().all(|&v| v == 0));
            let transfers = positions.upload_transfers();
            let input = GpuGicpInput {
                positions,
                covariances,
                weight_scale: 6000.0,
                covariance_offset: 1,
                transfers,
            };
            let output = input.equations_resident(&runtime, Isometry3::identity(), 2.0).unwrap();
            let w = words(&runtime, output.buffer());
            assert_eq!(w[0], 16);
            assert!(w[1..].iter().all(|&v| v == 0));
            assert!(output.download(&runtime).is_err());
        }
    }
}
