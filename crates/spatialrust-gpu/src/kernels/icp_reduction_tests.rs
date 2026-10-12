use super::*;
use spatialrust_math::{Quat, TransformPoint, Vec3};

fn tree(points: &[[f32; 3]]) -> KdTree {
    let columns: [Vec<_>; 3] = std::array::from_fn(|i| points.iter().map(|p| p[i]).collect());
    KdTree::from_slices(&columns[0], &columns[1], &columns[2])
}
fn read_words(runtime: &WgpuRuntime, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = runtime.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: RESULT_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = runtime.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, RESULT_BYTES);
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
fn reduction_plan_checks_storage_dispatch_uniform_and_shared_memory() {
    assert_eq!(std::mem::size_of::<Params>(), 112);
    let limits = wgpu::Limits::default();
    assert!(plan(u64::MAX, &limits).is_err());
    assert!(plan(u64::from(u32::MAX) + 1, &limits).is_err());
    assert_eq!(plan(65, &limits).unwrap(), 2);
    assert_eq!(plan(65537, &limits).unwrap(), 256);
    for case in 0..6 {
        let mut restricted = limits.clone();
        match case {
            0 => restricted.max_buffer_size = 175,
            1 => restricted.max_uniform_buffer_binding_size = 111,
            2 => restricted.max_storage_buffers_per_shader_stage = 5,
            3 => restricted.max_compute_workgroup_storage_size = 767,
            4 => restricted.max_compute_workgroup_size_x = 63,
            _ => restricted.max_compute_workgroups_per_dimension = 8,
        }
        assert!(plan(1, &restricted).is_err());
    }
}

#[test]
fn compensated_centroids_covariance_and_fitness_match_independent_f64_math() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [1, 3, 63, 64, 65, 257, 4097, 16385] {
        let source: Vec<_> = (0..n)
            .map(|i| [(i % 17) as f32 * 0.5, (i / 17 % 17) as f32 * 0.5, (i / 289) as f32 * 0.5])
            .collect();
        let target: Vec<_> =
            source.iter().map(|p| [p[0] + 0.03125, p[1] - 0.015625, p[2] + 0.0078125]).collect();
        let target_tree = tree(&target);
        let input = GpuIcpInput::upload(&runtime, &source, &target_tree).unwrap();
        let actual = input
            .moments_resident(&runtime, Isometry3::identity(), 0.25)
            .unwrap()
            .download(&runtime)
            .unwrap();
        assert_eq!(actual.correspondences as usize, n);
        let mean_s: [f64; 3] =
            std::array::from_fn(|a| source.iter().map(|p| f64::from(p[a])).sum::<f64>() / n as f64);
        let mean_t: [f64; 3] =
            std::array::from_fn(|a| target.iter().map(|p| f64::from(p[a])).sum::<f64>() / n as f64);
        for a in 0..3 {
            assert!((actual.source_centroid[a] - mean_s[a]).abs() < 1e-5);
            assert!((actual.target_centroid[a] - mean_t[a]).abs() < 1e-5);
            for b in 0..3 {
                let products: Vec<_> = source
                    .iter()
                    .zip(&target)
                    .map(|(s, t)| (f64::from(s[a]) - mean_s[a]) * (f64::from(t[b]) - mean_t[b]))
                    .collect();
                let expected = products.iter().sum::<f64>();
                let absolute_products = products.iter().map(|v| v.abs()).sum::<f64>();
                assert!(
                    (actual.cross_covariance[a][b] - expected).abs()
                        < 1e-7 + 1e-6 * absolute_products,
                    "n={n} a={a} b={b} actual={} expected={expected}",
                    actual.cross_covariance[a][b]
                );
            }
        }
        let error = source
            .iter()
            .zip(&target)
            .map(|(s, t)| (0..3).map(|a| f64::from(s[a] - t[a]).powi(2)).sum::<f64>())
            .sum::<f64>();
        assert!((actual.squared_error_sum - error).abs() < 1e-7 * (1.0 + error));
        assert_eq!(actual.transfers.device_to_host_bytes(), 176);
        assert_eq!(
            input.upload_transfers().host_to_device_bytes(),
            (n * 48 + target_tree.layout().nodes().len() * 32) as u64
        );
    }
}

#[test]
fn transformed_origins_ownership_and_concurrent_scratch_keep_owned_results() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let other = WgpuRuntime::new_headless().unwrap();
    for origin in [0.0, 1e6] {
        let points: Vec<_> = (0..65)
            .map(|i| {
                [
                    origin + (i % 5) as f32 * 0.5,
                    -origin + (i / 5 % 5) as f32 * 0.5,
                    (i / 25) as f32 * 0.5,
                ]
            })
            .collect();
        let pose = Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.125, -0.25, 0.0625));
        let target: Vec<_> = points
            .iter()
            .map(|p| {
                let q = pose.transform_point(Vec3::new(p[0], p[1], p[2]));
                [q.x, q.y, q.z]
            })
            .collect();
        let input = GpuIcpInput::upload(&runtime, &points, &tree(&target)).unwrap();
        assert!(input.moments_resident(&other, pose, 0.25).is_err());
        let retained = input.moments_resident(&runtime, pose, 0.25).unwrap();
        let expected = retained.download(&runtime).unwrap();
        assert_eq!(expected.correspondences, 65);
        assert_eq!(expected.squared_error_sum, 0.0);
        std::thread::scope(|scope| {
            let callers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        for _ in 0..2 {
                            assert_eq!(
                                input
                                    .moments_resident(&runtime, pose, 0.25)
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
}

#[test]
fn invalid_config_empty_missing_matches_and_transform_overflow_are_explicit() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = tree(&[[0.0; 3]]);
    for source in [vec![[f32::NAN, 0.0, 0.0]], vec![[f32::MAX, 0.0, 0.0], [-f32::MAX, 0.0, 0.0]]] {
        assert!(GpuIcpInput::upload(&runtime, &source, &target).is_err());
    }
    assert!(GpuIcpInput::upload(&runtime, &[[0.0; 3], [1e-22, 0.0, 0.0]], &target).is_err());
    assert!(GpuIcpInput::upload(&runtime, &[[0.0; 3]], &tree(&[])).is_err());
    let empty = GpuIcpInput::upload(&runtime, &[], &target).unwrap();
    assert_eq!(
        empty
            .moments_resident(&runtime, Isometry3::identity(), 0.25)
            .unwrap()
            .download(&runtime)
            .unwrap()
            .transfers
            .device_to_host_bytes(),
        0
    );
    let input = GpuIcpInput::upload(&runtime, &[[100.0; 3]], &target).unwrap();
    for distance in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        assert!(input.moments_resident(&runtime, Isometry3::identity(), distance).is_err());
    }
    assert!(input
        .moments_resident(
            &runtime,
            Isometry3::new(Quat::new(0.0, 0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 0.0)),
            0.25
        )
        .is_err());
    let none = input.moments_resident(&runtime, Isometry3::identity(), 0.25).unwrap();
    let words = read_words(&runtime, none.buffer());
    assert!(words.iter().all(|&v| v == 0));
    assert_eq!(none.download(&runtime).unwrap().correspondences, 0);
    let extreme = GpuIcpInput::upload(&runtime, &[[f32::MAX, 0.0, 0.0]], &target).unwrap();
    let invalid = extreme
        .moments_resident(
            &runtime,
            Isometry3::new(Quat::<f32>::identity(), Vec3::new(f32::MAX, 0.0, 0.0)),
            0.25,
        )
        .unwrap();
    let words = read_words(&runtime, invalid.buffer());
    assert_ne!(words[0] & 1, 0);
    assert!(words[1..].iter().all(|&v| v == 0));
    assert!(invalid.download(&runtime).is_err());
    let edge = GpuIcpInput::upload(&runtime, &[[0.25, 0.0, 0.0]], &target)
        .unwrap()
        .moments_resident(&runtime, Isometry3::identity(), 0.25)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(edge.correspondences, 1);
    assert_eq!(edge.squared_error_sum, 0.0625);
}
