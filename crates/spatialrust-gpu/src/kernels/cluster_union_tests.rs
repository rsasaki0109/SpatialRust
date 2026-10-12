use super::*;
use crate::kernels::euclidean_cluster_roots_gpu_with_receipt;

mod previous {
    use crate::kernels::build_voxel_segments_from_positions_gpu;
    use spatialrust_core::{SpatialError, SpatialResult, TransferDirection, TransferStats};
    use spatialrust_search::{KdTree, RadiusSearchIndex};
    use std::collections::VecDeque;
    include!("../../tests/fixtures/cluster_host_previous.rs");
    include!("../../tests/fixtures/cluster_kdtree_cpu.rs");
}

fn brute_force(xyz: &[Vec<f32>; 3], tolerance: f32) -> Vec<u32> {
    let n = xyz[0].len();
    let mut roots = vec![u32::MAX; n];
    for start in 0..n {
        if roots[start] != u32::MAX {
            continue;
        }
        let mut pending = vec![start];
        roots[start] = start as u32;
        while let Some(i) = pending.pop() {
            for (j, root) in roots.iter_mut().enumerate() {
                if *root != u32::MAX {
                    continue;
                }
                let dx = xyz[0][j] - xyz[0][i];
                let dy = xyz[1][j] - xyz[1][i];
                let dz = xyz[2][j] - xyz[2][i];
                if dx * dx + dy * dy + dz * dz <= tolerance * tolerance {
                    *root = start as u32;
                    pending.push(j);
                }
            }
        }
    }
    roots
}

fn fixture(n: usize, kind: usize) -> [Vec<f32>; 3] {
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    for i in 0..n {
        let j = if kind == 3 { n - 1 - i } else { i };
        let p = match kind {
            0 => [(j % 17) as f32 * 0.5 - 4.0, (j / 17 % 17) as f32 * 0.5, (j / 289) as f32 * 0.5],
            1 => [j as f32 * 0.75, 0.0, 0.0],
            2 => [j as f32 * 2.0, 0.0, 0.0],
            3 => [j as f32 * 0.75, 0.0, 0.0],
            4 => [(j / 7) as f32 * 4.0, 0.0, 0.0],
            _ => [0.0; 3],
        };
        for axis in 0..3 {
            xyz[axis].push(p[axis]);
        }
    }
    xyz
}

#[test]
fn gpu_minimum_roots_match_frozen_cpu_and_independent_graphs() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut words = 0;
    for n in [1_usize, 2, 17, 255, 256, 257, 513] {
        for kind in 0..6 {
            let xyz = fixture(n, kind);
            let expected = brute_force(&xyz, 1.0);
            let (prior, _) = previous::euclidean_cluster_roots_gpu_with_receipt(
                &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0,
            )
            .unwrap();
            assert_eq!(prior, expected);
            for _ in 0..3 {
                let (actual, transfers) = euclidean_cluster_roots_gpu_with_receipt(
                    &runtime, &xyz[0], &xyz[1], &xyz[2], 1.0,
                )
                .unwrap();
                assert_eq!(actual, expected, "n={n}, kind={kind}");
                assert_eq!(transfers.host_to_device_bytes(), n as u64 * 12);
                assert_eq!(transfers.device_to_host_bytes(), n as u64 * 4 + 4);
                words += n;
            }
        }
    }
    eprintln!("GPU_CLUSTER verified {words} minimum-root words against frozen CPU and independent graph traversal.");
}

#[test]
fn long_reversed_and_separated_components_cross_dispatch_boundaries() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [4097_usize, 65537] {
        for kind in [1, 2, 3, 4] {
            let xyz = fixture(n, kind);
            let expected: Vec<_> = (0..n)
                .map(|i| match kind {
                    1 | 3 => 0,
                    2 => i as u32,
                    _ => (i / 7 * 7) as u32,
                })
                .collect();
            let (actual, _) =
                euclidean_cluster_roots_gpu_with_receipt(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0)
                    .unwrap();
            assert_eq!(actual, expected, "n={n}, kind={kind}");
        }
    }
}

#[test]
fn radius_boundaries_and_infinite_radius_keep_minimum_roots() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let next = f32::from_bits(1.0_f32.to_bits() + 1);
    for spacing in [f32::from_bits(1.0_f32.to_bits() - 1), 1.0, next] {
        let xyz = [vec![-1.0, -1.0 + spacing, 8.0], vec![0.0; 3], vec![0.0; 3]];
        for tolerance in [f32::from_bits(1.0_f32.to_bits() - 1), 1.0, next] {
            let (actual, _) = euclidean_cluster_roots_gpu_with_receipt(
                &runtime, &xyz[0], &xyz[1], &xyz[2], tolerance,
            )
            .unwrap();
            assert_eq!(actual, brute_force(&xyz, tolerance));
        }
    }
    let x = [-f32::MAX, 0.0, f32::MAX];
    let (actual, transfers) =
        euclidean_cluster_roots_gpu_with_receipt(&runtime, &x, &x, &x, f32::INFINITY).unwrap();
    assert_eq!(actual, [0; 3]);
    assert_eq!(transfers.host_to_device_bytes(), 0);
    assert_eq!(transfers.device_to_host_bytes(), 12);
    let (finite, _) = euclidean_cluster_roots_gpu_with_receipt(&runtime, &x, &x, &x, 1.0).unwrap();
    assert_eq!(finite, [0, 1, 2]);
}

#[test]
fn invalid_inputs_reject_before_gpu_cache_or_scratch_work() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let before = runtime.scratch_buffer_pool_stats();
    for (x, y, z, radius) in [
        (&[f32::NAN][..], &[0.0][..], &[0.0][..], 1.0),
        (&[0.0], &[0.0], &[f32::INFINITY], 1.0),
        (&[0.0], &[0.0], &[0.0], f32::from_bits(1)),
        (&[0.0], &[], &[0.0], 1.0),
        (&[0.0], &[0.0], &[0.0], 0.0),
    ] {
        assert!(euclidean_cluster_roots_gpu_with_receipt(&runtime, x, y, z, radius).is_err());
    }
    assert!(runtime.cluster_pipelines.get().is_none());
    assert_eq!(runtime.scratch_buffer_pool_stats(), before);
    let (empty, transfers) =
        euclidean_cluster_roots_gpu_with_receipt(&runtime, &[], &[], &[], 1.0).unwrap();
    assert!(empty.is_empty());
    assert_eq!(transfers, TransferStats::default());
}

#[test]
fn plans_enforce_exact_byte_and_device_boundaries() {
    let limits = wgpu::Limits::default();
    for n in [1_u64, 255, 256, 257, 65537] {
        assert!(plan(n, false, &limits).is_ok());
    }
    for n in [0, u64::from(u32::MAX), u64::MAX] {
        assert!(plan(n, false, &limits).is_err());
    }
    for kind in 0..10 {
        let mut small = limits.clone();
        match kind {
            0 => small.max_storage_buffers_per_shader_stage = 6,
            1 => small.max_bindings_per_bind_group = 7,
            2 => small.max_uniform_buffer_binding_size = 15,
            3 => small.max_bind_groups = 0,
            4 => small.max_compute_invocations_per_workgroup = 255,
            5 => small.max_compute_workgroup_size_x = 255,
            6 => small.max_compute_workgroup_size_y = 0,
            7 => small.max_compute_workgroup_size_z = 0,
            8 => small.max_compute_workgroups_per_dimension = 0,
            _ => small.max_uniform_buffers_per_shader_stage = 0,
        }
        assert!(plan(257, false, &small).is_err(), "limit={kind}");
    }
    let mut bound = limits;
    bound.max_buffer_size = 4112;
    bound.max_storage_buffer_binding_size = 4112;
    assert!(plan(257, false, &bound).is_ok());
    bound.max_storage_buffer_binding_size = 4111;
    assert!(plan(257, false, &bound).is_err());
    bound.max_buffer_size = 1028;
    bound.max_storage_buffer_binding_size = 1028;
    assert!(plan(257, true, &bound).is_ok());
    bound.max_storage_buffer_binding_size = 1027;
    assert!(plan(257, true, &bound).is_err());
    bound.max_buffer_size = 16;
    bound.max_storage_buffer_binding_size = 16;
    bound.max_uniform_buffer_binding_size = 16;
    assert!(plan(1, true, &bound).is_ok());
    assert!(plan(1, false, &bound).is_err());
}

#[test]
fn dirty_parent_scratch_and_concurrent_calls_keep_minimum_roots_owned() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let dirty = runtime.scratch_pool.acquire_storage(&runtime, "dirty-parents", 513 * 4);
    runtime.queue().write_buffer(&dirty, 0, bytemuck::cast_slice(&vec![u32::MAX; 513]));
    runtime.scratch_pool.recycle(dirty.size(), dirty);
    let xyz = fixture(513, 5);
    assert_eq!(
        euclidean_cluster_roots_gpu_with_receipt(&runtime, &xyz[0], &xyz[1], &xyz[2], 1.0)
            .unwrap()
            .0,
        vec![0; 513]
    );
    let cache = runtime.cluster_pipelines.get().unwrap() as *const ClusterPipelines;
    std::thread::scope(|scope| {
        for kind in [0_usize, 2, 3, 4] {
            let runtime = &runtime;
            scope.spawn(move || {
                let xyz = fixture(4097, kind);
                let expected: Vec<_> = (0..4097)
                    .map(|i| match kind {
                        2 => i as u32,
                        4 => (i / 7 * 7) as u32,
                        _ => 0,
                    })
                    .collect();
                let first = euclidean_cluster_roots_gpu_with_receipt(
                    runtime, &xyz[0], &xyz[1], &xyz[2], 1.0,
                )
                .unwrap()
                .0;
                let second = euclidean_cluster_roots_gpu_with_receipt(
                    runtime, &xyz[0], &xyz[1], &xyz[2], 1.0,
                )
                .unwrap()
                .0;
                assert_eq!(first, expected);
                assert_eq!(second, expected);
            });
        }
    });
    assert_eq!(runtime.cluster_pipelines.get().unwrap() as *const ClusterPipelines, cache);
    assert!(runtime.scratch_buffer_pool_stats().reuse_count > 0);
}

#[path = "cluster_union_benchmark.rs"]
mod benchmark;
