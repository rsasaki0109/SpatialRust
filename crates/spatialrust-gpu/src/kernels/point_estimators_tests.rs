use super::super::{
    estimate_normals_gpu, estimate_normals_grid_gpu, estimate_plane_covariances_grid_gpu, GpuNormal,
};
use super::*;

#[allow(dead_code, unused_imports)]
#[path = "../../tests/fixtures/point_covariances_previous.rs"]
mod previous_covariance;
#[allow(dead_code, unused_imports)]
#[path = "../../tests/fixtures/point_normals_previous.rs"]
mod previous_knn;
#[allow(dead_code, unused_imports)]
#[path = "../../tests/fixtures/point_radius_normals_previous.rs"]
mod previous_radius;

fn fixture(n: usize, pattern: usize) -> [Vec<f32>; 3] {
    let mut xyz: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        let x = (i % 32) as f32 * 0.1;
        let y = (i / 32) as f32 * 0.1;
        let p = match pattern {
            0 => [x, y, 0.0],
            1 => [x, y, x * 0.37 - y * 0.19 + x * y * 0.023],
            2 => [0.0, 0.0, 0.0],
            _ => [i as f32 * 0.4, (i % 2) as f32 * 0.1, (i % 3) as f32 * 0.02],
        };
        for j in 0..3 {
            xyz[j].push(p[j]);
        }
    }
    xyz
}

fn neighbors(n: usize, k: u32) -> Vec<u32> {
    (0..n).flat_map(|i| (0..k as usize).map(move |j| ((i + j * 7) % n) as u32)).collect()
}

fn normal_bits(v: &[GpuNormal]) -> Vec<u32> {
    v.iter()
        .flat_map(|p| {
            [
                p.normal[0].to_bits(),
                p.normal[1].to_bits(),
                p.normal[2].to_bits(),
                p.curvature.to_bits(),
            ]
        })
        .collect()
}

fn current(
    runtime: &WgpuRuntime,
    kind: Kind,
    xyz: &[Vec<f32>; 3],
    indices: &[u32],
    k: u32,
    radius: f32,
    epsilon: f32,
) -> Vec<u32> {
    match kind {
        Kind::KnnNormals => normal_bits(
            &estimate_normals_gpu(runtime, &xyz[0], &xyz[1], &xyz[2], indices, k).unwrap(),
        ),
        Kind::RadiusNormals => normal_bits(
            &estimate_normals_grid_gpu(runtime, &xyz[0], &xyz[1], &xyz[2], radius).unwrap(),
        ),
        Kind::PlaneCovariances => {
            estimate_plane_covariances_grid_gpu(runtime, &xyz[0], &xyz[1], &xyz[2], radius, epsilon)
                .unwrap()
                .into_iter()
                .flat_map(|v| v.map(f32::to_bits))
                .collect()
        }
    }
}

fn previous(
    runtime: &WgpuRuntime,
    kind: Kind,
    xyz: &[Vec<f32>; 3],
    indices: &[u32],
    k: u32,
    radius: f32,
    epsilon: f32,
) -> Vec<u32> {
    match kind {
        Kind::KnnNormals => {
            previous_knn::estimate_normals_gpu(runtime, &xyz[0], &xyz[1], &xyz[2], indices, k)
                .unwrap()
                .into_iter()
                .flat_map(|p| {
                    [
                        p.normal[0].to_bits(),
                        p.normal[1].to_bits(),
                        p.normal[2].to_bits(),
                        p.curvature.to_bits(),
                    ]
                })
                .collect()
        }
        Kind::RadiusNormals => normal_bits(
            &previous_radius::estimate_normals_grid_gpu(runtime, &xyz[0], &xyz[1], &xyz[2], radius)
                .unwrap(),
        ),
        Kind::PlaneCovariances => previous_covariance::estimate_plane_covariances_grid_gpu(
            runtime, &xyz[0], &xyz[1], &xyz[2], radius, epsilon,
        )
        .unwrap()
        .into_iter()
        .flat_map(|v| v.map(f32::to_bits))
        .collect(),
    }
}

#[test]
fn standalone_estimators_match_all_frozen_f32_words() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut cases = 0;
    let mut words = 0;
    for n in [1, 2, 3, 17, 255, 256, 257, 1025] {
        for pattern in 0..4 {
            let xyz = fixture(n, pattern);
            for k in [1, 2, 7, 33] {
                let indices = neighbors(n, k);
                let actual = current(&runtime, Kind::KnnNormals, &xyz, &indices, k, 0.25, 0.001);
                assert_eq!(
                    actual,
                    previous(&runtime, Kind::KnnNormals, &xyz, &indices, k, 0.25, 0.001),
                    "kNN {n}/{pattern}/{k}"
                );
                cases += 1;
                words += actual.len();
            }
            for radius in [0.25, 0.75, f32::INFINITY] {
                let actual = current(&runtime, Kind::RadiusNormals, &xyz, &[], 0, radius, 0.001);
                assert_eq!(
                    actual,
                    previous(&runtime, Kind::RadiusNormals, &xyz, &[], 0, radius, 0.001),
                    "radius {n}/{pattern}/{radius}"
                );
                cases += 1;
                words += actual.len();
                for epsilon in [0.001, 1.0] {
                    let actual =
                        current(&runtime, Kind::PlaneCovariances, &xyz, &[], 0, radius, epsilon);
                    assert_eq!(
                        actual,
                        previous(&runtime, Kind::PlaneCovariances, &xyz, &[], 0, radius, epsilon),
                        "covariance {n}/{pattern}/{radius}/{epsilon}"
                    );
                    cases += 1;
                    words += actual.len();
                }
            }
        }
    }
    assert_eq!(cases, 416);
    assert_eq!(words, 464_896);
}

#[test]
fn point_estimator_plan_bounds_storage_uniform_indices_and_dispatch() {
    let limits = wgpu::Limits::default();
    let kinds = [Kind::KnnNormals, Kind::RadiusNormals, Kind::PlaneCovariances];
    for kind in kinds {
        let aux = if matches!(kind, Kind::KnnNormals) { vec![257 * 7] } else { vec![257, 1025] };
        let valid = plan(kind, 257, &aux, &limits).unwrap();
        assert_eq!(valid.groups, 2);
        assert_eq!(
            valid.output_bytes,
            257 * if matches!(kind, Kind::PlaneCovariances) { 32 } else { 16 }
        );
        for field in 0..12 {
            let mut bad = limits.clone();
            match field {
                0 => bad.max_buffer_size = 15,
                1 => bad.max_storage_buffer_binding_size = 15,
                2 => bad.max_compute_invocations_per_workgroup = 255,
                3 => bad.max_compute_workgroup_size_x = 255,
                4 => bad.max_compute_workgroup_size_y = 0,
                5 => bad.max_compute_workgroup_size_z = 0,
                6 => bad.max_compute_workgroups_per_dimension = 1,
                7 => bad.max_storage_buffers_per_shader_stage = 3 + aux.len() as u32,
                8 => bad.max_bind_groups = 0,
                9 => bad.max_bindings_per_bind_group = 4 + aux.len() as u32,
                10 => bad.max_uniform_buffers_per_shader_stage = 0,
                _ => bad.max_uniform_buffer_binding_size = kind.uniform_bytes() as u32 - 1,
            }
            assert!(plan(kind, 257, &aux, &bad).is_err(), "{kind:?}, limit {field}");
        }
        assert!(plan(kind, 0, &aux, &limits).is_err());
        assert!(plan(kind, u64::MAX, &aux, &limits).is_err());
        assert!(plan(kind, u64::from(u32::MAX) + 1, &aux, &limits).is_err());
        assert!(plan(kind, 257, &[], &limits).is_err());
        assert!(plan(kind, 257, &[u64::MAX], &limits).is_err());
    }
    assert!(plan(Kind::KnnNormals, 3, &[5], &limits).is_err());
    assert!(plan(Kind::RadiusNormals, 3, &[2, 2], &limits).is_err());
    assert!(plan(Kind::PlaneCovariances, 3, &[3, 1], &limits).is_err());
    let mut tiny = limits.clone();
    tiny.max_buffer_size = 32;
    assert!(plan(Kind::RadiusNormals, 1, &[1, 2], &tiny).is_err());
    assert!(plan(Kind::PlaneCovariances, 1, &[1, 2], &tiny).is_err());
    // Largest neighbor binding allowed by the default storage limit.
    let entries = u64::from(limits.max_storage_buffer_binding_size) / 4;
    assert!(plan(Kind::KnnNormals, 1, &[entries], &limits).is_ok());
    assert!(plan(Kind::KnnNormals, 1, &[entries + 1], &limits).is_err());
    let mut wide = limits.clone();
    wide.max_buffer_size = u64::MAX;
    wide.max_storage_buffer_binding_size = u32::MAX;
    wide.max_compute_workgroups_per_dimension = u32::MAX;
    let n = u64::from(u32::MAX) / 32;
    assert!(plan(Kind::PlaneCovariances, n, &[n, 2], &wide).is_ok());
    assert!(plan(Kind::PlaneCovariances, n + 1, &[n + 1, 2], &wide).is_err());
}

#[test]
fn invalid_and_empty_estimator_calls_allocate_nothing() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let xyz = fixture(3, 0);
    for (indices, k) in [(vec![], 0), (vec![0], 1), (vec![0, 1, 3], 1), (vec![u32::MAX; 3], 1)] {
        assert!(estimate_normals_gpu(&runtime, &xyz[0], &xyz[1], &xyz[2], &indices, k).is_err());
    }
    assert!(matches!(
        estimate_normals_gpu(&runtime, &xyz[0], &xyz[1], &[], &[0, 1, 2], 1),
        Err(SpatialError::BufferLengthMismatch { expected: 3, found: 0 })
    ));
    for radius in [0.0, -1.0, f32::NAN] {
        assert!(estimate_normals_grid_gpu(&runtime, &xyz[0], &xyz[1], &xyz[2], radius).is_err());
        assert!(estimate_plane_covariances_grid_gpu(
            &runtime, &xyz[0], &xyz[1], &xyz[2], radius, 0.01
        )
        .is_err());
    }
    assert!(estimate_normals_gpu(&runtime, &[], &[], &[], &[], 0).unwrap().is_empty());
    assert!(estimate_normals_grid_gpu(&runtime, &[], &[], &[], 0.0).unwrap().is_empty());
    assert!(estimate_plane_covariances_grid_gpu(&runtime, &[], &[], &[], f32::NAN, 0.0)
        .unwrap()
        .is_empty());
    // Exceeds the shared CPU grid cap: reject before sorting/upload on every adapter.
    assert!(estimate_normals_grid_gpu(&runtime, &[0.0, 64_000_004.0], &[0.0; 2], &[0.0; 2], 1.0)
        .is_err());
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, 0);
    assert!(runtime.point_estimator_pipelines.get().is_none());
}

#[test]
fn cold_cache_concurrency_moves_retained_results_and_pool_reuse_stay_independent() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let clouds = [fixture(257, 0), fixture(257, 1)];
    let indices = neighbors(257, 7);
    let kinds = [Kind::KnnNormals, Kind::RadiusNormals, Kind::PlaneCovariances];
    let expected: Vec<_> = (0..6)
        .map(|i| previous(&runtime, kinds[i % 3], &clouds[i % 2], &indices, 7, 0.25, 0.001))
        .collect();
    assert!(runtime.point_estimator_pipelines.get().is_none());
    let results = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..6)
            .map(|i| {
                let runtime = &runtime;
                let clouds = &clouds;
                let indices = &indices;
                scope.spawn(move || {
                    current(runtime, kinds[i % 3], &clouds[i % 2], indices, 7, 0.25, 0.001)
                })
            })
            .collect();
        jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
    });
    assert_eq!(results, expected);
    let runtime = Box::new(runtime);
    let cache = runtime.point_estimator_pipelines.get().unwrap();
    assert_eq!(cache.pipelines.iter().filter(|cell| cell.get().is_some()).count(), 3);
    for kind in kinds {
        current(&runtime, kind, &clouds[0], &indices, 7, 0.25, 0.001);
    }
    let before = runtime.scratch_buffer_pool_stats();
    for _ in 0..4 {
        for kind in kinds {
            current(&runtime, kind, &clouds[0], &indices, 7, 0.25, 0.001);
        }
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().allocation_count, before.allocation_count);
    assert!(runtime.scratch_buffer_pool_stats().reuse_count > before.reuse_count);
    // Host results retain their original words after later submissions reuse all buffers.
    assert_eq!(results, expected);
    runtime.set_scratch_buffer_pool_budget(1024);
    assert!(runtime.scratch_buffer_pool_stats().cached_bytes <= 1024);
    runtime.set_scratch_buffer_pool_budget(0);
    for (i, kind) in kinds.into_iter().enumerate() {
        assert_eq!(
            current(&runtime, kind, &clouds[0], &indices, 7, 0.25, 0.001),
            expected[[0, 4, 2][i]]
        );
    }
    assert_eq!(runtime.scratch_buffer_pool_stats().cached_bytes, 0);
    let other = WgpuRuntime::new_headless().unwrap();
    assert!(other.point_estimator_pipelines.get().is_none());
    current(&other, Kind::KnnNormals, &clouds[0], &indices, 7, 0.25, 0.001);
    let cache = other.point_estimator_pipelines.get().unwrap();
    assert!(cache.pipelines[0].get().is_some());
    assert!(cache.pipelines[1..].iter().all(|cell| cell.get().is_none()));
}

#[path = "point_estimators_benchmark.rs"]
mod benchmark;
