#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
#[path = "gpu_gicp_reduction.rs"]
mod reference;
use spatialrust_core::{HasPositions3, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
use spatialrust_registration::{transform_point_cloud, GicpConfig, GicpRegistration};
use spatialrust_search::{KdTree, NearestNeighborIndex};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_grid_gicp_matches_cpu_poses_stopping_fitness_and_transfer_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut maximum = 0.0_f32;
    let mut compared = 0;
    for n in [17, 63, 64, 65, 257, 1366] {
        for scale in [1e-4, 1.0, 1e4] {
            for radius in [0.075, 0.25, 0.5] {
                for stop in 0..3 {
                    let target = reference::corner(n, scale);
                    let source = reference::shifted(&target, scale);
                    let saved_source = source.clone();
                    let saved_target = target.clone();
                    let mut config =
                        reference::registration(scale, Some(radius * scale), stop).config();
                    config.k_neighbors = 6;
                    let registration = GicpRegistration::new(config);
                    let cpu = registration.align_with_diagnostics(&source, &target).unwrap();
                    let previous =
                        registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
                    let actual = registration
                        .align_with_gpu_resident_grid_covariances(&runtime, &source, &target, 128)
                        .unwrap_or_else(|e| {
                            panic!("n={n} scale={scale} radius={radius} stop={stop}: {e}")
                        });
                    maximum = maximum.max(
                        reference::compare_pose(
                            &source,
                            cpu.transform,
                            actual.result.transform,
                            scale,
                        ) / scale,
                    );
                    reference::compare_pose(
                        &source,
                        previous.result.transform,
                        actual.result.transform,
                        scale,
                    );
                    compared += source.len();
                    assert_eq!(actual.result.iterations, cpu.iterations);
                    assert_eq!(actual.result.converged, cpu.converged);
                    assert_eq!(actual.covariance_batches, 2);
                    assert_eq!(actual.evaluation_batches, actual.result.iterations + 1);
                    assert_eq!(actual.history.len(), actual.result.iterations);
                    let (sx, sy, sz) = source.positions3().unwrap();
                    let (tx, ty, tz) = target.positions3().unwrap();
                    let tt = KdTree::from_slices(tx, ty, tz);
                    let cells = [&source, &target]
                        .iter()
                        .map(|c| {
                            let (x, y, z) = c.positions3().unwrap();
                            spatialrust_search::grid_bounds(x, y, z, radius * scale)
                                .unwrap()
                                .1
                                .into_iter()
                                .map(u64::from)
                                .product::<u64>()
                        })
                        .sum::<u64>();
                    assert_eq!(
                        actual.transfers.host_to_device_bytes(),
                        20 * source.len() as u64
                            + 36 * target.len() as u64
                            + 32 * tt.layout().nodes().len() as u64
                            + 4 * (cells + 2)
                    );
                    assert_eq!(
                        actual.transfers.device_to_host_bytes(),
                        actual.evaluation_batches as u64 * 256
                    );
                    let mut error = 0.0;
                    for i in 0..source.len() {
                        let p =
                            actual.result.transform.transform_point(Vec3::new(sx[i], sy[i], sz[i]));
                        let j = tt.nearest_one(p.x, p.y, p.z).unwrap().index;
                        error += f64::from(p.x - tx[j]).powi(2)
                            + f64::from(p.y - ty[j]).powi(2)
                            + f64::from(p.z - tz[j]).powi(2);
                    }
                    error /= source.len() as f64;
                    assert!(
                        (actual.result.fitness - error).abs()
                            <= 1e-9 * f64::from(scale).powi(2) + 1e-4 * error
                    );
                    assert!(actual.history.iter().all(|row| row.correspondences >= 6
                        && row.evaluated_correspondences >= 6
                        && row.fitness.is_finite()
                        && row.update_magnitude.is_finite()));
                    assert_eq!(source, saved_source);
                    assert_eq!(target, saved_target);
                }
            }
        }
    }
    eprintln!("GPU_RESIDENT_GRID_GICP compared {compared} points in 162 conditions; maximum scale-normalized CPU pose-coordinate error {maximum}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_grid_gicp_rejects_unsupported_settings_and_preserves_concurrent_calls() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = reference::corner(85, 1.0);
    let (x, y, z) = target.positions3().unwrap();
    let mut b = PointCloudBuilder::xyz();
    for i in (0..target.len()).step_by(2) {
        b.push_point([x[i] + 0.015625, y[i], z[i] - 0.0078125]).unwrap();
    }
    for i in 0..7 {
        b.push_point([20.0 + i as f32, 20.0, 20.0]).unwrap();
    }
    let source = b.build().unwrap();
    let registration = reference::registration(1.0, Some(0.25), 0);
    let actual = registration
        .align_with_gpu_resident_grid_covariances(&runtime, &source, &target, 128)
        .unwrap();
    let cpu = registration.align_with_diagnostics(&source, &target).unwrap();
    reference::compare_pose(&source, cpu.transform, actual.result.transform, 1.0);
    assert!(actual.history.iter().all(|row| row.correspondences as usize == source.len() - 7));
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(
                        registration
                            .align_with_gpu_resident_grid_covariances(
                                &runtime, &source, &target, 128
                            )
                            .unwrap(),
                        actual
                    );
                })
            })
            .collect();
        for c in callers {
            c.join().unwrap();
        }
    });
    for config in [
        GicpConfig { k_neighbors: 2, ..Default::default() },
        GicpConfig { epsilon: 1e-5, ..Default::default() },
        GicpConfig { epsilon: 1.1, ..Default::default() },
        GicpConfig { epsilon: f64::NAN, ..Default::default() },
        GicpConfig { max_iterations: 0, ..Default::default() },
        GicpConfig { min_correspondences: 5, ..Default::default() },
        GicpConfig { transformation_epsilon: -1.0, ..Default::default() },
    ] {
        let config = GicpConfig { covariance_radius: Some(0.25), ..config };
        assert!(GicpRegistration::new(config)
            .align_with_gpu_resident_grid_covariances(&runtime, &source, &target, 128)
            .is_err());
    }
    assert!(GicpRegistration::new(GicpConfig::default())
        .align_with_gpu_resident_grid_covariances(&runtime, &source, &target, 128)
        .is_err());
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e20, 1e-22] {
        assert!(GicpRegistration::new(GicpConfig {
            covariance_radius: Some(radius),
            ..Default::default()
        })
        .align_with_gpu_resident_grid_covariances(&runtime, &source, &target, 128)
        .is_err());
    }
    for capacity in [0, 1] {
        assert!(registration
            .align_with_gpu_resident_grid_covariances(&runtime, &source, &target, capacity)
            .is_err());
    }
    let far = transform_point_cloud(
        &target,
        Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
    )
    .unwrap();
    assert!(registration
        .align_with_gpu_resident_grid_covariances(&runtime, &far, &target, 128)
        .unwrap_err()
        .to_string()
        .contains("only 0 correspondences"));
    let mut b = PointCloudBuilder::xyz();
    b.push_point([f32::INFINITY, 0.0, 0.0]).unwrap();
    let bad = b.build().unwrap();
    assert!(registration
        .align_with_gpu_resident_grid_covariances(&runtime, &bad, &target, 128)
        .is_err());
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_grid_gicp_requires_dense_grid_capacity_without_automatic_fallback() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let cloud = reference::corner(1366, 1.0);
    let registration =
        GicpRegistration::new(GicpConfig { covariance_radius: Some(0.01), ..Default::default() });
    assert!(registration
        .align_with_gpu_resident_grid_covariances(&runtime, &cloud, &cloud, 2)
        .unwrap_err()
        .to_string()
        .contains("cells"));
    let actual = registration
        .align_with_gpu_resident_radius_covariances(&runtime, &cloud, &cloud, 2)
        .unwrap();
    assert!(actual.result.converged);
    assert_eq!(actual.result.fitness, 0.0);
}
