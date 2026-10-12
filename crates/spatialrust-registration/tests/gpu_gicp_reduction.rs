#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]
use spatialrust_core::{HasPositions3, PointCloud, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
use spatialrust_registration::{transform_point_cloud, GicpConfig, GicpRegistration};

pub fn corner(n: usize, scale: f32) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        let a = (i % 17) as f32 * 0.125 * scale;
        let b = (i / 17) as f32 * 0.125 * scale;
        for p in [
            [a, b, 0.0],
            [a, 0.0, b + 0.0625 * scale],
            [0.0, a + 0.0625 * scale, b + 0.0625 * scale],
        ] {
            builder.push_point(p).unwrap();
        }
    }
    builder.build().unwrap()
}
pub fn shifted(cloud: &PointCloud, scale: f32) -> PointCloud {
    transform_point_cloud(
        cloud,
        Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.002),
            Vec3::new(0.01 * scale, -0.005 * scale, 0.008 * scale),
        ),
    )
    .unwrap()
}
pub fn registration(scale: f32, radius: Option<f32>, stop: usize) -> GicpRegistration {
    GicpRegistration::new(GicpConfig {
        max_iterations: 3,
        max_correspondence_distance: 0.25 * scale,
        k_neighbors: 6,
        initial_guess: Isometry3::new(
            Quat::<f32>::identity(),
            Vec3::new(-0.005 * scale, 0.003 * scale, 0.0),
        ),
        covariance_radius: radius,
        transformation_epsilon: if stop == 1 { scale.max(1.0) as f64 } else { 0.0 },
        fitness_epsilon: if stop == 2 { f64::from(scale).powi(2) } else { 0.0 },
        ..Default::default()
    })
}
pub fn compare_pose(cloud: &PointCloud, a: Isometry3<f32>, b: Isometry3<f32>, scale: f32) -> f32 {
    let (x, y, z) = cloud.positions3().unwrap();
    let mut maximum = 0.0_f32;
    for i in 0..cloud.len() {
        let p = Vec3::new(x[i], y[i], z[i]);
        let u = a.transform_point(p);
        let v = b.transform_point(p);
        let error = (u - v).x.abs().max((u - v).y.abs()).max((u - v).z.abs());
        maximum = maximum.max(error);
        assert!(error <= 1e-4 * scale, "point={i} error={error} scale={scale}");
    }
    maximum
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_gicp_reduction_matches_cpu_poses_covariance_paths_stopping_and_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut maximum = 0.0_f32;
    let mut compared = 0;
    for n in [17, 63, 64, 65, 257, 1366] {
        for scale in [1e-4, 1.0, 1e4] {
            for radius in [None, Some(0.25 * scale)] {
                for stop in 0..3 {
                    let target = corner(n, scale);
                    let source = shifted(&target, scale);
                    let saved_source = source.clone();
                    let saved_target = target.clone();
                    let registration = registration(scale, radius, stop);
                    let cpu = registration.align_with_diagnostics(&source, &target).unwrap();
                    let actual =
                        registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
                    maximum = maximum.max(
                        compare_pose(&source, cpu.transform, actual.result.transform, scale)
                            / scale,
                    );
                    compared += source.len();
                    assert_eq!(actual.result.iterations, cpu.iterations);
                    assert_eq!(actual.result.converged, cpu.converged);
                    assert_eq!(actual.history.len(), actual.result.iterations);
                    assert_eq!(actual.evaluation_batches, actual.result.iterations + 1);
                    assert_eq!(actual.covariance_batches, if radius.is_some() { 2 } else { 0 });
                    assert!(actual.history.iter().all(|row| row.correspondences >= 6
                        && row.evaluated_correspondences >= 6
                        && row.fitness.is_finite()
                        && row.update_magnitude.is_finite()));
                    let (x, y, z) = target.positions3().unwrap();
                    let tree = spatialrust_search::KdTree::from_slices(x, y, z);
                    let mut upload =
                        (source.len() * 48 + target.len() * 64 + tree.layout().nodes().len() * 32)
                            as u64;
                    let mut download = actual.evaluation_batches as u64 * 256;
                    if let Some(radius) = radius {
                        for cloud in [&source, &target] {
                            let (x, y, z) = cloud.positions3().unwrap();
                            let (_, dims) =
                                spatialrust_search::grid_bounds(x, y, z, radius).unwrap();
                            upload += cloud.len() as u64 * 16
                                + (dims.into_iter().map(u64::from).product::<u64>() + 1) * 4;
                            download += cloud.len() as u64 * 32;
                        }
                    }
                    assert_eq!(actual.transfers.host_to_device_bytes(), upload);
                    assert_eq!(actual.transfers.device_to_host_bytes(), download);
                    let (sx, sy, sz) = source.positions3().unwrap();
                    let mut error = 0.0;
                    for i in 0..source.len() {
                        let p =
                            actual.result.transform.transform_point(Vec3::new(sx[i], sy[i], sz[i]));
                        let j = tree
                            .nearest_one_within(p.x, p.y, p.z, (0.25 * scale).powi(2))
                            .unwrap()
                            .index;
                        error += f64::from(p.x - x[j]).powi(2)
                            + f64::from(p.y - y[j]).powi(2)
                            + f64::from(p.z - z[j]).powi(2);
                    }
                    error /= source.len() as f64;
                    assert!(
                        (actual.result.fitness - error).abs()
                            <= 1e-9 * f64::from(scale).powi(2) + 1e-4 * error
                    );
                    assert_eq!(source, saved_source);
                    assert_eq!(target, saved_target);
                }
            }
        }
    }
    eprintln!("GPU_GICP_REDUCTION compared {compared} points in 108 conditions; maximum scale-normalized CPU pose-coordinate error {maximum}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_gicp_reduction_handles_unequal_clouds_outliers_validation_and_concurrency() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = corner(85, 1.0);
    let (x, y, z) = target.positions3().unwrap();
    let mut builder = PointCloudBuilder::xyz();
    for i in (0..target.len()).step_by(2) {
        builder.push_point([x[i] + 0.015625, y[i], z[i] - 0.0078125]).unwrap();
    }
    for i in 0..7 {
        builder.push_point([100.0 + i as f32, 100.0, 100.0]).unwrap();
    }
    let source = builder.build().unwrap();
    for radius in [None, Some(0.5)] {
        let registration = registration(1.0, radius, 0);
        let expected = registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
        let cpu = registration.align_with_diagnostics(&source, &target).unwrap();
        compare_pose(&source, cpu.transform, expected.result.transform, 1.0);
        assert!(expected
            .history
            .iter()
            .all(|row| row.correspondences as usize == source.len() - 7));
        std::thread::scope(|scope| {
            let callers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        assert_eq!(
                            registration
                                .align_with_gpu_reduction(&runtime, &source, &target)
                                .unwrap(),
                            expected
                        )
                    })
                })
                .collect();
            for caller in callers {
                caller.join().unwrap();
            }
        });
    }
    for config in [
        GicpConfig { max_iterations: 0, ..Default::default() },
        GicpConfig { min_correspondences: 5, ..Default::default() },
        GicpConfig { epsilon: -1.0, ..Default::default() },
        GicpConfig { epsilon: f64::NAN, ..Default::default() },
        GicpConfig { max_correspondence_distance: f32::INFINITY, ..Default::default() },
        GicpConfig { transformation_epsilon: -1.0, ..Default::default() },
    ] {
        assert!(GicpRegistration::new(config)
            .align_with_gpu_reduction(&runtime, &target, &target)
            .is_err());
    }
    for radius in [0.0, -1.0, f32::NAN] {
        assert!(registration(1.0, Some(radius), 0)
            .align_with_gpu_reduction(&runtime, &target, &target)
            .is_err());
    }
    let far = transform_point_cloud(
        &target,
        Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
    )
    .unwrap();
    assert!(registration(1.0, None, 0)
        .align_with_gpu_reduction(&runtime, &far, &target)
        .unwrap_err()
        .to_string()
        .contains("only 0 correspondences"));
}
