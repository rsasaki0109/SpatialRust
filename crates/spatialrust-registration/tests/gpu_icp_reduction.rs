#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]
use spatialrust_core::{HasPositions3, PointCloud, PointCloudBuilder};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
use spatialrust_registration::{
    transform_point_cloud, IcpConfig, IcpConvergenceCriteria, IcpRegistration,
};

pub fn fixture(n: usize, kind: usize, scale: f32) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..n {
        builder
            .push_point([
                (i % 13) as f32 * 0.5 * scale,
                (i / 13 % 11) as f32 * 0.5 * scale,
                if kind == 1 { 0.0 } else { (i / 143) as f32 * 0.5 * scale },
            ])
            .unwrap();
    }
    builder.build().unwrap()
}
pub fn registration(scale: f32) -> IcpRegistration {
    IcpRegistration::new(IcpConfig {
        max_iterations: 4,
        max_correspondence_distance: 0.25 * scale,
        initial_guess: Isometry3::new(
            Quat::<f32>::identity(),
            Vec3::new(-0.001 * scale, 0.002 * scale, 0.0),
        ),
        ..Default::default()
    })
    .with_convergence_criteria(IcpConvergenceCriteria {
        translation_epsilon: 0.0,
        rotation_epsilon: 0.0,
        fitness_epsilon: 0.0,
    })
}
pub fn source(target: &PointCloud, scale: f32) -> PointCloud {
    transform_point_cloud(
        target,
        Isometry3::new(
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.003),
            Vec3::new(0.015 * scale, -0.01 * scale, 0.005 * scale),
        ),
    )
    .unwrap()
}
pub fn compare_pose(source: &PointCloud, a: Isometry3<f32>, b: Isometry3<f32>, scale: f32) -> f32 {
    let (x, y, z) = source.positions3().unwrap();
    let mut maximum = 0.0_f32;
    for i in 0..source.len() {
        let p = Vec3::new(x[i], y[i], z[i]);
        let u = a.transform_point(p);
        let v = b.transform_point(p);
        let error = (u - v).x.abs().max((u - v).y.abs()).max((u - v).z.abs());
        maximum = maximum.max(error);
        assert!(error <= 2e-5 * scale.max(1e-4), "point={i} error={error} scale={scale}");
    }
    maximum
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_icp_matches_cpu_poses_and_trace_counts_across_sizes_scales_and_unequal_clouds() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut maximum = 0.0_f32;
    let mut points = 0;
    for n in [17, 63, 64, 65, 257, 4097] {
        for kind in 0..3 {
            for scale in [1e-4, 1.0, 1e4] {
                let cloud = fixture(n, kind, scale);
                let target = fixture(n + if kind == 2 { 17 } else { 0 }, kind, scale);
                let source = source(&cloud, scale);
                let saved_source = source.clone();
                let saved_target = target.clone();
                let registration = registration(scale);
                let expected = registration.align_with_trace(&source, &target).unwrap();
                let actual =
                    registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
                maximum = maximum.max(
                    compare_pose(
                        &source,
                        actual.diagnostics.result.transform,
                        expected.result.transform,
                        scale,
                    ) / scale,
                );
                assert_eq!(actual.diagnostics.result.iterations, 4);
                assert!(!actual.diagnostics.result.converged);
                assert_eq!(actual.diagnostics.stop_reason, expected.stop_reason);
                for (a, b) in actual.diagnostics.history.iter().zip(&expected.history) {
                    assert_eq!(
                        (a.correspondences, a.evaluated_correspondences),
                        (b.correspondences, b.evaluated_correspondences)
                    );
                    assert!(a.fitness.is_finite() && a.fitness >= 0.0);
                }
                assert_eq!(actual.evaluation_batches, 5);
                assert_eq!(actual.transfers.device_to_host_bytes(), 5 * 176);
                let (x, y, z) = target.positions3().unwrap();
                let tree = spatialrust_search::KdTree::from_slices(x, y, z);
                assert_eq!(
                    actual.transfers.host_to_device_bytes(),
                    (n * 16 + target.len() * 32 + tree.layout().nodes().len() * 32) as u64
                );
                let (sx, sy, sz) = source.positions3().unwrap();
                let mut residual = 0.0;
                for i in 0..n {
                    let p = actual
                        .diagnostics
                        .result
                        .transform
                        .transform_point(Vec3::new(sx[i], sy[i], sz[i]));
                    let neighbor =
                        tree.nearest_one_within(p.x, p.y, p.z, (0.25 * scale).powi(2)).unwrap();
                    residual += f64::from(p.x - x[neighbor.index]).powi(2)
                        + f64::from(p.y - y[neighbor.index]).powi(2)
                        + f64::from(p.z - z[neighbor.index]).powi(2);
                }
                residual /= n as f64;
                assert!(
                    (actual.diagnostics.result.fitness - residual).abs()
                        <= 1e-10 * f64::from(scale).powi(2) + residual * 1e-4
                );
                assert_eq!(source, saved_source);
                assert_eq!(target, saved_target);
                points += n;
            }
        }
    }
    eprintln!("GPU_ICP_REDUCTION compared {points} transformed points in 54 conditions; maximum scale-normalized CPU pose-coordinate difference {maximum}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_icp_keeps_validation_and_reports_unsupported_trimming_and_insufficient_matches() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = fixture(65, 0, 1.0);
    for config in [
        IcpConfig { max_iterations: 0, ..Default::default() },
        IcpConfig { min_correspondences: 66, ..Default::default() },
        IcpConfig { max_correspondence_distance: f32::NAN, ..Default::default() },
        IcpConfig { max_correspondence_distance: f32::MAX, ..Default::default() },
    ] {
        let registration = IcpRegistration::new(config);
        assert_eq!(
            registration.align_with_trace(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_reduction(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    for fraction in [0.0, 1.1, f64::NAN] {
        let registration = registration(1.0).with_trim_fraction(fraction);
        assert_eq!(
            registration.align_with_trace(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_reduction(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    assert!(registration(1.0)
        .with_trim_fraction(0.75)
        .align_with_gpu_reduction(&runtime, &target, &target)
        .unwrap_err()
        .to_string()
        .contains("trim_fraction=1"));
    let far = transform_point_cloud(
        &target,
        Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
    )
    .unwrap();
    assert_eq!(
        registration(1.0).align_with_trace(&far, &target).unwrap_err().to_string(),
        registration(1.0)
            .align_with_gpu_reduction(&runtime, &far, &target)
            .unwrap_err()
            .to_string()
    );
    let converged = IcpRegistration::new(IcpConfig::default())
        .align_with_gpu_reduction(&runtime, &target, &target)
        .unwrap();
    assert!(converged.diagnostics.result.converged);
    assert_eq!(converged.diagnostics.result.iterations, 1);
    assert_eq!(converged.evaluation_batches, 2);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn resident_icp_full_calls_are_deterministic_with_concurrent_shared_runtime() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = fixture(257, 0, 1.0);
    let source = source(&target, 1.0);
    let registration = registration(1.0);
    let expected = registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(
                        registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap(),
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
