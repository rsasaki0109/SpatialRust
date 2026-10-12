#![cfg(all(feature = "register-icp-gpu", not(target_arch = "wasm32")))]

use spatialrust_core::{PointCloud, PointCloudBuilder, StandardSchemas};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_registration::{
    transform_point_cloud, IcpConfig, IcpConvergenceCriteria, IcpRegistration,
};

fn cloud(n: usize, shifted_outliers: bool) -> PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for i in 0..n {
        let mut p = [(i % 9) as f32 * 0.5, (i / 9 % 9) as f32 * 0.5, (i / 81) as f32 * 0.5];
        if shifted_outliers && i % 13 == 0 {
            p[0] += 0.125;
        }
        builder.push_point(p).unwrap();
    }
    builder.build().unwrap()
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_icp_matches_cpu_pose_trace_trimming_and_transfer_receipts() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [17, 65, 257, 4097] {
        let target = cloud(n, false);
        for trimmed in [false, true] {
            let source = cloud(n, trimmed);
            let shift =
                Isometry3::new(Quat::<f32>::identity(), Vec3::new(0.03125, -0.015625, 0.0078125));
            let source = transform_point_cloud(&source, shift).unwrap();
            let registration = IcpRegistration::new(IcpConfig {
                max_iterations: 6,
                max_correspondence_distance: 0.25,
                ..IcpConfig::default()
            })
            .with_trim_fraction(if trimmed { 0.75 } else { 1.0 });
            let expected = registration.align_with_trace(&source, &target).unwrap();
            let actual =
                registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
            assert_eq!(actual.diagnostics, expected, "n={n}, trimmed={trimmed}");
            let batches = expected.result.iterations * 2 + 1;
            assert_eq!(actual.correspondence_batches, batches);
            assert_eq!(actual.transfers.device_to_host_bytes(), (n * batches * 8) as u64);
            let (x, y, z) = spatialrust_core::HasPositions3::positions3(&target).unwrap();
            let tree = spatialrust_search::KdTree::from_slices(x, y, z);
            assert_eq!(
                actual.transfers.host_to_device_bytes(),
                (n * 16 + tree.layout().nodes().len() * 32 + n * batches * 16) as u64
            );
        }
    }
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_icp_matches_cpu_rotated_initial_guess_and_iteration_limit() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = cloud(257, false);
    let pose = Isometry3::new(
        Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.03),
        Vec3::new(0.015, -0.01, 0.005),
    );
    let source = transform_point_cloud(&target, pose).unwrap();
    let registration = IcpRegistration::new(IcpConfig {
        max_iterations: 3,
        initial_guess: Isometry3::new(Quat::<f32>::identity(), Vec3::new(-0.005, 0.005, 0.0)),
        ..IcpConfig::default()
    })
    .with_convergence_criteria(IcpConvergenceCriteria {
        translation_epsilon: 0.0,
        rotation_epsilon: 0.0,
        fitness_epsilon: 0.0,
    });
    let expected = registration.align_with_trace(&source, &target).unwrap();
    let actual = registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
    assert_eq!(actual.diagnostics, expected);
    assert_eq!(actual.correspondence_batches, 7);
    assert!(!actual.diagnostics.result.converged);
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_icp_keeps_cpu_validation_and_insufficient_correspondence_errors() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = cloud(65, false);
    for config in [
        IcpConfig { max_iterations: 0, ..IcpConfig::default() },
        IcpConfig { min_correspondences: 66, ..IcpConfig::default() },
        IcpConfig { max_correspondence_distance: f32::NAN, ..IcpConfig::default() },
        IcpConfig { max_correspondence_distance: f32::MAX, ..IcpConfig::default() },
    ] {
        let registration = IcpRegistration::new(config);
        assert_eq!(
            registration.align_with_trace(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_correspondences(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    for fraction in [0.0, 1.1, f64::NAN, 0.001] {
        let registration = IcpRegistration::new(IcpConfig::default()).with_trim_fraction(fraction);
        assert_eq!(
            registration.align_with_trace(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_correspondences(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    let far = transform_point_cloud(
        &target,
        Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
    )
    .unwrap();
    let registration = IcpRegistration::new(IcpConfig::default());
    assert_eq!(
        registration.align_with_trace(&far, &target).unwrap_err().to_string(),
        registration
            .align_with_gpu_correspondences(&runtime, &far, &target)
            .unwrap_err()
            .to_string()
    );
}
