#![cfg(all(feature = "register-gicp-gpu", not(target_arch = "wasm32")))]

use spatialrust_core::{
    HasPositions3, PointBuffer, PointCloud, PointCloudBuilder, StandardSchemas,
};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, Vec3};
use spatialrust_registration::{transform_point_cloud, GicpConfig, GicpRegistration};

fn corner(n: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for i in 0..n {
        let (a, b) = ((i % 17) as f32 * 0.125, (i / 17) as f32 * 0.125);
        builder.push_point([a, b, 0.0]).unwrap();
        builder.push_point([a, 0.0, b + 0.0625]).unwrap();
        builder.push_point([0.0, a + 0.0625, b + 0.0625]).unwrap();
    }
    builder.build().unwrap()
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_gicp_matches_cpu_pose_and_stopping_with_both_covariance_paths() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    for n in [17, 85, 257] {
        let target = corner(n);
        let pose = Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.02),
            Vec3::new(0.01, -0.005, 0.008),
        );
        let source = transform_point_cloud(&target, pose).unwrap();
        for covariance_radius in [None, Some(0.25)] {
            for stop in 0..3 {
                let config = GicpConfig {
                    max_iterations: 3,
                    max_correspondence_distance: 0.25,
                    k_neighbors: 6,
                    initial_guess: Isometry3::new(
                        Quat::<f32>::identity(),
                        Vec3::new(-0.005, 0.003, 0.0),
                    ),
                    covariance_radius,
                    transformation_epsilon: if stop == 0 { 1.0 } else { 0.0 },
                    fitness_epsilon: if stop == 1 { 1.0 } else { 0.0 },
                    ..GicpConfig::default()
                };
                let registration = GicpRegistration::new(config);
                let expected = registration.align_with_diagnostics(&source, &target).unwrap();
                let actual = registration
                    .align_with_gpu_correspondences(&runtime, &source, &target)
                    .unwrap();
                assert_eq!(
                    actual.result, expected,
                    "n={n}, radius={covariance_radius:?}, stop={stop}"
                );
                let batches = [2, 3, 7][stop];
                assert_eq!(actual.correspondence_batches, batches);
                assert_eq!(
                    actual.covariance_batches,
                    if covariance_radius.is_some() { 2 } else { 0 }
                );
                let points = n * 3;
                let (x, y, z) = target.positions3().unwrap();
                let tree = spatialrust_search::KdTree::from_slices(x, y, z);
                let mut upload =
                    (points * 16 + tree.layout().nodes().len() * 32 + points * batches * 16) as u64;
                let mut download = (points * batches * 8) as u64;
                if let Some(radius) = covariance_radius {
                    for cloud in [&source, &target] {
                        let (x, y, z) = cloud.positions3().unwrap();
                        let (_, dims) = spatialrust_search::grid_bounds(x, y, z, radius).unwrap();
                        let cells = dims.into_iter().map(u64::from).product::<u64>();
                        upload += points as u64 * 16 + (cells + 1) * 4;
                        download += points as u64 * 32;
                    }
                }
                assert_eq!(actual.transfers.host_to_device_bytes(), upload);
                assert_eq!(actual.transfers.device_to_host_bytes(), download);
            }
        }
    }
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_gicp_preserves_host_gate_edge_cases_and_propagates_gpu_errors() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = corner(85);
    let (schema, mut buffers, metadata) = target.clone().into_parts();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
    }
    let empty = PointCloud::try_from_parts(schema, buffers, metadata).unwrap();
    let registration = GicpRegistration::new(GicpConfig::default());
    assert_eq!(
        registration.align_with_diagnostics(&empty, &target).unwrap_err().to_string(),
        registration
            .align_with_gpu_correspondences(&runtime, &empty, &target)
            .unwrap_err()
            .to_string()
    );
    for gate in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let registration = GicpRegistration::new(GicpConfig {
            max_correspondence_distance: gate,
            ..GicpConfig::default()
        });
        assert_eq!(
            registration.align_with_gpu_correspondences(&runtime, &target, &target).unwrap().result,
            registration.align_with_diagnostics(&target, &target).unwrap()
        );
    }
    let registration =
        GicpRegistration::new(GicpConfig { max_iterations: 0, ..GicpConfig::default() });
    let actual = registration.align_with_gpu_correspondences(&runtime, &target, &target).unwrap();
    assert_eq!(actual.result, registration.align_with_diagnostics(&target, &target).unwrap());
    assert_eq!(actual.correspondence_batches, 1);
    for config in [
        GicpConfig { k_neighbors: 2, ..GicpConfig::default() },
        GicpConfig { min_correspondences: 1000, ..GicpConfig::default() },
    ] {
        let registration = GicpRegistration::new(config);
        assert_eq!(
            registration.align_with_diagnostics(&target, &target).unwrap_err().to_string(),
            registration
                .align_with_gpu_correspondences(&runtime, &target, &target)
                .unwrap_err()
                .to_string()
        );
    }
    for radius in [0.0, -1.0, f32::NAN] {
        let registration = GicpRegistration::new(GicpConfig {
            covariance_radius: Some(radius),
            ..GicpConfig::default()
        });
        // The legacy covariance route silently falls back; this explicit route
        // must return the GPU error rather than disguise execution placement.
        assert!(registration.align_with_gpu_correspondences(&runtime, &target, &target).is_err());
    }
    let bad_pose = GicpRegistration::new(GicpConfig {
        initial_guess: Isometry3::new(Quat::<f32>::identity(), Vec3::new(f32::INFINITY, 0.0, 0.0)),
        ..GicpConfig::default()
    });
    assert!(bad_pose
        .align_with_gpu_correspondences(&runtime, &target, &target)
        .unwrap_err()
        .to_string()
        .contains("finite transformed XYZ"));
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    builder.push_point([f32::NAN, 0.0, 0.0]).unwrap();
    let nonfinite = builder.build().unwrap();
    let registration = GicpRegistration::new(GicpConfig::default());
    assert!(registration
        .align_with_gpu_correspondences(&runtime, &nonfinite, &target)
        .unwrap_err()
        .to_string()
        .contains("finite XYZ"));
    assert!(registration
        .align_with_gpu_correspondences(&runtime, &target, &nonfinite)
        .unwrap_err()
        .to_string()
        .contains("finite XYZ"));
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_gicp_retains_source_indices_with_unequal_clouds_and_gated_outliers() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = corner(85);
    let (x, y, z) = target.positions3().unwrap();
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
    for i in (0..target.len()).step_by(2) {
        builder.push_point([x[i] + 0.015625, y[i], z[i] - 0.0078125]).unwrap();
    }
    for i in 0..7 {
        builder.push_point([100.0 + i as f32, 100.0, 100.0]).unwrap();
    }
    let source = builder.build().unwrap();
    for radius in [None, Some(0.5)] {
        let registration = GicpRegistration::new(GicpConfig {
            max_iterations: 3,
            covariance_radius: radius,
            max_correspondence_distance: 0.2,
            transformation_epsilon: 0.0,
            fitness_epsilon: 0.0,
            ..GicpConfig::default()
        });
        let actual =
            registration.align_with_gpu_correspondences(&runtime, &source, &target).unwrap();
        assert_eq!(actual.result, registration.align_with_diagnostics(&source, &target).unwrap());
        let expected = source.len() as u64 * 8 * actual.correspondence_batches as u64
            + if radius.is_some() { (source.len() + target.len()) as u64 * 32 } else { 0 };
        assert_eq!(actual.transfers.device_to_host_bytes(), expected);
    }
}
