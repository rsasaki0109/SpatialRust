#![cfg(all(feature = "register-icp-point-to-plane-gpu", not(target_arch = "wasm32")))]
use spatialrust_core::{
    DType, FieldSemantic, HasNormals3, HasPositions3, PointCloud, PointCloudBuilder, PointField,
    PointSchema,
};
use spatialrust_gpu::WgpuRuntime;
use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
use spatialrust_registration::{transform_point_cloud, PointToPlaneIcp, PointToPlaneIcpConfig};
use spatialrust_search::NearestNeighborIndex;

pub fn schema() -> PointSchema {
    let mut schema = PointSchema::new();
    for (name, semantic) in [
        ("x", FieldSemantic::PositionX),
        ("y", FieldSemantic::PositionY),
        ("z", FieldSemantic::PositionZ),
        ("normal_x", FieldSemantic::NormalX),
        ("normal_y", FieldSemantic::NormalY),
        ("normal_z", FieldSemantic::NormalZ),
    ] {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
    }
    schema
}
pub fn corner(n: usize, scale: f32, weighted: bool) -> PointCloud {
    let mut builder = PointCloudBuilder::new(schema());
    for i in 0..n {
        let a = (i % 17) as f32 * 0.125 * scale;
        let b = (i / 17) as f32 * 0.125 * scale;
        let weights = if weighted { [-2.0, 0.5, 1.0 + (i % 5) as f32 * 0.25] } else { [1.0; 3] };
        builder.push_point([a, b, 0.0, 0.0, 0.0, weights[0]]).unwrap();
        builder.push_point([a, 0.0, b + 0.0625 * scale, 0.0, weights[1], 0.0]).unwrap();
        builder
            .push_point([0.0, a + 0.0625 * scale, b + 0.0625 * scale, weights[2], 0.0, 0.0])
            .unwrap();
    }
    builder.build().unwrap()
}
pub fn shifted(cloud: &PointCloud, scale: f32) -> PointCloud {
    transform_point_cloud(
        cloud,
        Isometry3::new(
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.004),
            Vec3::new(0.01 * scale, -0.005 * scale, 0.008 * scale),
        ),
    )
    .unwrap()
}
pub fn registration(scale: f32, stop: usize) -> PointToPlaneIcp {
    PointToPlaneIcp::new(PointToPlaneIcpConfig {
        max_iterations: 3,
        max_correspondence_distance: 0.25 * scale,
        initial_guess: Isometry3::new(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -0.001),
            Vec3::new(-0.005 * scale, 0.003 * scale, 0.0),
        ),
        transformation_epsilon: if stop == 1 { f64::from(scale.max(1.0)) } else { 0.0 },
        fitness_epsilon: if stop == 2 { f64::from(scale).powi(2) } else { 0.0 },
        ..Default::default()
    })
}
pub fn compare_pose(cloud: &PointCloud, a: Isometry3<f32>, b: Isometry3<f32>, scale: f32) -> f32 {
    let (x, y, z) = cloud.positions3().unwrap();
    let mut maximum = 0.0_f32;
    for i in 0..cloud.len() {
        let p = Vec3::new(x[i], y[i], z[i]);
        let error = a.transform_point(p) - b.transform_point(p);
        let error = error.x.abs().max(error.y.abs()).max(error.z.abs());
        maximum = maximum.max(error);
        assert!(error <= 2e-4 * scale, "point={i} error={error} scale={scale}");
    }
    maximum
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_plane_reduction_matches_cpu_raw_normal_weights_poses_fitness_and_stopping() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let mut compared = 0;
    let mut maximum = 0.0_f32;
    for n in [23, 63, 64, 65, 257, 1366] {
        for scale in [1e-4, 1.0, 1e4] {
            for weighted in [false, true] {
                for stop in 0..3 {
                    let target = corner(n, scale, weighted);
                    let source = shifted(&target, scale);
                    let saved_source = source.clone();
                    let saved_target = target.clone();
                    let registration = registration(scale, stop);
                    let cpu =
                        registration.align_with_diagnostics(&source, &target).unwrap_or_else(|e| {
                            panic!("n={n} scale={scale} weighted={weighted} stop={stop}: {e}")
                        });
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
                    assert!(actual.history.iter().all(|row| row.correspondences >= 6
                        && row.evaluated_correspondences >= 6
                        && row.fitness.is_finite()
                        && row.update_magnitude.is_finite()));
                    let (x, y, z) = target.positions3().unwrap();
                    let (nx, ny, nz) = target.normals3().unwrap();
                    let tree = spatialrust_search::KdTree::from_slices(x, y, z);
                    assert_eq!(
                        actual.transfers.host_to_device_bytes(),
                        (source.len() * 16 + target.len() * 48 + tree.layout().nodes().len() * 32)
                            as u64
                    );
                    assert_eq!(
                        actual.transfers.device_to_host_bytes(),
                        actual.evaluation_batches as u64 * 256
                    );
                    let (sx, sy, sz) = source.positions3().unwrap();
                    let mut error = 0.0;
                    for i in 0..source.len() {
                        let p =
                            actual.result.transform.transform_point(Vec3::new(sx[i], sy[i], sz[i]));
                        let j = tree.nearest_one(p.x, p.y, p.z).unwrap().index;
                        let residual =
                            (p - Vec3::new(x[j], y[j], z[j])).dot(Vec3::new(nx[j], ny[j], nz[j]));
                        error += f64::from(residual).powi(2);
                    }
                    error /= source.len() as f64;
                    assert!(
                        (actual.result.fitness - error).abs()
                            <= 1e-9 * f64::from(scale).powi(2) + 2e-4 * error
                    );
                    assert_eq!(source, saved_source);
                    assert_eq!(target, saved_target);
                }
            }
        }
    }
    eprintln!("GPU_PLANE_REGISTRATION compared {compared} points in 108 conditions; maximum scale-normalized CPU pose-coordinate error {maximum}");
}

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn gpu_plane_reduction_validates_schema_singular_inputs_outliers_and_concurrency() {
    let runtime = WgpuRuntime::new_headless().unwrap();
    let target = corner(85, 1.0, true);
    let (x, y, z) = target.positions3().unwrap();
    let mut builder = PointCloudBuilder::xyz();
    for i in (0..target.len()).step_by(2) {
        builder.push_point([x[i] + 0.015625, y[i], z[i] - 0.0078125]).unwrap();
    }
    for i in 0..7 {
        builder.push_point([100.0 + i as f32, 100.0, 100.0]).unwrap();
    }
    let source = builder.build().unwrap();
    let registration = registration(1.0, 0);
    let expected = registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap();
    let cpu = registration.align_with_diagnostics(&source, &target).unwrap();
    compare_pose(&source, cpu.transform, expected.result.transform, 1.0);
    assert!(expected.history.iter().all(|row| row.correspondences as usize == source.len() - 7));
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(
                        registration.align_with_gpu_reduction(&runtime, &source, &target).unwrap(),
                        expected
                    );
                })
            })
            .collect();
        for c in callers {
            c.join().unwrap();
        }
    });
    for config in [
        PointToPlaneIcpConfig { max_iterations: 0, ..Default::default() },
        PointToPlaneIcpConfig { min_correspondences: 5, ..Default::default() },
        PointToPlaneIcpConfig { max_correspondence_distance: f32::INFINITY, ..Default::default() },
        PointToPlaneIcpConfig {
            max_correspondence_distance: f32::MIN_POSITIVE,
            ..Default::default()
        },
        PointToPlaneIcpConfig { transformation_epsilon: -1.0, ..Default::default() },
        PointToPlaneIcpConfig { fitness_epsilon: f64::NAN, ..Default::default() },
        PointToPlaneIcpConfig {
            initial_guess: Isometry3::new(Quat::new(0.0, 0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 0.0)),
            ..Default::default()
        },
    ] {
        assert!(PointToPlaneIcp::new(config)
            .align_with_gpu_reduction(&runtime, &source, &target)
            .is_err());
    }
    assert!(registration.align_with_gpu_reduction(&runtime, &source, &source).is_err());
    let mut bad_xyz = PointCloudBuilder::xyz();
    bad_xyz.push_point([f32::INFINITY, 0.0, 0.0]).unwrap();
    let bad_xyz = bad_xyz.build().unwrap();
    assert!(registration.align_with_gpu_reduction(&runtime, &bad_xyz, &target).is_err());
    assert!(PointToPlaneIcp::new(PointToPlaneIcpConfig {
        min_correspondences: 1000,
        ..Default::default()
    })
    .align_with_gpu_reduction(&runtime, &source, &target)
    .is_err());
    let empty = PointCloud::xyz();
    assert!(registration.align_with_gpu_reduction(&runtime, &empty, &target).is_err());
    let mut b = PointCloudBuilder::new(schema());
    for i in 0..65 {
        b.push_point([i as f32 * 0.125, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    }
    let plane = b.build().unwrap();
    assert!(registration
        .align_with_gpu_reduction(&runtime, &plane, &plane)
        .unwrap_err()
        .to_string()
        .contains("singular"));
    let far = transform_point_cloud(
        &target,
        Isometry3::new(Quat::<f32>::identity(), Vec3::new(100.0, 0.0, 0.0)),
    )
    .unwrap();
    assert!(registration
        .align_with_gpu_reduction(&runtime, &far, &target)
        .unwrap_err()
        .to_string()
        .contains("only 0 correspondences"));
    let mut b = PointCloudBuilder::new(schema());
    for i in 0..65 {
        b.push_point([i as f32 * 0.125, 0.0, 0.0, f32::NAN, 0.0, 1.0]).unwrap();
    }
    let bad = b.build().unwrap();
    assert!(registration.align_with_gpu_reduction(&runtime, &source, &bad).is_err());
}
