use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spatialrust_core::{
    DType, FieldSemantic, PointCloud, PointCloudBuilder, PointField, PointSchema, StandardSchemas,
};
use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
use spatialrust_registration::{
    transform_point_cloud, GicpConfig, GicpRegistration, IcpConfig, IcpRegistration, NdtConfig,
    NdtRegistration, PointCloudRegistration, PointToPlaneIcp, PointToPlaneIcpConfig,
    RegistrationResult,
};

fn schema_with_normals() -> PointSchema {
    PointSchema::new()
        .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
        .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
        .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32))
        .with_field(PointField::scalar("normal_x", FieldSemantic::NormalX, DType::F32))
        .with_field(PointField::scalar("normal_y", FieldSemantic::NormalY, DType::F32))
        .with_field(PointField::scalar("normal_z", FieldSemantic::NormalZ, DType::F32))
}

/// A box corner (3 perpendicular faces) with analytic normals.
fn corner(side: usize, with_normals: bool) -> PointCloud {
    let mut builder = if with_normals {
        PointCloudBuilder::new(schema_with_normals())
    } else {
        PointCloudBuilder::new(StandardSchemas::point_xyz())
    };
    let step = 1.5 / side as f32;
    for i in 0..side {
        for j in 0..side {
            let (a, b) = (i as f32 * step, j as f32 * step);
            if with_normals {
                builder.push_point([a, b, 0.0, 0.0, 0.0, 1.0]).unwrap();
                builder.push_point([a, 0.0, b + 0.02, 0.0, 1.0, 0.0]).unwrap();
                builder.push_point([0.0, a + 0.02, b + 0.02, 1.0, 0.0, 0.0]).unwrap();
            } else {
                builder.push_point([a, b, 0.0]).unwrap();
                builder.push_point([a, 0.0, b + 0.02]).unwrap();
                builder.push_point([0.0, a + 0.02, b + 0.02]).unwrap();
            }
        }
    }
    builder.build().unwrap()
}

fn misalignment() -> Isometry3<f32> {
    Isometry3::new(
        Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.03),
        Vec3::new(0.02, -0.015, 0.01),
    )
}

/// Probe error after composing the recovered transform with the misalignment.
fn probe_error(result: &RegistrationResult) -> f32 {
    let composed = result.transform.compose(misalignment());
    let probe = Vec3::new(0.4, 0.5, 0.3);
    let restored = composed.transform_point(probe);
    ((restored.x - probe.x).powi(2)
        + (restored.y - probe.y).powi(2)
        + (restored.z - probe.z).powi(2))
    .sqrt()
}

fn bench_registration(c: &mut Criterion) {
    let side = 50;
    let target_xyz = corner(side, false);
    let target_normals = corner(side, true);
    let source = transform_point_cloud(&target_xyz, misalignment()).unwrap();

    let icp = IcpRegistration::new(IcpConfig {
        max_correspondence_distance: 0.3,
        max_iterations: 40,
        ..IcpConfig::default()
    });
    let p2p = PointToPlaneIcp::new(PointToPlaneIcpConfig {
        max_correspondence_distance: 0.3,
        max_iterations: 40,
        ..PointToPlaneIcpConfig::default()
    });
    let gicp = GicpRegistration::new(GicpConfig {
        max_correspondence_distance: 0.3,
        max_iterations: 40,
        k_neighbors: 12,
        ..GicpConfig::default()
    });
    let ndt = NdtRegistration::new(NdtConfig {
        resolution: 0.2,
        max_iterations: 50,
        min_points_per_voxel: 4,
        ..NdtConfig::default()
    });

    // Print one-off accuracy so the benchmark run also reports recovery error.
    let n = source.len();
    eprintln!("registration accuracy ({n} points, probe error in meters):");
    eprintln!("  icp            = {:.5}", probe_error(&icp.align(&source, &target_xyz).unwrap()));
    eprintln!(
        "  point_to_plane = {:.5}",
        probe_error(&p2p.align(&source, &target_normals).unwrap())
    );
    eprintln!("  gicp           = {:.5}", probe_error(&gicp.align(&source, &target_xyz).unwrap()));
    eprintln!("  ndt            = {:.5}", probe_error(&ndt.align(&source, &target_xyz).unwrap()));

    let mut group = c.benchmark_group(format!("registration/{n}"));
    group.bench_function("icp", |b| {
        b.iter(|| black_box(icp.align(&source, &target_xyz).unwrap()));
    });
    group.bench_function("point_to_plane", |b| {
        b.iter(|| black_box(p2p.align(&source, &target_normals).unwrap()));
    });
    group.bench_function("gicp", |b| {
        b.iter(|| black_box(gicp.align(&source, &target_xyz).unwrap()));
    });
    #[cfg(feature = "register-gicp-gpu")]
    {
        let gicp_gpu = GicpRegistration::new(GicpConfig {
            max_correspondence_distance: 0.3,
            max_iterations: 40,
            covariance_radius: Some(0.1),
            ..GicpConfig::default()
        });
        group.bench_function("gicp_gpu", |b| {
            b.iter(|| black_box(gicp_gpu.align(&source, &target_xyz).unwrap()));
        });
    }
    group.bench_function("ndt", |b| {
        b.iter(|| black_box(ndt.align(&source, &target_xyz).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, bench_registration);
criterion_main!(benches);
