#![cfg(all(feature = "register-icp-point-to-plane-gpu", not(target_arch = "wasm32")))]

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_exposes_explicit_plane_gpu_path() {
    use spatialrust::{
        gpu::{GpuPointToPlaneInput, WgpuRuntime},
        GpuPointToPlaneIteration, GpuPointToPlaneReductionDiagnostics, PointToPlaneIcp,
        PointToPlaneIcpConfig,
    };
    let runtime = WgpuRuntime::new_headless().unwrap();
    let tree = spatialrust_search::KdTree::from_slices(&[0.0], &[0.0], &[0.0]);
    let input =
        GpuPointToPlaneInput::upload(&runtime, &[[0.1, 0.0, 0.0]], &tree, &[[2.0, 0.0, 0.0]])
            .unwrap();
    let eq = input
        .equations_resident(&runtime, spatialrust_math::Isometry3::identity(), 1.0)
        .unwrap()
        .download(&runtime)
        .unwrap();
    assert_eq!(eq.correspondences, 1);
    assert!((eq.squared_error_sum - 0.04).abs() < 1e-7);
    use spatialrust_core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};
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
    let mut b = PointCloudBuilder::new(schema);
    for i in 0..6 {
        for j in 0..6 {
            let a = i as f32 * 0.25;
            let c = j as f32 * 0.25;
            b.push_point([a, c, 0.0, 0.0, 0.0, 2.0]).unwrap();
            b.push_point([a, 0.0, c + 0.125, 0.0, -0.5, 0.0]).unwrap();
            b.push_point([0.0, a + 0.125, c + 0.125, 1.0, 0.0, 0.0]).unwrap();
        }
    }
    let cloud = b.build().unwrap();
    let result: GpuPointToPlaneReductionDiagnostics =
        PointToPlaneIcp::new(PointToPlaneIcpConfig::default())
            .align_with_gpu_reduction(&runtime, &cloud, &cloud)
            .unwrap();
    assert!(result.result.converged);
    assert_eq!(result.result.iterations, 1);
    assert_eq!(result.result.fitness, 0.0);
    assert_eq!(result.evaluation_batches, 2);
    assert_eq!(result.transfers.device_to_host_bytes(), 512);
    let row: &GpuPointToPlaneIteration = &result.history[0];
    assert_eq!(row.correspondences, 108);
}
