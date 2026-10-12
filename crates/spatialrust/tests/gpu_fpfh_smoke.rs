#![cfg(all(feature = "register-fpfh-gpu", not(target_arch = "wasm32")))]
use spatialrust::core::{DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema};
use spatialrust::gpu::WgpuRuntime;
use spatialrust::registration::{fpfh_descriptors, fpfh_descriptors_gpu};

#[test]
#[ignore = "requires a native wgpu adapter; run explicitly with --include-ignored"]
fn facade_gpu_fpfh_preserves_planar_descriptors_and_records_transfers() {
    let mut schema = PointSchema::new();
    for (name, semantic) in [
        ("x", FieldSemantic::PositionX),
        ("y", FieldSemantic::PositionY),
        ("z", FieldSemantic::PositionZ),
        ("nx", FieldSemantic::NormalX),
        ("ny", FieldSemantic::NormalY),
        ("nz", FieldSemantic::NormalZ),
    ] {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
    }
    let mut builder = PointCloudBuilder::new(schema);
    for i in 0..257 {
        builder
            .push_point([(i % 17) as f32 * 0.125, (i / 17) as f32 * 0.125, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    }
    let cloud = builder.build().unwrap();
    let runtime = WgpuRuntime::new_headless().unwrap();
    let expected = fpfh_descriptors(&cloud, 0.35).unwrap();
    let actual = fpfh_descriptors_gpu(&runtime, &cloud, 0.35, 64).unwrap();
    assert!(actual.transfers.host_to_device_bytes() >= 257 * 64);
    assert_eq!(actual.transfers.device_to_host_bytes(), 257 * 132 + 16);
    for (a, b) in actual.descriptors.iter().flatten().zip(expected.iter().flatten()) {
        assert!((a - b).abs() <= 0.002);
    }
    assert!(fpfh_descriptors_gpu(&runtime, &cloud, 0.35, 1).is_err());
}
