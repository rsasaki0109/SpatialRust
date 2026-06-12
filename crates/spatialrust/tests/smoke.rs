#[test]
fn reexports_core_types() {
    use spatialrust::{
        DType, FieldSemantic, HasPositions3, PointCloudBuilder, StandardSchemas, Vec3,
    };

    let schema = StandardSchemas::point_xyz();
    assert_eq!(schema.len(), 3);
    assert!(schema.find_semantic(FieldSemantic::PositionX).is_some());

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    let cloud = builder.build().unwrap();
    let (x, _, _) = cloud.positions3().unwrap();
    assert_eq!(x, &[1.0]);

    let v = Vec3::new(1.0_f32, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(DType::F32.size_bytes(), 4);
}

#[test]
fn math_reexports_compile() {
    use spatialrust::{symmetric_eigen3, HuberKernel, Mat3, Quat, RobustKernel};

    let kernel = HuberKernel::new(1.0);
    assert_eq!(kernel.weight(0.5), 1.0);
    let _ = Quat::<f32>::identity().to_mat3();
    let _ = symmetric_eigen3(Mat3::<f64>::identity());
}
