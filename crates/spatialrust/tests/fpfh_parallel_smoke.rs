#![cfg(feature = "register-fpfh")]

use spatialrust::{
    fpfh_descriptors, DType, FieldSemantic, PointCloudBuilder, PointField, PointSchema,
    FPFH_DESCRIPTOR_LEN,
};

#[test]
fn facade_fpfh_feature_keeps_order_and_exact_planar_bins() {
    let mut schema = PointSchema::new();
    for (name, semantic) in [
        ("px", FieldSemantic::PositionX),
        ("py", FieldSemantic::PositionY),
        ("pz", FieldSemantic::PositionZ),
        ("nx", FieldSemantic::NormalX),
        ("ny", FieldSemantic::NormalY),
        ("nz", FieldSemantic::NormalZ),
    ] {
        schema = schema.with_field(PointField::scalar(name, semantic, DType::F32));
    }
    let mut builder = PointCloudBuilder::new(schema);
    builder.push_point([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    builder.push_point([0.05, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let cloud = builder.build().unwrap();
    let descriptors = fpfh_descriptors(&cloud, 0.075).unwrap();
    assert_eq!(FPFH_DESCRIPTOR_LEN, 33);
    assert_eq!(descriptors.len(), cloud.len());
    for descriptor in descriptors {
        for (bin, value) in descriptor.into_iter().enumerate() {
            assert_eq!(value.to_bits(), if bin % 11 == 5 { 200.0_f32.to_bits() } else { 0 });
        }
    }
}
