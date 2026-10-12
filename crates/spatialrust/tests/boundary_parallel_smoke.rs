#![cfg(feature = "feature-boundary")]

use spatialrust::{
    BoundaryConfig, BoundaryDetector, DType, FieldSemantic, PointCloudBuilder, PointField,
    PointSchema,
};

#[test]
fn facade_boundary_preserves_rim_order_and_selected_cloud() {
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
    for i in 0..25 {
        builder
            .push_point([(i % 5) as f32 * 0.05, (i / 5) as f32 * 0.05, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    }
    let input = builder.build().unwrap();
    let detector = BoundaryDetector::new(BoundaryConfig {
        min_neighbors: 3,
        ..BoundaryConfig::with_radius(0.075)
    });
    let result = detector.detect(&input).unwrap();
    let expected: Vec<_> =
        (0..25).filter(|i| i % 5 == 0 || i % 5 == 4 || i / 5 == 0 || i / 5 == 4).collect();
    assert_eq!(result.indices, expected);
    assert_eq!(result.boundary.len(), 16);
    assert_eq!(result.boundary.schema(), input.schema());
    assert_eq!(result.boundary.metadata(), input.metadata());
    let mask = detector.boundary_mask(&input).unwrap();
    for (i, b) in mask.iter().enumerate() {
        assert_eq!(*b, expected.contains(&i));
    }
}
