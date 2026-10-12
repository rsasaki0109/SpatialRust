#![cfg(feature = "filter-mls")]

use spatialrust::filtering::PointCloudFilter;
use spatialrust::{MlsConfig, MlsSmoothing, PointCloudBuilder};

#[test]
fn facade_mls_preserves_point_order_and_planar_positions() {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..9 {
        builder.push_point([(i % 3) as f32 * 0.05, (i / 3) as f32 * 0.05, 0.0]).unwrap();
    }
    let input = builder.build().unwrap();
    let smoother = MlsSmoothing::new(MlsConfig::with_radius(0.2));
    let positions = smoother.smoothed_positions(&input).unwrap();
    let output = smoother.filter(&input).unwrap();
    assert_eq!(output.len(), input.len());
    assert_eq!(output.schema(), input.schema());
    assert_eq!(output.metadata(), input.metadata());
    for (i, p) in positions.iter().enumerate() {
        assert!((p.x - (i % 3) as f32 * 0.05).abs() < 1e-6);
        assert!((p.y - (i / 3) as f32 * 0.05).abs() < 1e-6);
        assert!(p.z.abs() < 1e-6);
    }
}
