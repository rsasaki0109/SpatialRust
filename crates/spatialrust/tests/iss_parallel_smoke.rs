#![cfg(feature = "feature-iss")]

use spatialrust::{IssKeypointConfig, IssKeypointDetector, PointCloudBuilder};

#[test]
fn facade_iss_preserves_equal_maxima_order_and_selected_cloud() {
    let mut builder = PointCloudBuilder::xyz();
    for x in [-0.06, 0.06] {
        for y in [-0.04, 0.04] {
            for z in [-0.02, 0.02] {
                builder.push_point([x, y, z]).unwrap();
            }
        }
    }
    builder.push_point([100.0, 100.0, 100.0]).unwrap();
    let input = builder.build().unwrap();
    let detector = IssKeypointDetector::new(IssKeypointConfig::with_radii(0.2, 1.0));
    let result = detector.detect(&input).unwrap();
    assert_eq!(result.indices, (0..8).collect::<Vec<_>>());
    assert_eq!(result.keypoints.len(), 8);
    assert_eq!(result.keypoints.schema(), input.schema());
    assert_eq!(result.keypoints.metadata(), input.metadata());
}
