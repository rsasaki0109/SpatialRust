#![cfg(feature = "metrics-distance")]

use spatialrust::{chamfer_distance, cloud_distances, hausdorff_distance, PointCloudBuilder};

#[test]
fn facade_metrics_preserve_asymmetric_directed_statistics() {
    let mut a = PointCloudBuilder::xyz();
    for x in [0.0, 2.0] {
        a.push_point([x, 0.0, 0.0]).unwrap();
    }
    let mut b = PointCloudBuilder::xyz();
    for x in [0.0, 1.0, 4.0] {
        b.push_point([x, 0.0, 0.0]).unwrap();
    }
    let (a, b) = (a.build().unwrap(), b.build().unwrap());
    let result = cloud_distances(&a, &b).unwrap();
    assert_eq!(result.mean_sq_a_to_b.to_bits(), 0.5_f64.to_bits());
    assert_eq!(result.mean_sq_b_to_a.to_bits(), (5.0_f64 / 3.0).to_bits());
    assert_eq!(result.max_a_to_b, 1.0);
    assert_eq!(result.max_b_to_a, 2.0);
    assert_eq!(chamfer_distance(&a, &b).unwrap(), 0.5 + 5.0 / 3.0);
    assert_eq!(hausdorff_distance(&a, &b).unwrap(), 2.0);
}
