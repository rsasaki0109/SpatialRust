#![cfg(feature = "search-kdtree")]

use spatialrust::search::{KdTree, NearestNeighborIndex, RadiusSearchIndex};
use spatialrust::{HasPositions3, PointCloudBuilder};

#[test]
fn facade_kdtree_handles_large_coincident_points_and_query_bounds() {
    let mut builder = PointCloudBuilder::xyz();
    for _ in 0..32771 {
        builder.push_point([1.0, 1.0, 1.0]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tree = KdTree::from_point_cloud(&cloud).unwrap();
    assert_eq!(tree.nearest_one(0.0, 0.0, 0.0).unwrap().distance_squared, 3.0);
    assert_eq!(tree.nearest_one_within(0.0, 0.0, 0.0, 2.99), None);
    assert_eq!(tree.radius_search(1.0, 1.0, 1.0, 0.0).len(), cloud.positions3().unwrap().0.len());
}
