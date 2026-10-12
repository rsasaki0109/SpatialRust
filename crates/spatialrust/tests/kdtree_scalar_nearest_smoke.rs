#![cfg(feature = "search-kdtree")]

use spatialrust::search::{KdTree, NearestNeighborIndex};

#[test]
fn facade_scalar_nearest_preserves_finite_ties_and_nan_queries() {
    let tree = KdTree::from_slices(&[-1.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
    let neighbor = tree.nearest_one(0.0, 0.0, 0.0).unwrap();
    assert_eq!(neighbor.index, 0);
    assert_eq!(neighbor.distance_squared, 1.0);
    let neighbor = tree.nearest_one(f32::NAN, 0.0, 0.0).unwrap();
    assert_eq!(neighbor.index, 2);
    assert!(neighbor.distance_squared.is_nan());
}
