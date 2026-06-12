#[cfg(feature = "search-kdtree")]
#[test]
fn kdtree_public_api() {
    use spatialrust::{KdTree, NearestNeighborIndex, PointCloudBuilder};

    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([0.0, 0.0, 0.0]).unwrap();
    builder.push_point([1.0, 0.0, 0.0]).unwrap();
    let cloud = builder.build().unwrap();

    let tree = KdTree::from_point_cloud(&cloud).unwrap();
    let nearest = tree.nearest_one(0.8, 0.0, 0.0).unwrap();
    assert_eq!(nearest.index, 1);
}
