#![cfg(feature = "search-parallel")]
#[test]
fn facade_exposes_exact_parallel_tree_construction() {
    use spatialrust::{KdTree, PointCloudBuilder, PARALLEL_KDTREE_BUILD_MIN_POINTS};
    let mut b = PointCloudBuilder::xyz();
    for i in 0..PARALLEL_KDTREE_BUILD_MIN_POINTS + 1 {
        b.push_point([(i % 127) as f32, (i / 127) as f32, 0.0]).unwrap();
    }
    let cloud = b.build().unwrap();
    let serial = KdTree::from_point_cloud(&cloud).unwrap();
    let parallel = KdTree::from_point_cloud_parallel(&cloud).unwrap();
    assert_eq!(serial.layout().point_order(), parallel.layout().point_order());
    assert_eq!(
        serial.layout().nodes().collect::<Vec<_>>(),
        parallel.layout().nodes().collect::<Vec<_>>()
    );
}
