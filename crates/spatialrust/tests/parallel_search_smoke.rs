#[cfg(feature = "search-parallel")]
#[test]
fn parallel_queries_preserve_serial_order_and_existing_results() {
    use spatialrust::{
        nearest_k_spatial_tensor, nearest_k_spatial_tensor_parallel_into,
        radius_search_spatial_tensor, radius_search_spatial_tensor_parallel_into, KdTree, Neighbor,
        PointCloudBuilder,
    };

    let mut builder = PointCloudBuilder::xyz();
    for i in 0..4103 {
        builder.push_point([i as f32, 0.0, 0.0]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let x = cloud.field("x").unwrap().as_f32().unwrap();
    let y = cloud.field("y").unwrap().as_f32().unwrap();
    let z = cloud.field("z").unwrap().as_f32().unwrap();
    let tree = KdTree::from_point_cloud(&cloud).unwrap();
    let tensor = cloud.spatial_tensor_chunks(1).unwrap();
    let prefix = vec![(usize::MAX, Neighbor { index: usize::MAX, distance_squared: -1.0 })];

    let mut expected = prefix.clone();
    radius_search_spatial_tensor(&tree, x, y, z, &tensor, 1.1, &mut expected);
    let mut actual = prefix.clone();
    radius_search_spatial_tensor_parallel_into(&tree, x, y, z, &tensor, 1.1, &mut actual);
    assert_eq!(actual, expected);

    let mut expected = prefix.clone();
    nearest_k_spatial_tensor(&tree, x, y, z, &tensor, 3, &mut expected);
    let mut actual = prefix;
    nearest_k_spatial_tensor_parallel_into(&tree, x, y, z, &tensor, 3, &mut actual);
    assert_eq!(actual, expected);
}

#[cfg(all(feature = "search-parallel", feature = "search-graph"))]
#[test]
fn parallel_graph_facade_preserves_serial_edge_sequence() {
    use spatialrust::{
        knn_graph, knn_graph_parallel, radius_graph, radius_graph_parallel, PointCloudBuilder,
    };
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..4103 {
        builder.push_point([(i / 2) as f32, 0.0, 0.0]).unwrap();
    }
    let cloud = builder.build().unwrap();
    assert_eq!(knn_graph_parallel(&cloud, 8).unwrap(), knn_graph(&cloud, 8).unwrap());
    assert_eq!(radius_graph_parallel(&cloud, 1.0).unwrap(), radius_graph(&cloud, 1.0).unwrap());
}
