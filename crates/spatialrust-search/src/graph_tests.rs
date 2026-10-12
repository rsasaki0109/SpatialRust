use super::*;
use spatialrust_core::PointCloudBuilder;

fn fixture(len: usize) -> PointCloud {
    if len == 0 {
        let mut buffers = spatialrust_core::PointBufferSet::new();
        for name in ["x", "y", "z"] {
            buffers.insert(name, spatialrust_core::PointBuffer::F32(Vec::new()));
        }
        return PointCloud::try_from_parts(
            spatialrust_core::StandardSchemas::point_xyz(),
            buffers,
            Default::default(),
        )
        .unwrap();
    }
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..len {
        // Duplicates, exact distance ties, negative coordinates and partial chunks.
        let j = i / 2;
        builder.push_point([(j % 67) as f32 - 33.0, ((j / 67) % 61) as f32 - 30.0, 0.0]).unwrap();
    }
    builder.build().unwrap()
}

// Frozen pre-change traversal: validates byte/order compatibility independently
// of the new shared graph executor and its chunk scheduling.
fn previous_graph(cloud: &PointCloud, query: GraphQuery) -> NeighborGraph {
    let (x, y, z) = cloud.positions3().unwrap();
    let tree = KdTree::from_slices(x, y, z);
    let mut edges = Vec::new();
    for i in 0..cloud.len() {
        let neighbors = match query {
            GraphQuery::Knn(k) => tree.nearest_k(x[i], y[i], z[i], k + 1),
            GraphQuery::Radius(r) => tree.radius_search(x[i], y[i], z[i], r),
        };
        for n in neighbors {
            if n.index != i {
                edges.push([i as u32, n.index as u32]);
            }
        }
    }
    NeighborGraph { num_nodes: cloud.len(), edges }
}

#[test]
fn graphs_preserve_frozen_neighbor_selection_and_edge_order() {
    for len in [0, 1, 17, 4095, 4096, 4103] {
        let cloud = fixture(len);
        for query in [
            GraphQuery::Knn(1),
            GraphQuery::Knn(8),
            GraphQuery::Knn(32),
            GraphQuery::Radius(0.01),
            GraphQuery::Radius(1.0),
            GraphQuery::Radius(1.5),
        ] {
            let expected = previous_graph(&cloud, query);
            let serial = match query {
                GraphQuery::Knn(k) => knn_graph(&cloud, k),
                GraphQuery::Radius(r) => radius_graph(&cloud, r),
            }
            .unwrap();
            assert_eq!(serial, expected, "serial len={len}");
            for available in [1, 2, 6] {
                assert_eq!(
                    build_graph(&cloud, query, available).unwrap(),
                    expected,
                    "len={len}, workers={available}"
                );
            }
        }
    }
}

#[test]
fn graphs_match_independent_distance_membership_and_exclude_self() {
    let cloud = fixture(37);
    let (x, y, z) = cloud.positions3().unwrap();
    for r in [0.01f32, 1.0, 1.5, f32::INFINITY] {
        let mut expected = Vec::new();
        for i in 0..cloud.len() {
            for j in 0..cloud.len() {
                let dx = x[i] - x[j];
                let dy = y[i] - y[j];
                let dz = z[i] - z[j];
                if i != j && dx * dx + dy * dy + dz * dz <= r * r {
                    expected.push([i as u32, j as u32]);
                }
            }
        }
        let mut actual = radius_graph(&cloud, r).unwrap().edges;
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }
    for k in [1, 8] {
        let graph = knn_graph(&cloud, k).unwrap();
        for i in 0..cloud.len() {
            let distance = |j: usize| {
                let dx = x[i] - x[j];
                let dy = y[i] - y[j];
                let dz = z[i] - z[j];
                dx * dx + dy * dy + dz * dz
            };
            let mut expected: Vec<_> = (0..cloud.len()).filter(|&j| j != i).map(distance).collect();
            expected.sort_by(f32::total_cmp);
            expected.truncate(k);
            let mut actual: Vec<_> = graph
                .edges
                .iter()
                .filter(|[source, _]| *source as usize == i)
                .map(|[_, target]| {
                    assert_ne!(*target as usize, i);
                    distance(*target as usize)
                })
                .collect();
            actual.sort_by(f32::total_cmp);
            assert_eq!(actual, expected, "query={i}, k={k}");
        }
    }
    // k >= len must reach every other point, even at usize::MAX.
    for k in [cloud.len(), usize::MAX] {
        let mut graph = knn_graph(&cloud, k).unwrap();
        assert_eq!(graph.num_nodes, cloud.len());
        graph.edges.sort_unstable();
        let expected: Vec<_> = (0..cloud.len())
            .flat_map(|i| {
                (0..cloud.len()).filter(move |&j| j != i).map(move |j| [i as u32, j as u32])
            })
            .collect();
        assert_eq!(graph.edges, expected);
        #[cfg(feature = "parallel")]
        assert_eq!(knn_graph_parallel(&cloud, k).unwrap(), knn_graph(&cloud, k).unwrap());
    }
}

#[test]
fn graph_validation_empty_and_maximum_node_indices() {
    for len in [0, 1, 17] {
        let cloud = fixture(len);
        assert!(knn_graph(&cloud, 0).is_err());
        for radius in [0.0, -1.0, f32::NEG_INFINITY, f32::NAN] {
            assert!(radius_graph(&cloud, radius).is_err());
            #[cfg(feature = "parallel")]
            assert!(radius_graph_parallel(&cloud, radius).is_err());
        }
        #[cfg(feature = "parallel")]
        assert!(knn_graph_parallel(&cloud, 0).is_err());
    }
    // A schema-only empty cloud still lacks positions, as in the old API.
    assert!(matches!(knn_graph(&PointCloud::xyz(), 1), Err(SpatialError::MissingField(_))));
    assert!(matches!(radius_graph(&PointCloud::xyz(), 1.0), Err(SpatialError::MissingField(_))));
    #[cfg(feature = "parallel")]
    {
        assert!(matches!(
            knn_graph_parallel(&PointCloud::xyz(), 1),
            Err(SpatialError::MissingField(_))
        ));
        assert!(matches!(
            radius_graph_parallel(&PointCloud::xyz(), 1.0),
            Err(SpatialError::MissingField(_))
        ));
    }
    assert!(knn_graph(&fixture(0), usize::MAX).unwrap().is_empty());
    assert!(radius_graph(&fixture(0), f32::INFINITY).unwrap().is_empty());
    assert!(validate_node_count(0).is_ok());
    assert!(validate_node_count(u32::MAX as usize).is_ok());
    if let Some(too_many) = (u32::MAX as usize).checked_add(1) {
        assert!(validate_node_count(too_many).is_err());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn public_parallel_graphs_and_concurrent_calls_keep_results() {
    let cloud = fixture(4103);
    let knn = knn_graph(&cloud, 8).unwrap();
    let radius = radius_graph(&cloud, 1.5).unwrap();
    std::thread::scope(|scope| {
        let tasks: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    assert_eq!(knn_graph_parallel(&cloud, 8).unwrap(), knn);
                    assert_eq!(radius_graph_parallel(&cloud, 1.5).unwrap(), radius);
                })
            })
            .collect();
        for task in tasks {
            task.join().unwrap();
        }
    });
}
