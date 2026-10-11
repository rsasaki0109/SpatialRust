//! Neighborhood graph construction (k-NN and radius graphs).
//!
//! Turns a point cloud into the edge list that graph neural networks consume
//! (PyG-style `edge_index`): each point becomes a node, with a directed edge to
//! every neighbor. Built on the KD-tree, so it scales to large clouds.

use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};

use crate::kdtree::KdTree;
use crate::{NearestNeighborIndex, RadiusSearchIndex};

/// A directed neighborhood graph over a point cloud.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeighborGraph {
    /// Number of nodes (input points).
    pub num_nodes: usize,
    /// Directed edges as `[source, target]` index pairs (no self-loops).
    pub edges: Vec<[u32; 2]>,
}

impl NeighborGraph {
    /// Number of directed edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Whether the graph has no edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Builds a directed k-nearest-neighbor graph: an edge from every point to each
/// of its `k` nearest neighbors (excluding itself).
///
/// Requests exceeding the cloud size search all points. Equal-distance neighbor
/// selection and order follow the KD-tree, including duplicate coordinates.
pub fn knn_graph(cloud: &PointCloud, k: usize) -> SpatialResult<NeighborGraph> {
    if k == 0 {
        return Err(SpatialError::InvalidArgument("k must be greater than zero".to_owned()));
    }
    let (x, y, z) = cloud.positions3()?;
    let len = cloud.len();
    validate_node_count(len)?;
    if len == 0 {
        return Ok(NeighborGraph { num_nodes: 0, edges: Vec::new() });
    }
    let tree = KdTree::from_slices(x, y, z);
    let count = k.saturating_add(1).min(len);
    let mut edges = Vec::with_capacity(len.checked_mul(k.min(len)).unwrap_or(0));
    for i in 0..len {
        for neighbor in tree.nearest_k(x[i], y[i], z[i], count) {
            if neighbor.index != i {
                edges.push([i as u32, neighbor.index as u32]);
            }
        }
    }
    Ok(NeighborGraph { num_nodes: len, edges })
}

/// Builds a directed radius graph: an edge from every point to each other point
/// within `radius`.
pub fn radius_graph(cloud: &PointCloud, radius: f32) -> SpatialResult<NeighborGraph> {
    if radius <= 0.0 || radius.is_nan() {
        return Err(SpatialError::InvalidArgument("radius must be positive".to_owned()));
    }
    let (x, y, z) = cloud.positions3()?;
    let len = cloud.len();
    validate_node_count(len)?;
    if len == 0 {
        return Ok(NeighborGraph { num_nodes: 0, edges: Vec::new() });
    }
    let tree = KdTree::from_slices(x, y, z);
    let mut edges = Vec::new();
    for i in 0..len {
        for neighbor in tree.radius_search(x[i], y[i], z[i], radius) {
            if neighbor.index != i {
                edges.push([i as u32, neighbor.index as u32]);
            }
        }
    }
    Ok(NeighborGraph { num_nodes: len, edges })
}

/// Builds a k-NN graph using bounded scoped CPU workers (`parallel` feature).
///
/// Returns exactly the same edge sequence as [`knn_graph`], including KD-tree
/// tie selection. One shared tree is built on the caller; contiguous 256-point
/// query chunks use at most the available CPU parallelism. Fewer than 4,096
/// points or one available worker run on the caller. Worker-local edge storage
/// and the final graph scale with edge count; dispatch metadata scales with
/// workers. Concurrent calls have independent worker budgets.
#[cfg(feature = "parallel")]
pub fn knn_graph_parallel(cloud: &PointCloud, k: usize) -> SpatialResult<NeighborGraph> {
    build_graph(cloud, GraphQuery::Knn(k), crate::parallel_chunked::available_workers())
}

/// Builds a radius graph using bounded scoped CPU workers (`parallel` feature).
///
/// Returns exactly the same edge sequence as [`radius_graph`]. Uses the same
/// worker, small-input and storage policy as [`knn_graph_parallel`]. Radius
/// graphs can contain quadratically many edges; parallel execution does not
/// bound the result size.
#[cfg(feature = "parallel")]
pub fn radius_graph_parallel(cloud: &PointCloud, radius: f32) -> SpatialResult<NeighborGraph> {
    build_graph(cloud, GraphQuery::Radius(radius), crate::parallel_chunked::available_workers())
}

#[cfg(any(feature = "parallel", test))]
#[derive(Clone, Copy)]
enum GraphQuery {
    Knn(usize),
    Radius(f32),
}

fn validate_node_count(len: usize) -> SpatialResult<()> {
    // The KD-tree's end offsets and graph indices are u32.
    if u32::try_from(len).is_err() {
        return Err(SpatialError::InvalidArgument("graph point count exceeds u32::MAX".to_owned()));
    }
    Ok(())
}

#[cfg(any(feature = "parallel", test))]
fn build_graph(
    cloud: &PointCloud,
    query: GraphQuery,
    _available: usize,
) -> SpatialResult<NeighborGraph> {
    match query {
        GraphQuery::Knn(0) => {
            return Err(SpatialError::InvalidArgument("k must be greater than zero".to_owned()));
        }
        GraphQuery::Radius(radius) if radius <= 0.0 || radius.is_nan() => {
            return Err(SpatialError::InvalidArgument("radius must be positive".to_owned()));
        }
        _ => {}
    }
    let (x, y, z) = cloud.positions3()?;
    let len = cloud.len();
    validate_node_count(len)?;
    if len == 0 {
        return Ok(NeighborGraph { num_nodes: 0, edges: Vec::new() });
    }
    let tree = KdTree::from_slices(x, y, z);
    // Select the query kind outside the per-point loop. A one-worker
    // parallel call also uses a specialized traversal and one caller range.
    let edges = match query {
        GraphQuery::Knn(k) => {
            let count = k.saturating_add(1).min(len);
            let capacity = len.checked_mul(k.min(len)).unwrap_or(0);
            collect_edges(cloud, _available, capacity, |i| tree.nearest_k(x[i], y[i], z[i], count))?
        }
        GraphQuery::Radius(radius) => {
            collect_edges(cloud, _available, 0, |i| tree.radius_search(x[i], y[i], z[i], radius))?
        }
    };
    Ok(NeighborGraph { num_nodes: len, edges })
}

#[cfg(any(feature = "parallel", test))]
fn collect_edges<Q>(
    cloud: &PointCloud,
    _available: usize,
    capacity: usize,
    query: Q,
) -> SpatialResult<Vec<[u32; 2]>>
where
    Q: Fn(usize) -> Vec<crate::Neighbor> + Sync,
{
    let query_band = |range: std::ops::Range<usize>, edges: &mut Vec<[u32; 2]>| {
        for i in range {
            for neighbor in query(i) {
                if neighbor.index != i {
                    edges.push([i as u32, neighbor.index as u32]);
                }
            }
        }
    };
    let mut edges = Vec::with_capacity(capacity);
    #[cfg(feature = "parallel")]
    if cloud.len() >= crate::PARALLEL_CHUNK_QUERY_MIN_POINTS && _available > 1 {
        let tensor = cloud.spatial_tensor_chunks(256)?;
        crate::parallel_chunked::dispatch_chunks(&tensor, _available, &mut edges, query_band);
        return Ok(edges);
    }
    query_band(0..cloud.len(), &mut edges);
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::{knn_graph, radius_graph};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn line(n: usize) -> spatialrust_core::PointCloud {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for i in 0..n {
            builder.push_point([i as f32, 0.0, 0.0]).unwrap();
        }
        builder.build().unwrap()
    }

    #[test]
    fn knn_graph_has_k_edges_per_node() {
        let cloud = line(10);
        let graph = knn_graph(&cloud, 2).unwrap();
        assert_eq!(graph.num_nodes, 10);
        // Every node has exactly k outgoing edges.
        assert_eq!(graph.num_edges(), 10 * 2);
        // No self-loops.
        assert!(graph.edges.iter().all(|[s, t]| s != t));
    }

    #[test]
    fn knn_nearest_neighbor_is_adjacent_on_a_line() {
        let cloud = line(5);
        let graph = knn_graph(&cloud, 1).unwrap();
        // Point 0's single nearest neighbor is point 1.
        let from_0: Vec<u32> =
            graph.edges.iter().filter(|[s, _]| *s == 0).map(|[_, t]| *t).collect();
        assert_eq!(from_0, vec![1]);
    }

    #[test]
    fn radius_graph_links_points_within_radius() {
        let cloud = line(5);
        // Radius 1.5 reaches the immediate neighbors on each side (spacing 1.0).
        let graph = radius_graph(&cloud, 1.5).unwrap();
        // Interior nodes have 2 neighbors, the two ends have 1 each: 3*2 + 2*1 = 8.
        assert_eq!(graph.num_edges(), 8);
    }

    #[test]
    fn rejects_bad_params() {
        let cloud = line(3);
        assert!(knn_graph(&cloud, 0).is_err());
        assert!(radius_graph(&cloud, 0.0).is_err());
    }
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod graph_tests;
