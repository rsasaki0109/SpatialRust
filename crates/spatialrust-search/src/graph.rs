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
pub fn knn_graph(cloud: &PointCloud, k: usize) -> SpatialResult<NeighborGraph> {
    if k == 0 {
        return Err(SpatialError::InvalidArgument("k must be greater than zero".to_owned()));
    }
    let (x, y, z) = cloud.positions3()?;
    let len = cloud.len();
    if len == 0 {
        return Ok(NeighborGraph { num_nodes: 0, edges: Vec::new() });
    }

    let tree = KdTree::from_slices(x, y, z);
    let mut edges = Vec::with_capacity(len * k);
    for i in 0..len {
        // k + 1 because the point finds itself first.
        for neighbor in tree.nearest_k(x[i], y[i], z[i], k + 1) {
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
