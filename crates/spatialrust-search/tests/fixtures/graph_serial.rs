// Frozen graph API loops from a804c23 for completed-operation comparisons.
use spatialrust_core::{HasPositions3, PointCloud, SpatialError, SpatialResult};
use spatialrust_search::{KdTree, NearestNeighborIndex, NeighborGraph, RadiusSearchIndex};

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
