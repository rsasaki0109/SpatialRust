//! Parallel dispatch for chunked spatial queries (`parallel` feature).
//!
//! Each [`SpatialTensor`] chunk is queried on its own thread; partial results are
//! merged without locks because [`KdTree`](crate::KdTree) search is read-only.

use spatialrust_core::SpatialTensor;

use crate::chunked::{
    nearest_k_spatial_tensor, radius_search_spatial_tensor, ChunkedNearestNeighborIndex,
    ChunkedRadiusSearchIndex,
};
use crate::Neighbor;

/// Minimum point count before threaded chunk dispatch is used.
pub const PARALLEL_CHUNK_QUERY_MIN_POINTS: usize = 4_096;

/// Parallel radius search over [`SpatialTensor`] chunks.
pub fn radius_search_spatial_tensor_parallel<I: ChunkedRadiusSearchIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    radius: f32,
) -> Vec<(usize, Neighbor)> {
    let mut out = Vec::new();
    radius_search_spatial_tensor_parallel_into(index, x, y, z, tensor, radius, &mut out);
    out
}

/// Parallel radius search, appending `(query_index, neighbor)` pairs to `out`.
pub fn radius_search_spatial_tensor_parallel_into<I: ChunkedRadiusSearchIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    radius: f32,
    out: &mut Vec<(usize, Neighbor)>,
) {
    if tensor.is_empty() {
        return;
    }
    if tensor.len() < PARALLEL_CHUNK_QUERY_MIN_POINTS {
        radius_search_spatial_tensor(index, x, y, z, tensor, radius, out);
        return;
    }

    let ranges: Vec<_> = tensor.chunks().map(|chunk| chunk.range()).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ranges.len());
        for query_range in ranges {
            handles.push(scope.spawn(|| {
                let mut local = Vec::new();
                index.radius_search_chunk_into(x, y, z, query_range, radius, &mut local);
                local
            }));
        }

        for handle in handles {
            out.extend(handle.join().expect("chunk radius search thread panicked"));
        }
    });
}

/// Parallel k-NN search over [`SpatialTensor`] chunks.
pub fn nearest_k_spatial_tensor_parallel<I: ChunkedNearestNeighborIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    k: usize,
) -> Vec<(usize, Neighbor)> {
    let mut out = Vec::new();
    nearest_k_spatial_tensor_parallel_into(index, x, y, z, tensor, k, &mut out);
    out
}

/// Parallel k-NN search, appending `(query_index, neighbor)` pairs to `out`.
pub fn nearest_k_spatial_tensor_parallel_into<I: ChunkedNearestNeighborIndex + Sync>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    k: usize,
    out: &mut Vec<(usize, Neighbor)>,
) {
    if tensor.is_empty() || k == 0 {
        return;
    }
    if tensor.len() < PARALLEL_CHUNK_QUERY_MIN_POINTS {
        nearest_k_spatial_tensor(index, x, y, z, tensor, k, out);
        return;
    }

    let ranges: Vec<_> = tensor.chunks().map(|chunk| chunk.range()).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(ranges.len());
        for query_range in ranges {
            handles.push(scope.spawn(|| {
                let mut local = Vec::new();
                index.nearest_k_chunk_into(x, y, z, query_range, k, &mut local);
                local
            }));
        }

        for handle in handles {
            out.extend(handle.join().expect("chunk k-NN search thread panicked"));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{
        nearest_k_spatial_tensor_parallel, radius_search_spatial_tensor_parallel,
        PARALLEL_CHUNK_QUERY_MIN_POINTS,
    };
    use crate::chunked::{nearest_k_spatial_tensor, radius_search_spatial_tensor};
    use crate::kdtree::KdTree;
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn grid_cloud(side: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, spatialrust_core::PointCloud) {
        let mut x = Vec::with_capacity(side * side);
        let mut y = Vec::with_capacity(side * side);
        let mut z = Vec::with_capacity(side * side);
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for row in 0..side {
            for col in 0..side {
                let px = col as f32 * 0.01;
                let py = row as f32 * 0.01;
                x.push(px);
                y.push(py);
                z.push(0.0);
                builder.push_point([px, py, 0.0]).unwrap();
            }
        }
        (x, y, z, builder.build().unwrap())
    }

    fn sort_neighbors(neighbors: &mut [(usize, crate::Neighbor)]) {
        neighbors.sort_by(|a, b| (a.0, a.1.index).cmp(&(b.0, b.1.index)));
    }

    #[test]
    fn parallel_radius_matches_sequential_on_large_grid() {
        let side = 70; // 4900 points > PARALLEL_CHUNK_QUERY_MIN_POINTS
        assert!(side * side >= PARALLEL_CHUNK_QUERY_MIN_POINTS);
        let (x, y, z, cloud) = grid_cloud(side);
        let tree = KdTree::from_slices(&x, &y, &z);
        let tensor = cloud.spatial_tensor_chunks(512).unwrap();

        let mut sequential = Vec::new();
        radius_search_spatial_tensor(&tree, &x, &y, &z, &tensor, 0.02, &mut sequential);

        let mut parallel = radius_search_spatial_tensor_parallel(
            &tree, &x, &y, &z, &tensor, 0.02,
        );

        sort_neighbors(&mut sequential);
        sort_neighbors(&mut parallel);
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn parallel_knn_matches_sequential_on_large_grid() {
        let side = 70;
        let (x, y, z, cloud) = grid_cloud(side);
        let tree = KdTree::from_slices(&x, &y, &z);
        let tensor = cloud.spatial_tensor_chunks(512).unwrap();

        let mut sequential = Vec::new();
        nearest_k_spatial_tensor(&tree, &x, &y, &z, &tensor, 8, &mut sequential);

        let mut parallel =
            nearest_k_spatial_tensor_parallel(&tree, &x, &y, &z, &tensor, 8);

        sort_neighbors(&mut sequential);
        sort_neighbors(&mut parallel);
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn parallel_falls_back_below_threshold() {
        let (x, y, z, cloud) = grid_cloud(10); // 100 points
        assert!(cloud.len() < PARALLEL_CHUNK_QUERY_MIN_POINTS);
        let tree = KdTree::from_slices(&x, &y, &z);
        let tensor = cloud.spatial_tensor_chunks(32).unwrap();

        let mut sequential = Vec::new();
        radius_search_spatial_tensor(&tree, &x, &y, &z, &tensor, 0.05, &mut sequential);

        let mut parallel =
            radius_search_spatial_tensor_parallel(&tree, &x, &y, &z, &tensor, 0.05);

        sort_neighbors(&mut sequential);
        sort_neighbors(&mut parallel);
        assert_eq!(parallel, sequential);
    }
}
