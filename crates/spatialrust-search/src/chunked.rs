//! Chunk-oriented query extensions for [`SpatialIndex`](crate::SpatialIndex) backends.
//!
//! These traits align query batching with [`SpatialTensor`](spatialrust_core::SpatialTensor)
//! chunk iteration for parallel CPU/GPU algorithms.

use std::ops::Range;

use spatialrust_core::SpatialTensor;

use crate::{Neighbor, RadiusSearchIndex};

/// Range of query point indices (matches [`SpatialTensor::chunks`] ranges).
pub type ChunkQueryRange = Range<usize>;

/// Radius search over contiguous query index ranges.
pub trait ChunkedRadiusSearchIndex: RadiusSearchIndex {
    /// Appends `(query_index, neighbor)` for each point index in `chunk`.
    fn radius_search_chunk_into(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        chunk: ChunkQueryRange,
        radius: f32,
        out: &mut Vec<(usize, Neighbor)>,
    );

    /// Radius search for a single indexed query point.
    fn radius_search_at(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        index: usize,
        radius: f32,
    ) -> Vec<Neighbor> {
        self.radius_search(x[index], y[index], z[index], radius)
    }
}

/// k-NN search over contiguous query index ranges.
pub trait ChunkedNearestNeighborIndex: crate::NearestNeighborIndex {
    /// Appends `(query_index, neighbor)` for each point index in `chunk`.
    fn nearest_k_chunk_into(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        chunk: ChunkQueryRange,
        k: usize,
        out: &mut Vec<(usize, Neighbor)>,
    );
}

/// Runs radius search for every chunk in a [`SpatialTensor`], appending tagged neighbors.
pub fn radius_search_spatial_tensor<I: ChunkedRadiusSearchIndex>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    radius: f32,
    out: &mut Vec<(usize, Neighbor)>,
) {
    for chunk in tensor.chunks() {
        index.radius_search_chunk_into(x, y, z, chunk.range(), radius, out);
    }
}

/// Runs k-NN search for every chunk in a [`SpatialTensor`], appending tagged neighbors.
pub fn nearest_k_spatial_tensor<I: ChunkedNearestNeighborIndex>(
    index: &I,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    tensor: &SpatialTensor<'_>,
    k: usize,
    out: &mut Vec<(usize, Neighbor)>,
) {
    for chunk in tensor.chunks() {
        index.nearest_k_chunk_into(x, y, z, chunk.range(), k, out);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        nearest_k_spatial_tensor, radius_search_spatial_tensor, ChunkedNearestNeighborIndex,
        ChunkedRadiusSearchIndex,
    };
    use crate::{brute::BruteForceIndex, kdtree::KdTree, RadiusSearchIndex};
    use spatialrust_core::{PointCloudBuilder, StandardSchemas};

    fn sample_cloud() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        )
    }

    fn chunked_matches_per_index<I>(index: &I, radius: f32)
    where
        I: ChunkedRadiusSearchIndex + RadiusSearchIndex,
    {
        let (x, y, z) = sample_cloud();
        let mut chunked = Vec::new();
        index.radius_search_chunk_into(&x, &y, &z, 1..4, radius, &mut chunked);

        let mut expected = Vec::new();
        for query in 1..4 {
            for neighbor in index.radius_search_at(&x, &y, &z, query, radius) {
                expected.push((query, neighbor));
            }
        }

        chunked.sort_by(|a, b| (a.0, a.1.index).cmp(&(b.0, b.1.index)));
        expected.sort_by(|a, b| (a.0, a.1.index).cmp(&(b.0, b.1.index)));
        assert_eq!(chunked, expected);
    }

    #[test]
    fn kdtree_chunked_radius_matches_per_index() {
        let (x, y, z) = sample_cloud();
        let tree = KdTree::from_slices(&x, &y, &z);
        chunked_matches_per_index(&tree, 1.5);
    }

    #[test]
    fn brute_chunked_radius_matches_per_index() {
        let (x, y, z) = sample_cloud();
        let index = BruteForceIndex::from_slices(&x, &y, &z);
        chunked_matches_per_index(&index, 1.5);
    }

    #[test]
    fn spatial_tensor_radius_matches_full_scan() {
        let (x, y, z) = sample_cloud();
        let tree = KdTree::from_slices(&x, &y, &z);

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for index in 0..x.len() {
            builder.push_point([x[index], y[index], z[index]]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = cloud.spatial_tensor_chunks(2).unwrap();

        let mut tensor_out = Vec::new();
        radius_search_spatial_tensor(&tree, &x, &y, &z, &tensor, 1.5, &mut tensor_out);

        let mut full_out = Vec::new();
        tree.radius_search_chunk_into(&x, &y, &z, 0..x.len(), 1.5, &mut full_out);

        tensor_out.sort_by(|a, b| (a.0, a.1.index).cmp(&(b.0, b.1.index)));
        full_out.sort_by(|a, b| (a.0, a.1.index).cmp(&(b.0, b.1.index)));
        assert_eq!(tensor_out, full_out);
    }

    #[test]
    fn spatial_tensor_nearest_k_matches_full_scan() {
        let (x, y, z) = sample_cloud();
        let tree = KdTree::from_slices(&x, &y, &z);

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        for index in 0..x.len() {
            builder.push_point([x[index], y[index], z[index]]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tensor = cloud.spatial_tensor_chunks(2).unwrap();

        let mut tensor_out = Vec::new();
        nearest_k_spatial_tensor(&tree, &x, &y, &z, &tensor, 2, &mut tensor_out);

        let mut full_out = Vec::new();
        tree.nearest_k_chunk_into(&x, &y, &z, 0..x.len(), 2, &mut full_out);

        tensor_out.sort_by(|a, b| {
            a.0.cmp(&b.0).then(
                a.1.distance_squared
                    .partial_cmp(&b.1.distance_squared)
                    .unwrap_or(std::cmp::Ordering::Equal),
            ).then(a.1.index.cmp(&b.1.index))
        });
        full_out.sort_by(|a, b| {
            a.0.cmp(&b.0).then(
                a.1.distance_squared
                    .partial_cmp(&b.1.distance_squared)
                    .unwrap_or(std::cmp::Ordering::Equal),
            ).then(a.1.index.cmp(&b.1.index))
        });
        assert_eq!(tensor_out, full_out);
    }
}
