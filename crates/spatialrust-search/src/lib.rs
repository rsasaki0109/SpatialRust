//! Spatial search structures for SpatialRust.
//!
//! KDTree, hash grid, and octree implementations live in this crate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod brute;
mod chunked;
mod staging;
mod traits;
mod uniform_grid;

#[cfg(feature = "search-kdtree")]
mod kdtree;

#[cfg(feature = "parallel")]
mod parallel_chunked;

#[cfg(feature = "search-graph")]
mod graph;

pub use brute::{brute_force_knn, brute_force_radius, BruteForceIndex};
pub use chunked::{
    nearest_k_spatial_tensor, radius_search_spatial_tensor, ChunkedNearestNeighborIndex,
    ChunkedRadiusSearchIndex, ChunkQueryRange,
};
pub use staging::{
    parallel_index_for_each, parallel_index_ranges, parallel_worker_count,
    parallel_worker_count_with_chunk, PARALLEL_STAGING_MIN_POINTS,
};
pub use traits::{NearestNeighborIndex, Neighbor, RadiusSearchIndex, SpatialIndex};
pub use uniform_grid::{
    build_grid, euclidean_cluster_roots, grid_bounds, uniform_grid_fits, MAX_UNIFORM_GRID_CELLS,
};

#[cfg(feature = "parallel")]
pub use parallel_chunked::{
    nearest_k_spatial_tensor_parallel, nearest_k_spatial_tensor_parallel_into,
    radius_search_spatial_tensor_parallel, radius_search_spatial_tensor_parallel_into,
    PARALLEL_CHUNK_QUERY_MIN_POINTS,
};

#[cfg(feature = "search-kdtree")]
pub use kdtree::KdTree;

#[cfg(feature = "search-graph")]
pub use graph::{knn_graph, radius_graph, NeighborGraph};
