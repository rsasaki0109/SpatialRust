//! Spatial search structures for SpatialRust.
//!
//! KDTree, hash grid, and octree implementations live in this crate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod brute;
mod traits;
mod uniform_grid;

#[cfg(feature = "search-kdtree")]
mod kdtree;

#[cfg(feature = "search-graph")]
mod graph;

pub use brute::{brute_force_knn, brute_force_radius, BruteForceIndex};
pub use traits::{NearestNeighborIndex, Neighbor, RadiusSearchIndex, SpatialIndex};
pub use uniform_grid::{
    build_grid, euclidean_cluster_roots, grid_bounds, uniform_grid_fits, MAX_UNIFORM_GRID_CELLS,
};

#[cfg(feature = "search-kdtree")]
pub use kdtree::KdTree;

#[cfg(feature = "search-graph")]
pub use graph::{knn_graph, radius_graph, NeighborGraph};
