//! Spatial search structures for SpatialRust.
//!
//! KDTree, hash grid, and octree implementations live in this crate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod brute;
mod traits;

#[cfg(feature = "search-kdtree")]
mod kdtree;

#[cfg(feature = "search-graph")]
mod graph;

pub use brute::{brute_force_knn, brute_force_radius, BruteForceIndex};
pub use traits::{NearestNeighborIndex, Neighbor, RadiusSearchIndex, SpatialIndex};

#[cfg(feature = "search-kdtree")]
pub use kdtree::KdTree;

#[cfg(feature = "search-graph")]
pub use graph::{knn_graph, radius_graph, NeighborGraph};
