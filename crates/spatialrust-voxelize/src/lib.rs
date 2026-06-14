//! Voxel occupancy / feature grids for SpatialRust.
//!
//! Turns a point cloud into a dense 3D grid — the tensor representation learned
//! models consume (3D CNNs, occupancy networks). The grid is row-major in
//! `(z, y, x)` order so it reshapes directly to an `(nz, ny, nx)` array.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "voxelize-occupancy")]
mod occupancy;

#[cfg(feature = "voxelize-occupancy")]
pub use occupancy::{voxelize, OccupancyGrid, VoxelFill, VoxelGridConfig};
