//! Point cloud comparison metrics for SpatialRust.
//!
//! Distance metrics quantify how close two clouds are — the standard way to
//! score registration, reconstruction, or downsampling against a reference.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "metrics-distance")]
mod distance;

#[cfg(feature = "metrics-distance")]
pub use distance::{chamfer_distance, cloud_distances, hausdorff_distance, CloudDistances};
