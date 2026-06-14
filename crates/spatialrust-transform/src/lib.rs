//! Geometric transforms and cloud utilities for SpatialRust.
//!
//! Affine transforms, recentering and scale normalization, cloud merging, and
//! bounding-volume computation — the small operations every pipeline needs.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "transform-ops")]
mod bounds;
#[cfg(feature = "transform-ops")]
mod ops;

#[cfg(feature = "transform-ops")]
pub use bounds::{Aabb, Obb};
#[cfg(feature = "transform-ops")]
pub use ops::{
    apply_transform, bounding_box, centroid, merge_clouds, normalize_unit_sphere,
    oriented_bounding_box, recenter, scale_cloud,
};
