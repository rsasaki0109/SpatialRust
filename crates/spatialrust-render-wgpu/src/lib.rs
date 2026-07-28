//! Explicit wgpu backend for SpatialRust visualization contracts.
//!
//! Geometry enters this crate only through caller-requested upload methods.
//! Uploads return a byte-exact transfer receipt, and device buffers remain
//! resident until explicitly recycled or dropped.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod geometry;
mod runtime;

pub use error::{RenderError, RenderResult};
pub use geometry::{GpuGeometry, GpuGeometryKind};
pub use runtime::WgpuRenderer;
