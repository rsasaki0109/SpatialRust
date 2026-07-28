//! Backend-independent visualization contracts.
//!
//! This crate describes borrowed geometry, cameras, visual styles, layers, and
//! explicit transfer receipts. It does not open windows, allocate GPU resources,
//! or copy geometry between devices.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod camera;
mod color;
mod error;
mod geometry;
mod layer;
mod style;
mod transfer;

pub use camera::{Camera, Projection};
pub use color::LinearRgba;
pub use error::{VizError, VizResult};
#[cfg(feature = "core")]
pub use geometry::point_cloud_positions;
pub use geometry::{
    LineListView, PointCloudView, PositionColumns3, Rgb8Columns, ScalarColumn, TriangleMeshView,
    VisualPrimitive,
};
pub use layer::{LayerId, VisualLayer, VisualScene};
pub use style::{ColorMap, PointColor, PointStyle, VisualStyle};
pub use transfer::{
    DeviceIdentity, TransferDirection, TransferEvent, TransferReceipt, VisualResidency,
};
