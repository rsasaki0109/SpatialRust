//! Audited Arrow C Data / Stream / Device adapters for SpatialRust records.
//!
//! `spatialrust-core` remains Arrow-free. This crate owns the C ABI and all
//! `unsafe` release callbacks at the export/import boundary.

#![warn(missing_docs)]

#[cfg(feature = "arrow-c-data")]
mod cdata;
#[cfg(feature = "arrow-c-data")]
mod error;
#[cfg(feature = "arrow-c-device")]
mod device;
#[cfg(feature = "arrow-c-stream")]
mod stream;

#[cfg(feature = "arrow-c-data")]
pub use cdata::{
    export_point_cloud_c_data, import_point_cloud_c_data, ArrowArray, ArrowSchema,
    ExportedArrowArray, ExportedArrowSchema,
};
#[cfg(feature = "arrow-c-data")]
pub use error::{ArrowBridgeError, ArrowBridgeResult};
#[cfg(feature = "arrow-c-device")]
pub use device::{
    export_point_cloud_device_array, import_point_cloud_device_array, ArrowDeviceArray,
    ArrowDeviceType, ExportedArrowDeviceArray,
};
#[cfg(feature = "arrow-c-stream")]
pub use stream::{export_record_source_c_stream, ArrowArrayStream, ExportedArrowArrayStream};
