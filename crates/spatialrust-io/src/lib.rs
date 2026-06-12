//! Point cloud and spatial data IO for SpatialRust.
//!
//! Format support is feature-gated. Core reader/writer traits live here.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod format;
mod options;
mod traits;

#[cfg(feature = "io-pcd")]
/// PCD (Point Cloud Data) readers and writers.
pub mod pcd;

#[cfg(feature = "io-ply")]
/// PLY (Polygon File Format) readers and writers.
pub mod ply;

#[cfg(feature = "io-las")]
/// LAS/LAZ (ASPRS LiDAR) readers and writers.
pub mod las;

#[cfg(feature = "io-e57")]
/// E57 (ASTM E2807) readers and writers.
pub mod e57;

#[cfg(feature = "io-copc")]
/// COPC (Cloud Optimized Point Cloud) readers and writers.
pub mod copc;

pub use error::IoError;
pub use format::{
    detect_point_cloud_format, read_point_cloud_file, read_point_cloud_file_with_format,
    write_point_cloud_file, write_point_cloud_file_with_format, PointCloudFileFormat,
};
pub use options::{ReadOptions, WriteOptions};
pub use traits::{PointReader, PointSink, PointStream, PointWriter};

#[cfg(feature = "io-pcd")]
pub use pcd::{
    infer_field_semantic, read_pcd, read_pcd_file, schema_from_pcd_fields, write_pcd,
    write_pcd_file, PcdDataKind, PcdFieldSpec, PcdHeader, PcdReader, PcdType, PcdWriteFormat,
    PcdWriter,
};

#[cfg(feature = "io-ply")]
pub use ply::{
    infer_property_semantic, ply_property_from_field, read_ply, read_ply_file,
    schema_from_ply_properties, write_ply, write_ply_file, PlyFormat, PlyHeader, PlyProperty,
    PlyPropertyKind, PlyReader, PlyWriteFormat, PlyWriter,
};

#[cfg(feature = "io-las")]
pub use las::{
    infer_las_field_semantic, read_las, read_las_file, schema_for_las_header,
    schema_from_point_cloud, write_las, write_las_file, LasReader, LasWriteFormat, LasWriter,
};

#[cfg(feature = "io-e57")]
pub use e57::{
    e57_prototype_from_schema, read_e57, read_e57_file, schema_for_e57_pointcloud,
    schema_from_point_cloud as e57_schema_from_point_cloud, write_e57, write_e57_file, E57Reader,
    E57Writer,
};

#[cfg(feature = "io-copc")]
pub use copc::{
    copc_level_for_resolution, read_copc, read_copc_file, read_copc_file_in_bounds,
    read_copc_file_info, read_copc_file_with_query, write_copc, write_copc_file, CopcBounds,
    CopcFileInfo, CopcQuery, CopcReader, CopcWriter,
};

#[cfg(feature = "io-copc-http")]
pub use copc::{
    read_copc_url, read_copc_url_info, read_copc_url_with_query, HttpByteSource,
};
