//! PLY (Polygon File Format) readers and writers for point clouds.

mod header;
mod reader;
mod schema;
mod writer;

pub use header::{PlyFormat, PlyHeader, PlyProperty, PlyPropertyKind};
#[cfg(feature = "streaming")]
pub use reader::PlyChunkSource;
pub use reader::{read_ply, read_ply_file, PlyReader};
pub use schema::{infer_property_semantic, ply_property_from_field, schema_from_ply_properties};
#[cfg(feature = "streaming")]
pub use writer::PlyChunkSink;
pub use writer::{write_ply, write_ply_file, PlyWriteFormat, PlyWriter};
