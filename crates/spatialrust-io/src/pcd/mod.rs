//! PCD (Point Cloud Data) readers and writers.

mod header;
mod reader;
mod schema;
mod writer;

pub use header::{PcdDataKind, PcdHeader};
#[cfg(feature = "streaming")]
pub use reader::PcdChunkSource;
pub use reader::{read_pcd, read_pcd_file, PcdReader};
pub use schema::{infer_field_semantic, schema_from_pcd_fields, PcdFieldSpec, PcdType};
#[cfg(feature = "streaming")]
pub use writer::PcdChunkSink;
pub use writer::{write_pcd, write_pcd_file, PcdWriteFormat, PcdWriter};
