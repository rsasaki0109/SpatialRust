//! COPC (Cloud Optimized Point Cloud) readers and writers.

mod query;

#[cfg(feature = "io-copc-http")]
mod http;

mod reader;
mod writer;

pub use copc_writer::CopcWriterParams;
pub use query::{copc_level_for_resolution, CopcBounds, CopcFileInfo, CopcQuery};
#[cfg(feature = "streaming")]
pub use reader::CopcChunkSource;
pub use reader::{
    read_copc, read_copc_file, read_copc_file_in_bounds, read_copc_file_info,
    read_copc_file_with_query, CopcReader,
};
pub use writer::{write_copc, write_copc_file, write_copc_file_with_params, CopcWriter};
#[cfg(feature = "streaming")]
pub use writer::{write_copc_stream, CopcStreamingWriteReceipt};

#[cfg(feature = "io-copc-http")]
pub use http::{read_copc_url, read_copc_url_info, read_copc_url_with_query, HttpByteSource};
