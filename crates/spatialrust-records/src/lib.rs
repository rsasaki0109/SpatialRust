//! Versioned spatial records, schema evolution, and chunked host streams.
//!
//! This crate stays Arrow-free. Arrow C Data/Stream/Device live in
//! `spatialrust-arrow` behind independent features.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod migrate;
mod record;
mod schema;
mod stream;

pub use error::{RecordsError, RecordsResult};
pub use migrate::{migrate_record, FieldFill, MigrationPolicy};
pub use record::SpatialRecord;
pub use schema::{
    compare_schemas, CompatVerdict, SchemaCompatReport, SchemaDescriptor, SchemaId, SchemaVersion,
};
pub use stream::{
    MemoryChunkSink, MemoryChunkSource, SpatialRecordSink, SpatialRecordSource,
};
