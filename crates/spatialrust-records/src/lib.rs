//! Versioned spatial records, schema evolution, and chunked host streams.
//!
//! This crate stays Arrow-free. Arrow C Data/Stream/Device live in
//! `spatialrust-arrow` behind independent features.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod bounded;
mod error;
mod migrate;
mod provenance;
mod record;
mod schema;
mod stream;
mod streaming;

pub use bounded::{
    record_storage_bytes, BoundedSpatialRecordSink, BoundedSpatialRecordSource, ChunkIdentity,
    LegacyBoundedSink, LegacyBoundedSource, PrefetchRecordSource, RecordBounds3,
    RecyclingMemoryChunkSource, SpatialRecordChunk,
};
pub use error::{RecordsError, RecordsResult};
pub use migrate::{migrate_record, FieldFill, MigrationPolicy};
pub use provenance::{RecordProvenance, RECORD_PROVENANCE_VERSION};
pub use record::SpatialRecord;
pub use schema::{
    compare_schemas, CompatVerdict, SchemaCompatReport, SchemaDescriptor, SchemaId, SchemaVersion,
};
pub use stream::{MemoryChunkSink, MemoryChunkSource, SpatialRecordSink, SpatialRecordSource};
pub use streaming::{
    canonical_streaming_workloads, CancellationToken, MemoryBudget, MemoryReservation,
    MemorySnapshot, MemoryTracker, StreamOptions, StreamOrdering, StreamingPhaseReceipt,
    StreamingReceipt, StreamingTransferDirection, StreamingTransferReceipt, StreamingWorkload,
    DEFAULT_STREAM_CHUNK_POINTS, DEFAULT_STREAM_MEMORY_BUDGET_BYTES, STREAMING_RECEIPT_SCHEMA,
    STREAMING_RECEIPT_VERSION,
};
