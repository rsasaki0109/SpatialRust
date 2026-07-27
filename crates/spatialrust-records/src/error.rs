//! Errors for versioned spatial records and chunked streams.

/// Result type for record and stream operations.
pub type RecordsResult<T> = Result<T, RecordsError>;

/// Shared record/stream failures.
#[derive(Debug, thiserror::Error)]
pub enum RecordsError {
    /// Invalid public configuration.
    #[error("invalid record configuration: {0}")]
    InvalidConfiguration(String),
    /// Schema identifiers or versions are incompatible.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    /// A required field is missing during migration.
    #[error("missing required field `{0}`")]
    MissingField(String),
    /// A tracked allocation would exceed the configured streaming memory budget.
    #[error(
        "streaming memory budget exceeded: requested {requested} bytes with {current} bytes \
         already reserved (limit {limit} bytes)"
    )]
    MemoryBudgetExceeded {
        /// Additional bytes requested by the operation.
        requested: u64,
        /// Bytes already reserved when the request was made.
        current: u64,
        /// Configured hard limit.
        limit: u64,
    },
    /// A streaming counter exceeded the representable receipt range.
    #[error("streaming receipt counter overflow: {0}")]
    ReceiptOverflow(String),
    /// A versioned streaming receipt did not satisfy its schema contract.
    #[error("invalid streaming receipt: {0}")]
    InvalidReceipt(String),
    /// Cooperative cancellation was observed at a streaming boundary.
    #[error("streaming operation cancelled")]
    Cancelled,
    /// A source emitted a chunk that violates its declared stream contract.
    #[error("invalid streaming chunk: {0}")]
    InvalidChunk(String),
    /// A background stream ended without an explicit end marker.
    #[error("streaming channel closed unexpectedly")]
    StreamClosed,
    /// Wrapped core spatial failure.
    #[error(transparent)]
    Spatial(#[from] spatialrust_core::SpatialError),
}
