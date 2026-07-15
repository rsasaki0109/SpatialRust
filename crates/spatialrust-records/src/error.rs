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
    /// Wrapped core spatial failure.
    #[error(transparent)]
    Spatial(#[from] spatialrust_core::SpatialError),
}
