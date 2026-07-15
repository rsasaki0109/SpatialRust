//! Errors for Arrow C ABI bridges.

/// Result type for Arrow bridge operations.
pub type ArrowBridgeResult<T> = Result<T, ArrowBridgeError>;

/// Failures converting SpatialRust records to or from Arrow C ABI values.
#[derive(Debug, thiserror::Error)]
pub enum ArrowBridgeError {
    /// Invalid configuration or unsupported layout.
    #[error("invalid Arrow bridge configuration: {0}")]
    InvalidConfiguration(String),
    /// Schema / dtype mismatch between Arrow and SpatialRust.
    #[error("Arrow schema mismatch: {0}")]
    SchemaMismatch(String),
    /// A required null pointer was encountered.
    #[error("null Arrow pointer: {0}")]
    NullPointer(String),
    /// Wrapped core error.
    #[error(transparent)]
    Spatial(#[from] spatialrust_core::SpatialError),
    /// Wrapped records error.
    #[error(transparent)]
    Records(#[from] spatialrust_records::RecordsError),
}
