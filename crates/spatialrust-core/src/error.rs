use thiserror::Error;

/// Result type used across SpatialRust crates.
pub type SpatialResult<T> = Result<T, SpatialError>;

/// Core error type for SpatialRust.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SpatialError {
    /// A required field is missing from the point cloud schema.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// Schema validation failed.
    #[error("schema validation failed: {0}")]
    SchemaValidation(String),

    /// Invalid argument supplied by the caller.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// IO-related error surfaced through higher-level crates.
    #[error("io error: {0}")]
    Io(String),

    /// Buffer length does not match point cloud length.
    #[error("buffer length mismatch: expected {expected}, found {found}")]
    BufferLengthMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        found: usize,
    },

    /// Unsupported dtype for the requested operation.
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDType(crate::DType),
}
