//! Runtime crate errors.

/// Result type for runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Failures in bounded pipelines and adapters.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Invalid configuration.
    #[error("invalid runtime configuration: {0}")]
    InvalidConfiguration(String),
    /// Backpressure / capacity exceeded.
    #[error("pipeline capacity exceeded: {0}")]
    CapacityExceeded(String),
    /// Explicit failure with diagnostic code.
    #[error("runtime failure `{code}`: {message}")]
    Failure {
        /// Machine-readable code.
        code: String,
        /// Human-readable message.
        message: String,
    },
}
