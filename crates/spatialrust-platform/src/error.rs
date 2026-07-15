//! Platform crate errors.

/// Result type for platform operations.
pub type PlatformResult<T> = Result<T, PlatformError>;

/// Failures in stability/conformance bookkeeping.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    /// Invalid configuration.
    #[error("invalid platform configuration: {0}")]
    InvalidConfiguration(String),
}
