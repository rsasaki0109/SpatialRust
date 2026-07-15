//! Distributed execution errors.

/// Result type for distribute operations.
pub type DistributeResult<T> = Result<T, DistributeError>;

/// Failures in partition graphs and transfers.
#[derive(Debug, thiserror::Error)]
pub enum DistributeError {
    /// Invalid configuration.
    #[error("invalid distribute configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing node or transfer.
    #[error("missing `{0}`")]
    Missing(String),
}
