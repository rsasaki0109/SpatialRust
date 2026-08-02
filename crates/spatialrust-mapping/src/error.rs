//! Mapping crate errors.

/// Result type for mapping operations.
pub type MappingResult<T> = Result<T, MappingError>;

/// Failures in trajectories and pose graphs.
#[derive(Debug, thiserror::Error)]
pub enum MappingError {
    /// Invalid configuration.
    #[error("invalid mapping configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing trajectory/node/edge.
    #[error("missing `{0}`")]
    Missing(String),
    /// Graph inconsistency.
    #[error("pose graph error: {0}")]
    Graph(String),
}
