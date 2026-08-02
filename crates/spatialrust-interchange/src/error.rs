//! Interchange errors.

/// Result type for interchange adapters.
pub type InterchangeResult<T> = Result<T, InterchangeError>;

/// Failures converting scenes to/from interchange formats.
#[derive(Debug, thiserror::Error)]
pub enum InterchangeError {
    /// Invalid configuration or payload.
    #[error("invalid interchange configuration: {0}")]
    InvalidConfiguration(String),
    /// Unsupported schema subset.
    #[error("unsupported interchange feature: {0}")]
    Unsupported(String),
}
