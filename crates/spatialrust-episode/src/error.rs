//! Episode crate errors.

/// Result type for episode workflows.
pub type EpisodeResult<T> = Result<T, EpisodeError>;

/// Failures in episode construction and evaluation.
#[derive(Debug, thiserror::Error)]
pub enum EpisodeError {
    /// Invalid configuration.
    #[error("invalid episode configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing episode asset.
    #[error("missing `{0}`")]
    Missing(String),
}
