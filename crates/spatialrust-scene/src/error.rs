//! Scene reconstruction errors.

/// Result type for scene operations.
pub type SceneResult<T> = Result<T, SceneError>;

/// Failures while building or extracting scenes.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    /// Invalid configuration.
    #[error("invalid scene configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing data for an operation.
    #[error("missing `{0}`")]
    Missing(String),
}
