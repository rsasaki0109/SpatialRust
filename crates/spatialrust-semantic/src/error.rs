//! Semantic crate errors.

/// Result type for semantic operations.
pub type SemanticResult<T> = Result<T, SemanticError>;

/// Failures in embeddings and semantic search.
#[derive(Debug, thiserror::Error)]
pub enum SemanticError {
    /// Invalid configuration.
    #[error("invalid semantic configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing entity or embedding.
    #[error("missing `{0}`")]
    Missing(String),
}
