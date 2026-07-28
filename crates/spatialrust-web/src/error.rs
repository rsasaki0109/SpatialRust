/// Web adapter result.
pub type WebResult<T> = Result<T, WebError>;

/// Portable Web state, range, or rendering error.
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    /// Serialized state/input was invalid.
    #[error("invalid Web viewer state: {0}")]
    InvalidState(String),
    /// A range or hard budget was invalid/exceeded.
    #[error("bounded Web range failure: {0}")]
    Range(String),
    /// Viewer state reducer rejected input.
    #[error("viewer input failed: {0}")]
    Viewer(String),
    /// WebGPU rendering failed.
    #[cfg(feature = "webgpu")]
    #[error("WebGPU bridge failed: {0}")]
    WebGpu(String),
}
