/// Result type returned by the wgpu rendering backend.
pub type RenderResult<T> = Result<T, RenderError>;

/// Errors returned by explicit rendering operations.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RenderError {
    /// Geometry is too large for the renderer's addressable counts or buffers.
    #[error("geometry size is unsupported: {0}")]
    GeometrySize(String),
    /// A GPU resource belongs to another renderer runtime.
    #[error("runtime mismatch: {0}")]
    RuntimeMismatch(String),
    /// A visualization transfer receipt could not be constructed.
    #[error("transfer receipt: {0}")]
    Transfer(String),
    /// A caller-requested GPU readback failed.
    #[error("readback failed: {0}")]
    Readback(String),
}
