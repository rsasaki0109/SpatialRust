/// Result returned by viewer operations.
pub type ViewerResult<T> = Result<T, ViewerError>;

/// Viewer validation or native-shell failure.
#[derive(Debug, thiserror::Error)]
pub enum ViewerError {
    /// A state or interaction value was invalid.
    #[error("invalid viewer state: {0}")]
    InvalidState(String),
    /// A referenced layer was not present.
    #[error("unknown viewer layer `{0}`")]
    UnknownLayer(String),
    /// Overlay input was malformed.
    #[error("invalid debug overlay: {0}")]
    InvalidOverlay(String),
    /// Native event-loop or window creation failed.
    #[cfg(feature = "native")]
    #[error("native viewer failure: {0}")]
    Native(String),
    /// A visualization contract rejected the requested value.
    #[error(transparent)]
    Viz(#[from] spatialrust_viz::VizError),
}
