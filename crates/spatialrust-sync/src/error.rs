//! Errors for sensor time and frame-graph operations.

/// Result type for sync crate operations.
pub type SyncResult<T> = Result<T, SyncError>;

/// Failures in time sync, frame graphs, and replay.
#[derive(Debug, thiserror::Error)]
pub enum SyncError {
    /// Invalid configuration.
    #[error("invalid sync configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing frame or topic.
    #[error("missing `{0}`")]
    #[allow(dead_code)]
    Missing(String),
    /// No transform path between frames.
    #[error("no transform path from `{from}` to `{to}`")]
    NoTransformPath {
        /// Source frame.
        from: String,
        /// Target frame.
        to: String,
    },
    /// Wrapped records error.
    #[error(transparent)]
    Records(#[from] spatialrust_records::RecordsError),
    /// Wrapped spatial error.
    #[error(transparent)]
    Spatial(#[from] spatialrust_core::SpatialError),
    /// Filesystem / IO failure.
    #[error("IO error: {0}")]
    Io(String),
    /// MCAP codec failure.
    #[error("MCAP error: {0}")]
    Mcap(String),
}
