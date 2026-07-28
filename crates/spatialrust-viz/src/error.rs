//! Visualization contract errors.

/// Result type used by visualization contracts.
pub type VizResult<T> = Result<T, VizError>;

/// Errors returned when visualization input violates a public contract.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VizError {
    /// A slice length or indexed geometry shape is invalid.
    #[error("invalid geometry: {0}")]
    InvalidGeometry(String),
    /// A style value is non-finite or outside its accepted range.
    #[error("invalid style: {0}")]
    InvalidStyle(String),
    /// A camera projection or view is invalid.
    #[error("invalid camera: {0}")]
    InvalidCamera(String),
    /// A layer identifier is empty or already present.
    #[error("invalid layer: {0}")]
    InvalidLayer(String),
    /// A transfer event is incomplete or invalid.
    #[error("invalid transfer: {0}")]
    InvalidTransfer(String),
}
