use spatialrust_image::ImageError;

/// Result type for image and vision operations.
pub type VisionResult<T> = Result<T, VisionError>;

/// Errors raised by image and vision algorithms.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum VisionError {
    /// Image construction or layout validation failed.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// An operation received unusable image dimensions.
    #[error("invalid image dimensions: {0}")]
    InvalidDimensions(String),
    /// A numeric parameter was invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Input collections or maps had incompatible shapes.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// A geometric transform could not be inverted.
    #[error("transform is singular")]
    SingularTransform,
}
