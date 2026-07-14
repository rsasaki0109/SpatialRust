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
    /// A descriptor buffer does not match its declared row layout.
    #[error("descriptor layout {rows}x{width} requires {expected} values, found {actual}")]
    DescriptorLayout {
        /// Descriptor row count.
        rows: usize,
        /// Values or bytes per descriptor.
        width: usize,
        /// Required flat storage length.
        expected: usize,
        /// Supplied flat storage length.
        actual: usize,
    },
    /// Keypoint and descriptor row counts differ.
    #[error("feature set has {keypoints} keypoints but {descriptors} descriptor rows")]
    FeatureCountMismatch {
        /// Number of keypoints.
        keypoints: usize,
        /// Number of descriptors.
        descriptors: usize,
    },
    /// A feature match references a keypoint outside its collection.
    #[error(
        "feature match index is out of bounds: query {query}/{queries}, train {train}/{trains}"
    )]
    MatchIndexOutOfBounds {
        /// Query index.
        query: usize,
        /// Query feature count.
        queries: usize,
        /// Train index.
        train: usize,
        /// Train feature count.
        trains: usize,
    },
    /// Inlier and residual arrays do not match the correspondence count.
    #[error(
        "geometry result for {correspondences} correspondences has {inliers} inlier flags and {residuals} residuals"
    )]
    GeometryResultLayout {
        /// Number of input correspondences.
        correspondences: usize,
        /// Number of inlier flags.
        inliers: usize,
        /// Number of residual values.
        residuals: usize,
    },
    /// A geometric transform could not be inverted.
    #[error("transform is singular")]
    SingularTransform,
}
