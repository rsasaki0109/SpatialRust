use crate::{Isometry3, Mat3, Real};

/// 3x3 symmetric covariance matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cov3<T: Real> {
    /// Symmetric covariance stored in row-major form.
    pub matrix: Mat3<T>,
}

/// Pose with optional translation covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pose3<T: Real> {
    /// Rigid pose.
    pub isometry: Isometry3<T>,
    /// Optional translation covariance.
    pub translation_covariance: Option<Cov3<T>>,
}

impl<T: Real> Cov3<T> {
    /// Creates a covariance matrix from a symmetric 3x3 matrix.
    #[must_use]
    pub const fn new(matrix: Mat3<T>) -> Self {
        Self { matrix }
    }
}

impl<T: Real> Pose3<T> {
    /// Creates a pose without covariance.
    #[must_use]
    pub const fn new(isometry: Isometry3<T>) -> Self {
        Self { isometry, translation_covariance: None }
    }

    /// Creates a pose with translation covariance.
    #[must_use]
    pub const fn with_covariance(isometry: Isometry3<T>, covariance: Cov3<T>) -> Self {
        Self { isometry, translation_covariance: Some(covariance) }
    }
}
