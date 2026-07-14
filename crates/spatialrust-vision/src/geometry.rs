//! Checked data contracts for calibrated and uncalibrated multiview geometry.

use spatialrust_camera::CameraIntrinsics;
use spatialrust_math::{Mat3, Vec2, Vec3};

use crate::{VisionError, VisionResult};

/// One ordered pixel correspondence from a source image to a target image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCorrespondence2 {
    source: Vec2<f64>,
    target: Vec2<f64>,
}

impl PointCorrespondence2 {
    /// Creates a correspondence with finite pixel coordinates.
    pub fn try_new(source: Vec2<f64>, target: Vec2<f64>) -> VisionResult<Self> {
        if ![source.x, source.y, target.x, target.y].into_iter().all(f64::is_finite) {
            return Err(VisionError::InvalidParameter(
                "2D correspondence coordinates must be finite".into(),
            ));
        }
        Ok(Self { source, target })
    }

    /// Returns the source-image pixel.
    pub const fn source(self) -> Vec2<f64> {
        self.source
    }

    /// Returns the target-image pixel.
    pub const fn target(self) -> Vec2<f64> {
        self.target
    }
}

/// Calibrated pinhole intrinsic matrix and its analytic inverse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMatrix3 {
    matrix: Mat3<f64>,
    inverse: Mat3<f64>,
}

impl CameraMatrix3 {
    /// Builds `K` and `K^-1` from validated camera intrinsics.
    #[must_use]
    pub fn from_intrinsics(intrinsics: CameraIntrinsics) -> Self {
        let matrix = Mat3::from_rows(
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        );
        let inverse = Mat3::from_rows(
            [1.0 / intrinsics.fx, 0.0, -intrinsics.cx / intrinsics.fx],
            [0.0, 1.0 / intrinsics.fy, -intrinsics.cy / intrinsics.fy],
            [0.0, 0.0, 1.0],
        );
        Self { matrix, inverse }
    }

    /// Returns the row-major intrinsic matrix `K`.
    pub const fn matrix(self) -> Mat3<f64> {
        self.matrix
    }

    /// Returns the row-major inverse intrinsic matrix `K^-1`.
    pub const fn inverse(self) -> Mat3<f64> {
        self.inverse
    }

    /// Converts a pixel into the homogeneous normalized camera plane.
    #[must_use]
    pub fn normalize_pixel(self, pixel: Vec2<f64>) -> Vec3<f64> {
        self.inverse.mul_vec3(Vec3::new(pixel.x, pixel.y, 1.0))
    }
}

fn validate_projective_matrix(matrix: Mat3<f64>, name: &str) -> VisionResult<Mat3<f64>> {
    if matrix.m.iter().flatten().any(|value| !value.is_finite()) {
        return Err(VisionError::InvalidParameter(format!(
            "{name} matrix elements must be finite"
        )));
    }
    let norm_squared = matrix.m.iter().flatten().map(|value| value * value).sum::<f64>();
    if norm_squared <= f64::EPSILON {
        return Err(VisionError::InvalidParameter(format!("{name} matrix must not be all zero")));
    }
    Ok(matrix)
}

macro_rules! projective_model {
    ($name:ident, $summary:literal, $label:literal) => {
        #[doc = $summary]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name(Mat3<f64>);

        impl $name {
            /// Creates a checked finite, non-zero projective matrix.
            pub fn try_new(matrix: Mat3<f64>) -> VisionResult<Self> {
                Ok(Self(validate_projective_matrix(matrix, $label)?))
            }

            /// Returns the row-major matrix without changing its arbitrary scale.
            pub const fn matrix(self) -> Mat3<f64> {
                self.0
            }
        }
    };
}

projective_model!(Homography3, "A pixel-to-pixel planar projective transform.", "homography");
projective_model!(
    Fundamental3,
    "An uncalibrated two-view epipolar constraint matrix.",
    "fundamental"
);
projective_model!(Essential3, "A calibrated two-view epipolar constraint matrix.", "essential");

/// Deterministic robust-model sampling and inlier classification settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustEstimationOptions {
    /// Maximum accepted geometric residual in pixels or normalized units.
    pub threshold: f64,
    /// Desired probability of sampling an outlier-free minimal set.
    pub confidence: f64,
    /// Hard iteration limit.
    pub max_iterations: usize,
    /// Reproducible pseudo-random sampling seed.
    pub seed: u64,
}

impl Default for RobustEstimationOptions {
    fn default() -> Self {
        Self { threshold: 1.0, confidence: 0.99, max_iterations: 2_000, seed: 0 }
    }
}

impl RobustEstimationOptions {
    /// Validates thresholds, confidence, and the iteration budget.
    pub fn validate(self) -> VisionResult<Self> {
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "geometry threshold must be finite and positive".into(),
            ));
        }
        if !self.confidence.is_finite() || self.confidence <= 0.0 || self.confidence >= 1.0 {
            return Err(VisionError::InvalidParameter(
                "geometry confidence must be finite and in (0, 1)".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(VisionError::InvalidParameter(
                "geometry max_iterations must be positive".into(),
            ));
        }
        Ok(self)
    }
}

/// A geometric model paired with one inlier decision and residual per input row.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricEstimate<Model> {
    model: Model,
    inliers: Vec<bool>,
    residuals: Vec<f64>,
}

/// Target-camera pose expressed in the source-camera coordinate frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativePose {
    rotation: Mat3<f64>,
    translation: Vec3<f64>,
}

impl RelativePose {
    /// Creates a checked proper rotation and finite non-zero translation.
    pub fn try_new(rotation: Mat3<f64>, translation: Vec3<f64>) -> VisionResult<Self> {
        if rotation.m.iter().flatten().any(|value| !value.is_finite())
            || ![translation.x, translation.y, translation.z].into_iter().all(f64::is_finite)
        {
            return Err(VisionError::InvalidParameter(
                "relative pose elements must be finite".into(),
            ));
        }
        let orthogonality = rotation.transpose().mul_mat3(rotation);
        let identity = Mat3::<f64>::identity();
        let maximum_error = orthogonality
            .m
            .iter()
            .flatten()
            .zip(identity.m.iter().flatten())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        if maximum_error > 1e-6 || determinant(rotation) < 1.0 - 1e-6 {
            return Err(VisionError::InvalidParameter(
                "relative pose rotation must be a proper orthonormal matrix".into(),
            ));
        }
        if translation.length() <= f64::EPSILON {
            return Err(VisionError::InvalidParameter(
                "relative pose translation must be non-zero".into(),
            ));
        }
        Ok(Self { rotation, translation })
    }

    /// Returns the source-to-target rotation.
    pub const fn rotation(self) -> Mat3<f64> {
        self.rotation
    }

    /// Returns the source-to-target translation, whose scale may be arbitrary.
    pub const fn translation(self) -> Vec3<f64> {
        self.translation
    }
}

/// One two-view triangulation with cheirality and reprojection diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangulatedPoint {
    position: Vec3<f64>,
    source_depth: f64,
    target_depth: f64,
    reprojection_error: f64,
}

impl TriangulatedPoint {
    pub(crate) fn try_new(
        position: Vec3<f64>,
        source_depth: f64,
        target_depth: f64,
        reprojection_error: f64,
    ) -> VisionResult<Self> {
        if ![position.x, position.y, position.z, source_depth, target_depth, reprojection_error]
            .into_iter()
            .all(f64::is_finite)
            || reprojection_error < 0.0
        {
            return Err(VisionError::InvalidParameter(
                "triangulation values must be finite and error non-negative".into(),
            ));
        }
        Ok(Self { position, source_depth, target_depth, reprojection_error })
    }

    /// Returns the point in source-camera coordinates.
    pub const fn position(self) -> Vec3<f64> {
        self.position
    }

    /// Returns its signed source-camera depth.
    pub const fn source_depth(self) -> f64 {
        self.source_depth
    }

    /// Returns its signed target-camera depth.
    pub const fn target_depth(self) -> f64 {
        self.target_depth
    }

    /// Returns the mean normalized-plane reprojection distance.
    pub const fn reprojection_error(self) -> f64 {
        self.reprojection_error
    }

    /// Returns whether the point lies in front of both cameras.
    pub fn has_positive_depth(self) -> bool {
        self.source_depth > 0.0 && self.target_depth > 0.0
    }
}

/// Essential-matrix pose disambiguation and per-correspondence triangulation.
#[derive(Clone, Debug, PartialEq)]
pub struct RelativePoseEstimate {
    pose: RelativePose,
    points: Vec<Option<TriangulatedPoint>>,
    positive_depth_count: usize,
}

impl RelativePoseEstimate {
    pub(crate) fn new(pose: RelativePose, points: Vec<Option<TriangulatedPoint>>) -> Self {
        let positive_depth_count =
            points.iter().flatten().filter(|point| point.has_positive_depth()).count();
        Self { pose, points, positive_depth_count }
    }

    /// Returns the selected source-to-target pose.
    pub const fn pose(&self) -> RelativePose {
        self.pose
    }

    /// Returns one optional triangulation per input correspondence.
    pub fn points(&self) -> &[Option<TriangulatedPoint>] {
        &self.points
    }

    /// Returns the number of points in front of both cameras.
    pub const fn positive_depth_count(&self) -> usize {
        self.positive_depth_count
    }
}

fn determinant(matrix: Mat3<f64>) -> f64 {
    matrix.m[0][0] * (matrix.m[1][1] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][1])
        - matrix.m[0][1] * (matrix.m[1][0] * matrix.m[2][2] - matrix.m[1][2] * matrix.m[2][0])
        + matrix.m[0][2] * (matrix.m[1][0] * matrix.m[2][1] - matrix.m[1][1] * matrix.m[2][0])
}

impl<Model> GeometricEstimate<Model> {
    /// Creates a result whose inlier and residual arrays match the input count.
    pub fn try_new(
        model: Model,
        correspondence_count: usize,
        inliers: Vec<bool>,
        residuals: Vec<f64>,
    ) -> VisionResult<Self> {
        if inliers.len() != correspondence_count || residuals.len() != correspondence_count {
            return Err(VisionError::GeometryResultLayout {
                correspondences: correspondence_count,
                inliers: inliers.len(),
                residuals: residuals.len(),
            });
        }
        if residuals.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(VisionError::InvalidParameter(
                "geometry residuals must be finite and non-negative".into(),
            ));
        }
        Ok(Self { model, inliers, residuals })
    }

    /// Returns the estimated model.
    pub const fn model(&self) -> &Model {
        &self.model
    }

    /// Returns one inlier decision per input correspondence.
    pub fn inliers(&self) -> &[bool] {
        &self.inliers
    }

    /// Returns one non-negative geometric residual per input correspondence.
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Returns the number of accepted correspondences.
    pub fn inlier_count(&self) -> usize {
        self.inliers.iter().filter(|&&value| value).count()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CameraMatrix3, GeometricEstimate, Homography3, PointCorrespondence2,
        RobustEstimationOptions,
    };
    use crate::VisionError;
    use spatialrust_camera::CameraIntrinsics;
    use spatialrust_math::{Mat3, Vec2};

    #[test]
    fn correspondence_and_projective_matrix_validation_is_strict() {
        assert!(PointCorrespondence2::try_new(Vec2 { x: 1.0, y: 2.0 }, Vec2 { x: 3.0, y: 4.0 },)
            .is_ok());
        assert!(PointCorrespondence2::try_new(
            Vec2 { x: f64::NAN, y: 2.0 },
            Vec2 { x: 3.0, y: 4.0 },
        )
        .is_err());
        assert!(Homography3::try_new(Mat3::from_rows(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ))
        .is_err());
    }

    #[test]
    fn camera_matrix_normalizes_pixels_analytically() {
        let intrinsics = CameraIntrinsics::try_new(500.0, 400.0, 320.0, 240.0, 640, 480).unwrap();
        let camera = CameraMatrix3::from_intrinsics(intrinsics);
        let normalized = camera.normalize_pixel(Vec2 { x: 420.0, y: 160.0 });
        assert!((normalized.x - 0.2).abs() < 1e-12);
        assert!((normalized.y + 0.2).abs() < 1e-12);
        assert_eq!(normalized.z, 1.0);
        assert_eq!(camera.matrix().mul_mat3(camera.inverse()), Mat3::<f64>::identity());
    }

    #[test]
    fn estimate_layout_and_robust_options_are_checked() {
        let model = Homography3::try_new(Mat3::<f64>::identity()).unwrap();
        let estimate =
            GeometricEstimate::try_new(model, 2, vec![true, false], vec![0.1, 2.0]).unwrap();
        assert_eq!(estimate.inlier_count(), 1);
        assert!(matches!(
            GeometricEstimate::try_new(model, 2, vec![true], vec![0.1, 2.0]),
            Err(VisionError::GeometryResultLayout { .. })
        ));
        assert!(RobustEstimationOptions { confidence: 1.0, ..Default::default() }
            .validate()
            .is_err());
    }
}
