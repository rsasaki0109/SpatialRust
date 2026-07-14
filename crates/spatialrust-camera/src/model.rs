use crate::BrownConrady;
use spatialrust_math::{Vec2, Vec3};

/// Camera model errors.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum CameraError {
    /// A focal length was zero, negative, or non-finite.
    #[error("camera focal lengths must be finite and positive")]
    InvalidFocalLength,
    /// The principal point was non-finite.
    #[error("camera principal point must be finite")]
    InvalidPrincipalPoint,
    /// Projection was requested for a point outside the positive camera half-space.
    #[error("point depth must be finite and positive, found {0}")]
    InvalidDepth(f64),
    /// Pixel coordinates were non-finite.
    #[error("pixel coordinates must be finite")]
    InvalidPixel,
}

/// Pinhole camera intrinsic parameters and image dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntrinsics {
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Principal point x coordinate in pixels.
    pub cx: f64,
    /// Principal point y coordinate in pixels.
    pub cy: f64,
    /// Calibrated image width.
    pub width: usize,
    /// Calibrated image height.
    pub height: usize,
}

impl CameraIntrinsics {
    /// Creates and validates camera intrinsics.
    pub fn try_new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: usize,
        height: usize,
    ) -> Result<Self, CameraError> {
        if !fx.is_finite() || !fy.is_finite() || fx <= 0.0 || fy <= 0.0 {
            return Err(CameraError::InvalidFocalLength);
        }
        if !cx.is_finite() || !cy.is_finite() {
            return Err(CameraError::InvalidPrincipalPoint);
        }
        Ok(Self { fx, fy, cx, cy, width, height })
    }
}

/// A pinhole camera with optional Brown–Conrady lens distortion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeCamera {
    /// Intrinsic calibration.
    pub intrinsics: CameraIntrinsics,
    /// Lens distortion coefficients.
    pub distortion: BrownConrady,
}

impl PinholeCamera {
    /// Creates a camera with no lens distortion.
    #[must_use]
    pub const fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            distortion: BrownConrady { k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0, k3: 0.0 },
        }
    }

    /// Attaches a Brown–Conrady distortion model.
    #[must_use]
    pub const fn with_distortion(mut self, distortion: BrownConrady) -> Self {
        self.distortion = distortion;
        self
    }

    /// Projects a camera-space point into distorted pixel coordinates.
    pub fn project(&self, point: Vec3<f64>) -> Result<Vec2<f64>, CameraError> {
        if !point.z.is_finite() || point.z <= 0.0 {
            return Err(CameraError::InvalidDepth(point.z));
        }
        let normalized = Vec2 { x: point.x / point.z, y: point.y / point.z };
        let distorted = self.distortion.distort(normalized);
        Ok(Vec2 {
            x: self.intrinsics.fx.mul_add(distorted.x, self.intrinsics.cx),
            y: self.intrinsics.fy.mul_add(distorted.y, self.intrinsics.cy),
        })
    }

    /// Unprojects a distorted pixel and metric depth into camera space.
    pub fn unproject(&self, pixel: Vec2<f64>, depth: f64) -> Result<Vec3<f64>, CameraError> {
        if !depth.is_finite() || depth <= 0.0 {
            return Err(CameraError::InvalidDepth(depth));
        }
        if !pixel.x.is_finite() || !pixel.y.is_finite() {
            return Err(CameraError::InvalidPixel);
        }
        let distorted = Vec2 {
            x: (pixel.x - self.intrinsics.cx) / self.intrinsics.fx,
            y: (pixel.y - self.intrinsics.cy) / self.intrinsics.fy,
        };
        let normalized = self.distortion.undistort(distorted);
        Ok(Vec3::new(normalized.x * depth, normalized.y * depth, depth))
    }
}

#[cfg(test)]
mod tests {
    use super::{CameraIntrinsics, PinholeCamera};
    use crate::BrownConrady;
    use spatialrust_math::Vec3;

    #[test]
    fn project_unproject_roundtrip_with_distortion() {
        let intrinsics = CameraIntrinsics::try_new(525.0, 520.0, 319.5, 239.5, 640, 480).unwrap();
        let camera = PinholeCamera::new(intrinsics).with_distortion(BrownConrady {
            k1: -0.15,
            k2: 0.02,
            p1: 0.001,
            p2: -0.001,
            k3: 0.0,
        });
        let point = Vec3::new(0.4, -0.2, 2.5);
        let pixel = camera.project(point).unwrap();
        let recovered = camera.unproject(pixel, point.z).unwrap();
        assert!((recovered.x - point.x).abs() < 1e-8);
        assert!((recovered.y - point.y).abs() < 1e-8);
        assert_eq!(recovered.z, point.z);
    }
}
