use core::f32::consts::PI;

use spatialrust_math::Vec3;

use crate::{VizError, VizResult};

/// Projection used to render a visual scene.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Projection {
    /// Perspective projection with a vertical field of view in radians.
    Perspective {
        /// Vertical field of view in radians.
        vertical_fov_radians: f32,
        /// Positive near clipping distance.
        near: f32,
        /// Far clipping distance greater than `near`.
        far: f32,
    },
    /// Orthographic projection with a positive vertical span.
    Orthographic {
        /// Visible vertical extent in world units.
        vertical_span: f32,
        /// Near clipping distance.
        near: f32,
        /// Far clipping distance greater than `near`.
        far: f32,
    },
}

impl Projection {
    /// Validates projection ranges.
    pub fn validate(self) -> VizResult<()> {
        match self {
            Self::Perspective { vertical_fov_radians, near, far } => {
                if !vertical_fov_radians.is_finite()
                    || vertical_fov_radians <= 0.0
                    || vertical_fov_radians >= PI
                    || !valid_clip_range(near, far)
                {
                    return Err(VizError::InvalidCamera(
                        "perspective FOV must be in (0, pi) and 0 < near < far".into(),
                    ));
                }
            }
            Self::Orthographic { vertical_span, near, far } => {
                if !vertical_span.is_finite()
                    || vertical_span <= 0.0
                    || !valid_clip_range(near, far)
                {
                    return Err(VizError::InvalidCamera(
                        "orthographic span must be positive and 0 < near < far".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn valid_clip_range(near: f32, far: f32) -> bool {
    near.is_finite() && far.is_finite() && near > 0.0 && far > near
}

/// Validated look-at camera.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Camera {
    /// Camera position in world coordinates.
    pub eye: Vec3<f32>,
    /// Look-at target in world coordinates.
    pub target: Vec3<f32>,
    /// Approximate world-space up direction.
    pub up: Vec3<f32>,
    /// Camera projection.
    pub projection: Projection,
}

impl Camera {
    /// Creates a validated look-at camera.
    pub fn try_new(
        eye: Vec3<f32>,
        target: Vec3<f32>,
        up: Vec3<f32>,
        projection: Projection,
    ) -> VizResult<Self> {
        projection.validate()?;
        if !finite_vec(eye) || !finite_vec(target) || !finite_vec(up) {
            return Err(VizError::InvalidCamera("view vectors must be finite".into()));
        }
        let forward = target - eye;
        if forward.length() <= f32::EPSILON {
            return Err(VizError::InvalidCamera("eye and target must differ".into()));
        }
        if up.length() <= f32::EPSILON || forward.cross(up).length() <= f32::EPSILON {
            return Err(VizError::InvalidCamera(
                "up must be non-zero and not parallel to the view direction".into(),
            ));
        }
        Ok(Self { eye, target, up: up.normalize(), projection })
    }
}

fn finite_vec(value: Vec3<f32>) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_PI_3;

    use spatialrust_math::Vec3;

    use super::{Camera, Projection};

    fn projection() -> Projection {
        Projection::Perspective { vertical_fov_radians: FRAC_PI_3, near: 0.1, far: 1_000.0 }
    }

    #[test]
    fn validates_look_at_basis() {
        let camera = Camera::try_new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            projection(),
        )
        .unwrap();
        assert_eq!(camera.up, Vec3::new(0.0, 1.0, 0.0));

        assert!(Camera::try_new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            projection(),
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_projection() {
        assert!(Projection::Perspective { vertical_fov_radians: 0.0, near: 0.1, far: 10.0 }
            .validate()
            .is_err());
        assert!(Projection::Orthographic { vertical_span: 0.0, near: 0.1, far: 10.0 }
            .validate()
            .is_err());
        assert!(Projection::Perspective { vertical_fov_radians: FRAC_PI_3, near: 10.0, far: 1.0 }
            .validate()
            .is_err());
    }
}
