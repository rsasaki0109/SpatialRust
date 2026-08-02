//! Oriented surfel clouds.

use spatialrust_math::Vec3;

use crate::{SceneError, SceneResult};

/// One oriented disc surfel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surfel {
    /// Center.
    pub position: Vec3<f32>,
    /// Unit normal.
    pub normal: Vec3<f32>,
    /// Disc radius.
    pub radius: f32,
}

/// Collection of surfels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfelCloud {
    surfels: Vec<Surfel>,
}

impl SurfelCloud {
    /// Creates an empty cloud.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes a validated surfel.
    pub fn push(&mut self, surfel: Surfel) -> SceneResult<()> {
        if !(surfel.radius.is_finite() && surfel.radius > 0.0) {
            return Err(SceneError::InvalidConfiguration("surfel radius must be > 0".into()));
        }
        if !(surfel.normal.length().is_finite()) || surfel.normal.length() < 1e-6 {
            return Err(SceneError::InvalidConfiguration("surfel normal must be non-zero".into()));
        }
        self.surfels.push(surfel);
        Ok(())
    }

    /// Returns surfels.
    #[must_use]
    pub fn as_slice(&self) -> &[Surfel] {
        &self.surfels
    }
}
