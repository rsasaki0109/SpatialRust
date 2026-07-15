//! Feature-gated Gaussian scene primitives (no renderer runtime).

use spatialrust_math::{Quat, Vec3};

use crate::{SceneError, SceneResult};

/// One anisotropic Gaussian primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPrimitive {
    /// Mean position.
    pub mean: Vec3<f32>,
    /// Per-axis scale.
    pub scale: Vec3<f32>,
    /// Orientation quaternion.
    pub rotation: Quat<f32>,
    /// Opacity in `[0, 1]`.
    pub opacity: f32,
    /// RGB color in `[0, 1]`.
    pub color: [f32; 3],
}

/// Host-side Gaussian scene container (renderer deferred).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GaussianScene {
    primitives: Vec<GaussianPrimitive>,
}

impl GaussianScene {
    /// Creates an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a validated Gaussian.
    pub fn push(&mut self, primitive: GaussianPrimitive) -> SceneResult<()> {
        if !(0.0..=1.0).contains(&primitive.opacity) {
            return Err(SceneError::InvalidConfiguration("opacity must be in [0, 1]".into()));
        }
        if primitive.color.iter().any(|c| !(0.0..=1.0).contains(c)) {
            return Err(SceneError::InvalidConfiguration("color channels must be in [0, 1]".into()));
        }
        self.primitives.push(primitive);
        Ok(())
    }

    /// Returns primitives.
    #[must_use]
    pub fn primitives(&self) -> &[GaussianPrimitive] {
        &self.primitives
    }
}
