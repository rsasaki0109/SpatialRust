use spatialrust_core::{PointCloud, SpatialResult};
use spatialrust_math::Isometry3;

/// Common trait for point cloud registration algorithms.
pub trait PointCloudRegistration {
    /// Human-readable registration algorithm name.
    fn name(&self) -> &'static str;

    /// Aligns `source` to `target` and returns the estimated transform.
    fn align(&self, source: &PointCloud, target: &PointCloud) -> SpatialResult<RegistrationResult>;
}

/// Result of a registration run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegistrationResult {
    /// Estimated transform mapping source into target frame.
    pub transform: Isometry3<f32>,
    /// Mean squared correspondence error at convergence.
    pub fitness: f64,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Whether a convergence criterion was met.
    pub converged: bool,
}
