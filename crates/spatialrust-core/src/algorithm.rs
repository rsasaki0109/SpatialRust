use crate::{ExecutionPolicy, SpatialResult};

/// Common trait implemented by spatial algorithms.
pub trait SpatialAlgorithm {
    /// Human-readable algorithm name.
    fn name(&self) -> &'static str;

    /// Executes the algorithm using the selected execution policy.
    fn execute(&self, policy: ExecutionPolicy) -> SpatialResult<()>;
}
