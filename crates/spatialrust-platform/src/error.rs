//! Platform crate errors.

/// Result type for platform operations.
pub type PlatformResult<T> = Result<T, PlatformError>;

/// Failures in stability/conformance bookkeeping.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    /// Invalid configuration.
    #[error("invalid platform configuration: {0}")]
    InvalidConfiguration(String),
    /// A performance sample exceeded its budget ceiling.
    #[error("performance budget `{budget_id}` exceeded: observed {observed} > ceiling {ceiling}")]
    BudgetExceeded {
        /// Budget identifier.
        budget_id: String,
        /// Observed measurement.
        observed: u64,
        /// Declared ceiling.
        ceiling: u64,
    },
    /// Aggregated release gate denial.
    #[error("release gate denied: {reasons:?}")]
    ReleaseGateDenied {
        /// Denial reasons.
        reasons: Vec<String>,
    },
}
