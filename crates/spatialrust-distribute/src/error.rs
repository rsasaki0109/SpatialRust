//! Distributed execution errors.

/// Result type for distribute operations.
pub type DistributeResult<T> = Result<T, DistributeError>;

/// Failures in partition graphs and transfers.
#[derive(Debug, thiserror::Error)]
pub enum DistributeError {
    /// Invalid configuration.
    #[error("invalid distribute configuration: {0}")]
    InvalidConfiguration(String),
    /// Missing node or transfer.
    #[error("missing `{0}`")]
    Missing(String),
    /// Partition graph contains a cycle.
    #[error("partition graph contains a cycle")]
    CycleDetected,
    /// Transfer queue reached its hard backpressure limit.
    #[error(
        "transfer queue `{queue}` at capacity: depth {depth} >= hard_limit {hard_limit}"
    )]
    CapacityExceeded {
        /// Queue / transfer name used for diagnostics.
        queue: String,
        /// Observed depth.
        depth: usize,
        /// Configured hard limit.
        hard_limit: usize,
    },
}
