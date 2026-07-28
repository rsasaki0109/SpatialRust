/// LOD operation result.
pub type LodResult<T> = Result<T, LodError>;

/// LOD validation, admission, or adapter error.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LodError {
    /// Index metadata is malformed.
    #[error("invalid LOD index: {0}")]
    InvalidIndex(String),
    /// Planner configuration or camera is invalid.
    #[error("invalid LOD planner state: {0}")]
    InvalidPlanner(String),
    /// A hard resource budget rejected work before allocation/upload.
    #[error("LOD budget exceeded: {0}")]
    BudgetExceeded(String),
    /// A node was missing.
    #[error("unknown LOD node {0}")]
    UnknownNode(u64),
    /// Record-memory lease admission failed.
    #[cfg(feature = "records")]
    #[error("record lease failed: {0}")]
    Records(String),
    /// COPC query adaptation failed.
    #[cfg(feature = "copc")]
    #[error("COPC LOD adapter failed: {0}")]
    Copc(String),
}
