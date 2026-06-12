use crate::DeviceKind;

/// Execution policy for spatial algorithms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExecutionPolicy {
    /// Single-threaded CPU execution.
    #[default]
    CpuSingle,
    /// Parallel CPU execution.
    CpuParallel,
    /// GPU execution on a device of the given kind.
    Gpu(DeviceKind),
    /// Automatic selection based on runtime heuristics.
    Auto,
}

impl ExecutionPolicy {
    /// Returns the device kind targeted by this policy when known.
    #[must_use]
    pub fn device_kind(&self) -> Option<DeviceKind> {
        match self {
            Self::CpuSingle | Self::CpuParallel => Some(DeviceKind::Cpu),
            Self::Gpu(kind) => Some(*kind),
            Self::Auto => None,
        }
    }
}
