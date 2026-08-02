use crate::{DeviceKind, ExecutionPolicy};

/// Runtime boundary for executing spatial algorithms on one backend.
///
/// The trait intentionally exposes only backend identity and policy
/// compatibility. Backend-specific queues, buffers, and transfer operations
/// belong in the crate that implements the runtime.
pub trait SpatialRuntime {
    /// Returns the device kind owned by this runtime.
    fn device_kind(&self) -> DeviceKind;

    /// Returns whether this runtime can satisfy the requested policy.
    #[must_use]
    fn supports_policy(&self, policy: ExecutionPolicy) -> bool {
        match policy.device_kind() {
            Some(kind) => kind == self.device_kind(),
            None => true,
        }
    }
}

/// Host runtime for single-threaded and parallel CPU algorithms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuRuntime;

impl SpatialRuntime for CpuRuntime {
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuRuntime, SpatialRuntime};
    use crate::{DeviceKind, ExecutionPolicy};

    #[test]
    fn cpu_runtime_accepts_cpu_and_auto_policies_only() {
        let runtime = CpuRuntime;
        assert_eq!(runtime.device_kind(), DeviceKind::Cpu);
        assert!(runtime.supports_policy(ExecutionPolicy::CpuSingle));
        assert!(runtime.supports_policy(ExecutionPolicy::CpuParallel));
        assert!(runtime.supports_policy(ExecutionPolicy::Auto));
        assert!(!runtime.supports_policy(ExecutionPolicy::Gpu(DeviceKind::Wgpu)));
    }
}
