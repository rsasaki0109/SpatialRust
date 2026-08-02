use crate::{DeviceKind, SpatialError, SpatialResult, TransferDirection, TransferStats};

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
    /// Validates that the policy names a meaningful execution request.
    pub fn validate(self) -> SpatialResult<()> {
        if matches!(self, Self::Gpu(DeviceKind::Cpu)) {
            return Err(SpatialError::InvalidArgument(
                "GPU execution policy cannot target the CPU device".to_owned(),
            ));
        }
        Ok(())
    }

    /// Returns whether the policy names a concrete execution backend.
    #[must_use]
    pub const fn is_explicit(self) -> bool {
        !matches!(self, Self::Auto)
    }

    /// Returns whether the policy requests an accelerator backend.
    #[must_use]
    pub const fn requests_gpu(self) -> bool {
        matches!(self, Self::Gpu(kind) if kind.is_gpu())
    }

    /// Returns whether the policy lets an algorithm choose a fallback backend.
    ///
    /// `Auto` is the only policy with fallback semantics. An explicit CPU or GPU
    /// policy is a request for that backend and must be reported as unsupported
    /// when the backend cannot satisfy it.
    #[must_use]
    pub const fn allows_fallback(self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Returns the device kind targeted by this policy when known.
    #[must_use]
    pub const fn device_kind(&self) -> Option<DeviceKind> {
        match self {
            Self::CpuSingle | Self::CpuParallel => Some(DeviceKind::Cpu),
            Self::Gpu(kind) => Some(*kind),
            Self::Auto => None,
        }
    }
}

/// Account of one algorithm execution.
///
/// The requested policy is kept separately from the resolved policy so callers
/// can distinguish an automatic CPU choice from an explicit CPU request. The
/// transfer counters describe the explicit host/device boundary copies made by
/// the operation; backend-specific internal work remains an implementation
/// detail of the executing crate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionReceipt {
    requested_policy: ExecutionPolicy,
    resolved_policy: ExecutionPolicy,
    transfers: TransferStats,
    stages: Vec<&'static str>,
}

impl ExecutionReceipt {
    /// Creates a receipt for a requested policy and the policy actually used.
    #[must_use]
    pub fn new(requested_policy: ExecutionPolicy, resolved_policy: ExecutionPolicy) -> Self {
        Self {
            requested_policy,
            resolved_policy,
            transfers: TransferStats::default(),
            stages: Vec::new(),
        }
    }

    /// Returns the policy supplied by the caller.
    #[must_use]
    pub const fn requested_policy(&self) -> ExecutionPolicy {
        self.requested_policy
    }

    /// Returns the backend policy selected for this execution.
    #[must_use]
    pub const fn resolved_policy(&self) -> ExecutionPolicy {
        self.resolved_policy
    }

    /// Returns transfer accounting for this execution.
    #[must_use]
    pub const fn transfer_stats(&self) -> TransferStats {
        self.transfers
    }

    /// Returns bytes copied from host memory to a device.
    #[must_use]
    pub const fn host_to_device_bytes(&self) -> u64 {
        self.transfers.host_to_device_bytes()
    }

    /// Returns bytes copied between buffers on a device.
    #[must_use]
    pub const fn device_to_device_bytes(&self) -> u64 {
        self.transfers.device_to_device_bytes()
    }

    /// Returns bytes copied from a device back to host memory.
    #[must_use]
    pub const fn device_to_host_bytes(&self) -> u64 {
        self.transfers.device_to_host_bytes()
    }

    /// Returns logical stages recorded by the algorithm or pipeline.
    #[must_use]
    pub fn stages(&self) -> &[&'static str] {
        &self.stages
    }

    /// Records an explicit transfer made by the operation.
    pub fn record_transfer(&mut self, direction: TransferDirection, bytes: u64) {
        self.transfers.record(direction, bytes);
    }

    /// Records a logical stage name in execution order.
    pub fn record_stage(&mut self, stage: &'static str) {
        self.stages.push(stage);
    }
}

/// Output value paired with the receipt for its execution.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionOutput<T> {
    output: T,
    receipt: ExecutionReceipt,
}

impl<T> ExecutionOutput<T> {
    /// Creates an output/receipt pair.
    #[must_use]
    pub const fn new(output: T, receipt: ExecutionReceipt) -> Self {
        Self { output, receipt }
    }

    /// Borrows the algorithm output.
    #[must_use]
    pub const fn output(&self) -> &T {
        &self.output
    }

    /// Borrows the execution receipt.
    #[must_use]
    pub const fn receipt(&self) -> &ExecutionReceipt {
        &self.receipt
    }

    /// Splits the output and receipt without cloning either value.
    #[must_use]
    pub fn into_parts(self) -> (T, ExecutionReceipt) {
        (self.output, self.receipt)
    }

    /// Returns only the algorithm output.
    #[must_use]
    pub fn into_output(self) -> T {
        self.output
    }

    /// Maps the output while preserving its execution receipt.
    #[must_use]
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> ExecutionOutput<U> {
        ExecutionOutput::new(map(self.output), self.receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionOutput, ExecutionPolicy, ExecutionReceipt};
    use crate::{DeviceKind, TransferDirection};

    #[test]
    fn policy_contract_distinguishes_auto_from_explicit_backend() {
        assert!(ExecutionPolicy::Auto.validate().is_ok());
        assert!(!ExecutionPolicy::Auto.is_explicit());
        assert!(ExecutionPolicy::Auto.allows_fallback());
        assert!(ExecutionPolicy::Gpu(DeviceKind::Wgpu).is_explicit());
        assert!(ExecutionPolicy::Gpu(DeviceKind::Wgpu).requests_gpu());
        assert!(!ExecutionPolicy::Gpu(DeviceKind::Wgpu).allows_fallback());
        assert!(ExecutionPolicy::CpuSingle.device_kind().unwrap().is_cpu());
        assert!(ExecutionPolicy::Gpu(DeviceKind::Cpu).validate().is_err());
    }

    #[test]
    fn receipt_keeps_requested_and_resolved_policies() {
        let mut receipt = ExecutionReceipt::new(ExecutionPolicy::Auto, ExecutionPolicy::CpuSingle);
        receipt.record_transfer(TransferDirection::HostToDevice, 24);
        receipt.record_stage("voxel");

        assert_eq!(receipt.requested_policy(), ExecutionPolicy::Auto);
        assert_eq!(receipt.resolved_policy(), ExecutionPolicy::CpuSingle);
        assert_eq!(receipt.host_to_device_bytes(), 24);
        assert_eq!(receipt.stages(), &["voxel"]);
    }

    #[test]
    fn execution_output_can_split_or_map() {
        let receipt = ExecutionReceipt::new(ExecutionPolicy::CpuSingle, ExecutionPolicy::CpuSingle);
        let output = ExecutionOutput::new(3_u32, receipt).map(|value| value * 2);
        let (value, receipt) = output.into_parts();
        assert_eq!(value, 6);
        assert_eq!(receipt.resolved_policy(), ExecutionPolicy::CpuSingle);
    }
}
