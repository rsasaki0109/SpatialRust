/// Device kind supported by SpatialRust execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceKind {
    /// Host CPU execution.
    #[default]
    Cpu,
    /// Portable GPU execution via wgpu/WebGPU.
    Wgpu,
    /// NVIDIA CUDA execution.
    Cuda,
}

impl DeviceKind {
    /// Returns whether this device kind represents host CPU execution.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns whether this device kind represents an accelerator device.
    #[must_use]
    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::Wgpu | Self::Cuda)
    }
}

/// Minimal device abstraction defined in core and extended by `spatialrust-gpu`.
pub trait Device: core::fmt::Debug + Send + Sync + 'static {
    /// Returns the kind of this device.
    fn kind(&self) -> DeviceKind;
}

/// Default CPU device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}
