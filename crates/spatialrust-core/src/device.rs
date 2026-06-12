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
