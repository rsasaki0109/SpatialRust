use spatialrust_core::{Device, DeviceKind};

/// Marker trait for GPU-capable devices.
pub trait GpuDevice: Device {}

/// Portable GPU device backed by wgpu.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WgpuDevice {
    label: String,
}

impl WgpuDevice {
    /// Creates a labeled wgpu device handle.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self { label: label.into() }
    }

    /// Returns the device label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl Device for WgpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Wgpu
    }
}

impl GpuDevice for WgpuDevice {}

#[cfg(feature = "gpu-wgpu")]
mod wgpu_backend {
    use super::WgpuDevice;

    impl WgpuDevice {
        /// Creates the default wgpu-backed device placeholder.
        ///
        /// Full adapter selection is implemented in later GPU milestones.
        #[must_use]
        pub fn default_adapter() -> Self {
            Self::new("wgpu-default")
        }
    }
}
