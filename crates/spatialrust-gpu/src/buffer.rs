use spatialrust_core::{Device, DeviceKind, SpatialError, SpatialResult};

/// Typed buffer allocated on a specific device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceBuffer<T> {
    len: usize,
    device_kind: DeviceKind,
    _marker: core::marker::PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    /// Creates a new device buffer placeholder.
    ///
    /// Actual allocation is implemented in later GPU milestones.
    pub fn new(len: usize, device: &dyn Device) -> SpatialResult<Self> {
        if len == 0 {
            return Err(SpatialError::InvalidArgument(
                "device buffer length must be greater than zero".to_owned(),
            ));
        }

        Ok(Self { len, device_kind: device.kind(), _marker: core::marker::PhantomData })
    }

    /// Returns the number of elements in the buffer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the device kind backing this buffer.
    #[must_use]
    pub const fn device_kind(&self) -> DeviceKind {
        self.device_kind
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceBuffer;
    use spatialrust_core::CpuDevice;

    #[test]
    fn rejects_zero_length_buffer() {
        let device = CpuDevice;
        let err = DeviceBuffer::<f32>::new(0, &device).unwrap_err();
        assert!(matches!(err, spatialrust_core::SpatialError::InvalidArgument(_)));
    }
}
