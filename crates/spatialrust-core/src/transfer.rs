/// Direction of an explicit data transfer between host and device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Copy from host memory into device memory.
    HostToDevice,
    /// Copy between buffers on the same device.
    DeviceToDevice,
    /// Copy from device memory back into host memory.
    DeviceToHost,
}

/// Byte accounting for explicit execution transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransferStats {
    host_to_device_bytes: u64,
    device_to_device_bytes: u64,
    device_to_host_bytes: u64,
}

impl TransferStats {
    /// Records bytes transferred in the given direction.
    pub fn record(&mut self, direction: TransferDirection, bytes: u64) {
        let counter = match direction {
            TransferDirection::HostToDevice => &mut self.host_to_device_bytes,
            TransferDirection::DeviceToDevice => &mut self.device_to_device_bytes,
            TransferDirection::DeviceToHost => &mut self.device_to_host_bytes,
        };
        *counter = counter.saturating_add(bytes);
    }

    /// Returns bytes copied from host memory into device memory.
    #[must_use]
    pub const fn host_to_device_bytes(self) -> u64 {
        self.host_to_device_bytes
    }

    /// Returns bytes copied between buffers on a device.
    #[must_use]
    pub const fn device_to_device_bytes(self) -> u64 {
        self.device_to_device_bytes
    }

    /// Returns bytes copied from device memory into host memory.
    #[must_use]
    pub const fn device_to_host_bytes(self) -> u64 {
        self.device_to_host_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{TransferDirection, TransferStats};

    #[test]
    fn records_transfer_bytes_by_direction() {
        let mut stats = TransferStats::default();
        stats.record(TransferDirection::HostToDevice, 12);
        stats.record(TransferDirection::HostToDevice, 8);
        stats.record(TransferDirection::DeviceToDevice, 4);
        stats.record(TransferDirection::DeviceToHost, 16);

        assert_eq!(stats.host_to_device_bytes(), 20);
        assert_eq!(stats.device_to_device_bytes(), 4);
        assert_eq!(stats.device_to_host_bytes(), 16);
    }
}
