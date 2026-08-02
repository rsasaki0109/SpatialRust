use crate::{VizError, VizResult};

/// Stable identity of an explicit rendering or compute device.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceIdentity {
    /// Backend name such as `wgpu` or `cuda`.
    pub backend: String,
    /// Adapter or device identifier supplied by the backend.
    pub device: String,
}

impl DeviceIdentity {
    /// Creates a non-empty backend/device identity.
    pub fn try_new(backend: impl Into<String>, device: impl Into<String>) -> VizResult<Self> {
        let backend = backend.into();
        let device = device.into();
        if backend.trim().is_empty() || device.trim().is_empty() {
            return Err(VizError::InvalidTransfer(
                "device backend and identifier must not be empty".into(),
            ));
        }
        Ok(Self { backend, device })
    }
}

/// Residency of visual data before or after an explicit transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VisualResidency {
    /// Caller-owned host memory.
    Host,
    /// Memory owned by a named device.
    Device(DeviceIdentity),
}

/// Direction of an explicit host/device transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TransferDirection {
    /// Host to device.
    Upload,
    /// Device to host.
    Readback,
    /// One explicitly named device to another.
    DeviceToDevice,
}

/// One named explicit data transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransferEvent {
    /// Caller-visible stage name.
    pub stage: String,
    /// Transfer direction.
    pub direction: TransferDirection,
    /// Source residency.
    pub source: VisualResidency,
    /// Destination residency.
    pub destination: VisualResidency,
    /// Number of transferred bytes.
    pub bytes: u64,
}

impl TransferEvent {
    /// Creates and validates an explicit transfer event.
    pub fn try_new(
        stage: impl Into<String>,
        direction: TransferDirection,
        source: VisualResidency,
        destination: VisualResidency,
        bytes: u64,
    ) -> VizResult<Self> {
        let stage = stage.into();
        if stage.trim().is_empty() {
            return Err(VizError::InvalidTransfer("transfer stage must not be empty".into()));
        }
        let residency_matches = match direction {
            TransferDirection::Upload => {
                matches!(&source, VisualResidency::Host)
                    && matches!(&destination, VisualResidency::Device(_))
            }
            TransferDirection::Readback => {
                matches!(&source, VisualResidency::Device(_))
                    && matches!(&destination, VisualResidency::Host)
            }
            TransferDirection::DeviceToDevice => {
                matches!(&source, VisualResidency::Device(_))
                    && matches!(&destination, VisualResidency::Device(_))
            }
        };
        if !residency_matches {
            return Err(VizError::InvalidTransfer(
                "transfer direction does not match source and destination residency".into(),
            ));
        }
        Ok(Self { stage, direction, source, destination, bytes })
    }
}

/// Ordered ledger of explicit visual-data transfers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransferReceipt {
    events: Vec<TransferEvent>,
}

impl TransferReceipt {
    /// Creates an empty receipt.
    #[must_use]
    pub const fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Appends an already validated event.
    pub fn push(&mut self, event: TransferEvent) {
        self.events.push(event);
    }

    /// Returns events in execution order.
    #[must_use]
    pub fn events(&self) -> &[TransferEvent] {
        &self.events
    }

    /// Returns the checked total number of transferred bytes.
    pub fn total_bytes(&self) -> VizResult<u64> {
        self.events.iter().try_fold(0_u64, |total, event| {
            total.checked_add(event.bytes).ok_or_else(|| {
                VizError::InvalidTransfer("transfer byte total overflowed u64".into())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceIdentity, TransferDirection, TransferEvent, TransferReceipt, VisualResidency,
    };

    #[test]
    fn receipt_requires_direction_to_match_residency() {
        let device = VisualResidency::Device(DeviceIdentity::try_new("wgpu", "adapter-0").unwrap());
        let upload = TransferEvent::try_new(
            "point-upload",
            TransferDirection::Upload,
            VisualResidency::Host,
            device.clone(),
            96,
        )
        .unwrap();
        let mut receipt = TransferReceipt::new();
        receipt.push(upload);
        assert_eq!(receipt.total_bytes().unwrap(), 96);

        assert!(TransferEvent::try_new(
            "wrong",
            TransferDirection::Readback,
            VisualResidency::Host,
            device,
            1,
        )
        .is_err());
    }

    #[test]
    fn total_bytes_fails_closed_on_overflow() {
        let device = VisualResidency::Device(DeviceIdentity::try_new("wgpu", "adapter-0").unwrap());
        let mut receipt = TransferReceipt::new();
        receipt.push(
            TransferEvent::try_new(
                "first",
                TransferDirection::Upload,
                VisualResidency::Host,
                device.clone(),
                u64::MAX,
            )
            .unwrap(),
        );
        receipt.push(
            TransferEvent::try_new(
                "second",
                TransferDirection::Upload,
                VisualResidency::Host,
                device,
                1,
            )
            .unwrap(),
        );
        assert!(receipt.total_bytes().is_err());
    }
}
