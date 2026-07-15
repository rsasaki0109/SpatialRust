//! Named host/device/network transfers.

/// Transfer direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Host to remote host.
    HostToNetwork,
    /// Remote host to host.
    NetworkToHost,
}

/// Kind of payload transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferKind {
    /// Explicit named copy (never implied).
    ExplicitCopy,
    /// Zero-copy handoff when both ends agree.
    ZeroCopyHandoff,
}

/// One named transfer in a distributed plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedTransfer {
    /// Transfer name used in telemetry.
    pub name: String,
    /// Direction.
    pub direction: TransferDirection,
    /// Kind.
    pub kind: TransferKind,
    /// Source node id.
    pub from: String,
    /// Destination node id.
    pub to: String,
    /// Approximate payload bytes.
    pub bytes: u64,
}
