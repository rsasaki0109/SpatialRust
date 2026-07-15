//! Named host/device/network transfers and measurable plans.

use crate::{DistributeError, DistributeResult, PartitionGraph};

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

impl NamedTransfer {
    /// Creates a validated named transfer.
    pub fn try_new(
        name: impl Into<String>,
        direction: TransferDirection,
        kind: TransferKind,
        from: impl Into<String>,
        to: impl Into<String>,
        bytes: u64,
    ) -> DistributeResult<Self> {
        let name = name.into();
        let from = from.into();
        let to = to.into();
        if name.is_empty() || from.is_empty() || to.is_empty() {
            return Err(DistributeError::InvalidConfiguration(
                "transfer name/from/to must be non-empty".into(),
            ));
        }
        if from == to {
            return Err(DistributeError::InvalidConfiguration(
                "transfer endpoints must differ".into(),
            ));
        }
        Ok(Self {
            name,
            direction,
            kind,
            from,
            to,
            bytes,
        })
    }

    /// Bytes counted as measurable copies (zero-copy handoffs are 0).
    #[must_use]
    pub fn counted_copy_bytes(&self) -> u64 {
        match self.kind {
            TransferKind::ExplicitCopy => self.bytes,
            TransferKind::ZeroCopyHandoff => 0,
        }
    }
}

/// Ordered plan of named transfers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransferPlan {
    transfers: Vec<NamedTransfer>,
}

impl TransferPlan {
    /// Creates an empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a transfer.
    pub fn push(&mut self, transfer: NamedTransfer) {
        self.transfers.push(transfer);
    }

    /// Returns transfers.
    #[must_use]
    pub fn transfers(&self) -> &[NamedTransfer] {
        &self.transfers
    }

    /// Sum of payload bytes (including zero-copy declared sizes).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.transfers.iter().map(|t| t.bytes).sum()
    }

    /// Sum of measurable explicit-copy bytes.
    #[must_use]
    pub fn counted_copy_bytes(&self) -> u64 {
        self.transfers.iter().map(NamedTransfer::counted_copy_bytes).sum()
    }

    /// Ensures every endpoint exists in the partition graph.
    pub fn validate_against(&self, graph: &PartitionGraph) -> DistributeResult<()> {
        for transfer in &self.transfers {
            if graph.partition_of_node(&transfer.from).is_none() {
                return Err(DistributeError::Missing(format!(
                    "transfer source node `{}`",
                    transfer.from
                )));
            }
            if graph.partition_of_node(&transfer.to).is_none() {
                return Err(DistributeError::Missing(format!(
                    "transfer destination node `{}`",
                    transfer.to
                )));
            }
        }
        Ok(())
    }
}

/// Append-only ledger of completed named transfers for telemetry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransferLedger {
    completed: Vec<NamedTransfer>,
}

impl TransferLedger {
    /// Creates an empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a completed transfer.
    pub fn record(&mut self, transfer: NamedTransfer) {
        self.completed.push(transfer);
    }

    /// Returns completed transfers.
    #[must_use]
    pub fn completed(&self) -> &[NamedTransfer] {
        &self.completed
    }

    /// Total measurable copy bytes recorded.
    #[must_use]
    pub fn counted_copy_bytes(&self) -> u64 {
        self.completed.iter().map(NamedTransfer::counted_copy_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NamedTransfer, TransferDirection, TransferKind, TransferLedger, TransferPlan,
    };
    use crate::{ExecutionPartition, PartitionGraph};

    #[test]
    fn plan_validates_and_counts_copies() {
        let mut graph = PartitionGraph::new();
        graph
            .insert_partition(ExecutionPartition::try_new("edge", vec!["cam".into()]).unwrap())
            .unwrap();
        graph
            .insert_partition(ExecutionPartition::try_new("host", vec!["scene".into()]).unwrap())
            .unwrap();
        let mut plan = TransferPlan::new();
        plan.push(
            NamedTransfer::try_new(
                "cam-to-scene",
                TransferDirection::HostToNetwork,
                TransferKind::ExplicitCopy,
                "cam",
                "scene",
                1024,
            )
            .unwrap(),
        );
        plan.push(
            NamedTransfer::try_new(
                "handoff",
                TransferDirection::HostToDevice,
                TransferKind::ZeroCopyHandoff,
                "scene",
                "cam",
                4096,
            )
            .unwrap(),
        );
        plan.validate_against(&graph).unwrap();
        assert_eq!(plan.total_bytes(), 5120);
        assert_eq!(plan.counted_copy_bytes(), 1024);

        let mut ledger = TransferLedger::new();
        ledger.record(plan.transfers()[0].clone());
        assert_eq!(ledger.counted_copy_bytes(), 1024);
    }
}
