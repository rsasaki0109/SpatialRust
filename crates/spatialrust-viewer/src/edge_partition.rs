//! Portable state for explicit edge-to-host partition execution.
//!
//! The state is intentionally independent of ROS, SQLite, and a transport
//! implementation. An adapter records the graph/queue decisions here, while
//! the viewer can render the same receipt and keep mapping admission separate
//! from successful packet transfer.

use std::collections::BTreeSet;

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized edge-partition state schema version.
pub const EDGE_PARTITION_STATE_VERSION: u32 = 1;

/// One named execution partition shown in the receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct EdgePartition {
    /// Stable partition identifier.
    pub id: String,
    /// Placement label, such as edge or host.
    pub placement: String,
    /// Node identifiers owned by the partition.
    pub node_ids: Vec<String>,
}

impl EdgePartition {
    /// Creates and validates a named execution partition.
    pub fn try_new(
        id: impl Into<String>,
        placement: impl Into<String>,
        node_ids: Vec<String>,
    ) -> ViewerResult<Self> {
        let partition = Self { id: id.into(), placement: placement.into(), node_ids };
        partition.validate()?;
        Ok(partition)
    }

    /// Validates partition identity and node uniqueness.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty() || self.placement.trim().is_empty() || self.node_ids.is_empty()
        {
            return Err(ViewerError::InvalidState(
                "edge partitions require an id, placement, and at least one node".into(),
            ));
        }
        let mut nodes = BTreeSet::new();
        for node_id in &self.node_ids {
            if node_id.trim().is_empty() || !nodes.insert(node_id) {
                return Err(ViewerError::InvalidState(
                    "edge partition node IDs must be non-empty and unique".into(),
                ));
            }
        }
        Ok(())
    }
}

/// One bounded packet transfer between two named execution nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct EdgePartitionTransfer {
    /// Source packet sequence from the upstream live-publish receipt.
    pub sequence: u64,
    /// Source ROS topic carried by this transfer.
    pub source_topic: String,
    /// Source execution node.
    pub from_node: String,
    /// Destination execution node.
    pub to_node: String,
    /// Payload bytes declared by the upstream packet.
    pub payload_bytes: u64,
    /// Measurable explicit-copy bytes recorded by the transfer ledger.
    pub counted_copy_bytes: u64,
    /// Backpressure signal observed at admission.
    pub queue_signal: String,
    /// Whether the transfer reached the destination ledger.
    pub completed: bool,
}

impl EdgePartitionTransfer {
    /// Creates and validates one explicit packet transfer receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        sequence: u64,
        source_topic: impl Into<String>,
        from_node: impl Into<String>,
        to_node: impl Into<String>,
        payload_bytes: u64,
        counted_copy_bytes: u64,
        queue_signal: impl Into<String>,
        completed: bool,
    ) -> ViewerResult<Self> {
        let transfer = Self {
            sequence,
            source_topic: source_topic.into(),
            from_node: from_node.into(),
            to_node: to_node.into(),
            payload_bytes,
            counted_copy_bytes,
            queue_signal: queue_signal.into(),
            completed,
        };
        transfer.validate()?;
        Ok(transfer)
    }

    /// Validates transfer endpoints, payload accounting, and queue evidence.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_topic.trim().is_empty()
            || self.from_node.trim().is_empty()
            || self.to_node.trim().is_empty()
            || self.from_node == self.to_node
            || self.payload_bytes == 0
            || self.counted_copy_bytes == 0
            || self.counted_copy_bytes > self.payload_bytes
            || self.queue_signal.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "edge partition transfers require distinct nodes and consistent byte receipts"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate graph, queue, source, and calibration gates for one run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct EdgePartitionSummary {
    /// Number of packets present in the upstream live-publish receipt.
    pub source_packet_count: u64,
    /// Number of packets admitted to the transfer queue.
    pub admitted_transfer_count: u64,
    /// Number of packets recorded as completed transfers.
    pub completed_transfer_count: u64,
    /// Sum of admitted transfer payload bytes.
    pub payload_bytes: u64,
    /// Sum of measurable explicit-copy bytes.
    pub counted_copy_bytes: u64,
    /// Maximum queue depth observed during the run.
    pub max_queue_depth: u64,
    /// Number of soft-watermark observations.
    pub soft_limit_trips: u64,
    /// Number of hard-limit rejections.
    pub hard_rejects: u64,
    /// Whether packet sequences were transferred in deterministic order.
    pub deterministic_order_verified: bool,
    /// Whether the upstream live-publish state passed its own admission gate.
    pub upstream_publish_ready: bool,
    /// Whether the current operation's source checksum matches.
    pub source_identity_match: bool,
    /// Whether the upstream source/frame gate passed.
    pub frame_identity_match: bool,
    /// Whether source-bound calibration registration is present.
    pub calibration_registered: bool,
    /// Whether a clock/frame calibration was actually applied.
    pub calibration_applied: bool,
    /// Timestamp domain retained by the upstream packets.
    pub time_basis: String,
}

impl EdgePartitionSummary {
    /// Creates and validates aggregate edge-partition counters.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_packet_count: u64,
        admitted_transfer_count: u64,
        completed_transfer_count: u64,
        payload_bytes: u64,
        counted_copy_bytes: u64,
        max_queue_depth: u64,
        soft_limit_trips: u64,
        hard_rejects: u64,
        deterministic_order_verified: bool,
        upstream_publish_ready: bool,
        source_identity_match: bool,
        frame_identity_match: bool,
        calibration_registered: bool,
        calibration_applied: bool,
        time_basis: impl Into<String>,
    ) -> ViewerResult<Self> {
        let summary = Self {
            source_packet_count,
            admitted_transfer_count,
            completed_transfer_count,
            payload_bytes,
            counted_copy_bytes,
            max_queue_depth,
            soft_limit_trips,
            hard_rejects,
            deterministic_order_verified,
            upstream_publish_ready,
            source_identity_match,
            frame_identity_match,
            calibration_registered,
            calibration_applied,
            time_basis: time_basis.into(),
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates monotonic transfer, queue, and calibration counters.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.time_basis.trim().is_empty()
            || self.admitted_transfer_count > self.source_packet_count
            || self.completed_transfer_count > self.admitted_transfer_count
            || (self.admitted_transfer_count > 0 && self.payload_bytes == 0)
            || (self.completed_transfer_count > 0 && self.counted_copy_bytes == 0)
            || self.counted_copy_bytes > self.payload_bytes
            || (self.max_queue_depth > 0 && self.admitted_transfer_count == 0)
            || (self.calibration_applied && !self.calibration_registered)
        {
            return Err(ViewerError::InvalidState(
                "edge partition summary has invalid counters or calibration ordering".into(),
            ));
        }
        Ok(())
    }
}

/// Portable source-bound receipt for explicit edge-to-host execution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct EdgePartitionState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Exact source identity used by this partition run.
    pub source: StudioSource,
    /// Absolute path to the upstream live-publish JSON receipt.
    pub upstream_live_publish_path: String,
    /// Absolute path to the calibration readiness receipt.
    pub calibration_readiness_path: String,
    /// Graph partitions represented by the receipt.
    pub partitions: Vec<EdgePartition>,
    /// Ordered packet transfer receipts.
    pub transfers: Vec<EdgePartitionTransfer>,
    /// Aggregate partition and admission metrics.
    pub summary: EdgePartitionSummary,
    /// Checksummed upstream and output artifacts.
    pub artifacts: Vec<ReplayArtifact>,
    /// Whether edge-to-host packet execution passed all source/queue gates.
    pub partition_ready: bool,
    /// Whether calibrated-world mapping is admitted.
    pub mapping_admitted: bool,
    /// Human-readable fail-closed reasons.
    pub blockers: Vec<String>,
}

impl EdgePartitionState {
    /// Creates state and derives partition/mapping admission from its receipts.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        upstream_live_publish_path: impl Into<String>,
        calibration_readiness_path: impl Into<String>,
        partitions: Vec<EdgePartition>,
        transfers: Vec<EdgePartitionTransfer>,
        summary: EdgePartitionSummary,
        artifacts: Vec<ReplayArtifact>,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let partition_ready = source.identity_matches
            && summary.source_identity_match
            && summary.upstream_publish_ready
            && summary.frame_identity_match
            && summary.deterministic_order_verified
            && summary.source_packet_count > 0
            && summary.admitted_transfer_count == summary.source_packet_count
            && summary.completed_transfer_count == summary.admitted_transfer_count
            && summary.hard_rejects == 0;
        let mapping_admitted = partition_ready && summary.calibration_applied;
        let state = Self {
            version: EDGE_PARTITION_STATE_VERSION,
            title: title.into(),
            source,
            upstream_live_publish_path: upstream_live_publish_path.into(),
            calibration_readiness_path: calibration_readiness_path.into(),
            partitions,
            transfers,
            summary,
            artifacts,
            partition_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates graph membership, transfer accounting, and admission gates.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != EDGE_PARTITION_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported edge partition state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty()
            || self.upstream_live_publish_path.trim().is_empty()
            || self.calibration_readiness_path.trim().is_empty()
            || self.partitions.len() < 2
        {
            return Err(ViewerError::InvalidState(
                "edge partition state requires title, input paths, and at least two partitions"
                    .into(),
            ));
        }
        self.source.validate()?;
        self.summary.validate()?;

        let mut partition_ids = BTreeSet::new();
        let mut node_ids = BTreeSet::new();
        let mut node_partition = std::collections::BTreeMap::new();
        for partition in &self.partitions {
            partition.validate()?;
            if !partition_ids.insert(&partition.id) {
                return Err(ViewerError::InvalidState("edge partition IDs must be unique".into()));
            }
            for node_id in &partition.node_ids {
                if !node_ids.insert(node_id) {
                    return Err(ViewerError::InvalidState(
                        "edge partition nodes must belong to exactly one partition".into(),
                    ));
                }
                node_partition.insert(node_id, &partition.id);
            }
        }

        let mut transfer_payload_bytes = 0_u64;
        let mut transfer_copy_bytes = 0_u64;
        let mut completed_transfer_count = 0_u64;
        for (expected_sequence, transfer) in self.transfers.iter().enumerate() {
            transfer.validate()?;
            if transfer.sequence != u64::try_from(expected_sequence).unwrap_or(u64::MAX)
                || !node_partition.contains_key(&transfer.from_node)
                || !node_partition.contains_key(&transfer.to_node)
                || node_partition.get(&transfer.from_node) == node_partition.get(&transfer.to_node)
            {
                return Err(ViewerError::InvalidState(
                    "edge partition transfers have invalid order or graph membership".into(),
                ));
            }
            transfer_payload_bytes =
                transfer_payload_bytes.checked_add(transfer.payload_bytes).ok_or_else(|| {
                    ViewerError::InvalidState("edge partition payload count overflow".into())
                })?;
            transfer_copy_bytes =
                transfer_copy_bytes.checked_add(transfer.counted_copy_bytes).ok_or_else(|| {
                    ViewerError::InvalidState("edge partition copy count overflow".into())
                })?;
            if transfer.completed {
                completed_transfer_count =
                    completed_transfer_count.checked_add(1).ok_or_else(|| {
                        ViewerError::InvalidState("edge partition completion count overflow".into())
                    })?;
            }
        }
        if transfer_payload_bytes != self.summary.payload_bytes
            || transfer_copy_bytes != self.summary.counted_copy_bytes
            || u64::try_from(self.transfers.len()).unwrap_or(u64::MAX)
                != self.summary.admitted_transfer_count
            || completed_transfer_count != self.summary.completed_transfer_count
        {
            return Err(ViewerError::InvalidState(
                "edge partition transfer totals disagree with summary".into(),
            ));
        }
        if (!self.source.identity_matches || !self.summary.upstream_publish_ready)
            && !self.transfers.is_empty()
        {
            return Err(ViewerError::InvalidState(
                "source- or upstream-mismatched edge state cannot contain transfers".into(),
            ));
        }

        let calculated_partition_ready = self.source.identity_matches
            && self.summary.source_identity_match
            && self.summary.upstream_publish_ready
            && self.summary.frame_identity_match
            && self.summary.deterministic_order_verified
            && self.summary.source_packet_count > 0
            && self.summary.admitted_transfer_count == self.summary.source_packet_count
            && self.summary.completed_transfer_count == self.summary.admitted_transfer_count
            && self.summary.hard_rejects == 0;
        if self.partition_ready != calculated_partition_ready {
            return Err(ViewerError::InvalidState(
                "partition_ready disagrees with source, upstream, graph, or queue gates".into(),
            ));
        }
        let calculated_mapping = self.partition_ready && self.summary.calibration_applied;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with partition and calibration gates".into(),
            ));
        }
        if self.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted edge partition mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked edge partition mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "edge partition blockers must not contain empty messages".into(),
            ));
        }

        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "edge partition artifacts must have unique roles and paths".into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn source(identity_matches: bool) -> StudioSource {
        StudioSource::try_new(
            "canonical bag",
            "/media/input.db3",
            SHA,
            if identity_matches {
                SHA
            } else {
                "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
            },
            identity_matches,
        )
        .unwrap()
    }

    fn partitions() -> Vec<EdgePartition> {
        vec![
            EdgePartition::try_new(
                "edge",
                "edge-host-0",
                vec!["packet-gate".into(), "live-publish".into()],
            )
            .unwrap(),
            EdgePartition::try_new(
                "host",
                "mapping-host-0",
                vec!["host-receive".into(), "mapping".into()],
            )
            .unwrap(),
        ]
    }

    fn summary(source_identity_match: bool) -> EdgePartitionSummary {
        EdgePartitionSummary::try_new(
            1,
            if source_identity_match { 1 } else { 0 },
            if source_identity_match { 1 } else { 0 },
            if source_identity_match { 128 } else { 0 },
            if source_identity_match { 128 } else { 0 },
            if source_identity_match { 1 } else { 0 },
            0,
            0,
            source_identity_match,
            source_identity_match,
            source_identity_match,
            source_identity_match,
            false,
            false,
            "PointCloud2 header stamp; no clock calibration applied",
        )
        .unwrap()
    }

    #[test]
    fn healthy_partition_is_ready_while_mapping_stays_blocked() {
        let transfer = EdgePartitionTransfer::try_new(
            0,
            "/lidar_front/points_raw",
            "packet-gate",
            "host-receive",
            128,
            128,
            "soft-limit",
            true,
        )
        .unwrap();
        let state = EdgePartitionState::try_new(
            "Edge Partition",
            source(true),
            "/media/live-publish.json",
            "/media/readiness.json",
            partitions(),
            vec![transfer],
            summary(true),
            Vec::new(),
            vec!["clock/frame calibration was not applied".into()],
        )
        .unwrap();
        assert!(state.partition_ready);
        assert!(!state.mapping_admitted);
        state.validate().unwrap();
    }

    #[test]
    fn source_mismatch_withholds_transfers() {
        let state = EdgePartitionState::try_new(
            "Edge Partition",
            source(false),
            "/media/live-publish.json",
            "/media/readiness.json",
            partitions(),
            Vec::new(),
            summary(false),
            Vec::new(),
            vec!["source SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.partition_ready);
        assert!(!state.mapping_admitted);
        assert!(state.transfers.is_empty());
    }

    #[test]
    fn rejects_transfer_within_one_partition() {
        let transfer = EdgePartitionTransfer::try_new(
            0,
            "/lidar_front/points_raw",
            "packet-gate",
            "live-publish",
            128,
            128,
            "ok",
            true,
        )
        .unwrap();
        let result = EdgePartitionState::try_new(
            "Edge Partition",
            source(true),
            "/media/live-publish.json",
            "/media/readiness.json",
            partitions(),
            vec![transfer],
            summary(true),
            Vec::new(),
            vec!["invalid graph edge".into()],
        );
        assert!(result.is_err());
    }
}
