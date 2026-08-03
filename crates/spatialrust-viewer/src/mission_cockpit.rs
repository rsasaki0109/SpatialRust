//! Portable state for the interactive, source-bound mission cockpit.
//!
//! The cockpit is deliberately a bounded inspection contract.  A frame keeps
//! a small, source-indexed sample for interaction while the original packet
//! point count, topic, frame, timestamp, and transfer receipt remain visible.
//! It never implies that a sample has been transformed into a calibrated world
//! frame.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized mission-cockpit state schema version.
pub const MISSION_COCKPIT_STATE_VERSION: u32 = 1;

/// Maximum number of sampled points retained in one interactive frame.
pub const MISSION_COCKPIT_MAX_SAMPLED_POINTS: usize = 4_096;

/// One finite XYZ point retained for bounded cockpit interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitPoint {
    /// X coordinate in the packet's original frame.
    pub x: f32,
    /// Y coordinate in the packet's original frame.
    pub y: f32,
    /// Z coordinate in the packet's original frame.
    pub z: f32,
}

impl MissionCockpitPoint {
    /// Creates a finite point.
    pub fn try_new(x: f32, y: f32, z: f32) -> ViewerResult<Self> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(ViewerError::InvalidState(
                "mission cockpit points must contain finite XYZ values".into(),
            ));
        }
        Ok(Self { x, y, z })
    }

    fn validate(&self) -> ViewerResult<()> {
        Self::try_new(self.x, self.y, self.z).map(|_| ())
    }
}

/// One packet frame with a bounded, source-indexed point sample.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitFrame {
    /// Zero-based packet sequence in deterministic replay order.
    pub sequence: u64,
    /// Source topic represented by the frame.
    pub source_topic: String,
    /// Published topic represented by the frame.
    pub publish_topic: String,
    /// Original packet frame identity.
    pub frame_id: String,
    /// Original PointCloud2 header timestamp in nanoseconds.
    pub stamp_nanos: u64,
    /// Full point count in the source packet.
    pub point_count: u64,
    /// Original source indices corresponding one-to-one with `sampled_points`.
    pub sampled_source_indices: Vec<u64>,
    /// Bounded XYZ sample retained for local interaction.
    pub sampled_points: Vec<MissionCockpitPoint>,
}

impl MissionCockpitFrame {
    /// Creates and validates one sampled packet frame.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        sequence: u64,
        source_topic: impl Into<String>,
        publish_topic: impl Into<String>,
        frame_id: impl Into<String>,
        stamp_nanos: u64,
        point_count: u64,
        sampled_source_indices: Vec<u64>,
        sampled_points: Vec<MissionCockpitPoint>,
    ) -> ViewerResult<Self> {
        let frame = Self {
            sequence,
            source_topic: source_topic.into(),
            publish_topic: publish_topic.into(),
            frame_id: frame_id.into(),
            stamp_nanos,
            point_count,
            sampled_source_indices,
            sampled_points,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Validates identity, source-index ordering, and bounded geometry.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_topic.trim().is_empty()
            || self.publish_topic.trim().is_empty()
            || self.frame_id.trim().is_empty()
            || self.point_count == 0
            || self.sampled_points.is_empty()
            || self.sampled_points.len() != self.sampled_source_indices.len()
            || self.sampled_points.len() > MISSION_COCKPIT_MAX_SAMPLED_POINTS
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit frames require bounded identity and point samples".into(),
            ));
        }
        let mut previous = None;
        for (source_index, point) in self.sampled_source_indices.iter().zip(&self.sampled_points) {
            if *source_index >= self.point_count
                || previous.is_some_and(|previous| *source_index <= previous)
            {
                return Err(ViewerError::InvalidState(
                    "mission cockpit source indices must be strictly increasing and in range"
                        .into(),
                ));
            }
            point.validate()?;
            previous = Some(*source_index);
        }
        Ok(())
    }
}

/// One visual layer exposed by the cockpit.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitLayer {
    /// Stable layer identity.
    pub id: String,
    /// User-facing layer label.
    pub label: String,
    /// Portable layer kind, such as `point-cloud` or `transfer-graph`.
    pub kind: String,
    /// Whether the layer is initially visible.
    pub visible: bool,
    /// Source topic identities represented by this layer.
    pub frame_ids: Vec<String>,
    /// RGB display color used by the portable dashboard.
    pub color_rgb: [u8; 3],
}

impl MissionCockpitLayer {
    /// Creates and validates one layer.
    pub fn try_new(
        id: impl Into<String>,
        label: impl Into<String>,
        kind: impl Into<String>,
        visible: bool,
        frame_ids: Vec<String>,
        color_rgb: [u8; 3],
    ) -> ViewerResult<Self> {
        let layer = Self {
            id: id.into(),
            label: label.into(),
            kind: kind.into(),
            visible,
            frame_ids,
            color_rgb,
        };
        layer.validate()?;
        Ok(layer)
    }

    /// Validates the layer identity and topic list.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty() || self.label.trim().is_empty() || self.kind.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit layers require identity, label, and kind".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        if self.frame_ids.iter().any(|id| id.trim().is_empty() || !ids.insert(id)) {
            return Err(ViewerError::InvalidState(
                "mission cockpit layer topic identities must be non-empty and unique".into(),
            ));
        }
        Ok(())
    }
}

/// One execution node shown in the transfer graph.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitNode {
    /// Stable node identity.
    pub id: String,
    /// Partition containing the node.
    pub partition_id: String,
    /// Placement label, for example `edge` or `host`.
    pub placement: String,
    /// Normalized display X coordinate in the graph panel.
    pub display_x: f32,
    /// Normalized display Y coordinate in the graph panel.
    pub display_y: f32,
}

impl MissionCockpitNode {
    /// Creates and validates one graph node.
    pub fn try_new(
        id: impl Into<String>,
        partition_id: impl Into<String>,
        placement: impl Into<String>,
        display_x: f32,
        display_y: f32,
    ) -> ViewerResult<Self> {
        let node = Self {
            id: id.into(),
            partition_id: partition_id.into(),
            placement: placement.into(),
            display_x,
            display_y,
        };
        node.validate()?;
        Ok(node)
    }

    /// Validates graph-node identity and display coordinates.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty()
            || self.partition_id.trim().is_empty()
            || self.placement.trim().is_empty()
            || !self.display_x.is_finite()
            || !self.display_y.is_finite()
            || !(0.0..=1.0).contains(&self.display_x)
            || !(0.0..=1.0).contains(&self.display_y)
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit nodes require bounded identity and display coordinates".into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate receipt for one directed execution lane.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitLink {
    /// Source node identity.
    pub from_node: String,
    /// Destination node identity.
    pub to_node: String,
    /// Number of packet transfers represented by this lane.
    pub transfer_count: u64,
    /// Number of completed packet transfers.
    pub completed_transfer_count: u64,
    /// Sum of packet payload bytes.
    pub payload_bytes: u64,
    /// Sum of explicit-copy bytes.
    pub counted_copy_bytes: u64,
}

impl MissionCockpitLink {
    /// Creates and validates one directed execution lane.
    pub fn try_new(
        from_node: impl Into<String>,
        to_node: impl Into<String>,
        transfer_count: u64,
        completed_transfer_count: u64,
        payload_bytes: u64,
        counted_copy_bytes: u64,
    ) -> ViewerResult<Self> {
        let link = Self {
            from_node: from_node.into(),
            to_node: to_node.into(),
            transfer_count,
            completed_transfer_count,
            payload_bytes,
            counted_copy_bytes,
        };
        link.validate()?;
        Ok(link)
    }

    /// Validates lane endpoints and byte/counter monotonicity.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.from_node.trim().is_empty()
            || self.to_node.trim().is_empty()
            || self.from_node == self.to_node
            || self.completed_transfer_count > self.transfer_count
            || (self.transfer_count > 0 && self.payload_bytes == 0)
            || self.counted_copy_bytes > self.payload_bytes
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit links require distinct nodes and consistent counters".into(),
            ));
        }
        Ok(())
    }
}

/// Cursor and bounds for the interactive packet timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitTimeline {
    /// Timestamp domain used by the packet headers.
    pub time_basis: String,
    /// First admitted frame timestamp, or zero when unavailable.
    pub start_nanos: u64,
    /// Last admitted frame timestamp, or zero when unavailable.
    pub end_nanos: u64,
    /// Initial cursor timestamp.
    pub cursor_nanos: u64,
    /// Number of admitted packet frames.
    pub frame_count: u64,
}

impl MissionCockpitTimeline {
    /// Creates and validates timeline bounds.
    pub fn try_new(
        time_basis: impl Into<String>,
        start_nanos: u64,
        end_nanos: u64,
        cursor_nanos: u64,
        frame_count: u64,
    ) -> ViewerResult<Self> {
        let timeline = Self {
            time_basis: time_basis.into(),
            start_nanos,
            end_nanos,
            cursor_nanos,
            frame_count,
        };
        timeline.validate()?;
        Ok(timeline)
    }

    /// Validates the unavailable and admitted timeline representations.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.time_basis.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "mission cockpit timeline requires a time basis".into(),
            ));
        }
        if self.frame_count == 0 {
            if self.start_nanos != 0 || self.end_nanos != 0 || self.cursor_nanos != 0 {
                return Err(ViewerError::InvalidState(
                    "empty mission cockpit timelines must have zero bounds".into(),
                ));
            }
        } else if self.start_nanos > self.end_nanos
            || self.cursor_nanos < self.start_nanos
            || self.cursor_nanos > self.end_nanos
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit timeline bounds or cursor are invalid".into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate counts and admission inputs shown by the cockpit.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitSummary {
    /// Number of upstream packets eligible for this cockpit.
    pub source_packet_count: u64,
    /// Number of packet frames retained in the cockpit.
    pub frame_count: u64,
    /// Sum of full source packet points.
    pub total_point_count: u64,
    /// Sum of bounded interactive sample points.
    pub sampled_point_count: u64,
    /// Number of packet transfers represented by graph links.
    pub transfer_count: u64,
    /// Number of completed packet transfers.
    pub completed_transfer_count: u64,
    /// Sum of packet payload bytes.
    pub payload_bytes: u64,
    /// Sum of explicit-copy bytes.
    pub counted_copy_bytes: u64,
    /// Upstream live-publish admission.
    pub upstream_publish_ready: bool,
    /// Upstream edge-partition admission.
    pub upstream_partition_ready: bool,
    /// Whether source-bound calibration registration exists.
    pub calibration_registered: bool,
    /// Whether calibrated transforms were applied.
    pub calibration_applied: bool,
    /// Timestamp domain shown in the timeline.
    pub time_basis: String,
}

impl MissionCockpitSummary {
    /// Creates and validates cockpit counters.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_packet_count: u64,
        frame_count: u64,
        total_point_count: u64,
        sampled_point_count: u64,
        transfer_count: u64,
        completed_transfer_count: u64,
        payload_bytes: u64,
        counted_copy_bytes: u64,
        upstream_publish_ready: bool,
        upstream_partition_ready: bool,
        calibration_registered: bool,
        calibration_applied: bool,
        time_basis: impl Into<String>,
    ) -> ViewerResult<Self> {
        let summary = Self {
            source_packet_count,
            frame_count,
            total_point_count,
            sampled_point_count,
            transfer_count,
            completed_transfer_count,
            payload_bytes,
            counted_copy_bytes,
            upstream_publish_ready,
            upstream_partition_ready,
            calibration_registered,
            calibration_applied,
            time_basis: time_basis.into(),
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates aggregate counters and calibration ordering.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.time_basis.trim().is_empty()
            || self.frame_count > self.source_packet_count
            || self.completed_transfer_count > self.transfer_count
            || (self.frame_count > 0 && self.total_point_count == 0)
            || (self.sampled_point_count > self.total_point_count)
            || (self.transfer_count > 0 && self.payload_bytes == 0)
            || self.counted_copy_bytes > self.payload_bytes
            || (self.calibration_applied && !self.calibration_registered)
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit summary has invalid counters or calibration ordering".into(),
            ));
        }
        Ok(())
    }
}

/// Portable source-bound state shared by the interactive cockpit surfaces.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MissionCockpitState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing cockpit title.
    pub title: String,
    /// Exact canonical source identity.
    pub source: StudioSource,
    /// Source topic to expected frame map.
    pub expected_frame_ids: BTreeMap<String, String>,
    /// Interactive packet timeline.
    pub timeline: MissionCockpitTimeline,
    /// Bounded packet frames.
    pub frames: Vec<MissionCockpitFrame>,
    /// Initial visual layer state.
    pub layers: Vec<MissionCockpitLayer>,
    /// Execution graph nodes.
    pub nodes: Vec<MissionCockpitNode>,
    /// Aggregate execution graph links.
    pub links: Vec<MissionCockpitLink>,
    /// Aggregate counts and admission inputs.
    pub summary: MissionCockpitSummary,
    /// Checksummed source and upstream artifacts.
    pub artifacts: Vec<ReplayArtifact>,
    /// Whether source-bound packet inspection is admitted.
    pub publish_ready: bool,
    /// Whether source-bound edge execution is admitted.
    pub partition_ready: bool,
    /// Whether calibrated-world mapping is admitted.
    pub mapping_admitted: bool,
    /// Human-readable fail-closed reasons.
    pub blockers: Vec<String>,
}

impl MissionCockpitState {
    /// Creates state and derives its three admission levels from the receipts.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        expected_frame_ids: BTreeMap<String, String>,
        timeline: MissionCockpitTimeline,
        frames: Vec<MissionCockpitFrame>,
        layers: Vec<MissionCockpitLayer>,
        nodes: Vec<MissionCockpitNode>,
        links: Vec<MissionCockpitLink>,
        summary: MissionCockpitSummary,
        artifacts: Vec<ReplayArtifact>,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let publish_ready = source.identity_matches
            && summary.upstream_publish_ready
            && summary.source_packet_count > 0
            && summary.frame_count == summary.source_packet_count
            && summary.frame_count == u64::try_from(frames.len()).unwrap_or(u64::MAX);
        let partition_ready = publish_ready
            && summary.upstream_partition_ready
            && summary.transfer_count == summary.source_packet_count
            && summary.completed_transfer_count == summary.transfer_count;
        let mapping_admitted = partition_ready && summary.calibration_applied;
        let state = Self {
            version: MISSION_COCKPIT_STATE_VERSION,
            title: title.into(),
            source,
            expected_frame_ids,
            timeline,
            frames,
            layers,
            nodes,
            links,
            summary,
            artifacts,
            publish_ready,
            partition_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates all frame, graph, receipt, and admission invariants.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != MISSION_COCKPIT_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported mission cockpit state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() || self.expected_frame_ids.is_empty() {
            return Err(ViewerError::InvalidState(
                "mission cockpit state requires title and expected frame identities".into(),
            ));
        }
        self.source.validate()?;
        self.timeline.validate()?;
        self.summary.validate()?;

        let mut expected_frames = BTreeSet::new();
        for (topic, frame_id) in &self.expected_frame_ids {
            if topic.trim().is_empty()
                || frame_id.trim().is_empty()
                || !expected_frames.insert(topic)
            {
                return Err(ViewerError::InvalidState(
                    "mission cockpit expected frame identities must be non-empty and unique".into(),
                ));
            }
        }

        let mut frame_topics = BTreeSet::new();
        let mut total_points = 0_u64;
        let mut sampled_points = 0_u64;
        for (expected_sequence, frame) in self.frames.iter().enumerate() {
            frame.validate()?;
            if frame.sequence != u64::try_from(expected_sequence).unwrap_or(u64::MAX)
                || !expected_frames.contains(&frame.source_topic)
                || self.expected_frame_ids.get(&frame.source_topic) != Some(&frame.frame_id)
            {
                return Err(ViewerError::InvalidState(
                    "mission cockpit frames have invalid sequence, topic, or frame identity".into(),
                ));
            }
            frame_topics.insert(frame.source_topic.clone());
            total_points = total_points.checked_add(frame.point_count).ok_or_else(|| {
                ViewerError::InvalidState("mission cockpit point count overflow".into())
            })?;
            sampled_points = sampled_points
                .checked_add(u64::try_from(frame.sampled_points.len()).unwrap_or(u64::MAX))
                .ok_or_else(|| {
                    ViewerError::InvalidState("mission cockpit sample count overflow".into())
                })?;
        }

        let mut layer_ids = BTreeSet::new();
        for layer in &self.layers {
            layer.validate()?;
            if !layer_ids.insert(&layer.id) {
                return Err(ViewerError::InvalidState(
                    "mission cockpit layer IDs must be unique".into(),
                ));
            }
        }

        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            node.validate()?;
            if !node_ids.insert(&node.id) {
                return Err(ViewerError::InvalidState(
                    "mission cockpit node IDs must be unique".into(),
                ));
            }
        }
        let mut link_transfer_count = 0_u64;
        let mut link_completed_count = 0_u64;
        let mut link_payload_bytes = 0_u64;
        let mut link_copy_bytes = 0_u64;
        for link in &self.links {
            link.validate()?;
            if !node_ids.contains(&link.from_node) || !node_ids.contains(&link.to_node) {
                return Err(ViewerError::InvalidState(
                    "mission cockpit links must reference known graph nodes".into(),
                ));
            }
            link_transfer_count =
                link_transfer_count.checked_add(link.transfer_count).ok_or_else(|| {
                    ViewerError::InvalidState("mission cockpit transfer count overflow".into())
                })?;
            link_completed_count = link_completed_count
                .checked_add(link.completed_transfer_count)
                .ok_or_else(|| {
                    ViewerError::InvalidState("mission cockpit completion count overflow".into())
                })?;
            link_payload_bytes =
                link_payload_bytes.checked_add(link.payload_bytes).ok_or_else(|| {
                    ViewerError::InvalidState("mission cockpit payload byte overflow".into())
                })?;
            link_copy_bytes =
                link_copy_bytes.checked_add(link.counted_copy_bytes).ok_or_else(|| {
                    ViewerError::InvalidState("mission cockpit copy byte overflow".into())
                })?;
        }
        if total_points != self.summary.total_point_count
            || sampled_points != self.summary.sampled_point_count
            || link_transfer_count != self.summary.transfer_count
            || link_completed_count != self.summary.completed_transfer_count
            || link_payload_bytes != self.summary.payload_bytes
            || link_copy_bytes != self.summary.counted_copy_bytes
            || self.timeline.frame_count != u64::try_from(self.frames.len()).unwrap_or(u64::MAX)
            || self.summary.frame_count != u64::try_from(self.frames.len()).unwrap_or(u64::MAX)
        {
            return Err(ViewerError::InvalidState(
                "mission cockpit frame, link, or timeline totals disagree with summary".into(),
            ));
        }
        if self.publish_ready && frame_topics.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted mission cockpit state must contain frames".into(),
            ));
        }
        let calculated_publish = self.source.identity_matches
            && self.summary.upstream_publish_ready
            && self.summary.source_packet_count > 0
            && self.summary.frame_count == self.summary.source_packet_count
            && self.summary.frame_count == u64::try_from(self.frames.len()).unwrap_or(u64::MAX);
        if self.publish_ready != calculated_publish {
            return Err(ViewerError::InvalidState(
                "publish_ready disagrees with source, upstream, or frame gates".into(),
            ));
        }
        let calculated_partition = calculated_publish
            && self.summary.upstream_partition_ready
            && self.summary.transfer_count == self.summary.source_packet_count
            && self.summary.completed_transfer_count == self.summary.transfer_count;
        if self.partition_ready != calculated_partition {
            return Err(ViewerError::InvalidState(
                "partition_ready disagrees with source, upstream, or transfer gates".into(),
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
                "admitted mission cockpit mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked mission cockpit mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "mission cockpit blockers must not contain empty messages".into(),
            ));
        }

        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "mission cockpit artifacts must have unique roles and paths".into(),
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

    fn source(matches: bool) -> StudioSource {
        StudioSource::try_new(
            "canonical bag",
            "/media/input.db3",
            SHA,
            if matches {
                SHA
            } else {
                "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
            },
            matches,
        )
        .unwrap()
    }

    fn frame(sequence: u64) -> MissionCockpitFrame {
        MissionCockpitFrame::try_new(
            sequence,
            "/lidar_front/points_raw",
            "/spatialrust/lidar_front/points_raw",
            "lidar_front",
            10 + sequence,
            2,
            vec![0, 1],
            vec![
                MissionCockpitPoint::try_new(sequence as f32, 0.0, 0.0).unwrap(),
                MissionCockpitPoint::try_new(sequence as f32, 1.0, 0.0).unwrap(),
            ],
        )
        .unwrap()
    }

    fn layers() -> Vec<MissionCockpitLayer> {
        vec![
            MissionCockpitLayer::try_new(
                "front",
                "Front lidar",
                "point-cloud",
                true,
                vec!["/lidar_front/points_raw".into()],
                [91, 220, 255],
            )
            .unwrap(),
            MissionCockpitLayer::try_new(
                "graph",
                "Execution graph",
                "transfer-graph",
                true,
                Vec::new(),
                [99, 231, 165],
            )
            .unwrap(),
        ]
    }

    fn state(matches: bool) -> MissionCockpitState {
        let frames = if matches { vec![frame(0)] } else { Vec::new() };
        let nodes = vec![
            MissionCockpitNode::try_new("edge", "edge", "edge", 0.2, 0.5).unwrap(),
            MissionCockpitNode::try_new("host", "host", "host", 0.8, 0.5).unwrap(),
        ];
        let links = if matches {
            vec![MissionCockpitLink::try_new("edge", "host", 1, 1, 128, 128).unwrap()]
        } else {
            Vec::new()
        };
        let summary = MissionCockpitSummary::try_new(
            if matches { 1 } else { 0 },
            u64::try_from(frames.len()).unwrap(),
            if matches { 2 } else { 0 },
            if matches { 2 } else { 0 },
            if matches { 1 } else { 0 },
            if matches { 1 } else { 0 },
            if matches { 128 } else { 0 },
            if matches { 128 } else { 0 },
            matches,
            matches,
            false,
            false,
            "header stamp",
        )
        .unwrap();
        MissionCockpitState::try_new(
            "Mission Cockpit",
            source(matches),
            BTreeMap::from([("/lidar_front/points_raw".into(), "lidar_front".into())]),
            MissionCockpitTimeline::try_new(
                "header stamp",
                if matches { 10 } else { 0 },
                if matches { 10 } else { 0 },
                if matches { 10 } else { 0 },
                u64::try_from(frames.len()).unwrap(),
            )
            .unwrap(),
            frames,
            layers(),
            nodes,
            links,
            summary,
            Vec::new(),
            if matches {
                vec!["calibration not applied".into()]
            } else {
                vec!["source SHA mismatch".into()]
            },
        )
        .unwrap()
    }

    #[test]
    fn healthy_packet_and_partition_are_admitted_but_mapping_stays_blocked() {
        let cockpit = state(true);
        assert!(cockpit.publish_ready);
        assert!(cockpit.partition_ready);
        assert!(!cockpit.mapping_admitted);
        cockpit.validate().unwrap();
    }

    #[test]
    fn source_mismatch_withholds_frames_and_execution() {
        let cockpit = state(false);
        assert!(!cockpit.publish_ready);
        assert!(!cockpit.partition_ready);
        assert!(!cockpit.mapping_admitted);
        assert!(cockpit.frames.is_empty());
    }

    #[test]
    fn rejects_non_finite_sample() {
        assert!(MissionCockpitPoint::try_new(f32::NAN, 0.0, 0.0).is_err());
    }

    #[test]
    fn rejects_unsorted_source_indices() {
        let result = MissionCockpitFrame::try_new(
            0,
            "topic",
            "publish",
            "frame",
            1,
            3,
            vec![1, 0],
            vec![
                MissionCockpitPoint::try_new(0.0, 0.0, 0.0).unwrap(),
                MissionCockpitPoint::try_new(1.0, 0.0, 0.0).unwrap(),
            ],
        );
        assert!(result.is_err());
    }
}
