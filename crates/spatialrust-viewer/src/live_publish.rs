//! Portable state for the source-bound ROS 2 live-publish bridge.
//!
//! The bridge state distinguishes transport readiness from calibrated mapping
//! admission. A bounded point-cloud stream may be published for inspection
//! through an explicit adapter, while source/frame identity failures withhold
//! packets and calibration remains a separate mapping gate.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized live-publish state schema version.
pub const LIVE_PUBLISH_STATE_VERSION: u32 = 1;

/// One source topic mapped to one explicit ROS 2 publish topic.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LivePublishTopic {
    /// Topic read from the source episode.
    pub source_topic: String,
    /// Topic exposed by the publish adapter.
    pub publish_topic: String,
    /// Fully-qualified ROS 2 message type.
    pub message_type: String,
    /// Number of messages present in the source bag for this topic.
    pub source_message_count: u64,
    /// Number of bounded records retained from this topic.
    pub retained_record_count: u64,
    /// Number of points retained from this topic.
    pub retained_point_count: u64,
    /// Number of messages published for this topic.
    pub published_message_count: u64,
    /// Number of points published for this topic.
    pub published_point_count: u64,
    /// Frame IDs observed in the published messages.
    pub frame_ids: Vec<String>,
}

impl LivePublishTopic {
    /// Creates and validates one source-to-publish topic mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_topic: impl Into<String>,
        publish_topic: impl Into<String>,
        message_type: impl Into<String>,
        source_message_count: u64,
        retained_record_count: u64,
        retained_point_count: u64,
        published_message_count: u64,
        published_point_count: u64,
        frame_ids: Vec<String>,
    ) -> ViewerResult<Self> {
        let topic = Self {
            source_topic: source_topic.into(),
            publish_topic: publish_topic.into(),
            message_type: message_type.into(),
            source_message_count,
            retained_record_count,
            retained_point_count,
            published_message_count,
            published_point_count,
            frame_ids,
        };
        topic.validate()?;
        Ok(topic)
    }

    /// Validates names, counters, and frame identity uniqueness.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_topic.trim().is_empty()
            || self.publish_topic.trim().is_empty()
            || self.message_type.trim().is_empty()
            || (self.retained_record_count > 0 && self.retained_point_count == 0)
            || (self.published_message_count > 0 && self.published_point_count == 0)
        {
            return Err(ViewerError::InvalidState(
                "live-publish topics require names, a message type, and consistent counters".into(),
            ));
        }
        let mut frames = BTreeSet::new();
        for frame_id in &self.frame_ids {
            if frame_id.trim().is_empty() || !frames.insert(frame_id) {
                return Err(ViewerError::InvalidState(
                    "live-publish topic frame IDs must be non-empty and unique".into(),
                ));
            }
        }
        Ok(())
    }
}

/// One encoded ROS 2 packet and its explicit loopback round-trip receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LivePublishPacket {
    /// Zero-based deterministic publish sequence.
    pub sequence: u64,
    /// Source episode topic.
    pub source_topic: String,
    /// Published ROS 2 topic.
    pub publish_topic: String,
    /// PointCloud2 frame ID.
    pub frame_id: String,
    /// PointCloud2 header stamp in nanoseconds.
    pub stamp_nanos: u64,
    /// Number of points represented by the packet.
    pub point_count: u64,
    /// Encoded CDR payload bytes handed to the adapter.
    pub payload_bytes: u64,
    /// Decoded loopback payload bytes returned by the adapter.
    pub roundtrip_payload_bytes: u64,
    /// Whether the decoded message exactly matched the encoded message.
    pub roundtrip_verified: bool,
}

impl LivePublishPacket {
    /// Creates and validates one publish packet receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        sequence: u64,
        source_topic: impl Into<String>,
        publish_topic: impl Into<String>,
        frame_id: impl Into<String>,
        stamp_nanos: u64,
        point_count: u64,
        payload_bytes: u64,
        roundtrip_payload_bytes: u64,
        roundtrip_verified: bool,
    ) -> ViewerResult<Self> {
        let packet = Self {
            sequence,
            source_topic: source_topic.into(),
            publish_topic: publish_topic.into(),
            frame_id: frame_id.into(),
            stamp_nanos,
            point_count,
            payload_bytes,
            roundtrip_payload_bytes,
            roundtrip_verified,
        };
        packet.validate()?;
        Ok(packet)
    }

    /// Validates packet identity and round-trip counters.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_topic.trim().is_empty()
            || self.publish_topic.trim().is_empty()
            || self.frame_id.trim().is_empty()
            || self.point_count == 0
            || self.payload_bytes == 0
            || (self.roundtrip_verified && self.roundtrip_payload_bytes != self.payload_bytes)
        {
            return Err(ViewerError::InvalidState(
                "live-publish packets require non-empty identity, points, and payload receipts"
                    .into(),
            ));
        }
        if self.roundtrip_verified && self.roundtrip_payload_bytes == 0 {
            return Err(ViewerError::InvalidState(
                "verified live-publish packets require a non-zero round-trip payload".into(),
            ));
        }
        Ok(())
    }
}

/// Explicit CPU/adapter transport counters for a publish run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LivePublishTransport {
    /// Adapter name, for example `in-process-loopback`.
    pub adapter: String,
    /// Fully-qualified ROS 2 message type.
    pub message_type: String,
    /// Queue policy used by the adapter.
    pub queue_policy: String,
    /// Maximum number of samples retained by the adapter queue.
    pub queue_capacity: u64,
    /// Number of packets handed to the adapter.
    pub published_message_count: u64,
    /// Number of packets received back from the adapter.
    pub received_message_count: u64,
    /// Number of explicit queue/backpressure events.
    pub backpressure_event_count: u64,
    /// Bytes encoded on the host before publish.
    pub host_encode_bytes: u64,
    /// Bytes decoded on the host after receive.
    pub host_decode_bytes: u64,
    /// Explicit host-to-device bytes; zero for the CPU loopback adapter.
    pub device_upload_bytes: u64,
    /// Explicit device-to-host bytes; zero for the CPU loopback adapter.
    pub device_readback_bytes: u64,
}

impl LivePublishTransport {
    /// Creates and validates a transport receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        adapter: impl Into<String>,
        message_type: impl Into<String>,
        queue_policy: impl Into<String>,
        queue_capacity: u64,
        published_message_count: u64,
        received_message_count: u64,
        backpressure_event_count: u64,
        host_encode_bytes: u64,
        host_decode_bytes: u64,
        device_upload_bytes: u64,
        device_readback_bytes: u64,
    ) -> ViewerResult<Self> {
        let transport = Self {
            adapter: adapter.into(),
            message_type: message_type.into(),
            queue_policy: queue_policy.into(),
            queue_capacity,
            published_message_count,
            received_message_count,
            backpressure_event_count,
            host_encode_bytes,
            host_decode_bytes,
            device_upload_bytes,
            device_readback_bytes,
        };
        transport.validate()?;
        Ok(transport)
    }

    /// Validates queue and message counters.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.adapter.trim().is_empty()
            || self.message_type.trim().is_empty()
            || self.queue_policy.trim().is_empty()
            || self.queue_capacity == 0
            || self.received_message_count > self.published_message_count
        {
            return Err(ViewerError::InvalidState(
                "live-publish transport has invalid adapter, queue, or message counters".into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate counts and admission inputs for one live-publish run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LivePublishSummary {
    /// Sum of source messages for the selected source topics.
    pub source_message_count: u64,
    /// Number of bounded records admitted before publish.
    pub selected_record_count: u64,
    /// Number of points in the bounded source episode.
    pub selected_point_count: u64,
    /// Conservative allocated bytes retained by the bounded source episode.
    pub selected_bytes: u64,
    /// Largest source allocation observed while reading the bag.
    pub peak_source_bytes: u64,
    /// Number of packets handed to the adapter.
    pub published_message_count: u64,
    /// Number of packets received back from the adapter.
    pub received_message_count: u64,
    /// Number of points represented by published packets.
    pub published_point_count: u64,
    /// Whether the deterministic episode order was verified.
    pub deterministic_order_verified: bool,
    /// Whether all emitted records matched the expected frame.
    pub frame_identity_match: bool,
    /// Whether the source-bound readiness receipt registered calibration.
    pub calibration_registered: bool,
    /// Whether a clock/frame transform was actually applied to packets.
    pub calibration_applied: bool,
    /// Timestamp domain exposed by the published messages.
    pub time_basis: String,
}

impl LivePublishSummary {
    /// Creates and validates publish counters and gates.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_message_count: u64,
        selected_record_count: u64,
        selected_point_count: u64,
        selected_bytes: u64,
        peak_source_bytes: u64,
        published_message_count: u64,
        received_message_count: u64,
        published_point_count: u64,
        deterministic_order_verified: bool,
        frame_identity_match: bool,
        calibration_registered: bool,
        calibration_applied: bool,
        time_basis: impl Into<String>,
    ) -> ViewerResult<Self> {
        let summary = Self {
            source_message_count,
            selected_record_count,
            selected_point_count,
            selected_bytes,
            peak_source_bytes,
            published_message_count,
            received_message_count,
            published_point_count,
            deterministic_order_verified,
            frame_identity_match,
            calibration_registered,
            calibration_applied,
            time_basis: time_basis.into(),
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates monotonic counters and calibration ordering.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.time_basis.trim().is_empty()
            || self.published_message_count > self.selected_record_count
            || self.received_message_count > self.published_message_count
            || (self.selected_record_count > 0 && self.selected_point_count == 0)
            || (self.published_message_count > 0 && self.published_point_count == 0)
            || (self.calibration_applied && !self.calibration_registered)
        {
            return Err(ViewerError::InvalidState(
                "live-publish summary has invalid counters or calibration ordering".into(),
            ));
        }
        Ok(())
    }
}

/// Portable source-bound state emitted by the ROS 2 live-publish bridge.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct LivePublishState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Exact input identity.
    pub source: StudioSource,
    /// Source-topic to expected frame identity map.
    pub expected_frame_ids: BTreeMap<String, String>,
    /// Time basis exposed by the bridge.
    pub time_basis: String,
    /// Source-to-publish topic inventory.
    pub topics: Vec<LivePublishTopic>,
    /// Explicit adapter and transfer counters.
    pub transport: LivePublishTransport,
    /// Ordered packet receipts.
    pub packets: Vec<LivePublishPacket>,
    /// Aggregate publish metrics and calibration gates.
    pub summary: LivePublishSummary,
    /// Checksummed state/dashboard/input artifacts.
    pub artifacts: Vec<ReplayArtifact>,
    /// Whether source-bound packet publish and round-trip checks passed.
    pub publish_ready: bool,
    /// Whether calibrated-world mapping is admitted.
    pub mapping_admitted: bool,
    /// Human-readable fail-closed reasons.
    pub blockers: Vec<String>,
}

impl LivePublishState {
    /// Creates state and derives publish/mapping admission from its receipts.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        expected_frame_ids: BTreeMap<String, String>,
        time_basis: impl Into<String>,
        topics: Vec<LivePublishTopic>,
        transport: LivePublishTransport,
        packets: Vec<LivePublishPacket>,
        summary: LivePublishSummary,
        artifacts: Vec<ReplayArtifact>,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let all_roundtrips_verified = !packets.is_empty()
            && packets.iter().all(|packet| {
                packet.roundtrip_verified && packet.roundtrip_payload_bytes == packet.payload_bytes
            });
        let packet_count = u64::try_from(packets.len()).map_err(|_| {
            ViewerError::InvalidState("live-publish packet count does not fit in u64".into())
        })?;
        let publish_ready = source.identity_matches
            && summary.frame_identity_match
            && summary.deterministic_order_verified
            && summary.selected_record_count > 0
            && packet_count == summary.selected_record_count
            && summary.published_message_count == summary.selected_record_count
            && summary.received_message_count == summary.published_message_count
            && all_roundtrips_verified;
        let mapping_admitted = publish_ready && summary.calibration_applied;
        let state = Self {
            version: LIVE_PUBLISH_STATE_VERSION,
            title: title.into(),
            source,
            expected_frame_ids,
            time_basis: time_basis.into(),
            topics,
            transport,
            packets,
            summary,
            artifacts,
            publish_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates packet, transport, artifact, and admission invariants.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != LIVE_PUBLISH_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported live-publish state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty()
            || self.time_basis.trim().is_empty()
            || self.expected_frame_ids.is_empty()
        {
            return Err(ViewerError::InvalidState(
                "live-publish state title, expected frame map, and time basis are required".into(),
            ));
        }
        self.source.validate()?;
        self.summary.validate()?;
        self.transport.validate()?;

        let mut source_topics = BTreeSet::new();
        let mut publish_topics = BTreeSet::new();
        for topic in &self.topics {
            topic.validate()?;
            if !source_topics.insert(&topic.source_topic)
                || !publish_topics.insert(&topic.publish_topic)
            {
                return Err(ViewerError::InvalidState(
                    "live-publish topics must have unique source and publish names".into(),
                ));
            }
            if topic.message_type != self.transport.message_type {
                return Err(ViewerError::InvalidState(
                    "live-publish topic message type disagrees with transport".into(),
                ));
            }
            let Some(frame_id) = self.expected_frame_ids.get(&topic.source_topic) else {
                return Err(ViewerError::InvalidState(
                    "live-publish topic is missing an expected frame identity".into(),
                ));
            };
            if frame_id.trim().is_empty() {
                return Err(ViewerError::InvalidState(
                    "live-publish expected frame identities must be non-empty".into(),
                ));
            }
        }

        let mut packet_points = 0_u64;
        let mut packet_payload_bytes = 0_u64;
        let mut roundtrip_payload_bytes = 0_u64;
        for (expected_sequence, packet) in self.packets.iter().enumerate() {
            packet.validate()?;
            if packet.sequence != u64::try_from(expected_sequence).unwrap_or(u64::MAX)
                || !source_topics.contains(&packet.source_topic)
                || !publish_topics.contains(&packet.publish_topic)
                || self.expected_frame_ids.get(&packet.source_topic) != Some(&packet.frame_id)
            {
                return Err(ViewerError::InvalidState(
                    "live-publish packets have invalid sequence, topic, or frame identity".into(),
                ));
            }
            packet_points = packet_points.checked_add(packet.point_count).ok_or_else(|| {
                ViewerError::InvalidState("live-publish packet point count overflow".into())
            })?;
            packet_payload_bytes =
                packet_payload_bytes.checked_add(packet.payload_bytes).ok_or_else(|| {
                    ViewerError::InvalidState("live-publish payload count overflow".into())
                })?;
            roundtrip_payload_bytes =
                roundtrip_payload_bytes.checked_add(packet.roundtrip_payload_bytes).ok_or_else(
                    || ViewerError::InvalidState("live-publish round-trip count overflow".into()),
                )?;
        }
        if !self.source.identity_matches && !self.packets.is_empty() {
            return Err(ViewerError::InvalidState(
                "source-mismatched live-publish state cannot contain packets".into(),
            ));
        }
        if packet_points != self.summary.published_point_count
            || packet_payload_bytes != self.transport.host_encode_bytes
            || roundtrip_payload_bytes != self.transport.host_decode_bytes
        {
            return Err(ViewerError::InvalidState(
                "live-publish packet totals disagree with transport summary".into(),
            ));
        }
        if self.transport.published_message_count != self.summary.published_message_count
            || self.transport.received_message_count != self.summary.received_message_count
            || u64::try_from(self.packets.len()).unwrap_or(u64::MAX)
                != self.transport.published_message_count
        {
            return Err(ViewerError::InvalidState(
                "live-publish message totals disagree with transport summary".into(),
            ));
        }
        let calculated_publish_ready = self.source.identity_matches
            && self.summary.frame_identity_match
            && self.summary.deterministic_order_verified
            && self.summary.selected_record_count > 0
            && u64::try_from(self.packets.len()).unwrap_or(u64::MAX)
                == self.summary.selected_record_count
            && self.summary.published_message_count == self.summary.selected_record_count
            && self.summary.received_message_count == self.summary.published_message_count
            && self.packets.iter().all(|packet| {
                packet.roundtrip_verified && packet.roundtrip_payload_bytes == packet.payload_bytes
            });
        if self.publish_ready != calculated_publish_ready {
            return Err(ViewerError::InvalidState(
                "publish_ready disagrees with source, frame, packet, or round-trip gates".into(),
            ));
        }
        let calculated_mapping = self.publish_ready && self.summary.calibration_applied;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with publish and calibration gates".into(),
            ));
        }
        if self.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted live-publish mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked live-publish mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "live-publish blockers must not contain empty messages".into(),
            ));
        }
        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "live-publish artifacts must have unique roles and paths".into(),
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

    fn topic() -> LivePublishTopic {
        LivePublishTopic::try_new(
            "/lidar_front/points_raw",
            "/spatialrust/lidar_front/points_raw",
            "sensor_msgs/msg/PointCloud2",
            1,
            1,
            2,
            1,
            2,
            vec!["lidar_front".into()],
        )
        .unwrap()
    }

    fn packet() -> LivePublishPacket {
        LivePublishPacket::try_new(
            0,
            "/lidar_front/points_raw",
            "/spatialrust/lidar_front/points_raw",
            "lidar_front",
            10,
            2,
            24,
            24,
            true,
        )
        .unwrap()
    }

    fn transport() -> LivePublishTransport {
        LivePublishTransport::try_new(
            "in-process-loopback",
            "sensor_msgs/msg/PointCloud2",
            "replace-latest-per-topic",
            1,
            1,
            1,
            0,
            24,
            24,
            0,
            0,
        )
        .unwrap()
    }

    fn summary(frame_identity_match: bool) -> LivePublishSummary {
        LivePublishSummary::try_new(
            1,
            1,
            2,
            24,
            24,
            1,
            1,
            2,
            true,
            frame_identity_match,
            false,
            false,
            "PointCloud2 header stamp; no clock calibration applied",
        )
        .unwrap()
    }

    #[test]
    fn healthy_publish_is_ready_while_mapping_stays_blocked() {
        let expected_frames =
            BTreeMap::from([("/lidar_front/points_raw".to_owned(), "lidar_front".to_owned())]);
        let state = LivePublishState::try_new(
            "Live Publish",
            source(true),
            expected_frames,
            "PointCloud2 header stamp",
            vec![topic()],
            transport(),
            vec![packet()],
            summary(true),
            Vec::new(),
            vec!["clock/frame calibration was not applied".into()],
        )
        .unwrap();
        assert!(state.publish_ready);
        assert!(!state.mapping_admitted);
        state.validate().unwrap();
    }

    #[test]
    fn source_mismatch_withholds_packets() {
        let expected_frames =
            BTreeMap::from([("/lidar_front/points_raw".to_owned(), "lidar_front".to_owned())]);
        let state = LivePublishState::try_new(
            "Live Publish",
            source(false),
            expected_frames,
            "PointCloud2 header stamp",
            vec![LivePublishTopic::try_new(
                "/lidar_front/points_raw",
                "/spatialrust/lidar_front/points_raw",
                "sensor_msgs/msg/PointCloud2",
                1,
                0,
                0,
                0,
                0,
                Vec::new(),
            )
            .unwrap()],
            LivePublishTransport::try_new(
                "in-process-loopback",
                "sensor_msgs/msg/PointCloud2",
                "replace-latest-per-topic",
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            .unwrap(),
            Vec::new(),
            LivePublishSummary::try_new(
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                false,
                false,
                false,
                false,
                "PointCloud2 header stamp",
            )
            .unwrap(),
            Vec::new(),
            vec!["source SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.publish_ready);
        assert!(!state.mapping_admitted);
        assert!(state.packets.is_empty());
    }
}
