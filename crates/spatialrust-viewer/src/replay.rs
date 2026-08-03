//! Portable state for the one-command deterministic replay demo.
//!
//! The state records the replay trace and its admission decision without
//! depending on rosbag2, SQLite, or a renderer. Adapters can populate it from
//! a bounded episode, while native/Web frontends can render the same trace and
//! fail-closed mapping gate.

use std::collections::BTreeSet;

use crate::{StudioSource, ViewerError, ViewerResult};

/// Current serialized one-command replay demo state schema version.
pub const REPLAY_DEMO_STATE_VERSION: u32 = 1;

/// One deterministic sample emitted by an episode replayer.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ReplaySample {
    /// Zero-based sequence number in deterministic replay order.
    pub sequence: u64,
    /// Logical source topic for the replayed record.
    pub topic: String,
    /// PointCloud2 header timestamp retained by the replay.
    pub stamp_nanos: u64,
    /// Number of points in the replayed record.
    pub point_count: u64,
    /// Topics paired with this record by the bounded sync window.
    pub paired_topics: Vec<String>,
}

impl ReplaySample {
    /// Creates one validated deterministic replay sample.
    pub fn try_new(
        sequence: u64,
        topic: impl Into<String>,
        stamp_nanos: u64,
        point_count: u64,
        paired_topics: Vec<String>,
    ) -> ViewerResult<Self> {
        let sample =
            Self { sequence, topic: topic.into(), stamp_nanos, point_count, paired_topics };
        sample.validate()?;
        Ok(sample)
    }

    /// Validates topic identity, point count, and paired-topic uniqueness.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.topic.trim().is_empty() || self.point_count == 0 {
            return Err(ViewerError::InvalidState(
                "replay samples require a topic and at least one point".into(),
            ));
        }
        let mut topics = BTreeSet::new();
        for paired_topic in &self.paired_topics {
            if paired_topic.trim().is_empty()
                || paired_topic == &self.topic
                || !topics.insert(paired_topic)
            {
                return Err(ViewerError::InvalidState(
                    "replay sample paired topics must be non-empty, distinct, and external".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Bounded episode and replay counters shown by the replay dashboard.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ReplaySummary {
    /// Number of records admitted into the bounded episode.
    pub episode_record_count: u64,
    /// Point count admitted into the bounded episode.
    pub episode_point_count: u64,
    /// Conservative allocated column bytes retained by the episode.
    pub episode_byte_count: u64,
    /// Number of records emitted by deterministic replay.
    pub replayed_record_count: u64,
    /// Number of primary/secondary bundles matched by the sync window.
    pub matched_bundle_count: u64,
    /// Largest matched timestamp delta in nanoseconds.
    pub max_matched_delta_ns: u64,
    /// Configured maximum timestamp delta in nanoseconds.
    pub max_delta_ns: u64,
    /// Wall time spent in bounded ingest and replay, in nanoseconds.
    pub replay_wall_ns: u64,
    /// Largest source allocation observed while reading the bag.
    pub peak_source_bytes: u64,
    /// Whether a second deterministic walk produced the same ordered trace.
    pub deterministic_order_verified: bool,
    /// Human-readable timestamp basis for the replay.
    pub time_basis: String,
    /// Whether a source-bound clock/frame calibration was applied.
    pub calibration_applied: bool,
}

/// Bounded inventory for one point-cloud topic in the replay episode.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ReplayTopic {
    /// ROS topic name.
    pub name: String,
    /// Total messages present in the source bag for this topic.
    pub bag_message_count: u64,
    /// Number of bounded records retained from this topic.
    pub retained_record_count: u64,
    /// Number of points retained from this topic.
    pub retained_point_count: u64,
    /// Frame IDs observed in the retained records.
    pub frame_ids: Vec<String>,
}

impl ReplayTopic {
    /// Creates and validates one topic inventory.
    pub fn try_new(
        name: impl Into<String>,
        bag_message_count: u64,
        retained_record_count: u64,
        retained_point_count: u64,
        frame_ids: Vec<String>,
    ) -> ViewerResult<Self> {
        let topic = Self {
            name: name.into(),
            bag_message_count,
            retained_record_count,
            retained_point_count,
            frame_ids,
        };
        topic.validate()?;
        Ok(topic)
    }

    /// Validates topic identity, retained counters, and frame IDs.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.name.trim().is_empty()
            || (self.retained_record_count > 0 && self.retained_point_count == 0)
        {
            return Err(ViewerError::InvalidState(
                "replay topic inventory has invalid identity or retained counters".into(),
            ));
        }
        let mut frames = BTreeSet::new();
        for frame_id in &self.frame_ids {
            if frame_id.trim().is_empty() || !frames.insert(frame_id) {
                return Err(ViewerError::InvalidState(
                    "replay topic frame IDs must be non-empty and unique".into(),
                ));
            }
        }
        Ok(())
    }
}

impl ReplaySummary {
    /// Creates and validates bounded replay counters.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        episode_record_count: u64,
        episode_point_count: u64,
        episode_byte_count: u64,
        replayed_record_count: u64,
        matched_bundle_count: u64,
        max_matched_delta_ns: u64,
        max_delta_ns: u64,
        replay_wall_ns: u64,
        peak_source_bytes: u64,
        deterministic_order_verified: bool,
        time_basis: impl Into<String>,
        calibration_applied: bool,
    ) -> ViewerResult<Self> {
        let summary = Self {
            episode_record_count,
            episode_point_count,
            episode_byte_count,
            replayed_record_count,
            matched_bundle_count,
            max_matched_delta_ns,
            max_delta_ns,
            replay_wall_ns,
            peak_source_bytes,
            deterministic_order_verified,
            time_basis: time_basis.into(),
            calibration_applied,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates counter ordering, sync bounds, and the timestamp basis.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.time_basis.trim().is_empty() {
            return Err(ViewerError::InvalidState("replay time basis must not be empty".into()));
        }
        if self.replayed_record_count > self.episode_record_count {
            return Err(ViewerError::InvalidState(
                "replayed record count cannot exceed episode record count".into(),
            ));
        }
        if self.matched_bundle_count > self.replayed_record_count {
            return Err(ViewerError::InvalidState(
                "matched bundle count cannot exceed replayed record count".into(),
            ));
        }
        if self.max_delta_ns == 0 && self.max_matched_delta_ns != 0 {
            return Err(ViewerError::InvalidState(
                "a non-zero matched delta requires a positive sync window".into(),
            ));
        }
        if self.max_matched_delta_ns > self.max_delta_ns {
            return Err(ViewerError::InvalidState(
                "matched timestamp delta exceeds the configured sync window".into(),
            ));
        }
        if self.calibration_applied && !self.deterministic_order_verified {
            return Err(ViewerError::InvalidState(
                "calibration cannot be applied to an unverified replay order".into(),
            ));
        }
        Ok(())
    }
}

/// Checksummed output artifact associated with a replay state.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ReplayArtifact {
    /// Logical role such as `state`, `dashboard`, or `manifest`.
    pub role: String,
    /// Absolute output path or source path represented by this receipt.
    pub path: String,
    /// Observed byte count.
    pub size_bytes: u64,
    /// Lowercase SHA-256 checksum.
    pub sha256: String,
}

impl ReplayArtifact {
    /// Creates and validates one replay artifact receipt.
    pub fn try_new(
        role: impl Into<String>,
        path: impl Into<String>,
        size_bytes: u64,
        sha256: impl Into<String>,
    ) -> ViewerResult<Self> {
        let artifact =
            Self { role: role.into(), path: path.into(), size_bytes, sha256: sha256.into() };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validates artifact identity and checksum shape.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.role.trim().is_empty() || self.path.trim().is_empty() || self.size_bytes == 0 {
            return Err(ViewerError::InvalidState(
                "replay artifacts require a role, path, and non-zero size".into(),
            ));
        }
        validate_sha256("replay artifact", &self.sha256)
    }
}

/// Portable state emitted by the one-command deterministic replay demo.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ReplayDemoState {
    /// Serialized replay state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Checksummed input identity.
    pub source: StudioSource,
    /// Bounded topic inventory used to build the episode.
    pub topics: Vec<ReplayTopic>,
    /// Bounded episode and replay metrics.
    pub summary: ReplaySummary,
    /// Ordered replay trace samples.
    pub samples: Vec<ReplaySample>,
    /// Checksummed state/dashboard/manifest artifacts.
    pub artifacts: Vec<ReplayArtifact>,
    /// Whether deterministic replay has passed its admission checks.
    pub replay_ready: bool,
    /// Whether downstream mapping is admitted; calibration is never implicit.
    pub mapping_admitted: bool,
    /// Fail-closed reasons for mapping or replay admission.
    pub blockers: Vec<String>,
}

impl ReplayDemoState {
    /// Creates a replay state and derives replay and mapping admission.
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        topics: Vec<ReplayTopic>,
        summary: ReplaySummary,
        samples: Vec<ReplaySample>,
        artifacts: Vec<ReplayArtifact>,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let sample_count = u64::try_from(samples.len()).map_err(|_| {
            ViewerError::InvalidState("replay sample count does not fit in u64".into())
        })?;
        let replay_ready = source.identity_matches
            && summary.deterministic_order_verified
            && summary.replayed_record_count > 0
            && summary.replayed_record_count == summary.episode_record_count
            && summary.replayed_record_count == sample_count;
        let mapping_admitted = replay_ready && summary.calibration_applied;
        let state = Self {
            version: REPLAY_DEMO_STATE_VERSION,
            title: title.into(),
            source,
            topics,
            summary,
            samples,
            artifacts,
            replay_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates the replay trace and all cross-panel admission invariants.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != REPLAY_DEMO_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported replay demo state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(ViewerError::InvalidState("replay demo title must not be empty".into()));
        }
        self.source.validate()?;
        let mut topic_names = BTreeSet::new();
        for topic in &self.topics {
            topic.validate()?;
            if !topic_names.insert(&topic.name) {
                return Err(ViewerError::InvalidState(
                    "replay topic inventory contains a duplicate topic".into(),
                ));
            }
        }
        self.summary.validate()?;

        for (expected_sequence, sample) in self.samples.iter().enumerate() {
            sample.validate()?;
            if sample.sequence != u64::try_from(expected_sequence).unwrap_or(u64::MAX) {
                return Err(ViewerError::InvalidState(
                    "replay samples must use contiguous zero-based sequence numbers".into(),
                ));
            }
        }

        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "replay artifacts must have unique roles and paths".into(),
                ));
            }
        }

        let sample_count = u64::try_from(self.samples.len()).unwrap_or(u64::MAX);
        let calculated_replay_ready = self.source.identity_matches
            && self.summary.deterministic_order_verified
            && self.summary.replayed_record_count > 0
            && self.summary.replayed_record_count == self.summary.episode_record_count
            && self.summary.replayed_record_count == sample_count;
        if self.replay_ready != calculated_replay_ready {
            return Err(ViewerError::InvalidState(
                "replay_ready disagrees with source, summary, or trace admission".into(),
            ));
        }
        let calculated_mapping = self.replay_ready && self.summary.calibration_applied;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with replay and calibration admission".into(),
            ));
        }
        if self.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted replay mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked replay mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "replay blockers must not contain empty messages".into(),
            ));
        }
        Ok(())
    }
}

fn validate_sha256(label: &str, value: &str) -> ViewerResult<()> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        return Err(ViewerError::InvalidState(format!(
            "{label} SHA-256 must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
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

    fn summary(calibration_applied: bool) -> ReplaySummary {
        ReplaySummary::try_new(
            2,
            8,
            96,
            2,
            1,
            4,
            10,
            20,
            128,
            true,
            "PointCloud2 header stamp; no clock calibration applied",
            calibration_applied,
        )
        .unwrap()
    }

    fn samples() -> Vec<ReplaySample> {
        vec![
            ReplaySample::try_new(0, "/front", 10, 4, vec!["/rear".into()]).unwrap(),
            ReplaySample::try_new(1, "/rear", 14, 4, vec!["/front".into()]).unwrap(),
        ]
    }

    #[test]
    fn blocked_mapping_state_roundtrips_with_serde() {
        let state = ReplayDemoState::try_new(
            "Replay demo",
            source(true),
            vec![ReplayTopic::try_new("/front", 2, 1, 4, vec!["front_frame".into()]).unwrap()],
            summary(false),
            samples(),
            Vec::new(),
            vec!["clock calibration not applied".into()],
        )
        .unwrap();
        assert!(state.replay_ready);
        assert!(!state.mapping_admitted);
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(serde_json::from_str::<ReplayDemoState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn source_mismatch_cannot_be_admitted() {
        let state = ReplayDemoState::try_new(
            "Replay demo",
            source(false),
            vec![ReplayTopic::try_new("/front", 2, 1, 4, vec!["front_frame".into()]).unwrap()],
            summary(false),
            samples(),
            Vec::new(),
            vec!["input SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.replay_ready);
        assert!(!state.mapping_admitted);
    }

    #[test]
    fn non_contiguous_trace_is_rejected() {
        let mut trace = samples();
        trace[1].sequence = 3;
        assert!(ReplayDemoState::try_new(
            "Replay demo",
            source(true),
            vec![ReplayTopic::try_new("/front", 2, 1, 4, vec!["front_frame".into()]).unwrap()],
            summary(false),
            trace,
            Vec::new(),
            vec!["trace is invalid".into()],
        )
        .is_err());
    }

    #[test]
    fn applied_calibration_requires_verified_order() {
        assert!(ReplaySummary::try_new(1, 1, 1, 1, 0, 0, 1, 1, 1, false, "header stamp", true,)
            .is_err());
    }
}
