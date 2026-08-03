//! Portable, source-bound health state for a spatial dataset.
//!
//! This module contains no rosbag2, SQLite, filesystem, or renderer types.
//! Adapters populate it from receipts and stage snapshots so a static
//! dashboard can expose integrity, lineage, and calibration gates together.

use std::collections::BTreeSet;

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized Dataset Health state schema version.
pub const DATASET_HEALTH_STATE_VERSION: u32 = 1;

/// One integrity or readiness check shown by the health dashboard.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DatasetHealthCheck {
    /// Stable check identifier.
    pub id: String,
    /// User-facing check label.
    pub label: String,
    /// Check outcome: pass, warning, or blocked.
    pub status: String,
    /// Whether a blocked result prevents dataset health readiness.
    pub critical: bool,
    /// Observed value or receipt fact.
    pub observed: String,
    /// Expected value or operation contract.
    pub expected: String,
    /// Human-readable explanation.
    pub detail: String,
}

impl DatasetHealthCheck {
    /// Creates and validates one health check.
    pub fn try_new(
        id: impl Into<String>,
        label: impl Into<String>,
        status: impl Into<String>,
        critical: bool,
        observed: impl Into<String>,
        expected: impl Into<String>,
        detail: impl Into<String>,
    ) -> ViewerResult<Self> {
        let check = Self {
            id: id.into(),
            label: label.into(),
            status: status.into(),
            critical,
            observed: observed.into(),
            expected: expected.into(),
            detail: detail.into(),
        };
        check.validate()?;
        Ok(check)
    }

    /// Validates identity, outcome vocabulary, and displayed values.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty()
            || self.label.trim().is_empty()
            || self.observed.trim().is_empty()
            || self.expected.trim().is_empty()
            || self.detail.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health checks require non-empty identity and explanation fields".into(),
            ));
        }
        validate_status(&self.status)
    }
}

/// Health counters for one canonical sensor topic.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DatasetHealthTopic {
    /// ROS topic name.
    pub name: String,
    /// Logical sensor role such as front or rear.
    pub role: String,
    /// Total messages reported by the source inventory.
    pub message_count: u64,
    /// Records retained by the bounded E2E episode.
    pub retained_record_count: u64,
    /// Points retained by the bounded E2E episode.
    pub retained_point_count: u64,
    /// Frame IDs observed in the retained records.
    pub frame_ids: Vec<String>,
    /// Topic outcome: pass, warning, or blocked.
    pub status: String,
}

impl DatasetHealthTopic {
    /// Creates and validates one topic health row.
    pub fn try_new(
        name: impl Into<String>,
        role: impl Into<String>,
        message_count: u64,
        retained_record_count: u64,
        retained_point_count: u64,
        frame_ids: Vec<String>,
        status: impl Into<String>,
    ) -> ViewerResult<Self> {
        let topic = Self {
            name: name.into(),
            role: role.into(),
            message_count,
            retained_record_count,
            retained_point_count,
            frame_ids,
            status: status.into(),
        };
        topic.validate()?;
        Ok(topic)
    }

    /// Validates topic counters, frame IDs, and outcome consistency.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.name.trim().is_empty() || self.role.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "Dataset Health topics require a name and logical role".into(),
            ));
        }
        validate_status(&self.status)?;
        if self.retained_record_count > 0 && self.retained_point_count == 0 {
            return Err(ViewerError::InvalidState(
                "retained Dataset Health records require retained points".into(),
            ));
        }
        let mut frames = BTreeSet::new();
        for frame_id in &self.frame_ids {
            if frame_id.trim().is_empty() || !frames.insert(frame_id) {
                return Err(ViewerError::InvalidState(
                    "Dataset Health topic frame IDs must be non-empty and unique".into(),
                ));
            }
        }
        if self.status == "pass"
            && (self.message_count == 0
                || self.retained_record_count == 0
                || self.retained_point_count == 0
                || self.frame_ids.is_empty())
        {
            return Err(ViewerError::InvalidState(
                "a passing Dataset Health topic must contain source and frame evidence".into(),
            ));
        }
        Ok(())
    }
}

/// Health and gate counters aggregated across checks and artifacts.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DatasetHealthSummary {
    /// Canonical source byte count.
    pub source_bytes: u64,
    /// Sum of source messages across health topics.
    pub source_message_count: u64,
    /// Sum of retained records across health topics.
    pub retained_record_count: u64,
    /// Sum of retained points across health topics.
    pub retained_point_count: u64,
    /// Number of checksummed artifacts represented by the dashboard.
    pub artifact_count: u64,
    /// Sum of checksummed artifact bytes.
    pub artifact_bytes: u64,
    /// Number of canonical topics represented.
    pub topic_count: u64,
    /// Number of stage snapshots represented.
    pub stage_count: u64,
    /// Total number of checks.
    pub check_count: u64,
    /// Number of passing checks.
    pub pass_count: u64,
    /// Number of warning checks.
    pub warning_count: u64,
    /// Number of blocked checks.
    pub blocked_count: u64,
    /// Number of blocked checks marked critical.
    pub critical_block_count: u64,
    /// Whether canonical source identity matched.
    pub source_identity_match: bool,
    /// Whether the requested frame matched canonical stage evidence.
    pub frame_identity_match: bool,
    /// Whether source-bound clock and frame calibration is registered.
    pub calibration_ready: bool,
}

impl DatasetHealthSummary {
    /// Creates and validates aggregate health counters.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_bytes: u64,
        source_message_count: u64,
        retained_record_count: u64,
        retained_point_count: u64,
        artifact_count: u64,
        artifact_bytes: u64,
        topic_count: u64,
        stage_count: u64,
        check_count: u64,
        pass_count: u64,
        warning_count: u64,
        blocked_count: u64,
        critical_block_count: u64,
        source_identity_match: bool,
        frame_identity_match: bool,
        calibration_ready: bool,
    ) -> ViewerResult<Self> {
        let summary = Self {
            source_bytes,
            source_message_count,
            retained_record_count,
            retained_point_count,
            artifact_count,
            artifact_bytes,
            topic_count,
            stage_count,
            check_count,
            pass_count,
            warning_count,
            blocked_count,
            critical_block_count,
            source_identity_match,
            frame_identity_match,
            calibration_ready,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates counter ordering and source/calibration relationships.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_bytes == 0 {
            return Err(ViewerError::InvalidState(
                "Dataset Health requires a non-empty canonical source".into(),
            ));
        }
        if self.retained_record_count > 0 && self.retained_point_count == 0 {
            return Err(ViewerError::InvalidState(
                "Dataset Health retained records require retained points".into(),
            ));
        }
        if self
            .pass_count
            .checked_add(self.warning_count)
            .and_then(|count| count.checked_add(self.blocked_count))
            != Some(self.check_count)
            || self.critical_block_count > self.blocked_count
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health check counters are inconsistent".into(),
            ));
        }
        if self.calibration_ready && (!self.source_identity_match || !self.frame_identity_match) {
            return Err(ViewerError::InvalidState(
                "calibration cannot be ready for a source or frame mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// Health status for one previously generated SpatialRust stage.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DatasetHealthStage {
    /// Stable stage identifier such as 145E.
    pub id: String,
    /// User-facing stage label.
    pub label: String,
    /// Stage outcome: pass, warning, or blocked.
    pub status: String,
    /// Whether the stage-specific inspection gate is ready.
    pub ready: bool,
    /// Whether the stage admitted calibrated mapping.
    pub mapping_admitted: bool,
    /// Whether the stage source matched the canonical source.
    pub source_identity_match: bool,
    /// Whether the stage explicitly checked the requested frame.
    pub frame_identity_match: Option<bool>,
    /// Number of stage files represented in the health manifest.
    pub artifact_count: u64,
    /// Short stage-specific evidence detail.
    pub detail: String,
}

impl DatasetHealthStage {
    /// Creates and validates one stage health row.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        id: impl Into<String>,
        label: impl Into<String>,
        status: impl Into<String>,
        ready: bool,
        mapping_admitted: bool,
        source_identity_match: bool,
        frame_identity_match: Option<bool>,
        artifact_count: u64,
        detail: impl Into<String>,
    ) -> ViewerResult<Self> {
        let stage = Self {
            id: id.into(),
            label: label.into(),
            status: status.into(),
            ready,
            mapping_admitted,
            source_identity_match,
            frame_identity_match,
            artifact_count,
            detail: detail.into(),
        };
        stage.validate()?;
        Ok(stage)
    }

    /// Validates stage gate consistency and source-bound status.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty()
            || self.label.trim().is_empty()
            || self.detail.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health stages require identity and evidence detail".into(),
            ));
        }
        validate_status(&self.status)?;
        if !self.source_identity_match && self.status != "blocked" {
            return Err(ViewerError::InvalidState(
                "a source-mismatched Dataset Health stage must be blocked".into(),
            ));
        }
        if self.frame_identity_match == Some(false) && self.status != "blocked" {
            return Err(ViewerError::InvalidState(
                "a frame-mismatched Dataset Health stage must be blocked".into(),
            ));
        }
        if self.mapping_admitted && (!self.ready || self.status != "pass") {
            return Err(ViewerError::InvalidState(
                "an admitted Dataset Health stage must be ready and passing".into(),
            ));
        }
        Ok(())
    }
}

/// Portable source-bound Dataset Health dashboard state.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DatasetHealthState {
    /// Serialized Dataset Health schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Canonical input identity.
    pub source: StudioSource,
    /// Canonical observed frame selected for health aggregation.
    pub frame_id: String,
    /// Frame required by the operation contract.
    pub expected_frame_id: String,
    /// Human-readable timestamp basis.
    pub time_basis: String,
    /// Canonical sensor topic rows.
    pub topics: Vec<DatasetHealthTopic>,
    /// Previously generated stage snapshots.
    pub stages: Vec<DatasetHealthStage>,
    /// Checksummed source and stage artifacts.
    pub artifacts: Vec<ReplayArtifact>,
    /// Integrity, lineage, and calibration checks.
    pub checks: Vec<DatasetHealthCheck>,
    /// Aggregate health metrics.
    pub summary: DatasetHealthSummary,
    /// Whether source/data health is ready for inspection.
    pub dataset_ready: bool,
    /// Whether calibrated downstream mapping is admitted.
    pub mapping_admitted: bool,
    /// Fail-closed reasons and calibration notices.
    pub blockers: Vec<String>,
}

impl DatasetHealthState {
    /// Creates a health state and derives dataset and mapping gates.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        frame_id: impl Into<String>,
        expected_frame_id: impl Into<String>,
        time_basis: impl Into<String>,
        topics: Vec<DatasetHealthTopic>,
        stages: Vec<DatasetHealthStage>,
        artifacts: Vec<ReplayArtifact>,
        checks: Vec<DatasetHealthCheck>,
        summary: DatasetHealthSummary,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let critical_block_count =
            checks.iter().filter(|check| check.status == "blocked" && check.critical).count();
        let dataset_ready = source.identity_matches
            && summary.source_identity_match
            && summary.frame_identity_match
            && critical_block_count == 0
            && !topics.is_empty()
            && !stages.is_empty();
        let mapping_admitted = dataset_ready && summary.calibration_ready;
        let state = Self {
            version: DATASET_HEALTH_STATE_VERSION,
            title: title.into(),
            source,
            frame_id: frame_id.into(),
            expected_frame_id: expected_frame_id.into(),
            time_basis: time_basis.into(),
            topics,
            stages,
            artifacts,
            checks,
            summary,
            dataset_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates source identity, receipts, counters, and both gates.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != DATASET_HEALTH_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported Dataset Health state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty()
            || self.frame_id.trim().is_empty()
            || self.expected_frame_id.trim().is_empty()
            || self.time_basis.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health title, frames, and time basis must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.summary.validate()?;
        if self.summary.source_identity_match != self.source.identity_matches
            || self.summary.frame_identity_match != (self.frame_id == self.expected_frame_id)
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health summary identity disagrees with source/frame fields".into(),
            ));
        }

        let mut topic_names = BTreeSet::new();
        let mut topic_message_count = 0_u64;
        let mut topic_record_count = 0_u64;
        let mut topic_point_count = 0_u64;
        for topic in &self.topics {
            topic.validate()?;
            if !topic_names.insert(&topic.name) {
                return Err(ViewerError::InvalidState(
                    "Dataset Health topics must have unique names".into(),
                ));
            }
            topic_message_count = topic_message_count
                .checked_add(topic.message_count)
                .ok_or_else(|| ViewerError::InvalidState("topic message count overflow".into()))?;
            topic_record_count = topic_record_count
                .checked_add(topic.retained_record_count)
                .ok_or_else(|| ViewerError::InvalidState("topic record count overflow".into()))?;
            topic_point_count = topic_point_count
                .checked_add(topic.retained_point_count)
                .ok_or_else(|| ViewerError::InvalidState("topic point count overflow".into()))?;
        }
        if topic_message_count != self.summary.source_message_count
            || topic_record_count != self.summary.retained_record_count
            || topic_point_count != self.summary.retained_point_count
            || self.summary.topic_count != u64::try_from(self.topics.len()).unwrap_or(u64::MAX)
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health topic counters disagree with the summary".into(),
            ));
        }

        let mut stage_ids = BTreeSet::new();
        let mut stage_artifact_count = 0_u64;
        for stage in &self.stages {
            stage.validate()?;
            if !stage_ids.insert(&stage.id) {
                return Err(ViewerError::InvalidState(
                    "Dataset Health stages must have unique IDs".into(),
                ));
            }
            stage_artifact_count = stage_artifact_count
                .checked_add(stage.artifact_count)
                .ok_or_else(|| ViewerError::InvalidState("stage artifact count overflow".into()))?;
        }
        if self.summary.stage_count != u64::try_from(self.stages.len()).unwrap_or(u64::MAX) {
            return Err(ViewerError::InvalidState(
                "Dataset Health stage count disagrees with the stage list".into(),
            ));
        }
        if stage_artifact_count == 0 && !self.stages.is_empty() {
            return Err(ViewerError::InvalidState(
                "Dataset Health stages must represent at least one artifact".into(),
            ));
        }

        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        let mut artifact_bytes = 0_u64;
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "Dataset Health artifacts require unique roles and paths".into(),
                ));
            }
            artifact_bytes = artifact_bytes
                .checked_add(artifact.size_bytes)
                .ok_or_else(|| ViewerError::InvalidState("artifact byte count overflow".into()))?;
        }
        if self.summary.artifact_count != u64::try_from(self.artifacts.len()).unwrap_or(u64::MAX)
            || self.summary.artifact_bytes != artifact_bytes
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health artifact counters disagree with receipts".into(),
            ));
        }

        let mut check_ids = BTreeSet::new();
        let mut pass_count = 0_u64;
        let mut warning_count = 0_u64;
        let mut blocked_count = 0_u64;
        let mut critical_block_count = 0_u64;
        for check in &self.checks {
            check.validate()?;
            if !check_ids.insert(&check.id) {
                return Err(ViewerError::InvalidState(
                    "Dataset Health checks must have unique IDs".into(),
                ));
            }
            match check.status.as_str() {
                "pass" => pass_count = pass_count.saturating_add(1),
                "warning" => warning_count = warning_count.saturating_add(1),
                "blocked" => {
                    blocked_count = blocked_count.saturating_add(1);
                    if check.critical {
                        critical_block_count = critical_block_count.saturating_add(1);
                    }
                }
                _ => unreachable!("check status validated above"),
            }
        }
        if self.summary.check_count != u64::try_from(self.checks.len()).unwrap_or(u64::MAX)
            || self.summary.pass_count != pass_count
            || self.summary.warning_count != warning_count
            || self.summary.blocked_count != blocked_count
            || self.summary.critical_block_count != critical_block_count
        {
            return Err(ViewerError::InvalidState(
                "Dataset Health check counters disagree with check rows".into(),
            ));
        }

        let calculated_dataset_ready = self.source.identity_matches
            && self.summary.source_identity_match
            && self.summary.frame_identity_match
            && critical_block_count == 0
            && !self.topics.is_empty()
            && !self.stages.is_empty();
        if self.dataset_ready != calculated_dataset_ready {
            return Err(ViewerError::InvalidState(
                "dataset_ready disagrees with source, checks, topics, or stages".into(),
            ));
        }
        let calculated_mapping = self.dataset_ready && self.summary.calibration_ready;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with health and calibration gates".into(),
            ));
        }
        if self.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted Dataset Health mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked Dataset Health mapping must expose blockers".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "Dataset Health blockers must not contain empty messages".into(),
            ));
        }
        Ok(())
    }
}

fn validate_status(status: &str) -> ViewerResult<()> {
    if matches!(status, "pass" | "warning" | "blocked") {
        Ok(())
    } else {
        Err(ViewerError::InvalidState(
            "Dataset Health status must be pass, warning, or blocked".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const OTHER_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn source(observed: &str, matches: bool) -> StudioSource {
        StudioSource::try_new("canonical bag", "/media/input.db3", SHA, observed, matches).unwrap()
    }

    fn topic() -> DatasetHealthTopic {
        DatasetHealthTopic::try_new("/front", "front", 2, 2, 8, vec!["lidar_front".into()], "pass")
            .unwrap()
    }

    fn stage() -> DatasetHealthStage {
        DatasetHealthStage::try_new(
            "145E",
            "Semantic Overlay",
            "pass",
            true,
            false,
            true,
            Some(true),
            3,
            "overlay state and receipts validated",
        )
        .unwrap()
    }

    fn check(id: &str, status: &str, critical: bool) -> DatasetHealthCheck {
        DatasetHealthCheck::try_new(
            id,
            id,
            status,
            critical,
            "observed",
            "expected",
            "health evidence",
        )
        .unwrap()
    }

    fn summary(source_matches: bool, frame_matches: bool) -> DatasetHealthSummary {
        DatasetHealthSummary::try_new(
            128,
            if source_matches { 2 } else { 0 },
            if source_matches { 2 } else { 0 },
            if source_matches { 8 } else { 0 },
            if source_matches { 1 } else { 0 },
            if source_matches { 128 } else { 0 },
            if source_matches { 1 } else { 0 },
            if source_matches { 1 } else { 0 },
            if source_matches { 2 } else { 1 },
            if source_matches { 1 } else { 0 },
            0,
            1,
            if source_matches { 0 } else { 1 },
            source_matches,
            frame_matches,
            false,
        )
        .unwrap()
    }

    #[test]
    fn healthy_dataset_can_be_ready_while_mapping_is_blocked() {
        let state = DatasetHealthState::try_new(
            "Dataset Health",
            source(SHA, true),
            "lidar_front",
            "lidar_front",
            "header stamp",
            vec![topic()],
            vec![stage()],
            vec![ReplayArtifact::try_new("source", "/media/input.db3", 128, SHA).unwrap()],
            vec![check("source", "pass", true), check("calibration", "blocked", false)],
            summary(true, true),
            vec!["clock and TF calibration are not registered".into()],
        )
        .unwrap();
        assert!(state.dataset_ready);
        assert!(!state.mapping_admitted);
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(serde_json::from_str::<DatasetHealthState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn source_mismatch_is_critical_and_fail_closed() {
        let state = DatasetHealthState::try_new(
            "Dataset Health",
            source(OTHER_SHA, false),
            "lidar_front",
            "lidar_front",
            "header stamp",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![check("source", "blocked", true)],
            summary(false, true),
            vec!["canonical source SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.dataset_ready);
        assert!(!state.mapping_admitted);
    }
}
