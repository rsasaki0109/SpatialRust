//! Portable state for the Spatial Studio multi-panel surface.
//!
//! This module intentionally contains no ROS, SQLite, renderer, or GPU types.
//! Adapters may populate the state from receipts, while native and Web
//! frontends can render the same source-bound admission decision.

use std::collections::BTreeSet;

use spatialrust_viz::LayerId;

use crate::{ViewerError, ViewerResult, ViewerState};

/// Current serialized Spatial Studio state schema version.
pub const STUDIO_STATE_VERSION: u32 = 1;

/// Checksummed identity of the source represented by a Studio session.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioSource {
    /// User-facing source label.
    pub label: String,
    /// Source path or URI shown in the Studio header.
    pub path: String,
    /// SHA-256 expected by the operation contract.
    pub expected_sha256: String,
    /// SHA-256 observed while opening the source.
    pub observed_sha256: String,
    /// Whether the expected and observed identities are exactly equal.
    pub identity_matches: bool,
}

impl StudioSource {
    /// Creates a source identity and rejects inconsistent checksum claims.
    pub fn try_new(
        label: impl Into<String>,
        path: impl Into<String>,
        expected_sha256: impl Into<String>,
        observed_sha256: impl Into<String>,
        identity_matches: bool,
    ) -> ViewerResult<Self> {
        let source = Self {
            label: label.into(),
            path: path.into(),
            expected_sha256: expected_sha256.into(),
            observed_sha256: observed_sha256.into(),
            identity_matches,
        };
        source.validate()?;
        Ok(source)
    }

    /// Validates source labels, paths, and the identity equality invariant.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.label.trim().is_empty() || self.path.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "Studio source label and path must not be empty".into(),
            ));
        }
        validate_sha256("expected source", &self.expected_sha256)?;
        validate_sha256("observed source", &self.observed_sha256)?;
        let calculated_match = self.expected_sha256 == self.observed_sha256;
        if self.identity_matches != calculated_match {
            return Err(ViewerError::InvalidState(
                "Studio source identity_matches disagrees with the checksums".into(),
            ));
        }
        Ok(())
    }
}

/// Metadata for one point-cloud or derived layer in the Studio panel.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioLayer {
    /// Stable layer identifier shared with renderer controls.
    pub id: String,
    /// User-facing display label.
    pub label: String,
    /// Logical layer role, such as `point_cloud` or `mesh`.
    pub role: String,
    /// Source topic or artifact identifier.
    pub topic: String,
    /// Observed frame ID, when the source receipt supplied one.
    pub frame_id: Option<String>,
    /// Number of messages represented by the layer.
    pub message_count: u64,
    /// Number of points represented by the layer receipt.
    pub point_count: u64,
    /// Whether geometry is available to a renderer.
    pub renderable: bool,
    /// Current panel visibility.
    pub visible: bool,
}

impl StudioLayer {
    /// Creates and validates one Studio layer descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        id: impl Into<String>,
        label: impl Into<String>,
        role: impl Into<String>,
        topic: impl Into<String>,
        frame_id: Option<String>,
        message_count: u64,
        point_count: u64,
        renderable: bool,
        visible: bool,
    ) -> ViewerResult<Self> {
        let layer = Self {
            id: id.into(),
            label: label.into(),
            role: role.into(),
            topic: topic.into(),
            frame_id,
            message_count,
            point_count,
            renderable,
            visible,
        };
        layer.validate()?;
        Ok(layer)
    }

    /// Validates layer identity and receipt-derived counts.
    pub fn validate(&self) -> ViewerResult<()> {
        LayerId::try_new(self.id.clone())?;
        if self.label.trim().is_empty()
            || self.role.trim().is_empty()
            || self.topic.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "Studio layer label, role, and topic must not be empty".into(),
            ));
        }
        if self.renderable && self.point_count == 0 {
            return Err(ViewerError::InvalidState(format!(
                "renderable Studio layer `{}` has no points",
                self.id
            )));
        }
        if let Some(frame_id) = &self.frame_id {
            if frame_id.trim().is_empty() {
                return Err(ViewerError::InvalidState(format!(
                    "Studio layer `{}` has an empty frame ID",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

/// Bounded timeline information shown by the Studio scrubber.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioTimeline {
    /// First timestamp in the source, when metadata is available.
    pub start_nanos: Option<u64>,
    /// Last timestamp bound in the source, when metadata is available.
    pub end_nanos: Option<u64>,
    /// Current scrubber position, when metadata is available.
    pub cursor_nanos: Option<u64>,
    /// Number of source samples represented by the session.
    pub sample_count: u64,
    /// Human-readable timestamp basis.
    pub time_basis: String,
    /// Whether an external clock calibration was applied.
    pub clock_calibrated: bool,
}

impl StudioTimeline {
    /// Creates a timeline and enforces complete timestamp bounds.
    pub fn try_new(
        start_nanos: Option<u64>,
        end_nanos: Option<u64>,
        cursor_nanos: Option<u64>,
        sample_count: u64,
        time_basis: impl Into<String>,
        clock_calibrated: bool,
    ) -> ViewerResult<Self> {
        let timeline = Self {
            start_nanos,
            end_nanos,
            cursor_nanos,
            sample_count,
            time_basis: time_basis.into(),
            clock_calibrated,
        };
        timeline.validate()?;
        Ok(timeline)
    }

    /// Validates timeline ordering and the unavailable-metadata representation.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.time_basis.trim().is_empty() {
            return Err(ViewerError::InvalidState("Studio timeline time basis is empty".into()));
        }
        match (self.start_nanos, self.end_nanos, self.cursor_nanos) {
            (Some(start), Some(end), Some(cursor)) => {
                if end < start || cursor < start || cursor > end || self.sample_count == 0 {
                    return Err(ViewerError::InvalidState(
                        "Studio timeline bounds or sample count are invalid".into(),
                    ));
                }
            }
            (None, None, None) => {
                if self.sample_count != 0 || self.clock_calibrated {
                    return Err(ViewerError::InvalidState(
                        "unavailable Studio timeline cannot contain samples or calibrated time"
                            .into(),
                    ));
                }
            }
            _ => {
                return Err(ViewerError::InvalidState(
                    "Studio timeline timestamps must be all present or all absent".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Calibration admission state displayed by the Studio gate panel.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioCalibration {
    /// Whether both required calibration artifacts are registered and ready.
    pub registration_ready: bool,
    /// Clock-artifact status, for example `registered` or `not_registered`.
    pub clock_status: String,
    /// Frame-artifact status, for example `registered` or `not_registered`.
    pub frame_status: String,
    /// Whether the calibration evidence is bound to the displayed source.
    pub source_bound: bool,
    /// Fail-closed reasons preventing calibration admission.
    pub blockers: Vec<String>,
}

impl StudioCalibration {
    /// Creates and validates a calibration gate state.
    pub fn try_new(
        registration_ready: bool,
        clock_status: impl Into<String>,
        frame_status: impl Into<String>,
        source_bound: bool,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let calibration = Self {
            registration_ready,
            clock_status: clock_status.into(),
            frame_status: frame_status.into(),
            source_bound,
            blockers,
        };
        calibration.validate()?;
        Ok(calibration)
    }

    /// Validates that ready state has no unresolved blockers.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.clock_status.trim().is_empty() || self.frame_status.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "Studio calibration statuses must not be empty".into(),
            ));
        }
        if self.registration_ready {
            if !self.source_bound || !self.blockers.is_empty() {
                return Err(ViewerError::InvalidState(
                    "ready Studio calibration must be source-bound and blocker-free".into(),
                ));
            }
            if self.clock_status != "registered" || self.frame_status != "registered" {
                return Err(ViewerError::InvalidState(
                    "ready Studio calibration requires registered clock and frame artifacts".into(),
                ));
            }
        } else if self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked Studio calibration must expose at least one blocker".into(),
            ));
        }
        Ok(())
    }
}

/// Source-bound frame inventory and composition status.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioFrameGraph {
    /// Frame IDs accepted from the source-bound inventory.
    pub observed_frames: Vec<String>,
    /// Number of accepted transform edges.
    pub edge_count: u64,
    /// Root selected by a composition operation, when any.
    pub root_frame: Option<String>,
    /// Whether transforms have been composed for downstream use.
    pub composed: bool,
    /// Whether the inventory belongs to the Studio source identity.
    pub source_bound: bool,
}

impl StudioFrameGraph {
    /// Creates and validates a non-composing or source-bound composed graph.
    pub fn try_new(
        observed_frames: Vec<String>,
        edge_count: u64,
        root_frame: Option<String>,
        composed: bool,
        source_bound: bool,
    ) -> ViewerResult<Self> {
        let graph = Self { observed_frames, edge_count, root_frame, composed, source_bound };
        graph.validate()?;
        Ok(graph)
    }

    /// Validates frame uniqueness and rejects unbound transform evidence.
    pub fn validate(&self) -> ViewerResult<()> {
        let mut frames = BTreeSet::new();
        for frame in &self.observed_frames {
            if frame.trim().is_empty() || !frames.insert(frame) {
                return Err(ViewerError::InvalidState(
                    "Studio frame inventory contains an empty or duplicate frame".into(),
                ));
            }
        }
        if !self.source_bound && (!self.observed_frames.is_empty() || self.edge_count != 0) {
            return Err(ViewerError::InvalidState(
                "unbound Studio frame evidence cannot be accepted into the graph".into(),
            ));
        }
        if self.composed
            && (!self.source_bound || self.edge_count == 0 || self.root_frame.is_none())
        {
            return Err(ViewerError::InvalidState(
                "composed Studio frame graph requires source-bound edges and a root".into(),
            ));
        }
        if let Some(root) = &self.root_frame {
            if root.trim().is_empty() || !frames.contains(root) {
                return Err(ViewerError::InvalidState(
                    "Studio frame graph root must be one of the observed frames".into(),
                ));
            }
        }
        Ok(())
    }
}

/// One measured pipeline stage in the Studio metrics panel.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioStageMetric {
    /// Stable stage name.
    pub name: String,
    /// Observed wall-clock duration in nanoseconds.
    pub wall_ns: u64,
}

/// Explicit memory and transfer counters for a Studio pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioPerformance {
    /// Sum or receipt-reported wall-clock duration in nanoseconds.
    pub observed_pipeline_wall_ns: u64,
    /// Ordered per-stage timings.
    pub stages: Vec<StudioStageMetric>,
    /// Largest observed source allocation in bytes.
    pub peak_source_bytes: u64,
    /// Host-to-device bytes explicitly transferred.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes explicitly transferred.
    pub device_to_host_bytes: u64,
    /// Copies that were not attributed to an explicit transfer boundary.
    pub hidden_device_copies: u64,
}

impl StudioPerformance {
    /// Creates and validates explicit stage metrics.
    pub fn try_new(
        observed_pipeline_wall_ns: u64,
        stages: Vec<StudioStageMetric>,
        peak_source_bytes: u64,
        host_to_device_bytes: u64,
        device_to_host_bytes: u64,
        hidden_device_copies: u64,
    ) -> ViewerResult<Self> {
        let performance = Self {
            observed_pipeline_wall_ns,
            stages,
            peak_source_bytes,
            host_to_device_bytes,
            device_to_host_bytes,
            hidden_device_copies,
        };
        performance.validate()?;
        Ok(performance)
    }

    /// Validates unique, named performance stages.
    pub fn validate(&self) -> ViewerResult<()> {
        let mut names = BTreeSet::new();
        for stage in &self.stages {
            if stage.name.trim().is_empty() || !names.insert(&stage.name) {
                return Err(ViewerError::InvalidState(
                    "Studio performance stages must have unique names".into(),
                ));
            }
        }
        Ok(())
    }
}

/// One portable state snapshot for the Spatial Studio surface.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct StudioState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing Studio title.
    pub title: String,
    /// Camera, viewport, and renderer-independent controls.
    pub viewer: ViewerState,
    /// Checksummed input identity.
    pub source: StudioSource,
    /// Point-cloud and derived layer metadata.
    pub layers: Vec<StudioLayer>,
    /// Timeline and timestamp basis.
    pub timeline: StudioTimeline,
    /// Calibration admission panel state.
    pub calibration: StudioCalibration,
    /// Source-bound TF inventory/composition state.
    pub frame_graph: StudioFrameGraph,
    /// Pipeline performance and transfer metrics.
    pub performance: StudioPerformance,
    /// Aggregated fail-closed reasons shown at the top level.
    pub blockers: Vec<String>,
    /// Whether downstream mapping is admitted by every required gate.
    pub mapping_admitted: bool,
}

impl StudioState {
    /// Creates a Studio state and derives its mapping admission decision.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        viewer: ViewerState,
        source: StudioSource,
        layers: Vec<StudioLayer>,
        timeline: StudioTimeline,
        calibration: StudioCalibration,
        frame_graph: StudioFrameGraph,
        performance: StudioPerformance,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let mapping_admitted = source.identity_matches
            && calibration.registration_ready
            && calibration.source_bound
            && frame_graph.source_bound
            && frame_graph.composed
            && performance.hidden_device_copies == 0;
        let state = Self {
            version: STUDIO_STATE_VERSION,
            title: title.into(),
            viewer,
            source,
            layers,
            timeline,
            calibration,
            frame_graph,
            performance,
            blockers,
            mapping_admitted,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates every panel and the cross-panel fail-closed admission rules.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != STUDIO_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported Spatial Studio state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(ViewerError::InvalidState("Studio title must not be empty".into()));
        }
        self.viewer.validate()?;
        self.source.validate()?;
        let mut ids = BTreeSet::new();
        for layer in &self.layers {
            layer.validate()?;
            if !ids.insert(&layer.id) {
                return Err(ViewerError::InvalidState(format!(
                    "duplicate Studio layer `{}`",
                    layer.id
                )));
            }
        }
        self.timeline.validate()?;
        self.calibration.validate()?;
        self.frame_graph.validate()?;
        self.performance.validate()?;

        let calculated_admission = self.source.identity_matches
            && self.calibration.registration_ready
            && self.calibration.source_bound
            && self.frame_graph.source_bound
            && self.frame_graph.composed
            && self.performance.hidden_device_copies == 0;
        if self.mapping_admitted != calculated_admission {
            return Err(ViewerError::InvalidState(
                "Studio mapping_admitted disagrees with its source/calibration/frame/performance gates"
                    .into(),
            ));
        }
        if self.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted Studio mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked Studio mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "Studio blockers must not contain empty messages".into(),
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
    use spatialrust_math::Vec3;
    use spatialrust_viz::{Camera, Projection};

    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn viewer() -> ViewerState {
        ViewerState::try_new(
            Camera::try_new(
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
            )
            .unwrap(),
            crate::ViewportSize::try_new(1280, 720).unwrap(),
        )
        .unwrap()
    }

    fn blocked_state() -> StudioState {
        StudioState::try_new(
            "Studio test",
            viewer(),
            StudioSource::try_new("bag", "/tmp/bag.db3", SHA, SHA, true).unwrap(),
            vec![StudioLayer::try_new(
                "front",
                "Front lidar",
                "point_cloud",
                "/lidar_front/points_raw",
                Some("lidar_front".into()),
                2,
                4,
                true,
                true,
            )
            .unwrap()],
            StudioTimeline::try_new(Some(10), Some(20), Some(10), 2, "header stamp", false)
                .unwrap(),
            StudioCalibration::try_new(
                false,
                "not_registered",
                "not_registered",
                false,
                vec!["calibration is missing".into()],
            )
            .unwrap(),
            StudioFrameGraph::try_new(Vec::new(), 0, None, false, false).unwrap(),
            StudioPerformance::try_new(10, Vec::new(), 20, 0, 0, 0).unwrap(),
            vec!["calibration is missing".into()],
        )
        .unwrap()
    }

    #[test]
    fn blocked_state_roundtrips_when_serde_is_enabled() {
        let state = blocked_state();
        assert!(!state.mapping_admitted);
        assert!(state.validate().is_ok());
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            let decoded: StudioState = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, state);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn studio_json_rejects_unknown_fields() {
        let state = blocked_state();
        let mut value: serde_json::Value = serde_json::to_value(state).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<StudioState>(value).is_err());
    }

    #[test]
    fn source_mismatch_cannot_be_hidden_by_mapping_admission() {
        let mut state = blocked_state();
        state.source.observed_sha256 =
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into();
        state.source.identity_matches = false;
        state.mapping_admitted = true;
        assert!(state.validate().is_err());
    }

    #[test]
    fn partial_timeline_and_unbound_frames_fail_closed() {
        assert!(StudioTimeline::try_new(Some(1), None, Some(1), 1, "stamp", false).is_err());
        assert!(StudioFrameGraph::try_new(vec!["map".into()], 1, None, false, false).is_err());
    }
}
