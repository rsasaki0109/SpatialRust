//! Source-bound clock and frame calibration evidence.
//!
//! This module validates a registration receipt without pretending to solve or
//! apply calibration.  A clock document must carry explicit quality values and
//! a frame document must contain a source-bound, acyclic path from one root to
//! both required lidar frames.  Downstream mapping still needs an application
//! receipt before it can be admitted.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::{
    CalibrationArtifact, ClockCalibration, FrameTransform, StudioSource, ViewerError, ViewerResult,
};

/// Current serialized calibration-evidence state schema version.
pub const CALIBRATION_EVIDENCE_STATE_VERSION: u32 = 1;

/// Explicit clock calibration evidence carried by a registration document.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct CalibrationEvidenceClock {
    /// Clock domain observed in the source records.
    pub source_domain: String,
    /// Clock domain to which the model would map timestamps.
    pub target_domain: String,
    /// Human-readable calibration method or provenance label.
    pub method: String,
    /// Numeric diagnostics and source-binding status.
    pub calibration: ClockCalibration,
}

impl CalibrationEvidenceClock {
    /// Creates and validates explicit clock evidence.
    pub fn try_new(
        source_domain: impl Into<String>,
        target_domain: impl Into<String>,
        method: impl Into<String>,
        calibration: ClockCalibration,
    ) -> ViewerResult<Self> {
        let clock = Self {
            source_domain: source_domain.into(),
            target_domain: target_domain.into(),
            method: method.into(),
            calibration,
        };
        clock.validate()?;
        Ok(clock)
    }

    /// Validates clock provenance, quality values, and registration ordering.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_domain.trim().is_empty()
            || self.target_domain.trim().is_empty()
            || self.method.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "calibration clock evidence requires domains and a method".into(),
            ));
        }
        self.calibration.validate()?;
        if self.calibration.status == "registered" {
            if self.source_domain == self.target_domain {
                return Err(ViewerError::InvalidState(
                    "registered clock evidence requires distinct source and target domains".into(),
                ));
            }
            if !self.calibration.source_bound || self.calibration.sample_count == 0 {
                return Err(ViewerError::InvalidState(
                    "registered clock evidence requires source binding and samples".into(),
                ));
            }
            for (label, value) in [
                ("p95 absolute offset", self.calibration.p95_abs_offset_nanos),
                ("clock uncertainty", self.calibration.uncertainty_nanos),
            ] {
                if value.map_or(true, |value| value < 0.0) {
                    return Err(ViewerError::InvalidState(format!(
                        "registered clock evidence requires non-negative {label}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Returns whether this clock document is ready for registration.
    #[must_use]
    pub fn registration_ready(&self) -> bool {
        self.calibration.status == "registered"
            && self.calibration.source_bound
            && self.calibration.sample_count > 0
            && self.calibration.p95_abs_offset_nanos.is_some_and(|value| value >= 0.0)
            && self.calibration.uncertainty_nanos.is_some_and(|value| value >= 0.0)
            && self.source_domain != self.target_domain
    }
}

/// Explicit root-to-sensor frame evidence carried by a registration document.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct CalibrationEvidenceFrame {
    /// Human-readable calibration method or provenance label.
    pub method: String,
    /// Root frame from which the sensor paths are evaluated.
    pub root_frame: String,
    /// Required sensor role to frame mapping, including `front` and `rear`.
    pub required_frames: BTreeMap<String, String>,
    /// All frame IDs present in the source-bound graph.
    pub frames: Vec<String>,
    /// Source-bound rigid edges in parent-to-child direction.
    pub edges: Vec<FrameTransform>,
    /// Whether the graph satisfies the root and required-sensor path checks.
    pub graph_ready: bool,
}

impl CalibrationEvidenceFrame {
    /// Creates a frame evidence document and derives its graph readiness.
    pub fn try_new(
        method: impl Into<String>,
        root_frame: impl Into<String>,
        required_frames: BTreeMap<String, String>,
        frames: Vec<String>,
        edges: Vec<FrameTransform>,
    ) -> ViewerResult<Self> {
        let frame = Self {
            method: method.into(),
            root_frame: root_frame.into(),
            required_frames,
            frames,
            edges,
            graph_ready: false,
        };
        let graph_ready = frame.calculate_graph_ready()?;
        let frame = Self { graph_ready, ..frame };
        frame.validate()?;
        Ok(frame)
    }

    /// Validates graph identity, topology, and the derived readiness bit.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.method.trim().is_empty() || self.root_frame.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "calibration frame evidence requires a method and root frame".into(),
            ));
        }
        let mut required_values = BTreeSet::new();
        for (role, frame) in &self.required_frames {
            if role.trim().is_empty() || frame.trim().is_empty() || !required_values.insert(frame) {
                return Err(ViewerError::InvalidState(
                    "required calibration sensor roles and frames must be unique and non-empty"
                        .into(),
                ));
            }
        }
        if !self.required_frames.contains_key("front") || !self.required_frames.contains_key("rear")
        {
            return Err(ViewerError::InvalidState(
                "calibration frame evidence requires front and rear sensor frames".into(),
            ));
        }
        let mut frame_ids = BTreeSet::new();
        for frame in &self.frames {
            if frame.trim().is_empty() || !frame_ids.insert(frame) {
                return Err(ViewerError::InvalidState(
                    "calibration frame graph IDs must be unique and non-empty".into(),
                ));
            }
        }
        let mut edges = BTreeSet::new();
        for edge in &self.edges {
            edge.validate()?;
            if !frame_ids.contains(&edge.parent_frame) || !frame_ids.contains(&edge.child_frame) {
                return Err(ViewerError::InvalidState(
                    "calibration frame edge refers to an unknown frame".into(),
                ));
            }
            if !edge.source_bound || !edge.accepted {
                return Err(ViewerError::InvalidState(
                    "calibration evidence edges must be source-bound and accepted".into(),
                ));
            }
            if !edges.insert((&edge.parent_frame, &edge.child_frame)) {
                return Err(ViewerError::InvalidState(
                    "calibration frame graph contains a duplicate edge".into(),
                ));
            }
        }
        if !graph_is_acyclic(&self.frames, &self.edges) {
            return Err(ViewerError::InvalidState(
                "calibration frame graph must be acyclic".into(),
            ));
        }
        let calculated_ready = self.calculate_graph_ready()?;
        if self.graph_ready != calculated_ready {
            return Err(ViewerError::InvalidState(
                "calibration frame graph_ready disagrees with graph topology".into(),
            ));
        }
        Ok(())
    }

    fn calculate_graph_ready(&self) -> ViewerResult<bool> {
        let mut frame_ids = BTreeSet::new();
        for frame in &self.frames {
            if frame.trim().is_empty() || !frame_ids.insert(frame) {
                return Err(ViewerError::InvalidState(
                    "calibration frame graph IDs must be unique and non-empty".into(),
                ));
            }
        }
        if !frame_ids.contains(&self.root_frame)
            || self.edges.is_empty()
            || !self.edges.iter().all(|edge| {
                edge.source_bound
                    && edge.accepted
                    && frame_ids.contains(&edge.parent_frame)
                    && frame_ids.contains(&edge.child_frame)
            })
            || !graph_is_acyclic(&self.frames, &self.edges)
        {
            return Ok(false);
        }
        Ok(self.required_frames.values().all(|target| {
            target != &self.root_frame
                && frame_ids.contains(target)
                && path_exists(&self.edges, &self.root_frame, target)
        }))
    }

    /// Returns whether a complete root-to-front/rear graph is registered.
    #[must_use]
    pub fn registration_ready(&self) -> bool {
        self.graph_ready
    }
}

/// Portable source-bound calibration registration state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct CalibrationEvidenceState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing title.
    pub title: String,
    /// Exact source identity to which both evidence documents refer.
    pub source: StudioSource,
    /// Checksummed clock evidence file receipt.
    pub clock_artifact: CalibrationArtifact,
    /// Checksummed frame evidence file receipt.
    pub frame_artifact: CalibrationArtifact,
    /// Parsed clock registration evidence.
    pub clock: CalibrationEvidenceClock,
    /// Parsed frame graph registration evidence.
    pub frame: CalibrationEvidenceFrame,
    /// Whether all registration gates passed.
    pub registration_ready: bool,
    /// Fail-closed reasons for registration or later application.
    pub blockers: Vec<String>,
}

impl CalibrationEvidenceState {
    /// Creates a state and derives source-bound registration admission.
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        clock_artifact: CalibrationArtifact,
        frame_artifact: CalibrationArtifact,
        clock: CalibrationEvidenceClock,
        frame: CalibrationEvidenceFrame,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let registration_ready = source.identity_matches
            && clock_artifact.status == "registered"
            && clock_artifact.source_bound
            && frame_artifact.status == "registered"
            && frame_artifact.source_bound
            && clock.registration_ready()
            && frame.registration_ready()
            && blockers.is_empty();
        let state = Self {
            version: CALIBRATION_EVIDENCE_STATE_VERSION,
            title: title.into(),
            source,
            clock_artifact,
            frame_artifact,
            clock,
            frame,
            registration_ready,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates evidence documents and the derived registration decision.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != CALIBRATION_EVIDENCE_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported calibration evidence state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "calibration evidence title must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.clock_artifact.validate()?;
        self.frame_artifact.validate()?;
        self.clock.validate()?;
        self.frame.validate()?;
        let calculated_registration = self.source.identity_matches
            && self.clock_artifact.status == "registered"
            && self.clock_artifact.source_bound
            && self.frame_artifact.status == "registered"
            && self.frame_artifact.source_bound
            && self.clock.registration_ready()
            && self.frame.registration_ready()
            && self.blockers.is_empty();
        if self.registration_ready != calculated_registration {
            return Err(ViewerError::InvalidState(
                "calibration registration_ready disagrees with evidence gates".into(),
            ));
        }
        if self.registration_ready && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted calibration evidence cannot contain blockers".into(),
            ));
        }
        if !self.registration_ready && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked calibration evidence must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "calibration evidence blockers must not be empty".into(),
            ));
        }
        Ok(())
    }
}

fn graph_is_acyclic(frames: &[String], edges: &[FrameTransform]) -> bool {
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut indegree: BTreeMap<&str, usize> = BTreeMap::new();
    for frame in frames {
        adjacency.entry(frame.as_str()).or_default();
        indegree.insert(frame.as_str(), 0);
    }
    for edge in edges.iter().filter(|edge| edge.accepted) {
        let Some(value) = indegree.get_mut(edge.child_frame.as_str()) else {
            return false;
        };
        adjacency.entry(edge.parent_frame.as_str()).or_default().push(edge.child_frame.as_str());
        *value += 1;
    }
    let mut queue = indegree
        .iter()
        .filter_map(|(frame, degree)| (*degree == 0).then_some(*frame))
        .collect::<Vec<_>>();
    let mut visited = 0_usize;
    while let Some(frame) = queue.pop() {
        visited += 1;
        for child in adjacency.get(frame).into_iter().flatten() {
            let Some(degree) = indegree.get_mut(child) else {
                return false;
            };
            *degree -= 1;
            if *degree == 0 {
                queue.push(child);
            }
        }
    }
    visited == indegree.len()
}

fn path_exists(edges: &[FrameTransform], root: &str, target: &str) -> bool {
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in edges.iter().filter(|edge| edge.accepted) {
        adjacency.entry(edge.parent_frame.as_str()).or_default().push(edge.child_frame.as_str());
    }
    let mut queue = VecDeque::from([root]);
    let mut visited = BTreeSet::from([root]);
    while let Some(frame) = queue.pop_front() {
        if frame == target {
            return true;
        }
        for child in adjacency.get(frame).into_iter().flatten() {
            if visited.insert(child) {
                queue.push_back(child);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn source(matches: bool) -> StudioSource {
        let observed = if matches {
            SHA
        } else {
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
        };
        StudioSource::try_new("canonical bag", "/media/canonical.db3", SHA, observed, matches)
            .unwrap()
    }

    fn artifacts(bound: bool) -> (CalibrationArtifact, CalibrationArtifact) {
        (
            CalibrationArtifact::try_new(
                "clock_evidence",
                "registered",
                Some("/media/clock.json".into()),
                Some(SHA.into()),
                bound,
            )
            .unwrap(),
            CalibrationArtifact::try_new(
                "frame_evidence",
                "registered",
                Some("/media/frame.json".into()),
                Some(SHA.into()),
                bound,
            )
            .unwrap(),
        )
    }

    fn clock(bound: bool) -> CalibrationEvidenceClock {
        CalibrationEvidenceClock::try_new(
            "ros2-external",
            "canonical",
            "fixture-clock-fit",
            ClockCalibration::try_new(
                "registered",
                "explicit clock model; not applied",
                12,
                Some(-2.0),
                Some(5.0),
                Some(0.1),
                Some(10.0),
                bound,
                false,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn frame(cycle: bool) -> ViewerResult<CalibrationEvidenceFrame> {
        let required = BTreeMap::from([
            ("front".into(), "lidar_front".into()),
            ("rear".into(), "lidar_rear".into()),
        ]);
        let mut edges = vec![
            FrameTransform::try_new(
                "base_link",
                "lidar_front",
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                None,
                true,
                true,
            )
            .unwrap(),
            FrameTransform::try_new(
                "base_link",
                "lidar_rear",
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                None,
                true,
                true,
            )
            .unwrap(),
        ];
        if cycle {
            edges.push(
                FrameTransform::try_new(
                    "lidar_front",
                    "base_link",
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    None,
                    true,
                    true,
                )
                .unwrap(),
            );
        }
        CalibrationEvidenceFrame::try_new(
            "fixture-extrinsic-fit",
            "base_link",
            required,
            vec!["base_link".into(), "lidar_front".into(), "lidar_rear".into()],
            edges,
        )
    }

    #[test]
    fn healthy_source_bound_evidence_is_registration_ready() {
        let (clock_artifact, frame_artifact) = artifacts(true);
        let state = CalibrationEvidenceState::try_new(
            "Calibration Evidence",
            source(true),
            clock_artifact,
            frame_artifact,
            clock(true),
            frame(false).unwrap(),
            Vec::new(),
        )
        .unwrap();
        assert!(state.registration_ready);
        state.validate().unwrap();
    }

    #[test]
    fn source_mismatch_withholds_registration() {
        let (clock_artifact, frame_artifact) = artifacts(true);
        let state = CalibrationEvidenceState::try_new(
            "Calibration Evidence",
            source(false),
            clock_artifact,
            frame_artifact,
            clock(true),
            frame(false).unwrap(),
            vec!["input SHA mismatch".into()],
        )
        .unwrap();
        assert!(!state.registration_ready);
    }

    #[test]
    fn missing_evidence_is_a_valid_blocked_state() {
        let required = BTreeMap::from([
            ("front".into(), "lidar_front".into()),
            ("rear".into(), "lidar_rear".into()),
        ]);
        let (clock_artifact, frame_artifact) = (
            CalibrationArtifact::try_new("clock_evidence", "not_registered", None, None, false)
                .unwrap(),
            CalibrationArtifact::try_new("frame_evidence", "not_registered", None, None, false)
                .unwrap(),
        );
        let state = CalibrationEvidenceState::try_new(
            "Calibration Evidence",
            source(true),
            clock_artifact,
            frame_artifact,
            CalibrationEvidenceClock::try_new(
                "unknown",
                "uncalibrated",
                "not_registered",
                ClockCalibration::try_new(
                    "not_registered",
                    "header stamp",
                    0,
                    None,
                    None,
                    None,
                    None,
                    false,
                    false,
                )
                .unwrap(),
            )
            .unwrap(),
            CalibrationEvidenceFrame::try_new(
                "not_registered",
                "base_link",
                required,
                Vec::new(),
                Vec::new(),
            )
            .unwrap(),
            vec!["clock evidence is not registered".into()],
        )
        .unwrap();
        assert!(!state.registration_ready);
        state.validate().unwrap();
    }

    #[test]
    fn cyclic_frame_graph_is_rejected() {
        assert!(frame(true).is_err());
    }
}
