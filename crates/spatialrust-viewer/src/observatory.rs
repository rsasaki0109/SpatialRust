//! Portable TF and calibration observability state.
//!
//! The observatory records evidence and admission decisions. It does not solve
//! calibration, apply transforms, or infer a frame root from incomplete data.

use std::collections::{BTreeMap, BTreeSet};

use crate::{StudioSource, ViewerError, ViewerResult};

/// Current serialized TF/calibration observatory schema version.
pub const CALIBRATION_OBSERVATORY_STATE_VERSION: u32 = 1;

/// Source-bound status for one calibration artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct CalibrationArtifact {
    /// Artifact kind, such as clock or frame.
    pub kind: String,
    /// Receipt status, such as registered or not_registered.
    pub status: String,
    /// Registered artifact path, when supplied.
    pub path: Option<String>,
    /// Registered artifact SHA-256, when supplied.
    pub sha256: Option<String>,
    /// Whether the artifact receipt is bound to the displayed source.
    pub source_bound: bool,
}

impl CalibrationArtifact {
    /// Creates and validates an artifact status.
    pub fn try_new(
        kind: impl Into<String>,
        status: impl Into<String>,
        path: Option<String>,
        sha256: Option<String>,
        source_bound: bool,
    ) -> ViewerResult<Self> {
        let artifact =
            Self { kind: kind.into(), status: status.into(), path, sha256, source_bound };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Validates registration fields and source-binding consistency.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.kind.trim().is_empty() || self.status.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "calibration artifact kind and status must not be empty".into(),
            ));
        }
        if self.status == "registered" {
            let path = self.path.as_deref().unwrap_or_default();
            let sha256 = self.sha256.as_deref().unwrap_or_default();
            if path.trim().is_empty() {
                return Err(ViewerError::InvalidState(format!(
                    "registered {} artifact has no path",
                    self.kind
                )));
            }
            validate_sha256(&format!("{} artifact", self.kind), sha256)?;
        } else if self.source_bound {
            return Err(ViewerError::InvalidState(format!(
                "unregistered {} artifact cannot be source-bound",
                self.kind
            )));
        }
        Ok(())
    }
}

/// Clock calibration observability and application status.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ClockCalibration {
    /// Receipt status for the clock artifact.
    pub status: String,
    /// Timestamp basis exposed to downstream consumers.
    pub time_basis: String,
    /// Number of clock correspondence samples.
    pub sample_count: u64,
    /// Median signed offset in nanoseconds, when measured.
    pub median_offset_nanos: Option<f64>,
    /// P95 absolute offset in nanoseconds, when measured.
    pub p95_abs_offset_nanos: Option<f64>,
    /// Estimated clock drift in parts per million, when measured.
    pub drift_ppm: Option<f64>,
    /// Estimated uncertainty in nanoseconds, when measured.
    pub uncertainty_nanos: Option<f64>,
    /// Whether clock evidence belongs to the displayed source.
    pub source_bound: bool,
    /// Whether the clock model was actually applied to the timeline.
    pub applied: bool,
}

impl ClockCalibration {
    /// Creates and validates clock observability fields.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        status: impl Into<String>,
        time_basis: impl Into<String>,
        sample_count: u64,
        median_offset_nanos: Option<f64>,
        p95_abs_offset_nanos: Option<f64>,
        drift_ppm: Option<f64>,
        uncertainty_nanos: Option<f64>,
        source_bound: bool,
        applied: bool,
    ) -> ViewerResult<Self> {
        let clock = Self {
            status: status.into(),
            time_basis: time_basis.into(),
            sample_count,
            median_offset_nanos,
            p95_abs_offset_nanos,
            drift_ppm,
            uncertainty_nanos,
            source_bound,
            applied,
        };
        clock.validate()?;
        Ok(clock)
    }

    /// Validates finite diagnostics and the applied-model invariant.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.status.trim().is_empty() || self.time_basis.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "clock status and time basis must not be empty".into(),
            ));
        }
        for (label, value) in [
            ("median offset", self.median_offset_nanos),
            ("p95 absolute offset", self.p95_abs_offset_nanos),
            ("clock drift", self.drift_ppm),
            ("clock uncertainty", self.uncertainty_nanos),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ViewerError::InvalidState(format!(
                    "clock {label} must be finite when present"
                )));
            }
        }
        if self.applied && (!self.source_bound || self.status != "registered") {
            return Err(ViewerError::InvalidState(
                "applied clock calibration must be registered and source-bound".into(),
            ));
        }
        Ok(())
    }
}

/// One rigid TF edge observed by the source-bound inventory.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct FrameTransform {
    /// Parent frame ID.
    pub parent_frame: String,
    /// Child frame ID.
    pub child_frame: String,
    /// Translation from parent to child in metres.
    pub translation_m: [f64; 3],
    /// Quaternion in x, y, z, w order.
    pub rotation_xyzw: [f64; 4],
    /// Transform timestamp, when carried by the source message.
    pub stamp_nanos: Option<u64>,
    /// Whether this edge belongs to the displayed source.
    pub source_bound: bool,
    /// Whether this edge is admitted for composition.
    pub accepted: bool,
}

impl FrameTransform {
    /// Creates and validates one rigid transform edge.
    pub fn try_new(
        parent_frame: impl Into<String>,
        child_frame: impl Into<String>,
        translation_m: [f64; 3],
        rotation_xyzw: [f64; 4],
        stamp_nanos: Option<u64>,
        source_bound: bool,
        accepted: bool,
    ) -> ViewerResult<Self> {
        let transform = Self {
            parent_frame: parent_frame.into(),
            child_frame: child_frame.into(),
            translation_m,
            rotation_xyzw,
            stamp_nanos,
            source_bound,
            accepted,
        };
        transform.validate()?;
        Ok(transform)
    }

    /// Validates frame names, finite values, and the acceptance invariant.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.parent_frame.trim().is_empty()
            || self.child_frame.trim().is_empty()
            || self.parent_frame == self.child_frame
        {
            return Err(ViewerError::InvalidState(
                "TF edge parent and child frames must be distinct and non-empty".into(),
            ));
        }
        if self.translation_m.iter().any(|value| !value.is_finite())
            || self.rotation_xyzw.iter().any(|value| !value.is_finite())
        {
            return Err(ViewerError::InvalidState(
                "TF edge translation and quaternion must be finite".into(),
            ));
        }
        let norm = self.rotation_xyzw.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm <= f64::EPSILON {
            return Err(ViewerError::InvalidState("TF edge quaternion must be non-zero".into()));
        }
        if self.accepted && !self.source_bound {
            return Err(ViewerError::InvalidState("an unbound TF edge cannot be accepted".into()));
        }
        Ok(())
    }
}

/// One portable TF/calibration observatory snapshot.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct CalibrationObservatoryState {
    /// Serialized observatory schema version.
    pub version: u32,
    /// User-facing observatory title.
    pub title: String,
    /// Checksummed input identity.
    pub source: StudioSource,
    /// Clock calibration artifact receipt.
    pub clock_artifact: CalibrationArtifact,
    /// Frame calibration artifact receipt.
    pub frame_artifact: CalibrationArtifact,
    /// Clock diagnostics and application status.
    pub clock: ClockCalibration,
    /// Source-bound frame IDs accepted by the inventory.
    pub frames: Vec<String>,
    /// Observed transform edges.
    pub edges: Vec<FrameTransform>,
    /// Number of edges rejected before they entered the graph.
    pub rejected_edge_count: u64,
    /// Whether the frame inventory belongs to the displayed source.
    pub frame_inventory_source_bound: bool,
    /// Root selected for a composition operation, when any.
    pub root_frame: Option<String>,
    /// Whether the graph has been composed for downstream use.
    pub composed: bool,
    /// Whether the accepted graph is acyclic.
    pub cycle_free: bool,
    /// Fail-closed reasons shown by the observatory.
    pub blockers: Vec<String>,
    /// Whether clock, frame, source, and graph gates all admit calibration use.
    pub calibration_admitted: bool,
}

impl CalibrationObservatoryState {
    /// Creates an observatory snapshot and derives its admission decision.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        clock_artifact: CalibrationArtifact,
        frame_artifact: CalibrationArtifact,
        clock: ClockCalibration,
        frames: Vec<String>,
        edges: Vec<FrameTransform>,
        rejected_edge_count: u64,
        frame_inventory_source_bound: bool,
        root_frame: Option<String>,
        composed: bool,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let cycle_free = graph_is_acyclic(&frames, &edges);
        let calibration_admitted = source.identity_matches
            && clock_artifact.status == "registered"
            && clock_artifact.source_bound
            && frame_artifact.status == "registered"
            && frame_artifact.source_bound
            && clock.source_bound
            && clock.applied
            && frame_inventory_source_bound
            && composed
            && cycle_free
            && !edges.is_empty()
            && edges.iter().all(|edge| edge.accepted)
            && blockers.is_empty();
        let state = Self {
            version: CALIBRATION_OBSERVATORY_STATE_VERSION,
            title: title.into(),
            source,
            clock_artifact,
            frame_artifact,
            clock,
            frames,
            edges,
            rejected_edge_count,
            frame_inventory_source_bound,
            root_frame,
            composed,
            cycle_free,
            blockers,
            calibration_admitted,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates receipt identity, graph topology, and fail-closed admission.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != CALIBRATION_OBSERVATORY_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported calibration observatory state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "calibration observatory title must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.clock_artifact.validate()?;
        self.frame_artifact.validate()?;
        self.clock.validate()?;

        let mut frames = BTreeSet::new();
        for frame in &self.frames {
            if frame.trim().is_empty() || !frames.insert(frame) {
                return Err(ViewerError::InvalidState(
                    "observatory frame IDs must be unique and non-empty".into(),
                ));
            }
        }
        let mut edges = BTreeSet::new();
        for edge in &self.edges {
            edge.validate()?;
            if !frames.contains(&edge.parent_frame) || !frames.contains(&edge.child_frame) {
                return Err(ViewerError::InvalidState(
                    "observatory edge refers to a frame outside the accepted inventory".into(),
                ));
            }
            if !edges.insert((&edge.parent_frame, &edge.child_frame)) {
                return Err(ViewerError::InvalidState(
                    "observatory frame graph contains a duplicate edge".into(),
                ));
            }
            if edge.accepted && (!edge.source_bound || !self.source.identity_matches) {
                return Err(ViewerError::InvalidState(
                    "accepted observatory edges must match the input source identity".into(),
                ));
            }
        }
        if !self.frame_inventory_source_bound && (!self.frames.is_empty() || !self.edges.is_empty())
        {
            return Err(ViewerError::InvalidState(
                "unbound frame inventory cannot populate observatory graph fields".into(),
            ));
        }
        if let Some(root) = &self.root_frame {
            if root.trim().is_empty() || !frames.contains(root) {
                return Err(ViewerError::InvalidState(
                    "observatory root must be one of the accepted frames".into(),
                ));
            }
        }
        if self.composed && (self.root_frame.is_none() || self.edges.is_empty()) {
            return Err(ViewerError::InvalidState(
                "composed observatory graph requires a root and accepted edges".into(),
            ));
        }
        if self.cycle_free != graph_is_acyclic(&self.frames, &self.edges) {
            return Err(ViewerError::InvalidState(
                "observatory cycle_free disagrees with graph topology".into(),
            ));
        }
        let calculated_admission = self.source.identity_matches
            && self.clock_artifact.status == "registered"
            && self.clock_artifact.source_bound
            && self.frame_artifact.status == "registered"
            && self.frame_artifact.source_bound
            && self.clock.source_bound
            && self.clock.applied
            && self.frame_inventory_source_bound
            && self.composed
            && self.cycle_free
            && !self.edges.is_empty()
            && self.edges.iter().all(|edge| edge.accepted)
            && self.blockers.is_empty();
        if self.calibration_admitted != calculated_admission {
            return Err(ViewerError::InvalidState(
                "calibration_admitted disagrees with observatory gates".into(),
            ));
        }
        if self.calibration_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted calibration cannot contain blockers".into(),
            ));
        }
        if !self.calibration_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked calibration must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState("observatory blockers must not be empty".into()));
        }
        Ok(())
    }
}

fn graph_is_acyclic(frames: &[String], edges: &[FrameTransform]) -> bool {
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut indegree: BTreeMap<&str, usize> = BTreeMap::new();
    for frame in frames {
        indegree.insert(frame.as_str(), 0);
        adjacency.entry(frame.as_str()).or_default();
    }
    for edge in edges.iter().filter(|edge| edge.accepted) {
        adjacency.entry(edge.parent_frame.as_str()).or_default().push(edge.child_frame.as_str());
        let Some(value) = indegree.get_mut(edge.child_frame.as_str()) else {
            return false;
        };
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

    fn source(identity_matches: bool) -> StudioSource {
        let observed = if identity_matches {
            SHA
        } else {
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
        };
        StudioSource::try_new("bag", "/tmp/bag.db3", SHA, observed, identity_matches).unwrap()
    }

    fn blocked() -> CalibrationObservatoryState {
        CalibrationObservatoryState::try_new(
            "TF Observatory",
            source(true),
            CalibrationArtifact::try_new("clock", "not_registered", None, None, false).unwrap(),
            CalibrationArtifact::try_new("frame", "not_registered", None, None, false).unwrap(),
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
            Vec::new(),
            Vec::new(),
            0,
            false,
            None,
            false,
            vec!["clock artifact is missing".into()],
        )
        .unwrap()
    }

    #[test]
    fn blocked_observatory_state_is_valid() {
        let state = blocked();
        assert!(!state.calibration_admitted);
        assert!(state.validate().is_ok());
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(serde_json::from_str::<CalibrationObservatoryState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn unbound_accepted_edge_is_rejected() {
        let edge = FrameTransform::try_new(
            "map",
            "lidar",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            None,
            false,
            true,
        );
        assert!(edge.is_err());
    }

    #[test]
    fn source_mismatch_cannot_admit_an_accepted_edge() {
        let edge = FrameTransform::try_new(
            "map",
            "lidar",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            None,
            true,
            true,
        )
        .unwrap();
        let result = CalibrationObservatoryState::try_new(
            "TF Observatory",
            source(false),
            CalibrationArtifact::try_new("clock", "not_registered", None, None, false).unwrap(),
            CalibrationArtifact::try_new("frame", "not_registered", None, None, false).unwrap(),
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
            vec!["map".into(), "lidar".into()],
            vec![edge],
            0,
            true,
            None,
            false,
            vec!["source mismatch".into()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn finite_transform_and_camera_types_are_available_without_unsafe() {
        let camera = Camera::try_new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 10.0 },
        )
        .unwrap();
        assert!(camera.eye.z.is_finite());
    }
}
