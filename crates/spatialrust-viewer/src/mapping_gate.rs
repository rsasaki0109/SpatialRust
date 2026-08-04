//! Source-bound bounded full-bag mapping admission state.
//!
//! This module is deliberately independent of ROS, SQLite, and renderer
//! implementations.  An adapter may populate the state after applying an
//! explicitly registered clock model and frame graph, while consumers can
//! inspect the same fail-closed decision without re-running the mapping.

use crate::{CalibrationEvidenceState, ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized bounded full-bag mapping state schema version.
pub const FULL_BAG_MAPPING_STATE_VERSION: u32 = 1;

/// Source and bounded-ingest totals for one full-bag mapping run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MappingSourceSummary {
    /// Front sensor topic consumed by the run.
    pub front_topic: String,
    /// Rear sensor topic consumed by the run.
    pub rear_topic: String,
    /// Number of PointCloud2 messages present in the front topic.
    pub front_bag_message_count: u64,
    /// Number of PointCloud2 messages present in the rear topic.
    pub rear_bag_message_count: u64,
    /// Number of bounded source chunks emitted for the front topic.
    pub front_chunk_count: u64,
    /// Number of bounded source chunks emitted for the rear topic.
    pub rear_chunk_count: u64,
    /// Number of complete bounded records retained from the front topic.
    pub front_record_count: u64,
    /// Number of complete bounded records retained from the rear topic.
    pub rear_record_count: u64,
    /// Total retained records across both topics.
    pub total_record_count: u64,
    /// Total retained points across both topics.
    pub total_point_count: u64,
    /// Conservative allocated column bytes retained by the episode.
    pub retained_bytes: u64,
    /// Largest raw source message allocation declared by the selected topics.
    pub peak_source_bytes: u64,
    /// First corrected timestamp, when a bag was processed.
    pub start_nanos: Option<u64>,
    /// Last corrected timestamp, when a bag was processed.
    pub end_nanos: Option<u64>,
    /// Whether every selected source record was consumed within the limits.
    pub full_bag_processed: bool,
    /// Whether a configured hard bound stopped source consumption.
    pub truncated: bool,
}

impl MappingSourceSummary {
    /// Validates source identities and bounded totals.
    pub fn validate(&self) -> ViewerResult<()> {
        let expected_total =
            self.front_record_count.checked_add(self.rear_record_count).ok_or_else(|| {
                ViewerError::InvalidState("mapping source record total overflow".into())
            })?;
        if self.front_topic.trim().is_empty()
            || self.rear_topic.trim().is_empty()
            || self.front_topic == self.rear_topic
            || self.front_bag_message_count < self.front_record_count
            || self.rear_bag_message_count < self.rear_record_count
            || self.front_chunk_count < self.front_record_count
            || self.rear_chunk_count < self.rear_record_count
            || self.total_record_count != expected_total
            || (self.total_record_count == 0 && self.total_point_count != 0)
            || self.truncated && self.full_bag_processed
        {
            return Err(ViewerError::InvalidState(
                "mapping source summary has invalid topic or bounded totals".into(),
            ));
        }
        if let (Some(start), Some(end)) = (self.start_nanos, self.end_nanos) {
            if start > end {
                return Err(ViewerError::InvalidState(
                    "mapping source timestamp bounds are reversed".into(),
                ));
            }
        } else if self.start_nanos.is_some() != self.end_nanos.is_some() {
            return Err(ViewerError::InvalidState(
                "mapping source timestamp bounds must be complete".into(),
            ));
        }
        if self.full_bag_processed
            && (self.total_record_count == 0
                || self.total_point_count == 0
                || self.retained_bytes == 0
                || self.peak_source_bytes == 0)
        {
            return Err(ViewerError::InvalidState(
                "a completed full-bag mapping run must retain source records and points".into(),
            ));
        }
        Ok(())
    }
}

/// Odometry and pose-graph receipt for the bounded full-bag run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MappingOdometrySummary {
    /// Topic used as the primary trajectory stream.
    pub topic: String,
    /// Original sensor frame of the primary trajectory stream.
    pub source_frame: String,
    /// Root frame after explicit frame-graph application.
    pub root_frame: String,
    /// Clock identifier used by corrected trajectory stamps.
    pub clock_id: String,
    /// Human-readable clock-domain label.
    pub clock_domain: String,
    /// Matcher and bounded-run description.
    pub matcher: String,
    /// Number of scans in the trajectory.
    pub scan_count: u64,
    /// Number of sequential relative motions.
    pub motion_count: u64,
    /// Number of pose-graph nodes.
    pub pose_graph_node_count: u64,
    /// Number of pose-graph edges.
    pub pose_graph_edge_count: u64,
    /// Whether odometry consumed the complete selected stream.
    pub complete: bool,
    /// Whether the selected stream was cut by a bound.
    pub truncated: bool,
}

impl MappingOdometrySummary {
    /// Validates trajectory and pose-graph totals.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.topic.trim().is_empty()
            || self.source_frame.trim().is_empty()
            || self.root_frame.trim().is_empty()
            || self.clock_id.trim().is_empty()
            || self.clock_domain.trim().is_empty()
            || self.matcher.trim().is_empty()
            || self.motion_count > self.scan_count
            || self.pose_graph_node_count != self.scan_count
            || self.pose_graph_edge_count != self.motion_count
            || self.truncated && self.complete
        {
            return Err(ViewerError::InvalidState(
                "mapping odometry summary has invalid identity or graph totals".into(),
            ));
        }
        if self.complete
            && (self.scan_count < 2 || self.motion_count.checked_add(1) != Some(self.scan_count))
        {
            return Err(ViewerError::InvalidState(
                "completed mapping odometry requires a connected scan trajectory".into(),
            ));
        }
        Ok(())
    }
}

/// TSDF and mesh receipt for one mapping run.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MappingTsdfSummary {
    /// Frame in which the volume is expressed.
    pub frame_id: String,
    /// Volume origin in metres.
    pub origin: [f32; 3],
    /// Voxel edge length in metres.
    pub voxel_size: f32,
    /// Number of voxels on each axis.
    pub dims: [usize; 3],
    /// TSDF truncation distance in metres.
    pub truncation: f32,
    /// Number of records integrated into the volume.
    pub integrated_record_count: u64,
    /// Number of point samples visited by integration.
    pub integrated_point_count: u64,
    /// Number of extracted mesh vertices.
    pub mesh_vertex_count: u64,
    /// Number of extracted mesh triangles.
    pub mesh_triangle_count: u64,
    /// Whether TSDF integration and extraction completed.
    pub complete: bool,
}

impl MappingTsdfSummary {
    /// Validates finite volume configuration and integration totals.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.frame_id.trim().is_empty()
            || self.origin.iter().any(|value| !value.is_finite())
            || !self.voxel_size.is_finite()
            || self.voxel_size <= 0.0
            || self.dims.contains(&0)
            || !self.truncation.is_finite()
            || self.truncation <= 0.0
            || self.mesh_vertex_count == 0 && self.mesh_triangle_count != 0
        {
            return Err(ViewerError::InvalidState(
                "mapping TSDF summary has invalid volume or mesh values".into(),
            ));
        }
        if self.complete && (self.integrated_record_count == 0 || self.integrated_point_count == 0)
        {
            return Err(ViewerError::InvalidState(
                "completed mapping TSDF requires integrated records and points".into(),
            ));
        }
        Ok(())
    }
}

/// Derived admission levels for bounded full-bag mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MappingGateSummary {
    /// Whether source-bound calibration evidence is registered.
    pub calibration_registered: bool,
    /// Whether the registered clock model was applied to timestamps.
    pub clock_applied: bool,
    /// Whether the registered frame graph was applied to sensor geometry.
    pub frame_graph_applied: bool,
    /// Whether the complete selected bag was retained within bounds.
    pub full_bag_processed: bool,
    /// Whether bounded odometry completed over the selected stream.
    pub odometry_complete: bool,
    /// Whether bounded TSDF integration and extraction completed.
    pub tsdf_complete: bool,
    /// Whether calibrated-world mapping may be consumed downstream.
    pub mapping_admitted: bool,
}

/// Portable source-bound bounded full-bag mapping state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct FullBagMappingState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Exact input identity used by the run.
    pub source: StudioSource,
    /// Optional parsed calibration state; absence is a blocking condition.
    pub calibration: Option<CalibrationEvidenceState>,
    /// Bounded source ingest totals.
    pub source_summary: MappingSourceSummary,
    /// Odometry receipt, when the source and calibration gates allowed it.
    pub odometry: Option<MappingOdometrySummary>,
    /// TSDF receipt, when mapping stages completed.
    pub tsdf: Option<MappingTsdfSummary>,
    /// Checksummed input and derived artifacts.
    pub artifacts: Vec<ReplayArtifact>,
    /// Derived admission levels.
    pub summary: MappingGateSummary,
    /// Human-readable fail-closed reasons.
    pub blockers: Vec<String>,
}

impl FullBagMappingState {
    /// Creates state and derives its mapping admission decision.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        calibration: Option<CalibrationEvidenceState>,
        source_summary: MappingSourceSummary,
        odometry: Option<MappingOdometrySummary>,
        tsdf: Option<MappingTsdfSummary>,
        artifacts: Vec<ReplayArtifact>,
        clock_applied: bool,
        frame_graph_applied: bool,
        mut blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let calibration_registered = calibration.as_ref().is_some_and(|calibration| {
            calibration.registration_ready
                && calibration.source.identity_matches
                && calibration.source.path == source.path
                && calibration.source.observed_sha256 == source.observed_sha256
        });
        let full_bag_processed = source_summary.full_bag_processed && !source_summary.truncated;
        let odometry_complete =
            odometry.as_ref().is_some_and(|odometry| odometry.complete && !odometry.truncated);
        let tsdf_complete = tsdf.as_ref().is_some_and(|tsdf| tsdf.complete);

        if !source.identity_matches {
            push_blocker(
                &mut blockers,
                "mapping input source identity does not match expected SHA-256",
            );
        }
        if calibration.is_none() {
            push_blocker(&mut blockers, "source-bound calibration evidence was not supplied");
        } else if !calibration_registered {
            push_blocker(
                &mut blockers,
                "source-bound calibration evidence registration is incomplete",
            );
        }
        if !full_bag_processed {
            push_blocker(
                &mut blockers,
                "full-bag ingest was not completed because an admission gate blocked execution or a configured bound was reached",
            );
        }
        if !clock_applied {
            push_blocker(
                &mut blockers,
                "registered clock model was not applied to the mapping timeline",
            );
        }
        if !frame_graph_applied {
            push_blocker(
                &mut blockers,
                "registered frame graph was not applied to sensor geometry",
            );
        }
        if !odometry_complete {
            push_blocker(&mut blockers, "full-bag frame-aware odometry did not complete");
        }
        if !tsdf_complete {
            push_blocker(&mut blockers, "full-bag TSDF integration did not complete");
        }

        let summary = MappingGateSummary {
            calibration_registered,
            clock_applied,
            frame_graph_applied,
            full_bag_processed,
            odometry_complete,
            tsdf_complete,
            mapping_admitted: source.identity_matches
                && calibration_registered
                && clock_applied
                && frame_graph_applied
                && full_bag_processed
                && odometry_complete
                && tsdf_complete
                && blockers.is_empty(),
        };
        let state = Self {
            version: FULL_BAG_MAPPING_STATE_VERSION,
            title: title.into(),
            source,
            calibration,
            source_summary,
            odometry,
            tsdf,
            artifacts,
            summary,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates source, calibration, stage receipts, and derived gates.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != FULL_BAG_MAPPING_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported full-bag mapping state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "full-bag mapping title must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.source_summary.validate()?;
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        if let Some(odometry) = &self.odometry {
            odometry.validate()?;
        }
        if let Some(tsdf) = &self.tsdf {
            tsdf.validate()?;
        }
        let mut roles = std::collections::BTreeSet::new();
        let mut paths = std::collections::BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !roles.insert(&artifact.role) || !paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "full-bag mapping artifacts must have unique roles and paths".into(),
                ));
            }
        }

        let calculated_calibration = self.calibration.as_ref().is_some_and(|calibration| {
            calibration.registration_ready
                && calibration.source.identity_matches
                && calibration.source.path == self.source.path
                && calibration.source.observed_sha256 == self.source.observed_sha256
        });
        let calculated_full =
            self.source_summary.full_bag_processed && !self.source_summary.truncated;
        let calculated_odometry =
            self.odometry.as_ref().is_some_and(|odometry| odometry.complete && !odometry.truncated);
        let calculated_tsdf = self.tsdf.as_ref().is_some_and(|tsdf| tsdf.complete);
        if self.summary.calibration_registered != calculated_calibration
            || self.summary.full_bag_processed != calculated_full
            || self.summary.odometry_complete != calculated_odometry
            || self.summary.tsdf_complete != calculated_tsdf
            || (self.summary.clock_applied && !calculated_calibration)
            || (self.summary.frame_graph_applied && !calculated_calibration)
        {
            return Err(ViewerError::InvalidState(
                "full-bag mapping summary disagrees with source or stage receipts".into(),
            ));
        }
        let calculated_mapping = self.source.identity_matches
            && calculated_calibration
            && self.summary.clock_applied
            && self.summary.frame_graph_applied
            && calculated_full
            && calculated_odometry
            && calculated_tsdf
            && self.blockers.is_empty();
        if self.summary.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with full-bag mapping gates".into(),
            ));
        }
        if self.summary.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted full-bag mapping cannot contain blockers".into(),
            ));
        }
        if !self.summary.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked full-bag mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "full-bag mapping blockers must not be empty".into(),
            ));
        }
        Ok(())
    }
}

fn push_blocker(blockers: &mut Vec<String>, blocker: impl Into<String>) {
    let blocker = blocker.into();
    if !blockers.iter().any(|existing| existing == &blocker) {
        blockers.push(blocker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationArtifact, CalibrationEvidenceClock, CalibrationEvidenceFrame, ClockCalibration,
        FrameTransform,
    };

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn source() -> StudioSource {
        StudioSource::try_new("fixture", "/media/fixture.db3", SHA, SHA, true).unwrap()
    }

    fn source_summary() -> MappingSourceSummary {
        MappingSourceSummary {
            front_topic: "/front".into(),
            rear_topic: "/rear".into(),
            front_bag_message_count: 0,
            rear_bag_message_count: 0,
            front_chunk_count: 0,
            rear_chunk_count: 0,
            front_record_count: 0,
            rear_record_count: 0,
            total_record_count: 0,
            total_point_count: 0,
            retained_bytes: 0,
            peak_source_bytes: 0,
            start_nanos: None,
            end_nanos: None,
            full_bag_processed: false,
            truncated: false,
        }
    }

    #[test]
    fn missing_calibration_is_a_valid_blocked_state() {
        let state = FullBagMappingState::try_new(
            "fixture mapping",
            source(),
            None,
            source_summary(),
            None,
            None,
            Vec::new(),
            false,
            false,
            Vec::new(),
        )
        .unwrap();
        assert!(!state.summary.mapping_admitted);
        assert!(state.blockers.iter().any(|blocker| blocker.contains("calibration")));
        state.validate().unwrap();
    }

    #[test]
    fn complete_source_bound_stages_are_admitted() {
        let calibration = CalibrationEvidenceState::try_new(
            "fixture calibration",
            source(),
            CalibrationArtifact::try_new(
                "clock_evidence",
                "registered",
                Some("/media/clock.json".into()),
                Some(SHA.into()),
                true,
            )
            .unwrap(),
            CalibrationArtifact::try_new(
                "frame_evidence",
                "registered",
                Some("/media/frame.json".into()),
                Some(SHA.into()),
                true,
            )
            .unwrap(),
            CalibrationEvidenceClock::try_new(
                "sensor",
                "external",
                "fixture clock fit",
                ClockCalibration::try_new(
                    "registered",
                    "anchored external clock",
                    2,
                    Some(1.0),
                    Some(2.0),
                    Some(0.0),
                    Some(3.0),
                    true,
                    false,
                )
                .unwrap(),
            )
            .unwrap(),
            CalibrationEvidenceFrame::try_new(
                "fixture extrinsic fit",
                "base_link",
                std::collections::BTreeMap::from([
                    ("front".into(), "front_frame".into()),
                    ("rear".into(), "rear_frame".into()),
                ]),
                vec!["base_link".into(), "front_frame".into(), "rear_frame".into()],
                vec![
                    FrameTransform::try_new(
                        "base_link",
                        "front_frame",
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        None,
                        true,
                        true,
                    )
                    .unwrap(),
                    FrameTransform::try_new(
                        "base_link",
                        "rear_frame",
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        None,
                        true,
                        true,
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
            Vec::new(),
        )
        .unwrap();
        let state = FullBagMappingState::try_new(
            "fixture mapping",
            source(),
            Some(calibration),
            MappingSourceSummary {
                front_topic: "/front".into(),
                rear_topic: "/rear".into(),
                front_bag_message_count: 2,
                rear_bag_message_count: 2,
                front_chunk_count: 2,
                rear_chunk_count: 2,
                front_record_count: 2,
                rear_record_count: 2,
                total_record_count: 4,
                total_point_count: 12,
                retained_bytes: 48,
                peak_source_bytes: 256,
                start_nanos: Some(1),
                end_nanos: Some(2),
                full_bag_processed: true,
                truncated: false,
            },
            Some(MappingOdometrySummary {
                topic: "/front".into(),
                source_frame: "front_frame".into(),
                root_frame: "base_link".into(),
                clock_id: "external".into(),
                clock_domain: "external-calibrated".into(),
                matcher: "fixture".into(),
                scan_count: 2,
                motion_count: 1,
                pose_graph_node_count: 2,
                pose_graph_edge_count: 1,
                complete: true,
                truncated: false,
            }),
            Some(MappingTsdfSummary {
                frame_id: "base_link".into(),
                origin: [0.0, 0.0, 0.0],
                voxel_size: 0.5,
                dims: [8, 8, 8],
                truncation: 1.0,
                integrated_record_count: 4,
                integrated_point_count: 12,
                mesh_vertex_count: 3,
                mesh_triangle_count: 1,
                complete: true,
            }),
            vec![ReplayArtifact::try_new("mesh", "/media/mesh.gltf", 1, SHA).unwrap()],
            true,
            true,
            Vec::new(),
        )
        .unwrap();
        assert!(state.summary.mapping_admitted);
        state.validate().unwrap();
    }

    #[test]
    fn summary_rejects_reversed_source_bounds() {
        let mut summary = source_summary();
        summary.start_nanos = Some(20);
        summary.end_nanos = Some(10);
        assert!(summary.validate().is_err());
    }
}
