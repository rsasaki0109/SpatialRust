//! Portable source-bound state for comparing two reconstructed maps.
//!
//! This module owns only validated comparison data. It does not decode glTF,
//! apply transforms, or decide which device renders the result. Adapters can
//! populate it from receipt-backed meshes while native and Web frontends share
//! the same heatmap and fail-closed admission decision.

use std::collections::BTreeSet;

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized Map Diff state schema version.
pub const MAP_DIFF_STATE_VERSION: u32 = 1;

/// Axis-aligned map bounds represented in integer micrometres.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MapDiffBounds {
    /// Minimum X coordinate in micrometres.
    pub min_x_um: i64,
    /// Minimum Y coordinate in micrometres.
    pub min_y_um: i64,
    /// Minimum Z coordinate in micrometres.
    pub min_z_um: i64,
    /// Maximum X coordinate in micrometres.
    pub max_x_um: i64,
    /// Maximum Y coordinate in micrometres.
    pub max_y_um: i64,
    /// Maximum Z coordinate in micrometres.
    pub max_z_um: i64,
}

impl MapDiffBounds {
    /// Creates and validates integer map bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        min_x_um: i64,
        min_y_um: i64,
        min_z_um: i64,
        max_x_um: i64,
        max_y_um: i64,
        max_z_um: i64,
    ) -> ViewerResult<Self> {
        let bounds = Self { min_x_um, min_y_um, min_z_um, max_x_um, max_y_um, max_z_um };
        bounds.validate()?;
        Ok(bounds)
    }

    /// Validates that every maximum is at least its corresponding minimum.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.max_x_um < self.min_x_um
            || self.max_y_um < self.min_y_um
            || self.max_z_um < self.min_z_um
        {
            return Err(ViewerError::InvalidState(
                "Map Diff bounds have a maximum below its minimum".into(),
            ));
        }
        Ok(())
    }
}

/// Receipt-backed description of one map participating in a comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MapDiffMap {
    /// User-facing map label such as `base` or `candidate`.
    pub label: String,
    /// Canonical input identity used to produce this map.
    pub source: StudioSource,
    /// Checksummed glTF or mesh artifact.
    pub artifact: ReplayArtifact,
    /// Frame in which the map coordinates are expressed.
    pub frame_id: String,
    /// Number of vertices in the map artifact.
    pub vertex_count: u64,
    /// Number of triangles in the map artifact.
    pub triangle_count: u64,
    /// Mesh coordinate bounds.
    pub bounds: MapDiffBounds,
}

impl MapDiffMap {
    /// Creates and validates one receipt-backed map descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        label: impl Into<String>,
        source: StudioSource,
        artifact: ReplayArtifact,
        frame_id: impl Into<String>,
        vertex_count: u64,
        triangle_count: u64,
        bounds: MapDiffBounds,
    ) -> ViewerResult<Self> {
        let map = Self {
            label: label.into(),
            source,
            artifact,
            frame_id: frame_id.into(),
            vertex_count,
            triangle_count,
            bounds,
        };
        map.validate()?;
        Ok(map)
    }

    /// Validates source, artifact, frame, and geometry metadata.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.label.trim().is_empty() || self.frame_id.trim().is_empty() {
            return Err(ViewerError::InvalidState(
                "Map Diff map label and frame ID must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.artifact.validate()?;
        self.bounds.validate()?;
        if self.vertex_count == 0 || self.triangle_count == 0 {
            return Err(ViewerError::InvalidState(
                "Map Diff maps require non-empty vertex and triangle counts".into(),
            ));
        }
        Ok(())
    }
}

/// One spatial heatmap cell in a map comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MapDiffCell {
    /// Zero-based row-major cell index.
    pub index: u32,
    /// Base-map vertices assigned to this cell.
    pub base_vertex_count: u64,
    /// Candidate-map vertices assigned to this cell.
    pub candidate_vertex_count: u64,
    /// Vertices compared by stable vertex index in this cell.
    pub compared_vertex_count: u64,
    /// Compared vertices beyond the configured change threshold.
    pub changed_vertex_count: u64,
    /// Largest displacement in this cell in micrometres.
    pub max_displacement_um: u64,
    /// Mean displacement in this cell in micrometres.
    pub mean_displacement_um: u64,
}

impl MapDiffCell {
    /// Creates and validates one heatmap cell.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        index: u32,
        base_vertex_count: u64,
        candidate_vertex_count: u64,
        compared_vertex_count: u64,
        changed_vertex_count: u64,
        max_displacement_um: u64,
        mean_displacement_um: u64,
    ) -> ViewerResult<Self> {
        let cell = Self {
            index,
            base_vertex_count,
            candidate_vertex_count,
            compared_vertex_count,
            changed_vertex_count,
            max_displacement_um,
            mean_displacement_um,
        };
        cell.validate()?;
        Ok(cell)
    }

    /// Validates count ordering and displacement statistics.
    pub fn validate(&self) -> ViewerResult<()> {
        // Comparisons are anchored to the base position. A moved candidate
        // vertex may land in another spatial cell, so candidate cell counts
        // are not an upper bound for this cell's stable-index comparisons.
        let max_compared = self.base_vertex_count;
        if self.compared_vertex_count > max_compared
            || self.changed_vertex_count > self.compared_vertex_count
            || self.mean_displacement_um > self.max_displacement_um
            || (self.compared_vertex_count == 0
                && (self.changed_vertex_count > 0
                    || self.max_displacement_um > 0
                    || self.mean_displacement_um > 0))
        {
            return Err(ViewerError::InvalidState(
                "Map Diff cell counts or displacement statistics are inconsistent".into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate vertex, topology, and admission metrics for a map comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MapDiffSummary {
    /// Number of base-map vertices.
    pub base_vertex_count: u64,
    /// Number of candidate-map vertices.
    pub candidate_vertex_count: u64,
    /// Number of vertices compared by stable index.
    pub compared_vertex_count: u64,
    /// Candidate-only vertices implied by a larger candidate map.
    pub added_vertex_count: u64,
    /// Base-only vertices implied by a larger base map.
    pub removed_vertex_count: u64,
    /// Compared vertices beyond the configured change threshold.
    pub changed_vertex_count: u64,
    /// Displacement threshold in micrometres.
    pub change_threshold_um: u64,
    /// Maximum compared displacement in micrometres.
    pub max_displacement_um: u64,
    /// Mean compared displacement in micrometres.
    pub mean_displacement_um: u64,
    /// P95 compared displacement in micrometres.
    pub p95_displacement_um: u64,
    /// Number of cells in the row-major heatmap.
    pub cell_count: u64,
    /// Whether the checksums of the two map artifacts are equal.
    pub geometry_hash_equal: bool,
    /// Whether decoded triangle index arrays are equal.
    pub topology_equal: bool,
    /// Whether both maps are bound to the same source checksum.
    pub source_identity_match: bool,
    /// Whether both maps use the same coordinate frame.
    pub frame_identity_match: bool,
    /// Whether a source-bound clock/frame calibration was applied.
    pub calibration_applied: bool,
}

impl MapDiffSummary {
    /// Creates and validates aggregate map-diff metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        base_vertex_count: u64,
        candidate_vertex_count: u64,
        compared_vertex_count: u64,
        added_vertex_count: u64,
        removed_vertex_count: u64,
        changed_vertex_count: u64,
        change_threshold_um: u64,
        max_displacement_um: u64,
        mean_displacement_um: u64,
        p95_displacement_um: u64,
        cell_count: u64,
        geometry_hash_equal: bool,
        topology_equal: bool,
        source_identity_match: bool,
        frame_identity_match: bool,
        calibration_applied: bool,
    ) -> ViewerResult<Self> {
        let summary = Self {
            base_vertex_count,
            candidate_vertex_count,
            compared_vertex_count,
            added_vertex_count,
            removed_vertex_count,
            changed_vertex_count,
            change_threshold_um,
            max_displacement_um,
            mean_displacement_um,
            p95_displacement_um,
            cell_count,
            geometry_hash_equal,
            topology_equal,
            source_identity_match,
            frame_identity_match,
            calibration_applied,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates count, hash, percentile, and calibration invariants.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.change_threshold_um == 0 {
            return Err(ViewerError::InvalidState(
                "Map Diff change threshold must be greater than zero".into(),
            ));
        }
        if self.compared_vertex_count > self.base_vertex_count.min(self.candidate_vertex_count)
            || self.changed_vertex_count > self.compared_vertex_count
            || self.mean_displacement_um > self.max_displacement_um
            || self.p95_displacement_um > self.max_displacement_um
            || (self.compared_vertex_count == 0
                && (self.cell_count > 0
                    || self.changed_vertex_count > 0
                    || self.max_displacement_um > 0
                    || self.mean_displacement_um > 0
                    || self.p95_displacement_um > 0))
        {
            return Err(ViewerError::InvalidState(
                "Map Diff summary counts or displacement statistics are inconsistent".into(),
            ));
        }
        let expected_added = self.candidate_vertex_count.saturating_sub(self.base_vertex_count);
        let expected_removed = self.base_vertex_count.saturating_sub(self.candidate_vertex_count);
        if self.added_vertex_count != expected_added
            || self.removed_vertex_count != expected_removed
        {
            return Err(ViewerError::InvalidState(
                "Map Diff added/removed counts disagree with map vertex counts".into(),
            ));
        }
        if self.geometry_hash_equal
            && (!self.topology_equal
                || self.base_vertex_count != self.candidate_vertex_count
                || self.compared_vertex_count != self.base_vertex_count
                || self.changed_vertex_count > 0
                || self.max_displacement_um > 0
                || self.mean_displacement_um > 0
                || self.p95_displacement_um > 0)
        {
            return Err(ViewerError::InvalidState(
                "equal Map Diff geometry hashes require identical geometry metrics".into(),
            ));
        }
        if self.calibration_applied && (!self.source_identity_match || !self.frame_identity_match) {
            return Err(ViewerError::InvalidState(
                "Map Diff calibration cannot be applied to unbound or mixed-frame maps".into(),
            ));
        }
        Ok(())
    }
}

/// Portable state for a source-bound map comparison dashboard.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct MapDiffState {
    /// Serialized Map Diff state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Base/reference map.
    pub base: MapDiffMap,
    /// Candidate/updated map.
    pub candidate: MapDiffMap,
    /// Aggregate comparison and admission metrics.
    pub summary: MapDiffSummary,
    /// Row-major spatial heatmap cells.
    pub cells: Vec<MapDiffCell>,
    /// Checksummed auxiliary artifacts represented by the state.
    pub artifacts: Vec<ReplayArtifact>,
    /// Whether the source-bound geometry comparison is valid for inspection.
    pub compare_ready: bool,
    /// Whether downstream calibrated mapping is admitted.
    pub mapping_admitted: bool,
    /// Fail-closed reasons for comparison or mapping admission.
    pub blockers: Vec<String>,
}

impl MapDiffState {
    /// Creates a Map Diff state and derives comparison/mapping admission.
    pub fn try_new(
        title: impl Into<String>,
        base: MapDiffMap,
        candidate: MapDiffMap,
        summary: MapDiffSummary,
        cells: Vec<MapDiffCell>,
        artifacts: Vec<ReplayArtifact>,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let compare_ready = summary.source_identity_match
            && summary.frame_identity_match
            && summary.compared_vertex_count > 0
            && summary.cell_count == u64::try_from(cells.len()).unwrap_or(u64::MAX);
        let mapping_admitted = compare_ready && summary.calibration_applied;
        let state = Self {
            version: MAP_DIFF_STATE_VERSION,
            title: title.into(),
            base,
            candidate,
            summary,
            cells,
            artifacts,
            compare_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates source binding, cell ordering, and derived admission flags.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != MAP_DIFF_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported Map Diff state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(ViewerError::InvalidState("Map Diff title must not be empty".into()));
        }
        self.base.validate()?;
        self.candidate.validate()?;
        self.summary.validate()?;

        let source_identity_match = self.base.source.identity_matches
            && self.candidate.source.identity_matches
            && self.base.source.expected_sha256 == self.candidate.source.expected_sha256
            && self.base.source.observed_sha256 == self.candidate.source.observed_sha256;
        if self.summary.source_identity_match != source_identity_match {
            return Err(ViewerError::InvalidState(
                "Map Diff source_identity_match disagrees with map source checksums".into(),
            ));
        }
        let frame_identity_match = self.base.frame_id == self.candidate.frame_id;
        if self.summary.frame_identity_match != frame_identity_match {
            return Err(ViewerError::InvalidState(
                "Map Diff frame_identity_match disagrees with map frame IDs".into(),
            ));
        }

        for (expected_index, cell) in self.cells.iter().enumerate() {
            cell.validate()?;
            if cell.index != u32::try_from(expected_index).unwrap_or(u32::MAX) {
                return Err(ViewerError::InvalidState(
                    "Map Diff cells must use contiguous row-major indices".into(),
                ));
            }
        }
        if self.summary.cell_count != u64::try_from(self.cells.len()).unwrap_or(u64::MAX) {
            return Err(ViewerError::InvalidState(
                "Map Diff cell_count disagrees with the heatmap cell list".into(),
            ));
        }

        let calculated_compare_ready = source_identity_match
            && frame_identity_match
            && self.summary.compared_vertex_count > 0
            && self.summary.cell_count > 0;
        if self.compare_ready != calculated_compare_ready {
            return Err(ViewerError::InvalidState(
                "compare_ready disagrees with source, frame, or geometry admission".into(),
            ));
        }
        let calculated_mapping = self.compare_ready && self.summary.calibration_applied;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with comparison and calibration admission".into(),
            ));
        }

        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "Map Diff artifacts must have unique roles and paths".into(),
                ));
            }
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked Map Diff mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "Map Diff blockers must not contain empty messages".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const OTHER_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

    fn source(observed_sha256: &str) -> StudioSource {
        StudioSource::try_new(
            "canonical bag",
            "/media/input.db3",
            SHA,
            observed_sha256,
            observed_sha256 == SHA,
        )
        .unwrap()
    }

    fn map(label: &str, observed_sha256: &str, artifact_role: &str) -> MapDiffMap {
        MapDiffMap::try_new(
            label,
            source(observed_sha256),
            ReplayArtifact::try_new(artifact_role, format!("/media/{label}.gltf"), 12, SHA)
                .unwrap(),
            "lidar_front",
            2,
            1,
            MapDiffBounds::try_new(0, 0, 0, 1_000_000, 1_000_000, 1_000_000).unwrap(),
        )
        .unwrap()
    }

    fn summary(source_identity_match: bool) -> MapDiffSummary {
        MapDiffSummary::try_new(
            2,
            2,
            if source_identity_match { 2 } else { 0 },
            0,
            0,
            0,
            1_000,
            0,
            0,
            0,
            if source_identity_match { 1 } else { 0 },
            source_identity_match,
            true,
            source_identity_match,
            true,
            false,
        )
        .unwrap()
    }

    fn cell() -> MapDiffCell {
        MapDiffCell::try_new(0, 2, 2, 2, 0, 0, 0).unwrap()
    }

    #[test]
    fn valid_diff_state_roundtrips_with_serde() {
        let state = MapDiffState::try_new(
            "Map Diff",
            map("base", SHA, "base-map"),
            map("candidate", SHA, "candidate-map"),
            summary(true),
            vec![cell()],
            Vec::new(),
            vec!["clock calibration not applied".into()],
        )
        .unwrap();
        assert!(state.compare_ready);
        assert!(!state.mapping_admitted);
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(serde_json::from_str::<MapDiffState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn source_mismatch_cannot_be_compare_ready() {
        let state = MapDiffState::try_new(
            "Map Diff",
            map("base", SHA, "base-map"),
            map("candidate", OTHER_SHA, "candidate-map"),
            summary(false),
            Vec::new(),
            Vec::new(),
            vec!["input SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.compare_ready);
        assert!(!state.mapping_admitted);
    }

    #[test]
    fn changed_cell_rejects_inconsistent_counts() {
        assert!(MapDiffCell::try_new(0, 1, 1, 1, 2, 4, 1).is_err());
        assert!(MapDiffSummary::try_new(
            2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, true, true, true, true, false
        )
        .is_err());
    }
}
