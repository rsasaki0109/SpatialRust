//! Portable source-bound state for a glTF/USD digital twin export.
//!
//! The state deliberately models a portable glTF asset plus an ASCII USDA
//! companion layer. It does not pull an OpenUSD runtime into the stable viewer
//! crate and it never turns a missing calibration artifact into a transform.

use std::collections::BTreeSet;

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized Digital Twin state schema version.
pub const DIGITAL_TWIN_STATE_VERSION: u32 = 1;

/// One format-specific asset in a Digital Twin bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DigitalTwinAsset {
    /// Stable asset identifier.
    pub id: String,
    /// Portable format name, currently glTF or USDA.
    pub format: String,
    /// Checksummed file receipt for the asset.
    pub artifact: ReplayArtifact,
    /// Coordinate frame represented by the asset.
    pub frame_id: String,
    /// Number of vertices represented by the asset.
    pub vertex_count: u64,
    /// Number of triangles represented by the asset.
    pub triangle_count: u64,
    /// Whether the asset preserves the source geometry byte-for-byte or by
    /// explicit reference.
    pub identity_preserved: bool,
    /// Human-readable geometry conversion mode.
    pub geometry_mode: String,
}

impl DigitalTwinAsset {
    /// Creates and validates one Digital Twin asset descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        id: impl Into<String>,
        format: impl Into<String>,
        artifact: ReplayArtifact,
        frame_id: impl Into<String>,
        vertex_count: u64,
        triangle_count: u64,
        identity_preserved: bool,
        geometry_mode: impl Into<String>,
    ) -> ViewerResult<Self> {
        let asset = Self {
            id: id.into(),
            format: format.into(),
            artifact,
            frame_id: frame_id.into(),
            vertex_count,
            triangle_count,
            identity_preserved,
            geometry_mode: geometry_mode.into(),
        };
        asset.validate()?;
        Ok(asset)
    }

    /// Validates asset identity, counts, frame, and artifact receipt.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty()
            || self.format.trim().is_empty()
            || self.frame_id.trim().is_empty()
            || self.geometry_mode.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "Digital Twin assets require an ID, format, frame, and geometry mode".into(),
            ));
        }
        if self.vertex_count == 0 || self.triangle_count == 0 {
            return Err(ViewerError::InvalidState(
                "Digital Twin assets require non-empty geometry counts".into(),
            ));
        }
        self.artifact.validate()
    }
}

/// Aggregate identity and geometry metrics for a Digital Twin bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DigitalTwinSummary {
    /// Vertex count in the source mesh.
    pub source_vertex_count: u64,
    /// Triangle count in the source mesh.
    pub source_triangle_count: u64,
    /// Vertex count in the glTF asset.
    pub gltf_vertex_count: u64,
    /// Triangle count in the glTF asset.
    pub gltf_triangle_count: u64,
    /// Vertex count represented by the USDA companion layer.
    pub usd_vertex_count: u64,
    /// Triangle count represented by the USDA companion layer.
    pub usd_triangle_count: u64,
    /// Number of format-specific assets in the bundle.
    pub asset_count: u64,
    /// Whether a source-bound semantic overlay is attached.
    pub semantic_layer_present: bool,
    /// Whether the canonical source checksum matched.
    pub source_identity_match: bool,
    /// Whether the expected and observed frames matched.
    pub frame_identity_match: bool,
    /// Whether geometry identity was preserved without an unapproved transform.
    pub geometry_identity_preserved: bool,
    /// Whether an explicit calibration transform was applied.
    pub calibration_applied: bool,
}

impl DigitalTwinSummary {
    /// Creates and validates aggregate Digital Twin metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        source_vertex_count: u64,
        source_triangle_count: u64,
        gltf_vertex_count: u64,
        gltf_triangle_count: u64,
        usd_vertex_count: u64,
        usd_triangle_count: u64,
        asset_count: u64,
        semantic_layer_present: bool,
        source_identity_match: bool,
        frame_identity_match: bool,
        geometry_identity_preserved: bool,
        calibration_applied: bool,
    ) -> ViewerResult<Self> {
        let summary = Self {
            source_vertex_count,
            source_triangle_count,
            gltf_vertex_count,
            gltf_triangle_count,
            usd_vertex_count,
            usd_triangle_count,
            asset_count,
            semantic_layer_present,
            source_identity_match,
            frame_identity_match,
            geometry_identity_preserved,
            calibration_applied,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates source counts and the no-hidden-transform invariant.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.source_vertex_count == 0 || self.source_triangle_count == 0 {
            return Err(ViewerError::InvalidState(
                "Digital Twin source geometry counts must be non-zero".into(),
            ));
        }
        if self.geometry_identity_preserved
            && (self.gltf_vertex_count != self.source_vertex_count
                || self.gltf_triangle_count != self.source_triangle_count
                || self.usd_vertex_count != self.source_vertex_count
                || self.usd_triangle_count != self.source_triangle_count)
        {
            return Err(ViewerError::InvalidState(
                "identity-preserving Digital Twin geometry must retain source counts".into(),
            ));
        }
        if self.calibration_applied && (!self.source_identity_match || !self.frame_identity_match) {
            return Err(ViewerError::InvalidState(
                "Digital Twin calibration cannot be applied without source and frame identity"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Portable, fail-closed state for the Digital Twin export.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct DigitalTwinState {
    /// Serialized Digital Twin state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Checksummed canonical source identity.
    pub source: StudioSource,
    /// Observed frame represented by the bundle.
    pub frame_id: String,
    /// Frame required by the operation contract.
    pub expected_frame_id: String,
    /// Human-readable timestamp basis from the source receipt.
    pub time_basis: String,
    /// glTF and USDA asset descriptors.
    pub assets: Vec<DigitalTwinAsset>,
    /// Optional source-bound semantic overlay artifact.
    pub semantic_layer: Option<ReplayArtifact>,
    /// Aggregate geometry and admission metrics.
    pub summary: DigitalTwinSummary,
    /// Whether a portable Digital Twin bundle is ready.
    pub twin_ready: bool,
    /// Whether downstream mapping is admitted.
    pub mapping_admitted: bool,
    /// Fail-closed reasons and inspection-only notices.
    pub blockers: Vec<String>,
}

impl DigitalTwinState {
    /// Creates a Digital Twin state and derives its two admission gates.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        frame_id: impl Into<String>,
        expected_frame_id: impl Into<String>,
        time_basis: impl Into<String>,
        assets: Vec<DigitalTwinAsset>,
        semantic_layer: Option<ReplayArtifact>,
        summary: DigitalTwinSummary,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let has_gltf = assets.iter().any(|asset| asset.format == "gltf");
        let has_usda = assets.iter().any(|asset| asset.format == "usda");
        let twin_ready = source.identity_matches
            && summary.source_identity_match
            && summary.frame_identity_match
            && summary.geometry_identity_preserved
            && has_gltf
            && has_usda;
        let mapping_admitted = twin_ready && summary.calibration_applied;
        let state = Self {
            version: DIGITAL_TWIN_STATE_VERSION,
            title: title.into(),
            source,
            frame_id: frame_id.into(),
            expected_frame_id: expected_frame_id.into(),
            time_basis: time_basis.into(),
            assets,
            semantic_layer,
            summary,
            twin_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates asset identity and both fail-closed admission gates.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != DIGITAL_TWIN_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported Digital Twin state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty()
            || self.frame_id.trim().is_empty()
            || self.expected_frame_id.trim().is_empty()
            || self.time_basis.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "Digital Twin title, frames, and time basis must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.summary.validate()?;

        let mut asset_ids = BTreeSet::new();
        let mut asset_formats = BTreeSet::new();
        let mut asset_paths = BTreeSet::new();
        let mut gltf_count = 0_u64;
        let mut usda_count = 0_u64;
        for asset in &self.assets {
            asset.validate()?;
            if !asset_ids.insert(&asset.id)
                || !asset_formats.insert(&asset.format)
                || !asset_paths.insert(&asset.artifact.path)
            {
                return Err(ViewerError::InvalidState(
                    "Digital Twin assets require unique IDs, formats, and paths".into(),
                ));
            }
            if asset.frame_id != self.frame_id {
                return Err(ViewerError::InvalidState(
                    "Digital Twin assets must use the state frame".into(),
                ));
            }
            match asset.format.as_str() {
                "gltf" => gltf_count = gltf_count.saturating_add(1),
                "usda" => usda_count = usda_count.saturating_add(1),
                _ => {
                    return Err(ViewerError::InvalidState(
                        "Digital Twin asset format must be gltf or usda".into(),
                    ));
                }
            }
            if asset.identity_preserved
                && (asset.vertex_count != self.summary.source_vertex_count
                    || asset.triangle_count != self.summary.source_triangle_count)
            {
                return Err(ViewerError::InvalidState(
                    "identity-preserving Digital Twin asset counts must match the source".into(),
                ));
            }
        }
        if self.summary.asset_count != u64::try_from(self.assets.len()).unwrap_or(u64::MAX) {
            return Err(ViewerError::InvalidState(
                "Digital Twin summary asset count disagrees with assets".into(),
            ));
        }
        if gltf_count > 1 || usda_count > 1 {
            return Err(ViewerError::InvalidState(
                "Digital Twin state permits at most one glTF and one USDA asset".into(),
            ));
        }

        if let Some(semantic_layer) = &self.semantic_layer {
            semantic_layer.validate()?;
            if !asset_paths.insert(&semantic_layer.path) {
                return Err(ViewerError::InvalidState(
                    "semantic Digital Twin layer path overlaps an asset path".into(),
                ));
            }
        }
        if self.summary.semantic_layer_present != self.semantic_layer.is_some() {
            return Err(ViewerError::InvalidState(
                "Digital Twin semantic layer flag disagrees with the artifact".into(),
            ));
        }

        let calculated_twin_ready = self.source.identity_matches
            && self.summary.source_identity_match
            && self.summary.frame_identity_match
            && self.summary.geometry_identity_preserved
            && gltf_count == 1
            && usda_count == 1;
        if self.twin_ready != calculated_twin_ready {
            return Err(ViewerError::InvalidState(
                "twin_ready disagrees with source, frame, geometry, or asset admission".into(),
            ));
        }
        let calculated_mapping = self.twin_ready && self.summary.calibration_applied;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with Digital Twin and calibration admission".into(),
            ));
        }
        if self.mapping_admitted && !self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "admitted Digital Twin mapping cannot contain blockers".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked Digital Twin mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "Digital Twin blockers must not contain empty messages".into(),
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

    fn source(observed: &str, identity_matches: bool) -> StudioSource {
        StudioSource::try_new("canonical bag", "/media/input.db3", SHA, observed, identity_matches)
            .unwrap()
    }

    fn artifact(role: &str) -> ReplayArtifact {
        ReplayArtifact::try_new(format!("{role}-asset"), format!("/media/{role}"), 128, SHA)
            .unwrap()
    }

    fn asset(id: &str, format: &str) -> DigitalTwinAsset {
        DigitalTwinAsset::try_new(
            id,
            format,
            artifact(format),
            "lidar_front",
            10,
            4,
            true,
            "identity-preserving",
        )
        .unwrap()
    }

    fn summary(source_identity_match: bool, frame_identity_match: bool) -> DigitalTwinSummary {
        DigitalTwinSummary::try_new(
            10,
            4,
            if source_identity_match { 10 } else { 0 },
            if source_identity_match { 4 } else { 0 },
            if source_identity_match { 10 } else { 0 },
            if source_identity_match { 4 } else { 0 },
            if source_identity_match { 2 } else { 0 },
            false,
            source_identity_match,
            frame_identity_match,
            source_identity_match,
            false,
        )
        .unwrap()
    }

    #[test]
    fn valid_bundle_is_ready_but_mapping_stays_blocked() {
        let state = DigitalTwinState::try_new(
            "Digital Twin",
            source(SHA, true),
            "lidar_front",
            "lidar_front",
            "PointCloud2 header stamp; no clock calibration applied",
            vec![asset("mesh-gltf", "gltf"), asset("mesh-usda", "usda")],
            None,
            summary(true, true),
            vec!["clock and TF calibration are not applied".into()],
        )
        .unwrap();
        assert!(state.twin_ready);
        assert!(!state.mapping_admitted);
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(serde_json::from_str::<DigitalTwinState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn source_mismatch_withholds_the_bundle() {
        let state = DigitalTwinState::try_new(
            "Digital Twin",
            source(OTHER_SHA, false),
            "lidar_front",
            "lidar_front",
            "header stamp",
            Vec::new(),
            None,
            summary(false, true),
            vec!["input SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.twin_ready);
        assert!(!state.mapping_admitted);
    }
}
