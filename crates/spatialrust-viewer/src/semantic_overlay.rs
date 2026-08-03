//! Portable, source-bound AI semantic overlay state.
//!
//! The state stores quantized coordinates and confidence values so native,
//! Web, and headless consumers share one deterministic contract. Model
//! execution and renderer uploads remain outside this module; adapters must
//! provide explicit model and artifact receipts.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ReplayArtifact, StudioSource, ViewerError, ViewerResult};

/// Current serialized AI semantic overlay state schema version.
pub const SEMANTIC_OVERLAY_STATE_VERSION: u32 = 1;

/// Confidence quantization scale used by semantic overlay receipts.
pub const SEMANTIC_CONFIDENCE_SCALE: u32 = 1_000_000;

/// Model and explicit host/device transfer receipt for one overlay run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct SemanticOverlayModel {
    /// Stable model/profile identifier.
    pub model_id: String,
    /// Backend identifier such as `mock` or `onnxruntime-cpu`.
    pub backend: String,
    /// Human-readable runtime/provenance note.
    pub runtime: String,
    /// Whether the model request was configured for deterministic execution.
    pub deterministic: bool,
    /// Number of input feature channels supplied per point.
    pub input_feature_count: u32,
    /// Number of class IDs declared by the model output contract.
    pub output_class_count: u32,
    /// Host bytes supplied to the model.
    pub input_host_bytes: u64,
    /// Host bytes returned by the model.
    pub output_host_bytes: u64,
    /// Explicit host-to-device bytes recorded for this run.
    pub device_upload_bytes: u64,
    /// Explicit device-to-host bytes recorded for this run.
    pub device_readback_bytes: u64,
}

impl SemanticOverlayModel {
    /// Creates and validates model metadata and transfer accounting.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        model_id: impl Into<String>,
        backend: impl Into<String>,
        runtime: impl Into<String>,
        deterministic: bool,
        input_feature_count: u32,
        output_class_count: u32,
        input_host_bytes: u64,
        output_host_bytes: u64,
        device_upload_bytes: u64,
        device_readback_bytes: u64,
    ) -> ViewerResult<Self> {
        let model = Self {
            model_id: model_id.into(),
            backend: backend.into(),
            runtime: runtime.into(),
            deterministic,
            input_feature_count,
            output_class_count,
            input_host_bytes,
            output_host_bytes,
            device_upload_bytes,
            device_readback_bytes,
        };
        model.validate()?;
        Ok(model)
    }

    /// Validates model identity, dimensions, and non-hidden transfer fields.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.model_id.trim().is_empty()
            || self.backend.trim().is_empty()
            || self.runtime.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "semantic overlay model identity and runtime must not be empty".into(),
            ));
        }
        if self.input_feature_count == 0 || self.output_class_count == 0 {
            return Err(ViewerError::InvalidState(
                "semantic overlay model dimensions must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// One class represented in the semantic overlay legend.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct SemanticOverlayClass {
    /// Stable model output class ID.
    pub class_id: u32,
    /// Human-readable class label.
    pub label: String,
    /// sRGB color used by renderer and dashboard.
    pub color_rgb: [u8; 3],
    /// Number of sampled entities assigned to this class.
    pub entity_count: u64,
    /// Mean confidence in millionths.
    pub mean_confidence_million: u32,
    /// Maximum confidence in millionths.
    pub max_confidence_million: u32,
}

impl SemanticOverlayClass {
    /// Creates and validates one class legend entry.
    pub fn try_new(
        class_id: u32,
        label: impl Into<String>,
        color_rgb: [u8; 3],
        entity_count: u64,
        mean_confidence_million: u32,
        max_confidence_million: u32,
    ) -> ViewerResult<Self> {
        let class = Self {
            class_id,
            label: label.into(),
            color_rgb,
            entity_count,
            mean_confidence_million,
            max_confidence_million,
        };
        class.validate()?;
        Ok(class)
    }

    /// Validates label and confidence statistics.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.label.trim().is_empty()
            || self.mean_confidence_million > SEMANTIC_CONFIDENCE_SCALE
            || self.max_confidence_million > SEMANTIC_CONFIDENCE_SCALE
            || self.mean_confidence_million > self.max_confidence_million
            || (self.entity_count == 0
                && (self.mean_confidence_million > 0 || self.max_confidence_million > 0))
        {
            return Err(ViewerError::InvalidState(
                "semantic overlay class label or confidence statistics are invalid".into(),
            ));
        }
        Ok(())
    }
}

/// One source-indexed semantic prediction rendered in the overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct SemanticOverlayEntity {
    /// Stable entity identifier from the overlay run.
    pub id: String,
    /// Index of the source point or feature row consumed by the model.
    pub source_index: u64,
    /// Model output class ID.
    pub class_id: u32,
    /// Resolved class label.
    pub label: String,
    /// Prediction confidence in millionths.
    pub confidence_million: u32,
    /// Centroid/point coordinate in the declared frame, in micrometres.
    pub centroid_um: [i64; 3],
}

impl SemanticOverlayEntity {
    /// Creates and validates one quantized semantic prediction.
    pub fn try_new(
        id: impl Into<String>,
        source_index: u64,
        class_id: u32,
        label: impl Into<String>,
        confidence_million: u32,
        centroid_um: [i64; 3],
    ) -> ViewerResult<Self> {
        let entity = Self {
            id: id.into(),
            source_index,
            class_id,
            label: label.into(),
            confidence_million,
            centroid_um,
        };
        entity.validate()?;
        Ok(entity)
    }

    /// Validates stable identity, coordinates, and confidence quantization.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.id.trim().is_empty()
            || self.label.trim().is_empty()
            || self.confidence_million > SEMANTIC_CONFIDENCE_SCALE
        {
            return Err(ViewerError::InvalidState(
                "semantic overlay entity identity, label, or confidence is invalid".into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate quality and source-binding metrics for an overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct SemanticOverlaySummary {
    /// Number of points/features available from the checked source artifact.
    pub input_point_count: u64,
    /// Number of points sampled into the model input.
    pub sampled_point_count: u64,
    /// Number of semantic predictions emitted by the model.
    pub entity_count: u64,
    /// Number of predictions with visible finite coordinates.
    pub visible_entity_count: u64,
    /// Number of class legend entries represented by predictions.
    pub class_count: u64,
    /// Mean prediction confidence in millionths.
    pub mean_confidence_million: u32,
    /// P95 prediction confidence in millionths.
    pub p95_confidence_million: u32,
    /// Sampled/input coverage in millionths.
    pub coverage_million: u32,
    /// Whether the checked source identity matched the operation contract.
    pub source_identity_match: bool,
    /// Whether the checked map frame matched the requested overlay frame.
    pub frame_identity_match: bool,
    /// Whether source-bound clock/frame calibration was applied.
    pub calibration_applied: bool,
}

impl SemanticOverlaySummary {
    /// Creates and validates aggregate overlay metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        input_point_count: u64,
        sampled_point_count: u64,
        entity_count: u64,
        visible_entity_count: u64,
        class_count: u64,
        mean_confidence_million: u32,
        p95_confidence_million: u32,
        coverage_million: u32,
        source_identity_match: bool,
        frame_identity_match: bool,
        calibration_applied: bool,
    ) -> ViewerResult<Self> {
        let summary = Self {
            input_point_count,
            sampled_point_count,
            entity_count,
            visible_entity_count,
            class_count,
            mean_confidence_million,
            p95_confidence_million,
            coverage_million,
            source_identity_match,
            frame_identity_match,
            calibration_applied,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates count, confidence, coverage, and calibration invariants.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.sampled_point_count > self.input_point_count
            || self.entity_count > self.sampled_point_count
            || self.visible_entity_count > self.entity_count
            || self.mean_confidence_million > SEMANTIC_CONFIDENCE_SCALE
            || self.p95_confidence_million > SEMANTIC_CONFIDENCE_SCALE
            || self.coverage_million > SEMANTIC_CONFIDENCE_SCALE
            || (self.entity_count == 0
                && (self.class_count > 0
                    || self.visible_entity_count > 0
                    || self.mean_confidence_million > 0
                    || self.p95_confidence_million > 0))
            || (self.input_point_count == 0
                && (self.sampled_point_count > 0 || self.coverage_million > 0))
            || (self.sampled_point_count == 0 && self.coverage_million > 0)
        {
            return Err(ViewerError::InvalidState(
                "semantic overlay summary counts or confidence metrics are invalid".into(),
            ));
        }
        if self.calibration_applied && (!self.source_identity_match || !self.frame_identity_match) {
            return Err(ViewerError::InvalidState(
                "semantic overlay calibration cannot be applied to an unbound or mixed-frame source"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Portable dashboard state for an AI semantic overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct SemanticOverlayState {
    /// Serialized state schema version.
    pub version: u32,
    /// User-facing dashboard title.
    pub title: String,
    /// Checksummed source identity consumed by the model adapter.
    pub source: StudioSource,
    /// Coordinate frame of all overlay entities.
    pub frame_id: String,
    /// Frame requested by the operation contract.
    pub expected_frame_id: String,
    /// Timestamp basis associated with the source artifact.
    pub time_basis: String,
    /// Explicit model/runtime and transfer receipt.
    pub model: SemanticOverlayModel,
    /// Class legend and aggregate counts.
    pub classes: Vec<SemanticOverlayClass>,
    /// Quantized predictions rendered by the dashboard/adapter.
    pub entities: Vec<SemanticOverlayEntity>,
    /// Checksummed input/output artifacts represented by this state.
    pub artifacts: Vec<ReplayArtifact>,
    /// Aggregate semantic quality and source-binding metrics.
    pub summary: SemanticOverlaySummary,
    /// Whether the source-bound semantic overlay is ready for inspection.
    pub overlay_ready: bool,
    /// Whether downstream calibrated mapping is admitted.
    pub mapping_admitted: bool,
    /// Fail-closed reasons for overlay or mapping admission.
    pub blockers: Vec<String>,
}

impl SemanticOverlayState {
    /// Creates a state and derives overlay/mapping admission flags.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        title: impl Into<String>,
        source: StudioSource,
        frame_id: impl Into<String>,
        expected_frame_id: impl Into<String>,
        time_basis: impl Into<String>,
        model: SemanticOverlayModel,
        classes: Vec<SemanticOverlayClass>,
        entities: Vec<SemanticOverlayEntity>,
        artifacts: Vec<ReplayArtifact>,
        summary: SemanticOverlaySummary,
        blockers: Vec<String>,
    ) -> ViewerResult<Self> {
        let overlay_ready = source.identity_matches
            && summary.source_identity_match
            && summary.frame_identity_match
            && summary.entity_count > 0
            && summary.entity_count == u64::try_from(entities.len()).unwrap_or(u64::MAX);
        let mapping_admitted = overlay_ready && summary.calibration_applied;
        let state = Self {
            version: SEMANTIC_OVERLAY_STATE_VERSION,
            title: title.into(),
            source,
            frame_id: frame_id.into(),
            expected_frame_id: expected_frame_id.into(),
            time_basis: time_basis.into(),
            model,
            classes,
            entities,
            artifacts,
            summary,
            overlay_ready,
            mapping_admitted,
            blockers,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validates all source, class, entity, artifact, and admission invariants.
    pub fn validate(&self) -> ViewerResult<()> {
        if self.version != SEMANTIC_OVERLAY_STATE_VERSION {
            return Err(ViewerError::InvalidState(format!(
                "unsupported semantic overlay state version {}",
                self.version
            )));
        }
        if self.title.trim().is_empty()
            || self.frame_id.trim().is_empty()
            || self.expected_frame_id.trim().is_empty()
            || self.time_basis.trim().is_empty()
        {
            return Err(ViewerError::InvalidState(
                "semantic overlay title, frame, and time basis must not be empty".into(),
            ));
        }
        self.source.validate()?;
        self.model.validate()?;
        self.summary.validate()?;
        if self.summary.source_identity_match != self.source.identity_matches {
            return Err(ViewerError::InvalidState(
                "semantic overlay source identity disagrees with the checked source".into(),
            ));
        }
        if self.summary.frame_identity_match != (self.frame_id == self.expected_frame_id) {
            return Err(ViewerError::InvalidState(
                "semantic overlay frame identity disagrees with frame fields".into(),
            ));
        }

        let mut class_ids = BTreeSet::new();
        let mut class_labels = BTreeSet::new();
        for class in &self.classes {
            class.validate()?;
            if !class_ids.insert(class.class_id) || !class_labels.insert(&class.label) {
                return Err(ViewerError::InvalidState(
                    "semantic overlay classes must have unique IDs and labels".into(),
                ));
            }
        }
        let class_by_id =
            self.classes.iter().map(|class| (class.class_id, class)).collect::<BTreeMap<_, _>>();
        let mut entity_ids = BTreeSet::new();
        let mut source_indices = BTreeSet::new();
        let mut counts = BTreeMap::<u32, (u64, u64, u32)>::new();
        let mut confidence = Vec::with_capacity(self.entities.len());
        for entity in &self.entities {
            entity.validate()?;
            if !entity_ids.insert(&entity.id) || !source_indices.insert(entity.source_index) {
                return Err(ViewerError::InvalidState(
                    "semantic overlay entity IDs and source indices must be unique".into(),
                ));
            }
            let class = class_by_id.get(&entity.class_id).ok_or_else(|| {
                ViewerError::InvalidState(
                    "semantic overlay entity references an unknown class".into(),
                )
            })?;
            if class.label != entity.label {
                return Err(ViewerError::InvalidState(
                    "semantic overlay entity label disagrees with its class legend".into(),
                ));
            }
            let entry = counts.entry(entity.class_id).or_default();
            entry.0 = entry.0.checked_add(1).ok_or_else(|| {
                ViewerError::InvalidState("semantic overlay class count overflow".into())
            })?;
            entry.1 =
                entry.1.checked_add(u64::from(entity.confidence_million)).ok_or_else(|| {
                    ViewerError::InvalidState("semantic overlay confidence sum overflow".into())
                })?;
            entry.2 = entry.2.max(entity.confidence_million);
            confidence.push(entity.confidence_million);
        }
        for class in &self.classes {
            let (count, sum, max) = counts.get(&class.class_id).copied().unwrap_or_default();
            if class.entity_count != count
                || class.max_confidence_million != max
                || (count > 0 && class.mean_confidence_million != (sum / count) as u32)
            {
                return Err(ViewerError::InvalidState(
                    "semantic overlay class statistics disagree with entities".into(),
                ));
            }
        }
        confidence.sort_unstable();
        let entity_count = u64::try_from(self.entities.len()).unwrap_or(u64::MAX);
        let visible_count = entity_count;
        let class_count = u64::try_from(counts.len()).unwrap_or(u64::MAX);
        let confidence_sum = confidence.iter().try_fold(0_u64, |sum, value| {
            sum.checked_add(u64::from(*value)).ok_or_else(|| {
                ViewerError::InvalidState("semantic overlay confidence sum overflow".into())
            })
        })?;
        let expected_mean = u32::try_from(confidence_sum.checked_div(entity_count).unwrap_or(0))
            .unwrap_or(u32::MAX);
        let expected_p95 = if confidence.is_empty() {
            0
        } else {
            confidence[confidence.len().saturating_mul(95).div_ceil(100).saturating_sub(1)]
        };
        let expected_coverage = if self.summary.input_point_count == 0 {
            0
        } else {
            u32::try_from(
                (u128::from(self.summary.sampled_point_count)
                    * u128::from(SEMANTIC_CONFIDENCE_SCALE)
                    / u128::from(self.summary.input_point_count))
                .min(u128::from(SEMANTIC_CONFIDENCE_SCALE)),
            )
            .unwrap_or(u32::MAX)
        };
        if self.summary.entity_count != entity_count
            || self.summary.visible_entity_count != visible_count
            || self.summary.class_count != class_count
            || self.summary.mean_confidence_million != expected_mean
            || self.summary.p95_confidence_million != expected_p95
            || self.summary.coverage_million != expected_coverage
        {
            return Err(ViewerError::InvalidState(
                "semantic overlay summary disagrees with class/entity data".into(),
            ));
        }

        let mut artifact_roles = BTreeSet::new();
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            artifact.validate()?;
            if !artifact_roles.insert(&artifact.role) || !artifact_paths.insert(&artifact.path) {
                return Err(ViewerError::InvalidState(
                    "semantic overlay artifacts must have unique roles and paths".into(),
                ));
            }
        }
        let calculated_ready = self.source.identity_matches
            && self.summary.source_identity_match
            && self.summary.frame_identity_match
            && self.summary.entity_count > 0;
        if self.overlay_ready != calculated_ready {
            return Err(ViewerError::InvalidState(
                "overlay_ready disagrees with source, frame, or prediction admission".into(),
            ));
        }
        let calculated_mapping = self.overlay_ready && self.summary.calibration_applied;
        if self.mapping_admitted != calculated_mapping {
            return Err(ViewerError::InvalidState(
                "mapping_admitted disagrees with semantic overlay calibration admission".into(),
            ));
        }
        if !self.mapping_admitted && self.blockers.is_empty() {
            return Err(ViewerError::InvalidState(
                "blocked semantic overlay mapping must expose at least one blocker".into(),
            ));
        }
        if self.blockers.iter().any(|blocker| blocker.trim().is_empty()) {
            return Err(ViewerError::InvalidState(
                "semantic overlay blockers must not contain empty messages".into(),
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

    fn source(observed: &str) -> StudioSource {
        StudioSource::try_new("canonical", "/media/input.db3", SHA, observed, observed == SHA)
            .unwrap()
    }

    fn model() -> SemanticOverlayModel {
        SemanticOverlayModel::try_new(
            "mock-semantic-classes",
            "mock",
            "deterministic test profile",
            true,
            4,
            3,
            48,
            24,
            0,
            0,
        )
        .unwrap()
    }

    fn classes() -> Vec<SemanticOverlayClass> {
        vec![SemanticOverlayClass::try_new(0, "ground", [54, 211, 153], 1, 800_000, 800_000)
            .unwrap()]
    }

    fn entities() -> Vec<SemanticOverlayEntity> {
        vec![SemanticOverlayEntity::try_new(
            "semantic:0",
            0,
            0,
            "ground",
            800_000,
            [1_000_000, 2_000_000, 3_000_000],
        )
        .unwrap()]
    }

    fn summary(source_match: bool, frame_match: bool) -> SemanticOverlaySummary {
        SemanticOverlaySummary::try_new(
            10,
            1,
            if source_match && frame_match { 1 } else { 0 },
            if source_match && frame_match { 1 } else { 0 },
            if source_match && frame_match { 1 } else { 0 },
            if source_match && frame_match { 800_000 } else { 0 },
            if source_match && frame_match { 800_000 } else { 0 },
            100_000,
            source_match,
            frame_match,
            false,
        )
        .unwrap()
    }

    #[test]
    fn valid_state_roundtrips_with_serde() {
        let state = SemanticOverlayState::try_new(
            "AI Semantic Overlay",
            source(SHA),
            "lidar_front",
            "lidar_front",
            "PointCloud2 header stamp",
            model(),
            classes(),
            entities(),
            Vec::new(),
            summary(true, true),
            vec!["clock calibration not applied".into()],
        )
        .unwrap();
        assert!(state.overlay_ready);
        assert!(!state.mapping_admitted);
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(serde_json::from_str::<SemanticOverlayState>(&json).unwrap(), state);
        }
    }

    #[test]
    fn source_mismatch_withholds_overlay() {
        let state = SemanticOverlayState::try_new(
            "AI Semantic Overlay",
            source(OTHER_SHA),
            "lidar_front",
            "lidar_front",
            "PointCloud2 header stamp",
            model(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            SemanticOverlaySummary::try_new(10, 0, 0, 0, 0, 0, 0, 0, false, true, false).unwrap(),
            vec!["input SHA-256 mismatch".into()],
        )
        .unwrap();
        assert!(!state.overlay_ready);
        assert!(!state.mapping_admitted);
    }

    #[test]
    fn entity_label_and_confidence_are_fail_closed() {
        assert!(SemanticOverlayEntity::try_new("", 0, 0, "ground", 1, [0, 0, 0]).is_err());
        assert!(SemanticOverlayEntity::try_new(
            "semantic:0",
            0,
            0,
            "ground",
            SEMANTIC_CONFIDENCE_SCALE + 1,
            [0, 0, 0],
        )
        .is_err());
    }
}
