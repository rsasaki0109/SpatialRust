//! Semantic spatial entities.

use spatialrust_math::Vec3;

use crate::Embedding;

/// Stable entity identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(pub String);

impl EntityId {
    /// Creates an entity id.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// Open-vocabulary label with confidence.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenVocabLabel {
    /// Free-form text label.
    pub text: String,
    /// Confidence in `[0, 1]`.
    pub confidence: f32,
}

/// One semantic spatial entity with optional embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticEntity {
    /// Entity id.
    pub id: EntityId,
    /// Optional centroid.
    pub centroid: Option<Vec3<f32>>,
    /// Open-vocabulary labels.
    pub labels: Vec<OpenVocabLabel>,
    /// Optional embedding for search/fusion.
    pub embedding: Option<Embedding>,
}
