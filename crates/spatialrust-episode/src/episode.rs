//! Episode containers over synchronized multimodal memory episodes.

use spatialrust_sync::MemoryEpisode;

use crate::{AnnotationLayer, EpisodeError, EpisodeResult, ModelProvenance};

/// Stable episode identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EpisodeId(pub String);

impl EpisodeId {
    /// Creates an episode id.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// One embodied-AI episode with optional annotations and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    /// Episode id.
    pub id: EpisodeId,
    /// Deterministic multimodal payload.
    pub memory: MemoryEpisode,
    /// Annotation layers.
    pub annotations: Vec<AnnotationLayer>,
    /// Model provenance records.
    pub provenance: Vec<ModelProvenance>,
}

impl Episode {
    /// Creates an episode with empty annotation/provenance lists.
    pub fn try_new(id: impl Into<EpisodeId>, memory: MemoryEpisode) -> EpisodeResult<Self> {
        let id = id.into();
        if id.0.is_empty() {
            return Err(EpisodeError::InvalidConfiguration("episode id must be non-empty".into()));
        }
        Ok(Self { id, memory, annotations: Vec::new(), provenance: Vec::new() })
    }
}

impl From<&str> for EpisodeId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for EpisodeId {
    fn from(value: String) -> Self {
        Self(value)
    }
}
