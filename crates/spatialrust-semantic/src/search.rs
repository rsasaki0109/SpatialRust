//! Multimodal fusion scoring and brute-force semantic search.

use crate::{
    cosine_similarity, Embedding, EntityId, SemanticEntity, SemanticError, SemanticResult,
};

/// Weighted fusion of embedding similarity and label confidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionScore {
    /// Final fused score.
    pub score: f32,
    /// Embedding similarity contribution.
    pub embedding: f32,
    /// Best label confidence contribution.
    pub label: f32,
}

/// Multimodal fusion weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultimodalFusion {
    /// Weight for embedding cosine.
    pub embedding_weight: f32,
    /// Weight for best open-vocab confidence.
    pub label_weight: f32,
}

impl Default for MultimodalFusion {
    fn default() -> Self {
        Self { embedding_weight: 0.7, label_weight: 0.3 }
    }
}

impl MultimodalFusion {
    /// Scores one entity against a query embedding.
    pub fn score(&self, entity: &SemanticEntity, query: &Embedding) -> SemanticResult<FusionScore> {
        let embedding = match &entity.embedding {
            Some(values) => cosine_similarity(values, query)?,
            None => 0.0,
        };
        let label = entity.labels.iter().map(|l| l.confidence).fold(0.0_f32, f32::max);
        let score = self.embedding_weight * embedding + self.label_weight * label;
        Ok(FusionScore { score, embedding, label })
    }
}

/// In-memory nearest-neighbor index over semantic entities.
#[derive(Clone, Debug, Default)]
pub struct SemanticSearchIndex {
    entities: Vec<SemanticEntity>,
}

impl SemanticSearchIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an entity.
    pub fn insert(&mut self, entity: SemanticEntity) {
        self.entities.push(entity);
    }

    /// Returns top-k entity ids by fused score.
    pub fn search(
        &self,
        query: &Embedding,
        fusion: MultimodalFusion,
        k: usize,
    ) -> SemanticResult<Vec<(EntityId, FusionScore)>> {
        if k == 0 {
            return Err(SemanticError::InvalidConfiguration("k must be positive".into()));
        }
        let mut scored = self
            .entities
            .iter()
            .map(|entity| fusion.score(entity, query).map(|score| (entity.id.clone(), score)))
            .collect::<SemanticResult<Vec<_>>>()?;
        scored
            .sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::{MultimodalFusion, SemanticSearchIndex};
    use crate::{Embedding, EntityId, OpenVocabLabel, SemanticEntity};

    #[test]
    fn ranks_entities_by_fusion() {
        let mut index = SemanticSearchIndex::new();
        index.insert(SemanticEntity {
            id: EntityId::new("a"),
            centroid: None,
            labels: vec![OpenVocabLabel { text: "chair".into(), confidence: 0.2 }],
            embedding: Some(Embedding::try_new(vec![1.0, 0.0]).unwrap()),
        });
        index.insert(SemanticEntity {
            id: EntityId::new("b"),
            centroid: None,
            labels: vec![OpenVocabLabel { text: "table".into(), confidence: 0.9 }],
            embedding: Some(Embedding::try_new(vec![0.0, 1.0]).unwrap()),
        });
        let hits = index
            .search(&Embedding::try_new(vec![1.0, 0.0]).unwrap(), MultimodalFusion::default(), 1)
            .unwrap();
        assert_eq!(hits[0].0, EntityId::new("a"));
    }
}
