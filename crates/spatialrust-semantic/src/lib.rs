//! Semantic entities, embeddings, fusion, and nearest-neighbor search.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod embedding;
mod entity;
mod error;
#[cfg(feature = "model")]
mod model;
mod search;

pub use embedding::{cosine_similarity, Embedding};
pub use entity::{record_entity_id, EntityId, OpenVocabLabel, SemanticEntity, SpatialRecordEntity};
pub use error::{SemanticError, SemanticResult};
#[cfg(feature = "model")]
pub use model::OnnxEntityEmbedder;
pub use search::{FusionScore, MultimodalFusion, SemanticSearchIndex};
