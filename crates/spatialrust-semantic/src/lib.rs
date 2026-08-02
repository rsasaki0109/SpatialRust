//! Semantic entities, embeddings, fusion, and nearest-neighbor search.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod embedding;
mod entity;
mod error;
mod search;

pub use embedding::{cosine_similarity, Embedding};
pub use entity::{EntityId, OpenVocabLabel, SemanticEntity};
pub use error::{SemanticError, SemanticResult};
pub use search::{FusionScore, MultimodalFusion, SemanticSearchIndex};
