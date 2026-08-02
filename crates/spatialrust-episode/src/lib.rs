//! Embodied-AI episodes, annotations, augmentation, and evaluation.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod annotate;
mod augment;
mod episode;
mod error;
mod eval;
mod provenance;

pub use annotate::{AnnotationKind, AnnotationLayer, AnnotationSpan};
pub use augment::{AugmentationOp, EpisodeAugmentor};
pub use episode::{Episode, EpisodeId};
pub use error::{EpisodeError, EpisodeResult};
pub use eval::{EvalMetric, EvalReport};
pub use provenance::{ModelProvenance, ProvenanceRecord};
