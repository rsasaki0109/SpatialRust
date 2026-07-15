//! Time-ranged annotation layers.

use crate::{EpisodeError, EpisodeResult};

/// Annotation payload type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    /// Bounding / free-form label.
    Label(String),
    /// Numeric scalar annotation.
    Scalar(String),
}

/// Inclusive annotation time span in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnnotationSpan {
    /// Start nanos.
    pub start_ns: u64,
    /// End nanos.
    pub end_ns: u64,
}

impl AnnotationSpan {
    /// Creates a validated span.
    pub fn try_new(start_ns: u64, end_ns: u64) -> EpisodeResult<Self> {
        if end_ns < start_ns {
            return Err(EpisodeError::InvalidConfiguration("annotation span end < start".into()));
        }
        Ok(Self { start_ns, end_ns })
    }
}

/// Named annotation layer over an episode.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationLayer {
    /// Layer name.
    pub name: String,
    /// Annotations.
    pub items: Vec<(AnnotationSpan, AnnotationKind)>,
}

impl AnnotationLayer {
    /// Creates an empty named layer.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), items: Vec::new() }
    }

    /// Pushes an annotation item.
    pub fn push(&mut self, span: AnnotationSpan, kind: AnnotationKind) {
        self.items.push((span, kind));
    }
}
