//! Dense embedding vectors.

use crate::{SemanticError, SemanticResult};

/// Dense float embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    values: Vec<f32>,
}

impl Embedding {
    /// Creates an embedding after rejecting empty/non-finite vectors.
    pub fn try_new(values: Vec<f32>) -> SemanticResult<Self> {
        if values.is_empty() {
            return Err(SemanticError::InvalidConfiguration("embedding must be non-empty".into()));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(SemanticError::InvalidConfiguration("embedding must be finite".into()));
        }
        Ok(Self { values })
    }

    /// Returns dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Borrows values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

/// Cosine similarity in `[-1, 1]`.
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> SemanticResult<f32> {
    if a.dim() != b.dim() {
        return Err(SemanticError::InvalidConfiguration("embedding dims must match".into()));
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return Err(SemanticError::InvalidConfiguration("zero-norm embedding".into()));
    }
    Ok(dot / denom)
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, Embedding};

    #[test]
    fn identical_vectors_have_unit_cosine() {
        let a = Embedding::try_new(vec![1.0, 0.0, 0.0]).unwrap();
        let b = Embedding::try_new(vec![2.0, 0.0, 0.0]).unwrap();
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-5);
    }
}
