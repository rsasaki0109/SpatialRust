//! Model provenance attached to episode outputs.

/// One model invocation provenance record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelProvenance {
    /// Model name / family.
    pub model: String,
    /// Model revision or digest.
    pub revision: String,
    /// Optional dataset / training run id.
    pub dataset: Option<String>,
}

/// Provenance bag retained on episodes.
pub type ProvenanceRecord = ModelProvenance;
