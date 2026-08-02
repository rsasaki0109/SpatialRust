//! Source lineage attached to versioned spatial records.

use crate::{RecordsError, RecordsResult};

/// Current version of the record-provenance contract.
pub const RECORD_PROVENANCE_VERSION: u32 = 1;

/// Generic source lineage for one [`crate::SpatialRecord`].
///
/// The contract deliberately stays independent of ROS, MCAP, Arrow, and
/// model runtimes. Adapters may identify their storage in `source_uri` and
/// their logical channel in `stream_id`; chunk-producing adapters can attach a
/// deterministic source sequence without changing the point schema.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "receipt-json", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "receipt-json", serde(deny_unknown_fields))]
pub struct RecordProvenance {
    /// Contract version for this provenance envelope.
    pub version: u32,
    /// Stable logical source identity.
    pub source_id: String,
    /// Optional local path or URI from which the record was read.
    #[cfg_attr(feature = "receipt-json", serde(skip_serializing_if = "Option::is_none"))]
    pub source_uri: Option<String>,
    /// Optional logical stream, topic, or channel within the source.
    #[cfg_attr(feature = "receipt-json", serde(skip_serializing_if = "Option::is_none"))]
    pub stream_id: Option<String>,
    /// Optional deterministic source sequence for this record or chunk.
    #[cfg_attr(feature = "receipt-json", serde(skip_serializing_if = "Option::is_none"))]
    pub sequence: Option<u64>,
}

impl RecordProvenance {
    /// Creates an unknown-source envelope for synthetic or legacy records.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            version: RECORD_PROVENANCE_VERSION,
            source_id: "unknown".to_owned(),
            source_uri: None,
            stream_id: None,
            sequence: None,
        }
    }

    /// Creates a validated envelope with a non-empty source identity.
    pub fn try_new(source_id: impl Into<String>) -> RecordsResult<Self> {
        let source_id = source_id.into();
        let provenance = Self {
            version: RECORD_PROVENANCE_VERSION,
            source_id,
            source_uri: None,
            stream_id: None,
            sequence: None,
        };
        provenance.validate()?;
        Ok(provenance)
    }

    /// Validates the version and required identity fields.
    pub fn validate(&self) -> RecordsResult<()> {
        if self.version != RECORD_PROVENANCE_VERSION {
            return Err(RecordsError::InvalidConfiguration(format!(
                "unsupported record provenance version {}; expected {}",
                self.version, RECORD_PROVENANCE_VERSION
            )));
        }
        if self.source_id.trim().is_empty() {
            return Err(RecordsError::InvalidConfiguration(
                "record provenance source_id must not be empty".into(),
            ));
        }
        if self.source_uri.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(RecordsError::InvalidConfiguration(
                "record provenance source_uri must not be empty".into(),
            ));
        }
        if self.stream_id.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(RecordsError::InvalidConfiguration(
                "record provenance stream_id must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Attaches a source path or URI.
    #[must_use]
    pub fn with_source_uri(mut self, source_uri: impl Into<String>) -> Self {
        self.source_uri = Some(source_uri.into());
        self
    }

    /// Attaches a logical stream, topic, or channel.
    #[must_use]
    pub fn with_stream_id(mut self, stream_id: impl Into<String>) -> Self {
        self.stream_id = Some(stream_id.into());
        self
    }

    /// Attaches a deterministic source sequence.
    #[must_use]
    pub const fn with_sequence(mut self, sequence: Option<u64>) -> Self {
        self.sequence = sequence;
        self
    }

    /// Removes a source sequence when an operation aggregates multiple inputs.
    #[must_use]
    pub const fn without_sequence(mut self) -> Self {
        self.sequence = None;
        self
    }
}

impl Default for RecordProvenance {
    fn default() -> Self {
        Self::unknown()
    }
}

#[cfg(test)]
mod tests {
    use super::{RecordProvenance, RECORD_PROVENANCE_VERSION};

    #[test]
    fn builds_lineage_without_protocol_specific_types() {
        let provenance = RecordProvenance::try_new("bag-01")
            .unwrap()
            .with_source_uri("/media/input/bag.db3")
            .with_stream_id("/lidar/points")
            .with_sequence(Some(4));
        assert_eq!(provenance.version, RECORD_PROVENANCE_VERSION);
        assert_eq!(provenance.source_id, "bag-01");
        assert_eq!(provenance.stream_id.as_deref(), Some("/lidar/points"));
        assert_eq!(provenance.sequence, Some(4));
        assert_eq!(provenance.without_sequence().sequence, None);
    }

    #[test]
    fn rejects_empty_source_identity() {
        assert!(RecordProvenance::try_new(" ").is_err());
    }

    #[test]
    fn rejects_unknown_version_and_empty_optional_identifiers() {
        let mut provenance = RecordProvenance::unknown();
        provenance.version = RECORD_PROVENANCE_VERSION + 1;
        assert!(provenance.validate().is_err());

        let mut provenance = RecordProvenance::unknown();
        provenance.source_uri = Some(String::new());
        assert!(provenance.validate().is_err());
    }
}
