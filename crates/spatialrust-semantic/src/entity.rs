//! Semantic spatial entities.

use spatialrust_core::{FrameId, HasPositions3, Timestamp};
use spatialrust_math::Vec3;
use spatialrust_records::{RecordProvenance, SpatialRecord};

use crate::{Embedding, SemanticError, SemanticResult};

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

/// A semantic entity derived from one versioned spatial record.
///
/// The embedded [`SemanticEntity`] remains compatible with the existing
/// search index, while this wrapper keeps the record's lineage and spatial
/// metadata alongside it. Model runtimes are intentionally not involved;
/// callers may supply an embedding from any explicit adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialRecordEntity {
    /// Search/index payload.
    pub entity: SemanticEntity,
    /// Source lineage copied from the input record.
    pub provenance: RecordProvenance,
    /// Coordinate frame of the centroid.
    pub frame_id: FrameId,
    /// Observation timestamp copied from the input record.
    pub timestamp: Timestamp,
}

impl SpatialRecordEntity {
    /// Builds a deterministic semantic entity from a spatial record.
    pub fn try_from_record(
        record: &SpatialRecord,
        labels: Vec<OpenVocabLabel>,
        embedding: Option<Embedding>,
    ) -> SemanticResult<Self> {
        record
            .provenance()
            .validate()
            .map_err(|error| SemanticError::InvalidConfiguration(error.to_string()))?;
        validate_labels(&labels)?;
        let centroid = centroid(record)?;
        let provenance = record.provenance().clone();
        let id = record_entity_id(&provenance);
        Ok(Self {
            entity: SemanticEntity { id, centroid, labels, embedding },
            provenance,
            frame_id: record.metadata().frame_id.clone(),
            timestamp: record.metadata().timestamp,
        })
    }

    /// Returns the search/index payload without losing the lineage wrapper.
    #[must_use]
    pub fn entity(&self) -> &SemanticEntity {
        &self.entity
    }
}

/// Derives a stable entity id from a record's protocol-independent lineage.
#[must_use]
pub fn record_entity_id(provenance: &RecordProvenance) -> EntityId {
    let stream = provenance.stream_id.as_deref().unwrap_or("record");
    let sequence =
        provenance.sequence.map_or_else(|| "unsequenced".to_owned(), |value| value.to_string());
    EntityId::new(format!("record:{}:{stream}:{sequence}", provenance.source_id))
}

fn centroid(record: &SpatialRecord) -> SemanticResult<Option<Vec3<f32>>> {
    let cloud = record.cloud();
    if cloud.is_empty() {
        return Ok(None);
    }
    let (x, y, z) = cloud.positions3()?;
    let mut sum = [0.0_f64; 3];
    for index in 0..cloud.len() {
        let values = [x[index], y[index], z[index]];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SemanticError::InvalidConfiguration(
                "record positions must be finite for semantic centroid".into(),
            ));
        }
        for axis in 0..3 {
            sum[axis] += f64::from(values[axis]);
        }
    }
    let count = cloud.len() as f64;
    let values = [sum[0] / count, sum[1] / count, sum[2] / count];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SemanticError::InvalidConfiguration("record centroid is not finite".into()));
    }
    Ok(Some(Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32)))
}

fn validate_labels(labels: &[OpenVocabLabel]) -> SemanticResult<()> {
    for label in labels {
        if label.text.trim().is_empty() {
            return Err(SemanticError::InvalidConfiguration(
                "semantic labels must be non-empty".into(),
            ));
        }
        if !label.confidence.is_finite() || !(0.0..=1.0).contains(&label.confidence) {
            return Err(SemanticError::InvalidConfiguration(
                "semantic label confidence must be finite in [0, 1]".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{record_entity_id, OpenVocabLabel, SpatialRecordEntity};
    use crate::Embedding;
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas, Timestamp,
    };
    use spatialrust_records::{RecordProvenance, SchemaVersion, SpatialRecord};

    fn sample_record() -> SpatialRecord {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0, 2.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![1.0, 3.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![2.0, 4.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::new("map", Timestamp::from_nanos(7)),
        )
        .unwrap();
        SpatialRecord::try_from_cloud_with_provenance(
            "point",
            SchemaVersion::new(1, 0),
            cloud,
            RecordProvenance::try_new("bag")
                .unwrap()
                .with_stream_id("/lidar")
                .with_sequence(Some(3)),
        )
        .unwrap()
    }

    #[test]
    fn derives_centroid_and_lineage_stable_entity_id() {
        let record = sample_record();
        let entity = SpatialRecordEntity::try_from_record(
            &record,
            vec![OpenVocabLabel { text: "surface".into(), confidence: 0.8 }],
            Some(Embedding::try_new(vec![1.0, 0.0]).unwrap()),
        )
        .unwrap();
        assert_eq!(entity.entity.id, record_entity_id(record.provenance()));
        assert_eq!(entity.entity.centroid, Some(spatialrust_math::Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(entity.frame_id.0, "map");
        assert_eq!(entity.timestamp.as_nanos(), 7);
        assert_eq!(entity.provenance, *record.provenance());
    }

    #[test]
    fn rejects_invalid_label_confidence() {
        let error = SpatialRecordEntity::try_from_record(
            &sample_record(),
            vec![OpenVocabLabel { text: "surface".into(), confidence: 1.1 }],
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("confidence"));
    }
}
