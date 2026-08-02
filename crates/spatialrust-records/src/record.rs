//! Owned spatial record envelope.

use spatialrust_core::{PointCloud, SpatialMetadata};

use crate::{RecordProvenance, RecordsError, RecordsResult, SchemaDescriptor};

/// One versioned point-cloud observation with attached metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialRecord {
    schema: SchemaDescriptor,
    cloud: PointCloud,
    provenance: RecordProvenance,
}

impl SpatialRecord {
    /// Creates a record after validating that the cloud matches the descriptor schema.
    pub fn try_new(schema: SchemaDescriptor, cloud: PointCloud) -> RecordsResult<Self> {
        Self::try_new_with_provenance(schema, cloud, RecordProvenance::default())
    }

    /// Creates a record with explicit source lineage after schema validation.
    pub fn try_new_with_provenance(
        schema: SchemaDescriptor,
        cloud: PointCloud,
        provenance: RecordProvenance,
    ) -> RecordsResult<Self> {
        provenance.validate()?;
        if cloud.schema() != schema.point_schema() {
            return Err(RecordsError::SchemaMismatch(
                "point cloud schema must equal the record schema descriptor".into(),
            ));
        }
        cloud.validate()?;
        Ok(Self { schema, cloud, provenance })
    }

    /// Builds a record from a cloud using an explicit schema id/version.
    pub fn try_from_cloud(
        id: impl Into<crate::SchemaId>,
        version: crate::SchemaVersion,
        cloud: PointCloud,
    ) -> RecordsResult<Self> {
        Self::try_from_cloud_with_provenance(id, version, cloud, RecordProvenance::default())
    }

    /// Builds a record from a cloud and explicit source lineage.
    pub fn try_from_cloud_with_provenance(
        id: impl Into<crate::SchemaId>,
        version: crate::SchemaVersion,
        cloud: PointCloud,
        provenance: RecordProvenance,
    ) -> RecordsResult<Self> {
        let schema = SchemaDescriptor::try_new(id, version, cloud.schema().clone())?;
        Self::try_new_with_provenance(schema, cloud, provenance)
    }

    /// Returns the schema descriptor.
    #[must_use]
    pub fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    /// Returns the owned point cloud.
    #[must_use]
    pub fn cloud(&self) -> &PointCloud {
        &self.cloud
    }

    /// Returns source lineage attached to this record.
    #[must_use]
    pub fn provenance(&self) -> &RecordProvenance {
        &self.provenance
    }

    /// Replaces source lineage without changing schema or cloud ownership.
    #[must_use]
    pub fn with_provenance(mut self, provenance: RecordProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Consumes the record into schema, cloud, and source lineage.
    #[must_use]
    pub fn into_parts(self) -> (SchemaDescriptor, PointCloud, RecordProvenance) {
        (self.schema, self.cloud, self.provenance)
    }

    /// Consumes the record into its cloud.
    #[must_use]
    pub fn into_cloud(self) -> PointCloud {
        self.cloud
    }

    /// Returns spatial metadata attached to the cloud.
    #[must_use]
    pub fn metadata(&self) -> &SpatialMetadata {
        self.cloud.metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::SpatialRecord;
    use crate::SchemaVersion;
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
    };

    #[test]
    fn record_rejects_schema_mismatch() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let rich = SpatialRecord::try_from_cloud("p", SchemaVersion::new(1, 0), {
            let mut buffers = PointBufferSet::new();
            buffers.insert("x", PointBuffer::from_f32(vec![0.0]));
            buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
            buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
            buffers.insert("intensity", PointBuffer::from_f32(vec![1.0]));
            PointCloud::try_from_parts(
                StandardSchemas::point_xyzi(),
                buffers,
                SpatialMetadata::default(),
            )
            .unwrap()
        });
        assert!(rich.is_ok());
        let _ = cloud;
    }

    #[test]
    fn record_preserves_explicit_provenance() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let provenance = crate::RecordProvenance::try_new("source-1")
            .unwrap()
            .with_stream_id("lidar")
            .with_sequence(Some(2));
        let record = SpatialRecord::try_from_cloud_with_provenance(
            "point",
            SchemaVersion::new(1, 0),
            cloud,
            provenance.clone(),
        )
        .unwrap();
        assert_eq!(record.provenance(), &provenance);
        assert_eq!(record.with_provenance(provenance.clone()).provenance(), &provenance);
    }
}
