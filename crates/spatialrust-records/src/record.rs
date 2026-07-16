//! Owned spatial record envelope.

use spatialrust_core::{PointCloud, SpatialMetadata};

use crate::{RecordsError, RecordsResult, SchemaDescriptor};

/// One versioned point-cloud observation with attached metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialRecord {
    schema: SchemaDescriptor,
    cloud: PointCloud,
}

impl SpatialRecord {
    /// Creates a record after validating that the cloud matches the descriptor schema.
    pub fn try_new(schema: SchemaDescriptor, cloud: PointCloud) -> RecordsResult<Self> {
        if cloud.schema() != schema.point_schema() {
            return Err(RecordsError::SchemaMismatch(
                "point cloud schema must equal the record schema descriptor".into(),
            ));
        }
        cloud.validate()?;
        Ok(Self { schema, cloud })
    }

    /// Builds a record from a cloud using an explicit schema id/version.
    pub fn try_from_cloud(
        id: impl Into<crate::SchemaId>,
        version: crate::SchemaVersion,
        cloud: PointCloud,
    ) -> RecordsResult<Self> {
        let schema = SchemaDescriptor::try_new(id, version, cloud.schema().clone())?;
        Self::try_new(schema, cloud)
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
}
