//! Explicit schema migration for spatial records.

use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud, PointField};

use crate::{
    compare_schemas, CompatVerdict, RecordsError, RecordsResult, SchemaDescriptor, SpatialRecord,
};

/// How to fill fields missing from the source cloud.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FieldFill {
    /// Fill numeric columns with zeros.
    #[default]
    Zeros,
}

/// Migration permissions when projecting a record onto a target schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MigrationPolicy {
    /// Drop source fields absent from the target.
    pub drop_unknown: bool,
    /// Fill target fields absent from the source.
    pub fill_missing: Option<FieldFill>,
}

impl Default for MigrationPolicy {
    fn default() -> Self {
        Self { drop_unknown: true, fill_missing: Some(FieldFill::Zeros) }
    }
}

/// Projects `record` onto `target` under an explicit migration policy.
pub fn migrate_record(
    record: &SpatialRecord,
    target: &SchemaDescriptor,
    policy: MigrationPolicy,
) -> RecordsResult<SpatialRecord> {
    let report = compare_schemas(target, record.schema());
    match report.verdict {
        CompatVerdict::Identical => {
            // Field order may still differ; always rebuild against `target`.
        }
        CompatVerdict::BackwardCompatible => {
            if !policy.drop_unknown {
                return Err(RecordsError::SchemaMismatch(
                    "source has extra fields; enable drop_unknown to migrate".into(),
                ));
            }
        }
        CompatVerdict::ForwardCompatible => {
            if policy.fill_missing.is_none() {
                return Err(RecordsError::SchemaMismatch(
                    "source is missing fields; configure fill_missing to migrate".into(),
                ));
            }
        }
        CompatVerdict::Incompatible => {
            return Err(RecordsError::SchemaMismatch(format!(
                "cannot migrate incompatible schemas: missing={:?} extra={:?} conflict={:?}",
                report.missing_in_actual, report.extra_in_actual, report.conflicting
            )));
        }
    }

    let len = record.cloud().len();
    let mut buffers = PointBufferSet::new();
    for field in target.point_schema().fields() {
        if let Ok(source) = record.cloud().field(&field.name) {
            buffers.insert(field.name.clone(), clone_buffer(source)?);
        } else {
            let fill = policy.fill_missing.ok_or_else(|| {
                RecordsError::MissingField(field.name.clone())
            })?;
            buffers.insert(field.name.clone(), filled_buffer(field, len, fill)?);
        }
    }
    let cloud = PointCloud::try_from_parts(
        target.point_schema().clone(),
        buffers,
        record.metadata().clone(),
    )?;
    SpatialRecord::try_new(target.clone(), cloud)
}

fn clone_buffer(buffer: &PointBuffer) -> RecordsResult<PointBuffer> {
    Ok(match buffer {
        PointBuffer::F32(values) => PointBuffer::F32(values.clone()),
        PointBuffer::F64(values) => PointBuffer::F64(values.clone()),
        PointBuffer::U8(values) => PointBuffer::U8(values.clone()),
        PointBuffer::U16(values) => PointBuffer::U16(values.clone()),
        PointBuffer::U32(values) => PointBuffer::U32(values.clone()),
        PointBuffer::I32(values) => PointBuffer::I32(values.clone()),
    })
}

fn filled_buffer(field: &PointField, len: usize, fill: FieldFill) -> RecordsResult<PointBuffer> {
    match fill {
        FieldFill::Zeros => {
            let mut buffer = PointBuffer::with_capacity(field.dtype, len);
            buffer.resize(len, field)?;
            Ok(buffer)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{migrate_record, FieldFill, MigrationPolicy};
    use crate::{SchemaDescriptor, SchemaVersion, SpatialRecord};
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
    };

    fn xyz_record() -> SpatialRecord {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0, 2.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![3.0, 4.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![5.0, 6.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud).unwrap()
    }

    #[test]
    fn migrates_xyz_to_xyzi_with_zero_fill() {
        let source = xyz_record();
        let target = SchemaDescriptor::try_new(
            "point",
            SchemaVersion::new(1, 1),
            StandardSchemas::point_xyzi(),
        )
        .unwrap();
        let migrated = migrate_record(
            &source,
            &target,
            MigrationPolicy { drop_unknown: true, fill_missing: Some(FieldFill::Zeros) },
        )
        .unwrap();
        assert_eq!(migrated.cloud().field("intensity").unwrap().as_f32().unwrap(), &[0.0, 0.0]);
        assert_eq!(migrated.schema().version.minor, 1);
    }

    #[test]
    fn drops_intensity_when_targeting_xyz() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![2.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![3.0]));
        buffers.insert("intensity", PointBuffer::from_f32(vec![9.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyzi(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let source =
            SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 1), cloud).unwrap();
        let target = SchemaDescriptor::try_new(
            "point",
            SchemaVersion::new(1, 0),
            StandardSchemas::point_xyz(),
        )
        .unwrap();
        let migrated = migrate_record(&source, &target, MigrationPolicy::default()).unwrap();
        assert!(migrated.cloud().field("intensity").is_err());
        assert_eq!(migrated.cloud().len(), 1);
    }
}
