use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use e57::{E57Writer as ExternalE57Writer, RecordValue, RawValues};
use spatialrust_core::{DType, FieldSemantic, HasPositions3, PointBuffer, PointCloud, PointField, PointSchema};

use crate::error::{e57_format, e57_parse, IoError};
use crate::e57::schema::{schema_from_point_cloud, validate_export_schema};
use crate::{PointWriter, WriteOptions};

/// Writes point clouds to E57 files.
pub struct E57Writer;

/// Creates a unique E57 GUID string.
#[must_use]
pub fn new_e57_guid(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{prefix}-{nanos}")
}

impl PointWriter for E57Writer {
    fn write(
        &mut self,
        cloud: &PointCloud,
        _options: &WriteOptions,
    ) -> spatialrust_core::SpatialResult<()> {
        let path = std::env::temp_dir().join(format!("spatialrust_e57_{}.e57", std::process::id()));
        write_e57_file(&path, cloud).map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))?;
        std::fs::remove_file(path).map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))?;
        Ok(())
    }
}

/// Writes a point cloud to an E57 file on disk.
pub fn write_e57(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    write_e57_file(path, cloud)
}

/// Writes a point cloud to an E57 file on disk.
pub fn write_e57_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    cloud.validate()?;
    validate_export_schema(cloud.schema())?;

    let (export_schema, prototype) = schema_from_point_cloud(cloud.schema());
    let file_guid = new_e57_guid("spatialrust");
    let mut writer = ExternalE57Writer::from_file(path, &file_guid)
        .map_err(|error| e57_format(error.to_string()))?;

    let scan_guid = new_e57_guid("scan");
    let mut pc_writer = writer
        .add_pointcloud(&scan_guid, prototype)
        .map_err(|error| e57_format(error.to_string()))?;

    for index in 0..cloud.len() {
        let values = point_values(cloud, &export_schema, index)?;
        pc_writer
            .add_point(values)
            .map_err(|error| e57_format(error.to_string()))?;
    }

    pc_writer
        .finalize()
        .map_err(|error| e57_format(error.to_string()))?;
    writer
        .finalize()
        .map_err(|error| e57_format(error.to_string()))
}

fn point_values(cloud: &PointCloud, schema: &PointSchema, index: usize) -> Result<RawValues, IoError> {
    let (x, y, z) = cloud.positions3()?;
    let mut values = Vec::with_capacity(schema.len());

    for field in schema.fields() {
        let value = match field.semantic {
            FieldSemantic::PositionX => RecordValue::Single(x[index]),
            FieldSemantic::PositionY => RecordValue::Single(y[index]),
            FieldSemantic::PositionZ => RecordValue::Single(z[index]),
            FieldSemantic::Intensity => {
                RecordValue::Single(read_cloud_field(cloud, field, index)?)
            }
            FieldSemantic::ColorR | FieldSemantic::ColorG | FieldSemantic::ColorB => {
                RecordValue::Integer(read_cloud_field(cloud, field, index)?.round() as i64)
            }
            _ => {
                return Err(e57_parse(format!(
                    "unsupported E57 field `{}` during export",
                    field.name
                )));
            }
        };
        values.push(value);
    }

    Ok(values)
}

fn read_cloud_field(cloud: &PointCloud, field: &PointField, index: usize) -> Result<f32, IoError> {
    let buffer = cloud.field(&field.name)?;
    match field.dtype {
        DType::F32 | DType::F16 => Ok(buffer.as_f32()?[index]),
        DType::U8 => {
            let PointBuffer::U8(values) = buffer else {
                return Err(spatialrust_core::SpatialError::UnsupportedDType(field.dtype).into());
            };
            Ok(f32::from(values[index]))
        }
        DType::U16 => {
            let PointBuffer::U16(values) = buffer else {
                return Err(spatialrust_core::SpatialError::UnsupportedDType(field.dtype).into());
            };
            Ok(f32::from(values[index]))
        }
        _ => Err(spatialrust_core::SpatialError::UnsupportedDType(field.dtype).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::write_e57_file;
    use crate::e57::reader::read_e57_file;
    use spatialrust_core::PointCloudBuilder;

    #[test]
    fn writes_xyz_cloud() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();
        let path = std::env::temp_dir().join(format!("spatialrust_e57_write_{}.e57", std::process::id()));
        write_e57_file(&path, &cloud).unwrap();
        let loaded = read_e57_file(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(loaded.len(), 1);
    }
}
