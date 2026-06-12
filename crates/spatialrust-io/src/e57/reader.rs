use std::path::Path;

use e57::{CartesianCoordinate, E57Reader as ExternalE57Reader, Point as E57Point, PointCloud as E57PointCloud};
use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointCloud, PointField, PointSchema, SpatialMetadata,
};

use crate::error::{e57_format, e57_parse, IoError};
use crate::e57::schema::schema_for_e57_pointcloud;
use crate::{PointReader, ReadOptions};

/// Reads point clouds from E57 files.
pub struct E57Reader {
    metadata: SpatialMetadata,
}

impl E57Reader {
    /// Opens an E57 file and parses its header eagerly.
    pub fn open(_path: impl AsRef<Path>) -> Result<Self, IoError> {
        Ok(Self {
            metadata: metadata_from_file(),
        })
    }
}

impl PointReader for E57Reader {
    fn schema(&self) -> spatialrust_core::SpatialResult<PointSchema> {
        Err(spatialrust_core::SpatialError::InvalidArgument(
            "E57 schema is scan-specific; use read_e57_file instead".to_owned(),
        ))
    }

    fn metadata(&self) -> spatialrust_core::SpatialResult<SpatialMetadata> {
        Ok(self.metadata.clone())
    }

    fn read(&mut self, _options: &ReadOptions) -> spatialrust_core::SpatialResult<PointCloud> {
        Err(spatialrust_core::SpatialError::InvalidArgument(
            "E57Reader requires read_e57_file for scan-aware loading".to_owned(),
        ))
    }
}

/// Reads all scans from an E57 file and merges them into one point cloud.
pub fn read_e57(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    read_e57_file(path)
}

/// Reads all scans from an E57 file and merges them into one point cloud.
pub fn read_e57_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    let mut reader =
        ExternalE57Reader::from_file(path).map_err(|error| e57_parse(error.to_string()))?;
    read_from_reader(&mut reader)
}

fn read_from_reader(
    reader: &mut ExternalE57Reader<impl std::io::Read + std::io::Seek>,
) -> Result<PointCloud, IoError> {
    let scans = reader.pointclouds();
    if scans.is_empty() {
        return Err(e57_format("E57 file contains no point clouds".to_owned()));
    }

    let schema = merged_schema(&scans);
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
    }

    for scan in scans {
        let mut iter = reader
            .pointcloud_simple(&scan)
            .map_err(|error| e57_parse(error.to_string()))?;
        iter.normalize_intensity(false);
        iter.normalize_color(false);

        for point in iter {
            let point = point.map_err(|error| e57_parse(error.to_string()))?;
            append_e57_point(&schema, &mut buffers, &point)?;
        }
    }

    PointCloud::try_from_parts(schema, buffers, metadata_from_file()).map_err(IoError::from)
}

fn merged_schema(scans: &[E57PointCloud]) -> PointSchema {
    let mut has_intensity = false;
    let mut has_color = false;
    for scan in scans {
        if scan.has_intensity() {
            has_intensity = true;
        }
        if scan.has_color() {
            has_color = true;
        }
    }

    let mut schema = schema_for_e57_pointcloud(&scans[0]);
    if has_intensity && schema.find_semantic(FieldSemantic::Intensity).is_none() {
        schema = schema.with_field(PointField::scalar(
            "intensity",
            FieldSemantic::Intensity,
            DType::F32,
        ));
    }
    if has_color && schema.find_semantic(FieldSemantic::ColorR).is_none() {
        schema = schema
            .with_field(PointField::scalar("r", FieldSemantic::ColorR, DType::U8))
            .with_field(PointField::scalar("g", FieldSemantic::ColorG, DType::U8))
            .with_field(PointField::scalar("b", FieldSemantic::ColorB, DType::U8));
    }
    schema
}

fn metadata_from_file() -> SpatialMetadata {
    SpatialMetadata {
        frame_id: spatialrust_core::FrameId::new("e57"),
        timestamp: spatialrust_core::Timestamp::from_nanos(0),
        sensor_origin: None,
        unit: "meter".to_owned(),
    }
}

fn append_e57_point(
    schema: &PointSchema,
    buffers: &mut PointBufferSet,
    point: &E57Point,
) -> Result<(), IoError> {
    let (x, y, z) = cartesian_xyz(point)?;
    for field in schema.fields() {
        let value = match field.semantic {
            FieldSemantic::PositionX => x,
            FieldSemantic::PositionY => y,
            FieldSemantic::PositionZ => z,
            FieldSemantic::Intensity => point.intensity.ok_or_else(|| {
                e57_parse("missing intensity for E57 point with intensity-enabled schema".to_owned())
            })?,
            FieldSemantic::ColorR => color_channel(point, |color| color.red)?,
            FieldSemantic::ColorG => color_channel(point, |color| color.green)?,
            FieldSemantic::ColorB => color_channel(point, |color| color.blue)?,
            _ => {
                return Err(e57_parse(format!(
                    "unsupported E57 export field `{}`",
                    field.name
                )));
            }
        };
        push_field(buffers, field, value)?;
    }
    Ok(())
}

fn cartesian_xyz(point: &E57Point) -> Result<(f32, f32, f32), IoError> {
    match point.cartesian {
        CartesianCoordinate::Valid { x, y, z } => Ok((x as f32, y as f32, z as f32)),
        CartesianCoordinate::Direction { .. } => Err(e57_parse(
            "direction-only Cartesian E57 points are not supported".to_owned(),
        )),
        CartesianCoordinate::Invalid => Err(e57_parse("invalid Cartesian E57 point".to_owned())),
    }
}

fn color_channel(
    point: &E57Point,
    channel: impl Fn(&e57::Color) -> f32,
) -> Result<f32, IoError> {
    point
        .color
        .as_ref()
        .map(|color| channel(color))
        .ok_or_else(|| e57_parse("missing color for E57 point with color-enabled schema".to_owned()))
}

fn push_field(
    buffers: &mut PointBufferSet,
    field: &PointField,
    value: f32,
) -> Result<(), IoError> {
    let buffer = buffers
        .get_mut(&field.name)
        .ok_or_else(|| spatialrust_core::SpatialError::MissingField(field.name.clone()))?;
    match field.dtype {
        DType::F32 | DType::F16 => {
            buffer.push_f32(value).map_err(IoError::from)?;
        }
        DType::U8 => {
            buffer.push_u8(value.round() as u8).map_err(IoError::from)?;
        }
        DType::U16 => {
            buffer.push_u16(value.round() as u16).map_err(IoError::from)?;
        }
        _ => return Err(spatialrust_core::SpatialError::UnsupportedDType(field.dtype).into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_e57_file;
    use crate::e57::writer::write_e57_file;
    use spatialrust_core::{HasIntensity, HasPositions3, PointCloudBuilder, StandardSchemas};

    #[test]
    fn roundtrip_xyz_intensity() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([1.0, 2.0, 3.0, 100.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0, 200.0]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!("spatialrust_e57_{}.e57", std::process::id()));
        write_e57_file(&path, &cloud).unwrap();
        let loaded = read_e57_file(&path).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(loaded.len(), 2);
        let (x, _, _) = loaded.positions3().unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4);
        assert!((loaded.intensity().unwrap()[1] - 200.0).abs() < 1e-3);
    }
}
