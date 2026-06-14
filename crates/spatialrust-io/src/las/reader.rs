use std::io::{BufRead, Seek};
use std::path::Path;

use las::{Header, Point, Reader};
use spatialrust_core::{
    DType, FieldSemantic, PointBuffer, PointBufferSet, PointCloud, PointSchema, SpatialMetadata,
};

use crate::error::{las_parse, IoError};
use crate::las::schema::schema_for_las_header;
use crate::{PointReader, ReadOptions};

/// Reads point clouds from LAS/LAZ files.
pub struct LasReader {
    reader: Reader,
    metadata: SpatialMetadata,
    schema: PointSchema,
    loaded: bool,
}

impl LasReader {
    /// Opens a LAS/LAZ file and parses its header eagerly.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let reader = Reader::from_path(path).map_err(|error| las_parse(error.to_string()))?;
        let header = reader.header();
        Ok(Self {
            schema: schema_for_las_header(header),
            metadata: metadata_from_header(header),
            reader,
            loaded: false,
        })
    }

    /// Returns the parsed LAS header.
    #[must_use]
    pub fn header(&self) -> &Header {
        self.reader.header()
    }

    /// Reads the point cloud payload.
    pub fn read_cloud(&mut self) -> Result<PointCloud, IoError> {
        if self.loaded {
            return Err(crate::error::las_format("LAS reader already consumed"));
        }
        self.loaded = true;
        read_points_from_reader(&mut self.reader, self.schema.clone(), self.metadata.clone())
    }
}

impl PointReader for LasReader {
    fn schema(&self) -> spatialrust_core::SpatialResult<PointSchema> {
        Ok(self.schema.clone())
    }

    fn metadata(&self) -> spatialrust_core::SpatialResult<SpatialMetadata> {
        Ok(self.metadata.clone())
    }

    fn read(&mut self, _options: &ReadOptions) -> spatialrust_core::SpatialResult<PointCloud> {
        self.read_cloud().map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))
    }
}

/// Reads a complete LAS/LAZ stream.
pub fn read_las<R: BufRead + Seek + Send + Sync + 'static>(
    reader: R,
) -> Result<PointCloud, IoError> {
    let mut las_reader = Reader::new(reader).map_err(|error| las_parse(error.to_string()))?;
    let header = las_reader.header().clone();
    let schema = schema_for_las_header(&header);
    let metadata = metadata_from_header(&header);
    read_points_from_reader(&mut las_reader, schema, metadata)
}

/// Reads a LAS/LAZ file from disk.
pub fn read_las_file(path: impl AsRef<Path>) -> Result<PointCloud, IoError> {
    let mut reader = LasReader::open(path)?;
    reader.read_cloud()
}

fn read_points_from_reader(
    reader: &mut Reader,
    schema: PointSchema,
    metadata: SpatialMetadata,
) -> Result<PointCloud, IoError> {
    let mut points = Vec::new();
    for point in reader.points() {
        points.push(point.map_err(|error| las_parse(error.to_string()))?);
    }
    point_cloud_from_las_points(schema, metadata, points)
}

/// Builds a point cloud from LAS points using a precomputed schema and metadata.
pub(crate) fn point_cloud_from_las_points(
    schema: PointSchema,
    metadata: SpatialMetadata,
    points: impl IntoIterator<Item = Point>,
) -> Result<PointCloud, IoError> {
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
    }

    for point in points {
        append_las_point(&schema, &mut buffers, &point)?;
    }

    PointCloud::try_from_parts(schema, buffers, metadata).map_err(IoError::from)
}

fn metadata_from_header(_header: &Header) -> SpatialMetadata {
    metadata_from_las_header()
}

pub(crate) fn metadata_from_las_header() -> SpatialMetadata {
    SpatialMetadata {
        frame_id: spatialrust_core::FrameId::new("las"),
        timestamp: spatialrust_core::Timestamp::from_nanos(0),
        sensor_origin: None,
        unit: "meter".to_owned(),
    }
}

fn append_las_point(
    schema: &PointSchema,
    buffers: &mut PointBufferSet,
    point: &Point,
) -> Result<(), IoError> {
    for field in schema.fields() {
        let value = read_las_field(point, field)?;
        push_field(buffers, field, value)?;
    }
    Ok(())
}

fn read_las_field(point: &Point, field: &spatialrust_core::PointField) -> Result<f64, IoError> {
    match field.semantic {
        FieldSemantic::PositionX => Ok(point.x),
        FieldSemantic::PositionY => Ok(point.y),
        FieldSemantic::PositionZ => Ok(point.z),
        FieldSemantic::Intensity => Ok(f64::from(point.intensity)),
        FieldSemantic::Label => Ok(f64::from(u8::from(point.classification))),
        FieldSemantic::TimeOffset => point
            .gps_time
            .ok_or_else(|| las_parse("missing gps_time for LAS point format".to_owned())),
        FieldSemantic::ColorR => point
            .color
            .map(|color| f64::from(color.red))
            .ok_or_else(|| las_parse("missing color for LAS point format".to_owned())),
        FieldSemantic::ColorG => point
            .color
            .map(|color| f64::from(color.green))
            .ok_or_else(|| las_parse("missing color for LAS point format".to_owned())),
        FieldSemantic::ColorB => point
            .color
            .map(|color| f64::from(color.blue))
            .ok_or_else(|| las_parse("missing color for LAS point format".to_owned())),
        _ => Err(las_parse(format!("unsupported LAS field `{}`", field.name))),
    }
}

fn push_field(
    buffers: &mut PointBufferSet,
    field: &spatialrust_core::PointField,
    value: f64,
) -> Result<(), IoError> {
    let buffer = buffers
        .get_mut(&field.name)
        .ok_or_else(|| spatialrust_core::SpatialError::MissingField(field.name.clone()))?;
    match field.dtype {
        DType::F32 | DType::F16 => {
            buffer.push_f32(value as f32).map_err(IoError::from)?;
            Ok(())
        }
        DType::F64 => {
            buffer.push_f64(value).map_err(IoError::from)?;
            Ok(())
        }
        DType::U8 => {
            buffer.push_u8(value.round() as u8).map_err(IoError::from)?;
            Ok(())
        }
        DType::U16 => {
            buffer.push_u16(value.round() as u16).map_err(IoError::from)?;
            Ok(())
        }
        DType::I32 => {
            buffer.push_i32(value.round() as i32).map_err(IoError::from)?;
            Ok(())
        }
        DType::U32 => {
            let PointBuffer::U32(values) = buffer else {
                return Err(spatialrust_core::SpatialError::UnsupportedDType(field.dtype).into());
            };
            values.push(value.round() as u32);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::read_las;
    use crate::las::writer::{write_las, LasWriteFormat};
    use spatialrust_core::{HasIntensity, HasPositions3, PointCloudBuilder, StandardSchemas};
    use std::io::Cursor;

    #[test]
    fn roundtrip_xyz_intensity() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([1.0, 2.0, 3.0, 100.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0, 200.0]).unwrap();
        let cloud = builder.build().unwrap();

        let mut cursor = write_las(Cursor::new(Vec::new()), &cloud, LasWriteFormat::Las).unwrap();
        cursor.set_position(0);
        let loaded = read_las(cursor).unwrap();

        assert_eq!(loaded.len(), 2);
        let (x, y, z) = loaded.positions3().unwrap();
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((y[1] - 5.0).abs() < 1e-5);
        assert!((z[1] - 6.0).abs() < 1e-5);
        assert!((loaded.intensity().unwrap()[0] - 100.0).abs() < 1e-3);
    }
}
