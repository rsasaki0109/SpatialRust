use std::io::Write;

use spatialrust_core::{DType, FieldSemantic, PointBuffer, PointCloud, PointSchema};

use crate::error::{ply_format, IoError};
use crate::ply::header::{PlyFormat, PlyHeader, PlyProperty, PlyPropertyKind};
use crate::ply::schema::{infer_property_semantic, ply_property_from_field};
use crate::{PointWriter, WriteOptions};

#[cfg(feature = "streaming")]
use crate::streaming::records_io;
#[cfg(feature = "streaming")]
use spatialrust_records::{
    BoundedSpatialRecordSink, RecordsError, RecordsResult, SchemaDescriptor, SpatialRecordChunk,
};

/// Output encoding for PLY writers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlyWriteFormat {
    /// ASCII PLY.
    #[default]
    Ascii,
    /// Binary little-endian PLY.
    BinaryLittleEndian,
}

/// Writes point clouds to PLY files or streams.
pub struct PlyWriter<W: Write> {
    writer: W,
    format: PlyWriteFormat,
}

impl<W: Write> PlyWriter<W> {
    /// Creates a new PLY writer.
    #[must_use]
    pub const fn new(writer: W, format: PlyWriteFormat) -> Self {
        Self { writer, format }
    }
}

impl<W: Write> PointWriter for PlyWriter<W> {
    fn write(
        &mut self,
        cloud: &PointCloud,
        _options: &WriteOptions,
    ) -> spatialrust_core::SpatialResult<()> {
        write_ply(&mut self.writer, cloud, self.format)
            .map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))
    }
}

/// Writes a point cloud to a PLY stream.
pub fn write_ply<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    format: PlyWriteFormat,
) -> Result<(), IoError> {
    cloud.validate()?;
    let properties = ply_properties_from_schema(cloud.schema())?;
    let header = PlyHeader {
        format: match format {
            PlyWriteFormat::Ascii => PlyFormat::Ascii,
            PlyWriteFormat::BinaryLittleEndian => PlyFormat::BinaryLittleEndian,
        },
        vertex_count: cloud.len(),
        properties,
    };
    header.write_header(writer)?;

    match format {
        PlyWriteFormat::Ascii => write_ascii_vertices(writer, cloud, &header.properties)?,
        PlyWriteFormat::BinaryLittleEndian => {
            write_binary_vertices(writer, cloud, &header.properties)?
        }
    }
    Ok(())
}

/// Writes a point cloud to a PLY file on disk.
pub fn write_ply_file(
    path: impl AsRef<std::path::Path>,
    cloud: &PointCloud,
    format: PlyWriteFormat,
) -> Result<(), IoError> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut writer = std::io::BufWriter::new(file);
    write_ply(&mut writer, cloud, format)
}

fn ply_properties_from_schema(schema: &PointSchema) -> Result<Vec<PlyProperty>, IoError> {
    schema.fields().iter().map(ply_property_from_field).collect()
}

/// Sequential PLY sink with an exact upfront vertex-count contract.
#[cfg(feature = "streaming")]
pub struct PlyChunkSink<W: Write> {
    writer: W,
    schema: SchemaDescriptor,
    properties: Vec<PlyProperty>,
    format: PlyWriteFormat,
    expected_points: u64,
    written_points: u64,
    finished: bool,
}

#[cfg(feature = "streaming")]
impl<W: Write> PlyChunkSink<W> {
    /// Writes the PLY header and creates a chunk sink.
    pub fn new(
        mut writer: W,
        schema: SchemaDescriptor,
        expected_points: u64,
        format: PlyWriteFormat,
    ) -> Result<Self, IoError> {
        let properties = ply_properties_from_schema(schema.point_schema())?;
        let vertex_count = usize::try_from(expected_points)
            .map_err(|_| ply_format("PLY vertex count does not fit usize"))?;
        PlyHeader {
            format: match format {
                PlyWriteFormat::Ascii => PlyFormat::Ascii,
                PlyWriteFormat::BinaryLittleEndian => PlyFormat::BinaryLittleEndian,
            },
            vertex_count,
            properties: properties.clone(),
        }
        .write_header(&mut writer)?;
        Ok(Self {
            writer,
            schema,
            properties,
            format,
            expected_points,
            written_points: 0,
            finished: false,
        })
    }
}

#[cfg(feature = "streaming")]
impl PlyChunkSink<std::io::BufWriter<std::fs::File>> {
    /// Creates a local PLY file sink.
    pub fn create(
        path: impl AsRef<std::path::Path>,
        schema: SchemaDescriptor,
        expected_points: u64,
        format: PlyWriteFormat,
    ) -> Result<Self, IoError> {
        Self::new(
            std::io::BufWriter::new(std::fs::File::create(path)?),
            schema,
            expected_points,
            format,
        )
    }
}

#[cfg(feature = "streaming")]
impl<W: Write> BoundedSpatialRecordSink for PlyChunkSink<W> {
    fn write_chunk(&mut self, chunk: &SpatialRecordChunk) -> RecordsResult<()> {
        if self.finished {
            return Err(RecordsError::InvalidConfiguration("PLY sink is already finished".into()));
        }
        let cloud = chunk.record().cloud();
        if cloud.schema() != self.schema.point_schema() {
            return Err(RecordsError::SchemaMismatch(
                "PLY chunk schema differs from sink schema".into(),
            ));
        }
        let next = self
            .written_points
            .checked_add(
                u64::try_from(cloud.len())
                    .map_err(|_| RecordsError::ReceiptOverflow("PLY chunk point count".into()))?,
            )
            .ok_or_else(|| RecordsError::ReceiptOverflow("PLY point count".into()))?;
        if next > self.expected_points {
            return Err(RecordsError::InvalidChunk(format!(
                "PLY sink expected {} points but received at least {next}",
                self.expected_points
            )));
        }
        match self.format {
            PlyWriteFormat::Ascii => {
                write_ascii_vertices(&mut self.writer, cloud, &self.properties)
            }
            PlyWriteFormat::BinaryLittleEndian => {
                write_binary_vertices(&mut self.writer, cloud, &self.properties)
            }
        }
        .map_err(records_io)?;
        self.written_points = next;
        Ok(())
    }

    fn finish(&mut self) -> RecordsResult<()> {
        if self.written_points != self.expected_points {
            return Err(RecordsError::InvalidChunk(format!(
                "PLY sink expected {} points but received {}",
                self.expected_points, self.written_points
            )));
        }
        self.writer.flush().map_err(IoError::from).map_err(records_io)?;
        self.finished = true;
        Ok(())
    }
}

fn write_ascii_vertices<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    properties: &[PlyProperty],
) -> Result<(), IoError> {
    for point_index in 0..cloud.len() {
        let mut first = true;
        for property in properties {
            if !first {
                write!(writer, " ")?;
            }
            first = false;
            write!(writer, "{}", read_scalar(cloud, property, point_index)?)?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

fn write_binary_vertices<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    properties: &[PlyProperty],
) -> Result<(), IoError> {
    for point_index in 0..cloud.len() {
        for property in properties {
            write_binary_scalar(writer, cloud, property, point_index)?;
        }
    }
    Ok(())
}

fn write_binary_scalar<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    property: &PlyProperty,
    point_index: usize,
) -> Result<(), IoError> {
    let field = find_field_for_property(cloud.schema(), property)?;
    let buffer = cloud.field(&field.name).map_err(IoError::from)?;

    match (field.dtype, property.kind) {
        (DType::F32 | DType::F16, PlyPropertyKind::Float) => {
            let value = buffer.as_f32().map_err(IoError::from)?[point_index];
            writer.write_all(&value.to_le_bytes())?;
        }
        (DType::F64, PlyPropertyKind::Double) => {
            let PointBuffer::F64(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        (DType::I32, PlyPropertyKind::Int) => {
            let PointBuffer::I32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        (DType::U32, PlyPropertyKind::UInt) => {
            let PointBuffer::U32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        (DType::U8, PlyPropertyKind::UChar) => {
            let PointBuffer::U8(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&[values[point_index]])?;
        }
        (DType::U16, PlyPropertyKind::UShort) => {
            let PointBuffer::U16(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        _ => return Err(ply_format(format!("cannot encode field `{}` to PLY", field.name))),
    }
    Ok(())
}

fn read_scalar(
    cloud: &PointCloud,
    property: &PlyProperty,
    point_index: usize,
) -> Result<f32, IoError> {
    let field = find_field_for_property(cloud.schema(), property)?;
    let buffer = cloud.field(&field.name).map_err(IoError::from)?;
    let value = match field.dtype {
        DType::F32 | DType::F16 => buffer.as_f32().map_err(IoError::from)?[point_index],
        DType::F64 => {
            let PointBuffer::F64(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            values[point_index] as f32
        }
        DType::U8 => {
            let PointBuffer::U8(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            f32::from(values[point_index])
        }
        DType::U16 => {
            let PointBuffer::U16(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            f32::from(values[point_index])
        }
        DType::I32 => {
            let PointBuffer::I32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            values[point_index] as f32
        }
        DType::U32 => {
            let PointBuffer::U32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            values[point_index] as f32
        }
    };
    Ok(value)
}

fn find_field_for_property<'a>(
    schema: &'a PointSchema,
    property: &PlyProperty,
) -> Result<&'a spatialrust_core::PointField, IoError> {
    let semantic = infer_property_semantic(&property.name);
    schema
        .fields()
        .iter()
        .find(|field| {
            field.name == property.name
                || (semantic != FieldSemantic::Unknown && field.semantic == semantic)
        })
        .ok_or_else(|| ply_format(format!("missing field for PLY property `{}`", property.name)))
}
