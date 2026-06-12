use std::io::BufRead;

use spatialrust_core::{
    DType, PointBuffer, PointBufferSet, PointCloud, PointSchema, SpatialMetadata,
};

use crate::error::{ply_format, ply_parse, IoError};
use crate::ply::header::{PlyFormat, PlyHeader, PlyPropertyKind};
use crate::ply::schema::schema_from_ply_properties;
use crate::{PointReader, ReadOptions};

/// Reads point clouds from PLY files or streams.
pub struct PlyReader<R: BufRead> {
    reader: R,
    header: PlyHeader,
    metadata: SpatialMetadata,
    schema: PointSchema,
    loaded: bool,
}

impl<R: BufRead> PlyReader<R> {
    /// Creates a reader and parses the PLY header eagerly.
    pub fn new(mut reader: R) -> Result<Self, IoError> {
        let header = PlyHeader::parse(&mut reader)?;
        let schema = schema_from_ply_properties(&header.properties)?;
        let metadata = metadata_from_header(&header);
        Ok(Self { reader, header, metadata, schema, loaded: false })
    }

    /// Returns the parsed PLY header.
    #[must_use]
    pub fn header(&self) -> &PlyHeader {
        &self.header
    }

    /// Reads the point cloud payload.
    pub fn read_cloud(&mut self) -> Result<PointCloud, IoError> {
        if self.loaded {
            return Err(ply_format("PLY reader already consumed"));
        }
        self.loaded = true;
        read_ply_body(&self.header, &mut self.reader, self.schema.clone(), self.metadata.clone())
    }
}

impl<R: BufRead> PointReader for PlyReader<R> {
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

/// Reads a complete PLY file from any buffered reader.
pub fn read_ply<R: BufRead>(reader: &mut R) -> Result<PointCloud, IoError> {
    let header = PlyHeader::parse(reader)?;
    let schema = schema_from_ply_properties(&header.properties)?;
    let metadata = metadata_from_header(&header);
    read_ply_body(&header, reader, schema, metadata)
}

fn read_ply_body<R: BufRead>(
    header: &PlyHeader,
    reader: &mut R,
    schema: PointSchema,
    metadata: SpatialMetadata,
) -> Result<PointCloud, IoError> {
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, header.vertex_count));
    }

    match header.format {
        PlyFormat::Ascii => read_ascii_vertices(reader, header, &schema, &mut buffers)?,
        PlyFormat::BinaryLittleEndian => read_binary_vertices(reader, header, &schema, &mut buffers)?,
    }

    PointCloud::try_from_parts(schema, buffers, metadata).map_err(IoError::from)
}

fn metadata_from_header(_header: &PlyHeader) -> SpatialMetadata {
    SpatialMetadata {
        frame_id: spatialrust_core::FrameId::new("ply"),
        timestamp: spatialrust_core::Timestamp::from_nanos(0),
        sensor_origin: None,
        unit: "meter".to_owned(),
    }
}

fn read_ascii_vertices<R: BufRead>(
    reader: &mut R,
    header: &PlyHeader,
    schema: &PointSchema,
    buffers: &mut PointBufferSet,
) -> Result<(), IoError> {
    for vertex_index in 0..header.vertex_count {
        let line = read_ascii_vertex_line(reader, vertex_index, header.vertex_count)?;
        let mut tokens = line.split_whitespace();
        for property in &header.properties {
            let token = tokens.next().ok_or_else(|| {
                ply_parse(format!(
                    "missing token for property `{}` on vertex {vertex_index}",
                    property.name
                ))
            })?;
            let value = token
                .parse::<f64>()
                .map_err(|_| ply_parse(format!("invalid ASCII value `{token}`")))?;
            push_to_field(buffers, schema, &property.name, value as f32)?;
        }
    }
    Ok(())
}

fn read_ascii_vertex_line<R: BufRead>(
    reader: &mut R,
    vertex_index: usize,
    vertex_count: usize,
) -> Result<String, IoError> {
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Err(ply_parse(format!(
                "unexpected EOF after {vertex_index} of {vertex_count} ASCII vertices"
            )));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        return Ok(trimmed.to_owned());
    }
}

fn read_binary_vertices<R: BufRead>(
    reader: &mut R,
    header: &PlyHeader,
    schema: &PointSchema,
    buffers: &mut PointBufferSet,
) -> Result<(), IoError> {
    let mut payload = vec![0_u8; header.vertex_stride() * header.vertex_count];
    std::io::Read::read_exact(&mut *reader, &mut payload).map_err(IoError::from)?;

    for vertex_index in 0..header.vertex_count {
        let start = vertex_index * header.vertex_stride();
        let mut offset = 0usize;
        for property in &header.properties {
            let size = property.kind.size_bytes();
            let chunk = &payload[start + offset..start + offset + size];
            offset += size;
            let value = read_binary_scalar(property.kind, chunk)?;
            push_to_field(buffers, schema, &property.name, value)?;
        }
    }
    Ok(())
}

fn read_binary_scalar(kind: PlyPropertyKind, chunk: &[u8]) -> Result<f32, IoError> {
    let value = match kind {
        PlyPropertyKind::Char => i8::from_le_bytes(chunk.try_into().expect("char")) as f32,
        PlyPropertyKind::UChar => f32::from(chunk[0]),
        PlyPropertyKind::Short => i16::from_le_bytes(chunk.try_into().expect("short")) as f32,
        PlyPropertyKind::UShort => f32::from(u16::from_le_bytes(chunk.try_into().expect("ushort"))),
        PlyPropertyKind::Int => i32::from_le_bytes(chunk.try_into().expect("int")) as f32,
        PlyPropertyKind::UInt => u32::from_le_bytes(chunk.try_into().expect("uint")) as f32,
        PlyPropertyKind::Float => f32::from_le_bytes(chunk.try_into().expect("float")),
        PlyPropertyKind::Double => f64::from_le_bytes(chunk.try_into().expect("double")) as f32,
    };
    Ok(value)
}

fn push_to_field(
    buffers: &mut PointBufferSet,
    schema: &PointSchema,
    name: &str,
    value: f32,
) -> Result<(), IoError> {
    let field = schema
        .fields()
        .iter()
        .find(|field| field.name == name)
        .ok_or_else(|| ply_format(format!("schema missing mapped field `{name}`")))?;

    let buffer = buffers
        .get_mut(name)
        .ok_or_else(|| ply_format(format!("buffer missing for field `{name}`")))?;

    match field.dtype {
        DType::F32 | DType::F16 => buffer.push_f32(value).map_err(IoError::from),
        DType::F64 => buffer.push_f64(f64::from(value)).map_err(IoError::from),
        DType::U8 => buffer.push_u8(value.round() as u8).map_err(IoError::from),
        DType::U16 => buffer.push_u16(value.round() as u16).map_err(IoError::from),
        DType::I32 => buffer.push_i32(value.round() as i32).map_err(IoError::from),
        DType::U32 => {
            let PointBuffer::U32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            values.push(value.round() as u32);
            Ok(())
        }
    }
}

/// Reads a PLY file from disk.
pub fn read_ply_file(path: impl AsRef<std::path::Path>) -> Result<PointCloud, IoError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut reader = std::io::BufReader::new(file);
    read_ply(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::read_ply;
    use crate::ply::writer::{write_ply, PlyWriteFormat};
    use spatialrust_core::{HasIntensity, HasPositions3, PointCloudBuilder, StandardSchemas};
    use std::io::Cursor;

    const SAMPLE_XYZ_ASCII: &str = "\
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
";

    const SAMPLE_XYZI_ASCII: &str = "\
ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float intensity
end_header
0.0 0.0 0.0 0.5
1.0 0.0 0.0 0.8
";

    #[test]
    fn reads_ascii_xyz() {
        let mut reader = Cursor::new(SAMPLE_XYZ_ASCII.as_bytes());
        let cloud = read_ply(&mut reader).unwrap();
        assert_eq!(cloud.len(), 3);
        let (x, y, z) = cloud.positions3().unwrap();
        assert_eq!(x, &[0.0, 1.0, 0.0]);
        assert_eq!(y, &[0.0, 0.0, 1.0]);
        assert_eq!(z, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn reads_ascii_xyzi() {
        let mut reader = Cursor::new(SAMPLE_XYZI_ASCII.as_bytes());
        let cloud = read_ply(&mut reader).unwrap();
        assert_eq!(cloud.intensity().unwrap(), &[0.5, 0.8]);
    }

    #[test]
    fn roundtrip_ascii_xyz() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();

        let mut buffer = Vec::new();
        write_ply(&mut buffer, &cloud, PlyWriteFormat::Ascii).unwrap();

        let mut reader = Cursor::new(buffer);
        let loaded = read_ply(&mut reader).unwrap();
        assert_eq!(loaded.len(), cloud.len());
        let (x, y, z) = loaded.positions3().unwrap();
        assert_eq!(x, cloud.field("x").unwrap().as_f32().unwrap());
        assert_eq!(y, cloud.field("y").unwrap().as_f32().unwrap());
        assert_eq!(z, cloud.field("z").unwrap().as_f32().unwrap());
    }

    #[test]
    fn roundtrip_binary_xyz() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.5, 1.5, 2.5]).unwrap();
        let cloud = builder.build().unwrap();

        let mut buffer = Vec::new();
        write_ply(&mut buffer, &cloud, PlyWriteFormat::BinaryLittleEndian).unwrap();

        let mut reader = Cursor::new(buffer);
        let loaded = read_ply(&mut reader).unwrap();
        let (x, _, _) = loaded.positions3().unwrap();
        assert!((x[0] - 0.5).abs() < 1e-6);
    }
}
