use std::io::BufRead;

use spatialrust_core::{
    DType, PointBuffer, PointBufferSet, PointCloud, PointSchema, SpatialMetadata,
};

use crate::error::{pcd_format, pcd_parse, IoError};
use crate::pcd::header::{read_binary_payload, PcdDataKind, PcdHeader};
use crate::pcd::schema::schema_from_pcd_fields;
use crate::{PointReader, ReadOptions};

/// Reads point clouds from PCD files or streams.
pub struct PcdReader<R: BufRead> {
    reader: R,
    header: PcdHeader,
    metadata: SpatialMetadata,
    schema: PointSchema,
    loaded: bool,
}

impl<R: BufRead> PcdReader<R> {
    /// Creates a reader and parses the PCD header eagerly.
    pub fn new(mut reader: R) -> Result<Self, IoError> {
        let (header, _) = PcdHeader::parse(&mut reader)?;
        let schema = schema_from_pcd_fields(&header.fields)?;
        let metadata = metadata_from_header(&header);
        Ok(Self { reader, header, metadata, schema, loaded: false })
    }

    /// Returns the parsed PCD header.
    #[must_use]
    pub fn header(&self) -> &PcdHeader {
        &self.header
    }

    /// Reads the point cloud payload.
    pub fn read_cloud(&mut self) -> Result<PointCloud, IoError> {
        if self.loaded {
            return Err(pcd_format("PCD reader already consumed"));
        }
        self.loaded = true;
        read_pcd_body(&self.header, &mut self.reader, self.schema.clone(), self.metadata.clone())
    }
}

impl<R: BufRead> PointReader for PcdReader<R> {
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

/// Reads a complete PCD file from any buffered reader.
pub fn read_pcd<R: BufRead>(reader: &mut R) -> Result<PointCloud, IoError> {
    let (header, _) = PcdHeader::parse(reader)?;
    let schema = schema_from_pcd_fields(&header.fields)?;
    let metadata = metadata_from_header(&header);
    read_pcd_body(&header, reader, schema, metadata)
}

fn read_pcd_body<R: BufRead>(
    header: &PcdHeader,
    reader: &mut R,
    schema: PointSchema,
    metadata: SpatialMetadata,
) -> Result<PointCloud, IoError> {
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, header.points));
    }

    match header.data {
        PcdDataKind::Ascii => read_ascii_payload(reader, header, &schema, &mut buffers)?,
        PcdDataKind::Binary => {
            let payload = read_binary_payload(reader, header.point_step() * header.points)?;
            decode_binary_payload(header, &schema, &payload, &mut buffers)?;
        }
        PcdDataKind::BinaryCompressed => {
            let payload = read_binary_compressed_payload(reader)?;
            decode_binary_compressed_payload(header, &schema, &payload, &mut buffers)?;
        }
    }

    PointCloud::try_from_parts(schema, buffers, metadata).map_err(IoError::from)
}

fn metadata_from_header(_header: &PcdHeader) -> SpatialMetadata {
    SpatialMetadata {
        frame_id: spatialrust_core::FrameId::new("pcd"),
        timestamp: spatialrust_core::Timestamp::from_nanos(0),
        sensor_origin: None,
        unit: "meter".to_owned(),
    }
}

fn read_ascii_payload<R: BufRead>(
    reader: &mut R,
    header: &PcdHeader,
    schema: &PointSchema,
    buffers: &mut PointBufferSet,
) -> Result<(), IoError> {
    let mut loaded = 0usize;
    while loaded < header.points {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Err(pcd_parse(format!(
                "unexpected EOF after {loaded} of {} ASCII points",
                header.points
            )));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut tokens = trimmed.split_whitespace();
        for field in &header.fields {
            if field.name.eq_ignore_ascii_case("rgb") {
                let token =
                    tokens.next().ok_or_else(|| pcd_parse("missing rgb token in ASCII PCD"))?;
                let packed = parse_packed_rgb(token)?;
                push_to_field(buffers, schema, "r", packed.0)?;
                push_to_field(buffers, schema, "g", packed.1)?;
                push_to_field(buffers, schema, "b", packed.2)?;
                continue;
            }

            for _ in 0..field.count {
                let token = tokens.next().ok_or_else(|| {
                    pcd_parse(format!("missing token for field `{}`", field.name))
                })?;
                let value = token
                    .parse::<f32>()
                    .map_err(|_| pcd_parse(format!("invalid ASCII value `{token}`")))?;
                push_to_field(buffers, schema, &field.name, value)?;
            }
        }
        loaded += 1;
    }
    Ok(())
}

fn parse_packed_rgb(token: &str) -> Result<(f32, f32, f32), IoError> {
    let float_value: f32 =
        token.parse().map_err(|_| pcd_parse(format!("invalid rgb value `{token}`")))?;
    let bits = float_value.to_bits();
    Ok((((bits >> 16) & 0xFF) as f32, ((bits >> 8) & 0xFF) as f32, (bits & 0xFF) as f32))
}

fn read_binary_compressed_payload<R: BufRead>(reader: &mut R) -> Result<Vec<u8>, IoError> {
    let mut size_buf = [0_u8; 4];
    reader.read_exact(&mut size_buf)?;
    let compressed_size = u32::from_le_bytes(size_buf) as usize;
    reader.read_exact(&mut size_buf)?;
    let uncompressed_size = u32::from_le_bytes(size_buf) as usize;

    let compressed = read_binary_payload(reader, compressed_size)?;
    lzf_decompress(&compressed, uncompressed_size)
}

fn lzf_decompress(input: &[u8], output_len: usize) -> Result<Vec<u8>, IoError> {
    let mut output = vec![0_u8; output_len];
    let mut ip = 0usize;
    let mut op = 0usize;

    while ip < input.len() {
        let ctrl = input[ip];
        ip += 1;

        if ctrl < 32 {
            let len = ctrl as usize + 1;
            if ip + len > input.len() || op + len > output.len() {
                return Err(pcd_format("truncated LZF literal run in binary_compressed PCD"));
            }
            output[op..op + len].copy_from_slice(&input[ip..ip + len]);
            ip += len;
            op += len;
            continue;
        }

        let mut len = (ctrl >> 5) as usize;
        let mut reference_offset = ((ctrl as usize & 0x1f) << 8) + 1;
        if len == 7 {
            if ip >= input.len() {
                return Err(pcd_format("truncated LZF length in binary_compressed PCD"));
            }
            len += input[ip] as usize;
            ip += 1;
        }
        if ip >= input.len() {
            return Err(pcd_format("truncated LZF back-reference in binary_compressed PCD"));
        }
        reference_offset += input[ip] as usize;
        ip += 1;

        let copy_len = len + 2;
        if reference_offset > op || op + copy_len > output.len() {
            return Err(pcd_format("invalid LZF back-reference in binary_compressed PCD"));
        }
        let ref_start = op - reference_offset;
        for offset in 0..copy_len {
            output[op + offset] = output[ref_start + offset];
        }
        op += copy_len;
    }

    if op != output.len() {
        return Err(pcd_format(format!(
            "LZF payload size mismatch: expected {}, decoded {op}",
            output.len()
        )));
    }
    Ok(output)
}

fn decode_binary_payload(
    header: &PcdHeader,
    schema: &PointSchema,
    payload: &[u8],
    buffers: &mut PointBufferSet,
) -> Result<(), IoError> {
    let point_step = header.point_step();
    if payload.len() != point_step * header.points {
        return Err(pcd_format(format!(
            "binary payload size mismatch: expected {}, found {}",
            point_step * header.points,
            payload.len()
        )));
    }

    for point_index in 0..header.points {
        let start = point_index * point_step;
        let end = start + point_step;
        decode_binary_point(&header.fields, &payload[start..end], schema, buffers)?;
    }
    Ok(())
}

fn decode_binary_compressed_payload(
    header: &PcdHeader,
    schema: &PointSchema,
    payload: &[u8],
    buffers: &mut PointBufferSet,
) -> Result<(), IoError> {
    let point_step = header.point_step();
    if payload.len() != point_step * header.points {
        return Err(pcd_format(format!(
            "binary_compressed payload size mismatch: expected {}, found {}",
            point_step * header.points,
            payload.len()
        )));
    }

    let mut field_base = 0usize;
    for field in &header.fields {
        let field_step = field.byte_size();
        for point_index in 0..header.points {
            let point_base = field_base + point_index * field_step;
            if field.name.eq_ignore_ascii_case("rgb") && field.count == 1 && field.size == 4 {
                let chunk = &payload[point_base..point_base + 4];
                let bits = u32::from_le_bytes(chunk.try_into().expect("rgb chunk"));
                push_to_field(buffers, schema, "r", ((bits >> 16) & 0xFF) as f32)?;
                push_to_field(buffers, schema, "g", ((bits >> 8) & 0xFF) as f32)?;
                push_to_field(buffers, schema, "b", (bits & 0xFF) as f32)?;
                continue;
            }

            for component in 0..field.count {
                let scalar_start = point_base + component * field.size;
                let scalar_end = scalar_start + field.size;
                let value = read_binary_scalar(field, &payload[scalar_start..scalar_end])?;
                push_to_field(buffers, schema, &field.name, value)?;
            }
        }
        field_base += field_step * header.points;
    }
    Ok(())
}

fn decode_binary_point(
    fields: &[crate::pcd::schema::PcdFieldSpec],
    bytes: &[u8],
    schema: &PointSchema,
    buffers: &mut PointBufferSet,
) -> Result<(), IoError> {
    let mut offset = 0usize;
    for field in fields {
        let size = field.byte_size();
        if offset + size > bytes.len() {
            return Err(pcd_parse("truncated binary PCD point"));
        }
        let field_start = offset;
        offset += size;

        if field.name.eq_ignore_ascii_case("rgb") && field.count == 1 && field.size == 4 {
            let chunk = &bytes[field_start..field_start + 4];
            let bits = u32::from_le_bytes(chunk.try_into().expect("rgb chunk"));
            push_to_field(buffers, schema, "r", ((bits >> 16) & 0xFF) as f32)?;
            push_to_field(buffers, schema, "g", ((bits >> 8) & 0xFF) as f32)?;
            push_to_field(buffers, schema, "b", (bits & 0xFF) as f32)?;
            continue;
        }

        for component in 0..field.count {
            let scalar_start = field_start + component * field.size;
            let scalar_end = scalar_start + field.size;
            let value = read_binary_scalar(field, &bytes[scalar_start..scalar_end])?;
            push_to_field(buffers, schema, &field.name, value)?;
        }
    }
    Ok(())
}

fn read_binary_scalar(
    field: &crate::pcd::schema::PcdFieldSpec,
    chunk: &[u8],
) -> Result<f32, IoError> {
    let value = match (field.kind, field.size) {
        (crate::pcd::schema::PcdType::F, 4) => f32::from_le_bytes(chunk.try_into().expect("f32")),
        (crate::pcd::schema::PcdType::F, 8) => {
            f64::from_le_bytes(chunk.try_into().expect("f64")) as f32
        }
        (crate::pcd::schema::PcdType::I, 4) => {
            i32::from_le_bytes(chunk.try_into().expect("i32")) as f32
        }
        (crate::pcd::schema::PcdType::U, 1) => f32::from(chunk[0]),
        (crate::pcd::schema::PcdType::U, 2) => {
            f32::from(u16::from_le_bytes(chunk.try_into().expect("u16")))
        }
        (crate::pcd::schema::PcdType::U, 4) => {
            u32::from_le_bytes(chunk.try_into().expect("u32")) as f32
        }
        _ => return Err(pcd_format(format!("unsupported binary field `{}`", field.name))),
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
        .ok_or_else(|| pcd_format(format!("schema missing mapped field `{name}`")))?;

    let buffer = buffers
        .get_mut(name)
        .ok_or_else(|| pcd_format(format!("buffer missing for field `{name}`")))?;

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

/// Reads a PCD file from disk.
pub fn read_pcd_file(path: impl AsRef<std::path::Path>) -> Result<PointCloud, IoError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut reader = std::io::BufReader::new(file);
    read_pcd(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::read_pcd;
    use crate::pcd::writer::{write_pcd, PcdWriteFormat};
    use spatialrust_core::{HasIntensity, HasPositions3, PointCloudBuilder, StandardSchemas};
    use std::io::Cursor;

    const SAMPLE_XYZ_ASCII: &str = "\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
";

    const SAMPLE_XYZI_ASCII: &str = "\
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
0.0 0.0 0.0 0.5
1.0 0.0 0.0 0.8
";

    fn binary_compressed_xyz_sample() -> Vec<u8> {
        let header = b"\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA binary_compressed
";
        let mut uncompressed = Vec::new();
        // PCL binary_compressed stores fields as structure-of-arrays:
        // x0, x1, y0, y1, z0, z1 for this XYZ sample.
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            uncompressed.extend_from_slice(&value.to_le_bytes());
        }

        let mut compressed = Vec::with_capacity(uncompressed.len() + 1);
        compressed.push((uncompressed.len() - 1) as u8);
        compressed.extend_from_slice(&uncompressed);

        let mut data = header.to_vec();
        data.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        data.extend_from_slice(&(uncompressed.len() as u32).to_le_bytes());
        data.extend_from_slice(&compressed);
        data
    }

    #[test]
    fn reads_ascii_xyz() {
        let mut reader = Cursor::new(SAMPLE_XYZ_ASCII.as_bytes());
        let cloud = read_pcd(&mut reader).unwrap();
        assert_eq!(cloud.len(), 3);
        let (x, y, z) = cloud.positions3().unwrap();
        assert_eq!(x, &[0.0, 1.0, 0.0]);
        assert_eq!(y, &[0.0, 0.0, 1.0]);
        assert_eq!(z, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn reads_ascii_xyzi() {
        let mut reader = Cursor::new(SAMPLE_XYZI_ASCII.as_bytes());
        let cloud = read_pcd(&mut reader).unwrap();
        assert_eq!(cloud.intensity().unwrap(), &[0.5, 0.8]);
    }

    #[test]
    fn roundtrip_ascii_xyz() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();

        let mut buffer = Vec::new();
        write_pcd(&mut buffer, &cloud, PcdWriteFormat::Ascii).unwrap();

        let mut reader = Cursor::new(buffer);
        let loaded = read_pcd(&mut reader).unwrap();
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
        write_pcd(&mut buffer, &cloud, PcdWriteFormat::Binary).unwrap();

        let mut reader = Cursor::new(buffer);
        let loaded = read_pcd(&mut reader).unwrap();
        let (x, _, _) = loaded.positions3().unwrap();
        assert!((x[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reads_binary_compressed_xyz() {
        let data = binary_compressed_xyz_sample();
        let mut reader = Cursor::new(data);
        let loaded = read_pcd(&mut reader).unwrap();
        let (x, y, z) = loaded.positions3().unwrap();
        assert_eq!(x, &[1.0, 2.0]);
        assert_eq!(y, &[3.0, 4.0]);
        assert_eq!(z, &[5.0, 6.0]);
    }
}
