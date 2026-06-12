use std::io::Write;

use spatialrust_core::{DType, FieldSemantic, PointCloud, PointField, PointSchema};

use crate::error::{pcd_format, IoError};
use crate::pcd::schema::{infer_field_semantic, PcdFieldSpec, PcdType};
use crate::{PointWriter, WriteOptions};

/// Output encoding for PCD writers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PcdWriteFormat {
    /// ASCII PCD.
    #[default]
    Ascii,
    /// Binary little-endian PCD.
    Binary,
}

/// Writes point clouds to PCD files or streams.
pub struct PcdWriter<W: Write> {
    writer: W,
    format: PcdWriteFormat,
}

impl<W: Write> PcdWriter<W> {
    /// Creates a new PCD writer.
    #[must_use]
    pub const fn new(writer: W, format: PcdWriteFormat) -> Self {
        Self { writer, format }
    }
}

impl<W: Write> PointWriter for PcdWriter<W> {
    fn write(
        &mut self,
        cloud: &PointCloud,
        _options: &WriteOptions,
    ) -> spatialrust_core::SpatialResult<()> {
        write_pcd(&mut self.writer, cloud, self.format)
            .map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))
    }
}

/// Writes a point cloud to a PCD stream.
pub fn write_pcd<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    format: PcdWriteFormat,
) -> Result<(), IoError> {
    cloud.validate()?;
    let specs = pcd_specs_from_schema(cloud.schema())?;
    write_header(writer, cloud, &specs, format)?;

    match format {
        PcdWriteFormat::Ascii => write_ascii_payload(writer, cloud, &specs)?,
        PcdWriteFormat::Binary => write_binary_payload(writer, cloud, &specs)?,
    }
    Ok(())
}

/// Writes a point cloud to a PCD file on disk.
pub fn write_pcd_file(
    path: impl AsRef<std::path::Path>,
    cloud: &PointCloud,
    format: PcdWriteFormat,
) -> Result<(), IoError> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    write_pcd(&mut writer, cloud, format)
}

fn write_header<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    specs: &[PcdFieldSpec],
    format: PcdWriteFormat,
) -> Result<(), IoError> {
    let data = match format {
        PcdWriteFormat::Ascii => "ascii",
        PcdWriteFormat::Binary => "binary",
    };

    writeln!(writer, "# .PCD v0.7 - Point Cloud Data file format")?;
    writeln!(writer, "VERSION 0.7")?;
    write_list_line(writer, "FIELDS", specs.iter().map(|spec| spec.name.as_str()))?;
    write_list_line(writer, "SIZE", specs.iter().map(|spec| spec.size.to_string()))?;
    write_list_line(
        writer,
        "TYPE",
        specs.iter().map(|spec| match spec.kind {
            PcdType::I => "I",
            PcdType::U => "U",
            PcdType::F => "F",
        }),
    )?;
    write_list_line(writer, "COUNT", specs.iter().map(|spec| spec.count.to_string()))?;
    writeln!(writer, "WIDTH {}", cloud.len())?;
    writeln!(writer, "HEIGHT 1")?;
    writeln!(writer, "VIEWPOINT 0 0 0 1 0 0 0")?;
    writeln!(writer, "POINTS {}", cloud.len())?;
    writeln!(writer, "DATA {data}")?;
    Ok(())
}

fn write_list_line<W: Write, I, S>(writer: &mut W, key: &str, values: I) -> Result<(), IoError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    write!(writer, "{key}")?;
    for value in values {
        write!(writer, " {}", value.as_ref())?;
    }
    writeln!(writer)?;
    Ok(())
}

fn write_ascii_payload<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    specs: &[PcdFieldSpec],
) -> Result<(), IoError> {
    for point_index in 0..cloud.len() {
        let mut first = true;
        for spec in specs {
            if !first {
                write!(writer, " ")?;
            }
            first = false;
            if spec.name.eq_ignore_ascii_case("rgb") {
                let r = read_scalar(cloud, "r", point_index)? as u32;
                let g = read_scalar(cloud, "g", point_index)? as u32;
                let b = read_scalar(cloud, "b", point_index)? as u32;
                let packed = (r << 16) | (g << 8) | b;
                write!(writer, "{packed}")?;
                continue;
            }
            write!(writer, "{}", read_scalar(cloud, &spec.name, point_index)?)?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

fn write_binary_payload<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    specs: &[PcdFieldSpec],
) -> Result<(), IoError> {
    for point_index in 0..cloud.len() {
        for spec in specs {
            if spec.name.eq_ignore_ascii_case("rgb") {
                let r = read_scalar(cloud, "r", point_index)? as u32;
                let g = read_scalar(cloud, "g", point_index)? as u32;
                let b = read_scalar(cloud, "b", point_index)? as u32;
                let packed = (r << 16) | (g << 8) | b;
                writer.write_all(&packed.to_le_bytes())?;
                continue;
            }
            write_binary_scalar(writer, cloud, spec, point_index)?;
        }
    }
    Ok(())
}

fn write_binary_scalar<W: Write>(
    writer: &mut W,
    cloud: &PointCloud,
    spec: &PcdFieldSpec,
    point_index: usize,
) -> Result<(), IoError> {
    let field = cloud
        .schema()
        .fields()
        .iter()
        .find(|field| field.name == spec.name)
        .ok_or_else(|| pcd_format(format!("missing field `{}`", spec.name)))?;
    let buffer = cloud.field(&field.name).map_err(IoError::from)?;

    match (field.dtype, spec.kind, spec.size) {
        (DType::F32 | DType::F16, PcdType::F, 4) => {
            let value = buffer.as_f32().map_err(IoError::from)?[point_index];
            writer.write_all(&value.to_le_bytes())?;
        }
        (DType::F64, PcdType::F, 8) => {
            let PointBuffer::F64(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        (DType::I32, PcdType::I, 4) => {
            let PointBuffer::I32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        (DType::U8, PcdType::U, 1) => {
            let PointBuffer::U8(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&[values[point_index]])?;
        }
        (DType::U16, PcdType::U, 2) => {
            let PointBuffer::U16(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        (DType::U32, PcdType::U, 4) => {
            let PointBuffer::U32(values) = buffer else {
                return Err(IoError::Core(spatialrust_core::SpatialError::UnsupportedDType(
                    field.dtype,
                )));
            };
            writer.write_all(&values[point_index].to_le_bytes())?;
        }
        _ => return Err(pcd_format(format!("cannot encode field `{}` to PCD", field.name))),
    }
    Ok(())
}

use spatialrust_core::PointBuffer;

fn read_scalar(cloud: &PointCloud, name: &str, point_index: usize) -> Result<f32, IoError> {
    let field = cloud
        .schema()
        .fields()
        .iter()
        .find(|field| field.name == name)
        .ok_or_else(|| pcd_format(format!("missing field `{name}`")))?;
    let buffer = cloud.field(name).map_err(IoError::from)?;
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

fn pcd_specs_from_schema(schema: &PointSchema) -> Result<Vec<PcdFieldSpec>, IoError> {
    let mut specs = Vec::new();
    let mut index = 0;
    while index < schema.len() {
        let field = &schema.fields()[index];
        if matches!(
            field.semantic,
            FieldSemantic::ColorR | FieldSemantic::ColorG | FieldSemantic::ColorB
        ) && field.semantic == FieldSemantic::ColorR
            && index + 2 < schema.len()
            && schema.fields()[index + 1].semantic == FieldSemantic::ColorG
            && schema.fields()[index + 2].semantic == FieldSemantic::ColorB
        {
            specs.push(PcdFieldSpec { name: "rgb".into(), size: 4, kind: PcdType::F, count: 1 });
            index += 3;
            continue;
        }

        specs.push(pcd_spec_from_field(field)?);
        index += 1;
    }
    Ok(specs)
}

fn pcd_spec_from_field(field: &PointField) -> Result<PcdFieldSpec, IoError> {
    let name = match field.semantic {
        FieldSemantic::PositionX => "x",
        FieldSemantic::PositionY => "y",
        FieldSemantic::PositionZ => "z",
        FieldSemantic::NormalX => "normal_x",
        FieldSemantic::NormalY => "normal_y",
        FieldSemantic::NormalZ => "normal_z",
        FieldSemantic::Intensity => "intensity",
        FieldSemantic::Curvature => "curvature",
        FieldSemantic::Ring => "ring",
        FieldSemantic::TimeOffset => "timestamp",
        FieldSemantic::Label => "label",
        _ => field.name.as_str(),
    }
    .to_owned();

    let (kind, size) = match field.dtype {
        DType::F32 | DType::F16 => (PcdType::F, 4),
        DType::F64 => (PcdType::F, 8),
        DType::I32 => (PcdType::I, 4),
        DType::U8 => (PcdType::U, 1),
        DType::U16 => (PcdType::U, 2),
        DType::U32 => (PcdType::U, 4),
    };

    let _semantic = infer_field_semantic(&name);
    Ok(PcdFieldSpec { name, size, kind, count: field.components })
}
