use std::io::{Seek, Write};

use las::point::{Classification, Format};
use las::{Builder, Color, Header, Point, Writer};
use spatialrust_core::{FieldSemantic, HasPositions3, PointCloud, PointField, PointSchema};

use crate::error::{las_format, las_parse, IoError};
use crate::las::schema::schema_from_point_cloud;
use crate::{PointWriter, WriteOptions};

/// Output encoding for LAS writers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LasWriteFormat {
    /// Uncompressed LAS.
    #[default]
    Las,
    /// LAZ compression (requires `io-laz` feature).
    Laz,
}

/// Writes point clouds to LAS/LAZ files or streams.
pub struct LasWriter<W: Write + Seek + Send + Sync + 'static> {
    writer: W,
    format: LasWriteFormat,
}

impl<W: Write + Seek + Send + Sync + 'static> LasWriter<W> {
    /// Creates a new LAS writer.
    #[must_use]
    pub const fn new(writer: W, format: LasWriteFormat) -> Self {
        Self { writer, format }
    }
}

impl<W: Write + Seek + Send + Sync + 'static> PointWriter for LasWriter<W> {
    fn write(
        &mut self,
        cloud: &PointCloud,
        _options: &WriteOptions,
    ) -> spatialrust_core::SpatialResult<()> {
        let buffer = write_las(std::io::Cursor::new(Vec::new()), cloud, self.format)
            .map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))?;
        self.writer
            .write_all(buffer.get_ref())
            .map_err(|error| spatialrust_core::SpatialError::Io(error.to_string()))
    }
}

/// Writes a point cloud to a LAS/LAZ stream and returns the inner writer.
pub fn write_las<W: Write + Seek + Send + Sync + 'static>(
    writer: W,
    cloud: &PointCloud,
    format: LasWriteFormat,
) -> Result<W, IoError> {
    cloud.validate()?;
    if format == LasWriteFormat::Laz {
        #[cfg(not(feature = "io-laz"))]
        {
            return Err(crate::error::laz_format(
                "LAZ output requires the io-laz feature".to_owned(),
            ));
        }
    }

    let (point_format, export_schema) = schema_from_point_cloud(cloud.schema())?;
    let header = header_from_cloud(point_format, format)?;
    let mut las_writer =
        Writer::new(writer, header).map_err(|error| las_format(error.to_string()))?;

    for index in 0..cloud.len() {
        let point = point_from_cloud(cloud, &export_schema, index, point_format)?;
        las_writer.write_point(point).map_err(|error| las_format(error.to_string()))?;
    }

    las_writer.into_inner().map_err(|error| las_format(error.to_string()))
}

/// Writes a point cloud to a LAS/LAZ file on disk.
pub fn write_las_file(
    path: impl AsRef<std::path::Path>,
    cloud: &PointCloud,
    format: LasWriteFormat,
) -> Result<(), IoError> {
    if format == LasWriteFormat::Laz {
        #[cfg(not(feature = "io-laz"))]
        {
            return Err(crate::error::laz_format(
                "LAZ output requires the io-laz feature".to_owned(),
            ));
        }
    }

    let (point_format, export_schema) = schema_from_point_cloud(cloud.schema())?;
    let header = header_from_cloud(point_format, format)?;
    let mut las_writer =
        Writer::from_path(path.as_ref(), header).map_err(|error| las_format(error.to_string()))?;

    for index in 0..cloud.len() {
        let point = point_from_cloud(cloud, &export_schema, index, point_format)?;
        las_writer.write_point(point).map_err(|error| las_format(error.to_string()))?;
    }

    las_writer.close().map_err(|error| las_format(error.to_string()))
}

fn header_from_cloud(point_format: Format, format: LasWriteFormat) -> Result<Header, IoError> {
    let mut builder = Builder::from((1, 2));
    builder.point_format = point_format;
    builder.system_identifier = "SpatialRust".to_owned();
    builder.generating_software = "SpatialRust".to_owned();

    if format == LasWriteFormat::Laz {
        builder.point_format.is_compressed = true;
    }

    builder.into_header().map_err(|error| las_format(error.to_string()))
}

pub(crate) fn point_from_cloud(
    cloud: &PointCloud,
    schema: &PointSchema,
    index: usize,
    format: Format,
) -> Result<Point, IoError> {
    let (x, y, z) = cloud.positions3()?;
    let mut point = Point {
        x: f64::from(x[index]),
        y: f64::from(y[index]),
        z: f64::from(z[index]),
        ..Default::default()
    };

    for field in schema.fields() {
        if matches!(
            field.semantic,
            FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ
        ) {
            continue;
        }
        let Some(field_name) = cloud_field_name_for_export(cloud, field) else {
            continue;
        };
        let value = read_cloud_field(cloud, field_name, index)?;
        apply_las_field(&mut point, field, value, format)?;
    }

    Ok(point)
}

fn cloud_field_name_for_export<'a>(
    cloud: &'a PointCloud,
    export_field: &'a PointField,
) -> Option<&'a str> {
    if cloud.field(&export_field.name).is_ok() {
        return Some(export_field.name.as_str());
    }
    cloud
        .schema()
        .find_semantic(export_field.semantic)
        .map(|field| field.name.as_str())
        .filter(|name| cloud.field(name).is_ok())
}

fn read_cloud_field(cloud: &PointCloud, field_name: &str, index: usize) -> Result<f64, IoError> {
    let buffer = cloud.field(field_name)?;
    Ok(match buffer {
        PointBuffer::F32(values) => f64::from(values[index]),
        PointBuffer::F64(values) => values[index],
        PointBuffer::U8(values) => f64::from(values[index]),
        PointBuffer::U16(values) => f64::from(values[index]),
        PointBuffer::I32(values) => f64::from(values[index]),
        PointBuffer::U32(values) => f64::from(values[index]),
    })
}

use spatialrust_core::PointBuffer;

fn apply_las_field(
    point: &mut Point,
    field: &PointField,
    value: f64,
    format: Format,
) -> Result<(), IoError> {
    match field.semantic {
        FieldSemantic::Intensity => point.intensity = value.round() as u16,
        FieldSemantic::Label => {
            point.classification = Classification::new(value.round() as u8)
                .map_err(|error| las_parse(error.to_string()))?;
        }
        FieldSemantic::TimeOffset => {
            if format.has_gps_time {
                point.gps_time = Some(value);
            }
        }
        FieldSemantic::ColorR | FieldSemantic::ColorG | FieldSemantic::ColorB => {
            if format.has_color {
                let color = point.color.get_or_insert_with(Color::default);
                match field.semantic {
                    FieldSemantic::ColorR => color.red = value.round() as u16,
                    FieldSemantic::ColorG => color.green = value.round() as u16,
                    FieldSemantic::ColorB => color.blue = value.round() as u16,
                    _ => {}
                }
            }
        }
        FieldSemantic::PositionX | FieldSemantic::PositionY | FieldSemantic::PositionZ => {}
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{write_las, write_las_file, LasWriteFormat};
    use spatialrust_core::PointCloudBuilder;
    use std::io::Cursor;

    #[test]
    fn writes_xyz_cloud() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();
        let bytes = write_las(Cursor::new(Vec::new()), &cloud, LasWriteFormat::Las).unwrap();
        assert!(!bytes.get_ref().is_empty());
    }

    #[test]
    fn writes_xyzirgb_cloud_with_u8_color_fields() {
        use spatialrust_core::StandardSchemas;

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzirgb());
        builder.push_point([1.0, 2.0, 3.0, 100.0, 255.0, 128.0, 64.0]).unwrap();
        let cloud = builder.build().unwrap();
        let bytes = write_las(Cursor::new(Vec::new()), &cloud, LasWriteFormat::Las).unwrap();
        assert!(!bytes.get_ref().is_empty());
    }

    #[cfg(feature = "io-laz")]
    #[test]
    fn writes_laz_file() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let path =
            std::env::temp_dir().join(format!("spatialrust_laz_write_{}.laz", std::process::id()));
        write_las_file(&path, &cloud, LasWriteFormat::Laz).unwrap();
        let loaded = crate::las::read_las_file(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(loaded.len(), 1);
    }
}
