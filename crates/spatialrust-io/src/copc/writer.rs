use std::path::Path;

use copc_core::{Bounds as CopcBounds, Error as CopcCoreError, Result as CopcCoreResult};
use copc_writer::{write_source, CopcPointFields, CopcPointSource, CopcWriterParams};
use spatialrust_core::{
    DType, FieldSemantic, HasPositions3, PointCloud, PointField, PointSchema,
};

use crate::error::{copc_format, copc_parse, IoError};
use crate::{PointWriter, WriteOptions};

/// Writes point clouds to COPC files.
pub struct CopcWriter;

impl PointWriter for CopcWriter {
    fn write(
        &mut self,
        _cloud: &PointCloud,
        _options: &WriteOptions,
    ) -> spatialrust_core::SpatialResult<()> {
        Err(spatialrust_core::SpatialError::InvalidArgument(
            "CopcWriter requires write_copc_file with a path ending in .copc.laz".to_owned(),
        ))
    }
}

/// Writes a point cloud to a COPC file on disk.
pub fn write_copc(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    write_copc_file(path, cloud)
}

/// Writes a point cloud to a COPC file on disk.
pub fn write_copc_file(path: impl AsRef<Path>, cloud: &PointCloud) -> Result<(), IoError> {
    validate_copc_output_path(path.as_ref())?;
    cloud.validate()?;
    if cloud.is_empty() {
        return Err(copc_format("cannot write an empty point cloud to COPC".to_owned()));
    }

    let has_color = cloud
        .schema()
        .fields()
        .iter()
        .any(|field| matches!(field.semantic, FieldSemantic::ColorR | FieldSemantic::ColorG | FieldSemantic::ColorB));
    let bounds = bounds_from_cloud(cloud)?;
    let source = PointCloudCopcSource { cloud };
    write_source(
        path.as_ref(),
        &source,
        has_color,
        bounds,
        &CopcWriterParams::default(),
    )
    .map_err(map_copc_writer_error)
}

fn validate_copc_output_path(path: &Path) -> Result<(), IoError> {
    let file_stem = path.file_stem().and_then(|stem| stem.to_str());
    let extension = path.extension().and_then(|ext| ext.to_str());
    match (file_stem, extension) {
        (Some(stem), Some(ext)) if ext.eq_ignore_ascii_case("laz") => {
            Path::new(stem)
                .extension()
                .and_then(|copc| copc.to_str())
                .filter(|copc| copc.eq_ignore_ascii_case("copc"))
                .ok_or_else(|| {
                    copc_format(format!(
                        "COPC output path must end with `.copc.laz`, got `{}`",
                        path.display()
                    ))
                })?;
            Ok(())
        }
        _ => Err(copc_format(format!(
            "COPC output path must end with `.copc.laz`, got `{}`",
            path.display()
        ))),
    }
}

fn bounds_from_cloud(cloud: &PointCloud) -> Result<CopcBounds, IoError> {
    let (x, y, z) = cloud.positions3()?;
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for index in 0..cloud.len() {
        min[0] = min[0].min(f64::from(x[index]));
        min[1] = min[1].min(f64::from(y[index]));
        min[2] = min[2].min(f64::from(z[index]));
        max[0] = max[0].max(f64::from(x[index]));
        max[1] = max[1].max(f64::from(y[index]));
        max[2] = max[2].max(f64::from(z[index]));
    }
    expand_degenerate_bounds(&mut min, &mut max);
    Ok(CopcBounds::new(
        (min[0], min[1], min[2]),
        (max[0], max[1], max[2]),
    ))
}

fn expand_degenerate_bounds(min: &mut [f64; 3], max: &mut [f64; 3]) {
    const EPS: f64 = 0.001;
    for axis in 0..3 {
        if !(max[axis] - min[axis]).is_normal() {
            min[axis] -= EPS;
            max[axis] += EPS;
        }
    }
}

fn map_copc_writer_error(error: CopcCoreError) -> IoError {
    match error {
        CopcCoreError::InvalidInput(message) => copc_format(message),
        other => copc_parse(other.to_string()),
    }
}

struct PointCloudCopcSource<'a> {
    cloud: &'a PointCloud,
}

impl CopcPointSource for PointCloudCopcSource<'_> {
    fn len(&self) -> usize {
        self.cloud.len()
    }

    fn xyz(&self, index: usize) -> (f64, f64, f64) {
        let (x, y, z) = self.cloud.positions3().expect("validated cloud positions");
        (
            f64::from(x[index]),
            f64::from(y[index]),
            f64::from(z[index]),
        )
    }

    fn fields(&self, index: usize) -> CopcCoreResult<CopcPointFields> {
        point_fields_from_cloud(self.cloud, index)
    }
}

fn point_fields_from_cloud(cloud: &PointCloud, index: usize) -> CopcCoreResult<CopcPointFields> {
    let (x, y, z) = cloud.positions3().map_err(|error| CopcCoreError::InvalidInput(error.to_string()))?;
    let schema = cloud.schema();
    Ok(CopcPointFields {
        x: f64::from(x[index]),
        y: f64::from(y[index]),
        z: f64::from(z[index]),
        intensity: read_optional_u16(cloud, schema, FieldSemantic::Intensity, "intensity", index)?
            .unwrap_or(0),
        return_number: read_optional_u8(cloud, schema, FieldSemantic::Unknown, "return_number", index)?
            .unwrap_or(1),
        number_of_returns: read_optional_u8(
            cloud,
            schema,
            FieldSemantic::Unknown,
            "number_of_returns",
            index,
        )?
        .unwrap_or(1),
        synthetic: 0,
        key_point: 0,
        withheld: 0,
        overlap: 0,
        scan_channel: 0,
        scan_direction_flag: 0,
        edge_of_flight_line: 0,
        classification: read_optional_u8(cloud, schema, FieldSemantic::Label, "classification", index)?
            .unwrap_or(0),
        user_data: 0,
        scan_angle: 0.0,
        point_source_id: read_optional_u16(
            cloud,
            schema,
            FieldSemantic::Unknown,
            "point_source_id",
            index,
        )?
        .unwrap_or(0),
        gps_time: read_optional_f64(cloud, schema, FieldSemantic::TimeOffset, "gps_time", index)?
            .unwrap_or(0.0),
        red: read_optional_u16(cloud, schema, FieldSemantic::ColorR, "red", index)?.unwrap_or(0),
        green: read_optional_u16(cloud, schema, FieldSemantic::ColorG, "green", index)?.unwrap_or(0),
        blue: read_optional_u16(cloud, schema, FieldSemantic::ColorB, "blue", index)?.unwrap_or(0),
    })
}

fn read_optional_u8(
    cloud: &PointCloud,
    schema: &PointSchema,
    semantic: FieldSemantic,
    fallback_name: &str,
    index: usize,
) -> CopcCoreResult<Option<u8>> {
    let Some(field) = find_field(schema, semantic, fallback_name) else {
        return Ok(None);
    };
    read_scalar_as_u8(cloud, field, index).map(Some)
}

fn read_optional_u16(
    cloud: &PointCloud,
    schema: &PointSchema,
    semantic: FieldSemantic,
    fallback_name: &str,
    index: usize,
) -> CopcCoreResult<Option<u16>> {
    let Some(field) = find_field(schema, semantic, fallback_name) else {
        return Ok(None);
    };
    read_scalar_as_u16(cloud, field, index).map(Some)
}

fn read_optional_f64(
    cloud: &PointCloud,
    schema: &PointSchema,
    semantic: FieldSemantic,
    fallback_name: &str,
    index: usize,
) -> CopcCoreResult<Option<f64>> {
    let Some(field) = find_field(schema, semantic, fallback_name) else {
        return Ok(None);
    };
    read_scalar_as_f64(cloud, field, index).map(Some)
}

fn find_field<'a>(
    schema: &'a PointSchema,
    semantic: FieldSemantic,
    fallback_name: &str,
) -> Option<&'a PointField> {
    schema
        .find_semantic(semantic)
        .or_else(|| schema.fields().iter().find(|field| field.name == fallback_name))
}

fn read_scalar_as_u8(cloud: &PointCloud, field: &PointField, index: usize) -> CopcCoreResult<u8> {
    let value = read_scalar_as_f64(cloud, field, index)?;
    Ok(value.round() as u8)
}

fn read_scalar_as_u16(cloud: &PointCloud, field: &PointField, index: usize) -> CopcCoreResult<u16> {
    let value = read_scalar_as_f64(cloud, field, index)?;
    Ok(value.round() as u16)
}

fn read_scalar_as_f64(cloud: &PointCloud, field: &PointField, index: usize) -> CopcCoreResult<f64> {
    use spatialrust_core::PointBuffer;

    let buffer = cloud
        .field(&field.name)
        .map_err(|error| CopcCoreError::InvalidInput(error.to_string()))?;
    match field.dtype {
        DType::F32 | DType::F16 => Ok(f64::from(
            buffer
                .as_f32()
                .map_err(|error| CopcCoreError::InvalidInput(error.to_string()))?[index],
        )),
        DType::F64 => {
            let PointBuffer::F64(values) = buffer else {
                return Err(CopcCoreError::InvalidInput(format!(
                    "unsupported dtype {:?} for field `{}`",
                    field.dtype, field.name
                )));
            };
            Ok(values[index])
        }
        DType::U8 => {
            let PointBuffer::U8(values) = buffer else {
                return Err(CopcCoreError::InvalidInput(format!(
                    "unsupported dtype {:?} for field `{}`",
                    field.dtype, field.name
                )));
            };
            Ok(f64::from(values[index]))
        }
        DType::U16 => {
            let PointBuffer::U16(values) = buffer else {
                return Err(CopcCoreError::InvalidInput(format!(
                    "unsupported dtype {:?} for field `{}`",
                    field.dtype, field.name
                )));
            };
            Ok(f64::from(values[index]))
        }
        DType::I32 => {
            let PointBuffer::I32(values) = buffer else {
                return Err(CopcCoreError::InvalidInput(format!(
                    "unsupported dtype {:?} for field `{}`",
                    field.dtype, field.name
                )));
            };
            Ok(f64::from(values[index]))
        }
        DType::U32 => {
            let PointBuffer::U32(values) = buffer else {
                return Err(CopcCoreError::InvalidInput(format!(
                    "unsupported dtype {:?} for field `{}`",
                    field.dtype, field.name
                )));
            };
            Ok(f64::from(values[index]))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{validate_copc_output_path, write_copc_file};
    use crate::copc::read_copc_file;
    use spatialrust_core::{HasPositions3, PointCloudBuilder};

    #[test]
    fn rejects_non_copc_extension() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([0.0, 0.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let path = std::env::temp_dir().join(format!("spatialrust_bad_ext_{}.laz", std::process::id()));
        let error = write_copc_file(&path, &cloud).unwrap_err();
        assert!(matches!(error, crate::IoError::CopcFormat(_)));
    }

    #[test]
    fn validate_copc_output_path_accepts_copc_laz_suffix() {
        assert!(validate_copc_output_path(Path::new("scan.copc.laz")).is_ok());
        assert!(validate_copc_output_path(Path::new("scan.laz")).is_err());
    }

    #[test]
    fn roundtrip_xyzi_cloud() {
        use spatialrust_core::{HasIntensity, StandardSchemas};

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzi());
        builder.push_point([1.0, 2.0, 3.0, 128.0]).unwrap();
        builder.push_point([4.0, 5.0, 6.0, 64.0]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!(
            "spatialrust_copc_xyzi_{}.copc.laz",
            std::process::id()
        ));
        write_copc_file(&path, &cloud).expect("write copc");
        let loaded = read_copc_file(&path).expect("read copc");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), cloud.len());
        let (src_x, src_y, src_z) = cloud.positions3().unwrap();
        let (out_x, out_y, out_z) = loaded.positions3().unwrap();
        let src_i = cloud.intensity().unwrap();
        let out_i = loaded.intensity().unwrap();
        for index in 0..cloud.len() {
            assert!((out_x[index] - src_x[index]).abs() < 1e-3);
            assert!((out_y[index] - src_y[index]).abs() < 1e-3);
            assert!((out_z[index] - src_z[index]).abs() < 1e-3);
            assert!((out_i[index] - src_i[index]).abs() < 1.0);
        }
    }

    #[test]
    fn roundtrip_xyzrgb_cloud() {
        use spatialrust_core::{PointBuffer, StandardSchemas};

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzrgb());
        builder.push_point([0.0, 0.0, 0.0, 10.0, 20.0, 30.0]).unwrap();
        builder.push_point([1.0, 1.0, 1.0, 40.0, 50.0, 60.0]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!(
            "spatialrust_copc_xyzrgb_{}.copc.laz",
            std::process::id()
        ));
        write_copc_file(&path, &cloud).expect("write copc");
        let loaded = read_copc_file(&path).expect("read copc");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), cloud.len());
        let (src_x, src_y, src_z) = cloud.positions3().unwrap();
        let (out_x, out_y, out_z) = loaded.positions3().unwrap();
        for index in 0..cloud.len() {
            assert!((out_x[index] - src_x[index]).abs() < 1e-3);
            assert!((out_y[index] - src_y[index]).abs() < 1e-3);
            assert!((out_z[index] - src_z[index]).abs() < 1e-3);
        }

        let PointBuffer::U8(src_r) = cloud.field("r").unwrap() else {
            panic!("expected u8 red channel");
        };
        let PointBuffer::U8(src_g) = cloud.field("g").unwrap() else {
            panic!("expected u8 green channel");
        };
        let PointBuffer::U8(src_b) = cloud.field("b").unwrap() else {
            panic!("expected u8 blue channel");
        };
        let PointBuffer::U16(out_r) = loaded.field("red").unwrap() else {
            panic!("expected u16 red channel");
        };
        let PointBuffer::U16(out_g) = loaded.field("green").unwrap() else {
            panic!("expected u16 green channel");
        };
        let PointBuffer::U16(out_b) = loaded.field("blue").unwrap() else {
            panic!("expected u16 blue channel");
        };
        for index in 0..cloud.len() {
            assert_eq!(u16::from(src_r[index]), out_r[index]);
            assert_eq!(u16::from(src_g[index]), out_g[index]);
            assert_eq!(u16::from(src_b[index]), out_b[index]);
        }
    }

    #[test]
    fn roundtrip_xyz_cloud() {
        let mut builder = PointCloudBuilder::xyz();
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        builder.push_point([1.5, 2.5, 3.5]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!(
            "spatialrust_copc_roundtrip_{}.copc.laz",
            std::process::id()
        ));
        write_copc_file(&path, &cloud).expect("write copc");
        let loaded = read_copc_file(&path).expect("read copc");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), cloud.len());
        let (src_x, src_y, src_z) = cloud.positions3().unwrap();
        let (out_x, out_y, out_z) = loaded.positions3().unwrap();
        for index in 0..cloud.len() {
            assert!((out_x[index] - src_x[index]).abs() < 1e-3);
            assert!((out_y[index] - src_y[index]).abs() < 1e-3);
            assert!((out_z[index] - src_z[index]).abs() < 1e-3);
        }
    }

    #[test]
    fn bounds_query_excludes_out_of_region_points() {
        use crate::copc::{read_copc_file_with_query, CopcBounds, CopcQuery};

        let mut builder = PointCloudBuilder::xyz();
        for x in 0..10 {
            for y in 0..10 {
                builder.push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0]).unwrap();
            }
        }
        builder.push_point([0.0, 0.0, 0.5]).unwrap();
        let cloud = builder.build().unwrap();

        let path = std::env::temp_dir().join(format!(
            "spatialrust_copc_bounds_query_{}.copc.laz",
            std::process::id()
        ));
        write_copc_file(&path, &cloud).expect("write copc");

        let bounds = CopcBounds::from_ranges((0.0, 0.85), (0.0, 0.85), (-0.01, 0.01));
        let loaded = read_copc_file_with_query(&path, &CopcQuery::bounds(bounds)).expect("query");
        let _ = std::fs::remove_file(&path);

        assert!(loaded.len() < cloud.len());
        assert!(loaded.len() >= 10);
        assert!(
            loaded
                .schema()
                .find_semantic(spatialrust_core::FieldSemantic::PositionX)
                .is_some()
        );
        let (x, _, _) = loaded.positions3().expect("positions3");
        assert_eq!(x.len(), loaded.len());
    }
}
