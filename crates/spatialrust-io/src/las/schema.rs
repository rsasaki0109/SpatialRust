use las::{point::Format, Header};
use spatialrust_core::{DType, FieldSemantic, PointField, PointSchema, StandardSchemas};

use crate::error::{las_format, IoError};

/// Maps a LAS-derived field name to a SpatialRust semantic.
#[must_use]
pub fn infer_las_field_semantic(name: &str) -> FieldSemantic {
    match name {
        "x" => FieldSemantic::PositionX,
        "y" => FieldSemantic::PositionY,
        "z" => FieldSemantic::PositionZ,
        "intensity" => FieldSemantic::Intensity,
        "classification" => FieldSemantic::Label,
        "return_number" => FieldSemantic::Unknown,
        "number_of_returns" => FieldSemantic::Unknown,
        "scan_angle" => FieldSemantic::Unknown,
        "point_source_id" => FieldSemantic::Unknown,
        "gps_time" => FieldSemantic::TimeOffset,
        "red" => FieldSemantic::ColorR,
        "green" => FieldSemantic::ColorG,
        "blue" => FieldSemantic::ColorB,
        _ => FieldSemantic::Unknown,
    }
}

/// Builds a SpatialRust schema for the fields exported from a LAS header.
pub fn schema_for_las_header(header: &Header) -> PointSchema {
    let format = header.point_format();
    let mut schema = StandardSchemas::point_xyzi().with_field(PointField::scalar(
        "classification",
        FieldSemantic::Label,
        DType::U8,
    ));

    if format.has_gps_time {
        schema = schema.with_field(PointField::scalar(
            "gps_time",
            FieldSemantic::TimeOffset,
            DType::F64,
        ));
    }

    if format.has_color {
        schema = schema
            .with_field(PointField::scalar("red", FieldSemantic::ColorR, DType::U16))
            .with_field(PointField::scalar("green", FieldSemantic::ColorG, DType::U16))
            .with_field(PointField::scalar("blue", FieldSemantic::ColorB, DType::U16));
    }

    schema
}

/// Selects a COPC-compatible LAS point format for writing the given cloud schema.
///
/// COPC accepts PDRF 1/3 (upgraded to 6/7) or native 6–8. SpatialRust maps clouds to PDRF 1 or 3.
pub fn schema_from_point_cloud_for_copc(
    schema: &PointSchema,
) -> Result<(Format, PointSchema), IoError> {
    let has_color = schema.fields().iter().any(|field| {
        matches!(
            field.semantic,
            FieldSemantic::ColorR | FieldSemantic::ColorG | FieldSemantic::ColorB
        )
    });

    let format = if has_color {
        Format::new(3).map_err(|error| las_format(error.to_string()))?
    } else {
        Format::new(1).map_err(|error| las_format(error.to_string()))?
    };

    let export_schema = export_schema_for_format(schema, &format);
    Ok((format, export_schema))
}

/// Selects a LAS point format for writing the given cloud schema.
pub fn schema_from_point_cloud(schema: &PointSchema) -> Result<(Format, PointSchema), IoError> {
    let has_color = schema.fields().iter().any(|field| {
        matches!(
            field.semantic,
            FieldSemantic::ColorR | FieldSemantic::ColorG | FieldSemantic::ColorB
        )
    });
    let has_gps_time =
        schema.fields().iter().any(|field| field.semantic == FieldSemantic::TimeOffset);

    let format = if has_color {
        Format::new(2).map_err(|error| las_format(error.to_string()))?
    } else if has_gps_time {
        Format::new(1).map_err(|error| las_format(error.to_string()))?
    } else {
        Format::new(0).map_err(|error| las_format(error.to_string()))?
    };

    let export_schema = export_schema_for_format(schema, &format);
    Ok((format, export_schema))
}

fn export_schema_for_format(source: &PointSchema, format: &Format) -> PointSchema {
    let mut schema = StandardSchemas::point_xyz().with_field(PointField::scalar(
        "classification",
        FieldSemantic::Label,
        DType::U8,
    ));

    if source.find_semantic(FieldSemantic::Intensity).is_some() {
        schema = schema.with_field(PointField::scalar(
            "intensity",
            FieldSemantic::Intensity,
            DType::F32,
        ));
    }

    if format.has_gps_time && source.find_semantic(FieldSemantic::TimeOffset).is_some() {
        schema = schema.with_field(PointField::scalar(
            "gps_time",
            FieldSemantic::TimeOffset,
            DType::F64,
        ));
    }

    if format.has_color {
        for (name, semantic) in [
            ("red", FieldSemantic::ColorR),
            ("green", FieldSemantic::ColorG),
            ("blue", FieldSemantic::ColorB),
        ] {
            if source.find_semantic(semantic).is_some() {
                schema = schema.with_field(PointField::scalar(name, semantic, DType::U16));
            }
        }
    }

    schema
}

#[cfg(test)]
mod tests {
    use super::{schema_for_las_header, schema_from_point_cloud};
    use las::{Builder, Header};
    use spatialrust_core::{FieldSemantic, StandardSchemas};

    #[test]
    fn builds_default_las_schema() {
        let header: Header = Builder::default().into_header().unwrap();
        let schema = schema_for_las_header(&header);
        assert!(schema.find_semantic(FieldSemantic::PositionX).is_some());
        assert!(schema.find_semantic(FieldSemantic::Label).is_some());
    }

    #[test]
    fn selects_format_zero_for_xyz_cloud() {
        let (format, _) =
            schema_from_point_cloud(&StandardSchemas::point_xyz()).expect("format selection");
        assert_eq!(format.to_u8().unwrap(), 0);
    }

    #[test]
    fn selects_format_two_for_rgb_cloud() {
        let (format, export_schema) =
            schema_from_point_cloud(&StandardSchemas::point_xyzrgb()).expect("format selection");
        assert_eq!(format.to_u8().unwrap(), 2);
        assert_eq!(export_schema.find_semantic(FieldSemantic::ColorR).unwrap().name, "red");
    }
}
