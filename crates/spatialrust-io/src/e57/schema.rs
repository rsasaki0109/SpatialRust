use e57::PointCloud as E57PointCloud;
use spatialrust_core::{DType, FieldSemantic, PointField, PointSchema, StandardSchemas};

use crate::error::{e57_format, IoError};

/// Builds a SpatialRust schema for points read from an E57 scan descriptor.
#[must_use]
pub fn schema_for_e57_pointcloud(pc: &E57PointCloud) -> PointSchema {
    let mut schema = StandardSchemas::point_xyz();
    if pc.has_intensity() {
        schema = schema.with_field(PointField::scalar(
            "intensity",
            FieldSemantic::Intensity,
            DType::F32,
        ));
    }
    if pc.has_color() {
        schema = schema
            .with_field(PointField::scalar("r", FieldSemantic::ColorR, DType::U8))
            .with_field(PointField::scalar("g", FieldSemantic::ColorG, DType::U8))
            .with_field(PointField::scalar("b", FieldSemantic::ColorB, DType::U8));
    }
    schema
}

/// Selects an E57 prototype and export schema for writing a SpatialRust cloud.
pub fn schema_from_point_cloud(schema: &PointSchema) -> (PointSchema, Vec<e57::Record>) {
    let export_schema = export_schema_for_cloud(schema);
    let prototype = e57_prototype_from_schema(&export_schema);
    (export_schema, prototype)
}

fn export_schema_for_cloud(source: &PointSchema) -> PointSchema {
    let mut schema = StandardSchemas::point_xyz();
    if source.find_semantic(FieldSemantic::Intensity).is_some() {
        schema = schema.with_field(PointField::scalar(
            "intensity",
            FieldSemantic::Intensity,
            DType::F32,
        ));
    }
    if source.find_semantic(FieldSemantic::ColorR).is_some() {
        schema = schema
            .with_field(PointField::scalar("r", FieldSemantic::ColorR, DType::U8))
            .with_field(PointField::scalar("g", FieldSemantic::ColorG, DType::U8))
            .with_field(PointField::scalar("b", FieldSemantic::ColorB, DType::U8));
    }
    schema
}

/// Builds an E57 point prototype from a SpatialRust schema.
#[must_use]
pub fn e57_prototype_from_schema(schema: &PointSchema) -> Vec<e57::Record> {
    let mut prototype = vec![
        e57::Record::CARTESIAN_X_F32,
        e57::Record::CARTESIAN_Y_F32,
        e57::Record::CARTESIAN_Z_F32,
    ];
    if schema.find_semantic(FieldSemantic::Intensity).is_some() {
        prototype.push(e57::Record::INTENSITY_UNIT_F32);
    }
    if schema.find_semantic(FieldSemantic::ColorR).is_some() {
        prototype.push(e57::Record::COLOR_RED_U8);
        prototype.push(e57::Record::COLOR_GREEN_U8);
        prototype.push(e57::Record::COLOR_BLUE_U8);
    }
    prototype
}

/// Validates that a SpatialRust cloud can be exported to E57.
pub fn validate_export_schema(schema: &PointSchema) -> Result<(), IoError> {
    schema.validate_positions().map_err(IoError::from)?;
    if schema.find_semantic(FieldSemantic::Intensity).is_none()
        && schema.find_semantic(FieldSemantic::ColorR).is_none()
        && schema.len() > 3
    {
        return Err(e57_format(
            "E57 export supports xyz with optional intensity and rgb only".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{e57_prototype_from_schema, schema_for_e57_pointcloud};
    use e57::PointCloud as E57PointCloud;
    use spatialrust_core::{FieldSemantic, StandardSchemas};

    #[test]
    fn builds_xyz_prototype() {
        let prototype = e57_prototype_from_schema(&StandardSchemas::point_xyz());
        assert_eq!(prototype.len(), 3);
    }

    #[test]
    fn builds_xyzi_prototype() {
        let prototype = e57_prototype_from_schema(&StandardSchemas::point_xyzi());
        assert_eq!(prototype.len(), 4);
    }

    #[test]
    fn schema_from_empty_e57_scan() {
        let pc = E57PointCloud::default();
        let schema = schema_for_e57_pointcloud(&pc);
        assert!(schema.find_semantic(FieldSemantic::PositionX).is_some());
    }
}
