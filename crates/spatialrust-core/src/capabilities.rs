use crate::{FieldSemantic, PointCloud, SpatialError, SpatialResult};

/// Capability trait for point clouds with 3D positions.
pub trait HasPositions3 {
    /// Returns x/y/z columns as `f32` slices.
    fn positions3(&self) -> SpatialResult<(&[f32], &[f32], &[f32])>;
}

/// Capability trait for point clouds with surface normals.
pub trait HasNormals3 {
    /// Returns normal x/y/z columns as `f32` slices.
    fn normals3(&self) -> SpatialResult<(&[f32], &[f32], &[f32])>;
}

/// Capability trait for point clouds with intensity values.
pub trait HasIntensity {
    /// Returns the intensity column as an `f32` slice.
    fn intensity(&self) -> SpatialResult<&[f32]>;
}

impl HasPositions3 for PointCloud {
    fn positions3(&self) -> SpatialResult<(&[f32], &[f32], &[f32])> {
        self.schema().validate_positions()?;
        let x = self.field_name_for_semantic(FieldSemantic::PositionX)?;
        let y = self.field_name_for_semantic(FieldSemantic::PositionY)?;
        let z = self.field_name_for_semantic(FieldSemantic::PositionZ)?;
        Ok((self.field(x)?.as_f32()?, self.field(y)?.as_f32()?, self.field(z)?.as_f32()?))
    }
}

impl HasNormals3 for PointCloud {
    fn normals3(&self) -> SpatialResult<(&[f32], &[f32], &[f32])> {
        let x = self.field_name_for_semantic(FieldSemantic::NormalX)?;
        let y = self.field_name_for_semantic(FieldSemantic::NormalY)?;
        let z = self.field_name_for_semantic(FieldSemantic::NormalZ)?;
        Ok((self.field(x)?.as_f32()?, self.field(y)?.as_f32()?, self.field(z)?.as_f32()?))
    }
}

impl HasIntensity for PointCloud {
    fn intensity(&self) -> SpatialResult<&[f32]> {
        let name = self.field_name_for_semantic(FieldSemantic::Intensity)?;
        self.field(name)?.as_f32()
    }
}

impl PointCloud {
    fn field_name_for_semantic(&self, semantic: FieldSemantic) -> SpatialResult<&str> {
        self.schema()
            .find_semantic(semantic)
            .map(|field| field.name.as_str())
            .ok_or_else(|| SpatialError::MissingField(format!("{semantic:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::{HasIntensity, HasNormals3, HasPositions3};
    use crate::{PointCloudBuilder, StandardSchemas};

    #[test]
    fn positions3_capability() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();
        let (x, y, z) = cloud.positions3().unwrap();
        assert_eq!(x, &[1.0]);
        assert_eq!(y, &[2.0]);
        assert_eq!(z, &[3.0]);
    }

    #[test]
    fn normal_and_intensity_capabilities() {
        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
        builder.push_point([0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0]).unwrap();
        let cloud = builder.build().unwrap();
        let (_, _, nz) = cloud.normals3().unwrap();
        assert_eq!(cloud.intensity().unwrap(), &[0.5]);
        assert_eq!(nz, &[0.0]);
    }
}
