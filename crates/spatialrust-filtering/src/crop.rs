//! Geometric crop and field-range filters.
//!
//! These are the cheap, ubiquitous preprocessing steps: keep (or drop) points
//! inside an axis-aligned box, or keep points whose value in some field falls in
//! a range (height slices, intensity thresholds, time windows).

use spatialrust_core::{
    HasPositions3, PointBuffer, PointBufferSet, PointCloud, SpatialError, SpatialResult,
};

use crate::filter::PointCloudFilter;

/// Axis-aligned bounding box used by [`CropBox`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Inclusive lower corner `(x, y, z)`.
    pub min: [f32; 3],
    /// Inclusive upper corner `(x, y, z)`.
    pub max: [f32; 3],
}

impl Aabb {
    /// Creates a box from its corners.
    #[must_use]
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Returns whether `point` lies within the (inclusive) box.
    #[must_use]
    pub fn contains(&self, point: [f32; 3]) -> bool {
        (0..3).all(|i| point[i] >= self.min[i] && point[i] <= self.max[i])
    }
}

/// Keeps (or, when `invert` is set, drops) points inside an axis-aligned box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropBox {
    bounds: Aabb,
    invert: bool,
}

impl CropBox {
    /// Keeps points inside `bounds`.
    #[must_use]
    pub const fn new(bounds: Aabb) -> Self {
        Self { bounds, invert: false }
    }

    /// Drops points inside `bounds` (keeps everything outside).
    #[must_use]
    pub const fn inverted(bounds: Aabb) -> Self {
        Self { bounds, invert: true }
    }

    /// Computes the keep mask for `input`.
    pub fn keep_mask(&self, input: &PointCloud) -> SpatialResult<Vec<bool>> {
        if self.bounds.min.iter().zip(self.bounds.max).any(|(lo, hi)| *lo > hi) {
            return Err(SpatialError::InvalidArgument(
                "crop box min must not exceed max on any axis".to_owned(),
            ));
        }
        let (x, y, z) = input.positions3()?;
        Ok((0..input.len())
            .map(|i| {
                let inside = self.bounds.contains([x[i], y[i], z[i]]);
                inside ^ self.invert
            })
            .collect())
    }
}

impl PointCloudFilter for CropBox {
    fn name(&self) -> &'static str {
        "CropBox"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        let mask = self.keep_mask(input)?;
        gather_mask(input, &mask)
    }
}

/// Keeps (or drops) points whose value in a named field falls within a range.
#[derive(Clone, Debug, PartialEq)]
pub struct PassThrough {
    field: String,
    min: f32,
    max: f32,
    invert: bool,
}

impl PassThrough {
    /// Keeps points whose `field` value is within `[min, max]` (inclusive).
    #[must_use]
    pub fn new(field: impl Into<String>, min: f32, max: f32) -> Self {
        Self { field: field.into(), min, max, invert: false }
    }

    /// Drops points whose `field` value is within `[min, max]`.
    #[must_use]
    pub fn inverted(field: impl Into<String>, min: f32, max: f32) -> Self {
        Self { field: field.into(), min, max, invert: true }
    }

    /// Computes the keep mask for `input`.
    pub fn keep_mask(&self, input: &PointCloud) -> SpatialResult<Vec<bool>> {
        if self.min > self.max {
            return Err(SpatialError::InvalidArgument(
                "pass-through min must not exceed max".to_owned(),
            ));
        }
        let values = input.field(&self.field)?.as_f32()?;
        Ok(values
            .iter()
            .map(|&v| {
                let inside = v >= self.min && v <= self.max;
                inside ^ self.invert
            })
            .collect())
    }
}

impl PointCloudFilter for PassThrough {
    fn name(&self) -> &'static str {
        "PassThrough"
    }

    fn filter(&self, input: &PointCloud) -> SpatialResult<PointCloud> {
        let mask = self.keep_mask(input)?;
        gather_mask(input, &mask)
    }
}

/// Builds a new cloud from the points where `mask` is true, preserving schema.
fn gather_mask(input: &PointCloud, mask: &[bool]) -> SpatialResult<PointCloud> {
    let indices: Vec<usize> =
        mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();

    let mut buffers = PointBufferSet::new();
    for field in input.schema().fields() {
        let source = input.field(&field.name)?;
        buffers.insert(field.name.clone(), gather_buffer(source, &indices));
    }
    PointCloud::try_from_parts(input.schema().clone(), buffers, input.metadata().clone())
}

fn gather_buffer(source: &PointBuffer, indices: &[usize]) -> PointBuffer {
    match source {
        PointBuffer::F32(v) => PointBuffer::from_f32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::F64(v) => PointBuffer::F64(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U8(v) => PointBuffer::U8(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U16(v) => PointBuffer::U16(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::U32(v) => PointBuffer::U32(indices.iter().map(|&i| v[i]).collect()),
        PointBuffer::I32(v) => PointBuffer::I32(indices.iter().map(|&i| v[i]).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatialrust_core::{DType, FieldSemantic, PointField, PointSchema};

    fn cloud_from_xyz(points: &[[f32; 3]]) -> PointCloud {
        let schema = PointSchema::new()
            .with_field(PointField::scalar("x", FieldSemantic::PositionX, DType::F32))
            .with_field(PointField::scalar("y", FieldSemantic::PositionY, DType::F32))
            .with_field(PointField::scalar("z", FieldSemantic::PositionZ, DType::F32));
        let mut buffers = PointBufferSet::new();
        buffers
            .insert("x".to_owned(), PointBuffer::from_f32(points.iter().map(|p| p[0]).collect()));
        buffers
            .insert("y".to_owned(), PointBuffer::from_f32(points.iter().map(|p| p[1]).collect()));
        buffers
            .insert("z".to_owned(), PointBuffer::from_f32(points.iter().map(|p| p[2]).collect()));
        PointCloud::try_from_parts(schema, buffers, Default::default()).unwrap()
    }

    fn grid() -> PointCloud {
        let mut pts = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                pts.push([ix as f32, iy as f32, 0.0]);
            }
        }
        cloud_from_xyz(&pts)
    }

    #[test]
    fn crop_box_keeps_inside() {
        let cloud = grid();
        let filter = CropBox::new(Aabb::new([1.0, 1.0, -1.0], [3.0, 3.0, 1.0]));
        let out = filter.filter(&cloud).unwrap();
        // x,y each in {1,2,3} -> 3x3 = 9 points.
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn crop_box_inverted_drops_inside() {
        let cloud = grid();
        let filter = CropBox::inverted(Aabb::new([1.0, 1.0, -1.0], [3.0, 3.0, 1.0]));
        let out = filter.filter(&cloud).unwrap();
        assert_eq!(out.len(), cloud.len() - 9);
    }

    #[test]
    fn pass_through_on_position_field() {
        let cloud = grid();
        // Keep points with x in [2, 4] -> x in {2,3,4} -> 3 columns * 5 rows = 15.
        let filter = PassThrough::new("x", 2.0, 4.0);
        let out = filter.filter(&cloud).unwrap();
        assert_eq!(out.len(), 15);
    }

    #[test]
    fn errors_on_inverted_range_and_missing_field() {
        let cloud = grid();
        assert!(CropBox::new(Aabb::new([3.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
            .keep_mask(&cloud)
            .is_err());
        assert!(PassThrough::new("x", 5.0, 1.0).keep_mask(&cloud).is_err());
        assert!(PassThrough::new("intensity", 0.0, 1.0).keep_mask(&cloud).is_err());
    }
}
