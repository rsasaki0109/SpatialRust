use crate::{VizError, VizResult};

#[cfg(feature = "core")]
use spatialrust_core::HasPositions3;

/// Borrowed structure-of-arrays XYZ position columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionColumns3<'a> {
    /// X coordinates.
    pub x: &'a [f32],
    /// Y coordinates.
    pub y: &'a [f32],
    /// Z coordinates.
    pub z: &'a [f32],
}

impl<'a> PositionColumns3<'a> {
    /// Creates a borrowed position view without copying.
    pub fn try_new(x: &'a [f32], y: &'a [f32], z: &'a [f32]) -> VizResult<Self> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(VizError::InvalidGeometry(
                "XYZ position columns must have equal lengths".into(),
            ));
        }
        Ok(Self { x, y, z })
    }

    /// Number of positions.
    #[must_use]
    pub fn len(self) -> usize {
        self.x.len()
    }

    /// Whether the view contains no positions.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.x.is_empty()
    }
}

/// Borrowed structure-of-arrays RGB color columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb8Columns<'a> {
    /// Red components.
    pub red: &'a [u8],
    /// Green components.
    pub green: &'a [u8],
    /// Blue components.
    pub blue: &'a [u8],
}

impl<'a> Rgb8Columns<'a> {
    /// Creates validated RGB columns for `point_count` points.
    pub fn try_new(
        red: &'a [u8],
        green: &'a [u8],
        blue: &'a [u8],
        point_count: usize,
    ) -> VizResult<Self> {
        if red.len() != point_count || green.len() != point_count || blue.len() != point_count {
            return Err(VizError::InvalidGeometry(
                "RGB columns must match the position count".into(),
            ));
        }
        Ok(Self { red, green, blue })
    }
}

/// Named borrowed scalar values used for point coloring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarColumn<'a> {
    /// Stable attribute name such as `intensity` or `cluster_id`.
    pub name: &'a str,
    /// One scalar per point.
    pub values: &'a [f32],
}

impl<'a> ScalarColumn<'a> {
    /// Creates a scalar column matching `point_count`.
    pub fn try_new(name: &'a str, values: &'a [f32], point_count: usize) -> VizResult<Self> {
        if name.trim().is_empty() {
            return Err(VizError::InvalidGeometry("scalar column name must not be empty".into()));
        }
        if values.len() != point_count {
            return Err(VizError::InvalidGeometry(
                "scalar column must match the position count".into(),
            ));
        }
        Ok(Self { name, values })
    }
}

/// Borrowed point-cloud geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCloudView<'a> {
    /// Point positions.
    pub positions: PositionColumns3<'a>,
    /// Optional RGB attributes.
    pub rgb: Option<Rgb8Columns<'a>>,
    /// Optional scalar attribute.
    pub scalar: Option<ScalarColumn<'a>>,
}

impl<'a> PointCloudView<'a> {
    /// Creates a position-only point-cloud view.
    #[must_use]
    pub const fn positions_only(positions: PositionColumns3<'a>) -> Self {
        Self { positions, rgb: None, scalar: None }
    }

    /// Attaches validated RGB attributes.
    pub fn with_rgb(mut self, rgb: Rgb8Columns<'a>) -> VizResult<Self> {
        if rgb.red.len() != self.positions.len() {
            return Err(VizError::InvalidGeometry(
                "RGB columns must match the position count".into(),
            ));
        }
        self.rgb = Some(rgb);
        Ok(self)
    }

    /// Attaches a validated scalar attribute.
    pub fn with_scalar(mut self, scalar: ScalarColumn<'a>) -> VizResult<Self> {
        if scalar.values.len() != self.positions.len() {
            return Err(VizError::InvalidGeometry(
                "scalar column must match the position count".into(),
            ));
        }
        self.scalar = Some(scalar);
        Ok(self)
    }
}

/// Borrowed pairs of line endpoints stored as interleaved XYZ values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineListView<'a> {
    /// Interleaved XYZ endpoint data; every six values form one segment.
    pub positions_xyz: &'a [f32],
}

impl<'a> LineListView<'a> {
    /// Creates a validated line-list view.
    pub fn try_new(positions_xyz: &'a [f32]) -> VizResult<Self> {
        if positions_xyz.len() % 6 != 0 {
            return Err(VizError::InvalidGeometry(
                "line-list positions must contain six values per segment".into(),
            ));
        }
        Ok(Self { positions_xyz })
    }

    /// Number of line segments.
    #[must_use]
    pub fn segment_count(self) -> usize {
        self.positions_xyz.len() / 6
    }
}

/// Borrowed indexed triangle mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMeshView<'a> {
    /// Interleaved XYZ vertex positions.
    pub positions_xyz: &'a [f32],
    /// Three vertex indices per triangle.
    pub indices: &'a [u32],
}

impl<'a> TriangleMeshView<'a> {
    /// Creates a validated indexed mesh view.
    pub fn try_new(positions_xyz: &'a [f32], indices: &'a [u32]) -> VizResult<Self> {
        if positions_xyz.len() % 3 != 0 || indices.len() % 3 != 0 {
            return Err(VizError::InvalidGeometry(
                "mesh positions and indices must contain complete triples".into(),
            ));
        }
        let vertex_count = positions_xyz.len() / 3;
        if indices.iter().any(|&index| index as usize >= vertex_count) {
            return Err(VizError::InvalidGeometry("mesh index is out of bounds".into()));
        }
        Ok(Self { positions_xyz, indices })
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(self) -> usize {
        self.positions_xyz.len() / 3
    }

    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(self) -> usize {
        self.indices.len() / 3
    }
}

/// Backend-independent borrowed visual geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VisualPrimitive<'a> {
    /// Point-cloud geometry.
    Points(PointCloudView<'a>),
    /// Independent line segments.
    Lines(LineListView<'a>),
    /// Indexed triangle mesh.
    Triangles(TriangleMeshView<'a>),
}

/// Creates a zero-copy point view from a core position capability.
///
/// The returned view borrows the source's structure-of-arrays columns. This
/// function never interleaves, uploads, or otherwise copies point data.
#[cfg(feature = "core")]
pub fn point_cloud_positions<'a>(source: &'a impl HasPositions3) -> VizResult<PointCloudView<'a>> {
    let (x, y, z) = source
        .positions3()
        .map_err(|error| VizError::InvalidGeometry(format!("position capability: {error}")))?;
    Ok(PointCloudView::positions_only(PositionColumns3::try_new(x, y, z)?))
}

#[cfg(test)]
mod tests {
    use super::{
        LineListView, PointCloudView, PositionColumns3, Rgb8Columns, ScalarColumn, TriangleMeshView,
    };

    #[test]
    fn borrowed_point_view_preserves_source_identity() {
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        let z = [5.0, 6.0];
        let positions = PositionColumns3::try_new(&x, &y, &z).unwrap();
        let rgb = Rgb8Columns::try_new(&[1, 2], &[3, 4], &[5, 6], 2).unwrap();
        let scalar = ScalarColumn::try_new("intensity", &[0.1, 0.2], 2).unwrap();
        let view = PointCloudView::positions_only(positions)
            .with_rgb(rgb)
            .unwrap()
            .with_scalar(scalar)
            .unwrap();

        assert!(core::ptr::eq(view.positions.x.as_ptr(), x.as_ptr()));
        assert_eq!(view.scalar.unwrap().name, "intensity");
    }

    #[test]
    fn rejects_mismatched_and_invalid_geometry() {
        assert!(PositionColumns3::try_new(&[0.0], &[], &[0.0]).is_err());
        assert!(LineListView::try_new(&[0.0; 5]).is_err());
        assert!(TriangleMeshView::try_new(&[0.0; 9], &[0, 1, 3]).is_err());
    }

    #[cfg(feature = "core")]
    #[test]
    fn core_adapter_borrows_point_cloud_columns() {
        use spatialrust_core::{HasPositions3, PointCloudBuilder, StandardSchemas};

        let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyz());
        builder.push_point([1.0, 2.0, 3.0]).unwrap();
        let cloud = builder.build().unwrap();
        let (x, _, _) = cloud.positions3().unwrap();
        let view = super::point_cloud_positions(&cloud).unwrap();
        assert!(core::ptr::eq(view.positions.x.as_ptr(), x.as_ptr()));
    }
}
