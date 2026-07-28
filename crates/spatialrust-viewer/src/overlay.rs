use spatialrust_math::Vec3;
use spatialrust_viz::{
    LayerId, LineListView, LinearRgba, PointCloudView, PointColor, PointStyle, PositionColumns3,
    ScalarColumn, VisualLayer, VisualPrimitive, VisualStyle,
};

use crate::{ViewerError, ViewerResult};

/// Stable category for an algorithm-debug overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OverlayKind {
    /// Per-point normal vectors.
    Normals,
    /// Voxel wire boxes.
    Voxels,
    /// Fitted plane patch.
    Plane,
    /// Cluster identifiers mapped as a scalar point attribute.
    Clusters,
    /// Registration correspondence segments.
    Correspondences,
    /// Axis-aligned bounds wire box.
    Bounds,
    /// Search-radius circle.
    SearchRadius,
}

impl OverlayKind {
    const fn slug(self) -> &'static str {
        match self {
            Self::Normals => "normals",
            Self::Voxels => "voxels",
            Self::Plane => "plane",
            Self::Clusters => "clusters",
            Self::Correspondences => "correspondences",
            Self::Bounds => "bounds",
            Self::SearchRadius => "search-radius",
        }
    }
}

/// Owned geometry retained by a debug overlay.
#[derive(Clone, Debug, PartialEq)]
pub enum OverlayGeometry {
    /// Interleaved independent line segments.
    Lines(Vec<f32>),
    /// SoA points and a scalar debug attribute.
    ScalarPoints {
        /// X positions.
        x: Vec<f32>,
        /// Y positions.
        y: Vec<f32>,
        /// Z positions.
        z: Vec<f32>,
        /// Scalar values.
        values: Vec<f32>,
        /// Stable attribute name.
        attribute: String,
    },
}

/// Owned algorithm-debug overlay with deterministic layer identity.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugOverlay {
    /// Overlay category.
    pub kind: OverlayKind,
    /// Stable visual-layer identity.
    pub id: LayerId,
    /// User-facing label.
    pub label: String,
    /// Owned geometry.
    pub geometry: OverlayGeometry,
    /// Rendering style.
    pub style: VisualStyle,
}

impl DebugOverlay {
    /// Creates per-point normal segments.
    pub fn normals(
        namespace: &str,
        positions: &[Vec3<f32>],
        normals: &[Vec3<f32>],
        scale: f32,
    ) -> ViewerResult<Self> {
        if positions.len() != normals.len() || !scale.is_finite() || scale <= 0.0 {
            return Err(ViewerError::InvalidOverlay(
                "normal positions/counts must match and scale must be positive".into(),
            ));
        }
        let mut lines = Vec::with_capacity(positions.len() * 6);
        for (&position, &normal) in positions.iter().zip(normals) {
            finite_vec(position)?;
            finite_vec(normal)?;
            push_segment(&mut lines, position, add(position, mul(normal.normalize(), scale)));
        }
        Self::lines(namespace, OverlayKind::Normals, "Normals", lines, color(0.2, 1.0, 0.2))
    }

    /// Creates wire boxes around voxel centers with one shared positive half extent.
    pub fn voxels(namespace: &str, centers: &[Vec3<f32>], half_extent: f32) -> ViewerResult<Self> {
        if !half_extent.is_finite() || half_extent <= 0.0 {
            return Err(ViewerError::InvalidOverlay(
                "voxel half extent must be finite and positive".into(),
            ));
        }
        let mut lines = Vec::with_capacity(centers.len() * 12 * 6);
        for &center in centers {
            finite_vec(center)?;
            append_box(
                &mut lines,
                sub_scalar(center, half_extent),
                add_scalar(center, half_extent),
            );
        }
        Self::lines(namespace, OverlayKind::Voxels, "Voxels", lines, color(0.0, 1.0, 1.0))
    }

    /// Creates a square wire patch tangent to a fitted plane.
    pub fn plane(
        namespace: &str,
        center: Vec3<f32>,
        normal: Vec3<f32>,
        half_extent: f32,
    ) -> ViewerResult<Self> {
        finite_vec(center)?;
        finite_vec(normal)?;
        if normal.length() <= f32::EPSILON || !half_extent.is_finite() || half_extent <= 0.0 {
            return Err(ViewerError::InvalidOverlay(
                "plane normal and half extent must be non-zero and finite".into(),
            ));
        }
        let normal = normal.normalize();
        let seed =
            if normal.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        let tangent = normal.cross(seed).normalize();
        let bitangent = normal.cross(tangent).normalize();
        let a = add(center, add(mul(tangent, half_extent), mul(bitangent, half_extent)));
        let b = add(center, add(mul(tangent, -half_extent), mul(bitangent, half_extent)));
        let c = add(center, add(mul(tangent, -half_extent), mul(bitangent, -half_extent)));
        let d = add(center, add(mul(tangent, half_extent), mul(bitangent, -half_extent)));
        let mut lines = Vec::with_capacity(24);
        for (from, to) in [(a, b), (b, c), (c, d), (d, a)] {
            push_segment(&mut lines, from, to);
        }
        Self::lines(namespace, OverlayKind::Plane, "Plane", lines, color(1.0, 1.0, 0.0))
    }

    /// Creates scalar-colored cluster points.
    pub fn clusters(
        namespace: &str,
        positions: &[Vec3<f32>],
        cluster_ids: &[u32],
    ) -> ViewerResult<Self> {
        if positions.len() != cluster_ids.len() {
            return Err(ViewerError::InvalidOverlay(
                "cluster IDs must match the position count".into(),
            ));
        }
        let mut x = Vec::with_capacity(positions.len());
        let mut y = Vec::with_capacity(positions.len());
        let mut z = Vec::with_capacity(positions.len());
        let mut values = Vec::with_capacity(positions.len());
        for (&position, &cluster_id) in positions.iter().zip(cluster_ids) {
            finite_vec(position)?;
            x.push(position.x);
            y.push(position.y);
            z.push(position.z);
            values.push(cluster_id as f32);
        }
        let max = values.iter().copied().fold(0.0_f32, f32::max).max(1.0);
        Ok(Self {
            kind: OverlayKind::Clusters,
            id: overlay_id(namespace, OverlayKind::Clusters)?,
            label: "Clusters".into(),
            geometry: OverlayGeometry::ScalarPoints {
                x,
                y,
                z,
                values,
                attribute: "cluster_id".into(),
            },
            style: VisualStyle::Points(PointStyle::try_new(
                3.0,
                PointColor::Scalar {
                    min: 0.0,
                    max: max + 1.0,
                    map: spatialrust_viz::ColorMap::Turbo,
                },
            )?),
        })
    }

    /// Creates registration correspondence segments.
    pub fn correspondences(
        namespace: &str,
        source: &[Vec3<f32>],
        target: &[Vec3<f32>],
    ) -> ViewerResult<Self> {
        if source.len() != target.len() {
            return Err(ViewerError::InvalidOverlay(
                "correspondence source/target counts must match".into(),
            ));
        }
        let mut lines = Vec::with_capacity(source.len() * 6);
        for (&from, &to) in source.iter().zip(target) {
            finite_vec(from)?;
            finite_vec(to)?;
            push_segment(&mut lines, from, to);
        }
        Self::lines(
            namespace,
            OverlayKind::Correspondences,
            "Correspondences",
            lines,
            color(1.0, 0.0, 1.0),
        )
    }

    /// Creates an axis-aligned bounds wire box.
    pub fn bounds(namespace: &str, min: Vec3<f32>, max: Vec3<f32>) -> ViewerResult<Self> {
        finite_vec(min)?;
        finite_vec(max)?;
        if min.x > max.x || min.y > max.y || min.z > max.z {
            return Err(ViewerError::InvalidOverlay(
                "bounds minimum must not exceed maximum".into(),
            ));
        }
        let mut lines = Vec::with_capacity(72);
        append_box(&mut lines, min, max);
        Self::lines(namespace, OverlayKind::Bounds, "Bounds", lines, LinearRgba::WHITE)
    }

    /// Creates a segmented XY search-radius circle.
    pub fn search_radius(
        namespace: &str,
        center: Vec3<f32>,
        radius: f32,
        segments: usize,
    ) -> ViewerResult<Self> {
        finite_vec(center)?;
        if !radius.is_finite() || radius <= 0.0 || segments < 3 {
            return Err(ViewerError::InvalidOverlay(
                "search radius must be positive with at least three segments".into(),
            ));
        }
        let mut lines = Vec::with_capacity(segments * 6);
        for index in 0..segments {
            let a = index as f32 * core::f32::consts::TAU / segments as f32;
            let b = (index + 1) as f32 * core::f32::consts::TAU / segments as f32;
            push_segment(
                &mut lines,
                add(center, Vec3::new(radius * a.cos(), radius * a.sin(), 0.0)),
                add(center, Vec3::new(radius * b.cos(), radius * b.sin(), 0.0)),
            );
        }
        Self::lines(
            namespace,
            OverlayKind::SearchRadius,
            "Search radius",
            lines,
            color(1.0, 1.0, 0.0),
        )
    }

    /// Borrows the owned overlay as a validated visual layer without copying.
    pub fn as_layer(&self) -> ViewerResult<VisualLayer<'_>> {
        let primitive = match &self.geometry {
            OverlayGeometry::Lines(lines) => VisualPrimitive::Lines(LineListView::try_new(lines)?),
            OverlayGeometry::ScalarPoints { x, y, z, values, attribute } => {
                let positions = PositionColumns3::try_new(x, y, z)?;
                let scalar = ScalarColumn::try_new(attribute, values, positions.len())?;
                VisualPrimitive::Points(
                    PointCloudView::positions_only(positions).with_scalar(scalar)?,
                )
            }
        };
        Ok(VisualLayer::try_new(
            self.id.clone(),
            self.label.clone(),
            primitive,
            self.style.clone(),
        )?)
    }

    fn lines(
        namespace: &str,
        kind: OverlayKind,
        label: &str,
        lines: Vec<f32>,
        color: LinearRgba,
    ) -> ViewerResult<Self> {
        LineListView::try_new(&lines)?;
        Ok(Self {
            kind,
            id: overlay_id(namespace, kind)?,
            label: label.into(),
            geometry: OverlayGeometry::Lines(lines),
            style: VisualStyle::Uniform(color),
        })
    }
}

fn overlay_id(namespace: &str, kind: OverlayKind) -> ViewerResult<LayerId> {
    if namespace.trim().is_empty() {
        return Err(ViewerError::InvalidOverlay("overlay namespace must not be empty".into()));
    }
    Ok(LayerId::try_new(format!("debug/{namespace}/{}", kind.slug()))?)
}

const fn color(red: f32, green: f32, blue: f32) -> LinearRgba {
    LinearRgba { red, green, blue, alpha: 1.0 }
}

fn append_box(lines: &mut Vec<f32>, min: Vec3<f32>, max: Vec3<f32>) {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    for (a, b) in [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        push_segment(lines, corners[a], corners[b]);
    }
}

fn push_segment(lines: &mut Vec<f32>, from: Vec3<f32>, to: Vec3<f32>) {
    lines.extend_from_slice(&[from.x, from.y, from.z, to.x, to.y, to.z]);
}

fn finite_vec(value: Vec3<f32>) -> ViewerResult<()> {
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(ViewerError::InvalidOverlay("overlay coordinates must be finite".into()));
    }
    Ok(())
}

fn add(lhs: Vec3<f32>, rhs: Vec3<f32>) -> Vec3<f32> {
    Vec3::new(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z)
}

fn mul(value: Vec3<f32>, scalar: f32) -> Vec3<f32> {
    Vec3::new(value.x * scalar, value.y * scalar, value.z * scalar)
}

fn sub_scalar(value: Vec3<f32>, scalar: f32) -> Vec3<f32> {
    Vec3::new(value.x - scalar, value.y - scalar, value.z - scalar)
}

fn add_scalar(value: Vec3<f32>, scalar: f32) -> Vec3<f32> {
    Vec3::new(value.x + scalar, value.y + scalar, value.z + scalar)
}

#[cfg(test)]
mod tests {
    use spatialrust_math::Vec3;
    use spatialrust_viz::VisualPrimitive;

    use super::{DebugOverlay, OverlayGeometry};

    #[test]
    fn canonical_overlays_have_stable_identity_and_geometry_counts() {
        let points = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let normals = [Vec3::new(0.0, 1.0, 0.0); 2];
        let fixtures = [
            DebugOverlay::normals("fixture", &points, &normals, 0.5).unwrap(),
            DebugOverlay::voxels("fixture", &points[..1], 0.5).unwrap(),
            DebugOverlay::plane("fixture", Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap(),
            DebugOverlay::correspondences("fixture", &points, &normals).unwrap(),
            DebugOverlay::bounds("fixture", Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
                .unwrap(),
            DebugOverlay::search_radius("fixture", points[0], 1.0, 16).unwrap(),
        ];
        let expected_segments = [2, 12, 4, 2, 12, 16];
        for (overlay, expected) in fixtures.iter().zip(expected_segments) {
            let VisualPrimitive::Lines(lines) = overlay.as_layer().unwrap().primitive else {
                panic!("fixture must be lines");
            };
            assert_eq!(lines.segment_count(), expected);
            assert!(overlay.id.as_str().starts_with("debug/fixture/"));
        }
        assert_eq!(
            DebugOverlay::normals("fixture", &points, &normals, 0.5).unwrap().id,
            fixtures[0].id
        );
    }

    #[test]
    fn cluster_overlay_preserves_point_and_scalar_counts() {
        let points = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0)];
        let overlay = DebugOverlay::clusters("segmentation", &points, &[4, 9]).unwrap();
        let OverlayGeometry::ScalarPoints { values, .. } = &overlay.geometry else {
            panic!("clusters must be scalar points");
        };
        assert_eq!(values, &[4.0, 9.0]);
        let VisualPrimitive::Points(points) = overlay.as_layer().unwrap().primitive else {
            panic!("cluster fixture must be points");
        };
        assert_eq!(points.positions.len(), 2);
        assert_eq!(points.scalar.unwrap().name, "cluster_id");
    }

    #[test]
    fn malformed_overlay_inputs_fail_closed() {
        let point = [Vec3::new(0.0, 0.0, 0.0)];
        assert!(DebugOverlay::normals("", &point, &point, 1.0).is_err());
        assert!(DebugOverlay::normals("x", &point, &[], 1.0).is_err());
        assert!(DebugOverlay::clusters("x", &point, &[]).is_err());
        assert!(DebugOverlay::search_radius("x", point[0], -1.0, 2).is_err());
        assert!(
            DebugOverlay::bounds("x", Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)).is_err()
        );
    }
}
