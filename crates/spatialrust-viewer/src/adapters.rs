#[cfg(any(feature = "scene", feature = "mapping", feature = "camera", feature = "semantic"))]
use spatialrust_math::Vec3;
#[cfg(any(feature = "scene", feature = "mapping", feature = "camera"))]
use spatialrust_viz::LinearRgba;
use spatialrust_viz::{
    LayerId, LineListView, PointCloudView, PositionColumns3, Rgb8Columns, ScalarColumn,
    TriangleMeshView, VisualLayer, VisualPrimitive, VisualStyle,
};
#[cfg(any(feature = "scene", feature = "semantic"))]
use spatialrust_viz::{PointColor, PointStyle};

#[cfg(any(feature = "scene", feature = "mapping", feature = "camera", feature = "semantic"))]
use crate::ViewerError;
use crate::ViewerResult;

/// Evidence describing one explicit scene-to-visual adaptation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterReceipt {
    /// Address identity of the first source element, or zero for empty input.
    pub source_identity: usize,
    /// Number of source primitives.
    pub source_count: usize,
    /// Number of output points, segments, or triangles.
    pub output_count: usize,
    /// Exact bytes allocated for generated visual geometry.
    pub generated_bytes: usize,
}

/// Owned geometry created by an explicit scene adapter.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaptedGeometry {
    /// SoA point positions with optional RGB and scalar columns.
    Points {
        /// X coordinates.
        x: Vec<f32>,
        /// Y coordinates.
        y: Vec<f32>,
        /// Z coordinates.
        z: Vec<f32>,
        /// Optional SoA red, green, and blue columns.
        rgb: Option<[Vec<u8>; 3]>,
        /// Optional scalar attribute name and values.
        scalar: Option<(String, Vec<f32>)>,
    },
    /// Interleaved independent line segments.
    Lines(Vec<f32>),
    /// Interleaved positions and triangle indices.
    Triangles {
        /// XYZ vertices.
        positions: Vec<f32>,
        /// Triangle indices.
        indices: Vec<u32>,
    },
}

/// One owned visual layer and its exact adapter receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptedVisual {
    /// Stable layer identity.
    pub id: LayerId,
    /// User-facing label.
    pub label: String,
    /// Generated geometry.
    pub geometry: AdaptedGeometry,
    /// Visual style.
    pub style: VisualStyle,
    /// Adaptation evidence.
    pub receipt: AdapterReceipt,
}

impl AdaptedVisual {
    /// Borrows generated storage as a visual layer without another copy.
    pub fn as_layer(&self) -> ViewerResult<VisualLayer<'_>> {
        let primitive = match &self.geometry {
            AdaptedGeometry::Points { x, y, z, rgb, scalar } => {
                let positions = PositionColumns3::try_new(x, y, z)?;
                let mut points = PointCloudView::positions_only(positions);
                if let Some([red, green, blue]) = rgb {
                    points = points.with_rgb(Rgb8Columns::try_new(
                        red,
                        green,
                        blue,
                        positions.len(),
                    )?)?;
                }
                if let Some((name, values)) = scalar {
                    points = points.with_scalar(ScalarColumn::try_new(
                        name,
                        values,
                        positions.len(),
                    )?)?;
                }
                VisualPrimitive::Points(points)
            }
            AdaptedGeometry::Lines(lines) => VisualPrimitive::Lines(LineListView::try_new(lines)?),
            AdaptedGeometry::Triangles { positions, indices } => {
                VisualPrimitive::Triangles(TriangleMeshView::try_new(positions, indices)?)
            }
        };
        Ok(VisualLayer::try_new(
            self.id.clone(),
            self.label.clone(),
            primitive,
            normalized_style(&self.geometry, &self.style),
        )?)
    }
}

fn normalized_style(_geometry: &AdaptedGeometry, style: &VisualStyle) -> VisualStyle {
    style.clone()
}

/// Borrows a reconstructed mesh directly, preserving source pointer identity.
#[cfg(feature = "scene")]
pub fn mesh_visual<'a>(
    id: LayerId,
    label: impl Into<String>,
    mesh: &'a spatialrust_scene::TriangleMesh,
    color: LinearRgba,
) -> ViewerResult<(VisualLayer<'a>, AdapterReceipt)> {
    let view = TriangleMeshView::try_new(&mesh.positions, &mesh.indices)?;
    let layer = VisualLayer::try_new(
        id,
        label,
        VisualPrimitive::Triangles(view),
        VisualStyle::Uniform(color),
    )?;
    Ok((
        layer,
        AdapterReceipt {
            source_identity: mesh.positions.as_ptr() as usize,
            source_count: mesh.triangle_count(),
            output_count: view.triangle_count(),
            generated_bytes: 0,
        },
    ))
}

/// Converts surfel centers/radii to point geometry.
#[cfg(feature = "scene")]
pub fn surfel_visual(
    namespace: &str,
    cloud: &spatialrust_scene::SurfelCloud,
) -> ViewerResult<AdaptedVisual> {
    let surfels = cloud.as_slice();
    let mut x = Vec::with_capacity(surfels.len());
    let mut y = Vec::with_capacity(surfels.len());
    let mut z = Vec::with_capacity(surfels.len());
    let mut radius = Vec::with_capacity(surfels.len());
    for surfel in surfels {
        x.push(surfel.position.x);
        y.push(surfel.position.y);
        z.push(surfel.position.z);
        radius.push(surfel.radius);
    }
    point_scalar_visual(
        namespace,
        "surfels",
        "Surfels",
        source_identity(surfels),
        surfels.len(),
        PointScalarColumns { x, y, z, name: "radius", values: radius },
    )
}

/// Converts Gaussian means, colors, and opacity to point geometry.
#[cfg(feature = "scene-gaussian")]
pub fn gaussian_visual(
    namespace: &str,
    scene: &spatialrust_scene::GaussianScene,
) -> ViewerResult<AdaptedVisual> {
    let primitives = scene.primitives();
    let mut x = Vec::with_capacity(primitives.len());
    let mut y = Vec::with_capacity(primitives.len());
    let mut z = Vec::with_capacity(primitives.len());
    let mut rgb = [
        Vec::with_capacity(primitives.len()),
        Vec::with_capacity(primitives.len()),
        Vec::with_capacity(primitives.len()),
    ];
    let mut opacity = Vec::with_capacity(primitives.len());
    for primitive in primitives {
        x.push(primitive.mean.x);
        y.push(primitive.mean.y);
        z.push(primitive.mean.z);
        for (column, channel) in rgb.iter_mut().zip(primitive.color) {
            column.push((channel * 255.0).round() as u8);
        }
        opacity.push(primitive.opacity);
    }
    let generated_bytes =
        bytes_f32(x.len() * 4)?.saturating_add(rgb.iter().map(Vec::len).sum::<usize>());
    Ok(AdaptedVisual {
        id: adapter_id(namespace, "gaussians")?,
        label: "Gaussians".into(),
        geometry: AdaptedGeometry::Points {
            x,
            y,
            z,
            rgb: Some(rgb),
            scalar: Some(("opacity".into(), opacity)),
        },
        style: VisualStyle::Points(PointStyle::try_new(
            4.0,
            PointColor::Scalar { min: 0.0, max: 1.0, map: spatialrust_viz::ColorMap::Viridis },
        )?),
        receipt: AdapterReceipt {
            source_identity: source_identity(primitives),
            source_count: primitives.len(),
            output_count: primitives.len(),
            generated_bytes,
        },
    })
}

/// Converts a stamped trajectory to line segments.
#[cfg(feature = "mapping")]
pub fn trajectory_visual(
    namespace: &str,
    trajectory: &spatialrust_mapping::Trajectory,
) -> ViewerResult<AdaptedVisual> {
    let samples = trajectory.samples();
    let mut lines = Vec::with_capacity(samples.len().saturating_sub(1) * 6);
    for window in samples.windows(2) {
        push_segment(
            &mut lines,
            window[0].pose.isometry.translation(),
            window[1].pose.isometry.translation(),
        );
    }
    lines_visual(
        namespace,
        "trajectory",
        "Trajectory",
        source_identity(samples),
        samples.len(),
        lines,
        LinearRgba { red: 0.2, green: 0.8, blue: 1.0, alpha: 1.0 },
    )
}

/// Converts pose-graph edges to deterministic line segments.
#[cfg(feature = "mapping")]
pub fn pose_graph_visual(
    namespace: &str,
    graph: &spatialrust_mapping::PoseGraph,
) -> ViewerResult<AdaptedVisual> {
    let mut lines = Vec::with_capacity(graph.edges().len() * 6);
    for edge in graph.edges() {
        let from = graph
            .nodes()
            .get(&edge.from.0)
            .ok_or_else(|| ViewerError::InvalidState("pose graph source node missing".into()))?;
        let to = graph
            .nodes()
            .get(&edge.to.0)
            .ok_or_else(|| ViewerError::InvalidState("pose graph target node missing".into()))?;
        push_segment(&mut lines, from.pose.isometry.translation(), to.pose.isometry.translation());
    }
    lines_visual(
        namespace,
        "pose-graph",
        "Pose graph",
        source_identity(graph.edges()),
        graph.edges().len(),
        lines,
        LinearRgba { red: 1.0, green: 0.4, blue: 0.1, alpha: 1.0 },
    )
}

/// Creates camera-frustum wire geometry at the camera origin.
#[cfg(feature = "camera")]
pub fn camera_frustum_visual(
    namespace: &str,
    camera: &spatialrust_camera::PinholeCamera,
    depth: f32,
) -> ViewerResult<AdaptedVisual> {
    if !depth.is_finite() || depth <= 0.0 {
        return Err(ViewerError::InvalidState("frustum depth must be finite and positive".into()));
    }
    let intrinsics = camera.intrinsics;
    let corners = [
        (0.0, 0.0),
        (intrinsics.width as f64, 0.0),
        (intrinsics.width as f64, intrinsics.height as f64),
        (0.0, intrinsics.height as f64),
    ];
    let mut points = Vec::with_capacity(4);
    for (x, y) in corners {
        let point = camera
            .unproject(spatialrust_math::Vec2 { x, y }, depth as f64)
            .map_err(|error| ViewerError::InvalidState(error.to_string()))?;
        points.push(Vec3::new(point.x as f32, point.y as f32, point.z as f32));
    }
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let mut lines = Vec::with_capacity(8 * 6);
    for &point in &points {
        push_segment(&mut lines, origin, point);
    }
    for index in 0..4 {
        push_segment(&mut lines, points[index], points[(index + 1) % 4]);
    }
    lines_visual(
        namespace,
        "camera-frustum",
        "Camera frustum",
        camera as *const _ as usize,
        1,
        lines,
        LinearRgba { red: 1.0, green: 1.0, blue: 0.0, alpha: 1.0 },
    )
}

/// Converts semantic entity centroids and best-label confidence to points.
#[cfg(feature = "semantic")]
pub fn semantic_visual(
    namespace: &str,
    entities: &[spatialrust_semantic::SemanticEntity],
) -> ViewerResult<AdaptedVisual> {
    let visible: Vec<_> =
        entities.iter().filter_map(|entity| entity.centroid.map(|p| (entity, p))).collect();
    let mut x = Vec::with_capacity(visible.len());
    let mut y = Vec::with_capacity(visible.len());
    let mut z = Vec::with_capacity(visible.len());
    let mut confidence = Vec::with_capacity(visible.len());
    for (entity, point) in &visible {
        x.push(point.x);
        y.push(point.y);
        z.push(point.z);
        confidence.push(entity.labels.iter().map(|label| label.confidence).fold(0.0_f32, f32::max));
    }
    point_scalar_visual(
        namespace,
        "semantic",
        "Semantic entities",
        source_identity(entities),
        entities.len(),
        PointScalarColumns { x, y, z, name: "confidence", values: confidence },
    )
}

#[cfg(any(feature = "scene", feature = "semantic"))]
struct PointScalarColumns<'a> {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    name: &'a str,
    values: Vec<f32>,
}

#[cfg(any(feature = "scene", feature = "semantic"))]
fn point_scalar_visual(
    namespace: &str,
    slug: &str,
    label: &str,
    identity: usize,
    source_count: usize,
    columns: PointScalarColumns<'_>,
) -> ViewerResult<AdaptedVisual> {
    let PointScalarColumns { x, y, z, name: scalar_name, values: scalar } = columns;
    let output_count = x.len();
    let generated_bytes = bytes_f32(
        x.len()
            .checked_add(y.len())
            .and_then(|value| value.checked_add(z.len()))
            .and_then(|value| value.checked_add(scalar.len()))
            .ok_or_else(|| ViewerError::InvalidState("adapter byte count overflow".into()))?,
    )?;
    let max = scalar.iter().copied().fold(0.0_f32, f32::max).max(1.0);
    Ok(AdaptedVisual {
        id: adapter_id(namespace, slug)?,
        label: label.into(),
        geometry: AdaptedGeometry::Points {
            x,
            y,
            z,
            rgb: None,
            scalar: Some((scalar_name.into(), scalar)),
        },
        style: VisualStyle::Points(PointStyle::try_new(
            3.0,
            PointColor::Scalar { min: 0.0, max, map: spatialrust_viz::ColorMap::Viridis },
        )?),
        receipt: AdapterReceipt {
            source_identity: identity,
            source_count,
            output_count,
            generated_bytes,
        },
    })
}

#[cfg(any(feature = "mapping", feature = "camera"))]
fn lines_visual(
    namespace: &str,
    slug: &str,
    label: &str,
    identity: usize,
    source_count: usize,
    lines: Vec<f32>,
    color: LinearRgba,
) -> ViewerResult<AdaptedVisual> {
    let output_count = lines.len() / 6;
    let generated_bytes = bytes_f32(lines.len())?;
    Ok(AdaptedVisual {
        id: adapter_id(namespace, slug)?,
        label: label.into(),
        geometry: AdaptedGeometry::Lines(lines),
        style: VisualStyle::Uniform(color),
        receipt: AdapterReceipt {
            source_identity: identity,
            source_count,
            output_count,
            generated_bytes,
        },
    })
}

#[cfg(any(feature = "scene", feature = "mapping", feature = "camera", feature = "semantic"))]
fn adapter_id(namespace: &str, slug: &str) -> ViewerResult<LayerId> {
    if namespace.trim().is_empty() {
        return Err(ViewerError::InvalidState("adapter namespace must not be empty".into()));
    }
    Ok(LayerId::try_new(format!("scene/{namespace}/{slug}"))?)
}

#[cfg(any(feature = "scene", feature = "mapping", feature = "semantic"))]
fn source_identity<T>(slice: &[T]) -> usize {
    if slice.is_empty() {
        0
    } else {
        slice.as_ptr() as usize
    }
}

#[cfg(any(feature = "scene", feature = "mapping", feature = "camera", feature = "semantic"))]
fn bytes_f32(count: usize) -> ViewerResult<usize> {
    count
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| ViewerError::InvalidState("adapter byte count overflow".into()))
}

#[cfg(any(feature = "mapping", feature = "camera"))]
fn push_segment(lines: &mut Vec<f32>, from: Vec3<f32>, to: Vec3<f32>) {
    lines.extend_from_slice(&[from.x, from.y, from.z, to.x, to.y, to.z]);
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "scene",
        feature = "mapping",
        feature = "camera",
        feature = "semantic"
    ))]
    use spatialrust_math::Vec3;
    #[cfg(any(feature = "scene", feature = "semantic"))]
    use spatialrust_viz::VisualPrimitive;
    #[cfg(feature = "scene")]
    use spatialrust_viz::{LayerId, LinearRgba};

    #[cfg(feature = "scene")]
    #[test]
    fn mesh_and_surfel_adapters_preserve_source_identity_and_counts() {
        let mesh = spatialrust_scene::TriangleMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 2],
        };
        let (layer, receipt) =
            super::mesh_visual(LayerId::try_new("mesh").unwrap(), "Mesh", &mesh, LinearRgba::WHITE)
                .unwrap();
        let VisualPrimitive::Triangles(view) = layer.primitive else {
            panic!("mesh adapter must produce triangles");
        };
        assert_eq!(receipt.source_identity, mesh.positions.as_ptr() as usize);
        assert_eq!(receipt.source_count, 1);
        assert_eq!(receipt.output_count, 1);
        assert_eq!(receipt.generated_bytes, 0);
        assert!(core::ptr::eq(view.positions_xyz.as_ptr(), mesh.positions.as_ptr()));

        let mut surfels = spatialrust_scene::SurfelCloud::new();
        surfels
            .push(spatialrust_scene::Surfel {
                position: Vec3::new(1.0, 2.0, 3.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                radius: 0.25,
            })
            .unwrap();
        let adapted = super::surfel_visual("map", &surfels).unwrap();
        assert_eq!(adapted.receipt.source_count, 1);
        assert_eq!(adapted.receipt.output_count, 1);
        let VisualPrimitive::Points(points) = adapted.as_layer().unwrap().primitive else {
            panic!("surfel adapter must produce points");
        };
        assert_eq!(points.scalar.unwrap().values, &[0.25]);
    }

    #[cfg(feature = "scene-gaussian")]
    #[test]
    fn gaussian_adapter_preserves_rgb_opacity_and_count() {
        let mut scene = spatialrust_scene::GaussianScene::new();
        scene
            .push(spatialrust_scene::GaussianPrimitive {
                mean: Vec3::new(1.0, 2.0, 3.0),
                scale: Vec3::new(1.0, 1.0, 1.0),
                rotation: spatialrust_math::Quat::<f32>::identity(),
                opacity: 0.5,
                color: [1.0, 0.5, 0.0],
            })
            .unwrap();
        let adapted = super::gaussian_visual("reconstruction", &scene).unwrap();
        assert_eq!(adapted.receipt.source_count, scene.len());
        let VisualPrimitive::Points(points) = adapted.as_layer().unwrap().primitive else {
            panic!("Gaussian adapter must produce points");
        };
        assert_eq!(points.rgb.unwrap().red, &[255]);
        assert_eq!(points.rgb.unwrap().green, &[128]);
        assert_eq!(points.scalar.unwrap().values, &[0.5]);
    }

    #[cfg(feature = "mapping")]
    fn stamped(x: f32, nanos: u64) -> spatialrust_mapping::StampedPose {
        spatialrust_mapping::StampedPose::new(
            spatialrust_sync::StampedTime::exact(
                "host",
                spatialrust_sync::ClockDomain::HostSteady,
                spatialrust_core::Timestamp::from_nanos(nanos),
            ),
            spatialrust_math::Pose3::new(spatialrust_math::Isometry3::new(
                spatialrust_math::Quat::<f32>::identity(),
                Vec3::new(x, 0.0, 0.0),
            )),
        )
    }

    #[cfg(feature = "mapping")]
    #[test]
    fn trajectory_and_pose_graph_have_exact_segment_parity() {
        let mut trajectory = spatialrust_mapping::Trajectory::new();
        trajectory.push(stamped(0.0, 0)).unwrap();
        trajectory.push(stamped(1.0, 1)).unwrap();
        trajectory.push(stamped(2.0, 2)).unwrap();
        let adapted = super::trajectory_visual("slam", &trajectory).unwrap();
        assert_eq!(adapted.receipt.source_count, 3);
        assert_eq!(adapted.receipt.output_count, 2);

        let mut graph = spatialrust_mapping::PoseGraph::new();
        graph.upsert_node("a", stamped(0.0, 0));
        graph.upsert_node("b", stamped(1.0, 1));
        graph
            .add_edge(spatialrust_mapping::PoseGraphEdge {
                from: spatialrust_mapping::PoseNodeId::new("a"),
                to: spatialrust_mapping::PoseNodeId::new("b"),
                to_t_from: spatialrust_math::Isometry3::identity(),
                loop_closure: false,
            })
            .unwrap();
        let graph_visual = super::pose_graph_visual("slam", &graph).unwrap();
        assert_eq!(graph_visual.receipt.source_count, 1);
        assert_eq!(graph_visual.receipt.output_count, 1);
    }

    #[cfg(feature = "camera")]
    #[test]
    fn frustum_has_four_rays_and_four_image_edges() {
        let camera = spatialrust_camera::PinholeCamera::new(
            spatialrust_camera::CameraIntrinsics::try_new(100.0, 100.0, 50.0, 40.0, 100, 80)
                .unwrap(),
        );
        let adapted = super::camera_frustum_visual("rgb", &camera, 2.0).unwrap();
        assert_eq!(adapted.receipt.source_count, 1);
        assert_eq!(adapted.receipt.output_count, 8);
        assert!(super::camera_frustum_visual("rgb", &camera, 0.0).is_err());
    }

    #[cfg(feature = "semantic")]
    #[test]
    fn semantic_adapter_filters_missing_centroids_and_keeps_confidence() {
        let entities = [
            spatialrust_semantic::SemanticEntity {
                id: spatialrust_semantic::EntityId::new("chair"),
                centroid: Some(Vec3::new(1.0, 2.0, 3.0)),
                labels: vec![spatialrust_semantic::OpenVocabLabel {
                    text: "chair".into(),
                    confidence: 0.8,
                }],
                embedding: None,
            },
            spatialrust_semantic::SemanticEntity {
                id: spatialrust_semantic::EntityId::new("unknown"),
                centroid: None,
                labels: Vec::new(),
                embedding: None,
            },
        ];
        let adapted = super::semantic_visual("room", &entities).unwrap();
        assert_eq!(adapted.receipt.source_count, 2);
        assert_eq!(adapted.receipt.output_count, 1);
        let VisualPrimitive::Points(points) = adapted.as_layer().unwrap().primitive else {
            panic!("semantic adapter must produce points");
        };
        assert_eq!(points.scalar.unwrap().values, &[0.8]);
    }
}
