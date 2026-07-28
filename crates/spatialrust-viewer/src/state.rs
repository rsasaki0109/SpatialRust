use spatialrust_viz::{Camera, LayerId, PointColor, VisualPrimitive, VisualScene, VisualStyle};

use crate::{ViewerError, ViewerResult};

/// Current serialized viewer-state schema version.
pub const VIEWER_STATE_VERSION: u32 = 1;

/// Positive viewport dimensions in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewportSize {
    /// Logical width.
    pub width: u32,
    /// Logical height.
    pub height: u32,
}

impl ViewportSize {
    /// Creates validated non-zero viewport dimensions.
    pub fn try_new(width: u32, height: u32) -> ViewerResult<Self> {
        if width == 0 || height == 0 {
            return Err(ViewerError::InvalidState("viewport dimensions must be non-zero".into()));
        }
        Ok(Self { width, height })
    }

    /// Width divided by height.
    #[must_use]
    pub fn aspect(self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// One attribute exposed by the point attribute inspector.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AttributeSummary {
    /// Stable attribute name.
    pub name: String,
    /// Scalar representation displayed to the user.
    pub data_type: String,
    /// Number of values.
    pub len: usize,
}

/// Owned presentation state for one borrowed scene layer.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerPresentation {
    /// Stable layer identity.
    pub id: LayerId,
    /// User-facing layer name.
    pub label: String,
    /// Current visibility.
    pub visible: bool,
    /// Editable visual style.
    pub style: VisualStyle,
    /// Number of logical primitives or points.
    pub element_count: usize,
    /// Attributes available for inspection.
    pub attributes: Vec<AttributeSummary>,
}

/// Current selection exposed by the inspector panel.
#[derive(Clone, Debug, PartialEq)]
pub struct InspectorSelection<'a> {
    /// Selected layer.
    pub layer: &'a LayerPresentation,
    /// Active point-color attribute, when any.
    pub active_attribute: Option<&'a str>,
}

/// Portable state shared by native, Web, Python, and notebook surfaces.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewerState {
    /// Schema version, currently [`VIEWER_STATE_VERSION`].
    pub version: u32,
    /// Active camera.
    pub camera: Camera,
    /// Logical viewport dimensions.
    pub viewport: ViewportSize,
    /// Ordered layer presentation state.
    pub layers: Vec<LayerPresentation>,
    /// Selected layer identity.
    pub selected_layer: Option<LayerId>,
    /// Validated data files queued by native drag/drop.
    pub pending_files: Vec<String>,
}

impl ViewerState {
    /// Creates an empty viewer state.
    pub fn try_new(camera: Camera, viewport: ViewportSize) -> ViewerResult<Self> {
        validate_camera(camera)?;
        Ok(Self {
            version: VIEWER_STATE_VERSION,
            camera,
            viewport,
            layers: Vec::new(),
            selected_layer: None,
            pending_files: Vec::new(),
        })
    }

    /// Synchronizes layer metadata from a borrowed visual scene.
    ///
    /// Existing visibility and style edits are retained for matching stable IDs.
    pub fn sync_scene(&mut self, scene: &VisualScene<'_>) {
        let mut next = Vec::with_capacity(scene.layers().len());
        for layer in scene.layers() {
            let existing = self.layers.iter().find(|entry| entry.id == layer.id);
            let (element_count, attributes) = describe_primitive(layer.primitive);
            next.push(LayerPresentation {
                id: layer.id.clone(),
                label: layer.label.clone(),
                visible: existing.map_or(layer.visible, |entry| entry.visible),
                style: existing.map_or_else(|| layer.style.clone(), |entry| entry.style.clone()),
                element_count,
                attributes,
            });
        }
        self.layers = next;
        if self
            .selected_layer
            .as_ref()
            .is_some_and(|id| !self.layers.iter().any(|layer| &layer.id == id))
        {
            self.selected_layer = None;
        }
    }

    /// Changes a layer's visibility.
    pub fn set_layer_visible(&mut self, id: &LayerId, visible: bool) -> ViewerResult<()> {
        self.layer_mut(id)?.visible = visible;
        Ok(())
    }

    /// Selects a layer for attribute inspection.
    pub fn select_layer(&mut self, id: Option<&LayerId>) -> ViewerResult<()> {
        if let Some(id) = id {
            if !self.layers.iter().any(|layer| &layer.id == id) {
                return Err(ViewerError::UnknownLayer(id.as_str().into()));
            }
            self.selected_layer = Some(id.clone());
        } else {
            self.selected_layer = None;
        }
        Ok(())
    }

    /// Replaces the visual style after validating attribute compatibility.
    pub fn set_layer_style(&mut self, id: &LayerId, style: VisualStyle) -> ViewerResult<()> {
        let layer = self.layer_mut(id)?;
        validate_style_attributes(layer, &style)?;
        layer.style = style;
        Ok(())
    }

    /// Returns the selected layer and active point attribute.
    #[must_use]
    pub fn inspector(&self) -> Option<InspectorSelection<'_>> {
        let id = self.selected_layer.as_ref()?;
        let layer = self.layers.iter().find(|layer| &layer.id == id)?;
        let active_attribute = match &layer.style {
            VisualStyle::Points(style) => match &style.color {
                PointColor::Rgb => Some("rgb"),
                PointColor::Scalar { .. } => layer
                    .attributes
                    .iter()
                    .find(|attribute| attribute.data_type == "f32")
                    .map(|attribute| attribute.name.as_str()),
                PointColor::Uniform(_) => None,
            },
            VisualStyle::Uniform(_) => None,
        };
        Some(InspectorSelection { layer, active_attribute })
    }

    /// Queues a supported point-cloud file from drag/drop.
    pub fn queue_dropped_file(&mut self, path: impl Into<String>) -> ViewerResult<()> {
        let path = path.into();
        let extension = path.rsplit('.').next().unwrap_or_default().to_ascii_lowercase();
        if !matches!(extension.as_str(), "pcd" | "ply" | "las" | "laz" | "copc" | "e57") {
            return Err(ViewerError::InvalidState(format!(
                "unsupported dropped file extension in `{path}`"
            )));
        }
        self.pending_files.push(path);
        Ok(())
    }

    fn layer_mut(&mut self, id: &LayerId) -> ViewerResult<&mut LayerPresentation> {
        self.layers
            .iter_mut()
            .find(|layer| &layer.id == id)
            .ok_or_else(|| ViewerError::UnknownLayer(id.as_str().into()))
    }
}

fn validate_camera(camera: Camera) -> ViewerResult<()> {
    Camera::try_new(camera.eye, camera.target, camera.up, camera.projection)?;
    Ok(())
}

fn describe_primitive(primitive: VisualPrimitive<'_>) -> (usize, Vec<AttributeSummary>) {
    match primitive {
        VisualPrimitive::Points(points) => {
            let mut attributes = vec![AttributeSummary {
                name: "position".into(),
                data_type: "vec3<f32>".into(),
                len: points.positions.len(),
            }];
            if points.rgb.is_some() {
                attributes.push(AttributeSummary {
                    name: "rgb".into(),
                    data_type: "rgb8".into(),
                    len: points.positions.len(),
                });
            }
            if let Some(scalar) = points.scalar {
                attributes.push(AttributeSummary {
                    name: scalar.name.into(),
                    data_type: "f32".into(),
                    len: scalar.values.len(),
                });
            }
            (points.positions.len(), attributes)
        }
        VisualPrimitive::Lines(lines) => (lines.segment_count(), Vec::new()),
        VisualPrimitive::Triangles(mesh) => (mesh.triangle_count(), Vec::new()),
    }
}

fn validate_style_attributes(layer: &LayerPresentation, style: &VisualStyle) -> ViewerResult<()> {
    if let VisualStyle::Points(point_style) = style {
        match point_style.color {
            PointColor::Rgb
                if !layer.attributes.iter().any(|attribute| attribute.name == "rgb") =>
            {
                return Err(ViewerError::InvalidState(
                    "RGB style requires an RGB layer attribute".into(),
                ));
            }
            PointColor::Scalar { .. }
                if !layer.attributes.iter().any(|attribute| attribute.data_type == "f32") =>
            {
                return Err(ViewerError::InvalidState(
                    "scalar style requires an f32 layer attribute".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use spatialrust_math::Vec3;
    use spatialrust_viz::{
        Camera, LayerId, LinearRgba, PointCloudView, PointColor, PointStyle, PositionColumns3,
        Projection, ScalarColumn, VisualLayer, VisualPrimitive, VisualScene, VisualStyle,
    };

    use super::{ViewerState, ViewportSize};

    fn camera() -> Camera {
        Camera::try_new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
        )
        .unwrap()
    }

    #[test]
    fn scene_sync_preserves_edits_and_inspects_attributes() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        let z = [0.0, 1.0];
        let scalar_values = [4.0, 8.0];
        let positions = PositionColumns3::try_new(&x, &y, &z).unwrap();
        let points = PointCloudView::positions_only(positions)
            .with_scalar(ScalarColumn::try_new("intensity", &scalar_values, 2).unwrap())
            .unwrap();
        let id = LayerId::try_new("cloud").unwrap();
        let layer = VisualLayer::try_new(
            id.clone(),
            "Cloud",
            VisualPrimitive::Points(points),
            VisualStyle::Points(
                PointStyle::try_new(
                    2.0,
                    PointColor::Scalar {
                        min: 0.0,
                        max: 10.0,
                        map: spatialrust_viz::ColorMap::Viridis,
                    },
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let mut scene = VisualScene::new();
        scene.add_layer(layer).unwrap();

        let mut state =
            ViewerState::try_new(camera(), ViewportSize::try_new(800, 600).unwrap()).unwrap();
        state.sync_scene(&scene);
        state.set_layer_visible(&id, false).unwrap();
        state.select_layer(Some(&id)).unwrap();
        assert_eq!(state.inspector().unwrap().active_attribute, Some("intensity"));

        state.sync_scene(&scene);
        assert!(!state.layers[0].visible);
        assert_eq!(state.layers[0].element_count, 2);
        assert_eq!(state.layers[0].attributes.len(), 2);

        assert!(state
            .set_layer_style(
                &id,
                VisualStyle::Points(PointStyle::try_new(1.0, PointColor::Rgb).unwrap())
            )
            .is_err());
        state.set_layer_style(&id, VisualStyle::Uniform(LinearRgba::WHITE)).unwrap();
    }

    #[test]
    fn validates_viewport_selection_and_drop_extensions() {
        assert!(ViewportSize::try_new(0, 1).is_err());
        let mut state =
            ViewerState::try_new(camera(), ViewportSize::try_new(1, 1).unwrap()).unwrap();
        assert!(state.select_layer(Some(&LayerId::try_new("missing").unwrap())).is_err());
        assert!(state.queue_dropped_file("scan.txt").is_err());
        state.queue_dropped_file("SCAN.PCD").unwrap();
        assert_eq!(state.pending_files, ["SCAN.PCD"]);
    }
}
