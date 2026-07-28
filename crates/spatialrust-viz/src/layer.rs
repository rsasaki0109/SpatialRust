use crate::{PointColor, VisualPrimitive, VisualStyle, VizError, VizResult};

/// Stable, non-empty layer identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerId(String);

impl LayerId {
    /// Creates a validated layer identifier.
    pub fn try_new(value: impl Into<String>) -> VizResult<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(VizError::InvalidLayer("layer identifier must not be empty".into()));
        }
        Ok(Self(value))
    }

    /// Returns the identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One named visual primitive and its presentation state.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualLayer<'a> {
    /// Stable layer identifier.
    pub id: LayerId,
    /// Human-readable label.
    pub label: String,
    /// Whether the layer should be rendered.
    pub visible: bool,
    /// Borrowed geometry.
    pub primitive: VisualPrimitive<'a>,
    /// Primitive style.
    pub style: VisualStyle,
}

impl<'a> VisualLayer<'a> {
    /// Creates a visible layer and validates style/geometry compatibility.
    pub fn try_new(
        id: LayerId,
        label: impl Into<String>,
        primitive: VisualPrimitive<'a>,
        style: VisualStyle,
    ) -> VizResult<Self> {
        validate_compatibility(&primitive, &style)?;
        Ok(Self { id, label: label.into(), visible: true, primitive, style })
    }
}

fn validate_compatibility(primitive: &VisualPrimitive<'_>, style: &VisualStyle) -> VizResult<()> {
    match (primitive, style) {
        (VisualPrimitive::Points(points), VisualStyle::Points(point_style)) => {
            match &point_style.color {
                PointColor::Rgb if points.rgb.is_none() => Err(VizError::InvalidStyle(
                    "RGB point style requires borrowed RGB columns".into(),
                )),
                PointColor::Scalar { .. } if points.scalar.is_none() => {
                    Err(VizError::InvalidStyle(
                        "scalar point style requires a borrowed scalar column".into(),
                    ))
                }
                _ => Ok(()),
            }
        }
        (VisualPrimitive::Points(_), VisualStyle::Uniform(_)) => Ok(()),
        (_, VisualStyle::Points(_)) => {
            Err(VizError::InvalidStyle("point style can only be applied to point geometry".into()))
        }
        (_, VisualStyle::Uniform(_)) => Ok(()),
    }
}

/// Ordered collection of uniquely identified visual layers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VisualScene<'a> {
    layers: Vec<VisualLayer<'a>>,
}

impl<'a> VisualScene<'a> {
    /// Creates an empty scene.
    #[must_use]
    pub const fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Adds a layer, rejecting duplicate identifiers.
    pub fn add_layer(&mut self, layer: VisualLayer<'a>) -> VizResult<()> {
        if self.layers.iter().any(|current| current.id == layer.id) {
            return Err(VizError::InvalidLayer(format!(
                "duplicate layer identifier `{}`",
                layer.id.as_str()
            )));
        }
        self.layers.push(layer);
        Ok(())
    }

    /// Returns layers in deterministic insertion order.
    #[must_use]
    pub fn layers(&self) -> &[VisualLayer<'a>] {
        &self.layers
    }

    /// Finds a layer by identifier.
    #[must_use]
    pub fn layer(&self, id: &LayerId) -> Option<&VisualLayer<'a>> {
        self.layers.iter().find(|layer| &layer.id == id)
    }

    /// Removes and returns a layer while preserving the order of remaining layers.
    pub fn remove_layer(&mut self, id: &LayerId) -> Option<VisualLayer<'a>> {
        let index = self.layers.iter().position(|layer| &layer.id == id)?;
        Some(self.layers.remove(index))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        LayerId, LineListView, LinearRgba, VisualLayer, VisualPrimitive, VisualScene, VisualStyle,
    };

    fn layer<'a>(id: &str, positions: &'a [f32]) -> VisualLayer<'a> {
        VisualLayer::try_new(
            LayerId::try_new(id).unwrap(),
            id,
            VisualPrimitive::Lines(LineListView::try_new(positions).unwrap()),
            VisualStyle::Uniform(LinearRgba::WHITE),
        )
        .unwrap()
    }

    #[test]
    fn scene_rejects_duplicate_ids_and_preserves_order() {
        let positions = [0.0; 6];
        let mut scene = VisualScene::new();
        scene.add_layer(layer("first", &positions)).unwrap();
        scene.add_layer(layer("second", &positions)).unwrap();
        assert!(scene.add_layer(layer("first", &positions)).is_err());
        assert_eq!(scene.layers()[0].id.as_str(), "first");

        let first = LayerId::try_new("first").unwrap();
        scene.remove_layer(&first).unwrap();
        assert_eq!(scene.layers()[0].id.as_str(), "second");
    }

    #[test]
    fn layer_rejects_style_without_required_attributes() {
        use crate::{ColorMap, PointCloudView, PointColor, PointStyle, PositionColumns3};

        let positions = PositionColumns3::try_new(&[0.0], &[0.0], &[0.0]).unwrap();
        let primitive = VisualPrimitive::Points(PointCloudView::positions_only(positions));
        let rgb_style = VisualStyle::Points(PointStyle::try_new(1.0, PointColor::Rgb).unwrap());
        assert!(VisualLayer::try_new(
            LayerId::try_new("points").unwrap(),
            "points",
            primitive,
            rgb_style,
        )
        .is_err());

        let scalar_style = VisualStyle::Points(
            PointStyle::try_new(
                1.0,
                PointColor::Scalar { min: 0.0, max: 1.0, map: ColorMap::Viridis },
            )
            .unwrap(),
        );
        assert!(VisualLayer::try_new(
            LayerId::try_new("scalar").unwrap(),
            "scalar",
            primitive,
            scalar_style,
        )
        .is_err());
    }
}
