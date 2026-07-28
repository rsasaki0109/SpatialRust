use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use spatialrust_viewer::{InputAction, ViewerController, ViewerState, ViewportSize};
use spatialrust_viz::{Camera, LayerId};

use crate::{WebError, WebResult};

/// Current portable Web viewer envelope version.
pub const WEB_VIEWER_STATE_VERSION: u32 = 1;

/// Versioned viewer state shared by native Rust, WASM, Python, and notebooks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebViewerState {
    /// Web envelope version.
    pub version: u32,
    /// Monotonic state revision.
    pub revision: u64,
    /// Portable viewer state.
    pub viewer: ViewerState,
}

impl WebViewerState {
    /// Wraps and validates a viewer state.
    pub fn try_new(viewer: ViewerState) -> WebResult<Self> {
        let state = Self { version: WEB_VIEWER_STATE_VERSION, revision: 0, viewer };
        state.validate()?;
        Ok(state)
    }

    /// Strict JSON serialization.
    pub fn to_json(&self) -> WebResult<String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| WebError::InvalidState(error.to_string()))
    }

    /// Strict JSON deserialization with schema and invariant validation.
    pub fn from_json(json: &str) -> WebResult<Self> {
        let state: Self = serde_json::from_str(json)
            .map_err(|error| WebError::InvalidState(error.to_string()))?;
        state.validate()?;
        Ok(state)
    }

    /// Applies one portable browser input and increments the revision exactly once.
    pub fn apply(&mut self, input: BrowserInput) -> WebResult<()> {
        self.validate()?;
        let action = input.into_action()?;
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| WebError::InvalidState("Web viewer revision overflow".into()))?;
        let mut next_viewer = self.viewer.clone();
        ViewerController::default()
            .apply(&mut next_viewer, action)
            .map_err(|error| WebError::Viewer(error.to_string()))?;
        self.viewer = next_viewer;
        self.revision = next_revision;
        Ok(())
    }

    /// Validates versions, camera, viewport, unique layers, and selection.
    pub fn validate(&self) -> WebResult<()> {
        if self.version != WEB_VIEWER_STATE_VERSION {
            return Err(WebError::InvalidState(format!(
                "unsupported Web viewer state version {}",
                self.version
            )));
        }
        Camera::try_new(
            self.viewer.camera.eye,
            self.viewer.camera.target,
            self.viewer.camera.up,
            self.viewer.camera.projection,
        )
        .map_err(|error| WebError::InvalidState(error.to_string()))?;
        ViewportSize::try_new(self.viewer.viewport.width, self.viewer.viewport.height)
            .map_err(|error| WebError::InvalidState(error.to_string()))?;
        let mut ids = BTreeSet::new();
        for layer in &self.viewer.layers {
            if !ids.insert(layer.id.as_str()) {
                return Err(WebError::InvalidState(format!(
                    "duplicate layer `{}`",
                    layer.id.as_str()
                )));
            }
        }
        if let Some(selected) = &self.viewer.selected_layer {
            if !ids.contains(selected.as_str()) {
                return Err(WebError::InvalidState(
                    "selected layer is absent from portable layers".into(),
                ));
            }
        }
        Ok(())
    }
}

/// JSON-friendly browser interaction event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BrowserInput {
    /// Orbit by logical-pixel deltas.
    Orbit {
        /// Horizontal delta.
        delta_x: f32,
        /// Vertical delta.
        delta_y: f32,
    },
    /// Pan by logical-pixel deltas.
    Pan {
        /// Horizontal delta.
        delta_x: f32,
        /// Vertical delta.
        delta_y: f32,
    },
    /// Signed wheel/gesture zoom delta.
    Zoom {
        /// Zoom delta.
        delta: f32,
    },
    /// Resize the logical viewport.
    Resize {
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },
    /// Toggle one stable layer.
    ToggleLayer {
        /// Layer identity.
        layer_id: String,
    },
    /// Select or clear a layer.
    SelectLayer {
        /// Optional layer identity.
        layer_id: Option<String>,
    },
    /// Queue a dropped file name/path.
    DropFile {
        /// Browser-provided name/path.
        path: String,
    },
}

impl BrowserInput {
    fn into_action(self) -> WebResult<InputAction> {
        Ok(match self {
            Self::Orbit { delta_x, delta_y } => InputAction::Orbit { delta_x, delta_y },
            Self::Pan { delta_x, delta_y } => InputAction::Pan { delta_x, delta_y },
            Self::Zoom { delta } => InputAction::Zoom(delta),
            Self::Resize { width, height } => InputAction::Resize(
                ViewportSize::try_new(width, height)
                    .map_err(|error| WebError::Viewer(error.to_string()))?,
            ),
            Self::ToggleLayer { layer_id } => InputAction::ToggleLayer(
                LayerId::try_new(layer_id).map_err(|error| WebError::Viewer(error.to_string()))?,
            ),
            Self::SelectLayer { layer_id } => InputAction::SelectLayer(
                layer_id
                    .map(LayerId::try_new)
                    .transpose()
                    .map_err(|error| WebError::Viewer(error.to_string()))?,
            ),
            Self::DropFile { path } => InputAction::FileDropped(path),
        })
    }
}

#[cfg(test)]
mod tests {
    use spatialrust_math::Vec3;
    use spatialrust_viewer::{ViewerState, ViewportSize};
    use spatialrust_viz::{Camera, Projection};

    use super::{BrowserInput, WebViewerState};

    fn state() -> WebViewerState {
        WebViewerState::try_new(
            ViewerState::try_new(
                Camera::try_new(
                    Vec3::new(0.0, 0.0, 5.0),
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                    Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
                )
                .unwrap(),
                ViewportSize::try_new(800, 600).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn state_json_roundtrip_and_input_revision_are_exact() {
        let mut state = state();
        let json = state.to_json().unwrap();
        assert_eq!(WebViewerState::from_json(&json).unwrap(), state);
        state.apply(BrowserInput::Zoom { delta: 1.0 }).unwrap();
        state.apply(BrowserInput::Resize { width: 1920, height: 1080 }).unwrap();
        assert_eq!(state.revision, 2);
        assert_eq!(state.viewer.viewport.width, 1920);
    }

    #[test]
    fn unknown_fields_versions_and_invalid_input_fail_closed() {
        let mut value: serde_json::Value =
            serde_json::from_str(&state().to_json().unwrap()).unwrap();
        value["extra"] = serde_json::json!(true);
        assert!(WebViewerState::from_json(&value.to_string()).is_err());

        let mut invalid_state = state();
        invalid_state.version = 99;
        assert!(invalid_state.validate().is_err());
        assert!(invalid_state.apply(BrowserInput::Resize { width: 0, height: 1 }).is_err());
        assert_eq!(invalid_state.revision, 0);

        let mut state = state();
        state.revision = u64::MAX;
        let before = state.viewer.clone();
        assert!(state.apply(BrowserInput::Zoom { delta: 1.0 }).is_err());
        assert_eq!(state.viewer, before);
        assert_eq!(state.revision, u64::MAX);
    }
}
