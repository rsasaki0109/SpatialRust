use spatialrust_math::Vec3;
use spatialrust_viz::{Camera, LayerId, Projection};

use crate::{ViewerError, ViewerResult, ViewerState, ViewportSize};

/// Backend-neutral input action consumed by [`ViewerController`].
#[derive(Clone, Debug, PartialEq)]
pub enum InputAction {
    /// Resize the logical viewport.
    Resize(ViewportSize),
    /// Orbit in logical pixels.
    Orbit {
        /// Horizontal drag delta.
        delta_x: f32,
        /// Vertical drag delta.
        delta_y: f32,
    },
    /// Pan in logical pixels.
    Pan {
        /// Horizontal drag delta.
        delta_x: f32,
        /// Vertical drag delta.
        delta_y: f32,
    },
    /// Zoom by a signed wheel/gesture delta.
    Zoom(f32),
    /// Replace the camera with a fit around world-space bounds.
    FocusBounds {
        /// Inclusive minimum bounds.
        min: Vec3<f32>,
        /// Inclusive maximum bounds.
        max: Vec3<f32>,
    },
    /// Toggle layer visibility.
    ToggleLayer(LayerId),
    /// Select a layer, or clear selection.
    SelectLayer(Option<LayerId>),
    /// Queue a dropped data file.
    FileDropped(String),
}

/// Deterministic orbit/pan/zoom and viewer-state input reducer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewerController {
    /// Orbit radians per logical pixel.
    pub orbit_sensitivity: f32,
    /// Pan fraction of camera distance per logical pixel.
    pub pan_sensitivity: f32,
    /// Exponential zoom sensitivity.
    pub zoom_sensitivity: f32,
}

impl Default for ViewerController {
    fn default() -> Self {
        Self { orbit_sensitivity: 0.005, pan_sensitivity: 0.002, zoom_sensitivity: 0.12 }
    }
}

impl ViewerController {
    /// Validates controller tuning.
    pub fn validate(self) -> ViewerResult<()> {
        if !self.orbit_sensitivity.is_finite()
            || !self.pan_sensitivity.is_finite()
            || !self.zoom_sensitivity.is_finite()
            || self.orbit_sensitivity <= 0.0
            || self.pan_sensitivity <= 0.0
            || self.zoom_sensitivity <= 0.0
        {
            return Err(ViewerError::InvalidState(
                "controller sensitivities must be finite and positive".into(),
            ));
        }
        Ok(())
    }

    /// Applies one input action and validates the resulting camera.
    pub fn apply(self, state: &mut ViewerState, action: InputAction) -> ViewerResult<()> {
        self.validate()?;
        match action {
            InputAction::Resize(viewport) => {
                state.viewport = viewport;
            }
            InputAction::Orbit { delta_x, delta_y } => {
                finite_pair(delta_x, delta_y, "orbit")?;
                orbit(
                    &mut state.camera,
                    delta_x * self.orbit_sensitivity,
                    delta_y * self.orbit_sensitivity,
                );
            }
            InputAction::Pan { delta_x, delta_y } => {
                finite_pair(delta_x, delta_y, "pan")?;
                pan(
                    &mut state.camera,
                    delta_x * self.pan_sensitivity,
                    delta_y * self.pan_sensitivity,
                );
            }
            InputAction::Zoom(delta) => {
                if !delta.is_finite() {
                    return Err(ViewerError::InvalidState("zoom delta must be finite".into()));
                }
                zoom(&mut state.camera, delta * self.zoom_sensitivity);
            }
            InputAction::FocusBounds { min, max } => {
                let vertical_fov_radians = match state.camera.projection {
                    Projection::Perspective { vertical_fov_radians, .. } => vertical_fov_radians,
                    Projection::Orthographic { .. } => 1.0,
                };
                state.camera = Camera::fit_perspective_bounds(
                    min,
                    max,
                    subtract(state.camera.target, state.camera.eye).normalize(),
                    state.camera.up,
                    vertical_fov_radians,
                    state.viewport.aspect(),
                    1.1,
                )?;
            }
            InputAction::ToggleLayer(id) => {
                let visible = state
                    .layers
                    .iter()
                    .find(|layer| layer.id == id)
                    .ok_or_else(|| ViewerError::UnknownLayer(id.as_str().into()))?
                    .visible;
                state.set_layer_visible(&id, !visible)?;
            }
            InputAction::SelectLayer(id) => state.select_layer(id.as_ref())?,
            InputAction::FileDropped(path) => state.queue_dropped_file(path)?,
        }
        Camera::try_new(
            state.camera.eye,
            state.camera.target,
            state.camera.up,
            state.camera.projection,
        )?;
        Ok(())
    }
}

fn finite_pair(x: f32, y: f32, name: &str) -> ViewerResult<()> {
    if !x.is_finite() || !y.is_finite() {
        return Err(ViewerError::InvalidState(format!("{name} delta must be finite")));
    }
    Ok(())
}

fn orbit(camera: &mut Camera, yaw: f32, pitch: f32) {
    let offset = subtract(camera.eye, camera.target);
    let radius = offset.length().max(f32::EPSILON);
    let direction = scale(offset, 1.0 / radius);
    let mut azimuth = direction.x.atan2(direction.z) + yaw;
    if !azimuth.is_finite() {
        azimuth = 0.0;
    }
    let elevation = direction.y.asin().clamp(-1.5, 1.5);
    let elevation = (elevation + pitch).clamp(-1.5, 1.5);
    let horizontal = elevation.cos();
    camera.eye = add(
        camera.target,
        scale(
            Vec3::new(horizontal * azimuth.sin(), elevation.sin(), horizontal * azimuth.cos()),
            radius,
        ),
    );
}

fn pan(camera: &mut Camera, delta_x: f32, delta_y: f32) {
    let view = subtract(camera.target, camera.eye);
    let distance = view.length().max(f32::EPSILON);
    let forward = scale(view, 1.0 / distance);
    let right = forward.cross(camera.up).normalize();
    let up = right.cross(forward).normalize();
    let translation = add(scale(right, -delta_x * distance), scale(up, delta_y * distance));
    camera.eye = add(camera.eye, translation);
    camera.target = add(camera.target, translation);
}

fn zoom(camera: &mut Camera, delta: f32) {
    let offset = subtract(camera.eye, camera.target);
    let factor = (-delta).exp().clamp(0.05, 20.0);
    camera.eye = add(camera.target, scale(offset, factor));
}

fn add(lhs: Vec3<f32>, rhs: Vec3<f32>) -> Vec3<f32> {
    Vec3::new(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z)
}

fn subtract(lhs: Vec3<f32>, rhs: Vec3<f32>) -> Vec3<f32> {
    Vec3::new(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z)
}

fn scale(value: Vec3<f32>, scalar: f32) -> Vec3<f32> {
    Vec3::new(value.x * scalar, value.y * scalar, value.z * scalar)
}

#[cfg(test)]
mod tests {
    use spatialrust_math::Vec3;
    use spatialrust_viz::{Camera, Projection};

    use crate::{InputAction, ViewerController, ViewerState, ViewportSize};

    fn state() -> ViewerState {
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
        .unwrap()
    }

    #[test]
    fn scripted_orbit_pan_zoom_resize_and_focus_are_valid() {
        let controller = ViewerController::default();
        let mut state = state();
        let original = state.camera;
        for action in [
            InputAction::Orbit { delta_x: 30.0, delta_y: -12.0 },
            InputAction::Pan { delta_x: 4.0, delta_y: 8.0 },
            InputAction::Zoom(2.0),
            InputAction::Resize(ViewportSize::try_new(1920, 1080).unwrap()),
            InputAction::FocusBounds {
                min: Vec3::new(-1.0, -2.0, -3.0),
                max: Vec3::new(1.0, 2.0, 3.0),
            },
        ] {
            controller.apply(&mut state, action).unwrap();
        }
        Camera::try_new(
            state.camera.eye,
            state.camera.target,
            state.camera.up,
            state.camera.projection,
        )
        .unwrap();
        assert_ne!(state.camera, original);
        assert_eq!(state.viewport.width, 1920);
    }

    #[test]
    fn rejects_non_finite_input_without_mutating_camera() {
        let controller = ViewerController::default();
        let mut state = state();
        let camera = state.camera;
        assert!(controller
            .apply(&mut state, InputAction::Orbit { delta_x: f32::NAN, delta_y: 0.0 })
            .is_err());
        assert_eq!(state.camera, camera);
    }
}
