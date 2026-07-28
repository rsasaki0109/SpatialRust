use std::sync::{Arc, Mutex};

use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{InputAction, ViewerController, ViewerError, ViewerResult, ViewerState, ViewportSize};

/// Native window creation options.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeViewerOptions {
    /// Window title.
    pub title: String,
    /// Initial logical width.
    pub width: u32,
    /// Initial logical height.
    pub height: u32,
}

impl Default for NativeViewerOptions {
    fn default() -> Self {
        Self { title: "SpatialRust Viewer".into(), width: 1280, height: 720 }
    }
}

/// Opt-in native window shell backed by winit.
///
/// It owns no geometry and performs no implicit upload. Applications explicitly
/// upload geometry through `spatialrust-render-wgpu`, then use this shell for
/// portable input and layer/inspector state.
pub struct NativeViewer {
    state: Arc<Mutex<ViewerState>>,
    controller: ViewerController,
    options: NativeViewerOptions,
}

impl NativeViewer {
    /// Creates a native viewer shell.
    pub fn try_new(state: ViewerState, options: NativeViewerOptions) -> ViewerResult<Self> {
        if options.title.trim().is_empty() || options.width == 0 || options.height == 0 {
            return Err(ViewerError::Native(
                "title and non-zero window dimensions are required".into(),
            ));
        }
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
            controller: ViewerController::default(),
            options,
        })
    }

    /// Shared state handle for application render/UI integration.
    #[must_use]
    pub fn state(&self) -> Arc<Mutex<ViewerState>> {
        Arc::clone(&self.state)
    }

    /// Opens the native window and runs until the user closes it.
    pub fn run(self) -> ViewerResult<()> {
        let event_loop =
            EventLoop::new().map_err(|error| ViewerError::Native(error.to_string()))?;
        let mut application = NativeApplication {
            state: self.state,
            controller: self.controller,
            options: self.options,
            window: None,
            cursor: None,
            drag: None,
        };
        event_loop.run_app(&mut application).map_err(|error| ViewerError::Native(error.to_string()))
    }
}

struct NativeApplication {
    state: Arc<Mutex<ViewerState>>,
    controller: ViewerController,
    options: NativeViewerOptions,
    window: Option<Window>,
    cursor: Option<(f64, f64)>,
    drag: Option<MouseButton>,
}

impl NativeApplication {
    fn apply(&self, action: InputAction) {
        if let Ok(mut state) = self.state.lock() {
            let _ = self.controller.apply(&mut state, action);
        }
    }
}

impl ApplicationHandler for NativeApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = WindowAttributes::default()
            .with_title(self.options.title.clone())
            .with_inner_size(LogicalSize::new(self.options.width, self.options.height));
        match event_loop.create_window(attributes) {
            Ok(window) => self.window = Some(window),
            Err(error) => {
                eprintln!("SpatialRust native viewer: {error}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.window.as_ref().map_or(true, |window| window.id() != window_id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Ok(viewport) = ViewportSize::try_new(size.width, size.height) {
                    self.apply(InputAction::Resize(viewport));
                }
            }
            WindowEvent::DroppedFile(path) => {
                self.apply(InputAction::FileDropped(path.to_string_lossy().into_owned()));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.drag = (state == ElementState::Pressed).then_some(button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let (Some((last_x, last_y)), Some(button)) = (self.cursor, self.drag) {
                    let delta_x = (position.x - last_x) as f32;
                    let delta_y = (position.y - last_y) as f32;
                    let action = if button == MouseButton::Left {
                        InputAction::Orbit { delta_x, delta_y }
                    } else {
                        InputAction::Pan { delta_x, delta_y }
                    };
                    self.apply(action);
                }
                self.cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 40.0,
                };
                self.apply(InputAction::Zoom(amount));
            }
            WindowEvent::RedrawRequested => {
                // The application owns explicit wgpu upload/render integration.
                // Continuous redraw keeps camera/UI observers responsive.
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
