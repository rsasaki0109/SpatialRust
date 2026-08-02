use std::sync::Arc;

use spatialrust_gpu::{WgpuPowerPreference, WgpuRuntime};
use spatialrust_render_wgpu::{GpuGeometry, HeadlessRender, RenderOptions, WgpuRenderer};
use spatialrust_viz::{LinearRgba, VisualStyle};

use crate::{WebError, WebResult, WebViewerState};

/// One WebGPU/device-resident frame tied to a portable state revision.
pub struct WebGpuFrame {
    /// Viewer state revision rendered.
    pub state_revision: u64,
    /// Device-resident color/depth result and explicit transfer receipt.
    pub render: HeadlessRender,
}

/// Web-compatible bridge from portable viewer state to the shared wgpu backend.
///
/// The bridge owns no scene geometry. Callers explicitly upload through
/// [`Self::renderer`] and pass the resulting device-resident handle to
/// [`Self::render`]. A frame remains device-resident unless the caller invokes
/// the renderer's explicit readback API.
pub struct WebGpuViewer {
    state: WebViewerState,
    renderer: WgpuRenderer,
}

impl WebGpuViewer {
    /// Asynchronously creates a WebGPU runtime and bridge.
    ///
    /// This is the normal `wasm32` entry point; it never blocks the browser
    /// main thread while requesting an adapter/device.
    pub async fn new_async(
        state: WebViewerState,
        preference: WgpuPowerPreference,
    ) -> WebResult<Self> {
        let runtime = WgpuRuntime::new_headless_async(preference)
            .await
            .map_err(|error| WebError::WebGpu(error.to_string()))?;
        Self::try_new(state, Arc::new(runtime))
    }

    /// Creates a bridge on a caller-selected wgpu runtime.
    pub fn try_new(state: WebViewerState, runtime: Arc<WgpuRuntime>) -> WebResult<Self> {
        state.validate()?;
        Ok(Self { state, renderer: WgpuRenderer::new(runtime) })
    }

    /// Portable state.
    #[must_use]
    pub const fn state(&self) -> &WebViewerState {
        &self.state
    }

    /// Replaces portable state after strict validation.
    pub fn set_state(&mut self, state: WebViewerState) -> WebResult<()> {
        state.validate()?;
        self.state = state;
        Ok(())
    }

    /// Renderer used for explicit geometry upload/recycle/readback.
    #[must_use]
    pub const fn renderer(&self) -> &WgpuRenderer {
        &self.renderer
    }

    /// Renders one already-uploaded layer with the current camera/viewport.
    pub fn render(
        &self,
        geometry: &GpuGeometry,
        style: VisualStyle,
        clear_color: LinearRgba,
    ) -> WebResult<WebGpuFrame> {
        let options = RenderOptions::try_new(
            self.state.viewer.viewport.width,
            self.state.viewer.viewport.height,
            self.state.viewer.camera,
            style,
            clear_color,
        )
        .map_err(|error| WebError::WebGpu(error.to_string()))?;
        let render = self
            .renderer
            .render_headless(geometry, &options)
            .map_err(|error| WebError::WebGpu(error.to_string()))?;
        Ok(WebGpuFrame { state_revision: self.state.revision, render })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use spatialrust_gpu::WgpuRuntime;
    use spatialrust_math::Vec3;
    use spatialrust_viewer::{ViewerState, ViewportSize};
    use spatialrust_viz::{
        Camera, LinearRgba, PointCloudView, PointColor, PointStyle, PositionColumns3, Projection,
        VisualPrimitive, VisualStyle,
    };

    use crate::WebViewerState;

    #[test]
    fn web_bridge_matches_shared_headless_renderer_pixels_and_revision() {
        let runtime = Arc::new(WgpuRuntime::new_headless().unwrap());
        let camera = Camera::try_new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
        )
        .unwrap();
        let mut state = WebViewerState::try_new(
            ViewerState::try_new(camera, ViewportSize::try_new(64, 64).unwrap()).unwrap(),
        )
        .unwrap();
        state.revision = 7;
        let viewer = super::WebGpuViewer::try_new(state, runtime).unwrap();
        let x = [0.0];
        let y = [0.0];
        let z = [0.0];
        let points = PointCloudView::positions_only(PositionColumns3::try_new(&x, &y, &z).unwrap());
        let (geometry, upload) = viewer.renderer().upload(VisualPrimitive::Points(points)).unwrap();
        assert_eq!(upload.total_bytes().unwrap(), 12);
        let style = VisualStyle::Points(
            PointStyle::try_new(
                8.0,
                PointColor::Uniform(LinearRgba::try_new(1.0, 0.0, 0.0, 1.0).unwrap()),
            )
            .unwrap(),
        );
        let frame = viewer.render(&geometry, style, LinearRgba::BLACK).unwrap();
        assert_eq!(frame.state_revision, 7);
        let image = viewer.renderer().readback_rgba(&frame.render.target).unwrap();
        let center = ((32 * 64 + 32) * 4) as usize;
        assert!(image.rgba[center] > 200);
        assert!(image.rgba[center + 1] < 20);
        assert_eq!(image.rgba.len(), 64 * 64 * 4);
    }
}
