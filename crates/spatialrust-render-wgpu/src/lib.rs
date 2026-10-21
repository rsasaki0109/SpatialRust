//! Explicit wgpu backend for SpatialRust visualization contracts.
//!
//! Geometry enters this crate only through caller-requested upload methods.
//! Uploads return a byte-exact transfer receipt, and device buffers remain
//! resident until explicitly recycled or dropped.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
#[cfg(feature = "wgpu")]
mod geometry;
#[cfg(feature = "wgpu")]
mod render;
#[cfg(feature = "wgpu")]
mod runtime;

pub use error::{RenderError, RenderResult};
#[cfg(feature = "wgpu")]
pub use geometry::{GpuGeometry, GpuGeometryKind};
#[cfg(feature = "wgpu")]
pub use render::{
    GpuRenderTarget, HeadlessRender, PickResult, ReadbackImage, RenderOptions, RenderReceipt,
    RENDER_TARGET_FORMAT,
};
#[cfg(feature = "wgpu")]
pub use runtime::WgpuRenderer;

// Native GPU tests share device initialization while keeping distinct renderers.
// Retain the previous construction path for wasm test compilation.
#[cfg(all(test, feature = "wgpu"))]
fn test_runtime() -> Result<std::sync::Arc<spatialrust_gpu::WgpuRuntime>, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        spatialrust_gpu::WgpuRuntime::shared().map_err(|error| error.to_string())
    }
    #[cfg(target_arch = "wasm32")]
    {
        spatialrust_gpu::WgpuRuntime::new_headless()
            .map(std::sync::Arc::new)
            .map_err(|error| error.to_string())
    }
}
