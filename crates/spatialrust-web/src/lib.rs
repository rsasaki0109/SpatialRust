//! Portable Web viewer state, bounded remote ranges, and an optional WebGPU bridge.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
#[cfg(feature = "webgpu")]
mod gpu;
mod range;
mod state;
#[cfg(feature = "wasm")]
mod wasm;

pub use error::{WebError, WebResult};
#[cfg(feature = "webgpu")]
pub use gpu::{WebGpuFrame, WebGpuViewer};
pub use range::{
    ByteRange, RangeAdmissionReceipt, RangeBudget, RangeCache, RangePlan, RangePlanner,
};
pub use state::{BrowserInput, WebViewerState, WEB_VIEWER_STATE_VERSION};
#[cfg(feature = "wasm")]
pub use wasm::{bounded_fetch_range, BrowserRangeAbort, BrowserViewer};
