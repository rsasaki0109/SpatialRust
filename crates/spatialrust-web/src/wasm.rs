use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::JsFuture;
use web_sys::{AbortController, Request, RequestInit, RequestMode, Response};

use spatialrust_math::Vec3;
use spatialrust_viewer::{ViewerState, ViewportSize};
use spatialrust_viz::{Camera, Projection};

use crate::{BrowserInput, ByteRange, RangeBudget, RangeCache, RangePlanner, WebViewerState};

fn js_error(error: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Browser-visible cooperative cancellation backed by `AbortController`.
#[wasm_bindgen]
pub struct BrowserRangeAbort {
    controller: AbortController,
}

#[wasm_bindgen]
impl BrowserRangeAbort {
    /// Creates a live cancellation handle.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BrowserRangeAbort, JsValue> {
        Ok(Self { controller: AbortController::new()? })
    }

    /// Aborts an in-progress fetch. Repeated calls are harmless.
    pub fn abort(&self) {
        self.controller.abort();
    }

    /// Whether cancellation was requested.
    #[wasm_bindgen(getter)]
    pub fn aborted(&self) -> bool {
        self.controller.signal().aborted()
    }
}

/// Executes one strictly bounded browser HTTP Range fetch.
///
/// The server must return `206`, an exact `Content-Length`, and a matching
/// response length. No bytes are copied into WASM until those headers pass.
#[wasm_bindgen]
pub async fn bounded_fetch_range(
    url: String,
    start: u64,
    end_exclusive: u64,
    max_response_bytes: u64,
    cancellation: &BrowserRangeAbort,
) -> Result<Uint8Array, JsValue> {
    let range = ByteRange::try_new(start, end_exclusive).map_err(js_error)?;
    if max_response_bytes == 0 || range.len() > max_response_bytes {
        return Err(js_error("requested range exceeds max_response_bytes"));
    }
    if cancellation.aborted() {
        return Err(js_error("range fetch cancelled before request"));
    }
    let options = RequestInit::new();
    options.set_method("GET");
    options.set_mode(RequestMode::Cors);
    options.set_signal(Some(&cancellation.controller.signal()));
    let request = Request::new_with_str_and_init(&url, &options)?;
    request.headers().set("Range", &range.http_header())?;
    let window = web_sys::window().ok_or_else(|| js_error("browser Window is unavailable"))?;
    let response =
        JsFuture::from(window.fetch_with_request(&request)).await?.dyn_into::<Response>()?;
    if response.status() != 206 {
        return Err(js_error(format!(
            "range server returned status {}, expected 206",
            response.status()
        )));
    }
    let length_header = response
        .headers()
        .get("Content-Length")?
        .ok_or_else(|| js_error("range response omitted Content-Length"))?;
    let declared: u64 = length_header.parse().map_err(|_| js_error("invalid Content-Length"))?;
    if declared != range.len() || declared > max_response_bytes {
        return Err(js_error(format!(
            "range Content-Length {declared} does not match requested {}",
            range.len()
        )));
    }
    let buffer = JsFuture::from(response.array_buffer()?).await?.dyn_into::<ArrayBuffer>()?;
    let bytes = Uint8Array::new(&buffer);
    if u64::from(bytes.length()) != declared {
        return Err(js_error("range body length differs from Content-Length"));
    }
    Ok(bytes)
}

/// WASM-facing portable viewer and bounded remote-range cache.
#[wasm_bindgen]
pub struct BrowserViewer {
    state: WebViewerState,
    planner: RangePlanner,
    cache: RangeCache,
}

#[wasm_bindgen]
impl BrowserViewer {
    /// Creates a deterministic empty viewer for browser smoke tests and simple embeds.
    pub fn new_default(
        width: u32,
        height: u32,
        range_budget_json: &str,
    ) -> Result<BrowserViewer, JsValue> {
        let camera = Camera::try_new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
        )
        .map_err(js_error)?;
        let viewer =
            ViewerState::try_new(camera, ViewportSize::try_new(width, height).map_err(js_error)?)
                .map_err(js_error)?;
        let state = WebViewerState::try_new(viewer).map_err(js_error)?;
        let budget: RangeBudget = serde_json::from_str(range_budget_json).map_err(js_error)?;
        Ok(Self {
            state,
            planner: RangePlanner::try_new(budget).map_err(js_error)?,
            cache: RangeCache::try_new(budget).map_err(js_error)?,
        })
    }

    /// Creates a browser viewer from strict state and range-budget JSON.
    #[wasm_bindgen(constructor)]
    pub fn new(state_json: &str, range_budget_json: &str) -> Result<BrowserViewer, JsValue> {
        let state = WebViewerState::from_json(state_json).map_err(js_error)?;
        let budget: RangeBudget = serde_json::from_str(range_budget_json).map_err(js_error)?;
        Ok(Self {
            state,
            planner: RangePlanner::try_new(budget).map_err(js_error)?,
            cache: RangeCache::try_new(budget).map_err(js_error)?,
        })
    }

    /// Returns strict portable viewer-state JSON.
    pub fn state_json(&self) -> Result<String, JsValue> {
        self.state.to_json().map_err(js_error)
    }

    /// Applies one strict [`BrowserInput`] JSON object.
    pub fn apply_input_json(&mut self, input_json: &str) -> Result<(), JsValue> {
        let input: BrowserInput = serde_json::from_str(input_json).map_err(js_error)?;
        self.state.apply(input).map_err(js_error)
    }

    /// Plans sorted/deduplicated bounded cache misses from a JSON range array.
    pub fn plan_ranges_json(
        &mut self,
        ranges_json: &str,
        cancelled: bool,
    ) -> Result<String, JsValue> {
        let ranges: Vec<ByteRange> = serde_json::from_str(ranges_json).map_err(js_error)?;
        let plan = self.planner.plan(ranges, &self.cache, cancelled).map_err(js_error)?;
        serde_json::to_string(&plan).map_err(js_error)
    }

    /// Explicitly copies a fetched JS byte array into the bounded WASM cache.
    ///
    /// The returned JSON receipt reports the exact copied and evicted bytes.
    pub fn admit_range(
        &mut self,
        start: u64,
        end_exclusive: u64,
        bytes: &Uint8Array,
    ) -> Result<String, JsValue> {
        let range = ByteRange::try_new(start, end_exclusive).map_err(js_error)?;
        let mut owned = vec![0_u8; bytes.length() as usize];
        bytes.copy_to(&mut owned);
        let receipt = self.cache.admit(range, owned).map_err(js_error)?;
        serde_json::to_string(&receipt).map_err(js_error)
    }

    /// Explicitly copies an exact cached range back into a JS byte array.
    pub fn cached_range(
        &mut self,
        start: u64,
        end_exclusive: u64,
    ) -> Result<Option<Uint8Array>, JsValue> {
        let range = ByteRange::try_new(start, end_exclusive).map_err(js_error)?;
        Ok(self.cache.get(range).map(Uint8Array::from))
    }
}
