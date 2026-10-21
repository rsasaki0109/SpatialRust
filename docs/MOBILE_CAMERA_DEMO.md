# Mobile AI-assisted camera playground

The [GitHub Pages camera playground](https://rsasaki0109.github.io/SpatialRust/camera/)
lets a visitor explicitly start their camera or play a labeled procedural sample.
It uses SpatialRust FAST features and forward/backward Lucas-Kanade in real Rust
WASM. A pinned MediaPipe SelfieSegmenter runs locally and excludes predicted
person neighborhoods from background feature tracks. The arrow shows median
**image-space background flow in pixels**. It does not estimate metric distance,
3D position, camera orientation or a SLAM trajectory.

Open the page over HTTPS, press **カメラ開始**, grant camera permission, and move
slowly toward a textured scene. **サンプル再生** works without camera permission.
**停止**, hiding the document and leaving the page stop capture and terminate
processing. Camera selection and AI masking can be changed while stopped. The
browser never requests a microphone, records a clip or persists a frame.

## Boundary and ownership

Only the optional `spatialrust-web/camera-tracking` feature adds image and vision
odometry dependencies. Existing core and vision algorithms are unchanged. The
native `CameraTracker` accepts 64–320 pixels per axis and at most 80 tracks. Its
copy-named WASM method validates lengths before copying JavaScript input. Invalid
sizes, mask labels and nonfinite or non-increasing times fail before retained
state changes; a gap over 500ms reseeds. Missing/lost tracking returns no motion.

The demo resizes to a 256-pixel long edge, transfers one `ImageBitmap` at a time,
and performs tracking in a worker. CPU image readback, grayscale conversion,
AI mask readback and the Rust input copy are explicit in the implementation.
There are no implicit production device transfers. Person exclusion covers both
previous/current point neighborhoods, including pyramid patch support; it does
not guarantee that every intermediate flow iterate avoids a person.

The model metadata contains one `selfie` confidence channel. The worker reads
that floating-point channel before the SDK callback releases it, validates its
shape/range and uses **confidence > 0.5** for the bounded binary exclusion mask.
It does not infer semantics from SDK category numbers. GPU inference is attempted
through the SDK's browser graphics delegate, with an explicit CPU backend display
if initialization falls back. SwiftShader evidence is software execution and is
not a physical GPU benchmark. An inference failure stops processing; it cannot
silently present unmasked tracking as AI-enabled.

## Static distribution and privacy

`docs/site/camera/assets.json` pins MediaPipe Tasks Vision **1.1.1**, model revision
**1**, loader variants and the upstream license by exact bytes and SHA256. The
complete static AI distribution is **41,087,603 bytes**, with a 64MiB asset limit
and 96MiB camera-site limit. A browser fetches the selected loader/WASM variant,
model and application; AI files for that variant total about 14MB. It does not
fetch every distributed fallback variant.
The build reuses identical assets and streams bounded downloads through private
partial files. Failed logs/downloads are retained. Vendor binaries stay outside
tracked source and are copied into the Pages build.

All runtime assets come from this site. A restrictive CSP blocks external
connections, frames and form submission. A blob module worker inherits the
page's policy and may import same-origin modules. This matters because the
[upstream SDK notice](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md)
allows SDK metrics; the application enforces its own connection restriction.
The browser check observes an external worker fetch blocked by `connect-src`.
This is a tested policy boundary, not a claim that upstream SDKs contain no
telemetry. Ordinary hosting access logs and the explicit external documentation
links are separate from camera processing.

The published page includes the upstream Apache license and links to the
[official model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter).
The exact pinned model metadata and actual inference, rather than a mock result,
are used in verification. Segmentation is imperfect and only targets people;
other moving objects can influence background flow.

## Build and release

Use the resolved lockfile's matching `wasm-bindgen-cli` version:

```sh
cargo build -p spatialrust-web --release --target wasm32-unknown-unknown --features wasm,camera-tracking --locked
python scripts/build_camera_demo.py --output target/camera-site/camera --wasm target/wasm32-unknown-unknown/release/spatialrust_web.wasm
```

Serve the resulting directory on localhost for development or over HTTPS for
camera use. The build emits a `camera/build.json` with commit, tool version and
per-file size/SHA bindings. The existing Docs site workflow builds the same
artifact and publishes it at `/SpatialRust/camera/`; the homepage links to it.
Native camera tests run in the three-OS visual matrix. The existing browser job
adds JS lifecycle/mask controls and real WASM/model checks without changing
required job names.

## Evidence and limits

Slice 146AH's fixed protocol is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/mobile-camera-plan.json`.
Native tests cover known translation, stationary texture, feature loss, masks,
size/input admission, clock/gap/reset behavior and finite bounded reports. JS
controls cover async cancellation, late permissions, stop/error cleanup,
one-frame ownership and confidence-mask threshold/resampling/input admission.
The browser checks real Rust WASM, actual model inference, fake capture only,
permission recovery, narrow/wide layouts and the worker's connection CSP.

Screenshots are inspected at 390px and desktop widths. All captured content is
procedural or the browser's fake camera; no real user camera is opened by tests.
The browser harness uses Chromium with a software graphics delegate. **Physical
Android/iPhone, iOS Safari, sustained mobile performance and real-scene tracking
accuracy are not qualified by these checks.** In particular, fast motion, weak
texture, lighting changes, occlusion and long processing gaps can lose tracking.

Development logs retain the initial weak-LK translation failure and the browser
CSP, loader and model-label failures before their corrections. The delivery
receipt binds successful checks as well as these failures. Native binaries and
493 prior scientific JSON/42 frozen artifacts remain hash-protected. The 80%
operational maturity gates for canonical sensor calibration, independent motion,
full-room accuracy, Linux hardware and integrated acceptance remain open.

The [source-bound delivery receipt](receipts/2026-10-21_mobile_camera.json) records
26 terminal owned commands and preserves all six development failures. Final
results are web default/camera/all **5/17/18**, vision **83 passed/two existing
ignored**, **14 JS**, **six asset** and **15 browser** controls, plus WASM,
Clippy and formatting checks. The final browser processes three AI frames and
shows actual feature tracking; the graphics delegate is software-backed.

All builds use the E-drive cache. A fixed 2GiB positive-cache guard stopped one
all-feature build; a path/time/hash-recorded reclamation removed **598,117,517
bytes** of newly generated debug symbols and incremental cache. It retained
binaries, failed logs and frozen evidence. Final positive cache growth is
**2,067,566,157 bytes**, with approximately **47.2MB** of owned site/log/screenshots;
the combined amount is about **2.115GB**, within the fixed bounds. No toolchain,
browser or sensor payload download was needed; SDK/model assets are static web
dependencies. Public deployment and exact-head CI are checked after the receipt
commit, with their records kept separately.
