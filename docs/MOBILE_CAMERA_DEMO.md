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
processing. Camera selection, AI masking and processing size can be changed while stopped. The
browser never requests a microphone, records a clip or persists a frame.

## Boundary and ownership

Only the optional `spatialrust-web/camera-tracking` feature adds image and vision
odometry dependencies. Existing core and vision algorithms are unchanged. The
native `CameraTracker` accepts 64–320 pixels per axis and at most 80 tracks. Its
copy-named WASM method validates lengths before copying JavaScript input. Invalid
sizes, mask labels and nonfinite or non-increasing times fail before retained
state changes; a gap over 500ms reseeds. Missing/lost tracking returns no motion.

The demo resizes to a 256-pixel long edge by default (192 in light mode), transfers one `ImageBitmap` at a time,
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

## Camera startup and changing frame geometry

A camera start commits presentation only while its session is current, including
after awaiting `video.play()`. A late completion after Stop or replacement cannot
restore the old canvas, label, size or backend. Stream/worker cleanup remains
owned by the existing session controller.

During capture the page derives its processing size from current intrinsic video
dimensions. It retains a 256-pixel long edge and admits rounded short sides of at
least 64 pixels. Unsupported extreme ratios fail with an explanation instead of
stretching the short side. Geometry changes happen only after the in-flight
result is consumed, so its image-space coordinates remain valid. An async bitmap
whose source dimensions changed is closed without submission. Resizing clears
old vectors and timing; the worker's existing dimension branch reconstructs the
Rust tracker and returns a warming report while retaining its AI instance.

These controls follow intrinsic image size changes, including portrait/landscape
transitions reported by a browser. They do not qualify every phone's orientation
behavior. Chromium tests resize an actual synthetic `canvas.captureStream`, hold
an actual worker result across rotation, then hold a bitmap across rotation back.
Another test delays `play()` while retaining metadata to check Stop and a
replacement sample session. Physical phone and iOS Safari remain untested.

## Camera source termination

The session watches acquired video tracks for `ended` and their stream for
`inactive`, including during worker initialization and video startup. It also
checks already ended/inactive state on acquisition, after startup, and before
each running camera animation step. This check precedes the busy-frame gate, so
an interrupted source can release resources while a worker result is pending.

Termination stops the worker and every owned stream track, clears old vectors,
mask, frequency and dimensions, restores the camera-off presentation and enables
Start, sample and settings. The message asks the visitor to start the camera
again or try a sample. It does not suggest changing AI for a source interruption.
There is no automatic permission request or camera restart.

Stream listeners belong to the session's abort controller. Stop removes them
before stopping tracks; queued old callbacks and late worker results cannot
affect a replacement session. Source-state polling covers a silent track stop
without relying solely on the browser's event delivery. Transient muting is not
treated as termination. A still-live source with frozen pixels or stalled
decoding remains outside this termination check; this is not a freshness timer.

## Processing size

While stopped, choose **処理サイズ → 軽量 · 192px** to reduce the tracking image.
Standard uses a 256-pixel long edge; light uses 192, for camera and sample alike.
The choice is captured at Start and stays fixed through loading and processing.
Stopping enables the control again and clears the displayed image size. Rotation
retains the chosen long edge, preserves aspect and warms a fresh tracker after
the previous result. The current processing dimensions appear below the stats.

For a 4:3 scene the image changes from **256×192 (49,152 pixels)** to
**192×144 (27,648 pixels)**, a **43.75% reduction in copied/tracked pixel count**.
This reduces grayscale/readback/mask image work; it is not a measured runtime,
power or accuracy gain. The AI model, its inference size, transfer ownership and
eight-frame-per-second submission cap remain unchanged. Light mode trades image
detail for less image work; it does not promise to fix tracking gaps over 500ms.

Each axis must still contain at least 64 pixels. At the presets' nominal sizes,
standard admits ratios up to 4:1, light up to 3:1 (with the same pixel rounding).
Unsupported ratios are rejected with a suggestion to try standard or another
camera; the image is never stretched to make its short side fit. The procedural
sample scales its title and motion with the selected size. No real person mask
is claimed for that sample scene.

## Submission cadence and displayed frequency

The presenter admits a frame using the current monotonic clock and waits at
least 125ms after the previous successful bitmap transfer returns. Async bitmap
preparation cannot consume that next interval. One capture/result remains in
flight. Each frame carries the monotonic time immediately before bitmap creation;
the existing Rust report echoes it. This is an application capture-invocation
clock, without physical sensor-clock or exposure-time calibration.

The displayed frequency uses timestamps of completed images, with an inclusive
two-second window and at most 17 retained samples. It divides image intervals by
their elapsed time, showing one decimal place. Worker result arrival jitter is
excluded from that calculation. First samples after Start, geometry reset or a
gap exceeding the window show **—** until a second image is available. Stop and
geometry changes clear the retained rate. Invalid/non-increasing timestamps fail
before the rate state changes; nonfinite arithmetic is never displayed.

The **処理 / ms** value remains the worker's combined image/AI/tracking time.
Frequency, combined worker time, camera capture FPS and model-only throughput
have different meanings; these displays do not establish physical GPU or phone
performance. The existing Rust gap admission, native algorithm, AI model and
explicit transfer/readback ownership remain unchanged.

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

## Source termination delivery evidence (146AL)

The [source receipt](receipts/2026-10-21_camera_source.json) binds eight terminal
commands and three parent failures: ended, inactive and silently stopped sources
retained the running presentation. Final results pass **39 JS**, **six asset**
and **42 browser** controls. Unit cases include already dead acquisition, worker
initialization, pending video startup, cleanup, late callbacks and an unrelated
audio event. Browser cases hold an actual worker result, stop actual fake-camera
tracks, check released resources/cleared fields/restart controls, then deliver
old source events and the held result into a replacement sample session.

Browser source events are synthetic; this does not qualify physical device
removal. The fixed contract is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-source-plan.json`.
Reuse the release WASM and **41,087,603** hard-linked vendor bytes with **zero**
compiler-cache growth. Local site/log/screenshots add about **3.44MB** beyond
shared vendor files. Preserve **291** prior camera files, 493 scientific JSON,
42 frozen native artifacts and the resolved lockfile. Live-source freezes,
physical phone/Safari behavior and operational 80% qualification remain open.

## Timing delivery evidence (146AK)

The [timing receipt](receipts/2026-10-21_camera_timing.json) binds seven terminal
commands and both reproduced parent-browser failures: a delayed bitmap consumed
the next submission interval, and delayed result delivery inflated displayed
frequency. Final checks pass **31 JS**, **six asset** and **36 browser** controls,
including actual WASM capture-time echoes, 125ms submission separation after a
600ms held bitmap, and frequency derived from completed image timestamps after
a held real worker result. Existing model, rotation, presets and lifecycle/CSP
controls remain. The saved capture is synthetic, without real user camera reads.

The fixed contract is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-timing-plan.json`.
Reuse the release WASM and **41,087,603** hard-linked vendor bytes; compiler-cache
growth is **zero**. Local build/log/screenshots add about **3.19MB** beyond shared
vendor files. Preserve all **230** prior camera files, 493 scientific JSON,
42 frozen native artifacts and the resolved lockfile. Physical camera clocks,
phone/Safari performance and the original operational 80% gates remain open.

## Light preset delivery evidence (146AJ)

The [light preset receipt](receipts/2026-10-21_camera_light.json) binds five terminal
commands, **25 JS**, **six asset** and **32 browser** controls. Both preset sizes
run real WASM tracking; both intrinsic rotations run the pinned AI model and
validate mask dimensions. Tests also check default size, frozen active settings,
restart at standard size, cleanup and existing CSP/lifecycle controls. Software
Chromium captures are synthetic; physical phones and Safari remain unqualified.

The fixed contract is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-light-plan.json`.
The local build hard-links the existing **41,087,603 vendor bytes**, preserves
all **173** prior camera files, 493 scientific JSON, 42 frozen native artifacts,
the release WASM and resolved lockfile. Compiler-cache growth is **zero**; the
local site, logs and screenshots add about **4.15MB** beyond the shared vendor
files. No model/browser/toolchain/sensor acquisition or Rust compilation occurs.

## Intrinsic geometry delivery evidence (146AI)

The [geometry receipt](receipts/2026-10-21_camera_geometry.json) binds eight
terminal commands, two reproduced parent-browser failures, **21 JS controls**,
**six asset controls** and **22 browser checks**. The tests use the actual Rust
WASM and pinned AI model. They retain a real worker result during a synthetic
portrait transition, hold a bitmap while changing back, and verify fresh tracker
warming, obsolete-capture rejection and one AI worker throughout. Late startup
is checked against both a stopped page and a replacement sample session.

The fixed contract is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-geometry-plan.json`.
Existing vendor payloads (**41,087,603 bytes**) are hard-linked into a separate
E-drive site after hash checks; they are not fetched or copied again. The same
release WASM, resolved lockfile, 493 scientific JSON, 42 frozen native artifacts
and all 111 previous camera files remain hash-protected. No Rust source,
dependency, native algorithm or transfer API changes, and no local Cargo
compilation is needed. Final pre-release logical owned size is about **44.6MB**,
including the shared vendor payload; non-shared owned files are about **3.5MB**
and compiler-cache growth is **zero**. Post-deployment verification has its own
32MiB streaming limit and does not retain downloaded vendor payload files.

These are software-browser checks of changing intrinsic frame dimensions,
not physical device, iOS Safari or real-motion accuracy acceptance. Exact-head
CI and the matching Pages deployment are verified after the receipt commit.

The first CI attempt correctly refused to overwrite an existing browser evidence
folder restored with the Rust target cache. Browser evidence now uses the
runner's temporary directory with the workflow run ID and attempt in its name.
The fresh-output invariant remains enforced; previous cached evidence is kept.
Application code and the successful local geometry controls are unchanged by
this CI path correction.
