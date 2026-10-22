# Public-photo camera sample (146AW)

Open the [camera playground](https://rsasaki0109.github.io/SpatialRust/camera/),
leave **AIで人物を除外** enabled and press **人物サンプル**. A public photograph
appears on the generated background; the existing local model tints predicted
person pixels while Rust/WASM tracks the surrounding background. Standard
256×192 and light 192×144 playback both work without camera permission.
**サンプル再生** retains the original procedural scene. The photo is requested
only after its own button is pressed.

The photograph is real, and its presentation motion is synthesized. The on-screen
**PHOTO SAMPLE · 動きは合成** label distinguishes this scene from live capture.
The feature exercises image-space masking and flow, with physical phone/Safari,
calibrated motion, metric pose and operational 80% acceptance remaining separate.

## Source and distribution

The unmodified 512×512 `astronaut.png` comes from scikit-image v0.25.2. Its
[official data documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)
identifies the NASA photograph as public domain. The page retains NASA /
scikit-image attribution. The pinned source is:

```text
https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png
791555 bytes
SHA256 88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5
```

The existing builder verifies the size/hash and self-hosts the image as
`vendor/person-sample.png`; the image binary is not committed. All nine previous
SDK/model/license asset pins remain identical. The same-origin CSP, model,
threshold and Rust/WASM tracker remain. Drawing the photograph uses the existing
canvas/bitmap/worker path with its explicit image and mask readbacks/copies.

## Loading and ownership

The new image loader observes the session's abort signal and accepts only the
expected decoded dimensions. Success, cancellation and image failure remove its
listeners. Stop releases the source; a late load/error cannot affect another
session. The existing 45-second startup deadline also covers image loading.
Cancellation returns application control and does not promise a bound on native
browser decoder work or its internal cache.

HTTP failure, invalid PNG and unexpected decoded dimensions stop the session and
offer an explicit restart. Procedural playback still works afterward. The photo
button is disabled during startup/playback to retain one session and one in-flight
bitmap. The existing camera and procedural paths do not request the photograph.

## Evidence and limits

The [source-bound receipt](receipts/2026-10-22_camera_person_sample.json) records
99 JS tests, including seven new image-load/cancellation controls, six asset
admission tests, 20 targeted photo browser controls and 106 full browser controls.
The full set retains the existing capture, result, camera-source, startup,
geometry, cadence, rate and transport regressions. The photo paths request no
camera or microphone; the wider harness uses fake cameras for lifecycle tests.

Actual model/WASM observations in both complete local browser runs were:

| Processing image | Predicted person pixels | Returned tracks | Endpoints inside predicted person mask |
| --- | ---: | ---: | ---: |
| 256×192 | 9,676 | 78 | 0 |
| 192×144 | 4,508 | 29 | 0 |

AI-off photo playback also returns background flow, with an empty mask and
backend `OFF`. Image failure, Stop during a held request, stale image events and
replacement playback recover. Mobile screenshots were inspected and the wider
suite checks 320-pixel and desktop layouts.

These counts are observations on one sample, not segmentation ground truth or
an accuracy benchmark. The browser reports a GPU delegate while running with
software WebGL; that label does not qualify physical GPU hardware or phone speed.
No latency, frame-rate or metric-motion improvement is claimed.

The 128MiB owned-root cap covers the reused assets and browser evidence. The nine
previous vendor files reuse 41,087,603 logical bytes through hardlinks. Only the
791,555-byte public image is newly acquired; the existing WASM is reused without
local Rust compilation, and shared build-cache growth is zero. All 2,615 files
in the previous LK output root, 493 scientific JSON and 42 frozen native artifacts
are verified unchanged. The receipt retains actual usage and peak samples.

## Reproduction

The prospective plan is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/camera-person-sample-plan.json`.
Use the existing builder with an already built camera-tracking WASM module and
the matching wasm-bindgen CLI, then run the checked-in browser harness with
installed Playwright/Chromium:

```text
node --test crates/spatialrust-web/web/camera_controller.test.mjs
python -m unittest discover -s scripts/tests -p test_camera_demo_assets.py
node scripts/test_camera_demo.cjs --site /absolute/path/site --output /absolute/path/fresh-photo-evidence --regression-case photo
node scripts/test_camera_demo.cjs --site /absolute/path/site --output /absolute/path/fresh-full-evidence
```

Every browser evidence directory must be fresh. Exact-head CI, matching Pages
publication and a public browser check complete the delivery after local evidence.
