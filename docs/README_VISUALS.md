# README opening animations

The opening group is rendered from actual SpatialRust results in a shared dark
palette, with readable labels and a link to the mobile camera playground.
`readme_hero.gif` presents the pipeline; the smaller animations show Euclidean
cluster labels, occupied voxel cells and Rust/WASM image-space feature tracking.
Nine GIFs form a full-width banner and eight gallery cards: depth-colored scan,
voxel filtering, plane segmentation, clusters, occupancy, depth slicing, image
tracks and image flow. Depth slicing is a geometric illustration over the same
scan, not an additional query API. Flow arrows are enlarged seven times for
legibility; numeric motion stays in original pixels.
The renderer limits each GIF to 1.5 MB and the group to 3 MB. It holds bounded
frame sequences in memory instead of writing intermediate frames to disk.

## Inputs and meaning

- Point clouds: the public PCL
  [`table_scene_lms400.pcd`](https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd)
  scan (460,400 points, 5,649,007 bytes). SHA256:
  `e285d415641e0d9de695b611db874cc8fe995e8089b77a50d6056d24d8cbcc58`.
  The renderer selects every sixth finite point (76,734 points), runs the Rust
  pipeline through the Python extension and uses its returned downsampled and
  labeled clouds. Exact output membership identifies removed plane points;
  the renderer does not fit its own plane or invent cluster labels. Occupied
  8 cm cells are returned by `spatialrust.voxelize`.
- Execution: voxel policy is explicitly `cpu`. The installed extension used
  for these assets does not expose resolved stage policies; its receipt records
  `null` and its binary hash. The visuals make no GPU execution or speed claim.
  Build the current extension to reproduce with current stage-policy receipts.
- Tracking: 48 generated 256×192 gray frames are passed to the actual
  `BrowserCameraTracker` WASM API at logical 125 ms intervals. All 47 frames
  after initialization track successfully. The renderer reconstructs the
  same input and checks every frame's byte hash before drawing returned tracks.
  No camera, person model, upload or physical-phone capture is involved; the
  animation is visibly marked **SYNTHETIC / AI OFF**. The playground separately
  offers opt-in camera use and optional local AI person masking.

Viewpoint movement and GIF playback are presentation choices. They do not show
live processing latency, GPU acceleration, metric pose or SLAM qualification.

## Regeneration

Use a built SpatialRust Python extension (`maturin develop --release` from
`crates/spatialrust-py`), NumPy, Pillow and Node.js. Reuse a built camera package
from `scripts/build_camera_demo.py`, or generate only its Rust/WASM package:

```sh
cargo build --release -p spatialrust-web --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir target/readme-camera-pkg target/wasm32-unknown-unknown/release/spatialrust_web.wasm
```

Use the wasm-bindgen CLI version matching Cargo's wasm-bindgen version. Keep
the input scan and the generated tracking report outside tracked directories.
The exporter and renderer perform no downloads. Then run from the repo root:

```sh
node scripts/readme_tracking_data.mjs --pkg target/readme-camera-pkg --output target/readme-tracking.json
python scripts/render_readme_heroes.py --input /absolute/path/table_scene_lms400.pcd --tracking target/readme-tracking.json --font /absolute/path/sans.ttf --output docs/assets
```

The exporter requires a fresh output file. The renderer updates only the nine
opening GIFs and `docs/assets/readme_heroes.json`; it does not modify other README
visuals. Use a local sans font such as DejaVu Sans or Segoe UI. The font hash,
Python extension hash, WASM hash, generator hashes, dependency versions, pipeline
parameters and asset hashes are recorded in that provenance receipt. Different
fonts or builds can change pixels and hashes; rerun the visual checks after
regeneration. No font binary, scan, WASM package or sensor data is committed.

The older `readme_mvp_preview` and `make_gifs.py` examples also write some of
these filenames. Run the opening-group renderer last when regenerating all
README media.

## Verification

The prior four-GIF group was about 1.54 MB. The expanded nine-GIF gallery stays
under the same 3 MB cap; exact sizes are in `assets/readme_heroes.json`.
Decode every frame, verify image
dimensions, looping, meaningful frame changes, the provenance hashes and the
combined size. Inspect all four pipeline phases and all eight cards at
desktop and phone widths. Check that the tracking link points to the playground
and that the synthetic/AI-off label and experimental 2D description remain visible.
