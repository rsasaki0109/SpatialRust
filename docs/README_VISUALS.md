# README opening animations

The nine opening GIFs use actual SpatialRust processing on a real public PCL
scan and a real TUM camera recording. A full-width pipeline banner is followed
by eight cards: scan input, voxel centroids, RANSAC membership, Euclidean
clusters, occupied cells, plane removal, ORB detections and dense image flow.
No synthetic image input, fabricated image motion or interpolated movie frames
are used in this opening group. Viewpoint animation presents the saved point
cloud outputs; it is not the algorithm's execution progress or latency.

## Point cloud input and execution

The public PCL [table_scene_lms400.pcd](https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd)
contains 460,400 measured points (5,649,007 bytes), SHA256
`e285d415641e0d9de695b611db874cc8fe995e8089b77a50d6056d24d8cbcc58`.
The renderer selects every sixth finite point (76,734 points) and invokes the
Rust pipeline through the Python extension with voxel policy explicitly `cpu`.
The returned voxel centroids, plane membership and Euclidean labels drive the
visuals. Plane removal displays the actual complement of the inlier plane.
Occupied 8 cm cells come from `spatialrust.voxelize`. The capture card shows the
input; depth colors and rotating viewpoints are display choices.

The installed extension used here does not expose resolved stage policies;
the receipt records `null` and its binary hash. These assets make no claim of
GPU execution or speed. Build the current extension to reproduce with its
current policy receipts.

## Real camera recording and processing

Source: [TUM RGB-D benchmark, freiburg1_xyz](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_xyz),
[official RGB movie](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz-rgb.avi).
The original 640x480, 30 fps movie is 8,059,298 bytes, SHA256
`1820b52939af2a2e7afdf78f6398cfbc7816399c8809b3836a9ba8da9bef8022`.

Attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers,
*A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
The recording is [CC BY 4.0](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license).
The GIFs reduce, annotate and slow the recording. The image content remains
third-party CC BY 4.0 material; it is not relicensed under the SDK's license.

- Decode original frames 0 through 63, in order, with OpenCV's AVI decoder.
  OpenCV performs no feature extraction, optical flow or tracking.
- Reduce each original RGB frame to 192x144 with Pillow BOX and convert to L8.
- Invoke actual `spatialrust.dense_flow_image` on all 63 original adjacent pairs
  with block radius 2 and search radius 3.
- On frames 1, 3, ..., 63, invoke actual `spatialrust.orb_features(max_features=80)`.
  Render these 32 images at 125 ms each. Each displayed flow result still uses
  its immediately preceding original frame, not the preceding displayed frame.
- Draw returned ORB coordinates and sample returned flow vectors on a fixed
  grid, skipping all non-finite flow pixels (also counted in the receipt).
  Vector coordinates use exactly the same 1.5x display scale (192x144 input shown at 288x216) as the image;
  no extra motion amplification or smoothing is applied.
- Record each source frame number, decoded RGB and actual grayscale hashes,
  preceding grayscale hash, full dense-flow hash, actual ORB coordinate count
  and hash (little-endian float64 Nx2), and descriptor hash in
  `assets/readme_heroes.json`, alongside the native extension,
  decoder and renderer identities. No fabricated detections or fitted arrows.

ORB detections are independent per frame, not persistent object IDs. Dense flow
is raw integer block matching with no confidence filtering; its arrows are not
validated correspondences. Movie frame rate is not a calibrated sensor clock.
These are visual processing examples, not accuracy measurements or proof of
metric camera pose, SLAM, AI inference or physical-phone performance. The linked
phone playground uses a separate Rust/WASM tracker and optional person model.
`real_orb_features.gif` and `real_dense_flow.gif` identify these two processing paths.

## Regeneration and limits

Use a built SpatialRust Python extension (`maturin develop --release` from
`crates/spatialrust-py`), NumPy, Pillow and OpenCV for AVI decoding. Obtain the
pinned public inputs explicitly and keep them outside tracked directories.
Neither renderer downloads data or installs dependencies.

```sh
python scripts/render_readme_heroes.py --input /absolute/path/table_scene_lms400.pcd --video /absolute/path/rgbd_dataset_freiburg1_xyz-rgb.avi --font /absolute/path/sans.ttf --output docs/assets
```

The renderer writes nine named GIFs and their provenance receipt. It processes
only 64 decoded video frames and keeps bounded frame sequences in memory;
intermediate PNGs, input movies, raw scans and native binaries are not committed.
Each GIF is limited to 1.5 MB and the complete group to 3 MB. Reuse a local font
such as DejaVu Sans or Segoe UI; its hash is included in the receipt. Different
fonts, decoders or native builds may change pixels and hashes.

Decode every GIF frame, check meaningful changes, looping and sizes, validate
receipt/source/output hashes, and inspect both processing results and input
images at desktop and phone widths. Verify source-frame indices and that the
video processing used Rust, independently of the display annotations.

The older synthetic `readme_tracking_data.mjs` exporter and other GIF examples
remain available for their own tests. They are not inputs to the opening gallery.
