# Migrating from OpenCV-centric vision to SpatialRust Vision 1

SpatialRust Vision 1 keeps image ownership, device placement, transfer cost,
and optional runtimes visible. The stable foundation follows SemVer; newer
geometry, video, odometry, photography, GPU, and execution-graph APIs remain
additive and provisional behind named features.

## Data and color conventions

- Replace an owning `cv::Mat` with `Image<T, CHANNELS>` and a borrowed ROI with
  `ImageView`. A view carries width, height, scalar row stride, and metadata.
- OpenCV commonly uses BGR. SpatialRust RGB APIs expect RGB; convert at the IO
  boundary and attach matching `ImageMetadata` rather than relying on an
  implicit global convention.
- Dense depth uses metric `f32` plus explicit scale/range options. Invalid
  depth/flow is represented and counted; it is not silently filled.
- NumPy entry points validate dtype and shape. Reusable `out=` paths make
  allocation ownership explicit for resize, grayscale, normalization, and CHW.

## API mapping

| OpenCV concept | SpatialRust Vision 1 |
| --- | --- |
| `cv::resize` | `resize` / `resize_into` with explicit `Interpolation` |
| `cv::cvtColor` | `rgb_to_gray[_into]`, `rgb_to_hsv` |
| `cv::filter2D` | `filter2d` (correlation) or explicitly named convolution |
| ORB + BFMatcher | `detect_and_describe_orb`, `match_descriptors` |
| `findHomography`, `solvePnP` | deterministic robust geometry/PnP contracts |
| StereoBM / RGB-D reprojection | `stereo_block_match`, camera RGB-D conversions |
| Farneback / video tracking | feature-gated dense flow, LK, background, tracker |
| Stitcher / exposure tools | feature-gated photography and bounded panorama |
| `UMat` / implicit acceleration | explicit `GpuImage::upload`, resident kernels, `readback` |
| ad-hoc threaded pipeline | bounded execution graph with named transfer receipts |

## CPU, GPU, and runtime boundaries

CPU functions never move an image to a device. GPU chains begin with an
explicit upload and end with a caller-requested readback; receipts identify
every transfer. Codec, ONNX, ROS 2, CUDA, and external video runtimes remain
separate features or adapter crates. Do not enable `full` merely to obtain one
algorithm in production—select the narrow feature that owns it.

## Error and determinism policy

SpatialRust returns typed errors for invalid dimensions, strides, transforms,
budgets, and feature contracts. Robust estimators accept deterministic seeds.
Tie-breaking, scan order, invalid borders, timestamp monotonicity, and track ID
lifecycle are documented so tests do not depend on incidental scheduling.

## Release and compatibility policy

Stable symbols in `docs/API_STABILITY.md` require a major version for breaking
changes. Provisional feature groups may evolve with migration notes. Before a
Vision 1 release, run the three-OS conformance workflow, Python/stub tests,
OpenCV comparison suites, unsafe audit, canonical benchmarks, both Vision 1
examples, and the `Vision1ReleaseGate` receipt. The required migration-policy
identifier is `vision-1`.
