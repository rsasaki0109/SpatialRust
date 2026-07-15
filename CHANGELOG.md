# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning policy (from 1.0.0)

From **1.0.0** onward SpatialRust follows SemVer strictly:

- **MAJOR** — breaking changes to APIs marked **Stable** in
  [`docs/API_STABILITY.md`](docs/API_STABILITY.md)
- **MINOR** — backward-compatible features; deprecations may land here
- **PATCH** — bug fixes and documentation corrections only

Breaking changes to **Stable** symbols require a major release, a CHANGELOG
entry, and migration notes. Deprecations are announced in a minor release and
removed no sooner than the next major (see `docs/API_STABILITY.md`).

## [Unreleased]

### Added

- **MCAP episode IO + TSDF meshing deepen**: `spatialrust-sync` `mcap` /
  facade `sync-mcap` writes and reads XYZ stamped records to MCAP files;
  `TsdfVolume::extract_mesh` uses truncation-band integration and marching
  tetrahedra instead of the occupied-voxel triangle proxy.

- **Canonical 2D → AI → 3D roadmap and image IO (Epic 83)**: `docs/ROADMAP.md`
  reserves Epics 83–90; new `spatialrust-image-io` provides bounded path,
  reader/writer, and memory PNG/JPEG/PNM codecs, independently gated TIFF and
  OpenEXR, typed pixels, Exif orientation handling, Python/NumPy bindings,
  property tests, and 640p/1080p/4K decode benchmarks.

- **Shared CPU filters (Epic 84A–84B)**: `spatialrust-vision` now exposes a
  common `BorderMode`, validated 1D/2D kernels, OpenCV-style filter2D
  correlation, explicit convolution, f32-output and separable filters, and
  normalized box/Gaussian blur, median and bilateral filters, signed
  Sobel/Scharr/Laplacian derivatives, and Gaussian pyramids. The feature
  includes strided-view property coverage, Python bindings/stubs, OpenCV
  comparison, and 640p/1080p/4K benchmarks.

- **CPU morphology (Epic 84C)**: validated rectangular, cross, elliptical,
  diamond, and custom structuring elements; explicit-anchor erode/dilate;
  open/close/gradient/top-hat/black-hat operations; additive feature and meta
  feature; u8/u16/f32 and strided-view tests; Python bindings; exact OpenCV
  comparisons; and 640p/1080p/4K benchmarks.

- **CPU image analysis (Epic 84D)**: fixed and adaptive thresholds, u8/u16
  Otsu selection, masked configurable histograms, exact u8 equalization,
  contrast-limited adaptive equalization, and checked summed-area tables;
  additive Rust/meta features, Python bindings/stubs, strided properties,
  OpenCV comparisons, and representative-resolution benchmarks.

- **Canny edge detection (Epic 84E)**: configurable 3/5/7 Sobel apertures,
  L1/L2 gradient magnitude, directional non-maximum suppression, 8-neighbor
  hysteresis, and inspectable intermediate stages; additive Rust/meta features,
  strided and property tests, Python binding/stub, 640p/1080p/4K benchmark, and
  exact OpenCV comparison across all six aperture/magnitude combinations.

- **Tensor foundation (Epic 85A)**: new dependency-light `spatialrust-tensor`
  crate with byte-addressable dtype, arbitrary-rank shape, signed element
  strides, checked byte offsets/spans, explicit device identity, safe borrowed
  CPU views, and named owned copies. Non-host device memory cannot be exposed as
  a Rust byte slice, and the meta-crate integration is opt-in through `tensor`.

- **Image/spatial tensor bridges (Epic 85B)**: packed interleaved images expose
  zero-copy HWC views, packed planar images expose zero-copy CHW views, and
  Schema-SoA `f32` point fields expose zero-copy one-dimensional views. Explicit
  `pack_*` operations handle padded/ROI images, with feature-alone tests and
  640p/1080p/4K packing benchmarks.

- **DLPack and Python tensor interoperability (Epic 85C–85D)**: audited
  `DLManagedTensorVersioned` major-version 1 CPU import/export, explicit deleter
  transfer, read-only/copy flags, signed strides and byte offsets, malformed ABI
  rejection, and zero-copy Python `__dlpack__`/`__dlpack_device__`. NumPy and
  PyTorch round trips preserve allocations and producer lifetimes; device or
  host copy requests remain explicit.

- **Inference contracts and ONNX Runtime CPU (Epic 86)**: new optional
  `spatialrust-ai` crate with named dynamic model metadata, stable backend and
  session traits, explicit input/output copy permissions, CPU ONNX Runtime,
  separately gated CUDA/TensorRT/DirectML providers, typed zero-copy I/O
  Binding, caller-preallocated u8/u16/f32 outputs, runtime-allocation retention,
  output-to-input chaining, Python bindings/stubs, reference-runtime comparison,
  and 640p/1080p/4K Criterion coverage. Multi-byte raw storage is rejected at
  zero-copy boundaries instead of being cast from an under-aligned byte buffer.

- **Feature2D and ORB matching (Epic 87)**: checked keypoint, binary/float
  descriptor, feature-set, and match contracts; Harris and Shi–Tomasi corners;
  OpenCV-exact FAST-9/16 detection and scores; deterministic multi-scale ORB
  with 256-bit rotated BRIEF; Hamming/L2 brute-force matching with ratio,
  cross-check, and distance filters; Python/NumPy bindings and stubs; property
  tests, OpenCV comparison, and 640p/1080p/4K Criterion baselines.

- **Camera geometry, motion, and stereo (Epic 88)**: checked correspondence and
  projective contracts; normalized DLT and deterministic RANSAC for homography,
  fundamental, and essential matrices; triangulation and essential pose
  disambiguation; EPnP-class PnP with iterative refine and RANSAC; sparse
  pyramidal Lucas–Kanade tracking; stereo rig, rectify remap grids, SAD block
  matching, and disparity-to-depth/XYZ reproject; Python bindings; OpenCV
  comparison with documented tolerances; property tests; and Criterion coverage.

- **Explicit GpuImage and chainable wgpu vision (Epic 89)**: `spatialrust-gpu`
  `gpu-image` feature adds `GpuImage` ownership with packed/`ImageView` upload,
  named stride packing, explicit readback, transfer receipts, and buffer
  recycle; device-resident `copy_gpu_image`; BT.601 `rgb_to_gray_gpu`; gray
  `box_blur_gpu` with replicate/constant-zero borders; facade `gpu-image` flag;
  headless chain tests that keep mid-pipeline `device_to_host_bytes == 0`; and
  640p/1080p/4K Criterion coverage. Texture-backed storage remains deferred.

- **Model adapters and image → AI → point-cloud pipelines (Epic 90)**:
  `MockInferenceBackend` / `ModelSource::Mock` for deterministic host inference
  without ONNX; vision `ai-adapters` for letterbox NCHW prep and
  tensor→`DepthMap`/`BinaryMask`/`Detection` decode; facade
  `ai-vision-pipeline` E2E covering RGB → mock depth → unproject → MVP.

- **Spatial records and Arrow bridges (Epic 91)**: new `spatialrust-records`
  with versioned `SpatialRecord`, schema compatibility/migration, and in-memory
  chunked sources/sinks; new `spatialrust-arrow` with Arrow C Data export/import,
  C Stream over record sources, and CPU C Device arrays; facade flags
  `records`, `arrow-c-data`, `arrow-c-stream`, `arrow-c-device`; ROADMAP 92–100
  activated with planned delivery slices.

- **Sensor time and frame graphs (Epic 92)**: new `spatialrust-sync` with clock
  domains/`SyncQuality`, stamped records, calibrated `FrameGraph` lookups,
  in-memory `MemoryEpisode` index, and `DeterministicReplayer` topic bundling;
  `mcap` feature reserved for file codecs; facade `sync` / `sync-mcap`.

- **Localization and mapping substrate (Epic 93)**: new `spatialrust-mapping`
  with `Trajectory`/`StampedPose`, `RelativeMotionEstimator`/`SyntheticOdometry`,
  `PoseGraph` root localization, and distance-based loop-closure candidates;
  facade `mapping`.

- **North-star substrate (Epics 94–100)**: new crates for scene reconstruction
  (`spatialrust-scene` TSDF/surfel/mesh + gated Gaussians), semantic intelligence
  (`spatialrust-semantic`), embodied episodes (`spatialrust-episode`), bounded
  robotics runtime (`spatialrust-runtime` + `ros2` gate), glTF/OpenUSD
  interchange (`spatialrust-interchange`), distributed execution
  (`spatialrust-distribute`), and platform stability (`spatialrust-platform`);
  facade `north-star` enables the stack without native heavy codecs.

- **North-star E2E demo**: facade feature `north-star-e2e` adds an integration
  test and example covering RGB → mock depth → stamped episode → pose graph →
  TSDF mesh → glTF JSON, plus semantic search, bounded runtime, partition graph,
  and platform conformance markers.

- **AI-ready image and vision foundation (Epics 75–79)**: mutable ROI views,
  planar/interleaved layouts and color metadata in `spatialrust-image`; new
  feature-gated `spatialrust-vision` preprocessing, warp, detection, mask/RLE,
  and dense spatial-map APIs; camera/point-cloud/pipeline bridges; Python/NumPy
  bindings and stubs; property tests, OpenCV comparison, Criterion benchmark,
  and an end-to-end vision-to-point-cloud demo.

- **OpenCV-oriented image/camera foundation**: new `spatialrust-image` typed
  packed buffers and strided zero-copy views; new `spatialrust-camera` pinhole
  projection/unprojection, Brown–Conrady distortion, aligned depth/RGB-D to
  XYZ/XYZRGB conversion, Python binding, MVP integration test, Criterion bench,
  synthetic demo, and OpenCV `rgbd.depthTo3d` comparison harness.

- **GPU Euclidean clustering** (`segment-euclidean-gpu`): WGSL uniform-grid label
  propagation, `GpuEuclideanClusterExtractor`,
  `EuclideanClusterExtractor::extract_with_policy`, MVP `cluster_policy`, and CLI
  `--cluster-policy auto|cpu|gpu`.
- **Euclidean cluster benchmark** (`bench/euclidean_cluster/`): CPU vs GPU timing
  harness with optional `--mvp-leaf` preprocess path
  (`notes/2026-07-03_euclidean_cluster_bench.md`).
- **MVP normal backend selection**: `MvpPipelineConfig::normal_policy`,
  `NormalEstimator::estimate_with_policy`, and CLI `--normal-policy auto|cpu|gpu`
  (included in `pipeline-mvp-gpu` / `feature-normal-gpu`).
- **MVP GPU normal radius**: `MvpPipelineConfig::normal_gpu_radius_scale` derives
  `search_radius` from voxel leaf when the wgpu backend is selected, enabling the
  GPU uniform-grid path (`notes/2026-07-03_mvp_normal_gpu_grid_radius.md`).
- **Provisional `SpatialTensor` chunked views** (`spatialrust-core`): zero-copy
  chunk iteration over `PointCloud` columns (`notes/2026-07-03_spatial_tensor_chunked_views.md`).

### Fixed

- GPU uniform-grid stages (Euclidean cluster, MVP normal radius) fall back to CPU when
  the spatial extent would exceed the wgpu cell cap.
- MVP integration tests: `EuclideanClusterConfig` partial initializers include
  `..Default::default()` for `gpu_min_points`.
- **GPU Euclidean cluster label propagation**: iteration count scales with grid
  span and point count; CPU fallback when propagation does not converge
  (`notes/2026-07-03_gpu_euclidean_cluster_fix.md`).
- **Grid union-find clustering** (Epic 67): replace WGSL Jacobi loops with CPU
  grid UF in `euclidean_cluster_roots_grid` (~4× faster on 460k pts, MVP path
  ~1.15×; `notes/2026-07-03_gpu_euclidean_cluster_speedup.md`).
- **Shared `uniform_grid` module** (Epic 68): `spatialrust-search` hosts grid
  bounds/build/cluster roots; GPU normals and cluster paths reuse it
  (`notes/2026-07-03_cpu_grid_euclidean_cluster.md`).
- **GPU cluster adaptive backend** (Epic 69): dense scans (≥50k pts) use KD-tree
  BFS in `GpuEuclideanClusterExtractor`; smaller clouds use parallel grid UF with
  deferred path compression (`notes/2026-07-03_gpu_euclidean_cluster_epic69.md`).
- **Chunked spatial index queries** (Epic 70): `ChunkedRadiusSearchIndex`,
  `ChunkedNearestNeighborIndex`, and `SpatialTensor` staging helpers in
  `spatialrust-search` (`notes/2026-07-03_spatial_index_chunked_queries.md`).
- **Parallel chunked queries** (Epic 71): threaded `SpatialTensor` chunk dispatch
  via `spatialrust-search/parallel` (`notes/2026-07-03_parallel_chunked_queries.md`).
- **AoSoA chunk packing** (Epic 72): interleaved XYZ buffers per chunk behind
  `tensor-aoso` (`notes/2026-07-03_aoso_chunk_packing.md`).
- **Parallel staging hot paths** (Epic 73): shared index-range staging in
  `spatialrust-search` wired into CPU normals and outlier filters
  (`notes/2026-07-03_parallel_staging_hot_paths.md`).
- **AoSoA GPU upload** (Epic 74): `GpuAoSoXyzChunk` and
  `upload_spatial_tensor_xyz_chunks` behind `gpu-aoso-staging`
  (`notes/2026-07-03_aoso_gpu_upload.md`).
- **AoSoA GPU voxel dispatch** (Epic 75): voxel keys are computed directly from
  uploaded interleaved XYZ chunks without re-uploading separate coordinate
  columns (`notes/2026-07-13_aoso_gpu_voxel_dispatch.md`).
- **AoSoA GPU voxel pipeline** (Epic 76): uploaded chunks are combined with
  GPU-to-GPU copies and flow through global key, sort/segment, and interleaved
  centroid reduction stages without intermediate readbacks
  (`notes/2026-07-13_aoso_gpu_voxel_pipeline.md`).
- **GPU-resident AoSoA result** (Epic 77): centroid results retain the combined
  interleaved position buffer for downstream kernels, with safe buffer access
  and explicit recycling (`notes/2026-07-13_gpu_resident_aoso_result.md`).
- **AoSoA attribute layouts** (Epic 78): explicit XYZ/intensity/normal
  stride-offset metadata, capability-based chunk packers, GPU upload wrappers,
  and global voxel `Average` / `First` aggregation
  (`notes/2026-07-13_aoso_attribute_layouts.md`).
- **AoSoA GPU normal chaining** (Epic 79): retained interleaved XYZ buffers feed
  normal estimation without position re-upload, and normal/curvature output can
  remain GPU-resident until explicit readback
  (`notes/2026-07-13_aoso_gpu_normals.md`).
- **GPU sparse radius grid** (Epic 80): radius cell keys, sorting, segment
  compaction, sparse neighbor lookup, and normal estimation run directly from
  retained AoSoA positions without CPU neighbor upload or dense cell allocation
  (`notes/2026-07-13_gpu_sparse_radius_grid.md`).
- **GPU-resident spatial frame** (Epic 81): schema-aware ownership of positions,
  segments, radius grids, normals, and attributes with device validation,
  explicit readback/recycling, transfer receipts, and a chained voxel-to-normal
  pipeline (`notes/2026-07-13_gpu_spatial_frame.md`).
- **Frame-native GPU operations** (Epic 82): radius-grid rebuilding, normal
  replacement, and attribute reduction execute through `GpuSpatialFrame`, with
  device/length validation, old-buffer recycling, and receipt updates
  (`notes/2026-07-13_frame_native_gpu_operations.md`).

## [1.0.0] — 2026-07-03

First stable release: **Stable** APIs in `docs/API_STABILITY.md`, MVP pipeline +
COPC IO, wgpu acceleration paths behind feature flags, and Python bindings with
stubtest CI.

### Added

- **Public COPC validation** (`bench/public_copc/`): reproducible harness and
  integration test `mvp_public_copc` exercising bounds/resolution partial reads
  on the public PCL `table_scene_lms400` sample through the MVP pipeline.
- **GPU RANSAC plane scoring** (`segment-ransac-plane-gpu`): WGSL inlier-count
  kernel, `GpuRansacPlaneSegmenter`, and `RansacPlaneSegmenter::segment_with_policy`
  with `ExecutionPolicy` selection (~11× on 460k points, ~2.7× after MVP
  downsampling in local release measurements).
- **MVP plane backend selection**: `MvpPipelineConfig::plane_policy`,
  `pipeline-mvp-gpu` feature, and CLI `--plane-policy auto|cpu|gpu`.
- **RANSAC plane benchmark** (`bench/ransac_plane/`): CPU vs GPU timing harness
  with optional `--mvp-leaf` MVP-preprocess path.
- **Public GPU buffer pool** (`GpuBufferPool` on `WgpuRuntime`): explicit
  upload/recycle for wgpu storage buffers (`upload_pod_storage`,
  `upload_u32_storage`, `clear_buffer_pool`).
- **PyG PointNet++ demo** (`crates/spatialrust-py/examples/pyg_pointnet_demo.py`):
  SpatialRust preprocessing → PyTorch Geometric tensors.
- **API stability guide** (`docs/API_STABILITY.md`): stability tiers and v1.0
  release checklist.
- **PCL comparison benchmark** (`bench/pcl_comparison/`): a reproducible,
  apples-to-apples timing harness against PCL 1.14 on voxel downsampling, normal
  estimation, and outlier removal, plus rotating cluster/voxel README GIFs
  generated through the Python bindings (`examples/make_gifs.py`).
- **Python type stubs** (`spatialrust.pyi` + `py.typed`, PEP 561): full type
  information for the compiled extension so editors and type checkers (mypy,
  pyright) get autocomplete and signature checking. CI runs `mypy.stubtest` on
  every push to keep the stubs in sync with the runtime API.
- **Python bindings** (`spatialrust-py`, PyO3 + maturin): NumPy interop for point
  clouds, `read`/`write` (PCD/PLY/LAS/COPC), `voxel_downsample`, `run_pipeline`,
  `region_growing`, and registration (`register_icp`, `register_point_to_plane`,
  `register_gicp`, `register_ndt`). Built as `abi3` wheels (CPython 3.8+).
- **Voxel occupancy grids** crate (`spatialrust-voxelize`, `voxelize-occupancy`):
  voxelizes a cloud into a dense 3D occupancy or count grid (the tensor learned
  models consume), exposed in the Python bindings as `voxelize` returning an
  `(nz, ny, nx)` NumPy array.
- **ML preprocessing example** (`examples/ml_preprocess.py`): the end-to-end
  "point cloud → model-ready tensors" pipeline (clean → unit-sphere normalize →
  FPS → voxel occupancy grid / LiDAR range image / k-NN `edge_index`), with a
  four-panel figure.
- **Neighborhood graph construction** (`spatialrust-search`, `search-graph`):
  directed k-NN and radius graphs over a cloud, exposed in the Python bindings
  as `knn_graph` / `radius_graph` returning a PyG-style `(2, E)` `edge_index`
  for graph neural networks.
- **Spherical range-image projection** (`spatialrust-voxelize`,
  `voxelize-range-image`): projects a rotating-LiDAR scan into the dense 2D
  range image used by LiDAR segmentation models, exposed in the Python bindings
  as `range_image` returning a `(height, width)` NumPy array.
- **Cloud transform utilities** crate (`spatialrust-transform`, `transform-ops`):
  4×4 affine transforms (positions + normals), recentering, scale and unit-sphere
  normalization, cloud merging, and AABB / PCA-based OBB computation, exposed in
  the Python bindings (`apply_transform`, `recenter`, `scale`,
  `normalize_unit_sphere`, `merge`, `centroid`, `bounding_box`,
  `oriented_bounding_box`).
- **Point cloud metrics** crate (`spatialrust-metrics`, `metrics-distance`):
  symmetric Chamfer and Hausdorff distances (plus directed statistics) for
  scoring registration / reconstruction against a reference, exposed in the
  Python bindings (`chamfer_distance`, `hausdorff_distance`).
- **Farthest Point Sampling** in `spatialrust-filtering` (`filter-fps`): even
  spatial downsampling to a target point count (the standard front-end for
  learned point-cloud models), exposed in the Python bindings
  (`farthest_point_sampling`).
- **MLS surface smoothing** in `spatialrust-filtering` (`filter-mls`): Moving
  Least Squares projection onto a local polynomial surface (order 1 or 2) that
  removes noise while preserving curvature, exposed in the Python bindings
  (`mls_smooth`).
- **Crop and range filters** in `spatialrust-filtering` (`filter-crop`):
  axis-aligned `CropBox` and field-range `PassThrough` (both invertible),
  exposed in the Python bindings (`crop_box`, `pass_through`).
- **Outlier removal filters** in `spatialrust-filtering` (`filter-outlier`):
  Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR), both
  exposed in the Python bindings (`statistical_outlier_removal`,
  `radius_outlier_removal`).
- **Registration suite** in `spatialrust-registration`:
  - Point-to-plane ICP (`register-icp-point-to-plane`).
  - Generalized ICP / GICP (`register-gicp`), with an optional GPU covariance
    path (`register-gicp-gpu`, ~1.7× faster covariance estimation).
  - NDT — Normal Distributions Transform (`register-ndt`), point-to-distribution
    with Levenberg–Marquardt.
  - FPFH + RANSAC global registration (`register-fpfh`): coarse alignment with
    no initial guess via Fast Point Feature Histograms and a RANSAC pose search,
    exposed in the Python bindings (`register_fpfh_ransac`).
  - Public FPFH descriptor API (`fpfh_descriptors`, `FpfhDescriptor`) and a
    keypoint-based registration path (ISS keypoints → FPFH → RANSAC) exposed in
    the Python bindings as `register_fpfh_keypoints`.
  - Registration backend selection in the MVP pipeline (`MvpRegistrationMethod`).
- **Boundary / edge point detection** (`spatialrust-features`,
  `feature-boundary`): flags hole-rim and scan-edge points via tangent-plane
  neighbor angle gaps, exposed in the Python bindings as `detect_boundary`.
- **Consistent normal orientation** (`spatialrust-features`,
  `feature-normal-orient`): MST/Prim propagation over a k-NN graph that flips
  estimated normals to agree in sign, exposed in the Python bindings as
  `orient_normals` (estimate + orient).
- **ISS keypoint detection** (`spatialrust-features`, `feature-iss`): Intrinsic
  Shape Signatures saliency with non-maximum suppression, returning a sparse
  keypoint sub-cloud; exposed in the Python bindings (`iss_keypoints`).
- **GPU normal estimation** (`spatialrust-features`, `feature-normal-gpu`): a wgpu
  path with a fully-GPU uniform-grid radius neighbor search (covariance + Jacobi
  eigensolver), up to ~50× faster than the CPU KD-tree estimator at 500k points.
- **Multi-plane segmentation** (`spatialrust-segmentation`,
  `segment-multi-plane`): sequential RANSAC that extracts the N dominant planes
  (floor, walls, ceiling) and labels each point by plane index, exposed in the
  Python bindings as `segment_multi_plane`.
- **DBSCAN segmentation** (`spatialrust-segmentation`, `segment-dbscan`):
  density-based clustering with explicit noise labeling, exposed in the Python
  bindings (`dbscan`).
- **Ground segmentation** (`spatialrust-segmentation`, `segment-ground`):
  grid-based ground/non-ground split using per-cell minimum heights eroded
  against neighbors (robust to slopes and rooftops), exposed in the Python
  bindings (`ground_segmentation`).
- **RANSAC sphere & cylinder fitting** (`spatialrust-segmentation`,
  `segment-ransac-primitives`): detect spheres (positions only) and cylinders
  (axis recovered from surface normals), partitioning inliers/outliers, exposed
  in the Python bindings (`ransac_sphere`, `ransac_cylinder`).
- **Region growing segmentation** (`spatialrust-segmentation`,
  `segment-region-growing`): normal-smoothness region growing with curvature
  seeding.
- Benchmarks comparing CPU vs GPU normal estimation and the four registration
  backends (speed + accuracy), with writeups under `notes/`.
- End-to-end Python example (`end_to_end.py`): outlier removal → downsample →
  DBSCAN → FPFH global registration → ICP refinement, rendered as a four-panel
  figure; accepts a real scan via `--input`.
- CI: Python wheel build/publish workflow; benchmark-compile and per-feature
  test matrix entries for all new features.

### Changed

- Split the AoSoA GPU staging implementation into focused `types`, `upload`,
  `attributes`, `normals`, `voxel`, and `tests` modules without changing its
  public API or measured performance
  (`notes/2026-07-13_aoso_module_cleanup.md`).
- Restrict internal normal-shader re-exports to `gpu-aoso-staging`, fixing
  `-D warnings` failures in narrower GPU/segmentation CI feature jobs.

- **Plane Auto threshold**: `DEFAULT_GPU_MIN_POINTS_PLANE` lowered from 100,000
  to 2,000 so MVP downsampled clouds (~2k points on the reference sample) select
  GPU under `Auto`; override with `RansacPlaneConfig::gpu_min_points`.
- Performance: radius outlier removal is **~16× faster** (1.70 s → 0.10 s on
  210k points) via a new `KdTree::radius_reaches` early-exit density test that
  stops once enough neighbors are found and allocates nothing. `radius_search`
  also no longer sorts results (no caller relied on the order). CPU voxel
  downsampling is **~2× faster** (0.047 s → 0.022 s on 210k points) via a fast
  integer hasher and a single-pass centroid accumulator for the default
  Centroid + Average case. Measured against PCL 1.14, SpatialRust now wins 3 of
  4 compared operations, with PCL's voxel grid keeping a ~2× edge
  (`bench/pcl_comparison/`).

### Fixed

- Workspace `[workspace.dependencies]` version requirements aligned to `1.0.0`
  so `cargo build` resolves after the release version bump.
- **Python binding tests**: a pytest suite (`crates/spatialrust-py/tests/`)
  covering the NumPy ⇄ Rust boundary of every exported function (shapes, dtypes,
  keyword signatures, and sane results on synthetic clouds), wired into CI as a
  `python-bindings` job that builds the extension with maturin and runs the
  tests on Python 3.8 and current — closing a gap where the bindings were only
  compile-checked, never imported or called.
- CI now runs clippy under `--all-features` (library targets) and a full
  `--all-features` test pass, closing a gap where clippy and combined-feature
  builds were only checked at default features; resolved the pre-existing
  clippy warnings this surfaced in the GPU and IO crates.
- Boundary detection panicked on isolated points when `min_neighbors` was 0
  (empty angle list indexed out of bounds); it now returns non-boundary.
- RANSAC samplers (plane and sphere / cylinder) drew indices from the low
  (short-period) bits of their LCG; they now use the well-mixed high bits via
  multiply-shift, making fits more reliable (the plane fix is what let
  multi-plane extraction find every plane).
- Gated the `io-copc-http` integration test behind its feature so
  `cargo test --workspace` builds with default features.
- Resolved pre-existing rustfmt and clippy drift surfaced by current stable
  toolchains so the CI fmt/clippy gates pass.

## [0.1.0] — MVP foundation

### Added

- MVP pipeline end-to-end: PCD/PLY/LAS/COPC IO, voxel downsampling (CPU + optional
  wgpu), normal estimation, RANSAC plane segmentation, Euclidean clustering, and
  point-to-point ICP.
- COPC partial reads (bounds + LOD) in the library and `spatialrust-mvp` CLI.
- wgpu voxel downsampling with automatic CPU/GPU policy selection.

[Unreleased]: https://github.com/rsasaki0109/SpatialRust/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rsasaki0109/SpatialRust/releases/tag/v1.0.0
[0.1.0]: https://github.com/rsasaki0109/SpatialRust/releases/tag/v0.1.0
