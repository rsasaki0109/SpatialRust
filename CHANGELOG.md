# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **PCL comparison benchmark** (`bench/pcl_comparison/`): a reproducible,
  apples-to-apples timing harness against PCL 1.14 on voxel downsampling, normal
  estimation, and outlier removal, plus rotating cluster/voxel README GIFs
  generated through the Python bindings (`examples/make_gifs.py`).
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

[Unreleased]: https://github.com/rsasaki0109/SpatialRust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rsasaki0109/SpatialRust/releases/tag/v0.1.0
