# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Python bindings** (`spatialrust-py`, PyO3 + maturin): NumPy interop for point
  clouds, `read`/`write` (PCD/PLY/LAS/COPC), `voxel_downsample`, `run_pipeline`,
  `region_growing`, and registration (`register_icp`, `register_point_to_plane`,
  `register_gicp`, `register_ndt`). Built as `abi3` wheels (CPython 3.8+).
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
  - Registration backend selection in the MVP pipeline (`MvpRegistrationMethod`).
- **GPU normal estimation** (`spatialrust-features`, `feature-normal-gpu`): a wgpu
  path with a fully-GPU uniform-grid radius neighbor search (covariance + Jacobi
  eigensolver), up to ~50× faster than the CPU KD-tree estimator at 500k points.
- **DBSCAN segmentation** (`spatialrust-segmentation`, `segment-dbscan`):
  density-based clustering with explicit noise labeling, exposed in the Python
  bindings (`dbscan`).
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

### Fixed

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
