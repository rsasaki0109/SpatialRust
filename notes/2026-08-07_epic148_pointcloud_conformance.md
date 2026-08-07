# Epic 148: point-cloud conformance program (PCL/PDAL comparison)

Date: 2026-08-07. Slices 148A/148B.

## Why

SpatialRust proves image correctness/speed against OpenCV with dated, honest
receipts (Epics 101–111). Epic 148 does the same for point clouds so the
"Rust-native point cloud speed" claims are reproducible and fair, following the
same fail-closed report contract style.

## What was built

- `bench/pcl_comparison/manifest.json` —
  `spatialrust.pointcloud-benchmark-manifest.v1` with `small` (460,400-point
  public PCL scan) and `full` (2M synthetic room) profiles, required
  statistics, publication receipt fields, and six workloads spanning
  `filtering`, `features`, and `io-transform` domains.
- `bench/pcl_comparison/report.py` —
  `spatialrust.pointcloud-comparison.v1` report contract. Stdlib-only: timed
  sampling, robust dispersion stats, environment receipt (PCL/PDAL/Open3D
  versions), make/emit/validate/load helpers, finite-value and required-key
  checks.
- `bench/pcl_comparison/test_report.py` — seven contract tests gating the
  schema without installing any library.
- `bench/pdal_comparison/` — `pdal_bench.py` (voxelcenternearest 0.05,
  filters.transformation translate, filters.reprojection) and `run.sh` that
  builds `bench_ops` and prints the side-by-side table. PDAL is comparison
  tooling only.
- `crates/spatialrust/examples/bench_ops.rs` — adds a `translate_xyz` workload
  behind `transform-ops` so SpatialRust can be compared against PDAL's
  transform filter.
- `bench/pcl_comparison/receipt-2026-08-07.json` — dated honest receipt.

## Dated result (Linux, libpcl-dev, 460,400-point public cloud)

| Operation | SpatialRust | PCL | Speedup |
| --- | ---: | ---: | :--- |
| Voxel downsample | 0.0104 s | 0.0177 s | 1.70× |
| Normal estimation | 0.2171 s | 0.9893 s | 4.56× |
| Statistical Outlier Removal | 0.2297 s | 1.1272 s | 4.91× |
| Radius Outlier Removal | 0.1088 s | 0.7200 s | 6.62× |

Output point counts match between libraries (voxel rounding differs by the
implementation's voxel-origin convention). These are honest single-run
numbers, not portability guarantees.

## Next slices

148C unifies PCL/PDAL/Open3D receipts into an aggregate runner with fail-closed
checks; 148D publishes docs and a dated receipt. PDAL/Open3D were not installed
on this host, so their runners are ready but their dated numbers must be
produced on a machine with those tools.
