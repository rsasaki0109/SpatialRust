# SpatialRust vs Open3D benchmark

A reproducible CPU timing comparison between SpatialRust and
[Open3D](https://www.open3d.org/) on the operations both libraries expose
through their straightforward point-cloud APIs. By default both process the
**exact same** public PCL `table_scene_lms400.pcd` scan.

## What it measures

| Operation | Parameters |
| --- | --- |
| Voxel-grid downsample | voxel size 0.05 |
| Normal estimation | k = 10 neighbors |
| Statistical Outlier Removal | neighbors = 16, std ratio = 1.0 |
| Radius Outlier Removal | radius 0.1, min points 4 |

## Running

```bash
# needs: open3d plus a Rust toolchain for the SpatialRust bench example
python bench/open3d_comparison/run.py
```

The script downloads the public PCL sample into `target/bench-data/`, builds the
SpatialRust `bench_ops` example, runs both libraries, and prints a side-by-side
timing table. Use `--input cloud.pcd` to benchmark another PCD. The
`--synthetic-points 200000` fallback also needs NumPy and the SpatialRust Python
extension because it writes the generated cloud through the Python bindings.

On Unix-like shells, `bench/open3d_comparison/run.sh` is a convenience
wrapper around the Python runner.

## Indicative results

Measured on one Windows machine (Open3D 0.19.0, Python 3.12, release
SpatialRust build, 460,400-point public PCL `table_scene_lms400.pcd`):

| Operation | SpatialRust | Open3D | Speedup |
| --- | ---: | ---: | ---: |
| Voxel downsample | 0.0132 s | 0.0234 s | **1.77× faster** |
| Normal estimation | 0.1997 s | 0.4946 s | **2.48× faster** |
| Statistical Outlier Removal | 0.2105 s | 0.6565 s | **3.12× faster** |
| Radius Outlier Removal | 0.1049 s | 66.4701 s | **633.65× faster** |

## Notes

Open3D may use internal parallelism depending on how its wheel was built and on
your environment variables. Treat these as local, apples-to-apples input
comparisons rather than a controlled CPU microbenchmark. Record the CPU, Open3D
version, Python version, and thread settings when publishing numbers.
