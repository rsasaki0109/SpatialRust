# SpatialRust vs Open3D benchmark

A reproducible CPU timing comparison between SpatialRust and
[Open3D](https://www.open3d.org/) on the operations both libraries expose
through their straightforward point-cloud APIs. Both process the **exact same**
synthetic scan, written once to a PCD file.

## What it measures

| Operation | Parameters |
| --- | --- |
| Voxel-grid downsample | voxel size 0.05 |
| Normal estimation | k = 10 neighbors |
| Statistical Outlier Removal | neighbors = 16, std ratio = 1.0 |
| Radius Outlier Removal | radius 0.1, min points 4 |

## Running

```bash
# needs: open3d, numpy, and the SpatialRust Python extension installed into
#        ../../.venv (maturin develop --release)
python bench/open3d_comparison/run.py 200000
```

The script reuses the deterministic synthetic cloud generator from
`bench/pcl_comparison/gen_cloud.py`, builds the SpatialRust `bench_ops` example,
runs both libraries, and prints a side-by-side timing table.

On Unix-like shells, `bench/open3d_comparison/run.sh 200000` is a convenience
wrapper around the Python runner.

## Notes

Open3D may use internal parallelism depending on how its wheel was built and on
your environment variables. Treat these as local, apples-to-apples input
comparisons rather than a controlled CPU microbenchmark. Record the CPU, Open3D
version, Python version, and thread settings when publishing numbers.
