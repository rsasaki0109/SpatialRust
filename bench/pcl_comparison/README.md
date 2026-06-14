# SpatialRust vs PCL benchmark

A reproducible, apples-to-apples timing comparison between SpatialRust and
[PCL](https://pointclouds.org/) on the operations both libraries implement. Both
process the **exact same** synthetic scan (written to a PCD file) with matching
parameters.

## What it measures

| Operation | Parameters |
| --- | --- |
| Voxel-grid downsample | leaf size 0.05 |
| Normal estimation | k = 10 neighbors, single-threaded |
| Statistical Outlier Removal | mean k = 16, std-mul = 1.0 |
| Radius Outlier Removal | radius 0.1, min neighbors 4 |

## Running

```bash
# needs: libpcl-dev, g++, eigen3, and the SpatialRust Python extension
#        installed into ../../.venv (maturin develop --release)
bench/pcl_comparison/run.sh 200000
```

The script generates the cloud (`gen_cloud.py`), builds the SpatialRust
`bench_ops` example and the PCL `pcl_bench.cpp`, runs both, and prints a table.

## Indicative results

Measured on one machine (PCL 1.14, release builds, 210k points). Throughput
varies by CPU; run it yourself for numbers on your hardware.

| Operation | SpatialRust | PCL | Speedup |
| --- | ---: | ---: | ---: |
| Normal estimation | 0.32 s | 0.48 s | **1.5× faster** |
| Statistical Outlier Removal | 0.28 s | 0.50 s | **1.8× faster** |
| Voxel downsample | 0.043 s | 0.011 s | 0.25× (PCL ~4× faster) |
| Radius Outlier Removal | 1.70 s | 0.32 s | 0.19× (PCL ~5× faster) |

SpatialRust is faster on neighborhood-statistics operations (normals, SOR) and
slower on voxel downsampling (PCL's hashed grid is highly tuned) and radius
filtering — both are active optimization targets. These are honest single-run
numbers, including where PCL wins.

> Note: comparisons use each library's straightforward default API on a CPU,
> single-threaded where that is the default. They are indicative, not a
> rigorously controlled benchmark.
