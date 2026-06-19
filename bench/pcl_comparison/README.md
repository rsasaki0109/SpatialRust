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

On Windows, the repository also includes a CMake/vcpkg runner. It expects MSYS2
UCRT64 tools under `C:\msys64` and PCL installed by vcpkg under `C:\vcpkg`:

```powershell
powershell -ExecutionPolicy Bypass -File bench\pcl_comparison\run.ps1 -Points 200000
```

The script generates the cloud (`gen_cloud.py`), builds the SpatialRust
`bench_ops` example and the PCL `pcl_bench.cpp`, runs both, and prints a table.

## Indicative results

Measured on one Windows machine (PCL 1.15.1 via vcpkg, MSYS2 g++ 16.1.0,
release builds, 210k points). Throughput varies by CPU; run it yourself for
numbers on your hardware.

| Operation | SpatialRust | PCL | Speedup |
| --- | ---: | ---: | ---: |
| Radius Outlier Removal | 0.0597 s | 0.9998 s | **16.75× faster** |
| Statistical Outlier Removal | 0.1786 s | 1.1352 s | **6.36× faster** |
| Normal estimation | 0.1514 s | 1.0885 s | **7.19× faster** |
| Voxel downsample | 0.0068 s | 0.0099 s | **1.46× faster** |

SpatialRust is faster on neighborhood-statistics and density operations (radius
outlier removal uses an early-exit density test; normals and SOR win too; voxel
downsampling uses a specialized XYZ centroid path with compact `u32` voxel keys
for the common min-origin case). These are honest single-run numbers; rerun the
harness on your target hardware before making a portability claim.

> Note: comparisons use each library's straightforward default API on a CPU,
> single-threaded where that is the default. They are indicative, not a
> rigorously controlled benchmark.
