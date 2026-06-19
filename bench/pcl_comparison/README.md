# SpatialRust vs PCL benchmark

A reproducible, apples-to-apples timing comparison between SpatialRust and
[PCL](https://pointclouds.org/) on the operations both libraries implement. Both
process the **exact same** public PCL `table_scene_lms400.pcd` scan by default,
with matching parameters.

## What it measures

| Operation | Parameters |
| --- | --- |
| Voxel-grid downsample | leaf size 0.05 |
| Normal estimation | k = 10 neighbors, single-threaded |
| Statistical Outlier Removal | mean k = 16, std-mul = 1.0 |
| Radius Outlier Removal | radius 0.1, min neighbors 4 |

## Running

```bash
# needs: libpcl-dev, g++, eigen3, Python, and a Rust toolchain
bench/pcl_comparison/run.sh
```

On Windows, the repository also includes a CMake/vcpkg runner. It expects MSYS2
UCRT64 tools under `C:\msys64` and PCL installed by vcpkg under `C:\vcpkg`:

```powershell
powershell -ExecutionPolicy Bypass -File bench\pcl_comparison\run.ps1
```

The script downloads the public PCL sample into `target/bench-data/`, builds the
SpatialRust `bench_ops` example and the PCL `pcl_bench.cpp`, runs both, and
prints a table. Use `--input cloud.pcd` on Unix or `-InputPcd cloud.pcd` on
Windows to benchmark another PCD. Use `--synthetic 200000` on Unix or
`-SyntheticPoints 200000` on Windows to run the deterministic synthetic scene;
that fallback also needs NumPy and the SpatialRust Python extension.

## Indicative results

Measured on one Windows machine (PCL 1.15.1 via vcpkg, MSYS2 g++ 16.1.0,
release builds, 460,400-point public PCL `table_scene_lms400.pcd`). Throughput
varies by CPU; run it yourself for numbers on your hardware.

| Operation | SpatialRust | PCL | Speedup |
| --- | ---: | ---: | ---: |
| Radius Outlier Removal | 0.0899 s | 1.8784 s | **20.89× faster** |
| Statistical Outlier Removal | 0.1664 s | 2.0933 s | **12.58× faster** |
| Normal estimation | 0.1461 s | 1.9750 s | **13.52× faster** |
| Voxel downsample | 0.0104 s | 0.0181 s | **1.74× faster** |

SpatialRust is faster on neighborhood-statistics and density operations (radius
outlier removal uses an early-exit density test; normals and SOR win too; voxel
downsampling uses a specialized XYZ centroid path with compact `u32` voxel keys
for the common min-origin case). These are honest single-run numbers; rerun the
harness on your target hardware before making a portability claim.

> Note: comparisons use each library's straightforward default API on a CPU,
> single-threaded where that is the default. They are indicative, not a
> rigorously controlled benchmark.
