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

For local ICP failure analysis, build the additional `pcl_icp_bench` CMake target
and run it through the common SpatialRust/Open3D comparison:

```bash
cmake -S bench/pcl_comparison -B target/pcl-comparison -DCMAKE_BUILD_TYPE=Release
cmake --build target/pcl-comparison --target pcl_icp_bench
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/compare_icp_failures.py \
  --pcl-binary target/pcl-comparison/pcl_icp_bench --output-dir target/three-library-icp
SPATIALRUST_PCL_ICP_BINARY=target/pcl-comparison/pcl_icp_bench \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  bench/pcl_comparison/test_pcl_icp_failures.py -o addopts='' -q
```

The comparator explicitly instantiates PCL's header implementations of ICP and
SVD, linking its common/search dependencies. It consumes identical seeded f32
points through a persistent process. Native timing excludes ASCII input and
process startup. Zero thresholds and a shared update cap are configured, but
PCL can stop on exactly zero transform changes; actual update counts are saved.
This stopping difference and f32/f64 update precision preclude a literal claim
of identical numerical work. JSON hashes the native comparator and runner code;
HTML compares recovery by condition using common full-source SciPy evaluation.

In the 2026-10-10 controlled study, PCL 1.15.0, Open3D 0.19.0 and SpatialRust
all recovered 138/234 synthetic conditions, with no disagreement on any paired
condition. Nearby outliers under broad distance gates defeated all three.
This is shared failure evidence, not a real-sensor accuracy or speed ranking.
PCL executed 15–100 updates in the run, while SpatialRust used all 100.

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

Measured on one Linux machine (PCL 1.14.x via libpcl-dev, g++ 16, release
Rust build, 460,400-point public PCL `table_scene_lms400.pcd`, single-threaded
where that is the default). Throughput varies by CPU and PCL build; run it
yourself for numbers on your hardware. Dated receipt: `receipt-2026-08-07.json`.

| Operation | SpatialRust | PCL | Speedup |
| --- | ---: | ---: | :--- |
| Voxel downsample | 0.0104 s | 0.0177 s | **1.70× faster** |
| Normal estimation | 0.2171 s | 0.9893 s | **4.56× faster** |
| Statistical Outlier Removal | 0.2297 s | 1.1272 s | **4.91× faster** |
| Radius Outlier Removal | 0.1088 s | 0.7200 s | **6.62× faster** |

SpatialRust is faster on neighborhood-statistics and density operations (radius
outlier removal uses an early-exit density test; normals and SOR win too; voxel
downsampling uses a specialized XYZ centroid path with compact `u32` voxel keys
for the common min-origin case). These are honest single-run numbers; rerun the
harness on your target hardware before making a portability claim.

> Note: comparisons use each library's straightforward default API on a CPU,
> single-threaded where that is the default. They are indicative, not a
> rigorously controlled benchmark.
