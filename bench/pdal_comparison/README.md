# SpatialRust vs PDAL benchmark

A reproducible comparison between SpatialRust and
[PDAL](https://pdal.io/) on the operations both implement, using the **exact
same** public PCL `table_scene_lms400.pcd` scan (or a synthetic room scene)
with matching parameters. PDAL is comparison tooling only; it is never a
production dependency.

## What it measures

| Operation | PDAL pipeline | Parameters |
| --- | --- | --- |
| Voxel-grid downsample | `filters.voxelcenternearest` | cell = 0.05 |
| Translate XYZ | `filters.transformation` | +1 m on each axis |
| Reprojection | `filters.reprojection` | EPSG:4979 → EPSG:4978 |

PDAL does not expose a point-cloud normal/outlier API matching PCL/SpatialRust,
so those workloads are covered by the PCL and Open3D harnesses. `voxel
downsample`, `translate`, and `reprojection` are the PDAL-facing workloads.

## Running

```bash
# needs: pdal (>= 2.4), Python, and a Rust toolchain
bench/pdal_comparison/run.sh
```

The script downloads the public PCL sample into `target/bench-data/`, builds the
SpatialRust `bench_ops` example, runs PDAL pipelines, and prints a side-by-side
table. Use `--input cloud.pcd` or `--synthetic 200000` to change the input.

## Indicative results

Measured on one local machine (PDAL 2.x, release Rust build, 460,400-point
public PCL `table_scene_lms400.pcd`). Throughput varies by hardware and PDAL
build; run the harness yourself for numbers on your machine.

| Operation | SpatialRust | PDAL | Winner |
| --- | ---: | ---: | :--- |
| (pending dated run) | — | — | — |

These are honest single-run numbers; rerun the harness on your target hardware
before making a portability claim.
