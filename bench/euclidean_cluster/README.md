# Euclidean cluster CPU vs GPU benchmark

Compares [`EuclideanClusterExtractor`](../../crates/spatialrust-segmentation/src/cluster.rs) (CPU) against [`GpuEuclideanClusterExtractor`](../../crates/spatialrust-segmentation/src/cluster_gpu.rs) (wgpu + WGSL label propagation) on the public PCL [`table_scene_lms400.pcd`](https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd) sample.

## Run

```bash
python bench/euclidean_cluster/run.py
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File bench/euclidean_cluster/run.ps1
```

Options:

```bash
python bench/euclidean_cluster/run.py --repeat 5 --warmup 1
python bench/euclidean_cluster/run.py --mvp-leaf 0.05
python bench/euclidean_cluster/run.py --full-cloud
```

Default path (`--mvp-leaf 0.05`): voxel downsample → normals → plane RANSAC → cluster **plane outliers** (matches MVP pipeline input).

## Output

CSV on stdout:

```csv
backend,seconds,cluster_count,point_count
cpu,0.0123,4,182000
gpu,0.0045,4,182000
```

Stderr includes load summary and `speedup (cpu/gpu): Nx`.

## Direct cargo invocation

```bash
python bench/pcl_comparison/fetch_public_cloud.py
cargo run --release -p spatialrust --example bench_euclidean_cluster \
  --features segment-euclidean,segment-euclidean-gpu,segment-ransac-plane,io-pcd,gpu-wgpu,filter-voxel,feature-normal -- \
  target/bench-data/table_scene_lms400.pcd --mvp-leaf 0.05 --repeat 3
```

Results are recorded in `notes/2026-07-03_euclidean_cluster_bench.md`.
