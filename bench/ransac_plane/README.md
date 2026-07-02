# RANSAC plane CPU vs GPU benchmark

Compares [`RansacPlaneSegmenter`](../../crates/spatialrust-segmentation/src/plane.rs) (CPU) against [`GpuRansacPlaneSegmenter`](../../crates/spatialrust-segmentation/src/plane_gpu.rs) (wgpu + WGSL) on the public PCL [`table_scene_lms400.pcd`](https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd) sample (460,400 points).

## Run

```bash
python bench/ransac_plane/run.py
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File bench/ransac_plane/run.ps1
```

Options:

```bash
python bench/ransac_plane/run.py --iterations 1000 --repeat 5 --warmup 1
python bench/ransac_plane/run.py --mvp-leaf 0.05
```

The `--mvp-leaf` flag runs voxel downsampling + normal estimation before RANSAC,
matching the MVP pipeline point count (~2k on `table_scene_lms400` at leaf=0.05).

## Output

CSV on stdout:

```csv
backend,seconds,inlier_count,iterations
cpu,0.1234,123456,1000
gpu,0.0456,123456,1000
```

Stderr includes load summary and `speedup (cpu/gpu): Nx`.

## Direct cargo invocation

```bash
python bench/pcl_comparison/fetch_public_cloud.py
cargo run --release -p spatialrust --example bench_ransac_plane \
  --features segment-ransac-plane,segment-ransac-plane-gpu,io-pcd,gpu-wgpu -- \
  target/bench-data/table_scene_lms400.pcd --repeat 3
```

Results are recorded in `notes/2026-07-03_ransac_plane_bench.md`.
