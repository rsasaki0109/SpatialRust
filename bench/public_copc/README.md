# Public COPC validation (Epic 60)

Reproducible end-to-end check that SpatialRust can:

1. Load the public PCL [`table_scene_lms400.pcd`](https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd) sample (460,400 points).
2. Write it as COPC, apply **bounds** and **resolution** queries, and run the MVP pipeline on the result.
3. Run the Python clean → cluster → global registration → ICP pipeline with
   count, convergence, transform-error, and dataset-hash assertions.
4. Optionally stream a non-empty bounded region from a **public HTTP COPC**
   (PDAL Autzen Stadium on S3).

## Run

```bash
python bench/public_copc/run.py
```

Windows:

```powershell
python bench/public_copc/run.py
.venv/Scripts/python.exe bench/public_copc/validate_python.py
```

HTTP smoke (requires network):

```bash
python bench/public_copc/run.py --http
```

## What it executes

| Step | Command |
| --- | --- |
| Fetch PCD | `python bench/pcl_comparison/fetch_public_cloud.py` |
| Local COPC + MVP | `cargo test -p spatialrust --features mvp --test mvp_public_copc --release` |
| Python real-data E2E | `.venv/Scripts/python.exe bench/public_copc/validate_python.py` |
| HTTP Autzen (optional) | `cargo test ... --features mvp,mvp-http -- --ignored` |

Override the input path:

```bash
export SPATIALRUST_PUBLIC_PCD=/path/to/table_scene_lms400.pcd
python bench/public_copc/run.py
```

## Notes

Original COPC results are recorded in
`notes/2026-07-03_public_copc_mvp.md`; the strengthened Rust/Python/HTTP
validation is recorded in `notes/2026-07-16_real_data_validation.md`.
