# Public RGB-D multi-adapter conformance

Date: 2026-10-19. Operational slice 146B. This protocol is fixed before first
dataset execution. Results are pending; initialization alone is not execution.

The [fixed plan](../benchmarks/tum-fr1-xyz-gpu-conformance-plan.json) selects
depth indices 0, 399 and 797 from the already observed public TUM RGB-D
Freiburg 1 XYZ development sequence. All three original associated RGB images
retain their 640x480 geometry. Every valid 0.1..5m depth pixel is projected in
row order with the publisher optical formula in float64 and cast once to
float32. This preparation is a CPU operation; GPU depth projection is not claimed.
No reference trajectory is read. Dataset pixels and derived point payloads
remain outside Git.

The archive SHA-256 is
`a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98`
(448204271 bytes). The prepared source manifest is
`e7e98da77bed799ac281168bd6d81d737692cf7aa25b1c8443fb0485b0e87434`.
Input preparation checks that manifest, original image checksums, calibration,
timestamp associations, dimensions and depth counts. Generated manifests bind
every original and derived payload. The Rust runner checks its caller-supplied
manifest hash and each payload hash/size before consuming owned bytes.

Source: [official archive](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz),
[publisher format/calibration](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
and [license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license).
Dataset attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers,
A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012. Dataset
license is CC BY 4.0. Registered depth already uses RGB optical coordinates;
the published default intrinsics are an approximation, not newly measured
sensor/clock calibration. No extra distortion or depth/extrinsic correction is applied.

## Fixed operations and acceptance

| Operation | Independent reference | Gate |
| --- | --- | --- |
| RGB to gray | Scalar integer BT.601 | Every pixel exact |
| Replicate box blur 11x11 and 259x3 | CPU two-dimensional padded integer integral | Every pixel exact |
| 256-bin histogram of 11x11 blur | Scalar counts | Every bin exact |
| 0.03125m voxel centroids | Ordered CPU map, float64 sums | Complete keys/counts/membership exact; every coordinate error <=1e-5m |
| 0.03125m radius counts, 64 fixed offset queries | Brute-force float64 squared distances | Every count exact; preserve count-only overflow |

Both high/low power preferences must execute every frame. The hardware
coverage gate requires distinct integrated and discrete device classes;
preferences returning one device do not prove two-device coverage. Any Linux
software adapter is labeled separately and does not count as hardware coverage.
Do not remove failed frames or revise tolerance/frame selection after execution.
The CPU builds the KD-tree index; resident radius traversal/counts run on GPU.

All image consumers submit before the first explicit host readback. Resident
image receipts must record the single physical RGBA8 source upload and zero
mid-chain host download. Three explicit texture downloads plus one histogram
download total 3687424 bytes per frame. Point transfers separately report
position upload, the existing four-byte compaction synchronization, centroid
download, optional metadata audit, KD-tree/query uploads and radius readback.
Counts follow public buffer layouts/receipts; uniforms and driver staging are
excluded. Pool counters must stay within their declared retained-byte budgets;
they do not measure live allocations, peak memory or driver/staging overhead.

## Reproduction

Use the source preparation in `C:/Users/rsasa/Workspace/SpatialRust/scripts/prepare_public_gpu_rgbd.py`
with a verified prepared directory, its manifest SHA-256, and a new external
output directory. The observed input/output directories will be recorded with
results. The Rust runner takes the resulting directory/hash and `high` or `low`:

```text
cargo run -p spatialrust-gpu --release --features gpu-image \
  --example gpu_public_rgbd_conformance -- INPUT_DIRECTORY MANIFEST_SHA256 high
```

The preparation uses existing NumPy/Pillow dependencies. Example-only JSON and
hash dependencies are dev dependencies, already present in the workspace lock.
No core, runtime API, shader, default selection or implicit device copy changes.
Independent CPU reference and source/PNG/geometry/IO rejection tests run before
the first dataset execution. Source, executable and protocol hashes are frozen
with the complete results. A failed process/case is retained.

This is correctness and transfer conformance on three development frames,
not a timing ranking, full algorithm/scene portability certificate, calibrated
canonical Autoware acceptance, accurate motion, or completion of the operational
80% target. The failed motion results in
[the image-motion study](TUM_RGBD_MOTION_VALIDATION.md) remain unchanged.
