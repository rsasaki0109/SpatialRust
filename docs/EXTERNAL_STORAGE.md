# External storage for point-cloud IO

SpatialRust keeps data placement explicit. The point-cloud core does not know
about disks, and IO readers/writers do not silently copy a dataset to another
device. Applications can provide separate input and output roots at the CLI or
Python boundary.

## Rust and CLI

Relative logical paths are resolved under the matching root. Absolute paths
remain explicit and bypass the root; relative paths containing `..` are
rejected.

```bash
cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- \
  --input-root /media/sasaki/aiueo/datasets \
  --output-root /media/sasaki/aiueo/spatialrust-results \
  --manifest runs/scan.json \
  boreas/scan.las runs/scan.ply
```

The bounded streaming CLI accepts the same root flags and writes the existing
workflow receipt separately from the file manifest:

```bash
cargo run -p spatialrust --features streaming-cli --bin spatialrust-stream -- \
  --input-root /media/sasaki/aiueo/datasets \
  --output-root /media/sasaki/aiueo/spatialrust-results \
  --receipt runs/stream-receipt.json \
  --manifest runs/stream-manifest.json \
  boreas/scan.pcd runs/scan.laz
```

The `spatialrust-io/io-manifest` feature adds `DatasetManifest`,
`FileReceipt`, and SHA-256 hashing. A manifest has version `1` and ordered
`entries`; local input/output entries contain `size_bytes` and `sha256`. An
HTTP(S) COPC input is recorded as an `input` URI entry without a local size or
checksum because the CLI does not materialize it before streaming.

## Python

The binding exposes the same explicit roots without changing the default
behavior:

```python
import spatialrust as sr

cloud = sr.read("boreas/scan.las", input_root="/media/sasaki/aiueo/datasets")
sr.write(
    "runs/labeled.ply",
    cloud,
    output_root="/media/sasaki/aiueo/spatialrust-results",
    manifest_path="runs/labeled.json",
)

result = sr.run_pipeline_files(
    "boreas/scan.las",
    "runs/labeled.ply",
    input_root="/media/sasaki/aiueo/datasets",
    output_root="/media/sasaki/aiueo/spatialrust-results",
    manifest_path="runs/mvp.json",
)
```

`open_point_cloud_stream(..., input_root=...)` applies the same input
resolution to bounded streaming. The output root only controls the destination
path supplied by the caller; no intermediate dataset copy is created.
