# Epic 122 bounded record stream receipt

Date: 2026-07-27

## Outcome

SpatialRust now carries each emitted record in a lease whose lifetime is also
the lifetime of its exact tracked-memory reservation. Existing
`SpatialRecordSource` and `SpatialRecordSink` signatures remain unchanged.

## Delivered

- `ChunkIdentity` with deterministic sequence and global point offset.
- `SpatialRecordChunk` with optional finite XYZ bounds and tracked
  column-capacity accounting.
- Additive `BoundedSpatialRecordSource` and `BoundedSpatialRecordSink` traits.
- Legacy source/sink adapters that reserve the declared maximum before pulling.
- Single-worker `PrefetchRecordSource` with ordered delivery, bounded
  `sync_channel` backpressure, cancellation, and preflight memory admission.
- `RecyclingMemoryChunkSource`, which returns column buffer sets to a
  source-owned pool on chunk drop.
- Safe `PointCloud::into_parts`, the ownership-preserving inverse of
  `try_from_parts`, plus mutable buffer-set iteration for allocation-free
  clearing; no stream policy or dependency was added to core.

## Validation

```text
cargo test -p spatialrust-core
cargo test -p spatialrust-records --all-features
cargo test -p spatialrust-records --no-default-features
cargo run -p spatialrust-records --example bounded_record_stream
cargo clippy -p spatialrust-records --all-targets --all-features -- -D warnings
cargo test --workspace
cargo fmt --all -- --check
```

The steady-state recycling test emits three chunks with one buffer-set
allocation. Prefetch admission accounts for the queue plus one producer-held
chunk before the worker starts.
