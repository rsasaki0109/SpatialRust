# Epic 136 record-semantic viewer bridge — 2026-08-03

## Scope

`spatialrust-viewer` now exposes `spatial_record_entity_visual`, which accepts
`&[SpatialRecordEntity]` directly. The adapter keeps the wrapper slice as the
receipt source and materializes only renderer-facing XYZ and best-label
confidence columns. `AdapterReceipt::generated_bytes` reports the exact
four-column `f32` payload generated for the visual layer; provenance, frame,
timestamp, and embeddings are not cloned into a second semantic collection.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo fmt --all -- --check
cargo test -p spatialrust-viewer --features semantic
cargo clippy -p spatialrust-viewer --features semantic --all-targets -- -D warnings
```

Tests cover source-slice identity, source/output counts, confidence scalar
preservation, and the existing semantic centroid filtering behavior.
