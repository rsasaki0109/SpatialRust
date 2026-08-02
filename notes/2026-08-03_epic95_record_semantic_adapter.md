# Epic 95 record semantic adapter — 2026-08-03

## Scope

`spatialrust-semantic` now exposes `SpatialRecordEntity`, a narrow adapter from
one versioned `SpatialRecord` to the existing semantic search payload:

- derives a deterministic entity id from `RecordProvenance`;
- validates open-vocabulary label text/confidence;
- computes a finite f64-accumulated XYZ centroid from `HasPositions3`;
- retains the record's provenance, frame, and timestamp alongside the
  searchable `SemanticEntity`.

An embedding is explicitly supplied by the caller. The adapter does not depend
on ONNX, CUDA, or any model runtime; a later model-specific adapter can produce
the `Embedding` under its own feature boundary.

## Validation

Commands run from `/home/sasaki/workspace/SpatialRust`:

```text
cargo fmt --all
cargo test -p spatialrust-semantic
```

Tests cover deterministic lineage-derived ids, centroid calculation, and
fail-closed label confidence validation.
