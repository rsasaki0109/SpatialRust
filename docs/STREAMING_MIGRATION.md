# Migrating point-cloud workflows to bounded streaming in 1.2

SpatialRust 1.2 adds an opt-in out-of-core path. Existing `PointCloud`,
`SpatialTensor`, `SpatialRecordSource`, format readers/writers, and
`MvpPipeline` signatures are unchanged. Move only workflows that need bounded
resident memory; small in-memory workflows can remain as they are.

## Select the bounded features

- Rust composition: `pipeline-streaming`.
- All-format CLI: `streaming-cli`.
- Direct format adapters: enable `spatialrust-io/streaming` plus only the
  required `io-*` formats.
- Receipt JSON: `records-receipt-json`.

HTTP COPC is isolated behind `io-copc-http`. Default Python wheels intentionally
exclude its TLS stack; use `spatialrust-stream` for remote COPC and pass a local
file to Python.

## Replace whole-cloud ownership at the boundary

Open `PcdChunkSource`, `PlyChunkSource`, `LasChunkSource`, or
`CopcChunkSource`, then pass it to `StreamingPipeline`. Every
`SpatialRecordChunk` owns a memory reservation that lasts exactly as long as
the record borrow. Consume or drop a chunk before pulling another unless the
configured budget intentionally allows both.

Use `LegacyBoundedSource` and `LegacyBoundedSink` when an existing synchronous
record implementation already has a trustworthy maximum chunk size. The old
traits remain available.

Prefetch admission accounts for `prefetch_chunks + 2` maximum-sized leases:
the bounded queue, one producer lease blocked on a full queue, and one consumer
lease. Budgets sized to the earlier queue-plus-consumer intuition are rejected
before the worker starts.

## Size memory and spool limits explicitly

`MemoryBudget` is a hard ceiling for tracked native buffers, not a target.
Chunk maps briefly hold input and output leases together. Deterministic voxel
aggregation additionally reserves a bounded sort run or merge state and writes
fixed-width records to a `BoundedSpool`.

Set all four controls for production voxel work:

- `chunk_points` for leased source/output columns;
- `memory_budget_bytes` for the shared tracker;
- `run_points` and `max_runs` for sort and merge state;
- `spool_limit_bytes` for temporary disk extent.

Admission fails before allocation or spool growth exceeds a declared limit.
Cancellation is cooperative at chunk/record boundaries and reservations are
released on drop.

## Preserve deterministic behavior

Chunk identity is `(sequence, point_offset)` in source order. Crop and
transform rebuild contiguous output identity. Global voxel aggregation sorts
by `(voxel key, source point offset)`, so changing source chunk or run size
does not change output order or floating-point accumulation order.

PCD `binary_compressed` input is deliberately rejected by the bounded adapter
because its encoded layout requires whole-field decompression. Use binary or
ASCII PCD, or convert once through another bounded format.

## Keep transfers and Python retention visible

The 1.2 pipeline is CPU-only and performs no implicit host/device transfer.
Future GPU stages must record explicit uploads/readbacks in the receipt.

Python `PointCloudStream` uses the same Rust iterator, but each yielded
`PointCloud` owns a Python-visible copy. Retaining many yielded clouds is
caller-managed memory outside the native stream budget.

## Reproduce the release gate

```powershell
cargo test -p spatialrust-platform streaming
cargo run -p spatialrust --no-default-features `
  --features platform,pipeline-streaming `
  --example streaming_1_2_release_gate
```

`Streaming12ReleaseGate` denies missing, skipped, duplicated, over-budget, or
unacknowledged migration evidence and reports all reasons together.
