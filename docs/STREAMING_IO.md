# Bounded point-cloud I/O

SpatialRust 1.2 adds opt-in format adapters under the
`spatialrust-io/streaming` feature. They implement the leased contracts from
`spatialrust-records` and leave the existing whole-cloud readers and writers
unchanged.

## Sources

- `PcdChunkSource`: ASCII and interleaved little-endian binary PCD.
- `PlyChunkSource`: ASCII and little-endian binary, vertex-only PLY.
- `LasChunkSource`: sequential LAS and feature-enabled LAZ.
- `CopcChunkSource`: deterministic `(level, x, y, z)` node traversal over local
  files or HTTP range requests, with optional bounds and LOD.

Every source reserves column capacity before allocation. Binary PCD/PLY
payload scratch, COPC compressed/decompressed node storage, and decoded COPC
point vectors are admitted through the same `MemoryTracker`. ASCII point
records use a fixed 16 KiB stack buffer and reject longer records. Header and
hierarchy metadata are format caches rather than point payload and should be
reported separately from tracked chunk memory in execution receipts.

The bounded PCD source deliberately rejects `binary_compressed`: that encoding
is field-major and cannot be emitted seek-free without first transposing its
entire payload. The existing whole-cloud reader remains available. Applications
that accept disk staging should use an explicit `BoundedSpool` and record its
extent.

## Sinks

`PcdChunkSink`, `PlyChunkSink`, and `LasChunkSink` write each lease
synchronously. Their constructors require the exact expected point count;
schema changes, overflow, short final streams, and writes after `finish` fail
closed.

`write_copc_stream` consumes any bounded record source. It computes the COPC
writer's fixed-width spill extent from the declared point count and refuses to
pull the first chunk if `SpoolOptions::limit_bytes` is too small. Cancellation
is shared with the source and polled by both the source iterator and COPC
writer.

`BoundedSpool` is a general seekable `.part` file. It rejects a write before
the configured maximum extent is crossed, removes uncommitted files on drop,
and refuses to overwrite a destination on commit.

## Feature examples

```powershell
cargo run -p spatialrust-io --example bounded_pcd_to_ply `
  --features "io-pcd,io-ply,streaming" -- input.pcd output.ply
```

Holding more than one chunk at once is allowed, but every live lease remains
charged to the shared memory budget. Drop a processed chunk before pulling the
next one when the budget is sized for one chunk.
