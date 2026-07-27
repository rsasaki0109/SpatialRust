# Chunk-safe streaming operations

The `spatialrust-pipeline/pipeline-streaming` feature composes leased record
sources without expanding `spatialrust-core`.

## Chunk-local map

`ChunkMapSource::crop` applies inclusive XYZ bounds and emits only non-empty
chunks. `ChunkMapSource::transform` applies an affine transform to positions
and the linear transform plus normalization to complete normal triplets.
Other numeric fields are copied unchanged.

Both operations reserve output column capacity on the upstream source's
`MemoryTracker` before allocation. The input lease stays live during the map,
so a budget sized for only one chunk fails instead of temporarily holding two
unaccounted chunks. Output sequence and point offsets are rebuilt contiguously
after crop.

## Global reduction

`reduce_positions` makes one ordered pass and returns total points, finite
points, finite XYZ bounds, and a compensated-sum centroid. Non-finite positions
contribute to `point_count` but not bounds or centroid.

## Deterministic global voxel centroid

`StreamingVoxelSource::try_build` uses two bounded phases:

1. Convert each finite point to a fixed-width record containing voxel key,
   source point offset, and every numeric field as `f64`; sort bounded runs by
   `(key, offset)` and write them to one `BoundedSpool`.
2. Merge at most `max_runs` cursors, aggregate each voxel in original source
   order, and emit leased chunks in lexicographic voxel-key order.

The source validates contiguous input identities. Therefore changing source
chunk size or in-memory run size does not change aggregation order or output.
All numeric attributes use centroid/mean aggregation; integer outputs are
rounded and clamped to their declared dtype. Points with a non-finite position
are skipped.

`run_points` bounds sort memory, `max_runs` bounds file handles and merge heap,
and `SpoolOptions` bounds the total single-file spill extent. Run buffers,
merge records, accumulators, and emitted columns share the upstream memory
tracker. `spool_bytes()`, `run_count()`, and the tracker snapshot provide
receipt inputs.

## End-to-end workflows

`StreamingPipeline` type-erases bounded sources while retaining their shared
memory tracker and cancellation token. Its `crop`, `transform`, and `voxel`
builders feed either a Rust iterator or `run_to_sink`; input/output chunks,
tracked bytes, peak memory, voxel phase time, and spill extent are written to
one versioned `StreamingReceipt`.

The `spatialrust-stream` binary accepts local PCD/PLY/LAS/LAZ/COPC or an
HTTP(S) COPC URL, writes LAS/LAZ without knowing the filtered point count in
advance, and emits receipt JSON:

```powershell
cargo run -p spatialrust --features streaming-cli --bin spatialrust-stream -- `
  input.copc.laz output.laz --chunk-points 65536 --memory-budget 268435456 `
  --crop 0 0 -10 100 100 20 --voxel 0.1 --receipt receipt.json
```

The Python extension uses the same Rust iterator:

```python
stream = spatialrust.open_point_cloud_stream(
    "input.pcd", chunk_points=65_536, memory_budget_bytes=268_435_456,
)
for chunk in stream:
    consume(chunk)
print(stream.receipt_json())
```

`stream.cancel()` requests cooperative cancellation. Each yielded Python
`PointCloud` owns a copy, so retaining Python outputs is caller-managed and is
not included in the native pipeline budget.

```powershell
cargo run -p spatialrust-pipeline --example bounded_voxel `
  --no-default-features --features pipeline-streaming
```
