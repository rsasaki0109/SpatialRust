# Epic 121 bounded-streaming contract receipt

Date: 2026-07-27

## Outcome

SpatialRust 1.2 now has a runtime-independent contract for bounded point-cloud
streaming in `spatialrust-records`. The stable in-memory `spatialrust-core`
surface is unchanged and `SpatialTensor` remains provisional.

## Delivered

- `StreamOptions` with a positive chunk size, hard `MemoryBudget`, bounded
  prefetch count, and deterministic ordering policy.
- Concurrent `MemoryTracker` reservations that fail before the limit is
  exceeded and release automatically on drop.
- Clone-visible cooperative `CancellationToken`.
- Versioned `spatialrust.streaming.receipt` v1 accounting for points, chunks,
  IO bytes, named phases, tracked peak memory, spill bytes, and explicit
  host/device transfers.
- Strict optional JSON import/export behind `receipt-json`; unknown fields and
  unsupported versions fail closed.
- Canonical 1M/10M/100M point workloads at 16K/64K/256K chunk sizes.
- Runnable `streaming_receipt` synthetic 1M-point baseline.

## Validation

```text
cargo test -p spatialrust-records --all-features
cargo run -p spatialrust-records --example streaming_receipt --features receipt-json
cargo test -p spatialrust-records --no-default-features
cargo clippy -p spatialrust-records --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

The receipt intentionally tracks bytes owned through `MemoryTracker`; operating
system RSS is observational and is not used as the deterministic hard limit.
