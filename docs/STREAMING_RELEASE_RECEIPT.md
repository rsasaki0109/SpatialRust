# SpatialRust 1.2 bounded-streaming release receipt

Decision: **allowed**

The canonical executable receipt uses four XYZ points, two-point source and
sort chunks, deterministic voxel aggregation, a 1 MiB memory budget, and a
1 MiB spool limit. It verifies the accounting path on every supported host;
the canonical 1M/10M/100M workload manifest remains available for production
benchmarking without making machine-independent latency claims.

| Measurement | Observed | Ceiling |
| --- | ---: | ---: |
| Configured memory budget | 1,048,576 bytes | 268,435,456 bytes |
| Peak tracked memory | 552 bytes | 268,435,456 bytes and configured budget |
| Configured spool limit | 1,048,576 bytes | 2,147,483,648 bytes |
| Fixed-width spill | 240 bytes | 2,147,483,648 bytes and configured limit |
| Live tracked bytes after finish | 0 | 0 |
| Hidden host-copy bytes | 0 | 0 |
| Host-to-device bytes | 0 | 0 |
| Device-to-host bytes | 0 | 0 |
| Determinism mismatches | 0 | 0 |
| Maximum open spill/run files | 4 | 1,025 |

Required receipt families:

- [x] Epic 121 memory/cancellation/receipt contract
- [x] Epic 122 leased/recycled bounded record streams
- [x] Epic 123 local/HTTP format streams and spool contract
- [x] Epic 124 chunk-safe maps/reductions/deterministic voxel
- [x] Epic 125 Rust/CLI/Python end-to-end workflow

Release conformance runs the records, all-format IO, pipeline, real CLI E2E,
platform gate, and receipt example on Linux, Windows, and macOS. Python 3.8 and
3.12 extension tests plus x86_64/aarch64 wheel builds remain independent CI
gates.

Reproduce the machine-checked decision:

```powershell
cargo test -p spatialrust-platform streaming
cargo run -p spatialrust --no-default-features `
  --features platform,pipeline-streaming `
  --example streaming_1_2_release_gate
```
