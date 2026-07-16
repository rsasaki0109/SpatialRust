# HTTP COPC connection reuse on public Autzen data

Date: 2026-07-16

## Change

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-io\src\copc\http.rs`
now stores one `ureq::Agent` inside `HttpByteSource`. Cloned sources and
parallel range workers clone the agent, which shares its connection pool and
TLS state. Previous code called the `ureq::get`/`head` convenience functions
for every range; those functions construct a new agent and discard its pool.

The public API, query bounds, point decoding, eight-range concurrency default,
and CPU ownership semantics are unchanged.

## Correctness

- Existing HTTP range unit tests pass.
- Local HTTP COPC query output remains identical to the local-file query.
- The public Autzen central-XY query returned exactly 889,058 points before and
  after the change.

## Measurement

Host: Windows x86_64, Rust 1.97.0, release build, default network and thread
settings. Remote source:
`https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz`.

Command:

```powershell
cargo test -p spatialrust --features mvp,mvp-http --test mvp_public_copc --release -- --ignored --nocapture mvp_http_autzen_copc_bounds_smoke
```

| Implementation | Runs (seconds) | Reported value |
| --- | --- | ---: |
| New agent per range | 95.99 | 95.99 |
| Shared agent/pool | 59.69, 37.57, 61.83 | median 59.69 |

The same 889,058-point output improved by 1.61× against the single pre-change
observation, a 37.8% elapsed-time reduction. Network conditions are externally
variable, so this is a dated workload-specific observation, not a portable
latency guarantee.

## Validation

```powershell
cargo test -p spatialrust-io --features io-copc-http copc::http::tests
cargo test -p spatialrust-io --features io-copc-http --test copc_http_local
cargo clippy -p spatialrust-io --features io-copc-http --all-targets -- -D warnings
```
