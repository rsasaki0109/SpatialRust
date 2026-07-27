# Epic 126 SpatialRust 1.2 release-gate receipt

Date: 2026-07-27

## Delivered

- `Streaming12ReleaseGate` with strict conformance, receipt, example,
  migration-policy, memory, spill, cleanup, copy, device-transfer,
  determinism, and file-handle evidence.
- Executable canonical voxel workflow producing measured gate inputs and
  generated Markdown.
- Dedicated Linux/Windows/macOS bounded-streaming conformance matrix plus
  feature-isolation entries.
- Stable bounded-record foundation registry, provisional adapter/workflow
  boundary, migration guide, release receipt, architecture/ROADMAP/CHANGELOG
  integration, and 1.2.0 package versions.

## Verification

- `cargo test -p spatialrust-platform streaming`
- `cargo run -p spatialrust --no-default-features --features platform,pipeline-streaming --example streaming_1_2_release_gate`
- `cargo clippy -p spatialrust-platform --all-targets -- -D warnings`
- `cargo clippy -p spatialrust --no-default-features --features platform,pipeline-streaming --example streaming_1_2_release_gate -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo check --manifest-path crates/spatialrust-py/Cargo.toml`
- `maturin build --release --manifest-path crates/spatialrust-py/Cargo.toml`

## Relevant files

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-platform\src\streaming.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust\examples\streaming_1_2_release_gate.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\.github\workflows\ci.yml`
- `C:\Users\rsasa\Workspace\SpatialRust\docs\STREAMING_MIGRATION.md`
- `C:\Users\rsasa\Workspace\SpatialRust\docs\STREAMING_RELEASE_RECEIPT.md`
