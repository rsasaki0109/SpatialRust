# Verified Rust toolchain surfaces

The workspace package baseline is Rust 1.75. This applies to the minimal data
model and geometry surface below; it is not a promise that every optional feature,
GPU/backend, Python extension, benchmark or dev dependency builds on Rust 1.75.

## Rust 1.75.0 verified locally

```bash
cargo +1.75.0 generate-lockfile
cargo +1.75.0 check --locked -p spatialrust -p spatialrust-core -p spatialrust-vision \
  --features spatialrust-vision/geometry
cargo +1.75.0 test --locked -p spatialrust-core -p spatialrust-math \
  --all-features --lib
```

The library check covers the default meta crate, math, core, image, camera and vision geometry. All 36
core/math library tests pass, including serde/bytemuck and tensor-aoso feature
compilation. The same generate-lockfile/check sequence passes in an extracted
source tree with no pre-existing lockfile. CI now repeats these commands with
Rust 1.75.0. Remote CI success is not yet verified locally.

`thiserror` is pinned to `=2.0.17` because the former unconstrained 2.x resolution
selected 2.0.21, which requires Rust 1.77. That produced a confirmed build failure
on Rust 1.75 even for core. The compatible release declares Rust 1.61 and passes
the actual Rust 1.75 checks. Future upgrades require repeating the MSRV gate.

Library lockfiles are intentionally untracked in this repository. Generating
with Cargo 1.75 avoids a newer Cargo lockfile format and checks fresh dependency
resolution; subsequent checks use `--locked` to freeze that run's graph.

## Current stable toolchain

Full CPU vision and Python/native integration are locally tested using Rust 1.99,
Linux x86_64 and CPython 3.12. Optional parallel vision uses Rayon 1.12, which
requires Rust 1.80. Some benchmark/dev dependencies require Rust 1.85 and current
optional dependency resolutions can require newer versions. Their full minimum
toolchain is not established by the base gate. Use current stable for full builds.

Do not interpret a library compilation as a runtime GPU, browser, macOS, Windows,
ARM or Python 3.8 conformance result. Those runtimes still require separate checks.
