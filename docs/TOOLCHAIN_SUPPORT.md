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

The root library lockfile is intentionally untracked in this repository. Generating
with Cargo 1.75 avoids a newer Cargo lockfile format and checks fresh dependency
resolution; subsequent checks use `--locked` to freeze that run's graph.

The separately distributed Python extension tracks
`/workspace/SpatialRust-python-delivery/crates/spatialrust-py/Cargo.lock`.
Previously the blanket ignore rule omitted it from fresh Git checkouts, so an
actual dependency-resolving `cargo metadata --locked` failed before wheel builds.
The frozen graph now resolves offline in a fresh source tree (342 packages), and
the generated source distribution contains exactly the tracked lockfile bytes.
This graph targets current stable, independently of the Rust 1.75 library gate.

Python wheel CI also runs on relevant main-branch changes and builds and tests
installed default wheels on Windows and macOS. These jobs gate tag publication;
their configuration is not evidence of remote runtime success until CI completes.

The same Linux x86_64 `cp38-abi3` default wheel was installed into an isolated
CPython 3.8.20 environment with NumPy 1.24.4, PyArrow 17.0.0 and pytest 8.3.5.
All 491 current applicable tests pass with exactly three expected ONNX skips;
installed native/stub/marker bytes and conversion/ICP/support/PCD checks match
the wheel. This verifies the advertised Python floor and NumPy 1.x compatibility
on this Linux host. Native runtime CI now includes Ubuntu/Python 3.8 alongside
Windows/macOS Python 3.12. Python 3.8 is end-of-life; support here describes
compatibility, not upstream maintenance. Local receipts are under
`/workspace/SpatialRust-python-delivery/target/python38-wheel-validation/`.

The identical wheel also passes 491 tests with the same three ONNX skips on
CPython 3.14.7, NumPy 2.5.3 and PyArrow 26.0.0, with archive/native/type/runtime
verification, under `target/python314-wheel-validation/`. These two endpoints
plus the existing Python 3.12 checks are specific verified configurations;
intermediate versions and free-threaded interpreters are not separately verified.
Ubuntu/Python 3.14 is included in native runtime CI. Superseded branch CI runs
are canceled by ref-based concurrency; tag publication runs are preserved.

## Current stable toolchain

Full CPU vision and Python/native integration are locally tested using Rust 1.99,
Linux x86_64 and CPython 3.12. Optional parallel vision uses Rayon 1.12, which
requires Rust 1.80. Some benchmark/dev dependencies require Rust 1.85 and current
optional dependency resolutions can require newer versions. Their full minimum
toolchain is not established by the base gate. Use current stable for full builds.

Do not interpret a library compilation as a runtime GPU, browser, macOS, Windows,
ARM or Python 3.8 conformance result. Those runtimes still require separate checks.
