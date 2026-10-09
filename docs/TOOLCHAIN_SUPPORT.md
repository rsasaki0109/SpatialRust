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

Earlier complete-suite checkpoint (2026-10-09, main `def46e3`): the existing
identical Linux abi3 wheel passes 537 tests with exactly three expected ONNX
skips in both isolated CPython 3.8/NumPy 1.24 and CPython 3.14/NumPy 2.5
environments. The ONNX-enabled CPython 3.12 installation passes all 540 tests.
The additional research scripts do not require Open3D to run these default-wheel
tests. These counts supersede the earlier default-wheel suite counts below;
the source-archive rebuild remains a separately verified historical checkpoint.

At 2026-10-09 22:48 UTC, GitHub API job metadata for main `def46e3` shows all
seven Python-wheel jobs and all 30 CI jobs queued without an assigned runner.
The configured labels include `ubuntu-latest`, `windows-latest` and `macos-14`;
the workflows do not require self-hosted runners. No execution failure is
available to diagnose, and queue metadata alone does not establish an account,
billing or GitHub-service cause. Successful local checks cannot substitute for
these platform runtime gates. Main CI now uses the same workflow/ref concurrency
policy as wheel CI to cancel superseded branch runs while preserving tags.
This limits future obsolete work; it does not assign runners or retroactively
cancel old runs that had no concurrency group.

Post-holdout complete-suite checkpoint (2026-10-10, client timezone Asia/Tokyo):
all 561 tests pass with the ONNX-enabled CPython 3.12 installation. The existing
identical default abi3 wheel passes 558 tests with exactly three expected ONNX
skips in both isolated CPython 3.8/NumPy 1.24 and CPython 3.14/NumPy 2.5
installations. No native code changes or wheel rebuild were needed for the new
plan, summary and display tools. Their default-wheel tests do not require Open3D
or Matplotlib; actual holdout fitting and optional plots use the comparison
virtual environment. Remote Windows/macOS runtime verification remains open.

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
verification, under `/workspace/SpatialRust-python-delivery/target/python314-wheel-validation/`. These two endpoints
plus the existing Python 3.12 checks are specific verified configurations;
intermediate versions and free-threaded interpreters are not separately verified.
Ubuntu/Python 3.14 is included in native runtime CI. Superseded branch CI runs
are canceled by ref-based concurrency; tag publication runs are preserved.

The source archive containing the frozen lockfile also rebuilds successfully
under CPython 3.14 with the actual PEP 517 option
`--config-settings=maturin.build-args=--locked`. The build uses a separate Cargo
target directory and disables pip's wheel cache, so it cannot substitute the
previously installed wheel. Its `cp38-abi3-linux_x86_64` wheel passes all 499
current applicable tests with the same three ONNX skips in another isolated
Python 3.14 environment, and installed native/stub/marker bytes plus runtime
checks match this newly built artifact. The plain Linux tag verifies this host;
it does not establish manylinux portability. Artifacts and receipt are under
`/workspace/SpatialRust-python-delivery/target/frozen-source-wheel-locked-dist/`
and `/workspace/SpatialRust-python-delivery/target/frozen-source-wheel-validation/`.

## Locale-independent report files

Report/configuration text IO explicitly uses UTF-8 across the Python alignment
examples and study tools. Previously both reference-pose and publisher-score
CLIs failed with `UnicodeEncodeError` when Python UTF-8 mode and locale coercion
were disabled under `LC_ALL=C`: Japanese provenance and the ≤ symbol could not
be written, despite the HTML declaring UTF-8. Two CLI regressions reproduce both
failures before the fix and now verify valid UTF-8 artifacts under those settings.
The change only specifies text encodings; registration calculations are unchanged.
This locally verified locale fix does not establish Windows/macOS runtime success.

## Current stable toolchain

Full CPU vision and Python/native integration are locally tested using Rust 1.99,
Linux x86_64 and CPython 3.12. Optional parallel vision uses Rayon 1.12, which
requires Rust 1.80. Some benchmark/dev dependencies require Rust 1.85 and current
optional dependency resolutions can require newer versions. Their full minimum
toolchain is not established by the base gate. Use current stable for full builds.

Do not interpret a library compilation as a runtime GPU, browser, macOS, Windows,
ARM or Python 3.8 conformance result. Those runtimes still require separate checks.
