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
seven Python-wheel jobs and the first 30 returned CI jobs queued without an
assigned runner. The CI request used the API default page size; it did not
establish the status of the remaining jobs.
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

## Remote runtime evidence and Windows receipt repair

Complete pagination for main `767f128` observes 103 CI jobs and seven wheel
jobs. The earlier statements about every CI job lacking a runner were not
established by the first-page observations. `scripts/inspect_github_ci.py` now
fetches all pages, checks complete unique job membership against `total_count`,
and separates queued, active, successful, failed and skipped outcomes. A queued
workflow status does not imply that every job is queued. Metadata is sampled
across requests rather than as an atomic snapshot.

On that commit, macOS vision, visual and streaming conformance jobs and Linux
Python 3.12 binding checks complete successfully. The macOS 14/Python 3.12
installed-wheel runtime job also succeeds, including exact installed-byte,
archive/native/type and ICP/support/PCD verification. This is one hosted macOS
configuration; its hardware architecture is not inferred from the label.

Windows builds the wheel successfully but its installed-runtime step fails.
Public check annotations identify `cannot identify native extension` and
`cannot identify the loaded native extension`. The trimming and convergence
study CLIs searched package siblings for `*.so`; Windows loads a `.pyd` binary.
Those two scripts and the equivalent public-candidate validator now fingerprint
`importlib.import_module('spatialrust.spatialrust').__file__`, which identifies
the actual loaded binary without scanning siblings. Regression tests execute
real native studies with a controlled `.pyd`-only wrapper directory and verify
the imported binary hash rather than the non-loaded sibling. Linux passing
checks do not establish that the repaired Windows remote gate has passed.

Remote logs redirect to `productionresultssa3.blob.core.windows.net`, which the
current proxy denies. Only public check metadata/annotations were read; no
credentials were requested or extracted. The synthetic Redwood/ICL-NUIM
fragment source also requires `redwood-data.org`, currently denied by the proxy.
An additive environment draft preserves the existing three custom domains and
adds only that official dataset domain. Saving the draft does not activate it;
review/save and Publish are required before retrying that dataset operation.

## Managed environment recovery (2026-10-10)

The current cloud snapshot has Rust 1.99 under `/workspace/.spatialrust-env`;
its existing activation script restores PATH. Network-enabled executor calls
successfully fetch GitHub and the allowed 3DMatch project host. Earlier failures
from executor calls without network permission do not establish a dead proxy.
The Redwood, official fragment-archive and canonical Autoware S3 hosts remain
explicitly denied; a saved historical environment draft is not an active change.

A fresh frozen default wheel passes 565 tests with three expected ONNX skips and
exact installed-byte/runtime verification. Its six-row native/Open3D replay uses
freshly acquired publisher demo files with reference/input hashes. The prior
comparison installation's missing ONNX loader path is also identified and repaired
locally. These results do not supply independent sensor ground truth or measured
operational calibration. See `notes/2026-10-10_environment_data_recovery.md` for
activation, data hashes, remaining gates and the HEAD-response fixture regression.
