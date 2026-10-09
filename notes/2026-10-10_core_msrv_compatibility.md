# Restore the declared base Rust compatibility

Actual Rust 1.75.0 validation failed because the unconstrained workspace
`thiserror = "2"` selected 2.0.21, requiring Rust 1.77. The workspace now pins
`=2.0.17`, whose declared MSRV is 1.61. API and algorithms are unchanged.
Root and Python lockfiles are regenerated locally but are intentionally untracked.

The new CI job generates a lockfile with Cargo 1.75, freezes resolution, checks
the default meta crate, core and vision geometry, then runs core/math all-feature
library tests. Fresh resolution is tested in an extracted source archive at
`/workspace/SpatialRust-python-delivery/target/msrv-fresh-source/`, overlaid with
the pinned manifest. A separate target avoids source-identity cache collisions.
The dtolnay Rust 1.75.0 action ref is confirmed by read-only Git lookup.

Validation:

- Rust 1.75.0: base/meta/geometry check and 36 core/math all-feature tests pass.
- Rust 1.99: 174 core/math/full-vision tests and 29 CPU-registration tests pass.
- Rebuilt ONNX Python extension: 450 tests, 20 optional comparison tests, strict
  Clippy and stubtest pass.
- Rebuilt isolated default wheel: 447 tests pass, exactly 3 expected ONNX skips;
  archive native/type bytes, actual imported paths, ICP/support and PCD roundtrip
  verification pass.

The README badge and `/workspace/SpatialRust-python-delivery/docs/TOOLCHAIN_SUPPORT.md`
now distinguish base Rust 1.75 from full-feature/dev/Python builds. Optional
parallel vision's Rayon requires Rust 1.80; dev dependencies can require 1.85 or
newer. Base compilation does not establish runtime GPU or other-OS conformance.
Remote CI execution remains unverified.

The public comparison's previous native binary is archived before rebuilding:
`/workspace/SpatialRust-python-delivery/target/public-geotransformer-comparison-final/baseline-native.so`.
Its hash matches that comparison's receipt. The repaired default wheel and receipt
are under `/workspace/SpatialRust-python-delivery/target/msrv-repaired-wheel/`.

Provisional maturity reaches 74%, not the requested 90%. The primary 3DMatch
dataset host returns a proxy CONNECT 403 and is absent from the current custom
allowlist. A configuration draft adds only `3dmatch.cs.princeton.edu`; the save
is confirmed with `requires_publish: true`. It does not apply access immediately.
Review/save and Publish in environment settings are required before retrying
the independently sourced multi-pair evaluation gate. No credentials are needed
or requested. Platform runtime and operational evidence are also still open.
