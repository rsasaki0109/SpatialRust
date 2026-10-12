# Complete GPU covariance fallback writes

## Reproduction and correction

Plane covariance output has two vec4 rows per point. The public result takes
`[c00, c11, c22, c01]` from the first row and `[c02, c12]` from the second.
For fewer than three radius neighbors, the shader wrote only the first row,
assuming the remaining components were initially zero. Scratch acquisitions
are uninitialized and can contain previous results. Reuse therefore exposed
stale cross terms, including NaNs, in the intended epsilon-scaled identity.

Before the correction, an independently poisoned native output retained
`[1.5, 1.5, 1.5, 1.5]` in the second row of a singleton. A public API call with
a deliberately poisoned compatible pooled allocation returned NaN cross terms
instead of zeros. Both new regressions failed against the parent before the
production change and pass afterward. Reproduction logs remain private at:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-covariance-fallback-repro.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-covariance-fallback-public-repro.log`

The early-return branch now explicitly writes a zero second vec4. The first
row retains the existing epsilon diagonal and zero c01. The public six-word
fallback is `[epsilon, epsilon, epsilon, 0, 0, 0]`; the two second-row padding
words are zero as well. This is an output-completeness correction, not another
covariance algorithm or a placement change. Gather order, distance/count math,
the supported eigensolver/regularization and host decoding remain unchanged.

Production source:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/covariances_grid.rs`.

The existing entry guard enforces `i < n`. Covariance planning already bounds
`n <= u32::MAX/2` and allocates `2*n` vec4 rows, so `n+i < 2*n` fits the legal
output and u32 addressing. The added store uses that same address as the
supported branch. It requires no global buffer clear, extra dispatch/barrier,
new allocation/pipeline, public signature, feature, dependency or unsafe code.
Full uploads, scratch submission/recycling, one compute/result-copy submission,
mapping synchronization and `32*n` result staging bytes remain unchanged.

## Regression coverage

Three native tests at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/covariance_output_tests.rs`
cover:

- **266,112 current/frozen raw output and canary words** across ninety paired
  cases. Ten sizes cover singleton/pair/triple support and partial final groups
  through 1,025 points. Three epsilon values include negative zero; three
  prefilled patterns include positive/negative finite values and NaN bits.
  Independent membership identifies the insufficient-support samples, whose
  diagonal and zero cross terms/padding are checked exactly. Supported samples
  retain all frozen output bits without imposing a unique eigenbasis on
  degenerate data. An extra canary row remains unchanged in both implementations.
- Twenty-four public size/epsilon cases deliberately seed the compatible
  output-size pool allocation, submitting every poison write before recycling.
  Singleton/pair bands independently require the epsilon identity. A second
  same-shape call performs no new scratch allocation, reuses all six leases,
  and retains the first independently owned host result. Negative-zero diagonal
  bits and positive-zero cross terms are checked explicitly.
- Four concurrent callers start with eight poisoned output-size resources,
  use distinct epsilon values, and retain/check three results each after later
  calls. Outputs remain independent and complete under shared runtime reuse.

The frozen shader at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/covariance_output_previous.wgsl`
matches parent source `ab837794a3f9b56d605da02ca6e15770037b1a52` after line-ending
normalization. Its insufficient-support second rows are intentionally asserted
to retain the poison: the test records the old defect, rather than changing
the frozen source to conceal it. Only that branch's missing store differs in
the current shader.

The facade smoke at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_covariance_fallback_smoke.rs`
first computes a tilted plane with nonzero c02/c12. It then processes 257
isolated points of the same output size and checks exact epsilon identities,
storage reuse and preservation of the previous host result. This exercises
natural prior results instead of relying solely on synthetic poison. It needs
a native adapter and is ignored by default on GPU-less CI, then run explicitly
with `--include-ignored` on a compatible host.

Native Windows, Rust/Cargo 1.98.1 and wgpu 24: **171 GPU tests pass** (144
unit plus 27 integration), zero failures and fifteen ignored diagnostics.
All three new tests pass in minimal `gpu-wgpu`; the explicit facade smoke and
GICP GPU covariance alignment pass. Strict default/all-feature all-target GPU
Clippy and facade Clippy pass. Three WASM builds pass: minimal GPU, GPU image/
AoSoA, and minimal facade GPU. This establishes compilation, not browser
execution. Formatting and diff checks pass.

```text
cargo test -p spatialrust-gpu --no-default-features --features gpu-wgpu --lib covariances_grid::output_tests -- --test-threads=1 --nocapture
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo test -p spatialrust --no-default-features --features gpu-wgpu --test gpu_covariance_fallback_smoke -- --include-ignored --test-threads=1
cargo test -p spatialrust-registration --no-default-features --features register-gicp-gpu --lib aligns_with_gpu_covariances -- --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features gpu-wgpu --test gpu_covariance_fallback_smoke -- -D warnings
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu,gpu-image,gpu-aoso-staging
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu
```

No throughput improvement is claimed for this correctness fix. It writes the
already allocated output completely instead of adding initialization work to
the executor or introducing hidden copies. Full validation logs remain private
under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-covariance-fallback-*`.
Published evidence omits the local adapter model. Delivery retains the existing
114 named successful CI checks and an exact source/dependency commit merge guard.
