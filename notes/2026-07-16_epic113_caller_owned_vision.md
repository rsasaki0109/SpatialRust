# Epic 113 caller-owned CPU vision receipt

Date: 2026-07-16

Epic 113 closes the caller-owned output and reusable CPU scratch contract for
Gaussian blur, Sobel, rectangular morphology, Canny, and exact Euclidean
distance transform.

## Delivered surface

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\filter.rs`
  exposes packed Gaussian `*_into` execution with caller-owned output,
  `GaussianBlurU8Workspace`, stable capacity reuse, cached kernels, and explicit
  reserved-byte reporting.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\advanced_filter.rs`
  exposes caller-owned direct, absolute, threshold, paired-gradient, and fused
  L1 Sobel outputs. The direct signed path writes without a full-image
  intermediate.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\morphology.rs`
  owns full-image and per-worker line scratch in
  `RectMorphologyWorkspace`.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\canny.rs`
  accepts packed or strided output views and retains gradient, magnitude-ring,
  state, and frontier storage in `CannyWorkspace`.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py\src\lib.rs`
  supports Python `out=` identity for all four algorithm families and now
  exposes `GaussianBlurWorkspace` alongside the existing morphology, Canny,
  and distance-transform workspaces. Omitting it retains the convenience
  thread-local pool; passing it makes scratch ownership explicit.

No CPU entry point selects a GPU or performs a device transfer. NumPy overlap
checks reject input/output aliasing before mutable output access.

## Correctness and ownership gates

- Gaussian packed output length, NumPy shape, contiguity, overlap, strided
  input, output identity, capacity reuse, and reserved-byte stability.
- Sobel packed output lengths, derivative/channel contracts, strided input,
  metadata preservation, Python output identity, and overlap rejection.
- Morphology packed output length, rectangular-workspace eligibility, generic
  fallback parity, strided input, output identity, overlap rejection, and
  steady-state full-image/worker/line capacity.
- Canny dimension validation, packed/strided output padding preservation,
  binary parity, Python output identity, and steady-state allocation bounds.
- Exact EDT caller-owned output and workspace coverage remains recorded by
  Epic 113E.

## Validation

```powershell
cargo test -p spatialrust-vision
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --no-run
cargo check --manifest-path crates\spatialrust-py\Cargo.toml
pytest crates\spatialrust-py\tests
python -m mypy.stubtest spatialrust --ignore-missing-stub --allowlist crates\spatialrust-py\stubtest_allowlist.txt --ignore-unused-allowlist
```

This receipt introduces no new performance comparison or portable speed claim.
It records ownership, allocation-reuse, and correctness contracts only.
