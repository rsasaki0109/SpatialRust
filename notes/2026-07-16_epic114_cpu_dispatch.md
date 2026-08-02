# Epic 114 safe CPU dispatch receipt

Date: 2026-07-16

Epic 114 closes the shared size-aware CPU dispatch contract without changing
SpatialRust 1.1.0 public ownership, stride, border, or transfer semantics.

## Shared policy

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\dispatch.rs`
is the single source for the current CPU scheduling boundaries:

- 100,000 components for lightweight row-parallel resize and RGB-to-gray work.
- 262,144 components for row kernels that own per-worker Sobel/CHW work.
- 1,000,000 components for Gaussian, morphology, Canny, and larger shared
  gradient row/band work.
- Eight rows per scheduling tile for outputs at least 2,000 rows tall.

The policy is pure and deterministic. Parallel work additionally requires at
least two independent rows, planes, or columns. Worker counts are capped by
both the current Rayon pool and the independent item count, and the scalar path
always reports one worker.

## Fast paths and fallbacks

- Packed `u8` resize plans select specialized execution only for one- or
  three-channel input and output whose row strides equal their packed widths.
- Other channel counts, explicitly strided views, unsupported interpolation
  shapes, generic component types, and non-specialized borders retain their
  existing safe implementations.
- Sobel and Canny use packed caller-owned `f32`/`i16` intermediates while
  preserving strided `u8` input and output contracts.
- CPU entry points do not upload to a device or hide a CPU/GPU transfer.

## Worker and scratch bounds

- Sobel scratch is bounded by `workers * 3 * width * sizeof(i16)`.
- Canny magnitude scratch is bounded by
  `workers * 3 * width * sizeof(i32)`.
- Gaussian band scratch is bounded by the selected worker bands plus the
  vertical kernel halo; the caller-owned workspace retains that capacity.
- Rectangular morphology creates at most one reusable line-buffer set per
  bounded worker and retains its full-image intermediates in the caller-owned
  workspace.

The dispatch policy does not create a separate thread pool. Rayon kernels use
the current bounded pool; CHW packing creates at most one scoped worker per
channel and only above the shared 262,144-component boundary.

## Correctness gates

```powershell
cargo test -p spatialrust-vision --no-default-features --features full
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-features --lib -- -D warnings
cargo test --workspace --no-run
```

The vision suite covers exact threshold and worker bounds, packed selector
decisions, packed/strided parity, padding preservation, channel fallbacks,
border behavior, parallel seams, and the existing property tests. This receipt
adds no new portable performance claim.
