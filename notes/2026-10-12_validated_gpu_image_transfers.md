# Validated GPU image transfers

## Scope

GPU image upload previously checked channels and conversion to u32 dimensions,
then allocated a texture before packing RGBA rows. It did not check the runtime's
2D texture limit or representability of the physical packed allocation. Readback
calculated row padding in u32 and allocated a staging buffer before checking its
size against the device limit. Unsupported dimensions could reach wgpu validation;
large synthetic dimensions could overflow row/storage/capacity arithmetic.

The implementation in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/gpu_image.rs`
now computes private transfer plans before work allocation:

- Upload converts dimensions to u32, validates positive dimensions and 1..=4
  channels, checks both axes against `max_texture_dimension_2d`, checks `width * 4`
  row bytes and u64 physical storage bytes, and validates the packed host length
  against the target's `isize::MAX`/usize capacity. Only then does it acquire a
  texture and pack/upload the image.
- Readback keeps ownership/channel rejection first, then checks texture axes and
  RGBA row bytes. It calculates 256-byte row padding in u64, checks conversion to
  the u32 copy-layout field, checks full padded staging bytes against
  `max_buffer_size`, and validates host staging/logical-output capacities before
  buffer allocation, encoder creation or copy submission.
- Internal `GpuImage::from_parts` uses checked u64 physical storage arithmetic.
  Its signature and existing positive-dimension/channel validation remain.

The padded buffer policy is unchanged: every row, including the last, reserves
its full aligned stride. For example, 65x3 RGBA texels occupy 780 physical bytes;
readback stages through 1,536 bytes. The exact staging limit succeeds and one
byte below it rejects. Upload does not allocate a caller-visible GPU buffer and
is not restricted by `max_buffer_size`; neither texture transfer requires
compute or storage-buffer binding capabilities. The plans are capacity/device
checks rather than new peak-memory or texture-retention budgets. Allocation
availability remains dependent on the host/backend.

Successful behavior is preserved: strided/ROI packing, zero unused RGBA
components, logical channel order, metadata, texture pooling and explicit
upload/readback. Receipt stages are unchanged; payload accounting still uses
four physical bytes per pixel rather than padded staging bytes or logical
channel bytes. Each successful explicit readback adds another physical image
payload to its receipt. Rejected calls leave receipts and texture-pool counters
unchanged. No shader, compute pipeline, public signature, dependency, unsafe code
or implicit transfer is added. This slice establishes transfer validation and
compatibility; it makes no throughput improvement claim.

## Correctness

Five native tests in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/transfer_tests.rs`
cover:

- **12,300 independent upload/readback layouts**: all widths 1..=1,025, heights
  1/3/257 and channels 1/2/3/4. Independent bit-mask row alignment checks both
  sides of each 256-byte boundary, physical lengths, staging lengths and logical
  output capacities.
- Zero dimensions, invalid channels, exact/over-limit texture axes, zero texture
  capability, u32 RGBA row overflow, the largest representable aligned row and
  its successor, usize dimension conversion, exact/over-limit host lengths,
  u64 physical-storage overflow and exact/one-byte-under padded-buffer limits.
  Zero compute/storage-binding limits remain acceptable for transfers. Large
  synthetic limits test arithmetic only, without allocating giant images.
- **120 independent full-RGBA GPU cases** across channels 1..=4, ten shapes and
  packed/strided/ROI inputs. Shapes cover singleton axes, 63/64/65 and 127/128/129
  row boundaries and 257x19 odd tails. Pixel-coordinate/channel formulas verify
  host packing and every uploaded physical byte, including unused zeros. Two
  logical readbacks per case preserve metadata and cumulative payload/stages.
  Texture aliases used for physical validation are dropped rather than pooled.
- Invalid public uploads use small actual host inputs beyond the real device's
  texture axes; empty/unsupported-channel uploads also reject. Foreign runtime
  and channel readbacks retain their existing rejection behavior. Metadata-only
  negative readback probes keep a tiny backing texture, reject before work and
  restore valid dimensions before recycling. Overflowing internal storage is
  rejected. A wgpu validation error scope stays empty, receipts/pool counters
  stay unchanged, caches stay cold, and a subsequent valid readback succeeds.
- Four concurrent uploads retain independent live results. Readbacks survive
  runtime movement, twelve dirty-texture reuse cycles and setting the retention
  budget to zero while an image is live. Pooled texture allocation count stays
  unchanged during the reuse cycles; zero budget discards recycled textures.

The facade test is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_image_transfer_smoke.rs`.
It exercises a strided 65x7 RGB ROI through the public re-exports and verifies
payload/stage transfer boundaries. New native GPU tests require an available
headless runtime rather than silently skipping execution.

## Validation

Native Windows with Rust/Cargo 1.98.1 and wgpu 24: **159 image/point GPU tests
pass**, comprising 132 unit and 27 integration tests, with twelve ignored
diagnostics and no failures. The facade smoke passes. The five new tests also
pass in the minimal `gpu-image` configuration. Strict all-target Clippy passes
for default/all-feature GPU configurations and the facade smoke. Feature-off and
wgpu-only builds pass. GPU image, GPU image/AoSoA and facade GPU image WASM builds
pass; browser execution is not established. Formatting and diff checks pass.
Shader math, resident image/point chains, existing stage-profile receipts and
explicit transfers are covered by the full regressions.

```text
cargo test -p spatialrust-gpu --no-default-features --features gpu-image --lib transfer_tests -- --test-threads=1 --nocapture
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo test -p spatialrust --no-default-features --features gpu-image --test gpu_image_transfer_smoke
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features gpu-image --test gpu_image_transfer_smoke -- -D warnings
cargo check -p spatialrust-gpu --no-default-features
cargo check -p spatialrust-gpu --no-default-features --features gpu-wgpu
cargo check -p spatialrust-gpu --no-default-features --features gpu-image --target wasm32-unknown-unknown
cargo check -p spatialrust-gpu --no-default-features --features gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown
cargo check -p spatialrust --no-default-features --features gpu-image --target wasm32-unknown-unknown
cargo fmt --all --check
git diff --check
```

Private ignored execution logs remain under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-image-transfers-*`.
Public notes omit the local adapter model. Workflows/check names are unchanged;
all 114 named CI checks and an exact source commit gate merge.
