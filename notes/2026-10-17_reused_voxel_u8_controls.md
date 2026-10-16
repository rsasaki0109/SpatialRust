# Reuse immutable voxel u8 batch controls

Parent: `7e9678fa4ba877fa1491a1bebf0d5e34bbd89139` (merged PR 249).

Wide byte-attribute recording previously created a new 16-byte uniform for
every packed first-value or average pass. Each admitted wide recorder now owns
a private single-entry cache keyed by all original uniform words: cell count,
point count, active channel count and zero padding. Equal consecutive keys reuse
the buffer. Changed keys create a new immutable buffer with the original
UNIFORM-only mapped initialization. Earlier encoded passes retain their buffers.
No queue write changes their parameters, and calls/devices share no controls.

The original single-pass wrapper retains a fresh cache per call, including the
two-to-four-channel recorder and existing reference/benchmark fixtures. Pipeline
selection still precedes control acquisition on every pass. First/average mode
does not change control bytes, so equal keys can safely use one immutable
uniform with either pipeline; no pipeline or bind group is cached here.

## Source request ledger

These counts apply at the admitted four-wide byte-attribute recorder boundary,
for either first-value or average mode. Surrounding XYZ/f32 stages retain their
controls. Calls with at most four byte channels retain their original routing
and single-pass control creation.

| Byte channels handed to wide recorder | Uniform creation requests before / after |
| --- | --- |
| Five | 2 / 2 (packed + original scalar tail) |
| Seven | 2 / 2 (four-channel + three-channel batches) |
| Eight | 2 / 1 |
| Sixteen | 4 / 1 |
| Seventeen | 5 / 2 (packed + original scalar tail) |
| Nineteen | 5 / 2 (four full batches + three-channel tail) |
| Twenty | 5 / 1 |

Controls remain sixteen bytes. Nineteen channels initialize 32 rather than 80
summed source control bytes; twenty initialize 16 rather than 80. These are
buffer creation/initialization requests, not physical allocations or measured
transfer/peak-memory savings. Each packed pass compares the exact key. No
latency benchmark was run; latency and bandwidth effects are unmeasured.

## Preserved execution

Planners and fallback priority, four-channel batch width, original scalar
one-channel tail, empty-cell zero, inactive read-only aliases, uniform values,
pipeline lookup order, shaders, sequential f32 byte averages and WGSL rounding,
bindings and dispatches remain. Inputs keep their original borrowed/padded
packing and upload labels/order. The reusable output still copies active packed
planes, including zeroed final-word padding, to the original staging offsets
before overwrite. Payload copies, submits, maps, waits, ownership, errors and
public APIs remain. No buffer pool, core, dependency or image code changes.

## Validation

- **338 native tests:** 135 image, 183 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **1,532,792 guarded words** from 128 mode/parameter contexts / 256 passes
  queued before the first submit/map. Cached outputs match the complete frozen
  PR 249 pass and independent CPU byte expectations exactly, including packed
  zero padding and guards. The CPU fixture avoids half ties. Adjacent contexts
  change only cell count, then only point count; channels 1/2/3/4 repeat/change
  with different inputs. Both pipelines use the same key in adjacent passes.
  Cases include 1/17/255/256/257/65,537 cells, unaligned input lengths, permuted
  indices, empty cells, sixteen-word staging prefix/suffix and eight output-tail
  guards. Results remain owned after controls, source buffers and runtime drop.
- Existing wide-byte tests verify **146 admission conditions**,
  **27,596,192 exact reference/CPU cell bytes**,
  **655,369 rounding/long-sum reference/CPU cell bytes** and
  **672,578,280 complete reference/CPU elements** across four public
  APIs. They cover every ordered pair of byte values (including half ties), f32
  sums beyond 2^24, all packed tails, sparse/empty cells, adjacent staging guards,
  no-write rejection, errors and owned output capacities. Existing concurrent
  GPU callers and broader readback/image regressions also pass.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds with WASM warnings denied; workspace formatting and whitespace.
- **493 unchanged original GPU/image Rust/WGSL files** and **498 validated
  source/executable hashes**. Source audit restores the complete original pass,
  wide recording loop and remaining source by removing only control reuse and
  the native test hook. It checks all original control words/descriptor,
  pipeline selection, bindings and dispatch order.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementations:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_batch.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_wide.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_control_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_u8_control_reference.rs`.
Ignored receipts: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/u8-controls-*`.
The local adapter name is omitted from tracked files and delivery text.
