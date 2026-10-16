# Reuse immutable voxel average controls

Parent: `2ea81eb56c28b8411076e60bd40de6df171c8f5c` (merged PR 248).

Standalone batched f32 averages and combined attribute-average recording
previously created a fresh 16-byte uniform for each packed pass. Each invocation
now owns a private single-entry cache, comparing all original words: cell count,
point count, active channel count and zero padding. Equal consecutive keys reuse
the buffer. A changed key creates a new immutable buffer using the original
UNIFORM-only mapped initialization. Earlier encoded passes retain their buffers;
no queue write can change their parameters. Calls and devices share no controls.

The single-pass helper is now native-test-only because both production callers
use the cached recorder. Its fresh-cache-per-call behavior remains available to
the existing low-level regression and benchmark fixtures.

## Source request ledger

Counts below apply to one four-wide attribute recorder, either the standalone
batched average call or the combined attribute recorder receiving that many
channels. Outer fused XYZ/attribute prefixes and byte stages keep their original
controls; their attributes can reduce the count handed to this recorder.

| Attributes at recorder boundary | Uniform creation requests before / after |
| --- | --- |
| None | 0 / 0 |
| One | 1 / 1 (original scalar path) |
| Four | 1 / 1 |
| Eight | 2 / 1 |
| Sixteen | 4 / 1 |
| Seventeen | 5 / 2 (packed + original scalar tail) |
| Nineteen | 5 / 2 (four full batches + three-channel tail) |

Controls still contain sixteen bytes. Nineteen attributes initialize 32 rather
than 80 summed source control bytes. These are buffer creation/initialization
requests, not physical allocation or measured transfer/peak-memory savings.
Each packed pass adds an exact key comparison. No latency benchmark was run;
latency and bandwidth effects are unmeasured.

## Preserved execution

Planners, selection priority, four-wide batches, scalar one-channel tails,
empty-cell positive zero, inactive read-only aliases, all uniform values,
pipeline lookup order, shaders, sequential f32 summation/division, bindings and
dispatches remain. Input uploads and reusable packed outputs retain their
labels/sizes/order. Active output planes are copied to the original staging
offsets before reuse. Payload copies, submits, maps, waits, readback ownership,
error order and public APIs remain. No buffer pool or image code changes.

## Validation

- **337 native tests:** 135 image, 182 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **3,055,494 guarded words** from 64 contexts / 128 passes queued before the
  first submit/map. Cached outputs match the complete frozen PR 248 pass
  bitwise, including guards. Bounded finite inputs also match independent
  sequential CPU averages within four f32 epsilons scaled by max(1, abs(value));
  zero bits match exactly. Adjacent contexts change only cell count, then only
  point count; channel counts 1/2/3/4 repeat and change with different inputs.
  Cases include 1/17/255/256/257/65,537 cells, permuted point indices, empty cells,
  16-word staging prefix/suffix and eight output-tail guards. Results remain
  owned after controls, source buffers and runtime drop.
- Existing average regressions verify **15,005,136 exact reference
  words** and **496,273,824 complete reference/CPU elements** across
  standalone averages, first-XYZ plus averaged attributes, and centroids plus
  averaged attributes. They cover partial batches, fused wide prefixes, sparse
  cells, nonfinite values, guards, changed inputs, owned capacities, errors and
  concurrent calls.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds with WASM warnings denied; workspace formatting and whitespace.
- **492 unchanged original GPU/image Rust/WGSL files** and **496 validated
  source/executable hashes**. Source audit restores the complete original pass,
  both production loops and remaining source by removing only the cache and
  native test hook; it checks the original control descriptor and every word.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_average_batch.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_average_control_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_average_control_reference.rs`.
Ignored receipts: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/average-controls-*`.
The local adapter name is omitted from tracked files and delivery text.
