# Reuse immutable packed voxel gather controls

Parent: `b21341b67e8c7fefde34845a0e3404a68cf96da8` (merged PR 247).

Combined packed first-attribute recording and standalone packed submission
windows previously created a fresh 16-byte uniform for every packed pass.
Each recorder now owns a single-entry control cache keyed by every original
uniform word: cell count, point count, active channel count and zero padding.
Equal consecutive keys reuse the buffer. A changed key creates a new buffer
with the original mapped initialization; it never rewrites a submitted buffer.
The command encoder retains earlier resources after a cache entry is replaced.

Each combined recorder or standalone window gets a fresh cache. No controls
are shared between calls, windows or runtime devices. The existing single-pass
wrapper keeps a fresh cache per call, including standalone packed XYZ callers.
Pipeline lookup precedes control acquisition, as before. Keys are checked
after the original input uploads. No global cache, pool or public API changes.

## Source request ledger

All entries below retain the original four-channel batch width. Controls still
contain sixteen bytes and use the original UNIFORM-only mapped initialization.
These counts cover attribute controls at the named recording boundary. Any
surrounding XYZ/byte stages retain their original controls.

| Recording boundary | Uniform creation requests before / after |
| --- | --- |
| Combined, no attributes | 0 / 0 |
| Combined, one attribute | 1 / 1 (original scalar control) |
| Combined, four attributes | 1 / 1 |
| Combined, eight attributes | 2 / 1 |
| Combined, sixteen attributes | 4 / 1 |
| Combined, seventeen attributes | 5 / 2 (packed + original scalar tail) |
| Combined, nineteen attributes | 5 / 2 (four full batches + three-channel tail) |
| Standalone, two full batches in one original eight-channel window | 2 / 1 |
| Standalone, nineteen attributes in original 8/8/3 windows | 5 / 3 |

The combined nineteen-attribute case initializes 32 rather than 80 summed
source control bytes. These are source buffer/initialization requests, not
physical allocations or measured transfer/peak-memory/bandwidth savings.
Each packed pass additionally compares the exact key; total latency is
unmeasured. No latency benchmark was performed for this resource cleanup.

## Preserved execution

Planner/selection and packed priority, original batch widths, output reuse,
scalar combined tails (positive zero in empty cells), standalone one-channel
packed tails (point zero), inactive read-only aliases, uniform values, shaders,
bindings and dispatches are unchanged. Inputs and active output copies retain
their labels, sizes, offsets and order. Compute passes, payload transfers,
submissions, maps, waits, readback ownership and upload recycling are retained.
Submission windows remain bounded by the original plan. Both runtime buffer
pools and image code are unchanged.

## Validation

- **336 native tests:** 135 image, 181 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **3,055,494 guarded original/CPU words** from 64 parameter contexts and 128
  compute passes queued before the first submit/map. Adjacent contexts change
  only cell count, then only point count; channel counts repeat/change through
  1/2/3/4. Different inputs/segments with equal parameter keys also remain correct.
  Cases include 1/17/255/256/257/65,537 cells, NaN payloads, signed zeros,
  infinities, subnormals, repeated/descending starts and empty cells. Sixteen-word
  staging prefix/suffix guards and eight output-tail words stay intact. Earlier
  results remain owned after control/source buffers and the runtime are dropped.
- Existing output-reuse regressions additionally verify **47,647,888 raw**,
  **30,010,272 forced-width complete** and **496,273,824 public
  parent/CPU words/elements**. They cover widths two/four, scalar versus packed
  empty tails, original boundaries, pooled uploads, complete public calls,
  changed inputs, error precedence, concurrency and owned results.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds with WASM warnings denied; workspace formatting and whitespace.
- **489 unchanged original GPU/image Rust/WGSL files** and **494 validated
  source/executable hashes**. The complete original packed pass is frozen in
  the native queued test. Source comparison restores the original pass, both
  recording loops and the remaining source by removing only control caching.
  It verifies the original uniform words/descriptor and bind/dispatch behavior.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementations:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_packed.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_standalone_packed.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_control_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_control_reference.rs`.
Ignored receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/packed-gather-controls-*`.
The local adapter name is omitted from tracked files and delivery text.
