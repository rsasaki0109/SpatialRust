# Reuse immutable legacy voxel gather controls

Parent: `389fb3f07106abc6e19fe46f31e0badd0647d879` (merged PR 250).

The coalesced standalone first-value fallback records at most two original
batches per submission window. It previously created a 16-byte uniform for each
two-/four-output pass. Each window now owns a private single-entry cache keyed
by every original uniform word (cell count, point count, active channel count,
zero padding) and the original descriptor label. Equal consecutive keys reuse
the immutable buffer. A changed key creates a new buffer with the original
UNIFORM-only mapped initialization; earlier commands retain their resources.
Windows, calls and devices share no controls.

The original single-pass helpers retain fresh per-call controls for combined
recorders and reference/benchmark fixtures. Four-output pipeline availability
is checked before control acquisition, retaining the original error path.
Descriptor labels also remain correct when a cache sees different pass widths.
The preferred packed route retains priority. This change affects the coalesced
fallback when the packed planner rejects a call; it does not move ordinary
calls away from the preferred path.

## Source request ledger

Counts cover the coalesced fallback only, using its original two-batch windows.
Input uploads, reusable outputs, staging copies and waits retain their original
counts. Other combined/scalar/XYZ/byte paths keep their controls.

| Recording boundary | Uniform creation requests before / after |
| --- | --- |
| Two full width-two batches in one four-channel window | 2 / 1 |
| Two full width-four batches in one eight-channel window | 2 / 1 |
| Width-four window with four then two/three channels | 2 / 2 |
| Width-four nineteen channels in original 8/8/3 windows | 5 / 3 |
| Width-two nineteen channels in original 4/4/4/4/3 windows | 10 / 6 |

Uniforms remain sixteen bytes. Width-four nineteen-channel recording initializes
48 instead of 80 summed source control bytes. These are buffer creation and
initialization requests, not physical allocations or measured transfer/peak-
memory savings. Each cached pass compares the exact key and label. No latency
benchmark was run; latency and bandwidth effects are unmeasured.

## Preserved execution

Selection and planner bounds, submission window sizes, input/empty bindings,
original two-/four-output pipeline and active channel count, point-zero empty
cell behavior, inactive output guards, all control data/labels, shaders, bind
groups and dispatch order remain. The per-window output array still copies
every active plane to its original staging offset before reuse. Payload copies,
submits, maps, waits, result ownership and error order remain. No public API,
pool, core, dependency or image code changes.

## Validation

- **339 native tests:** 135 image, 184 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **6,379,488 guarded original/CPU words** from 96 contexts / 192
  passes queued before the first submit/map. Both complete PR 250 pass bodies
  are frozen independently. Active bits, inactive outputs and all guards match
  those passes and CPU expectations exactly. Cases change only cell count,
  then only point count, repeat keys with different inputs, switch output width
  with equal uniform data, and cover channel counts 1/2/3/4. Sizes include
  1/17/255/256/257/65,537 cells, repeated/descending starts, empty cells, permuted
  indices, NaN payloads, signed zeros, infinities and subnormal bits. Four
  independent outputs have eight tail words each; staging has sixteen prefix
  and suffix words. Owned results survive control/source buffers and runtime
  drop. Devices without four-output support still run the width-two cases.
- Existing fallback regressions verify **6,108,264 raw reference/CPU
  words**, **102,322,584 complete fallback/reference/CPU words** and
  **688,238 concurrently retained owned words**. They force
  both original widths, reject packed bindings, cover every partial tail and
  original submission boundary, validate capacities and retain results across
  input mutation, repeated calls, concurrency and runtime drop.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds with WASM warnings denied; workspace formatting and whitespace.
- **495 unchanged original GPU/image Rust/WGSL files** and **500 validated
  source/executable hashes**. Source audit restores both complete passes, the
  bounded window recorder and all remaining source by removing only control
  caching and the native test hook; original uniform words/labels, availability
  errors, binding/dispatch and preferred/combined routes are checked.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementations:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_submit.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_legacy_control_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_legacy_control_reference.rs`.
Ignored receipts: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/legacy-controls-*`.
The local adapter name is omitted from tracked files and delivery text.
