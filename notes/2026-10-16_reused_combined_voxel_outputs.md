# Reuse combined voxel attribute outputs

Parent: `0e828335d71aca718d6df7f70dba64472669fb39` (merged PR 245).

The combined XYZ/first-f32-attribute recorder keeps its existing empty guard and
preferred packed selector. Extract its legacy fallback into a private recorder
and allocate the active output array once per invocation, sized to the first
batch. Each batch borrows only its active prefix, runs the original scalar,
two-channel or four-channel kernel, then copies every active plane to its
original unified staging offset before the next batch overwrites the array.

The previous helper for absent four-channel slots is unchanged: a three-channel
tail still allocates one distinct dummy, while a full four-channel batch has
none. Scalar empty-cell tails remain positive zero; multi-channel empty cells
still gather point zero. Input uploads/recycling, controls, kernel selection,
copy offsets/lengths, transfers, compute passes, submissions, maps, waits and
owned readback results are preserved. Output arrays remain private to each
invocation; calls do not share outputs. No shader, public API or pool changes.

## Source request ledger

Let `A` be the attribute count, `W` the existing device batch width and `S=4*C`
the bytes per output plane for `C` cells. Active output requests become
`A -> min(A,W)`. Existing dummy requests remain unchanged. Zero attributes
retain the early return, and a single batch requests the same number of outputs.

| Four-wide attributes | Active output requests before / after | Active + dummy requests before / after | Fewer summed descriptor bytes |
| --- | --- | --- | --- |
| 4 | 4 / 4 | 4 / 4 | 0 |
| 8 | 8 / 4 | 8 / 4 | 4*S |
| 17 | 17 / 4 | 17 / 4 | 13*S |
| 19 | 19 / 4 | 20 / 5 | 15*S |
| 31 | 31 / 4 | 32 / 5 | 27*S |

At 262,144 cells, `S` is one MiB: 19 attributes remove fifteen MiB of summed
output descriptors over recording. These are source buffer creation requests
and descriptor sums, not measured physical allocations or peak memory.
The active array is created before the first batch upload; previous per-batch
output creation followed that batch's uploads. Command copies now introduce
explicit reuse dependencies within the same encoder, before later writes.

## Validation

- **332 native tests:** 132 image, 180 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **19,406,168 guarded original/CPU words** across 1/3/17/255/256/257/65,537
  cells and 0/1/2/3/4/5/6/7/8/9/11/12/17/19 attributes. Tests directly call the
  extracted fallback, bypassing preferred selection. NaN payloads, infinities,
  signed zeros, arbitrary words, reversed indices, repeated/descending starts
  and empty cells match the original. The XYZ-prefix guard and 32-word staging
  suffix remain sentinel-filled. Sixteen extra words per output plane retain
  their original zero initialization through reuse and copying.
- **285,334,110 guarded recorder/CPU word comparisons** through 262,145
  cells and 31 attributes. The original and current fallback and preferred
  recorders agree; a changed host input remains correct after upload recycling,
  pool clearing and earlier results being retained. Existing entries in the
  upload-recycle vector remain before the appended channel buffers.
- Four concurrent independent calls retain **803,088 guarded
  owned words** after runtime/buffer drop. All three native markers confirm
  that the validation device exercised the four-channel capability.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds with WASM warnings denied; workspace formatting and whitespace.
- **486 unchanged original GPU/image Rust/WGSL sources** and **490 validated
  source/executable hashes**. Inlining the private recorder and restoring only
  output allocation scope/active references reproduces the original selector
  and fallback. The frozen original fallback and selector also restore exactly.

The fixtures exercise complete attribute recording, uploads, staging copies,
submission and owned readback. Their XYZ prefix is a guard, not computed XYZ.
They are not measurements of a complete public combined XYZ call. Existing
public combined-call regressions also pass in the native suite.

## Measurement scope

This slice establishes the source request reduction and correctness. No latency
benchmark was performed; no speed, physical allocation, bandwidth or peak-memory
improvement is claimed. Preferred packed/fused algorithm placement is unchanged.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_combined_reuse_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_combined_reuse_reference.rs`.
Ignored receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-combined-reuse-*`.
The local adapter name is omitted from tracked files and delivery text.
