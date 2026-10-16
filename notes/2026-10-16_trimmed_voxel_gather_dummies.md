# Trim inactive combined voxel gather outputs

Parent: `2e7f384759d034d0ad3d98e86cfa972d945d6943` (merged PR 244).

The legacy combined XYZ/first-attribute fallback created four full-size dummy
outputs for every three-/four-channel batch, even though a three-channel batch
binds only one and a four-channel batch binds none. Extract that recording block
into a private helper and allocate only the missing output slots.

The upstream match invokes the helper only for three/four active channels with
the four-channel pipeline available. Its channel count is the active output
slice length, equal to the original chunk count. Remaining active and dummy
outputs stay distinct. The unused read-only input keeps the existing four-byte
empty buffer. No shader, binding layout, public API, dependency or pool changes.

## Source request ledger

Let `S = 4*C` bytes per output plane for `C` cells.

| Active channels | Dummy outputs before / after | Active + dummy outputs before / after | Fewer summed output descriptor bytes |
| --- | --- | --- | --- |
| 3 | 4 / 1 | 7 / 4 | 3*S |
| 4 | 4 / 0 | 8 / 4 | 4*S |

At 262,144 cells, `S` is one MiB. A four-channel batch removes four MiB of
summed dummy descriptors; a three-channel batch removes three MiB. In a
four-wide 19-attribute fallback, dummy requests become **20 -> 1** and all
attribute output requests **39 -> 20**, reducing summed descriptors by 19 MiB
at that cell count. These are source API requests/descriptor totals, rather
than measured physical allocations, peak memory or bandwidth.

The preferred packed/fused paths keep priority. This helper is reached only
in the existing combined fallback; devices using scalar/two-channel batches
retain their previous path. Input uploads/recycling, uniform controls,
pipeline choice, channel math and active output copies at the original XYZ
staging offset are preserved. Scalar empty-cell tails return positive zero;
multi-channel tails retain point zero, as before. Compute passes, submissions,
maps, waits, readback ownership and payload transfers are unchanged. There is
no shared output reuse or additional GPU dependency in this slice.

## Validation

- **329 native tests:** 132 image, 177 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **5,180,820 guarded raw parent/CPU word comparisons** for three/four
  active channels, three data presets and 1/3/17/255/256/257/65,537 cells.
  Inputs include signed zeros, NaN payloads, infinities and arbitrary words,
  reversed permutations, repeated starts and descending/empty cells. A sentinel
  XYZ-prefix region, sixteen words after each active output and a 32-word
  staging suffix remain unchanged. The prefix is a guard, not an XYZ result.
- **122,988,675 guarded batch/parent/CPU words** through 262,145 cells
  and 19 attributes, including scalar positive-zero tails and multi-channel
  point-zero tails, changed host input and owned results after pool clearing.
- Four concurrent independent calls retain **557,278 guarded
  owned words** after buffers/runtime drop. All three native markers confirm
  that the four-channel capability was exercised on the validation device.
- All-features/all-targets Clippy with warnings denied, CPU-only and minimal/CI
  WASM builds (WASM warnings denied), workspace formatting and whitespace checks.
- **483 unchanged original GPU/image Rust/WGSL sources** and **488 validated
  source/executable hashes**. The original outer fallback restores exactly by
  inlining the extracted block. The frozen block matches that complete original
  block; the new helper differs only in missing-slot allocation/indexing.
- Eight additional private recording fixture conditions pass with three/four
  active channels and up to 1,048,577 cells, three warmups and thirty alternating
  pairs. Their non-isolated timings are excluded from performance evidence.

The guarded fixtures directly exercise the extracted block with its surrounding
uploads, outputs and copies. They deliberately bypass public packed/fused
selection; they are not measurements of a complete public combined XYZ call.
Existing public combined-call regressions additionally pass, and unchanged
selection/copy/XYZ behavior is checked by source comparison.

## Measurements

Neither bounded 60-second scheduling window obtained an isolated run because competing work remained present. No timing samples are accepted as performance evidence. A separate eight-condition guarded fixture run passed with all timings excluded. No public-call latency, physical allocation or peak-memory improvement is established.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_dummy_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_dummy_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_dummy_benchmark.rs`.
Ignored receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-gather-dummy-*`.
The local adapter name is omitted from tracked files and delivery text.
