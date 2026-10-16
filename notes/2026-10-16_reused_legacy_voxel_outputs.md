# Reuse legacy voxel gather outputs per submission

Parent: `0529552b82960c80305d4684b60244b17e2d6a9b` (merged PR 243).

The original coalesced first-value gather fallback created separate two-/four-
buffer output arrays for every batch. Its existing submission windows contain
at most two batches. The private window recorder now allocates one output array
before the batch loop, and each active plane is copied to its original staging
offset before the next batch overwrites those outputs.

The array width is the original first batch's rounded width: two buffers for
one/two channels and four for three/four. Per-batch kernel selection is retained,
including the two-output kernel for a one/two-channel tail after a four-channel
batch. Inactive bindings remain distinct buffers; only active planes are copied.
Empty windows return before allocating anything, as their original empty loop did.

## Placement and exact behavior

The preferred packed-output route keeps priority. This change applies to the
original coalesced fallback when the packed route rejects, for example because
multiple output planes exceed one storage-binding limit while individual planes
fit, or on a device admitting six rather than seven storage buffers per stage.
Public selection, validation and both planners are unchanged. Tests directly
execute valid two-/four-channel fallback plans on the available device and use
reduced binding limits to check packed rejection plus legacy admission. They
do not allocate oversized buffers to force the default public route.

The complete current-parent fallback call and its window recorder are frozen
in native fixtures. Original uploads, empty read-only bindings, two-/four-channel
shaders, uniforms, pipeline selection, active copy count/offsets, staging sizes,
submissions, map/wait and owned readback are preserved. In particular, every
standalone tail retains the original point-zero value for empty/descending cells.
There is no scalar-tail substitution, new shader, public API, dependency or pool.

## Source-level output request ledger

Let `C` be cells, `S = 4*C` bytes per plane and `W` be batch width (2 or 4).
Previously each batch requested its rounded two-/four-buffer output array; now
each nonempty submission window requests one such array. A window containing
two full four-channel batches changes **8 -> 4 output buffer requests** and
**8*S -> 4*S summed output descriptor bytes**. At 262,144 cells that is 8 MiB
to 4 MiB. A complete 17-channel, four-wide call has windows 8/8/1: **18 -> 10
requests**, 18 MiB to 10 MiB summed descriptor bytes at that cell count. The
number of batches, input buffers, uniform buffers, copies, submissions, maps,
waits and payload transfer bytes remains unchanged.

These are source requests and summed descriptor sizes, not measured physical
allocations, peak residency or bandwidth. Reuse adds a dependency from each
copy to the next write; latency requires measurement on this fallback route.

## Validation

- **326 native tests:** 132 image, 174 voxel/readback, 18 filtering and 2 facade.
  Ignored GPU correctness tests ran; benchmark tests were excluded from this set.
- New direct-window tests compare **6,108,264 current-parent/CPU words**
  across empty windows, every tail, 1/3/17/255/256/257/65,537 cells, reversed
  permutations, repeated/descending starts, raw signed zeros/NaNs/infinities and
  32 untouched staging suffix words. Both existing batch widths are exercised.
- Complete fallback tests compare **102,322,584 parent/CPU/owned words**,
  through 262,145 cells and 17 channels, with exact capacities, changed host
  inputs and pool clearing. Four concurrent calls retain **688,238
  owned words** after GPU buffers and runtime drop.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds (WASM warnings denied); workspace formatting and whitespace checks.
- **480 unchanged original GPU/image Rust/WGSL sources** and **485 validated
  source/executable hashes**. Only output hoisting/sizing, the empty guard and
  native test registration change original code. Public selection, planners,
  shaders and complete parent executor/copy bodies are checked unchanged.
- Eight forced fallback conditions additionally pass: widths 2/4, 16 through
  1,048,577 cells and 5/9/11/17 channels, three warmups and 30 alternating pairs.
  The functional-only run excludes all its timings from performance evidence.

## Measurements

Neither bounded 60-second scheduling window obtained an isolated run because competing work remained present. No timing sample is accepted as performance evidence. A separate eight-condition functional run passed with every timing sample excluded. Latency, physical allocations and peak memory remain unmeasured.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_gather_submit.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_legacy_reuse_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_legacy_reuse_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_legacy_reuse_benchmark.rs`.
Ignored receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-legacy-reuse-*`.
The local adapter name is omitted from tracked files and delivery text.
