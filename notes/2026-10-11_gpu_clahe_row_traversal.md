# GPU CLAHE row traversal

Date: 2026-10-11. Repository: `rsasaki0109/SpatialRust`.
Slice: `GPU CLAHE padded traversal`; branch: `codex/gpu-clahe-row-traversal`.
Baseline source: PR #133, `ade339d037d5f0b83126ac941197e0ec772a1286`.
The flat lookup shader is unchanged from PR #132 at that source.

## Implementation and bounds

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_lookup.wgsl`
  replaces flat-index traversal in tiles larger than 256 padded pixels with
  power-of-two row strips. The input texture, histogram atomics, clipping/CDF,
  binary64-compatible LUT rounding and workgroup tile loop are preserved.
- For tile width W and height H, a=ceil(log2(W)), b=min(8,floor(log2(H))) and
  s=min(a,max(5,8-b)). There are 2^s column residues and 256/2^s row residues.
  A lane starts at column `lane & (2^s-1)` and row `lane >> s`, stepping by
  those respective strides. Every padded coordinate has exactly one pair of
  residues, and therefore exactly one owning lane. Ordinarily columns cap at
  32; short tiles increase that cap to keep lanes occupied, while narrow tiles
  reduce columns and increase row residues. No adapter-specific dispatch policy
  or pipeline specialization is introduced.
- Tiles of area <=256 assign one pixel to each active lane. This preserves the
  small-tile path without executing a nested pixel loop. Workgroup barriers are
  outside the varying pixel loops and before each histogram reuse.
- Reflect101 returns an in-bounds coordinate directly. Only padded positions
  compute a remainder; length-one axes map to zero. Padded width/height and area
  are still checked by the existing host plan. Stride increments occur only
  when the remaining extent exceeds the stride, avoiding u32 wrap. Histogram
  counts remain bounded by the validated tile area.
- The existing 256-lane workgroup, roughly 3 KiB local storage and 4 KiB device
  requirement remain. Scratch is still `1024*tiles_x*tiles_y + 16*(width+height)`
  bytes, leased through the application submission. There are still three
  passes, one submission, caller-owned output, explicit payload transfers and
  explicit collection of 48 timestamp-query bytes. Public signatures,
  metadata, feature/dependency boundaries and resource pools are unchanged.

## Correctness and verification

`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_traversal_tests.rs`
uses the production lookup shader with only its final LUT expression replaced
by a raw-bin load. This exposes missing pixels that clipping or byte rounding
could conceal. An independent CPU oracle repeatedly folds signed positions at
image boundaries instead of using shader modulo or lane assignment.

The proof compares **3,297,792 raw histogram bins across 114 tile layouts**:
widths around 1/2/4/16/32/64 lane partitions, 255/256/257-pixel area boundaries,
one-row and one-column images, odd dimensions, mixed/all-hot bins, grids 1x1,
3x5 and 17x19 including grids larger than the image. A single workgroup handles
all tiles for each layout; destination words start at u32::MAX. This exercises
histogram clearing, repeated reflected padding and workgroup tile reuse.

The complete GPU feature test suite passes **112 tests**: 88 unit, 4 CDF,
5 CLAHE, 5 equalization, 3 histogram and 7 timing tests; three explicit
performance diagnostics are ignored by default. Existing CLAHE coverage checks
CPU byte parity for odd grids, ROI/strides, all-hot/extreme bytes, clipping and
half-rounding, RGB-to-gray resident chains, metadata, retained outputs, stale
scratch, runtime moves, concurrency and three-pass profiling. Existing point
GPU proofs also pass, including the prior radix changes.

All-feature/all-target native Clippy with warnings denied, combined image/AoSoA
WASM library Clippy with warnings denied, formatting and diff whitespace checks
pass. WASM is a compile check; browser execution of these shaders is unverified.
Logs:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-regressions.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-clippy.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-wasm.log`

## Native paired measurements

The diagnostic in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/clahe_lookup_benchmark.rs`
compares the frozen current flat shader in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/clahe_lookup_flat.wgsl`
to production traversal. The older PR #131 serial diagnostic remains available;
its timing is not used to attribute this change's gain.

Hardware: local GPU, Vulkan backend, driver 596.36, Intel
Core i7-9750H (6 cores/12 logical CPUs), Windows, Rust/Cargo 1.98.1, wgpu 24.0.5.
Each fixture has four warmups and 40 alternating paired rounds on the same
runtime, resident source, uniform parameters, layout and workgroup dispatch,
with independent LUT destinations. Both passes run in one ordered encoder.
Timestamp collection is outside measured passes and reads 32 query bytes per
pair. Final LUTs from both variants match independent CPU clipping and rounding.
The full GPU regression completed before this diagnostic; user processes were
untouched.

The 72 fixtures cover 12 image/grid shapes, mixed and all-hot inputs, and clip
limits 0, 2 and 40. Medians and full observed ranges, all 40 paired samples,
median paired flat/rows ratios and 95% bootstrap intervals (10,000 draws,
seed 20261011) are retained in the local JSON. Ratios above one favor the new
traversal. These measure the **lookup pass**, excluding geometry/application,
uploads, host allocation/encoding, readback and synchronization outside the
pass. They are not full-image or end-to-end frame speedups, nor cross-adapter
performance guarantees.

An identical second sweep checks the initial sweep's large variation. At clip 2,
the repeat reports paired medians of 1.420/1.437 for VGA 8x8 mixed/hot,
2.066/2.008 for 4K 8x8, and 1.642/1.633 for 4K 32x32. The first 4K 8x8 mixed
sweep had a much higher median ratio of 4.477 [3.081,4.512]; that larger figure
is not presented as a repeatable speedup. The tables/JSON for both sweeps are
retained. The table below is the **repeat sweep**, using the identical source
and fixture order, with four new warmups and 40 new pairs per fixture.

The short-tile fix restores the 1027x1 repeat medians to 1.095/1.043 at clip 2;
some single-axis confidence intervals include parity. Tiny 17x19 images are
near parity, with small measured regressions in some conditions. There is no
claim of a universal no-regression result. Absolute clocks and times vary, and
the bootstrap intervals describe these paired observations, not future workload
or adapter guarantees.

| Image | Grid | Distribution | Clip | Flat median (us) | Rows median (us) | Paired flat/rows median [95% CI] |
|---|---|---|---:|---:|---:|---:|
| 17x19 | 8x8 | mixed | 0 | 11.392 | 11.424 | 0.999 [0.919, 1.094] |
| 17x19 | 8x8 | mixed | 2 | 11.312 | 11.552 | 0.988 [0.905, 1.077] |
| 17x19 | 8x8 | mixed | 40 | 11.312 | 11.552 | 0.963 [0.920, 1.041] |
| 17x19 | 8x8 | hot | 0 | 11.440 | 11.360 | 1.006 [0.916, 1.090] |
| 17x19 | 8x8 | hot | 2 | 11.792 | 11.472 | 1.027 [0.921, 1.098] |
| 17x19 | 8x8 | hot | 40 | 11.536 | 11.328 | 1.000 [0.944, 1.098] |
| 640x480 | 8x8 | mixed | 0 | 27.888 | 19.792 | 1.415 [1.322, 1.496] |
| 640x480 | 8x8 | mixed | 2 | 28.320 | 19.808 | 1.420 [1.331, 1.486] |
| 640x480 | 8x8 | mixed | 40 | 27.856 | 19.552 | 1.422 [1.345, 1.488] |
| 640x480 | 8x8 | hot | 0 | 28.448 | 20.352 | 1.392 [1.329, 1.468] |
| 640x480 | 8x8 | hot | 2 | 29.024 | 20.432 | 1.437 [1.332, 1.490] |
| 640x480 | 8x8 | hot | 40 | 28.960 | 20.336 | 1.435 [1.339, 1.487] |
| 640x480 | 64x64 | mixed | 0 | 352.224 | 344.480 | 1.023 [1.021, 1.024] |
| 640x480 | 64x64 | mixed | 2 | 365.184 | 356.592 | 1.024 [1.023, 1.026] |
| 640x480 | 64x64 | mixed | 40 | 352.832 | 344.624 | 1.025 [1.022, 1.026] |
| 640x480 | 64x64 | hot | 0 | 349.408 | 341.520 | 1.024 [1.020, 1.025] |
| 640x480 | 64x64 | hot | 2 | 367.072 | 358.080 | 1.025 [1.022, 1.027] |
| 640x480 | 64x64 | hot | 40 | 364.496 | 355.936 | 1.025 [1.023, 1.028] |
| 3840x2160 | 8x8 | mixed | 0 | 522.080 | 253.312 | 2.070 [2.064, 2.074] |
| 3840x2160 | 8x8 | mixed | 2 | 520.624 | 251.584 | 2.066 [2.060, 2.071] |
| 3840x2160 | 8x8 | mixed | 40 | 511.984 | 248.432 | 2.058 [2.050, 2.062] |
| 3840x2160 | 8x8 | hot | 0 | 515.984 | 255.456 | 2.014 [2.011, 2.020] |
| 3840x2160 | 8x8 | hot | 2 | 511.552 | 254.736 | 2.008 [2.002, 2.013] |
| 3840x2160 | 8x8 | hot | 40 | 511.392 | 254.704 | 2.007 [2.006, 2.012] |
| 3840x2160 | 32x32 | mixed | 0 | 458.832 | 279.504 | 1.642 [1.639, 1.645] |
| 3840x2160 | 32x32 | mixed | 2 | 461.744 | 281.024 | 1.642 [1.641, 1.644] |
| 3840x2160 | 32x32 | mixed | 40 | 458.624 | 279.728 | 1.639 [1.637, 1.644] |
| 3840x2160 | 32x32 | hot | 0 | 470.336 | 287.264 | 1.638 [1.633, 1.640] |
| 3840x2160 | 32x32 | hot | 2 | 472.480 | 289.280 | 1.633 [1.629, 1.638] |
| 3840x2160 | 32x32 | hot | 40 | 473.952 | 289.696 | 1.635 [1.634, 1.639] |
| 1x1027 | 1x1 | mixed | 0 | 10.528 | 9.904 | 1.063 [0.923, 1.197] |
| 1x1027 | 1x1 | mixed | 2 | 10.608 | 9.776 | 1.074 [0.941, 1.182] |
| 1x1027 | 1x1 | mixed | 40 | 10.224 | 9.744 | 1.038 [0.943, 1.161] |
| 1x1027 | 1x1 | hot | 0 | 10.320 | 10.160 | 1.012 [0.920, 1.140] |
| 1x1027 | 1x1 | hot | 2 | 10.320 | 9.968 | 1.039 [0.932, 1.147] |
| 1x1027 | 1x1 | hot | 40 | 10.448 | 9.888 | 1.069 [0.989, 1.159] |
| 1027x1 | 1x1 | mixed | 0 | 10.160 | 9.440 | 1.071 [0.997, 1.132] |
| 1027x1 | 1x1 | mixed | 2 | 10.464 | 9.568 | 1.095 [0.983, 1.202] |
| 1027x1 | 1x1 | mixed | 40 | 10.320 | 10.160 | 1.035 [0.956, 1.155] |
| 1027x1 | 1x1 | hot | 0 | 10.416 | 9.728 | 1.065 [0.953, 1.175] |
| 1027x1 | 1x1 | hot | 2 | 10.192 | 9.616 | 1.043 [0.951, 1.141] |
| 1027x1 | 1x1 | hot | 40 | 10.272 | 9.568 | 1.082 [1.002, 1.114] |
| 1027x2 | 1x1 | mixed | 0 | 12.976 | 10.608 | 1.237 [1.094, 1.354] |
| 1027x2 | 1x1 | mixed | 2 | 13.040 | 10.640 | 1.223 [1.123, 1.340] |
| 1027x2 | 1x1 | mixed | 40 | 13.152 | 10.736 | 1.201 [1.108, 1.350] |
| 1027x2 | 1x1 | hot | 0 | 13.584 | 11.152 | 1.234 [1.109, 1.347] |
| 1027x2 | 1x1 | hot | 2 | 13.392 | 10.656 | 1.253 [1.143, 1.338] |
| 1027x2 | 1x1 | hot | 40 | 13.392 | 10.896 | 1.218 [1.121, 1.363] |
| 1027x3 | 1x1 | mixed | 0 | 16.080 | 13.008 | 1.226 [1.120, 1.301] |
| 1027x3 | 1x1 | mixed | 2 | 15.744 | 12.928 | 1.234 [1.107, 1.320] |
| 1027x3 | 1x1 | mixed | 40 | 15.520 | 12.944 | 1.214 [1.113, 1.316] |
| 1027x3 | 1x1 | hot | 0 | 16.112 | 13.232 | 1.216 [1.161, 1.282] |
| 1027x3 | 1x1 | hot | 2 | 16.544 | 13.456 | 1.193 [1.110, 1.296] |
| 1027x3 | 1x1 | hot | 40 | 40.032 | 33.184 | 1.214 [1.102, 1.298] |
| 1027x7 | 1x1 | mixed | 0 | 62.240 | 42.848 | 1.446 [1.377, 1.529] |
| 1027x7 | 1x1 | mixed | 2 | 62.384 | 43.216 | 1.438 [1.367, 1.534] |
| 1027x7 | 1x1 | mixed | 40 | 67.056 | 46.544 | 1.441 [1.377, 1.512] |
| 1027x7 | 1x1 | hot | 0 | 68.928 | 49.072 | 1.408 [1.356, 1.460] |
| 1027x7 | 1x1 | hot | 2 | 68.848 | 49.152 | 1.398 [1.343, 1.436] |
| 1027x7 | 1x1 | hot | 40 | 70.304 | 49.552 | 1.408 [1.348, 1.495] |
| 257x259 | 3x5 | mixed | 0 | 54.064 | 41.472 | 1.309 [1.228, 1.385] |
| 257x259 | 3x5 | mixed | 2 | 53.936 | 42.624 | 1.265 [1.196, 1.354] |
| 257x259 | 3x5 | mixed | 40 | 55.984 | 42.656 | 1.315 [1.227, 1.374] |
| 257x259 | 3x5 | hot | 0 | 58.448 | 45.088 | 1.305 [1.217, 1.347] |
| 257x259 | 3x5 | hot | 2 | 57.584 | 44.960 | 1.283 [1.213, 1.362] |
| 257x259 | 3x5 | hot | 40 | 59.360 | 47.120 | 1.256 [1.203, 1.353] |
| 1025x1027 | 13x11 | mixed | 0 | 232.544 | 190.768 | 1.221 [1.194, 1.234] |
| 1025x1027 | 13x11 | mixed | 2 | 232.912 | 191.008 | 1.222 [1.206, 1.232] |
| 1025x1027 | 13x11 | mixed | 40 | 236.176 | 194.512 | 1.216 [1.197, 1.241] |
| 1025x1027 | 13x11 | hot | 0 | 242.192 | 198.992 | 1.221 [1.198, 1.244] |
| 1025x1027 | 13x11 | hot | 2 | 243.696 | 197.904 | 1.235 [1.203, 1.240] |
| 1025x1027 | 13x11 | hot | 40 | 305.104 | 278.688 | 1.103 [1.085, 1.120] |

## Rejected prototype

The first prototype always capped larger tiles at 32 columns. It improved the
VGA/4K tile pass but slowed 1027x1 images to paired ratios around 0.60-0.65,
because only 32 lanes processed that single row. The final shape assignment
increases columns for short tiles instead. Its frozen source and 54-fixture
measurement are kept locally for review:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-prototype.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-short-tiles-rejected.log`

## Reproduction and delivery

From `C:/Users/rsasa/Workspace/SpatialRust`, with Rust tools on PATH:

```powershell
cargo test -p spatialrust-gpu --all-features --lib --tests -- --test-threads=1
cargo test -p spatialrust-gpu --all-features --release --lib paired_flat_rows_clahe_lookup_timestamps -- --ignored --nocapture
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --no-default-features --features gpu-wgpu,gpu-image,gpu-aoso-staging --target wasm32-unknown-unknown --lib -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Measurements and aggregation receipts:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-paired.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-paired-estimates.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-paired-table.md`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-paired-repeat.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-paired-repeat-estimates.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-paired-repeat-table.md`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/summarize-clahe-rows.py`

The change is delivered as one focused commit and PR, stacked on frozen PR #133
source `ade339d037d5f0b83126ac941197e0ec772a1286` while that dependency remains
open, or directly on main after its merge. The dependency-aware guard at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-ci-merge.ps1`
freezes both heads and requires all 114 named checks to be SUCCESS, no pending
or failed checks and MERGEABLE/CLEAN before merging. It retargets to main only
after the frozen dependency merges, preserves user-edited PR text and local
branches, and never force-pushes. The guard's all-green, ten rejection cases and
retarget tests plus a live dry run are performed before background monitoring.
Its receipt is
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-clahe-rows-ci-merge.log`.

PR #132 merged at `6f565b54acbd8928dcd22416db2bd3b5f81c7b91`; PR #133 is
retargeted to main and continues its existing frozen-head CI guard. This slice
does not alter the pending source heads or unrelated PRs.
