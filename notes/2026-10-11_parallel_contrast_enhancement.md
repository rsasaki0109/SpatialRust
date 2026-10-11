# Parallel CPU contrast enhancement — 2026-10-11

Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.
Parent: `bca97b98aefe43e775e219b430c46491be4efad7`, PR 124,
`codex/bounded-gpu-temporary-storage`. The CPU analysis implementation is
unchanged between PR 123 and this parent.

## Implementation and contract

`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\src\analysis.rs`
now applies an equalization lookup through the existing bounded row mapper at
100,000 pixels. Its preceding u8 histogram already uses bounded local integer
histograms from PR 123. Constant-image and empty-image behavior is preserved.
Small/one-worker lookup application retains the original scalar loop.

CLAHE has two independently dispatched stages at 1,000,000 input pixels:

- Tile lookup generation groups contiguous tiles into at most the current
  Rayon's pool worker count (also capped by tile count). Every active tile owns
  one 256-element usize histogram; counts, clipping, residual redistribution,
  reflect-101 sampling and cumulative rounding are unchanged.
- Interpolation groups disjoint output rows into at most the current pool's
  worker count (also capped by output row count). A shared horizontal table
  stores the two tile indices and f64 interpolation weight for each column.
  This removes repeated horizontal division/floor/clamping across rows and
  replaces per-pixel view access with validated row slices. The original f64
  arithmetic order and ties-to-even conversion remain unchanged.

The scalar interpolation implementation is retained for one-worker pools,
small images and single-row output. A large single-row image can still dispatch
independent lookup tiles; it cannot dispatch independent output rows. A single
lookup tile can generate its lookup serially and interpolate multiple rows in
parallel. Pixel results and metadata match 1/2/6-worker execution exactly.

Lookup storage remains `tiles_x * tiles_y * 256` bytes, and the output remains
`width * height` bytes. Multiplication and addressable lookup byte size are now
checked before allocation for nonempty images. These checks reject overflow;
they do not impose a new fixed memory budget on user-selected tile grids.
Parallel interpolation adds one width-sized geometry allocation shared by all
workers (24 bytes/column on this 64-bit build: 92,160 bytes at 3840 columns).
No per-worker full-image copy is created. Tile histogram scratch is 2 KiB per
active tile on 64-bit targets, and the number active is bounded by the pool.
Small inputs avoid initializing the global pool through the existing dispatcher.

No public algorithm signatures, default features or core dependencies change.
Rayon stays under the existing optional `imgproc-analysis` feature. GPU transfer
behavior and algorithms are unchanged by this slice.

The source review also found that the legacy parallel tensor query helper in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-search\src\parallel_chunked.rs`
starts one OS thread per tensor chunk. A separate planned ROADMAP row records
bounding these query threads by available workers; this PR remains focused on
CPU contrast enhancement.

## Correctness

Five new tests cover:

- Padded grayscale ROI equalization (1025 × 1027), exact 1/2/6-worker output and
  nondefault metadata.
- Odd ROI dimensions with 8×8, 7×5 and 1×1 CLAHE grids; clipping disabled,
  minimum clipping, ordinary clipping and effectively unlimited clipping;
  exact 1/2/6-worker output and metadata.
- Direct scalar/parallel interpolation parity for 1×1,000,003,
  2×500,003 and 1,000,003×1 views, using four distinct lookup functions.
- Empty/tiny/one-row/one-column scalar cases, grids larger than small images,
  large constant-image behavior, and a known ties-to-even CDF fixture
  `[42,85,128,170,212,255]`.
- Lookup grid and byte-size overflow before allocation, plus invalid clip/grid
  settings.

Full vision tests: 151 unit tests + 13 property tests passed. Analysis-only unit tests: 21 passed. Full and
analysis-only native Clippy pass with warnings denied. Analysis-only WASM and
no-default-feature builds pass with warnings denied. These are browser compile
checks, not browser runtime/performance evidence.

## Measurements and limitations

Windows, Intel Core i7-9750H (6 physical/12 logical CPUs), Rust/Cargo 1.98.1.
Other desktop/Python work continued. CPU load changed sharply during sequential
Criterion runs. No unrelated process was stopped or reprioritized.

The parent release benchmark executable was copied before editing analysis to
`C:\Users\rsasa\Workspace\SpatialRust\target\automation\contrast-before.exe`.
The new executable was rebuilt after the change. Added CLAHE allocation
preflight diagnostics were verified absent from the reference and present in
the new binary, preventing accidental reuse of the old artifact.

Criterion used 10 samples per case (the existing benchmark fixes this count),
3-second warmup and at least 5-second measurement, with separate parent-6,
new-6 and new-1 processes. All input generation was outside timing. Values below
are slope estimates in milliseconds and 95% bootstrap intervals.

| Operation/size | Parent, 6 workers | New, 6 workers | New, 1 worker |
| --- | --- | --- | --- |
| Equalization 1080p | 3.861 [3.748,3.954] | 1.543 [1.518,1.569] | 25.272 [22.532,28.430] |
| Equalization 4K | 16.357 [14.580,18.707] | 52.041 [42.949,59.668] | 37.415 [36.109,38.346] |
| CLAHE 1080p | 39.096 [38.354,40.230] | 44.995 [11.953,70.773] | 139.872 [113.207,162.899] |
| CLAHE 4K | 159.010 [153.583,171.883] | 152.995 [113.671,179.379] | 221.562 [207.865,237.477] |

These separate-process figures cannot establish a reliable before/after speedup
or absence of regression: even unchanged scalar paths and individual sample
latencies varied substantially with system load. The unfavorable and noisy
observations are retained here rather than discarded.

To reduce order/drift effects, the new
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-vision\examples\parallel_contrast.rs`
compares prebuilt local pools within one process. It warms each pool twice,
rotates 1/2/6-worker order over 12 rounds per workload, records completed-call
times, and checks exact pixels/metadata after timing. Setup, input generation,
validation and output destruction are excluded from the timer. This example
uses the same `(index & 255)` input pattern and CLAHE 2.0 / 8×8 settings as the
Criterion benchmark.

| Workload | 1 worker median [range], ms | 2 workers median [range], ms | 6 workers median [range], ms |
| --- | --- | --- | --- |
| Equalization 1080p | 7.053 [6.244,7.962] | 1.658 [1.433,2.214] | 1.122 [1.029,1.326] |
| Equalization 4K | 26.377 [22.065,33.818] | 5.959 [5.079,7.346] | 3.453 [2.691,4.095] |
| CLAHE 1080p | 49.790 [40.232,72.542] | 14.410 [11.231,18.752] | 9.293 [7.226,12.969] |
| CLAHE 4K | 167.156 [150.193,191.742] | 52.394 [42.337,77.755] | 32.562 [27.978,35.365] |

Local median ratios for 1 versus 6 workers are 6.3× / 7.6× for 1080p / 4K
equalization and 5.4× / 5.1× for CLAHE. They include parallel histogram/lookup
work, row access and shared horizontal geometry, not just adding threads to an
identical kernel. The rotating same-process comparison shows useful behavior
under this workload but is not a controlled or portable performance guarantee.
Ranges are observed min/max, not confidence intervals.

## Reproduction

From `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
cargo test -p spatialrust-vision --features full
cargo test -p spatialrust-vision --features imgproc-analysis --lib
cargo clippy -p spatialrust-vision --features full --all-targets -- -D warnings
cargo clippy -p spatialrust-vision --features imgproc-analysis --all-targets -- -D warnings
cargo run -p spatialrust-vision --release --features imgproc-analysis --example parallel_contrast
$env:RAYON_NUM_THREADS = '6'
cargo bench -p spatialrust-vision --features imgproc-analysis --bench analysis -- 'clahe_8x8/(1080p|4k)|equalize_histogram/(1080p|4k)' --warm-up-time 3 --measurement-time 5
$env:RUSTFLAGS = '-Dwarnings'
cargo check -p spatialrust-vision --target wasm32-unknown-unknown --features imgproc-analysis
cargo check -p spatialrust-vision --no-default-features
cargo fmt --all -- --check
```

Ignored local artifacts under
`C:\Users\rsasa\Workspace\SpatialRust\target\automation`:
`contrast-before-criterion.log`, `contrast-after-6-criterion.log`,
`contrast-after-1-criterion.log`, `contrast-criterion-estimates.json`, and
`contrast-interleaved.log`. Criterion baseline data is under
`C:\Users\rsasa\Workspace\SpatialRust\target\criterion`.

CI and the dependency PRs remain the delivery gates.
