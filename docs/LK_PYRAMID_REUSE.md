# Forward/backward LK pyramid reuse (146AU)

Robust CPU LK tracking now builds one private owned pair of f32 image pyramids
and borrows it in both directions. For built-in u8/u16/f32/f64 pixels, four
pyramid constructions become two. The one-direction public API uses the same
private preparation and tracking code. Public signatures, dependencies, core,
device transfers, pyramid sampling and per-point numerical operations remain.

PixelComponent is open to custom implementations whose scalar conversion may
have observable effects. A safe TypeId check selects only the four built-in
types; custom components retain the original two public calls and conversion
observations. There is no unsafe code or new trait restriction.

Each call owns its pyramids with no retained cache or caller alias. At 320x320
with three levels, each f32 pyramid contains 537,600 logical payload bytes;
the pair contains 1,075,200. Peak pyramid payload remains two pyramids. Halving
construction work does not promise half the tracker time or resident memory.

## Compatibility evidence

The [source-bound receipt](receipts/2026-10-22_lk_pyramid_reuse.json) records
90 passing vision tests before implementation and 90 afterward,
plus 24 web tests and a no-feature vision build (zero tests). Seven new characterization
tests retain the original two-public-call reference unchanged across both runs.
They compare exact coordinate/error bits and statuses for four pixel types,
odd dimensions, levels 1/2/3/5/32, strided subviews, nonfinite values, signed
zero, repeated/empty points, accepted identity tracks, options/shape/threshold
error priority and an observable custom converter. Source comparison also
checks unchanged point-loop operations, interpolation, pyramid construction
kernels and existing optical-flow tests.

An independent frozen native executable from the previous Rust implementation
matches all 126 complete camera reports over 18 conditions. Its source has no
Rust changes through the parent merge. Actual parent and candidate WASM match
252 complete reports and 84 rejected calls over 42 conditions, including reset
and state recovery. Existing 92 JS controls, six asset checks and 86 actual
WASM/model browser controls pass. Captures are synthetic. Physical-phone and
iOS Safari qualification remain open.

Both feature-scoped Clippy checks deny warnings, and workspace formatting passes.
One launcher failed before child creation because its version-suffix parser
also split the word vision. Preserve that original owner and failure record;
a fresh corrected owner runs all 19 successful guarded local commands.
Exact-head CI, merge and the matching public Pages/browser checks follow the
local receipt.

## Interleaved CPU observations

The prospective protocol is
C:/Users/rsasa/Workspace/SpatialRust/benchmarks/lk-pyramid-reuse-plan.json.
The driver is
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-web/examples/lk_pyramid_reuse.rs.
Run it with:

```text
cargo run -p spatialrust-web --release --no-default-features --features camera-tracking --example lk_pyramid_reuse --locked --offline
```

Two fresh native release processes compare the frozen two-public-call flow
with the candidate robust API in the same executable. Each condition has four
checked warmups per function, then four baseline/candidate/candidate/baseline
rounds: eight timed calls per function. Every timed and warmup result is checked
outside the timer. Inputs, output ownership and result allocations are retained.

The table shows milliseconds per robust call; ratios below 1 favor reuse. Keep
all samples and every condition, including adverse observations, in the receipt.
One Windows host worker is used. CPU affinity/frequency, other host load and
allocation effects are uncontrolled; there are no confidence intervals. These
measurements cover CPU LK, without proving physical-phone, GPU, AI-inference
or end-to-end browser speed.

| Image | Points | Parent 1 ms | New 1 ms | New/parent 1 | Parent 2 ms | New 2 ms | New/parent 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 192x144 | 0 | 0.677 | 0.331 | 0.489 | 0.657 | 0.326 | 0.496 |
| 192x144 | 8 | 10.555 | 10.099 | 0.957 | 10.061 | 10.019 | 0.996 |
| 192x144 | 80 | 77.125 | 76.683 | 0.994 | 73.148 | 76.354 | 1.044 |
| 256x192 | 0 | 1.006 | 0.501 | 0.498 | 1.184 | 0.595 | 0.503 |
| 256x192 | 8 | 8.198 | 7.787 | 0.950 | 9.054 | 8.694 | 0.960 |
| 256x192 | 80 | 93.479 | 96.306 | 1.030 | 124.876 | 118.447 | 0.949 |
| 320x240 | 0 | 1.478 | 0.726 | 0.491 | 1.749 | 0.864 | 0.494 |
| 320x240 | 8 | 12.166 | 11.207 | 0.921 | 14.125 | 13.539 | 0.959 |
| 320x240 | 80 | 111.254 | 117.147 | 1.053 | 106.745 | 97.264 | 0.911 |

Of 18 paired medians, 15 are lower and
3 higher for the candidate. Empty-point calls still prepare the images,
preserving admission/conversion behavior; they isolate preparation overhead
and do not represent a useful tracking workload.

## Storage and remaining qualification

Fresh offline Cargo builds use a dedicated E-drive target with a prospective
1GiB logical-file limit, zero permitted shared-cache growth, two Cargo jobs,
one owned command at a time, five-second samples and a 900-second deadline.
The final local root contains 866,316,622 logical bytes, including
41,087,603 bytes of reused vendor hardlinks. Shared-cache growth is zero.
Preserve all 2,437 earlier camera/renderer files, 493 scientific JSON,
42 frozen native artifacts, room/reference outputs, both previous raw WASM
files and the resolved lock. No sensor, model, SDK, browser or toolchain
acquisition occurs. Public verification streams at most 40MiB.

Operational 80% acceptance still needs calibration/timing and independent
motion, full-room accuracy, qualifying physical Linux hardware and integrated
mapping/semantics/viewer/interchange/release evidence. This CPU preparation
optimization does not close those gates.
