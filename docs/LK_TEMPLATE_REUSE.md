# Exact LK template reuse (146AV)

CPU sparse LK now lazily saves the previous-image center value and two gradients
for each admitted patch offset. A point and pyramid level have fixed previous
coordinates and immutable f32 pixels, so later iterations reuse the exact f64
values. A cache hit removes five repeated previous-image bilinear samples;
the next-image sample is still computed on every admitted iteration.

Both image bounds are tested every iteration before cache access. An offset
first becoming eligible later is filled then; an offset becoming ineligible is
still excluded. Row-major accumulation, structure-tensor products, updates,
convergence, thresholds, pyramid construction, guesses and statuses retain
their original expressions/order. Rechecking bounds each iteration preserves
changes in the admitted sample set.

Only radii at most five with more than one configured iteration enable reuse.
Other settings retain the original sampling expressions. A call-local optional
array holds at most 121 slots, each three f64 values: 2,904 logical value bytes.
A compile-time assertion limits the complete optional array representation to
4,096 bytes in each native/WASM build. This excludes other call-frame/runtime
memory and does not guarantee process RSS. No new heap allocation or cache
across points, levels, calls or frames is introduced.

Public APIs, dependencies, core, existing owned pyramid reuse, custom scalar
conversion observations, caller ownership and explicit CPU/GPU transfers remain.
This is CPU work reuse in the existing tracker.

## Correctness and qualification

The [source-bound receipt](receipts/2026-10-22_lk_template_reuse.json) records
96 passing vision tests before implementation and 96 afterward,
plus 24 web tests and the no-feature vision build (zero tests).
Six new controls compare the full frozen parent LK module, copied before
implementation from parent merge b3ac5274490a03981f97fa550079727a96ed0eff, with the public API.
They cover u8/u16/f32/f64, odd sizes, several radii on both sides of the reuse
boundary, one/multiple iterations, pyramids, motion near image borders,
nonfinite values, signed zero, duplicate/empty points, strided sources, repeated
calls, input ownership, errors and successful stationary tracks. Coordinates
are checked as exact f64 bits and statuses/errors match. Existing seven robust
pyramid-pair controls, including observable custom conversions, also pass.

The untouched parent module, characterization tests and benchmark driver are
identical between parent and candidate test runs. Source inspection verifies
unchanged preparation, pyramid/sampling helpers and accumulation/update tail.
The independent frozen parent camera executable matches all 126 complete
reports. Actual parent/candidate WASM matches 252 reports and 84 rejected calls
over 42 cases, including reset and state recovery. Existing 92 JS, six asset
and 86 actual WASM/model browser controls pass. Two feature-scoped Clippy checks
deny warnings and workspace formatting passes.

Captures are synthetic; physical phone and iOS Safari qualification remain open.
Exact-head CI, merge and matching public Pages/browser checks follow the local
receipt. This change does not establish metric pose or operational 80% acceptance.

## Interleaved CPU measurements

The prospective protocol is
C:/Users/rsasa/Workspace/SpatialRust/benchmarks/lk-template-reuse-plan.json.
The driver is
C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/examples/lk_template_reuse.rs.
It has no new dependency and can be run with:

```text
cargo run -p spatialrust-vision --release --no-default-features --features odometry --example lk_template_reuse --locked --offline
```

Two fresh native release processes compare the complete frozen parent module
and its already shared-pyramid robust call graph with the candidate in one
executable. Both stationary and translated images use the same fixed inputs.
Each condition has four checked warmups per function followed by four
baseline/candidate/candidate/baseline rounds: eight timed calls per function.
Full coordinate/status/error bit snapshots are compared outside every timer.

The table reports milliseconds per robust call; ratios below 1 favor reuse.
Keep all conditions and samples, including adverse rows. One Windows host
worker is used; affinity/frequency, other host load and allocation effects are
uncontrolled, with no confidence intervals. Stationary inputs may converge on
their first iteration and pay reuse bookkeeping without a cache hit. Empty
points are a preparation-only control and never enter the changed per-point
kernel. The frozen module is local to the benchmark executable while the
candidate calls the library; compiler placement/inlining differences are an
additional uncontrolled factor. Empty-input timing differences cannot be
attributed to template reuse. These observations do not establish
physical-phone, GPU, AI-inference or end-to-end camera speed.

| Image | Points | Motion | Parent 1 ms | New 1 ms | Ratio 1 | Parent 2 ms | New 2 ms | Ratio 2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 192x144 | 0 | stationary | 0.361 | 0.393 | 1.088 | 0.306 | 0.348 | 1.136 |
| 192x144 | 0 | translated | 0.367 | 0.415 | 1.130 | 0.336 | 0.385 | 1.144 |
| 192x144 | 8 | stationary | 0.963 | 1.036 | 1.075 | 0.924 | 1.009 | 1.092 |
| 192x144 | 8 | translated | 9.602 | 2.926 | 0.305 | 9.819 | 3.064 | 0.312 |
| 192x144 | 80 | stationary | 6.184 | 6.169 | 0.998 | 6.965 | 8.733 | 1.254 |
| 192x144 | 80 | translated | 81.583 | 22.837 | 0.280 | 74.665 | 20.511 | 0.275 |
| 256x192 | 0 | stationary | 0.515 | 0.585 | 1.136 | 0.478 | 0.533 | 1.115 |
| 256x192 | 0 | translated | 0.537 | 0.607 | 1.130 | 0.465 | 0.524 | 1.127 |
| 256x192 | 8 | stationary | 1.045 | 1.105 | 1.058 | 0.900 | 0.990 | 1.100 |
| 256x192 | 8 | translated | 7.952 | 2.711 | 0.341 | 8.764 | 2.712 | 0.309 |
| 256x192 | 80 | stationary | 6.234 | 6.636 | 1.065 | 6.377 | 6.204 | 0.973 |
| 256x192 | 80 | translated | 133.824 | 36.503 | 0.273 | 130.929 | 32.903 | 0.251 |
| 320x240 | 0 | stationary | 0.785 | 0.886 | 1.130 | 1.241 | 1.229 | 0.990 |
| 320x240 | 0 | translated | 0.917 | 0.899 | 0.980 | 0.922 | 1.044 | 1.132 |
| 320x240 | 8 | stationary | 2.282 | 2.592 | 1.136 | 1.455 | 1.545 | 1.062 |
| 320x240 | 8 | translated | 17.473 | 4.530 | 0.259 | 17.871 | 6.082 | 0.340 |
| 320x240 | 80 | stationary | 8.067 | 12.271 | 1.521 | 9.703 | 8.816 | 0.909 |
| 320x240 | 80 | translated | 140.645 | 42.444 | 0.302 | 156.546 | 43.727 | 0.279 |

Of 36 paired medians, 17 are lower and
19 higher for the candidate. All observations are retained without
selecting a favorable process or omitting short/empty/stationary workloads.

After observing the empty-point/compiler-layout differences, the already
captured independent native camera outputs were inspected as an exploratory
secondary check. No new timing runs were added. These are separate parent then
candidate processes, one per flavor, with six seven-frame samples per condition.
All 18 conditions and raw samples are in the receipt: 15 lower and
3 higher candidate medians. This was not a prospective primary metric;
process order, host load and frequency remain uncontrolled. It is not pooled
with the primary surveys and does not prove physical-phone/browser speed.

## Bounded storage and remaining work

Offline builds use a fresh E-drive target with a 1GiB logical-file cap,
zero allowed shared-cache growth, two Cargo jobs, one owned command at a time,
five-second samples and a 900-second deadline. The local root contains
870,361,727 logical bytes, including 41,087,603 reused vendor hardlink
bytes. Shared-cache growth is zero. All 5,038 earlier camera/renderer files,
493 scientific JSON, 42 frozen native artifacts, room/reference outputs,
previous raw WASM and the resolved lock remain. No sensor, model, SDK, browser
or toolchain acquisition occurs. Public verification has a 40MiB stream cap.

Calibration/timing with independent motion, full-room accuracy, qualifying
physical Linux hardware and integrated mapping/semantics/viewer/interchange/
release evidence are still required for operational 80% acceptance.
