# Resident point-to-plane ICP equations — 2026-10-13

This slice adds `PointToPlaneIcp::align_with_gpu_reduction` under the optional
`register-icp-point-to-plane-gpu` feature. Original source positions, target
positions/tree and raw target normals remain resident across evaluations. GPU
passes transform originals, search for unbounded nearest targets, apply the
inclusive Euclidean distance gate, prepare plane rows and reduce Hessian,
gradient and raw-normal plane squared fitness. CPU tree construction, the
normalized 6x6 solve, existing finite pose composition and stopping remain.
Ordinary CPU point-to-plane ICP is unchanged.

## Math and precision contract

For transformed source `p`, matched target `q` and its raw normal `n`, CPU uses
`J = [p cross n | n]` and `r = (p-q) dot n`. Let `a` be the actual
device-transformed original source bounding-box midpoint, `s` the larger source
or target bounding-box diagonal, `d` the configured match distance, and `b`
the largest absolute component across all target normals (one if all zero).
GPU uses `p'=(p-a)/s`, `n'=n/b`, `r'=((p-q)/d) dot n'` and
`J'=[p' cross n' | n']`. One common normal scale preserves relative length
weights, signs and zero rows; normals are never normalized individually.

GPU reduces the 21 upper Hessian terms, six gradient terms and plane-error sum
with compensated two-f32 accumulation. CPU decodes sums in f64 and solves
`H*u=-g`, then converts `omega=u[0..3]*d/s`, `local_t=u[3..6]*d` and
`world_t=local_t-omega cross a`. This is a linearized change of basis; the
existing finite axis-angle delta and left composition remain. Physical plane
error is restored in f64 using `sum(r'^2)*d^2*b^2` and divided by the number of
gated pairs, including zero-normal pairs as in CPU. This is raw-normal plane
error; it is not Euclidean or per-normal unit distance.

The explicit f32 path can change transformed coordinates, correspondence ties,
distance gates, pose updates, fitness and convergence. It does not promise
bitwise CPU identity. Degenerate normals/surfaces can yield singular equations;
the error propagates. Extreme relative normal weights can lose contributions
in f32 products even when upload is representable.

## Residency, ownership and transfers

`GpuPointToPlaneInput::upload` accepts original source positions, a borrowed CPU
target KD-tree and original-order raw target normals. `equations_resident`
submits five ordered passes without host waits or payload copies. Private
scratch holds 16*S transformed queries, 8*S matches, 32*S prepared rows and
240*G partial bytes, with `G=min(ceil(S/64),256)`. Leases survive submission;
the 256-byte output is independently owned and survives input drop/pool reuse.
The caller explicitly invokes `download` to copy, wait and validate.

Initial payload upload is **16*S + 48*T + 32*M bytes**, where `M` is the target
tree node count. Each nonempty evaluation downloads **256 bytes**. A completed
registration with `I` updates evaluates `I+1` times: initial equations and
post-update equations reused by the next solve/final fitness. No transformed
points or matches are read back and no iteration uploads payload again. An
empty primitive source has zero readback; the high-level alignment rejects
empty clouds. Receipts count payload copies; 96-byte control uniforms and
device scratch writes are excluded consistently with existing GPU APIs.

Resident layout: four u32 status/count/fitness-count/reserved words, actual
anchor as vec4<f32> (w=0), then 28 compensated vec2<f32> sums. Nonzero status
invalidates all payload, which is zeroed on device before exposure. Status bits
1/2/4 flag transform, match or nonfinite arithmetic. GPU consumers must inspect
status before using equations. Decode verifies counts, finite sums/anchor,
nonnegative diagonals and plane error. Mismatched runtimes reject.

Configuration requires at least one iteration, at least six configured
correspondences, finite nonnegative stopping thresholds, a finite unit pose
(quaternion norm tolerance 1e-4), and positive finite match distance whose f32
square is finite/nonzero. Positions and normals must be finite. Nonzero cloud
extent squared and the common normal scale must be normal representable f32;
nonzero normal components must remain normal f32 after common scaling. Counts,
byte products, storage and dispatch limits are checked before allocations.
Required compute limits include eight storage buffers, nine bindings,
96 uniform bytes, 64 invocations and 1024 workgroup bytes. Errors propagate
without automatic CPU fallback.

## Correctness evidence

- Independent f64 raw-normal Jacobian/Hessian/gradient/plane-error references
  for source counts 1/3/63/64/65/257/4097, nonidentity poses, translated anchors,
  signed nonunit/zero normals and common normal scales 1e-8/1/1e8. Maximum
  component error divided by `1e-4 + sum(abs(contributions))` was
  **3.5880697583e-5**; tolerance is 5e-5, including CPU/device transform rounding.
- Independent original world-equation solve checks normalized-step conversion.
- Explicit high-level integration compares **99,252 points in 108 conditions**:
  six sizes, coordinate scales 1e-4/1/1e4, unit/weighted normals and fixed,
  transform or fitness stopping. Maximum scale-normalized CPU pose-coordinate
  difference was **9.536743e-7**. Fitness is independently recomputed with raw
  normals and receipt sizes are checked exactly; inputs stay unchanged.
- Inclusive gate, outliers/unequal clouds, empty inputs, malformed schema,
  nonfinite geometry/normals, invalid configuration/pose/scales, singular
  systems, owned outputs, four concurrent callers, runtime mismatches,
  transform overflow/status zeroing and device-limit boundaries are covered.
- The meta-crate smoke test exercises both the primitive and completed
  alignment with weighted normals. Two feature-isolation CI cases cover the
  registration crate and facade.

## Completed-call measurement protocol

Run the exact ignored test alone in release mode:

```text
cargo test -p spatialrust-registration --no-default-features --features register-icp-point-to-plane-gpu --release --test gpu_point_to_plane_reduction_benchmark completed_resident_plane_calls -- --include-ignored --exact --nocapture --test-threads=1
```

Two isolated surveys cover 257/4097/16384/65537 sources and three profiles:
U = unit-normal corner surfaces, W = signed varied raw-normal lengths,
D = duplicated weighted corner points with 17 additional targets. Both source
and target keep the same lattice spacing/layout. Each full call performs three
updates with zero stopping thresholds. A caller-owned runtime exists before
timing; each full call rebuilds its CPU tree/preparation and uploads input.
Pipeline/scratch caches are warmed, with three warmups and 20 cyclically
interleaved CPU/full-GPU/prepared-resident samples per condition. No other
build, test or statistical processing runs during a survey.

CPU timings include existing transforms, matching, f64 equations, solve,
composition and fitness. Full GPU timings include CPU packing/tree/upload,
device execution, every copy/wait, CPU solve/composition and final fitness.
Prepared resident timings measure **one evaluation with completion wait**,
excluding preparation/upload/download/CPU solve; its result is checked after
the timer. All full-call outputs and prepared results are validated after
timing. Logs report backend and device type without an adapter model name.

Measurements used Windows, Rust 1.98.1 release optimization and the Vulkan
backend on a discrete compute adapter. Both compared full paths use the same
compiled search/parallel features. Times below are milliseconds; CPU/GPU ratio
is the paired geometric mean (values above one favor GPU). Brackets show a
within-survey 95% paired log-bootstrap interval from 10,000 resamples, with
seed `20261013 + 12*(survey-1) + condition_index` and sorted endpoints
249/9749. Medians and ratios differ because ratios include every sample.

| Survey | Sources | Profile | CPU full median | GPU full median | Prepared resident median | CPU/GPU ratio [95% CI] |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 257 | U | 0.321 | 4.265 | 0.240 | 0.07 [0.07, 0.07] |
| 1 | 257 | W | 0.324 | 4.344 | 0.246 | 0.07 [0.07, 0.08] |
| 1 | 257 | D | 0.326 | 4.610 | 0.245 | 0.07 [0.07, 0.07] |
| 1 | 4,097 | U | 5.659 | 5.093 | 0.410 | 1.12 [1.09, 1.14] |
| 1 | 4,097 | W | 5.455 | 5.169 | 0.392 | 1.08 [1.03, 1.12] |
| 1 | 4,097 | D | 6.922 | 5.151 | 0.406 | 1.35 [1.29, 1.40] |
| 1 | 16,384 | U | 24.680 | 7.982 | 0.870 | 3.12 [3.02, 3.26] |
| 1 | 16,384 | W | 24.216 | 13.601 | 2.163 | 2.17 [1.87, 2.50] |
| 1 | 16,384 | D | 33.649 | 17.301 | 2.962 | 1.93 [1.87, 2.01] |
| 1 | 65,537 | U | 110.556 | 54.049 | 8.942 | 2.05 [2.01, 2.10] |
| 1 | 65,537 | W | 111.380 | 54.074 | 8.812 | 2.03 [1.96, 2.09] |
| 1 | 65,537 | D | 178.137 | 61.281 | 10.125 | 2.87 [2.64, 3.11] |
| 2 | 257 | U | 0.368 | 5.571 | 0.310 | 0.07 [0.06, 0.07] |
| 2 | 257 | W | 0.399 | 6.712 | 0.332 | 0.06 [0.06, 0.07] |
| 2 | 257 | D | 0.369 | 5.442 | 0.284 | 0.07 [0.06, 0.07] |
| 2 | 4,097 | U | 6.089 | 5.772 | 0.412 | 1.06 [1.02, 1.10] |
| 2 | 4,097 | W | 6.235 | 5.571 | 0.409 | 1.10 [1.05, 1.15] |
| 2 | 4,097 | D | 9.423 | 5.771 | 0.427 | 1.68 [1.59, 1.78] |
| 2 | 16,384 | U | 38.101 | 17.950 | 2.644 | 2.35 [2.15, 2.60] |
| 2 | 16,384 | W | 34.980 | 19.640 | 3.102 | 1.78 [1.68, 1.89] |
| 2 | 16,384 | D | 47.622 | 20.758 | 3.472 | 2.26 [2.12, 2.39] |
| 2 | 65,537 | U | 150.361 | 59.498 | 9.077 | 2.56 [2.43, 2.69] |
| 2 | 65,537 | W | 154.035 | 59.465 | 9.129 | 2.61 [2.47, 2.76] |
| 2 | 65,537 | D | 210.989 | 61.392 | 9.860 | 3.33 [3.18, 3.48] |

All 18 conditions at 4,097 or more points had within-survey confidence intervals
above one: full-call gains ranged **1.06–3.33x**. The 65,537-point conditions
ranged **2.03–3.33x**. All six 257-point conditions were slower on GPU (ratios
0.062–0.075); upload/allocation/copy/wait overhead dominates such small calls.
Survey-to-survey timing variation is substantial, particularly at 16,384
points, so these intervals describe samples within each survey and do not
establish a portable crossover size or gains on other devices/data. The API
stays explicit and CPU remains the ordinary default.

All sampled completed calls and prepared results matched their warmed
reference outputs. Every final evaluation retained all source correspondences.
Maximum CPU/device pose-coordinate difference across measured conditions was
**4.7683716e-7**. CPU and GPU plane fitness were below 1.6e-17; relative fitness
errors at this near-zero scale are not useful accuracy claims.

| Sources | U/W full H2D bytes | D full H2D bytes | Full D2H bytes (all profiles) |
| ---: | ---: | ---: | ---: |
| 257 | 17,504 | 19,280 | 1,024 |
| 4,097 | 278,624 | 280,528 | 1,024 |
| 16,384 | 1,114,080 | 1,115,984 | 1,024 |
| 65,537 | 4,456,544 | 4,458,448 | 1,024 |

These are GPU payload receipts, not a comparison with CPU transfers (CPU has
none). In particular, the fixed readback is four 256-byte evaluations for
three updates, regardless of source count.

## Final verification and delivery

- GPU all-feature library: **192 passed, 20 ignored**; integration targets:
  **27 passed, 3 ignored**. The new four native math/ownership tests passed.
- Registration all-feature library: **35 passed, 1 ignored**. Both new native
  registration integration tests explicitly passed; facade smoke: one passed.
- Both exact release surveys passed with all 24 conditions and 20 samples per
  stage. No ignored benchmark was selected by a broad test filter.
- All-feature/all-target Clippy for GPU/registration/facade, final opt-in facade
  Clippy, CPU-only registration Clippy, WASM optional registration, minimal
  facade and final workspace formatting passed. No new unsafe code, core/math/
  search edits, dependencies or automatic placement rules were introduced.
- A source comparison against parent commit `8f0c648` confirms the entire
  ordinary CPU point-to-plane implementation is unchanged apart from optional
  child-module wiring. Public adapter-name scan reported zero matches.
- Raw 20-sample records and bootstrap results are preserved locally in
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-point-to-plane-reduction-measurements.json`.
  Survey logs are
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-point-to-plane-reduction-bench-1.log`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-point-to-plane-reduction-bench-2.log`.
  These ignored artifacts contain backend/device type, without a local adapter
  model. Repository feature and algorithm CI remain required before merge.
