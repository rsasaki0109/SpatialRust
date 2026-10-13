# Resident trimmed ICP moment precision

Date: 2026-10-13. Named slice: accurate two-word GPU moments for trimmed ICP,
with explicit precision selection and the existing CPU pose-solve boundary.

## API and execution

Resident trimmed ICP now defaults to `GpuIcpTrimPrecision::DoubleSingle`.
The explicit alternatives are
`GpuIcpInput::trimmed_moments_resident_with_precision` and
`IcpRegistration::align_with_gpu_trimmed_reduction_with_precision`; choosing
`CompensatedF32` retains the initial route's coordinate/product arithmetic.
The precision enum is exported by GPU, registration and the facade under their
existing GPU feature flags. Both methods validate and propagate errors without
CPU fallback or automatic backend placement. Ordinary CPU ICP stays unchanged.

The new route carries two f32 words through point-minus-anchor offsets,
normalization, centroid division, centering and covariance products. Integer
24x24-bit mantissa multiplication preserves the high/low product, rather than
assuming fused multiply-add behavior. Dependent bitcasts use the private
runtime uniform zero word to preserve rounded f32 intermediate values through
compensated addition/merging. The host always initializes that word to zero.
This prevents algebraic elimination of the correction terms on the measured
backend, without exposing another user parameter or copying data to CPU.

WGSL permits floating-point reassociation, and its `fma` can be implemented as
ordinary multiplication followed by addition. An error-free product cannot
assume IEEE fused behavior. See the primary
[WGSL floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy)
and [fma specification](https://www.w3.org/TR/WGSL/#fma-builtin).

The earlier experiment's low correction words were all zero on this backend.
The preserved-rounding route retains them. At the initial large gated pose,
source/target membership, device queries and selected indices agreed exactly
with scalar matching. The initial maximum mean error fell from
0.00000043506811664428824 to 0.00000000000005684341886080802, and maximum
covariance error from 74.13663649559021 to 0.000025928020477294922 against
independent f64 sums over identical GPU-selected points. These observations
identify moment rounding as an initial source of trajectory divergence; later
f32 transforms/search can still change computed distances or selected sets.

Five reduction pipelines are initialized lazily per precision on one runtime;
the stable distance/index selector is shared. Source/target/tree upload,
six canonical phases plus selection digits, scratch ownership, workgroup sizes,
binding/shared-memory limits, status bits and the 176-byte output layout remain.
Fraction one uses canonical untrimmed kernels for either precision and preserves
their complete diagnostics. The all-gated fitness arithmetic remains canonical;
the new arithmetic applies to retained centroids/covariance, not its denominator.
CPU retains target-tree construction, Horn solving, composition and stopping.

There is no additional pass, point/match payload readback, host wait, repeated
upload, dependency or unsafe boundary. For S source points, T target points and
M target nodes, H2D stays `16*S+32*T+32*M`; I updates require I+1 evaluations and
`176*(I+1)` D2H bytes. Uniform controls remain excluded from payload receipts.
Both precision caches and shared scratch support concurrent callers.

DoubleSingle improves retained-moment accuracy, but its operations and storage
still use f32 words. Subnormal/underflow behavior, extreme dynamic range,
matching near gates/ties and degenerate geometry retain numerical limits. It
does not supply native f64 device execution or guarantee CPU-identical poses
for arbitrary inputs. No change is made to source transform/search arithmetic.

## Validation

Ten native GPU trimming primitive tests passed, including:

- 2,076 independent binary64 comparisons for compensated addition, exact
  two-product, corrected division and two-word products: 519 input rows with
  cancellation/product residuals, signs, integer boundaries and seeded varied
  exponents. Maximum relative error was 0.000000000000012316562842982828;
  the enforced tolerance is 1e-11.
- Three fixed-pose f64 oracles with 262,145 source points. Owned snapshots copy
  queries/matches/selection before scratch recycling, compare thresholds to
  independent sorting, and accumulate selected moments in f64. Queries,
  matches, selected sets and all-gated error are identical between precision
  choices. Across poses, maximum precise mean error was
  0.0000000000004334310688136611 and covariance error
  0.00004225969314575195. The mean limit is 1e-9; covariance uses an absolute
  floor plus 5e-12 times the sum of absolute reference products.
- A frozen complete shader from source commit
  `741efaf4614f0b53651ce45c119f5fd7c5d1e50e` verifies that the explicit f32
  choice retains its earlier shader expressions. Existing fraction/count
  rounding, membership/ordering, all-gated fitness, ownership/concurrency,
  numeric error/overflow and malformed-readback checks also pass.

The large native registration suite checks 27 trajectories: sizes
4097/65537/262145, unique/partially gated/duplicate profiles with fractions
0.5/0.8/0.9, and one/two/three updates. Every precise result meets the unchanged
3e-5 CPU pose-coordinate agreement threshold. Maximum difference is
0.0000076293945, compared with the explicit previous choice's 0.0008154977.
Counts, independent all-gated fitness, iteration limit/nonconvergence and full
receipts agree. Fraction one is exactly canonical, and four callers alternate
both precision choices while retaining each mode's complete expected result.

The existing 486 size/profile/scale/fraction/stopping conditions and three
rotated initial guesses passed after the precision change. Maximum
scale-normalized CPU pose difference remains 0.0000019073486. The old benchmark
now enforces its original pose threshold on the formerly adverse large case
without an exception. Facade smoke tests, all-feature registration library
tests (35 passed, three intentional ignored), Clippy all targets/features,
CPU-only ICP, optional ICP WASM and formatting checks passed.

## Completed-call performance

Two separate sequential native release surveys completed successfully on
Vulkan with a discrete device, without recording the adapter model. Each
survey covers N=257/4097/65537/262145 source points and N+17 target points.
U means unique distances, G partially gated points, and D duplicate/tied
points; their trim fractions are 0.5/0.8/0.9 respectively. Every alignment
performs exactly three updates with zero stopping epsilons and a 0.25 gate.
The same input poses and cloud generation are used for all routes.

Each of six stages receives three completed warmups and 20 paired rounds.
The stage order rotates each round: CPU; legacy GPU correspondences with
CPU trimming/moments; explicit CompensatedF32 resident reduction; default
DoubleSingle resident reduction; prepared F32; prepared DoubleSingle.
Compilation, input generation and independent validation occur outside the
timers. No local builds, tests or statistical analysis ran concurrently
with either survey. Both runs use already warmed pipeline caches; cold
shader compilation and construction of the runtime are excluded.

Complete alignment timers include CPU packing/tree construction, explicit
uploads, GPU work/completion, fixed-size readbacks, CPU Horn solve and
composition, history and final all-gated fitness. Prepared timers use one
already uploaded input and the SAME final DoubleSingle pose for both
precisions; they include submission and wait_idle completion, and exclude
input preparation/upload, output readback and the CPU pose solve. Prepared
outputs are downloaded and checked after timing. Every timed call also
matches its own full warm control result exactly. Counts, independent f64
fitness and the unchanged 3e-5 precise CPU pose threshold are checked for
all profiles, including the previously adverse large gated case.

Times below are medians in milliseconds. Ratios are the geometric mean of
20 paired baseline/DoubleSingle time ratios; values above one mean the
precise route is faster than that baseline. Parenthesized intervals use
10,000 paired log-ratio bootstrap draws (indices 249/9749), seeded with
20261013 + 3*record_index + baseline_index. They describe within-survey
variation only. Scheduling changed absolute times between surveys; these
synthetic, single-backend measurements do not establish universal speedups.

| Survey | Source/target | Profile/trim | CPU ms | Legacy GPU ms | F32 ms | DoubleSingle ms | Prepared F32/DS ms | F32/DS pose error | CPU/DS ratio (95%) | Legacy/DS ratio (95%) | F32/DS ratio (95%) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 257/274 | U/0.5 | 0.285 | 7.998 | 5.553 | 5.384 | 0.702/0.591 | 5.96e-08/0 | 0.05x (0.05-0.06) | 1.49x (1.40-1.60) | 1.00x (0.94-1.06) |
| 1 | 257/274 | G/0.8 | 0.320 | 9.696 | 5.769 | 5.827 | 0.703/0.642 | 2.38e-07/0 | 0.06x (0.05-0.06) | 1.49x (1.37-1.61) | 1.00x (0.94-1.07) |
| 1 | 257/274 | D/0.9 | 0.380 | 8.523 | 5.474 | 5.411 | 0.681/0.612 | 1.19e-07/0 | 0.07x (0.06-0.07) | 1.50x (1.41-1.60) | 0.98x (0.92-1.05) |
| 1 | 4,097/4,114 | U/0.5 | 7.293 | 10.957 | 6.783 | 6.848 | 0.838/1.650 | 0/0 | 0.97x (0.90-1.05) | 1.52x (1.43-1.61) | 0.96x (0.91-1.03) |
| 1 | 4,097/4,114 | G/0.8 | 9.075 | 11.479 | 6.968 | 7.236 | 0.853/0.964 | 9.54e-07/2.38e-07 | 1.21x (1.11-1.32) | 1.55x (1.46-1.64) | 0.96x (0.90-1.03) |
| 1 | 4,097/4,114 | D/0.9 | 9.395 | 11.748 | 7.728 | 7.839 | 0.913/0.964 | 9.54e-07/9.54e-07 | 1.23x (1.13-1.34) | 1.48x (1.39-1.58) | 0.95x (0.90-1.01) |
| 1 | 65,537/65,554 | U/0.5 | 166.437 | 65.097 | 29.881 | 36.442 | 3.127/4.915 | 1.98e-06/0 | 4.55x (4.31-4.85) | 1.79x (1.69-1.93) | 0.80x (0.75-0.86) |
| 1 | 65,537/65,554 | G/0.8 | 174.936 | 63.685 | 32.264 | 38.300 | 3.928/5.179 | 5.25e-06/4.66e-10 | 4.87x (4.57-5.20) | 1.68x (1.59-1.78) | 0.85x (0.80-0.89) |
| 1 | 65,537/65,554 | D/0.9 | 204.321 | 66.742 | 31.049 | 37.248 | 3.127/4.895 | 9.54e-07/1.43e-06 | 5.67x (5.28-6.09) | 1.86x (1.77-1.97) | 0.84x (0.81-0.88) |
| 1 | 262,145/262,162 | U/0.5 | 832.833 | 270.507 | 149.388 | 154.841 | 11.603/16.674 | 7.15e-07/0 | 6.13x (5.14-7.38) | 1.82x (1.58-2.08) | 0.97x (0.89-1.07) |
| 1 | 262,145/262,162 | G/0.8 | 1081.770 | 290.580 | 138.351 | 144.208 | 13.604/15.220 | 0.000815/7.63e-06 | 7.10x (6.25-8.23) | 2.00x (1.78-2.28) | 0.96x (0.84-1.11) |
| 1 | 262,145/262,162 | D/0.9 | 1036.076 | 275.925 | 123.198 | 136.274 | 12.489/14.292 | 5.72e-06/1.91e-06 | 7.59x (6.95-8.32) | 2.03x (1.86-2.24) | 0.90x (0.83-0.97) |
| 2 | 257/274 | U/0.5 | 0.622 | 25.688 | 15.749 | 16.893 | 1.098/1.261 | 5.96e-08/0 | 0.04x (0.03-0.04) | 1.59x (1.32-1.91) | 1.01x (0.92-1.11) |
| 2 | 257/274 | G/0.8 | 0.503 | 13.182 | 8.732 | 9.280 | 1.292/1.355 | 2.38e-07/0 | 0.05x (0.05-0.06) | 1.46x (1.39-1.53) | 0.98x (0.94-1.02) |
| 2 | 257/274 | D/0.9 | 0.566 | 12.493 | 8.976 | 9.110 | 1.440/1.446 | 1.19e-07/0 | 0.06x (0.06-0.06) | 1.39x (1.29-1.50) | 0.97x (0.91-1.04) |
| 2 | 4,097/4,114 | U/0.5 | 11.265 | 16.485 | 11.219 | 13.794 | 2.037/2.606 | 0/0 | 0.78x (0.70-0.86) | 1.21x (1.11-1.35) | 0.83x (0.79-0.87) |
| 2 | 4,097/4,114 | G/0.8 | 15.925 | 27.373 | 15.722 | 17.740 | 3.060/3.792 | 9.54e-07/2.38e-07 | 0.77x (0.72-0.83) | 1.37x (1.30-1.44) | 0.85x (0.80-0.91) |
| 2 | 4,097/4,114 | D/0.9 | 12.697 | 16.876 | 12.075 | 14.239 | 2.274/2.778 | 9.54e-07/9.54e-07 | 0.86x (0.77-0.94) | 1.20x (1.16-1.25) | 0.85x (0.83-0.89) |
| 2 | 65,537/65,554 | U/0.5 | 238.765 | 98.429 | 54.655 | 64.162 | 7.585/9.317 | 1.98e-06/0 | 3.80x (3.56-4.08) | 1.56x (1.48-1.65) | 0.86x (0.84-0.88) |
| 2 | 65,537/65,554 | G/0.8 | 264.383 | 83.826 | 38.227 | 40.462 | 2.684/3.642 | 5.25e-06/4.66e-10 | 5.87x (4.97-6.85) | 1.94x (1.74-2.16) | 0.90x (0.84-0.96) |
| 2 | 65,537/65,554 | D/0.9 | 261.298 | 88.903 | 59.278 | 67.621 | 8.030/8.679 | 9.54e-07/1.43e-06 | 4.28x (3.92-4.67) | 1.47x (1.34-1.61) | 0.88x (0.85-0.91) |
| 2 | 262,145/262,162 | U/0.5 | 604.312 | 208.490 | 103.102 | 109.633 | 11.129/12.643 | 7.15e-07/0 | 5.93x (5.51-6.40) | 2.03x (1.88-2.19) | 0.97x (0.91-1.04) |
| 2 | 262,145/262,162 | G/0.8 | 700.950 | 237.571 | 112.726 | 127.851 | 13.543/15.225 | 0.000815/7.63e-06 | 5.79x (5.17-6.43) | 1.86x (1.74-1.99) | 0.93x (0.89-0.97) |
| 2 | 262,145/262,162 | D/0.9 | 875.522 | 251.212 | 111.059 | 120.424 | 5.507/6.887 | 5.72e-06/1.91e-06 | 7.28x (6.51-8.14) | 2.06x (1.85-2.30) | 0.91x (0.81-1.02) |

For 262,145 points, complete DoubleSingle calls are 5.79-7.59x faster than
CPU and 1.82-2.06x faster than legacy GPU correspondence calls across both
surveys. Precision has a cost: they take 1.03-1.11x the time of the explicit
previous resident F32 route by paired geometric ratios. For 65,537 points,
the CPU gain is 3.80-5.87x, with 1.12-1.25x the previous resident time.
For 4,097 points the CPU/precise ratio ranges from 0.77 to 1.23 and depends
on the profile/survey. At 257 points CPU is 14.7-27.5x faster. No automatic
CPU/GPU choice is introduced. The explicit F32 option leaves that arithmetic
tradeoff to the caller.

The adverse gated 262,145-point pose difference is 0.0008154977 for the
previous arithmetic and 0.0000076293945 for DoubleSingle in both surveys.
The duplicate 65,537-point profile has a slightly larger pose difference
with DoubleSingle (0.0000014305115 versus 0.0000009536743); both meet the
same threshold. Improved moments do not imply a uniformly smaller pose
error for every geometry and selected-set trajectory.

At N=262145/T=262162, three updates require four fixed-size readbacks:
legacy H2D/D2H are 34,604,528/14,680,120 payload bytes; both resident
precisions use 13,633,200/704. Precision adds no payload transfer.

Private raw timing records and derived artifacts remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trimmed-precision-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trimmed-precision-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-icp-trimmed-precision-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-icp-trimmed-precision.py`

Reproduce from `C:/Users/rsasa/Workspace/SpatialRust`: compile with `--no-run`
first, then run this exact test alone twice, sequentially, with no overlapping
local builds or benchmarks:

```text
cargo test -p spatialrust-registration --no-default-features --features register-icp-gpu --release --test gpu_icp_trimmed_precision_benchmark completed_trimmed_icp_precision_calls -- --include-ignored --exact --nocapture --test-threads=1
```

Source references:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/icp_trim_moments.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_math_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_trim_precision_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/icp_trimmed_f32_previous.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/src/icp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_icp_trimmed_precision.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_icp_trimmed_precision_benchmark.rs`
