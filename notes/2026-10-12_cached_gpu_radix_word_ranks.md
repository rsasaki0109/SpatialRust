# Cached GPU voxel radix word ranks

## Implementation

For more than 2,048 input workgroups (524,288 points), the existing cached
histogram computes each digit's eight mask-word population counts. It now also
writes exclusive word prefixes at `word * 16 + digit`. After the existing
cached-path barrier, each point adds its word prefix to the population count of
preceding bits within its own word. This replaces up to seven repeated mask-word
loads/popcounts per point. The sixteen digit lanes overwrite all 128 prefixes,
including empty digits and partial final groups. Stable ranks remain in 0..=255.

The cache adds 512 workgroup bytes: the histogram uses 1,088 bytes, below the
existing prefix kernel's 2,052-byte requirement. It adds no barrier, global
scratch, allocation, dispatch, dependency, unsafe code or public API. The small
specialization retains the original local-rank loop. The cutoff, lazy pipeline
selection, prefix/scatter code, host executor, signed x/y/z ordering, tie order,
24 digits/72 passes and profiling/explicit-transfer boundaries are retained.
There is no host repair in the raw sort comparisons.

Production shader:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_radix.wgsl`.
Native correctness and measurement sources:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_rank_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_rank_benchmark.rs`

## Correctness and build evidence

Two new native tests compare both legacy/cached specializations against the
frozen parent and an independent CPU oracle. The intermediate test redirties
histogram, subtotal and rank buffers before every dispatch and checks
**8,235,972 exact u32 words**. Fifteen sizes cover singleton, 32-bit mask-word
boundaries, workgroup boundaries and odd tails through 65,537 points; three
distributions cover mixed signed/extreme/tied keys, coincident keys and reverse
structured keys. Every axis is checked at shifts 0, 4 and 28. Legacy subtotal
buffers must remain untouched; cached subtotal words must match independent
earlier-digit counts.

Complete raw device sorts compare the frozen/current outputs against signed
CPU tuple order and original point indices for twelve size/distribution cases:
257, 524,288, 524,289 and 1,000,003 points. Existing radix tests additionally
exercise maximum u32 prefix mass, chunk boundaries, stale pooled scratch,
simultaneously live outputs, scratch reuse and concurrent lazy initialization.
The facade test crosses the selection cutoff, preserves all tied point indices
and checks the existing four-byte construction readback before explicit metadata
readback. Its source is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust/tests/gpu_radix_word_rank_smoke.rs`.

The frozen shader at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/voxel_radix_ranks_previous.wgsl`
matches parent source `43d31daaf32eb14d88a90fd623b3c14585aee52d` after line-ending
normalization. The native `previous_sort` executor copies that parent's host
function, changing only its name and injecting prepared pipelines instead of
looking them up. Bind-group layout access follows the runtime. These specific
substitutions were verified against the parent; production host/cache code and
the shader's prefix/scatter suffix remain unchanged.

Native Windows, Rust/Cargo 1.98.1 and wgpu 24: **165 GPU tests pass** (138 unit,
27 integration), zero failures and fourteen ignored diagnostics. The two new
tests also pass with minimal `gpu-wgpu`. Six GPU segmentation tests, two
attribute-bearing MVP CPU/GPU voxel comparisons and the facade smoke pass.
Strict default/all-feature all-target GPU Clippy and facade Clippy pass. Three
WASM compilation checks pass: minimal GPU, GPU image/AoSoA, and minimal facade
GPU. These establish compilation, not browser execution or browser performance.
Formatting and diff checks pass.

```text
cargo test -p spatialrust-gpu --all-features --tests -- --test-threads=1 --nocapture
cargo test -p spatialrust-gpu --no-default-features --features gpu-wgpu --lib rank_tests -- --test-threads=1 --nocapture
cargo test -p spatialrust --no-default-features --features gpu-wgpu --test gpu_radix_word_rank_smoke -- --test-threads=1
cargo test -p spatialrust-segmentation --no-default-features --features segment-euclidean-gpu --lib
cargo test -p spatialrust --features mvp,pipeline-mvp-gpu --test mvp_pipeline gpu_voxel_matches_cpu -- --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --all-targets -- -D warnings
cargo clippy -p spatialrust --no-default-features --features gpu-wgpu --test gpu_radix_word_rank_smoke -- -D warnings
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu,gpu-image,gpu-aoso-staging
cargo check -p spatialrust --target wasm32-unknown-unknown --no-default-features --features gpu-wgpu
```

## Paired native completion measurements

Release measurements use Vulkan on a discrete GPU with timestamp queries, 256-lane radix workgroups and independent full-output checks. Each
condition has three warmup pairs followed by twenty alternating old/new pairs.
Five sizes and three distributions include controls below/at the cutoff and
selected work immediately above it, through 2,000,003 points. Own test/build
processes completed before the final repeated measurements.

The wall timer covers the private complete sort call: scratch acquisition,
uniform/bind-group creation, command encoding, submission and device wait.
Input upload and its completion, pipeline cold setup, profiler creation,
timestamp collection, full output readback/checks and output drop are outside the
timer. The frozen executor receives prepared pipelines; current execution uses
the actual production cache lookup. Timestamp sums cover the 24 histogram,
prefix and scatter passes separately; collection verifies 72 stages and 1,152
query bytes. Every warmup and measured raw output matches the CPU oracle.
This measures sorting, not the entire voxel pipeline or an upload-inclusive
CPU/GPU placement boundary.

Ratios below are medians of the twenty paired old/new completion ratios; values
above one favor the cache. Intervals are percentile bootstrap 95% intervals of
that median (10,000 resamples, seed 20261012). They describe within-survey
variation, not portability or between-run uncertainty. Wall times are separate
medians, so their quotient can differ from the paired ratio.

First final survey:

| Points | Pattern | Old/new histogram us | Old/new wall ms | Paired ratio (95% CI) |
| ---: | --- | ---: | ---: | ---: |
| 65,537 | mixed | 248.96/249.20 | 2.632/2.693 | 0.999 (0.970-1.016) |
| 65,537 | coincident | 224.51/225.10 | 2.137/2.167 | 1.001 (0.983-1.017) |
| 65,537 | structured | 227.17/228.05 | 3.224/3.185 | 0.999 (0.957-1.073) |
| 524,288 | mixed | 1381.71/1387.25 | 8.772/8.748 | 1.002 (0.973-1.006) |
| 524,288 | coincident | 1134.00/1139.89 | 6.496/6.661 | 0.984 (0.952-1.011) |
| 524,288 | structured | 1323.22/1330.02 | 7.385/7.534 | 0.988 (0.970-1.011) |
| 524,289 | mixed | 1882.53/1673.12 | 9.547/9.360 | 1.010 (0.976-1.031) |
| 524,289 | coincident | 1726.51/1775.89 | 6.775/6.634 | 1.002 (0.989-1.028) |
| 524,289 | structured | 2038.46/2044.67 | 8.441/8.282 | 1.011 (0.998-1.039) |
| 1,000,003 | mixed | 3552.90/3096.82 | 16.291/15.687 | 1.030 (1.013-1.043) |
| 1,000,003 | coincident | 2372.48/2438.14 | 9.939/9.998 | 0.995 (0.957-1.001) |
| 1,000,003 | structured | 3029.41/3007.92 | 11.839/11.703 | 1.010 (0.993-1.027) |
| 2,000,003 | mixed | 4709.68/4143.30 | 21.420/20.912 | 1.027 (1.011-1.034) |
| 2,000,003 | coincident | 3969.38/4061.73 | 16.577/16.713 | 0.990 (0.981-1.013) |
| 2,000,003 | structured | 4844.32/4800.21 | 18.898/18.763 | 0.998 (0.993-1.010) |

Immediate repeat:

| Points | Pattern | Old/new histogram us | Old/new wall ms | Paired ratio (95% CI) |
| ---: | --- | ---: | ---: | ---: |
| 65,537 | mixed | 246.74/248.38 | 2.375/2.388 | 0.997 (0.986-1.007) |
| 65,537 | coincident | 222.59/223.20 | 2.157/2.161 | 0.999 (0.977-1.021) |
| 65,537 | structured | 225.18/226.66 | 2.980/3.195 | 0.943 (0.802-1.048) |
| 524,288 | mixed | 1372.00/1374.80 | 8.297/8.745 | 0.996 (0.960-1.008) |
| 524,288 | coincident | 1129.01/1133.26 | 5.922/6.016 | 0.995 (0.987-0.999) |
| 524,288 | structured | 1153.76/1158.59 | 6.733/6.942 | 0.983 (0.971-1.008) |
| 524,289 | mixed | 1513.39/1348.50 | 8.363/8.238 | 1.046 (1.012-1.074) |
| 524,289 | coincident | 1630.96/1664.54 | 6.277/6.193 | 1.014 (0.998-1.036) |
| 524,289 | structured | 1761.95/1759.54 | 7.176/7.231 | 0.995 (0.961-1.029) |
| 1,000,003 | mixed | 3416.99/3009.30 | 15.632/15.513 | 1.011 (0.997-1.037) |
| 1,000,003 | coincident | 2767.46/2858.46 | 10.632/10.723 | 0.997 (0.985-1.014) |
| 1,000,003 | structured | 3464.94/3448.40 | 12.228/12.263 | 1.017 (1.004-1.026) |
| 2,000,003 | mixed | 4668.06/4062.59 | 20.900/20.261 | 1.036 (1.021-1.042) |
| 2,000,003 | coincident | 3958.85/4047.73 | 16.517/16.633 | 0.996 (0.980-1.003) |
| 2,000,003 | structured | 4545.89/4504.93 | 18.577/18.569 | 0.990 (0.979-1.003) |

The selected mixed-input histogram stage improves by about 11-13% in both final
surveys. At 2,000,003 mixed points, completed-sort ratios are 1.027 and 1.036,
with both intervals above one. The 524,289 mixed case has ratios 1.010 and 1.046;
the first interval includes one. The million-point mixed case has ratios 1.030
and 1.011; the repeat interval includes one. Whole-sort gains are modest and
variable even when the histogram stage improves.

Coincident/structured selected inputs do not establish a consistent completed
sort improvement. Coincident histogram times increase about 2-3%; these adverse
stage measurements are retained. An initial cache-only survey also measured
0.970 (0.953-1.007) at 524,289 coincident points. No distribution-dependent
shortcut is adopted. Below/at-cutoff controls retain all measured variation,
including repeat ratios 0.943 (0.802-1.048) for 65,537 structured points and 0.995
(0.987-0.999) for 524,288 coincident points. The latter small adverse difference
is not evidence of a portable regression or improvement. The legacy source loop
is retained, but identical measured latency is not promised. No universal speedup,
CPU placement crossover or end-to-end voxel pipeline gain is claimed.

## Deferred single-digit lane shortcut

A native prototype also sets the rank directly to the input lane when all
valid lanes belong to one digit. Its output passed the same correctness tests,
but an alternating cache-only/cache-plus-shortcut ablation did not establish
additional improvement. Across all nine selected size/distribution conditions,
paired median ratios ranged from 0.988 to 1.005 and every 95% interval included
one. Production therefore adopts only the word-prefix cache. The prototype is
retained solely as a native fixture at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/voxel_radix_lane_shortcut.wgsl`.

Native reproduction (unset `SPATIALRUST_RADIX_RANK_COMPARE` for parent/current;
set it to `cache-only` only for the deferred shortcut ablation):

```text
cargo test -p spatialrust-gpu --release --no-default-features --features gpu-wgpu --lib paired_previous_word_rank_sorts -- --ignored --nocapture --test-threads=1
```

All full paired samples and validation logs remain private under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-radix-word-ranks-*`.
Published measurements describe the backend and workload without an adapter
model name. Delivery uses the existing 114 named successful CI checks and an
exact source/dependency commit merge guard; failed, pending, missing or skipped
required checks do not authorize merging.
