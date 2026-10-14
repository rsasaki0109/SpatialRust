# Exact GPU bilateral byte candidates

Base: merged PR #194, `1ac226c624f42ffdaccab8041f1148fee79bc787`.

The resident bilateral shader retains its circular sampling, joint L1 color
distance, exact binary64 coefficient bits and ordered positive binary64
products/sums. Only the final byte conversion changes. The former conversion
used eight integer decisions over 0..=255 for each channel.

The new conversion proposes a byte from the positive normalized high mantissa
words. These are exact 21-bit integers. A normal f32 power-of-two scale is formed
only for numerator/denominator exponent gaps -1..=8; the estimate is rounded and
clamped to 0..=255. It is only a search starting point, not an approximate output.
Exact integer comparisons verify both relevant byte boundaries. Interior
successful proposals need two comparisons; endpoint proposals need one.

If the lower comparison rejects a candidate, only the proven upper bound changes.
If the upper comparison reaches the next byte, only the proven lower bound
changes. The resulting closed interval is searched with the existing exact
midpoint comparison until one integer remains, at most eight further decisions.
Any candidate in 0..=255 is therefore correct, independently of floating estimate
accuracy. Both fused and unfused estimate arithmetic remain safe under this
verification. No precision-dependent tolerance, native shader float64 or CPU
fallback is introduced.

The comparison retains the binary64 division-round-to-nearest-even boundary:
the exact numerator/denominator ratio must reach the midpoint between k+0.5 and
its binary64 predecessor before CPU division can round to that half-byte value.
The half-byte significand is even, and the subsequent CPU byte conversion is
half-up. The special predecessor spacing below 0.5 and exact 128-bit mantissa
products are unchanged. A zero numerator returns zero. Gaps below -1 compare
the first boundary exactly, and gaps above 8 safely clamp to 255; estimate
arithmetic never sees extreme scales.

Public APIs, admitted parameters/normal-or-zero coefficients, runtime binding,
metadata, ownership, resource/coordinate/dispatch validation, lazy runtime cache,
one 16x16 device pass, explicitly allocated scratch and transfer receipts are
unchanged. Driver registers/spills are not measured by the allocation statement.
Preparation remains an explicit image-independent control upload, and filtering
has no image-payload transfer, host wait or host fallback. No dependency, unsafe
boundary or core change is added. Generic CPU bilateral execution is untouched.

## Paths and reference

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_round.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_round_reference.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_candidate_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_candidate_proof.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_candidate_benchmark.rs`

The frozen round source is byte-identical to PR #194's original shader. Its
submission fixture retains the original filter body with the pipeline selected
by the fixture. The other shader text, coefficient preparation and input texture
are shared. Pipelines are compiled before accepted samples: the production path
retains its warm OnceLock lookup, while the frozen fixture receives its prebuilt
pipeline. This fixture is private native test code, not a second public API.

## Verification

Six selected native bilateral tests pass in 13.90 s. The existing 79,822 binary64
division cases still match CPU exactly. A new proof checks 655,023 rows and
1,965,069 byte outputs against CPU hardware division/round/clamp: normal candidate,
frozen eight-decision result and forced-candidate result. All 256 proposed bytes,
including deliberately distant wrong guesses, are tested at every half-byte
threshold with predecessor/current/successor numerators and three denominator
patterns. All normalized exponent gaps -1022..=1023, extreme mantissa words,
large/small denominator scales, zero and extreme numerator/denominator ratios
are included. Zero bypasses the nonzero bracket helper, as in production.

The image matrix compares both current and frozen GPU outputs with CPU across
3,000 conditions / 1,158,000 logical bytes per GPU path: 1..=4 channels, circular
and even diameters, zero/normal/uniform weights, singleton/strip/workgroup tails,
noise/constants/impulses and both borders. The actual production accumulators
retain all 14,960 CPU binary64 values. Existing ROI/metadata, runtime/channel
rejection, cold concurrency, live output/source recycling and resident RGB
bilateral/gray/Sobel/CHW chain checks pass.

All 105 native image regressions pass in 97.62 s, covering the shared binary64
helpers, other image kernels, lazy cache and temporary ownership.

GPU all-feature/all-target Clippy passes with `-D warnings` (15.62 s).
The native public-facade bilateral/gray/Sobel/CHW smoke test passes (3.16 s),
GPU-disabled compilation passes, and the WASM facade with
`gpu-image,vision-imgproc-filter` compiles. The pre-existing filter-only
`LIGHT_PARALLEL_COMPONENTS` dead-code warning remains in that WASM combination;
the changed GPU crate's all-feature/all-target Clippy is clean. Rustfmt and diff
checks pass. No production or benchmark source changed after measurement.

## Measurement boundaries

Rust 1.98.1 release executable; Vulkan, discrete device type, 12 available CPU
workers and a 64 MiB texture retention budget. Both surveys accepted attempt 1,
with zero observed build/test/benchmark overlap and empty native stderr.
Each controller requires three quiet observations five seconds apart, monitors
other named build/benchmark processes between 500 ms child waits, and rejects
the entire attempt if overlap appears. It may cancel only its own benchmark.
This bounded process observation is not a proof of complete machine idleness.
Surveys ran sequentially (11.95 s and 12.82 s); statistics ran afterward.

Each of eight conditions has three warm rounds and twenty samples per route.
Six routes rotate cyclically each round: frozen/current new-plan, frozen/current
reused-plan, frozen/current resident. Images, CPU references, shader compilation
and warmup occur before accepted samples. Output validation and host image drop
are outside timers. There is no timed CPU comparison or cross-run historical
absolute-time comparison; all ratios use frozen/current samples from this run.

- **newplan:** prepare/upload/drop fresh coefficients, image upload, filter,
  readback/unpack, source/output recycling and plan drop are timed. Pipeline
  caches are warm; this is fresh-plan cost, not cold shader compilation.
- **reuse:** identical image boundary with an already prepared plan; coefficient
  preparation/drop are excluded.
- **resident:** pre-uploaded source and prepared plan; filter and wait-idle
  completion are timed. Readback validation and output recycling are afterward.

Each image-boundary call retains 4*pixels physical image upload/output/readback
bytes. Fresh/reused plans have identical coefficient bytes; the table records
them separately. These comparisons isolate final GPU byte conversion rather
than the earlier LUT-versus-generic-CPU algorithm change. The same pass, input,
coefficient table, allocation/recycling and output validation are used.

Paired old/new geometric means and percentile 95% intervals use 10,000 log-scale
bootstrap samples, endpoints 249/9749. Seed is
`2026101407 + 24*(survey-1) + 3*condition_index + route_index`, with routes
newplan/reuse/resident numbered 0/1/2. These are twenty correlated paired
observations per condition, not a hardware-independent latency guarantee.
Profiles 0/2 are deterministic noise/all-255. All sixteen condition records and
48 ratios are retained below, including unfavorable and inconclusive outcomes.

## Interpretation

1080p RGB / diameter 5 new-plan old/new ratios are
1.085661 [1.072016, 1.098474] and 1.076282 [1.045229, 1.112525]. This is about
7-8% less completed time. Reused-plan ratios are 1.084651 and 1.120555; resident
ratios are 1.194082 and 1.194885 (about 16% less time).
Large noisy RGB and RGBA image-boundary gains reproduce in both surveys.
Non-tiny resident intervals remain above one in both surveys.

Tiny images do not establish a resident improvement. Survey 1 tiny reused-plan
ratio is adverse, 0.971648 [0.940495, 0.998384], while its survey 2 interval crosses
one. Several small/grayscale image-boundary intervals cross one; all-255
new-plan improvement is inconclusive in survey 1. There is no blanket faster
claim or automatic CPU/GPU placement change. The earlier small-image CPU
advantage remains relevant. Ratios, not separate medians, determine the paired
inference; medians below are descriptive.

## Complete measurements

| Survey | Shape / channels / diameter / profile | Route | Previous ms | Candidate ms | Paired ratio [95%] |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 65x33 / 1 / 3 / 0 | newplan | 0.989 | 0.974 | 1.029x (1.001-1.064) |
| 1 | 65x33 / 1 / 3 / 0 | reuse | 0.965 | 0.977 | 0.972x (0.940-0.998) |
| 1 | 65x33 / 1 / 3 / 0 | resident | 0.176 | 0.178 | 1.006x (0.969-1.051) |
| 1 | 257x193 / 3 / 5 / 0 | newplan | 1.128 | 1.115 | 1.015x (0.993-1.037) |
| 1 | 257x193 / 3 / 5 / 0 | reuse | 1.088 | 1.071 | 1.033x (1.006-1.062) |
| 1 | 257x193 / 3 / 5 / 0 | resident | 0.464 | 0.389 | 1.177x (1.147-1.213) |
| 1 | 1024x1024 / 1 / 3 / 0 | newplan | 5.858 | 5.643 | 1.032x (1.014-1.050) |
| 1 | 1024x1024 / 1 / 3 / 0 | reuse | 5.786 | 5.509 | 1.046x (1.030-1.061) |
| 1 | 1024x1024 / 1 / 3 / 0 | resident | 1.359 | 1.085 | 1.226x (1.197-1.258) |
| 1 | 1024x1024 / 3 / 5 / 0 | newplan | 9.199 | 8.561 | 1.078x (1.052-1.105) |
| 1 | 1024x1024 / 3 / 5 / 0 | reuse | 9.265 | 8.462 | 1.077x (1.055-1.098) |
| 1 | 1024x1024 / 3 / 5 / 0 | resident | 4.104 | 3.520 | 1.184x (1.171-1.197) |
| 1 | 1024x1024 / 4 / 7 / 0 | newplan | 14.464 | 13.373 | 1.083x (1.069-1.099) |
| 1 | 1024x1024 / 4 / 7 / 0 | reuse | 14.316 | 13.257 | 1.084x (1.073-1.096) |
| 1 | 1024x1024 / 4 / 7 / 0 | resident | 9.547 | 8.443 | 1.129x (1.124-1.134) |
| 1 | 1920x1080 / 3 / 5 / 0 | newplan | 16.214 | 14.780 | 1.086x (1.072-1.098) |
| 1 | 1920x1080 / 3 / 5 / 0 | reuse | 16.074 | 14.904 | 1.085x (1.071-1.099) |
| 1 | 1920x1080 / 3 / 5 / 0 | resident | 7.489 | 6.271 | 1.194x (1.189-1.199) |
| 1 | 1024x1024 / 3 / 5 / 2 | newplan | 8.978 | 8.441 | 1.033x (0.979-1.075) |
| 1 | 1024x1024 / 3 / 5 / 2 | reuse | 8.952 | 8.603 | 1.060x (1.039-1.089) |
| 1 | 1024x1024 / 3 / 5 / 2 | resident | 3.798 | 3.320 | 1.146x (1.135-1.157) |
| 1 | 1024x1024 / 1 / 7 / 0 | newplan | 8.356 | 8.209 | 1.011x (1.000-1.023) |
| 1 | 1024x1024 / 1 / 7 / 0 | reuse | 8.281 | 8.003 | 1.025x (1.012-1.039) |
| 1 | 1024x1024 / 1 / 7 / 0 | resident | 3.511 | 3.369 | 1.075x (1.046-1.125) |
| 2 | 65x33 / 1 / 3 / 0 | newplan | 1.084 | 1.110 | 0.989x (0.957-1.030) |
| 2 | 65x33 / 1 / 3 / 0 | reuse | 1.101 | 1.092 | 0.992x (0.967-1.013) |
| 2 | 65x33 / 1 / 3 / 0 | resident | 0.199 | 0.195 | 1.003x (0.960-1.044) |
| 2 | 257x193 / 3 / 5 / 0 | newplan | 1.247 | 1.182 | 0.990x (0.907-1.083) |
| 2 | 257x193 / 3 / 5 / 0 | reuse | 1.208 | 1.175 | 0.995x (0.921-1.065) |
| 2 | 257x193 / 3 / 5 / 0 | resident | 0.460 | 0.419 | 1.057x (1.002-1.105) |
| 2 | 1024x1024 / 1 / 3 / 0 | newplan | 6.081 | 5.859 | 0.990x (0.914-1.044) |
| 2 | 1024x1024 / 1 / 3 / 0 | reuse | 5.980 | 5.831 | 1.012x (0.979-1.042) |
| 2 | 1024x1024 / 1 / 3 / 0 | resident | 1.313 | 1.068 | 1.201x (1.101-1.279) |
| 2 | 1024x1024 / 3 / 5 / 0 | newplan | 9.267 | 8.591 | 1.079x (1.058-1.100) |
| 2 | 1024x1024 / 3 / 5 / 0 | reuse | 9.319 | 8.544 | 1.087x (1.064-1.112) |
| 2 | 1024x1024 / 3 / 5 / 0 | resident | 3.982 | 3.357 | 1.175x (1.153-1.190) |
| 2 | 1024x1024 / 4 / 7 / 0 | newplan | 14.898 | 13.737 | 1.087x (1.069-1.105) |
| 2 | 1024x1024 / 4 / 7 / 0 | reuse | 14.622 | 13.416 | 1.116x (1.081-1.167) |
| 2 | 1024x1024 / 4 / 7 / 0 | resident | 9.556 | 8.441 | 1.133x (1.115-1.147) |
| 2 | 1920x1080 / 3 / 5 / 0 | newplan | 16.489 | 15.397 | 1.076x (1.045-1.113) |
| 2 | 1920x1080 / 3 / 5 / 0 | reuse | 16.788 | 14.937 | 1.121x (1.082-1.179) |
| 2 | 1920x1080 / 3 / 5 / 0 | resident | 7.525 | 6.310 | 1.195x (1.188-1.202) |
| 2 | 1024x1024 / 3 / 5 / 2 | newplan | 8.779 | 8.271 | 1.071x (1.047-1.094) |
| 2 | 1024x1024 / 3 / 5 / 2 | reuse | 8.727 | 8.181 | 1.057x (1.044-1.070) |
| 2 | 1024x1024 / 3 / 5 / 2 | resident | 3.806 | 3.308 | 1.154x (1.147-1.162) |
| 2 | 1024x1024 / 1 / 7 / 0 | newplan | 8.529 | 8.287 | 1.009x (0.973-1.033) |
| 2 | 1024x1024 / 1 / 7 / 0 | reuse | 8.586 | 8.292 | 1.017x (0.993-1.043) |
| 2 | 1024x1024 / 1 / 7 / 0 | resident | 3.466 | 3.282 | 1.073x (1.048-1.114) |

## Execution receipts

| File | SHA-256 |
| --- | --- |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.rs` | `a7f3755e5673c7ee007cf1f07b13a0c6632c2a00acf611d4c27bec83dbcf832d` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_round.wgsl` | `300d11e4ef60bfe38b2ce6649b0c28db874bbc6888197c7c430387a845d5648f` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_tests.rs` | `de3fcd9f01f445d0ddec3265fec1f9c0529bb0d4ebfba6a3c76dbfa208054fc9` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_candidate_benchmark.rs` | `ad2da128ded6d2aaedd453a2149aa3ce58307cb907b5fc5af69829538b287a8c` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_candidate_proof.rs` | `733612d04ebfaa737270aef76fbe334dd5803ca05c2c3bb23134090cebc950ee` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_candidate_tests.rs` | `42a0c1e1e7e5771f2f3190bab2ff3c2b60859083ab68e3e798680b249ee28301` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral_round_reference.wgsl` | `0332f72147ee8241df8475a3ed2593ef9ea08f92b9ef2fe06da12c4c8b0b253f` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/bilateral.wgsl` | `502000c889e07aba915af5f9d42d2b67fee8558a8aa705cfef49aaaa19584e1c` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/binary64_positive.wgsl` | `896f45d335ad68b4cf2cfdbb684a2341d36de355d85ae5d772fe56868cd234c8` |
| `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/rounding_u32.wgsl` | `550563e68282f955662bb5190c3b1d1e998aec2d0dc63f43f0146cf4cba5a690` |
| `C:/Users/rsasa/Workspace/SpatialRust/target/release/deps/spatialrust_gpu-62f845495446a7e6.exe` | `e121e1f09d91e50748cf411e116b33455b7fc4c5494c14bc06be16126401ba0b` |

Worktree-byte hashes were captured before measurement and rechecked after the
measurements. Local controller/raw-log/statistical files are retained under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`.
