# GPU point-to-plane query/row storage reuse

Date: 2026-10-13. Named slice: reuse transformed source queries after nearest
matching in the resident point-to-plane ICP reduction.

## Execution and memory

The transform pass writes 16-byte query rows. The existing unbounded nearest
search finishes in a separate compute pass before pair preparation. Each
preparation invocation builds a complete local pair, reading only its own
transformed row after the gate, then overwrites that row with the same rotation
Jacobian and plane residual
previously stored in the first half of a separate 32-byte pair. A separate
16-byte row retains the normalized raw normal and contribution flag. Rejected
pairs return a local zero pair. Each invocation unconditionally stores both
halves once, removing the old initial zero store followed by a valid pair store
and avoiding a query read for gated-out pairs. The following reductions
reconstruct the same logical pair from these two bindings.

The five ordered passes and single submission stay unchanged. No workgroup
barrier, copy, clear pass, additional submission, host wait or production payload
transfer is added. Existing pass boundaries order the changed storage accesses;
the backend still manages device resource barriers. Source positions and target
positions/tree/raw normals stay
immutable. Nearest matching, inclusive Euclidean gating, common normal scale,
centered Jacobian, compensated sum order and CPU decode/solve/world-step
conversion retain their arithmetic. Zero-normal pairs retain their gated count
and zero weight. Nonzero status still invalidates and zeros the entire output.

For nonempty source count S, G=min(ceil(S/64),256):

| Private binding | Previous bytes | Reused bytes |
| --- | ---: | ---: |
| Transformed queries / later rotation-residual | 16*S | 16*S |
| Nearest matches | 8*S | 8*S |
| Prepared pairs / now normals-counts | 32*S | 16*S |
| Compensated partials | 240*G | 240*G |
| Total | 56*S + 240*G | 40*S + 240*G |

The saving is exactly **16*S bytes**: 28.57% of the source-sized scratch, and
16,777,232 bytes at 1,048,577 sources. These are requested private working
buffer sizes, excluding resident input, owned output, uniforms, staging,
retained caches and driver overhead; they are not total process VRAM savings.
Four private leases, eight storage bindings, nine total bindings, 96-byte
uniforms, 1024-byte workgroup storage and the 256-byte owned output remain.
The largest source-sized binding falls from 32*S to 16*S. Source/normal counts,
checked byte products, partial/result sizes, invocation and dispatch limits
still reject unsupported work before allocation. Empty source skips dispatch
and has zero readback as before.

Payload receipts remain **16*S + 48*T + 32*M uploaded bytes** and **256 bytes
downloaded per nonempty evaluation**. T is target count and M is tree-node
count. Controls and scratch writes remain outside the existing payload receipt.
All copies/waits remain explicit, and ordinary CPU registration retains its
existing path.

## Correctness references

The actual previous WGSL is frozen from parent source
`95432965534735f8ba3e9efccf930bc31e01c7b0` (PR #186). The native reference also
retains that host's 32*S planning/allocation, bind group, ordered search and
submission. Host reference changes only rename methods and inject a frozen
pipeline cache; no tolerance-based emulation substitutes for the old shader.

- **324 conditions / 20,736 output u32 words** match bit for bit. Counts
  0/1/3/63/64/65/257/4097/65537 cross single-target, lattice, duplicate/unequal
  targets and zero-normal profiles, common normal scales 1e-8/1/1e8, a nonidentity
  pose and distance gates 0.001/0.25/100. Comparisons include all status, counts,
  anchor, high/low sums and reserved words. Every nonempty old/new call acquires
  four pre-poisoned all-one scratch buffers through the actual pool with four
  confirmed cache hits and no extra allocation.
- Production allocation helper requests four distinct leases with exact sizes
  at 1/65/65537/1048577 sources. Cold allocation count and retained bytes match
  40*S+240*G. A synthetic 64-KiB binding limit accepts 4096 sources in the new
  plan while rejecting the old 32-byte plan; one extra source/normal, excess
  dispatch and u32 count bounds remain checked. Existing plan tests retain
  result/partial, storage-binding, uniform and workgroup limit checks.
- Twelve varied queued calls and eight concurrent callers keep independently
  owned result words and decoded equations equal to their frozen references,
  including after input drop and repeated scratch reuse.
- Existing independent f64 raw-normal math, normalized-to-world solve,
  inclusive boundary, invalid numeric/schema/runtime, empty input, transform
  overflow/status zeroing and ownership tests remain. The maximum independent
  normalized component error remains **3.5880697583e-5**, below the unchanged
  5e-5 tolerance.
- Completed registration checks use the original CPU comparisons/tolerances
  for **99,252 points in 108 coordinate-scale/normal-weight/pose/stopping
  conditions**. Maximum normalized CPU pose-coordinate error remains
  **9.536743e-7**, below the unchanged 2e-4 tolerance. Separate outlier, singular,
  schema, configuration and concurrency cases also pass.

## Completed-call measurement protocol

Run the exact ignored native test alone in release mode:

```text
cargo test -p spatialrust-gpu --all-features --lib --release kernels::point_to_plane_reduction::tests::reuse_query_tests::benchmark::completed_reused_query_calls -- --include-ignored --exact --nocapture --test-threads=1
```

Two independent surveys each measure ten conditions. Both versions use
independent runtimes with the default 64-MiB scratch retention policy, cleared
between conditions, and the same compiled search/parallel features. Three
warmups precede twenty paired rounds, rotating old full/new full/old resident/
new resident ordering. Every sampled result and payload receipt is validated;
all samples are retained. Both routes share the valid-case immutable input
upload routine; the frozen equations call keeps its old planning/allocation.
Full calls include CPU target-tree construction,
packing/upload, controls/output/scratch, all GPU passes, download/wait/decode
and owner drops. Prepared resident calls include the public submission and
completion wait; their upload/tree preparation and download/validation are
outside timing. These are one-equation evaluations, not complete multi-update
registration trajectories or isolated shader timestamps.

Profiles are a source lattice with shifted targets and signed varied raw
normal lengths; duplicated sources with zero/nonunit raw normals and 17 extra
targets; and a synthetic single target that isolates large-source reduction
from tree traversal. The 0.001 gate in the single-target profile rejects every
pair, while 100 retains all pairs. The published table records actual target
and correspondence counts. Synthetic large-source cases do not establish
performance on equally large target trees.

The runner waits for a quiet process window and monitors other Cargo, compiler,
linker, spatialrust test and benchmark executables throughout each attempt.
Any overlapping attempt is discarded in full without stopping foreign work.
No build, test or statistical analysis runs during an accepted survey.
Backend and device type are recorded without a local adapter model name.
Ratios are paired geometric means, above one favoring reuse. Within-survey
95% log-bootstrap intervals use 10,000 resamples, seed
20261013+2*record_index+comparison_index, and sorted endpoints 249/9749.

Measurements use Windows, Rust 1.98.1 release optimization, Vulkan on a
discrete compute adapter and twelve Rayon workers. Both final surveys passed
the exact single test, in 9.31 and 9.12 seconds. Both first attempts were
accepted with no detected overlapping build/test/benchmark. All twenty records
contain twenty positive samples per stage. Medians are milliseconds; ratios
and intervals include every sample and are not adjusted for multiple comparisons.

| Survey | Sources/targets | Profile/distance/count | Old full ms | Reused full ms | Old resident ms | Reused resident ms | Old/reused full (95%) | Old/reused resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 257/274 | lattice/0.25/257 | 1.068 | 1.065 | 0.336 | 0.336 | 0.982x (0.950-1.013) | 1.002x (0.896-1.118) |
| 1 | 4,097/4,114 | lattice/0.25/4,097 | 1.900 | 1.982 | 0.565 | 0.528 | 0.965x (0.932-0.998) | 1.041x (0.955-1.137) |
| 1 | 4,097/4,114 | duplicates/0.25/4,097 | 1.891 | 1.855 | 0.506 | 0.507 | 0.992x (0.938-1.050) | 1.027x (0.948-1.109) |
| 1 | 65,537/65,554 | lattice/0.25/65,537 | 21.997 | 22.616 | 2.137 | 2.097 | 0.961x (0.889-1.025) | 1.032x (1.009-1.055) |
| 1 | 65,537/65,554 | duplicates/0.25/65,537 | 17.953 | 18.538 | 1.906 | 1.839 | 0.952x (0.919-0.985) | 1.066x (1.026-1.128) |
| 1 | 65,537/1 | single/100/65,537 | 2.646 | 3.351 | 1.311 | 1.280 | 0.769x (0.705-0.813) | 1.026x (0.981-1.082) |
| 1 | 262,145/1 | single/0.001/0 | 6.524 | 6.339 | 1.998 | 1.963 | 1.010x (0.970-1.050) | 1.037x (1.016-1.060) |
| 1 | 262,145/1 | single/100/262,145 | 8.475 | 6.563 | 2.036 | 1.948 | 1.291x (1.242-1.339) | 1.039x (1.013-1.066) |
| 1 | 1,048,577/1 | single/0.001/0 | 21.259 | 22.700 | 4.711 | 4.670 | 0.947x (0.927-0.974) | 1.012x (1.002-1.022) |
| 1 | 1,048,577/1 | single/100/1,048,577 | 21.445 | 22.883 | 4.719 | 4.622 | 0.950x (0.930-0.973) | 1.020x (1.010-1.030) |
| 2 | 257/274 | lattice/0.25/257 | 1.175 | 1.208 | 0.352 | 0.355 | 0.994x (0.976-1.014) | 0.989x (0.881-1.106) |
| 2 | 4,097/4,114 | lattice/0.25/4,097 | 1.758 | 1.772 | 0.490 | 0.487 | 0.992x (0.959-1.019) | 0.998x (0.928-1.074) |
| 2 | 4,097/4,114 | duplicates/0.25/4,097 | 1.957 | 1.889 | 0.517 | 0.517 | 1.043x (0.993-1.122) | 1.004x (0.927-1.088) |
| 2 | 65,537/65,554 | lattice/0.25/65,537 | 18.934 | 21.019 | 2.082 | 2.040 | 0.928x (0.878-0.977) | 1.019x (1.002-1.037) |
| 2 | 65,537/65,554 | duplicates/0.25/65,537 | 18.154 | 17.714 | 1.902 | 1.928 | 1.019x (0.957-1.082) | 1.026x (0.990-1.072) |
| 2 | 65,537/1 | single/100/65,537 | 2.585 | 3.244 | 1.343 | 1.323 | 0.783x (0.755-0.817) | 0.967x (0.877-1.033) |
| 2 | 262,145/1 | single/0.001/0 | 6.090 | 6.261 | 2.030 | 1.949 | 0.973x (0.933-1.016) | 1.030x (0.999-1.060) |
| 2 | 262,145/1 | single/100/262,145 | 7.860 | 6.040 | 2.040 | 1.947 | 1.288x (1.247-1.331) | 1.042x (1.013-1.070) |
| 2 | 1,048,577/1 | single/0.001/0 | 21.702 | 22.459 | 4.702 | 4.659 | 0.942x (0.910-0.974) | 1.010x (0.998-1.022) |
| 2 | 1,048,577/1 | single/100/1,048,577 | 23.381 | 24.608 | 4.720 | 4.650 | 0.979x (0.951-1.009) | 1.015x (1.005-1.025) |

Three resident conditions have intervals above one in both surveys:
65,537-source shifted lattice (1.032x/1.019x), 262,145-source fully gated
single-target (1.039x/1.042x), and 1,048,577-source fully gated single-target
(1.020x/1.015x). These modest completed-resident gains include host binding,
resource and driver effects; they are not isolated shader throughput. Smaller
conditions are inconclusive. Duplicate and all-rejected profiles have positive
resident intervals only in the first survey; they do not establish a repeated
speedup.

Full calls are mixed, with retained adverse cases. The fully gated
262,145-source single-target full call improves 1.291x/1.288x in both surveys,
but the 65,537-source single-target full call worsens (old/new ratios
0.769x/0.783x, both intervals below one). The 1,048,577-source all-rejected
full call also worsens 0.947x/0.942x with both intervals below one; the fully
gated version worsens in the first survey and is inconclusive in the second.
A 4,097-source lattice and 65,537-source duplicate condition worsen in the first
survey, and the 65,537-source lattice worsens in the second. No universal
full-call improvement is claimed. Full CPU preparation/upload/allocation/decode
can dominate these evaluations, and large full-call differences cannot be
attributed solely to the small resident execution improvement. Complete
multi-update registration timing and other adapters remain unmeasured here.

All four scratch leases fit the default 64-MiB retained budget for every measured
condition. After completion, the actual old/new retained-byte counters equal
their respective requested working bytes in every record. Retention policy and
allocation count stay unchanged; these are working/retained buffer payloads,
not a complete GPU memory measurement. The exact memory reduction holds for
both faster and slower conditions:

| Sources | Previous scratch bytes | Reused scratch bytes | Saved bytes |
| ---: | ---: | ---: | ---: |
| 257 | 15,592 | 11,480 | 4,112 |
| 4,097 | 245,032 | 179,480 | 65,552 |
| 65,537 | 3,731,512 | 2,682,920 | 1,048,592 |
| 262,145 | 14,741,560 | 10,547,240 | 4,194,320 |
| 1,048,577 | 58,781,752 | 42,004,520 | 16,777,232 |


## Verification

- Native GPU all-feature library: **243 passed, 29 ignored**. This includes
  all seven point-to-plane correctness tests (the measurement test stays
  ignored), image/point kernels, runtime/cache and scratch regressions.
  After the local-pair store refinement, the affected seven tests are rerun
  explicitly; the rest of the kernels/runtime/cache code is unchanged.
- Explicit release registration integration: **two tests passed**, including
  the 108-condition accuracy/receipt checks and validation/concurrency suite.
- GPU all-feature/all-target Clippy, CPU-only GPU build, optional GPU WASM
  build and minimal point-to-plane GPU facade build pass.
- Source WGSL fixture matches the pinned parent exactly after newline
  normalization. No new dependency, public unsafe API, automatic placement or
  core/math/search/registration implementation change is introduced.

## Source and local evidence paths

- Production host:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\point_to_plane_reduction.rs`.
- Production shader:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\shaders\point_to_plane_reduction.wgsl`.
- Native frozen host/allocation/ownership tests:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\point_to_plane_reuse_query_tests.rs`.
- Frozen shader:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\tests\fixtures\point_to_plane_reuse_queries_previous.wgsl`.
- Reproducible ignored survey:
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\kernels\point_to_plane_reuse_query_benchmark.rs`.
- Ignored local logs:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-plane-reuse-transformed-queries-benchmark-1.log`
  and `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-plane-reuse-transformed-queries-benchmark-2.log`.
- Raw twenty-sample records and analysis:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-plane-reuse-transformed-queries-measurements.json`.
