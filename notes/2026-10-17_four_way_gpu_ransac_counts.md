# Four-way integer RANSAC count reduction

Parent: `3ba645aba0cc7e8debe6341d80e853e75b317e25` (merged PR 256).

The point-parallel plane scorer now reduces its 256 lane counts in four-way
stages instead of a binary tree. This changes one private WGSL reduction loop.
The RANSAC planner, full public API and floating point distance/model arithmetic
are byte-for-byte unchanged.

## Exact reduction and synchronization

After every lane writes its count and the existing barrier completes, active
lanes reduce four disjoint source blocks into the first block. Stage strides
are 64, 16, 4 and 1. For each active lane, its first input is its own destination;
the other three inputs are in inactive blocks, which no lane writes in that
stage. Each retained barrier makes all writes visible before the next stage.
All lanes encounter the same four barriers. No subgroup or warp-size assumption
is introduced. Model publication, count initialization and the final barrier
before the next hypothesis remain unchanged.

| Reduction source work per hypothesis | Binary parent | Four-way |
| --- | ---: | ---: |
| Stages / reduction barriers | 8 | 4 |
| Active lane updates | 255 | 85 |
| Integer additions | 255 | 255 |

These are source-level counts; generated instructions and physical shared-memory
transactions are not established. Total explicit main-loop barriers per model
become eleven to seven when including model publication, initial counts and the
final model-reuse guard. Shared storage capacity remains identical.

Every real lane count is nonnegative, and their sum is at most the validated
u32 point count. Every partial sum is bounded by that same total, so changing the
integer grouping introduces neither overflow nor a numerical rounding change.
The CPU test oracle sums raw counts in u64 before narrowing. Degenerate model
counts remain zero. Score normal/distance bits and every per-point threshold
test retain the parent expressions and order.

The production host, serial shader, dispatch threshold (1024 points), group cap,
layouts, lazy caches, controls, allocation/pool rules, errors, ownership and
explicit XYZ/hypothesis uploads and score download remain unchanged. No new
public API, dependency, core, scratch buffer, pass, transfer, wait or unsafe code
is introduced. There is still one compute/readback-copy submission per call.

## Correctness and integration

- **366 native correctness tests:** 145 image, 184 voxel/readback, 8 RANSAC,
  8 segmentation, 18 filtering, 2 image-transfer
  facade and 1 Gaussian facade. Include ignored GPU correctness tests and exclude
  benchmarks. All-feature/all-target Clippy with warnings denied, CPU-only,
  minimal/CI WASM with warnings denied, formatting and whitespace pass.
- **881 independent raw reductions per shader**, including maximum u32 mass in
  each of 256 single active lanes, zero, balanced remainder distributions at
  binary/four-way boundaries and 512 uneven weighted distributions. Both actual
  production reduction loops are extracted unchanged into GPU harnesses, then
  checked against a direct u64 sum and untouched leading/trailing output guards.
  The test covers distributions beyond those attainable by ordinary grid strides.
- **218,700 complete raw score words** compared to the frozen immediate
  parent, including dirty output initialization, tiny/odd/tail sizes through
  1,000,003 points, degenerate and oblique planes, exact/adjacent thresholds,
  negative/NaN/infinite thresholds, and forced one/three-group dispatch across
  513 hypotheses. Independent CPU axis counts verify distance-boundary behavior.
- **80,040 complete-parent API score words** across 36 paired API
  fixtures, serial controls around the 1024-point threshold, all five score words,
  retained output ownership and stable pooled XYZ allocations on warm reuse.
  Existing invalid/no-op/cold-cache/concurrency and segmentation checks pass.
- **512 original GPU/image Rust/WGSL files unchanged**, with **520 validated
  source/executable hashes**. Only the parallel shader reduction and an appended
  native test-module declaration differ in original files. The parent parallel
  shader is frozen byte-for-byte, and its complete planner/pipelines/API are
  frozen with explicit type/name/visibility/external-cache substitutions.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=gpu_pass_and_full_api checks_drop_excluded=true`. Adapter model names are omitted.

Two sequential surveys each use three warm pairs and thirty measured pairs
across ten cases. Eight use the changed point-parallel shader; two serial
controls use the identical serial shader. Old/new order alternates by round.
The entire **40-row** result set is retained below. Changed parallel cases have:

- GPU pass: **6 favorable / 10 inconclusive / 0 adverse**.
- Complete upload/score/readback API: **2 favorable / 14 inconclusive / 0 adverse**.

Unchanged serial controls have pass **0 favorable / 4 inconclusive / 0 adverse** and full API
**0 favorable / 4 inconclusive / 0 adverse**. No pointwise interval is adverse.

At 1024 points / 17 hypotheses, pass medians are 0.005744 -> 0.005648 ms and 0.005872 -> 0.005680 ms;
full API medians are 1.131050 -> 1.162500 ms and 1.144150 -> 1.121100 ms. At 65,537 points / 257 hypotheses,
pass medians are 0.126880 -> 0.122592 ms and 0.129040 -> 0.124032 ms; full API medians are
1.464350 -> 1.480550 ms and 1.452250 -> 1.339300 ms. Every other condition remains visible below.

At 1,024 points / 513 hypotheses, pass medians are
0.017536 -> 0.013792 ms and 0.017728 -> 0.013920 ms (about 21% lower in both surveys), with favorable
pointwise intervals in both. Its full-API intervals cross one in both surveys.
The two favorable full-API intervals occur only in survey two, at 8,193 points /
513 hypotheses and 65,537 points / 257 hypotheses; they do not establish a
repeated transfer-inclusive gain. The 1,000,003-point / 17-hypothesis pass intervals cross one in both surveys;
the large case does not show a repeated speedup. No input-specific or device-specific
dispatch heuristic is inferred from these local measurements.

- `pass`: GPU timestamps bracket one old/new scoring compute pass, using equal
  pre-uploaded inputs, layouts/uniforms/dispatch and independently owned outputs.
  Both passes share a command buffer and alternate order. Explicit collection
  reads 32 query bytes after the passes; final score/CPU checks follow timing.
  Upload, pipeline setup, readback, host wait and API construction are excluded.
- `full`: the complete public scoring call, including XYZ/hypothesis staging,
  buffers, encoder, compute/readback copy submission, mapping/wait, result decode
  and the API's own XYZ recycling. Assertions and returned-vector destruction
  follow timing. The complete immediate-parent API uses a separately warmed
  explicit cache; the current call uses its original runtime cache.

No resident public RANSAC API is simulated or added. First compilation, fixture
creation and expected-output preparation stay outside timing. Full calls return
owned scores and every result is checked against the frozen complete API.

Ratios are geometric means of paired original/new samples; above one favors the
new route. Pointwise percentile intervals resample paired log ratios 10,000
times, seed `2026101707 + 100*run + 10*case + route_index`, route order pass/full,
with sorted bounds 249/9749. No multiple-comparison correction is applied.
Zero times would suppress ratios; raw samples remain in ignored receipts.

Run 1: observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization samples `[[[8, 0]], [[9, 0]]]` before / `[[[5, 0]], [[15, 0]]]` after.

Run 2: observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization samples `[[[15, 0]], [[19, 0]]]` before / `[[[5, 0]], [[17, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all GPU/memory
samples at most 5%. Surveys wait two seconds between snapshots; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post samples cannot
establish isolation throughout every pair. These are shared-host diagnostic
observations, with no portable throughput or application-latency guarantee.
Process identities remain in ignored receipts.

| Run | Points / hypotheses | Boundary | Original median ms | New median ms | Paired original/new ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 100 / 1 | pass | 0.008480 | 0.008528 | 0.983485 [0.938265, 1.029067] |
| 1 | 100 / 1 | full | 1.206050 | 1.154450 | 1.016014 [0.988945, 1.043703] |
| 1 | 1023 / 17 | pass | 0.044672 | 0.044704 | 0.998609 [0.991911, 1.005495] |
| 1 | 1023 / 17 | full | 1.145000 | 1.129900 | 0.995974 [0.962851, 1.031654] |
| 1 | 1024 / 1 | pass | 0.005520 | 0.005216 | 1.047612 [0.991911, 1.104443] |
| 1 | 1024 / 1 | full | 1.102300 | 1.153050 | 0.989564 [0.955903, 1.027144] |
| 1 | 1024 / 17 | pass | 0.005744 | 0.005648 | 1.042662 [0.979422, 1.107632] |
| 1 | 1024 / 17 | full | 1.131050 | 1.162500 | 0.998320 [0.967659, 1.032864] |
| 1 | 1024 / 513 | pass | 0.017536 | 0.013792 | 1.281764 [1.251536, 1.314257] |
| 1 | 1024 / 513 | full | 1.157450 | 1.141450 | 0.972446 [0.906608, 1.028430] |
| 1 | 8193 / 17 | pass | 0.011872 | 0.011488 | 1.015239 [0.974782, 1.053492] |
| 1 | 8193 / 17 | full | 1.140450 | 1.150750 | 0.982868 [0.945684, 1.020646] |
| 1 | 8193 / 513 | pass | 0.049312 | 0.045840 | 1.082934 [1.067510, 1.097811] |
| 1 | 8193 / 513 | full | 1.260050 | 1.282550 | 0.998337 [0.970460, 1.028897] |
| 1 | 65537 / 17 | pass | 0.058672 | 0.058672 | 1.020104 [0.998530, 1.057286] |
| 1 | 65537 / 17 | full | 1.344950 | 1.373350 | 1.006383 [0.971486, 1.044796] |
| 1 | 65537 / 257 | pass | 0.126880 | 0.122592 | 1.040045 [1.033641, 1.046727] |
| 1 | 65537 / 257 | full | 1.464350 | 1.480550 | 1.010309 [0.982290, 1.038278] |
| 1 | 1000003 / 17 | pass | 1.347488 | 1.346864 | 1.007975 [0.994920, 1.028671] |
| 1 | 1000003 / 17 | full | 4.726250 | 4.769450 | 0.984128 [0.959169, 1.008957] |
| 2 | 100 / 1 | pass | 0.008464 | 0.008448 | 1.015221 [0.962270, 1.072504] |
| 2 | 100 / 1 | full | 1.129400 | 1.115450 | 1.001546 [0.961508, 1.044317] |
| 2 | 1023 / 17 | pass | 0.044928 | 0.044960 | 0.980312 [0.955553, 1.001126] |
| 2 | 1023 / 17 | full | 1.167200 | 1.164250 | 1.031432 [0.973581, 1.115001] |
| 2 | 1024 / 1 | pass | 0.005792 | 0.005456 | 1.041673 [0.975774, 1.111568] |
| 2 | 1024 / 1 | full | 1.107900 | 1.087400 | 0.996689 [0.956261, 1.035553] |
| 2 | 1024 / 17 | pass | 0.005872 | 0.005680 | 1.045983 [0.977638, 1.119161] |
| 2 | 1024 / 17 | full | 1.144150 | 1.121100 | 1.021663 [0.990775, 1.053066] |
| 2 | 1024 / 513 | pass | 0.017728 | 0.013920 | 1.269151 [1.234081, 1.303929] |
| 2 | 1024 / 513 | full | 1.162400 | 1.138950 | 1.013788 [0.983624, 1.044277] |
| 2 | 8193 / 17 | pass | 0.011936 | 0.011584 | 1.010564 [0.964914, 1.050092] |
| 2 | 8193 / 17 | full | 1.125950 | 1.168700 | 0.965893 [0.919287, 1.011294] |
| 2 | 8193 / 513 | pass | 0.049664 | 0.046192 | 1.081940 [1.063163, 1.101765] |
| 2 | 8193 / 513 | full | 1.168700 | 1.109150 | 1.040866 [1.010129, 1.075425] |
| 2 | 65537 / 17 | pass | 0.059312 | 0.059008 | 1.003222 [0.935850, 1.075629] |
| 2 | 65537 / 17 | full | 1.282450 | 1.249050 | 1.007410 [0.966986, 1.045198] |
| 2 | 65537 / 257 | pass | 0.129040 | 0.124032 | 1.033270 [1.002196, 1.061897] |
| 2 | 65537 / 257 | full | 1.452250 | 1.339300 | 1.060728 [1.006561, 1.127376] |
| 2 | 1000003 / 17 | pass | 1.340992 | 1.341024 | 1.026384 [0.980979, 1.074204] |
| 2 | 1000003 / 17 | full | 4.487550 | 4.491450 | 1.002717 [0.986523, 1.018928] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Production: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/ransac_plane_parallel.wgsl`.
Native fixtures: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_counts_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_counts_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_counts_parallel_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/ransac_counts_benchmark.rs`.
Ignored validation, raw samples and provenance: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/ransac-counts-*`.
