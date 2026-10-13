# Direct SoA registration target upload

This slice removes the intermediate original-order target `Vec<[f32; 3]>`
from shared `GpuIcpInput::upload`. The CPU target tree already owns three
original-order coordinate slices. Target geometry now consumes those slices
directly; a private `GpuQueryBatch::upload_soa` packs them directly into the
existing padded device records.

The eliminated vector occupied 12*T host bytes and previously remained live
through tree, source and target upload. This is an allocation/layout fact,
not a measured process-RSS reduction. Existing 16*T padded target staging,
tree-order point/node staging, source staging and covariance/normal storage
remain. There is no new public API, dependency, unsafe boundary or core change.

The previous merged source is `f3c5032824149c34b7fc4126599ff85077bbfe34`
(PR #190, source `d8a97d79aa06fa014720c24c5bc9c6ad0235175a`). The test-local
previous upload is transcribed from that merge. It continues to use the shared,
unchanged AoS geometry and public AoS query uploader; the new production path
uses SoA target geometry/packing. This comparison isolates target staging from
the earlier geometry-worker and covariance-upload changes.

Source preparation is unchanged. Target geometry uses the same threshold of
262,144 points, at most one scoped worker per 65,536 points and available native
CPU parallelism, with serial small/one-worker/WASM processing. Workers return
24-byte bounds summaries. Shape checks precede scanning or thread startup;
finite checks and final normal-extent validation precede GPU input allocation.
Private query upload checks lengths, count/buffer/dispatch limits and finite XYZ
before packing/device allocation. No automatic CPU/GPU placement is introduced.

Midpoints and extent squares retain original f64 arithmetic and axis order;
derived zero anchors remain canonical +0. Uploaded original XYZ bits, including
signed zero/subnormals, retain their original order and values. The device still
holds both tree-order and original-order targets. Input layouts, shaders,
scratch pools, pose solvers, match gates, stopping, status and owned output
lifetimes remain unchanged. Shared upload receipt is still
16*S + 32*T + 32*M host-to-device bytes and zero device-to-host bytes.

Production source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_geometry_preparation.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs`

Validation and measurement paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_soa_upload_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhood_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_soa_upload_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-registration/tests/gpu_gicp_reuse_query_benchmark.rs`

New valid geometry comparisons cover 76,032,240 points plus 1,048,580 points
across four concurrent preparations. Eight profiles include a 3D hash grid,
planar/line data, coincident positive/mixed/negative zeros, large offsets and
subnormals. Four requested worker counts plus the automatic policy are tested
at empty/single/small sizes and both sides of 65,536, 131,072 and 262,144,
through 524,289 points. All anchor bits and f64 extent bits match the unchanged
AoS path exactly, without zero-sign exceptions or numeric tolerances.

Invalid tests exercise unequal coordinate lengths, NaN and both infinities in
every axis at chunk/start/end positions, extent overflow and subnormal geometry.
Constant f32::MAX coordinates retain zero extent. Query-packing tests compare
all XYZ/padding bits for eight signed-zero masks, subnormal/extreme values and
nontrivial original order; shape and nonfinite cases reject. Native query tests
compare AoS/SoA k-NN rows for 20 fixtures and three k values, including ties and
duplicates, empty allocation/receipts and runtime mismatch.

Native shared upload compares 144 evaluation conditions across eight target
sizes through 524,289 and six profiles. All 6,336 raw result words match the
merged upload exactly, including status/count, exposed anchors and compensated
sums. Decoded results, source/target frames, scale bits and full transfer
receipts match; queued outputs remain valid after inputs are dropped. Empty
sources, empty/overflow/subnormal targets and runtime mismatch are checked.

The local release run passes 64 native tests: 24 ICP, 24 GICP, seven plane and
nine neighborhood tests. Existing independent numeric references, frozen
equation words and resident covariance/error/ownership checks remain in scope.

Eight registration integration targets pass 29 test executions (20 unique
functions; shared references repeat), covering 1,269 named trajectory conditions:
54 ICP, 486 trimmed ICP, 27 large precision trajectories, 108 plane and 594
GICP covariance-route conditions. These unique matrices evaluate 4,040,037
source points. Existing tolerances were not widened. Maximum normalized pose
errors remain 1.4305115e-6 (ICP), 1.9073486e-6 (trimmed), 7.6293945e-6 (large
precision), and 9.536743e-7 (plane/GICP). The previous explicit f32 precision
mode's adverse 8.154977e-4 large-trajectory error is unchanged.

GPU all-feature/all-target and combined registration-feature Clippy, CPU-only
GPU check, minimal native facade, combined ICP/GICP/plane WASM facade and
rustfmt checks pass. Clippy caught a newer array-reference helper in test code;
explicit slice references restore Rust 1.75 compatibility. All five new tests
pass again after this test-only repair. Production code and numeric tolerances
were unchanged by the repair.

Two native release surveys use Vulkan discrete GPU execution, 12 available CPU
workers, Rust 1.98.1 and default 64 MiB scratch budgets. The primitive benchmark
has 14 fixtures, three warmups and 20 paired samples per variant/stage; four
routes rotate in order each round. Each fixture's CPU target tree and source
fixture are prepared outside timers. Host timing includes old target AoS
allocation/copy, unchanged AoS geometry and vector drop, versus direct borrowed
SoA geometry. Complete timing includes actual shared upload, one ordinary
moment submission, 176-byte download/wait/decode and owned input/output drop.
It is a completed moment primitive, not a full registration solve. Upload,
submission, download/wait and drop phase samples sum exactly to each full sample.

Both primitive surveys are accepted on their first attempts (17.32 and 14.97
seconds). All timed decoded results and full transfer receipts match. For the
three 1,048,577-target profiles, host paired ratios are 1.902-2.188x across both
surveys, with all six 95% intervals above one. The 3D host medians are
6.784/6.881 -> 3.302/3.720 ms. This benefit concerns target preparation, not all
upload work or the device kernels.

Repeated complete-primitive intervals above one occur at 131,073 targets
(1.130/1.075x), 262,143 (1.169/1.132x), 262,145 (1.087/1.037x), 524,289
(1.052/1.123x) and the million-target planar fixture (1.065/1.101x). Other
conditions have intervals that include one in at least one survey. In
particular, million-target 3D complete ratios are only 1.047/1.059x, with the
first interval including one; the faster host stage does not imply a matching
whole-call speedup. No complete-primitive interval is entirely below one.

The identical public GICP benchmark source is measured against the preserved
PR #190 release executable, whose source equals the merged parent. It covers
eight cases through 65,537 sources and slightly unequal targets, CPU k-NN and
resident GPU k-NN/radius/grid covariance preparation, three forced pose updates
and four equation evaluations. Caller fixtures stay outside the timer; internal
CPU tree/covariance preparation, explicit upload, all GPU work/readbacks,
solves/stopping and owned GPU resource drops stay inside. Returned CPU diagnostic
validation/drop is outside the timer. Old/new processes stay alive but
receive one command at a time. Three warmups precede 20 alternated paired
samples per case and survey. All 640 timed full diagnostics match exactly,
including pose, fitness, iteration/evaluation counts and transfer receipts.

Both public surveys and all 16 individual conditions are accepted on their
first attempts. Public alignment has no repeated full-call interval above one
or below one. Retain the adverse first CPU k-NN 65,537-point interval
0.951x [0.912, 0.992]; its second interval is 1.026x [0.938, 1.123]. The second
16,384-point GPU-grid interval is favorable (1.209x [1.025, 1.398]), while its
first interval includes one. Several intervals are broad. No consistent public
registration speedup is established, and these cases do not measure
million-target full registration. All condition tables follow, including adverse
and inconclusive samples.

The controllers monitor named Cargo/compiler/linker/test/benchmark process
classes and require three quiet observations. Primitive observation spacing is
five seconds; public per-condition spacing is two seconds. A detected overlap
discards the entire primitive survey or full public condition. Accepted data
have no observed overlap; statistical analysis runs only after all measurements
finish. This guard does not prove every machine process is idle. No foreign
process was terminated. Scratch is warm for measured samples; cache/pipeline
startup and caller tree construction are excluded from primitive measurements.

Intervals bootstrap 20 paired log ratios with 10,000 resamples and sorted
endpoints 249/9749; they are unadjusted across multiple conditions. Seed is
2026101401 + 28*(survey-1) + 2*case_index + stage for primitive host/full
(stage 0/1), and 2026101401 + 8*(survey-1) + case_index for public calls.
Ratios use geometric means; table times use separate medians. Sub-microsecond
host controls are timer-noise limited and rounded to three decimal milliseconds.
Phase medians need not sum to the median full time.

Executable/source identities:

- Previous public executable SHA-256: `22D24CDB3CD9D4039ABBE30D51F4BBE42C28ADA8B4B8861D1AE111AB0BD9A785`
- Current public executable SHA-256: `8E47EE0DED19FB7614CE8D5D5016B87333299BD193CAE95827B46AB5CF4E5425`
- Identical public benchmark source SHA-256: `34343FCC04D26E903FFA5B764BDF5C27EAEE8470D9CCB7416AD2539389F6C0D8`
- Final native primitive/test executable SHA-256: `D3AEAC830AA6A50AEA82D1B02845D9206EE07EA4841D9B517A4E283925F74499`

Raw samples, phase arrays, controller overlap receipts, hashes and analysis
remain local under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:

- `direct-soa-registration-baseline.json`
- `run-direct-soa-registration-survey.ps1`
- `run-direct-soa-registration-public-survey.ps1`
- `analyze-direct-soa-registration.py`
- `direct-soa-registration-benchmark-1.log` / `-2.log`
- `direct-soa-registration-measurements.json`
- `direct-soa-registration-table.md` / `direct-soa-registration-phase-table.md`
- `direct-soa-registration-public-benchmark-1.json` / `-2.json`
- `direct-soa-registration-public-measurements.json` / `direct-soa-registration-public-table.md`

Exact native primitive test:
`kernels::icp_reduction::geometry_preparation::soa_tests::benchmark::paired_completed_soa_upload_calls`.
Public test: `paired_completed_registration_calls`; it requires the paired
stdin/stdout controller. Both remain ignored in ordinary test runs and require
an isolated native release environment. Reports describe backend/capabilities
without naming the local adapter.

## Completed primitive and host preparation

| Survey | Sources / targets / profile | Workers | Stage | Old ms | New ms | Old/new paired ratio [95%] |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 / 1 / 0 | 1 | host | 0.000 | 0.000 | 1.022x (0.904-1.147) |
| 1 | 1 / 1 / 0 | 1 | full | 1.543 | 1.452 | 1.054x (1.000-1.114) |
| 1 | 257 / 257 / 0 | 1 | host | 0.005 | 0.002 | 1.924x (1.777-2.064) |
| 1 | 257 / 257 / 0 | 1 | full | 2.028 | 2.057 | 1.028x (0.967-1.086) |
| 1 | 4,097 / 4,097 / 0 | 1 | host | 0.027 | 0.019 | 1.425x (1.306-1.605) |
| 1 | 4,097 / 4,097 / 0 | 1 | full | 1.552 | 1.579 | 1.065x (0.996-1.146) |
| 1 | 257 / 65,537 / 0 | 1 | host | 0.448 | 0.342 | 1.359x (1.275-1.456) |
| 1 | 257 / 65,537 / 0 | 1 | full | 3.950 | 3.839 | 1.048x (0.989-1.123) |
| 1 | 65,537 / 65,537 / 0 | 1 | host | 0.444 | 0.345 | 1.289x (1.220-1.364) |
| 1 | 65,537 / 65,537 / 0 | 1 | full | 5.429 | 5.131 | 1.077x (1.016-1.135) |
| 1 | 257 / 131,073 / 0 | 1 | host | 1.418 | 0.755 | 1.906x (1.809-2.026) |
| 1 | 257 / 131,073 / 0 | 1 | full | 8.528 | 7.354 | 1.130x (1.075-1.190) |
| 1 | 257 / 262,143 / 0 | 1 | host | 2.762 | 1.484 | 1.877x (1.782-1.980) |
| 1 | 257 / 262,143 / 0 | 1 | full | 13.759 | 11.377 | 1.169x (1.111-1.226) |
| 1 | 257 / 262,144 / 0 | 4 | host | 2.393 | 1.655 | 1.480x (1.434-1.533) |
| 1 | 257 / 262,144 / 0 | 4 | full | 12.698 | 11.884 | 1.066x (1.008-1.128) |
| 1 | 257 / 262,145 / 0 | 4 | host | 2.620 | 1.764 | 1.416x (1.340-1.487) |
| 1 | 257 / 262,145 / 0 | 4 | full | 13.913 | 12.758 | 1.087x (1.037-1.135) |
| 1 | 257 / 524,289 / 0 | 8 | host | 4.273 | 2.383 | 1.694x (1.456-1.869) |
| 1 | 257 / 524,289 / 0 | 8 | full | 25.170 | 24.625 | 1.052x (1.008-1.098) |
| 1 | 257 / 1,048,577 / 0 | 12 | host | 6.784 | 3.302 | 2.099x (1.977-2.224) |
| 1 | 257 / 1,048,577 / 0 | 12 | full | 56.069 | 55.278 | 1.047x (0.998-1.103) |
| 1 | 257 / 1,048,577 / 1 | 12 | host | 7.055 | 3.245 | 2.046x (1.784-2.243) |
| 1 | 257 / 1,048,577 / 1 | 12 | full | 58.407 | 52.335 | 1.065x (1.008-1.126) |
| 1 | 257 / 1,048,577 / 2 | 12 | host | 6.745 | 3.308 | 2.057x (1.970-2.140) |
| 1 | 257 / 1,048,577 / 2 | 12 | full | 50.152 | 47.266 | 1.046x (0.981-1.110) |
| 1 | 262,145 / 1 / 0 | 1 | host | 0.001 | 0.001 | 1.703x (1.359-2.251) |
| 1 | 262,145 / 1 / 0 | 1 | full | 5.988 | 5.986 | 0.977x (0.918-1.033) |
| 2 | 1 / 1 / 0 | 1 | host | 0.001 | 0.000 | 1.188x (0.996-1.429) |
| 2 | 1 / 1 / 0 | 1 | full | 1.642 | 1.537 | 1.029x (0.949-1.112) |
| 2 | 257 / 257 / 0 | 1 | host | 0.003 | 0.002 | 1.495x (1.408-1.609) |
| 2 | 257 / 257 / 0 | 1 | full | 1.689 | 1.691 | 0.993x (0.925-1.069) |
| 2 | 4,097 / 4,097 / 0 | 1 | host | 0.027 | 0.022 | 1.315x (1.244-1.433) |
| 2 | 4,097 / 4,097 / 0 | 1 | full | 1.692 | 1.655 | 0.991x (0.947-1.033) |
| 2 | 257 / 65,537 / 0 | 1 | host | 0.434 | 0.335 | 1.290x (1.205-1.375) |
| 2 | 257 / 65,537 / 0 | 1 | full | 3.978 | 3.832 | 1.046x (0.996-1.097) |
| 2 | 65,537 / 65,537 / 0 | 1 | host | 0.479 | 0.359 | 1.285x (1.234-1.339) |
| 2 | 65,537 / 65,537 / 0 | 1 | full | 5.539 | 5.402 | 1.015x (0.959-1.075) |
| 2 | 257 / 131,073 / 0 | 1 | host | 1.231 | 0.678 | 1.834x (1.738-1.954) |
| 2 | 257 / 131,073 / 0 | 1 | full | 6.877 | 6.389 | 1.075x (1.021-1.132) |
| 2 | 257 / 262,143 / 0 | 1 | host | 2.435 | 1.307 | 1.850x (1.787-1.915) |
| 2 | 257 / 262,143 / 0 | 1 | full | 11.530 | 10.245 | 1.132x (1.102-1.161) |
| 2 | 257 / 262,144 / 0 | 4 | host | 2.119 | 1.539 | 1.348x (1.314-1.384) |
| 2 | 257 / 262,144 / 0 | 4 | full | 11.157 | 10.590 | 1.059x (0.994-1.134) |
| 2 | 257 / 262,145 / 0 | 4 | host | 2.038 | 1.543 | 1.356x (1.305-1.414) |
| 2 | 257 / 262,145 / 0 | 4 | full | 11.010 | 10.485 | 1.037x (1.003-1.070) |
| 2 | 257 / 524,289 / 0 | 8 | host | 3.562 | 1.987 | 1.797x (1.713-1.877) |
| 2 | 257 / 524,289 / 0 | 8 | full | 21.250 | 18.772 | 1.123x (1.074-1.171) |
| 2 | 257 / 1,048,577 / 0 | 12 | host | 6.881 | 3.720 | 1.902x (1.791-2.021) |
| 2 | 257 / 1,048,577 / 0 | 12 | full | 60.846 | 56.684 | 1.059x (1.012-1.112) |
| 2 | 257 / 1,048,577 / 1 | 12 | host | 6.328 | 2.988 | 2.032x (1.900-2.148) |
| 2 | 257 / 1,048,577 / 1 | 12 | full | 50.873 | 46.538 | 1.101x (1.038-1.166) |
| 2 | 257 / 1,048,577 / 2 | 12 | host | 6.432 | 2.916 | 2.188x (2.091-2.287) |
| 2 | 257 / 1,048,577 / 2 | 12 | full | 43.227 | 39.174 | 1.106x (1.065-1.152) |
| 2 | 262,145 / 1 / 0 | 1 | host | 0.001 | 0.001 | 1.147x (1.038-1.282) |
| 2 | 262,145 / 1 / 0 | 1 | full | 5.769 | 5.526 | 1.010x (0.942-1.087) |

## Public alignment

| Survey | Sources | Profile | Covariance route | Old ms | New ms | Old/new paired ratio [95%] |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 257 | lattice | CPU k-NN | 7.056 | 7.247 | 0.979x (0.911-1.051) |
| 1 | 4,097 | lattice | CPU k-NN | 15.584 | 16.057 | 0.975x (0.911-1.054) |
| 1 | 4,097 | duplicate/unequal | GPU k-NN | 7.627 | 7.009 | 1.038x (0.969-1.113) |
| 1 | 16,384 | lattice | GPU grid | 11.263 | 10.783 | 0.987x (0.903-1.077) |
| 1 | 65,537 | lattice | CPU k-NN | 81.456 | 86.667 | 0.951x (0.912-0.992) |
| 1 | 65,537 | lattice | GPU k-NN | 36.168 | 35.479 | 1.022x (0.931-1.119) |
| 1 | 65,537 | lattice | GPU radius | 93.332 | 93.734 | 1.004x (0.974-1.037) |
| 1 | 65,537 | duplicate/unequal | GPU grid | 52.577 | 59.924 | 0.874x (0.670-1.093) |
| 2 | 257 | lattice | CPU k-NN | 8.318 | 13.297 | 0.826x (0.672-1.014) |
| 2 | 4,097 | lattice | CPU k-NN | 23.911 | 19.143 | 1.175x (0.942-1.451) |
| 2 | 4,097 | duplicate/unequal | GPU k-NN | 10.585 | 10.891 | 1.004x (0.811-1.245) |
| 2 | 16,384 | lattice | GPU grid | 23.313 | 15.821 | 1.209x (1.025-1.398) |
| 2 | 65,537 | lattice | CPU k-NN | 93.669 | 97.759 | 1.026x (0.938-1.123) |
| 2 | 65,537 | lattice | GPU k-NN | 34.220 | 31.211 | 1.128x (0.954-1.365) |
| 2 | 65,537 | lattice | GPU radius | 40.290 | 39.981 | 0.982x (0.823-1.177) |
| 2 | 65,537 | duplicate/unequal | GPU grid | 30.521 | 29.118 | 1.035x (0.846-1.245) |

## Completed primitive phases

| Survey | Sources / targets / profile | Variant | Upload ms | Submit ms | Download/wait ms | Drop ms | Full ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 / 1 / 0 | old | 0.030 | 0.176 | 1.318 | 0.006 | 1.543 |
| 1 | 1 / 1 / 0 | new | 0.027 | 0.161 | 1.230 | 0.005 | 1.452 |
| 1 | 257 / 257 / 0 | old | 0.057 | 0.273 | 1.714 | 0.009 | 2.028 |
| 1 | 257 / 257 / 0 | new | 0.062 | 0.280 | 1.676 | 0.008 | 2.057 |
| 1 | 4,097 / 4,097 / 0 | old | 0.141 | 0.173 | 1.235 | 0.006 | 1.552 |
| 1 | 4,097 / 4,097 / 0 | new | 0.139 | 0.177 | 1.261 | 0.006 | 1.579 |
| 1 | 257 / 65,537 / 0 | old | 2.202 | 0.239 | 1.467 | 0.006 | 3.950 |
| 1 | 257 / 65,537 / 0 | new | 2.086 | 0.240 | 1.441 | 0.006 | 3.839 |
| 1 | 65,537 / 65,537 / 0 | old | 3.284 | 0.237 | 1.779 | 0.007 | 5.429 |
| 1 | 65,537 / 65,537 / 0 | new | 3.028 | 0.222 | 1.800 | 0.006 | 5.131 |
| 1 | 257 / 131,073 / 0 | old | 5.842 | 0.327 | 1.635 | 0.463 | 8.528 |
| 1 | 257 / 131,073 / 0 | new | 4.753 | 0.324 | 1.714 | 0.473 | 7.354 |
| 1 | 257 / 262,143 / 0 | old | 10.985 | 0.270 | 1.554 | 0.452 | 13.759 |
| 1 | 257 / 262,143 / 0 | new | 8.625 | 0.231 | 1.597 | 0.420 | 11.377 |
| 1 | 257 / 262,144 / 0 | old | 10.068 | 0.263 | 1.673 | 0.476 | 12.698 |
| 1 | 257 / 262,144 / 0 | new | 9.435 | 0.277 | 1.635 | 0.436 | 11.884 |
| 1 | 257 / 262,145 / 0 | old | 10.755 | 0.270 | 1.713 | 0.654 | 13.913 |
| 1 | 257 / 262,145 / 0 | new | 10.223 | 0.270 | 1.751 | 0.609 | 12.758 |
| 1 | 257 / 524,289 / 0 | old | 21.214 | 0.315 | 2.451 | 1.020 | 25.170 |
| 1 | 257 / 524,289 / 0 | new | 19.411 | 0.274 | 2.599 | 1.144 | 24.625 |
| 1 | 257 / 1,048,577 / 0 | old | 49.302 | 0.299 | 4.378 | 2.034 | 56.069 |
| 1 | 257 / 1,048,577 / 0 | new | 47.455 | 0.287 | 4.295 | 2.086 | 55.278 |
| 1 | 257 / 1,048,577 / 1 | old | 49.570 | 0.278 | 4.731 | 1.902 | 58.407 |
| 1 | 257 / 1,048,577 / 1 | new | 44.131 | 0.310 | 4.953 | 1.955 | 52.335 |
| 1 | 257 / 1,048,577 / 2 | old | 44.077 | 0.283 | 4.084 | 1.933 | 50.152 |
| 1 | 257 / 1,048,577 / 2 | new | 36.974 | 0.306 | 7.047 | 1.884 | 47.266 |
| 1 | 262,145 / 1 / 0 | old | 3.933 | 0.255 | 1.782 | 0.006 | 5.988 |
| 1 | 262,145 / 1 / 0 | new | 3.986 | 0.266 | 1.794 | 0.006 | 5.986 |
| 2 | 1 / 1 / 0 | old | 0.032 | 0.175 | 1.415 | 0.005 | 1.642 |
| 2 | 1 / 1 / 0 | new | 0.030 | 0.176 | 1.329 | 0.005 | 1.537 |
| 2 | 257 / 257 / 0 | old | 0.042 | 0.179 | 1.446 | 0.006 | 1.689 |
| 2 | 257 / 257 / 0 | new | 0.042 | 0.185 | 1.462 | 0.006 | 1.691 |
| 2 | 4,097 / 4,097 / 0 | old | 0.150 | 0.175 | 1.354 | 0.005 | 1.692 |
| 2 | 4,097 / 4,097 / 0 | new | 0.152 | 0.171 | 1.294 | 0.005 | 1.655 |
| 2 | 257 / 65,537 / 0 | old | 2.140 | 0.238 | 1.461 | 0.006 | 3.978 |
| 2 | 257 / 65,537 / 0 | new | 2.104 | 0.206 | 1.484 | 0.006 | 3.832 |
| 2 | 65,537 / 65,537 / 0 | old | 3.335 | 0.229 | 1.856 | 0.006 | 5.539 |
| 2 | 65,537 / 65,537 / 0 | new | 3.185 | 0.232 | 1.948 | 0.007 | 5.402 |
| 2 | 257 / 131,073 / 0 | old | 4.853 | 0.242 | 1.339 | 0.394 | 6.877 |
| 2 | 257 / 131,073 / 0 | new | 4.359 | 0.254 | 1.385 | 0.380 | 6.389 |
| 2 | 257 / 262,143 / 0 | old | 8.989 | 0.260 | 1.597 | 0.445 | 11.530 |
| 2 | 257 / 262,143 / 0 | new | 7.736 | 0.198 | 1.585 | 0.411 | 10.245 |
| 2 | 257 / 262,144 / 0 | old | 8.926 | 0.246 | 1.599 | 0.412 | 11.157 |
| 2 | 257 / 262,144 / 0 | new | 8.005 | 0.191 | 1.594 | 0.405 | 10.590 |
| 2 | 257 / 262,145 / 0 | old | 8.375 | 0.246 | 1.614 | 0.474 | 11.010 |
| 2 | 257 / 262,145 / 0 | new | 7.996 | 0.198 | 1.585 | 0.483 | 10.485 |
| 2 | 257 / 524,289 / 0 | old | 16.957 | 0.270 | 2.459 | 0.979 | 21.250 |
| 2 | 257 / 524,289 / 0 | new | 14.795 | 0.263 | 2.410 | 0.965 | 18.772 |
| 2 | 257 / 1,048,577 / 0 | old | 53.485 | 0.291 | 4.024 | 1.925 | 60.846 |
| 2 | 257 / 1,048,577 / 0 | new | 49.583 | 0.289 | 5.730 | 1.894 | 56.684 |
| 2 | 257 / 1,048,577 / 1 | old | 43.701 | 0.280 | 4.502 | 1.851 | 50.873 |
| 2 | 257 / 1,048,577 / 1 | new | 39.194 | 0.301 | 4.479 | 1.774 | 46.538 |
| 2 | 257 / 1,048,577 / 2 | old | 35.151 | 0.262 | 4.804 | 1.757 | 43.227 |
| 2 | 257 / 1,048,577 / 2 | new | 32.529 | 0.302 | 3.913 | 1.977 | 39.174 |
| 2 | 262,145 / 1 / 0 | old | 3.713 | 0.226 | 1.631 | 0.005 | 5.769 |
| 2 | 262,145 / 1 / 0 | new | 3.566 | 0.218 | 1.610 | 0.006 | 5.526 |
