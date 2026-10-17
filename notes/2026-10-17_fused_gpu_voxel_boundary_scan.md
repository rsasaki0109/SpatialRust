# Fused GPU voxel boundary scan

Parent: `1b062b52314297d223a7516835dac55f133f32ae` (merged PR 263).

The first inclusive-scan block now computes the original signed-key boundary
predicate directly from sorted entries. Its unchanged four-way tree consumes
those binary values in workgroup memory. The separate mark pass and its full
flag write/read disappear. Upper scan levels, offsets and final prefix-based
segment writing retain their original integer operations.

## Exactness, ordering and resource effects

For valid input i=0 emits one; later values are one exactly when an x/y/z key
differs from the previous entry. The predecessor load is guarded by i>0 and the
original element count. Point-index words do not participate in key comparison.
Inactive lanes contribute zero and still execute every original workgroup
barrier. A block's first lane compares with the preceding global entry, so runs
crossing workgroups and hierarchy levels retain the original boundaries.

For N>0 the boundary mass is between one and N <= u32::MAX. Every intermediate
tree/hierarchy subtotal is bounded by that mass. There is no new floating math
or integer reassociation in the tree, and no wrap. The first block still emits
its local inclusive prefix and total; the original hierarchy and offsets make
the final prefix globally inclusive. Final keys, starts, indices and CPU-derived
lengths retain the parent values and order.

The new pipeline shares the original four-binding scan layout, with a read-only
16-byte entry source and separate writable prefix/totals. Only the first level
uses entries; upper levels continue reading u32 totals. Every bind group has
distinct storage buffers. Owned indices are written in the final segment pass
and remain outside scratch leases. Prefix/totals leases retain their original
submission lifetime and scalar-copy ordering. Empty production input returns
before scan/cache/encoder work, with its original zero scalar readback.

| N>0 property | Immediate parent | Current |
| --- | --- | --- |
| Boundary values | Mark pass writes owned indices | Generated inside first scan block |
| Initial scan input | Materialized u32 flags | Sorted entries |
| First-level tree and upper hierarchy | Four-way u32 scan | Same operations |
| Total compute passes | 2+scan_pass_count(N) | 1+scan_pass_count(N) |
| Final prefix/segment outputs | Owned keys/indices/starts | Same values/ownership |
| Construction scalar readback | 4 bytes | 4 bytes |
| Requested scratch bytes/resources | Parent sizes | Identical |
| Optional query readback | 16 bytes per original pass | 16 fewer bytes |

One dispatch, one bind group and one 16-byte uniform initialization request are
removed per nonempty compaction. The source-level full flag store/load payload
falls by 8*N bytes (8,000,024 bytes at N=1,000,003). This counts removed u32
shader accesses; it is not measured physical memory traffic or a new VRAM
allocation saving. The preceding PR already reduced private flag requests to
four compatibility bytes, which remain for the final legacy layout. Scratch
requests and owned allocation sizes therefore stay identical to PR 263.
No payload upload/readback, CPU fallback, API, dependency or feature is added.

Profiling keeps the original scan-block/offset/write names and removes only
voxel-compact-mark. The first scan-block duration now includes boundary creation.
The profiled two-grid frame preflight reserves two fewer stages: exact capacities
are 0/9/153/157 for 0/1/2/257 points. Query collection remains explicit and is
separate from the frame's original transfer receipt.

## Correctness and integration

- **393 native correctness tests:** 151 image, 200 voxel/readback, 5 existing
  frame tests, 8 RANSAC, 8 segmentation, 18 filtering, 2 transfer facade and
  1 Gaussian facade. Ignored correctness tests are included, benchmarks excluded.
  Clippy on all features/targets with warnings denied, CPU-only, minimal/CI WASM
  with warnings denied, formatting and whitespace checks pass.
- **68,052,310 raw parent/u64 words and guards** across **60 paired fixtures**.
  Original mark plus first scan block and the actual fused pipeline match exact
  local prefixes/totals, source contents and poisoned suffixes/inactive bindings.
  An explicit zero-count raw dispatch checks all-lane barrier participation;
  production zero input does not dispatch.
- **67,944,740 complete-hierarchy parent/u64 words and guards** across **50 paired
  fixtures**. The actual fused encoder matches the frozen complete scan planner,
  untouched input/unused storage and global prefixes through three levels. Zero
  count requests no scratch. Block and hierarchy fixtures cover singleton,
  four-point, 257-point, one-cell and irregular runs, signed extreme keys,
  reversed point indices, boundary sizes and 1,000,003 points.
- **24,036,610 complete parent/CPU logical values** across 25 paired conditions
  verify raw device index order, public metadata/CPU sorting, exact four-byte/zero
  construction receipts and retained outputs across later calls/pool clearing.
- Fresh/warm/four-byte/zero-budget pools preserve exact requested/retained sizes
  and resource counts. Warm repeats allocate no new scratch. Optional profiles
  verify exactly one removed mark stage and query pair per compaction, totaling
  48 bytes across three compared calls. Earlier memory tests retain their
  original exact/minimum storage assertions and now require this precise stage
  removal instead of the historical mark-stage count.
- Public frame checks use the exact capacities above, compare unprofiled outputs
  and transfer receipts, and reject one-too-small capacity before scratch/upload
  work. Unsupported timestamp adapters still run all storage/output comparisons.
- **540 original GPU/image sources unchanged** and
  **555 source/executable hashes**. The complete parent compactor,
  scan planner/hierarchy, shaders and constructors are frozen with mechanical
  name/visibility/import/external-cache substitutions only. The new tree body
  and original upper hierarchy are source-audited unchanged. Measured sources
  and executable are identical to final broad validation.

## Paired diagnostics

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=mark_scan_gpu_resident_compactor_and_full wall_order=balanced checks_drop_excluded=true`. Timestamp queries are supported;
adapter model names are omitted. Two surveys discard three warmups and record
thirty pairs for twelve (N,run) cases: (1,1), (256,4), (257,4), (65537,1),
(65537,257), (65537,65537), (1000003,1), (1000003,4), (1000003,257),
(1000003,1000003), (2000003,irregular), (2000003,2000003). Run=0 cycles
1/2/3/255/256/257/1023/4097. All 72 rows remain, without filtering by interval
direction or zero duration.

- Boundary generation plus complete scan GPU time: **24 favorable / 0 inconclusive / 0 adverse**.
- Completed resident compactor: **11 favorable / 13 inconclusive / 0 adverse**.
- Upload/compact/readback: **2 favorable / 22 inconclusive / 0 adverse**.

Adverse pointwise intervals: none.

Adopted for removing a separate full-array stage while preserving exact outputs,
storage and receipts, supported by consistent GPU-interval improvements. The
mostly inconclusive full-call intervals do not establish a broad application
latency improvement.

Timing boundaries and interpretation:

- The GPU route sums actual mark+hierarchy passes for the parent and actual fused
  hierarchy passes for current. Each pair uses separate poisoned output/flag
  controls and the same read-only entries in one submission. Orders alternate
  fifteen times in each direction. Query bytes are 16*(2*scan_pass_count(N)+1)
  per pair. Setup, final segment writing, scalar synchronization, query collection
  and CPU verification are outside these GPU intervals. Timestamp sums exclude
  inter-pass gaps. Different pass counts also change query instrumentation;
  unprofiled wall routes separately measure completed calls. The current flag
  control is untouched test instrumentation.
- Resident wall time starts from uploaded entries and includes the full call,
  allocation requests/encoding and original four-byte scalar synchronization.
  Full time starts before upload and ends after actual metadata/index readback,
  CPU conversion and per-cell sorting. Every completed output is checked; checks
  and drops are outside timing. GPU and wall routes use the same run fixtures.
- Wall order uses variant=offset^(round%4). Each resident/full route has fifteen
  parent-first and fifteen current-first pairs; each variant visits every ordinal
  position seven/eight times. Three warmups and pool policy are shared. Both
  frozen parent caches are supplied externally; current retains runtime cache
  lookups. New shader compilation can affect first use; cold latency is unmeasured.
- Estimates are paired log(old/new) geometric means with 10,000 paired bootstrap
  resamples, seeds 2026101714+1000*survey+10*case+routeIndex (case zero-based,
  pass/resident/full indices 0/1/2), sorted bounds 249/9749. Above one favors
  current. These are pointwise percentile 95% intervals without correction for
  multiple comparisons. All thirty samples remain; zero durations leave ratios
  unresolved. No outlier filtering/retry. Bootstrap follows both timed surveys.

Pre/post two-second process/GPU observations:

- Survey 1: observed-quiet=False; pre={"busy_cpu_cores": [0.27548869298001627, 0.38328861632002265, 1.2756324261900753], "busy_process_count": 3, "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}; post={"busy_cpu_cores": [0.31179405120962983, 0.8706701807363247], "busy_process_count": 2, "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}.

- Survey 2: observed-quiet=False; pre={"busy_cpu_cores": [0.25296498494366193, 0.6235881024192597, 0.976562500015067], "busy_process_count": 3, "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}; post={"busy_cpu_cores": [0.2982316616873534, 0.3099270209692104, 0.9999532185987731], "busy_process_count": 3, "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}.

Pre/post samples cannot establish isolation during every pair. This shared
host/backend/device class and fixed matrix are affected by scheduling, clock/
thermal state, timestamp granularity and transfer/CPU costs. Removed dispatches
and source-level flag accesses are structural; timings remain workload-dependent.
All adverse/inconclusive rows stay visible. No portable, cold, application-wide
or transfer-inclusive speed guarantee follows from these diagnostics.

| Survey | N | Run | Route | Parent ms | Current ms | Paired old/new | Pointwise 95% interval | Result | Seed | Zero old/new |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | pass | 0.007296 | 0.004480 | 1.595036 | [1.524461, 1.669290] | favorable | 2026102714 | 0/0 |
| 1 | 1 | 1 | resident | 1.279500 | 1.286750 | 1.009531 | [0.951162, 1.064307] | inconclusive | 2026102715 | 0/0 |
| 1 | 1 | 1 | full | 4.879300 | 4.696100 | 1.062662 | [1.015158, 1.125970] | favorable | 2026102716 | 0/0 |
| 1 | 256 | 4 | pass | 0.007664 | 0.004736 | 1.584530 | [1.533929, 1.637272] | favorable | 2026102724 | 0/0 |
| 1 | 256 | 4 | resident | 1.257100 | 1.280350 | 0.995525 | [0.944306, 1.050396] | inconclusive | 2026102725 | 0/0 |
| 1 | 256 | 4 | full | 4.876050 | 4.712250 | 1.025656 | [0.990498, 1.061942] | inconclusive | 2026102726 | 0/0 |
| 1 | 257 | 4 | pass | 0.044848 | 0.034288 | 1.243734 | [1.183650, 1.306475] | favorable | 2026102734 | 0/0 |
| 1 | 257 | 4 | resident | 1.375550 | 1.303050 | 1.032468 | [0.987329, 1.082302] | inconclusive | 2026102735 | 0/0 |
| 1 | 257 | 4 | full | 4.717450 | 4.644350 | 1.019712 | [0.982952, 1.058953] | inconclusive | 2026102736 | 0/0 |
| 1 | 65537 | 1 | pass | 0.036256 | 0.030528 | 1.188480 | [1.181915, 1.196467] | favorable | 2026102744 | 0/0 |
| 1 | 65537 | 1 | resident | 1.587450 | 1.573450 | 1.002821 | [0.957062, 1.046396] | inconclusive | 2026102745 | 0/0 |
| 1 | 65537 | 1 | full | 7.150600 | 7.031050 | 0.973943 | [0.897868, 1.032425] | inconclusive | 2026102746 | 0/0 |
| 1 | 65537 | 257 | pass | 0.036080 | 0.030496 | 1.181293 | [1.176259, 1.186131] | favorable | 2026102754 | 0/0 |
| 1 | 65537 | 257 | resident | 1.561600 | 1.481600 | 1.032622 | [1.001489, 1.066446] | favorable | 2026102755 | 0/0 |
| 1 | 65537 | 257 | full | 5.436800 | 5.446800 | 1.021862 | [0.998349, 1.048026] | inconclusive | 2026102756 | 0/0 |
| 1 | 65537 | 65537 | pass | 0.036160 | 0.030544 | 1.183852 | [1.177814, 1.189789] | favorable | 2026102764 | 0/0 |
| 1 | 65537 | 65537 | resident | 1.702550 | 1.609250 | 1.009317 | [0.917283, 1.079563] | inconclusive | 2026102765 | 0/0 |
| 1 | 65537 | 65537 | full | 5.764300 | 5.736150 | 1.008190 | [0.975317, 1.039037] | inconclusive | 2026102766 | 0/0 |
| 1 | 1000003 | 1 | pass | 3.338144 | 2.534976 | 1.227811 | [1.092390, 1.310454] | favorable | 2026102774 | 0/0 |
| 1 | 1000003 | 1 | resident | 6.343500 | 5.568500 | 1.054174 | [1.004807, 1.118063] | favorable | 2026102775 | 0/0 |
| 1 | 1000003 | 1 | full | 44.478500 | 45.250600 | 1.002138 | [0.976523, 1.027402] | inconclusive | 2026102776 | 0/0 |
| 1 | 1000003 | 4 | pass | 1.272352 | 0.997744 | 1.281621 | [1.270083, 1.293115] | favorable | 2026102784 | 0/0 |
| 1 | 1000003 | 4 | resident | 7.913050 | 7.577500 | 1.030304 | [0.984242, 1.077933] | inconclusive | 2026102785 | 0/0 |
| 1 | 1000003 | 4 | full | 33.377250 | 33.401400 | 1.044948 | [0.996847, 1.123167] | inconclusive | 2026102786 | 0/0 |
| 1 | 1000003 | 257 | pass | 1.260384 | 0.985008 | 1.280415 | [1.269822, 1.291502] | favorable | 2026102794 | 0/0 |
| 1 | 1000003 | 257 | resident | 7.688950 | 7.424150 | 1.059776 | [1.007736, 1.126742] | favorable | 2026102795 | 0/0 |
| 1 | 1000003 | 257 | full | 27.204300 | 26.764300 | 0.977457 | [0.904093, 1.027944] | inconclusive | 2026102796 | 0/0 |
| 1 | 1000003 | 1000003 | pass | 1.260432 | 0.989200 | 1.278897 | [1.267793, 1.289797] | favorable | 2026102804 | 0/0 |
| 1 | 1000003 | 1000003 | resident | 8.021950 | 7.681200 | 1.051868 | [1.030145, 1.075160] | favorable | 2026102805 | 0/0 |
| 1 | 1000003 | 1000003 | full | 26.039350 | 26.024700 | 1.036355 | [0.989245, 1.103568] | inconclusive | 2026102806 | 0/0 |
| 1 | 2000003 | 0 | pass | 2.486400 | 1.940992 | 1.282302 | [1.276917, 1.287711] | favorable | 2026102814 | 0/0 |
| 1 | 2000003 | 0 | resident | 6.139350 | 6.083800 | 1.042002 | [1.011016, 1.090153] | favorable | 2026102815 | 0/0 |
| 1 | 2000003 | 0 | full | 27.039000 | 26.397750 | 1.009480 | [0.990980, 1.027642] | inconclusive | 2026102816 | 0/0 |
| 1 | 2000003 | 2000003 | pass | 2.499936 | 1.939744 | 1.256314 | [1.154314, 1.335873] | favorable | 2026102824 | 0/0 |
| 1 | 2000003 | 2000003 | resident | 13.824300 | 13.149050 | 1.028565 | [0.998786, 1.060182] | inconclusive | 2026102825 | 0/0 |
| 1 | 2000003 | 2000003 | full | 43.715050 | 43.819200 | 0.987339 | [0.946408, 1.022825] | inconclusive | 2026102826 | 0/0 |
| 2 | 1 | 1 | pass | 0.007456 | 0.004480 | 1.599814 | [1.524520, 1.676586] | favorable | 2026103714 | 0/0 |
| 2 | 1 | 1 | resident | 1.429200 | 1.414250 | 1.015799 | [0.954608, 1.083421] | inconclusive | 2026103715 | 0/0 |
| 2 | 1 | 1 | full | 4.974450 | 5.289550 | 1.000670 | [0.957462, 1.049688] | inconclusive | 2026103716 | 0/0 |
| 2 | 256 | 4 | pass | 0.007520 | 0.004832 | 1.547386 | [1.490578, 1.606030] | favorable | 2026103724 | 0/0 |
| 2 | 256 | 4 | resident | 1.401800 | 1.378600 | 0.985592 | [0.927566, 1.042163] | inconclusive | 2026103725 | 0/0 |
| 2 | 256 | 4 | full | 4.873400 | 5.009750 | 0.985618 | [0.954845, 1.018230] | inconclusive | 2026103726 | 0/0 |
| 2 | 257 | 4 | pass | 0.014736 | 0.011936 | 1.239182 | [1.213814, 1.264928] | favorable | 2026103734 | 0/0 |
| 2 | 257 | 4 | resident | 1.369850 | 1.288400 | 1.024753 | [0.979486, 1.067077] | inconclusive | 2026103735 | 0/0 |
| 2 | 257 | 4 | full | 4.761650 | 4.672950 | 0.997173 | [0.962630, 1.032557] | inconclusive | 2026103736 | 0/0 |
| 2 | 65537 | 1 | pass | 0.185840 | 0.157744 | 1.195193 | [1.063585, 1.347505] | favorable | 2026103744 | 0/0 |
| 2 | 65537 | 1 | resident | 2.598300 | 2.579800 | 0.996022 | [0.940714, 1.040924] | inconclusive | 2026103745 | 0/0 |
| 2 | 65537 | 1 | full | 9.635400 | 9.864250 | 1.013756 | [0.976462, 1.058295] | inconclusive | 2026103746 | 0/0 |
| 2 | 65537 | 257 | pass | 0.192560 | 0.154752 | 1.269972 | [1.155343, 1.392764] | favorable | 2026103754 | 0/0 |
| 2 | 65537 | 257 | resident | 2.459500 | 2.282850 | 1.049210 | [1.013091, 1.084033] | favorable | 2026103755 | 0/0 |
| 2 | 65537 | 257 | full | 7.605550 | 7.447550 | 1.019632 | [0.996631, 1.046228] | inconclusive | 2026103756 | 0/0 |
| 2 | 65537 | 65537 | pass | 0.200576 | 0.178160 | 1.229114 | [1.111365, 1.358939] | favorable | 2026103764 | 0/0 |
| 2 | 65537 | 65537 | resident | 2.461950 | 2.435950 | 1.007988 | [0.978447, 1.039356] | inconclusive | 2026103765 | 0/0 |
| 2 | 65537 | 65537 | full | 7.717800 | 7.775500 | 1.002630 | [0.977687, 1.028036] | inconclusive | 2026103766 | 0/0 |
| 2 | 1000003 | 1 | pass | 3.354016 | 2.532720 | 1.295871 | [1.278184, 1.314447] | favorable | 2026103774 | 0/0 |
| 2 | 1000003 | 1 | resident | 15.512250 | 14.709300 | 1.055181 | [1.019287, 1.086791] | favorable | 2026103775 | 0/0 |
| 2 | 1000003 | 1 | full | 67.923350 | 67.657750 | 1.012045 | [0.991461, 1.032323] | inconclusive | 2026103776 | 0/0 |
| 2 | 1000003 | 4 | pass | 1.276464 | 1.001120 | 1.277194 | [1.267075, 1.287465] | favorable | 2026103784 | 0/0 |
| 2 | 1000003 | 4 | resident | 7.894250 | 7.608950 | 1.066532 | [1.018063, 1.131631] | favorable | 2026103785 | 0/0 |
| 2 | 1000003 | 4 | full | 30.900950 | 30.944100 | 0.995553 | [0.960053, 1.031712] | inconclusive | 2026103786 | 0/0 |
| 2 | 1000003 | 257 | pass | 1.258048 | 0.990512 | 1.276655 | [1.265481, 1.288559] | favorable | 2026103794 | 0/0 |
| 2 | 1000003 | 257 | resident | 7.952250 | 7.824650 | 1.043705 | [1.006933, 1.082734] | favorable | 2026103795 | 0/0 |
| 2 | 1000003 | 257 | full | 25.540500 | 25.739000 | 1.001526 | [0.975326, 1.029119] | inconclusive | 2026103796 | 0/0 |
| 2 | 1000003 | 1000003 | pass | 1.261984 | 0.991440 | 1.279557 | [1.267626, 1.291580] | favorable | 2026103804 | 0/0 |
| 2 | 1000003 | 1000003 | resident | 7.999900 | 7.738050 | 1.031747 | [1.003602, 1.059961] | favorable | 2026103805 | 0/0 |
| 2 | 1000003 | 1000003 | full | 24.979300 | 24.633200 | 1.009275 | [0.984227, 1.033633] | inconclusive | 2026103806 | 0/0 |
| 2 | 2000003 | 0 | pass | 2.479040 | 1.727648 | 1.246409 | [1.219597, 1.281280] | favorable | 2026103814 | 0/0 |
| 2 | 2000003 | 0 | resident | 11.296600 | 10.598850 | 1.057045 | [1.040833, 1.072734] | favorable | 2026103815 | 0/0 |
| 2 | 2000003 | 0 | full | 34.691600 | 33.607250 | 1.025998 | [1.004412, 1.046200] | favorable | 2026103816 | 0/0 |
| 2 | 2000003 | 2000003 | pass | 0.430352 | 0.345776 | 1.243721 | [1.241068, 1.246427] | favorable | 2026103824 | 0/0 |
| 2 | 2000003 | 2000003 | resident | 8.695850 | 9.229350 | 0.972861 | [0.920025, 1.028371] | inconclusive | 2026103825 | 0/0 |
| 2 | 2000003 | 2000003 | full | 38.800400 | 38.735950 | 1.013396 | [0.980484, 1.052010] | inconclusive | 2026103826 | 0/0 |

## Reproduction and evidence

Native fixtures are in `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/`:
voxel_boundary_fusion_tests.rs, voxel_boundary_fusion_benchmark.rs and complete
parent compactor/scanner reference .rs/.wgsl files. Run voxel_scan::fused:: with
all features in release mode, including ignored tests and excluding benchmark.
Run benchmark_complete_voxel_boundary_fusion_calls alone with
--ignored --nocapture --test-threads=1 for the paired matrix.

Complete raw arrays, logs and machine-specific validation/provenance/measurement
receipts are ignored under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-boundary-fusion-`.
Every aggregate row is published above. These local receipts are not required by
the production library. Published artifacts omit adapter model names.
