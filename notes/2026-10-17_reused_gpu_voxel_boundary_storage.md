# GPU voxel boundary/index lifetime reuse

Parent: `b0a2b57961215e5e23ff34f611f1703c323f73d3` (merged PR 262).

The owned point-index buffer now holds boundary flags until the existing scan
consumes them. The final write pass reconstructs boundaries from adjacent
inclusive-prefix values, then overwrites the temporary flags with final point
indices. The separate private flag request decreases from 4*N to four bytes.
No new pass, transfer, CPU operation, feature, dependency or public API is added.

## Exact boundary and lifetime reasoning

For nonempty sorted input the original mark shader emits f[0]=1 and f[i] in
0/1 according to key equality. The unchanged scan produces
P[i]=sum(f[0]..f[i]). Its mass is at most N <= u32::MAX, so there is no wrap.
For i>0, P[i] differs from P[i-1] exactly when f[i]=1. The new write shader
uses that comparison and explicitly handles i=0 before loading any predecessor.
Its original cell ID P[i]-1, key/start stores and point-index copy remain.
There is no new subtraction or floating arithmetic in boundary recovery.

The mark pass binds the full owned index buffer at flag binding 2, with the
four-byte flag placeholder at inactive index binding 7. The scan reads the
owned buffer into the original separate private inclusive-prefix buffer.
The write pass binds the owned buffer at active index binding 7 and the
placeholder at inactive flag binding 2. Each group has distinct storage-buffer
handles; the scan source and destination are also distinct. No overlapping
storage bindings or in-place scan are used.

The command order remains mark, complete hierarchical scan, write, scalar copy.
The first scan block pass consumes the mark output; upper hierarchy/offset
passes use their original prefix/sum buffers. The final write starts after
these passes and reads only the completed prefix for boundaries. Every valid
index element is overwritten with its original final value before return.
The four-byte cell-count synchronization still completes before exposing the
owned result. Empty input retains its original no-dispatch/zero-readback path.

The owned index allocation remains outside scratch leases and is never recycled
into the private pool. Its size/usage/ownership are unchanged. Published outputs
therefore survive later calls, input mutation/drop, pool clearing and concurrent
submissions under the same original ownership rules. The four-byte placeholder
is leased until submission and may contain stale data because neither active
mark nor write binding accesses it. Legacy pipeline entrypoints/layouts remain
available; production still uses only mark/write plus its separate scan cache.

| Stage/storage | Parent | Current |
| --- | --- | --- |
| Mark flags | private 4*N-byte flags | owned 4*N-byte indices |
| Scan input | private flags | owned indices holding flags |
| Scan output/hierarchy | separate private prefix/sums | same |
| Write boundary | flags[i] | P[0] boundary, otherwise P[i]!=P[i-1] |
| Write final indices | owned indices | overwrite owned temporary flags |
| Private flag request for N>0 | 4*N bytes | 4 bytes |
| Construction scalar readback | 4 bytes | 4 bytes |

For each nonempty call requested scratch decreases by **4*N-4 bytes**:
4,000,008 bytes for 1,000,003 points and 8,000,008 for 2,000,003 points.
This is additional to PR 262's discarded-counter reduction. No new output,
buffer lease, compute dispatch, explicit copy, map/wait, upload or readback is
introduced. Pool retention still depends on budget and previous contents;
fresh-pool comparisons verify the exact retained difference. These are requested
payloads, excluding driver overhead and physical adapter-memory measurements.

## Correctness and integration

- **383 native correctness tests:** 151 image, 195 voxel/readback, 8 RANSAC,
  8 segmentation, 18 filtering, 2 transfer facade and 1 Gaussian facade.
  Ignored GPU correctness tests are included and benchmarks excluded.
  All-feature/all-target Clippy with warnings denied, CPU-only, minimal/CI WASM
  with warnings denied, formatting and whitespace checks pass.
- **80,211,910 phased parent/CPU words and guards** across **55 paired fixtures**.
  A mark plus complete hierarchy is submitted first. Before overwrite, the
  current owned indices must equal independently computed binary key boundaries;
  parent flags must equal the same values, and both full prefixes must match a
  sequential u64 oracle. Keys/starts and inactive outputs remain poisoned.
  A second submission writes final segments; every final index, key/padding,
  start, prefix and suffix guard is verified. The current actual four-byte
  placeholder, both discarded-counter bindings and unused scan output retain
  poison. This covers zero/1/2/3/255/256/257/511/1025/65537/1,000,003 points;
  singleton, four-point, 257-point, all-one-cell and irregular runs; signed
  extreme keys, reverse input indices, partial blocks and three scan levels.
  The intermediate readback is test instrumentation; production stays in its
  original single submission.
- **24,036,610 complete parent/CPU logical values** in 25 paired conditions
  (50 full calls) check raw device order, owned metadata, derived CPU lengths,
  original four-byte/zero scalar receipts and retained outputs across later
  calls and pool clearing. Existing downstream tests cover concurrent builds,
  stale scratch, attributes, averages, stable sort and input mutation/drop.
- Fresh-pool comparisons at N=1/257/65537 verify **263,168 aggregate bytes**
  saved and unchanged retained resource counts. Warm repeats allocate no new
  scratch; four/zero-byte budgets preserve outputs and bounded retention.
  When supported, timestamp stage names/capacity and query bytes match the
  parent. Adapters without timestamps still run storage/output comparisons.
  The earlier discarded-counter test now enforces its original minimum saving,
  allowing this additional reduction; the new immediate-parent test checks the
  exact additional 4*N-4 bytes.
- **539 original GPU/image sources unchanged** and
  **548 source/executable hashes**. Production changes are confined
  to the private flag size, mark/scan bindings and write-boundary recovery.
  The immediate-parent complete compactor/helpers, shader and pipeline
  constructor/layout/helpers are frozen with name/visibility/import/reference
  and external-cache substitutions only. The measured sources/executable are
  identical to final broad validation.

## Paired completed-call diagnostics

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=write_gpu_resident_compactor_and_full wall_order=balanced checks_drop_excluded=true`. Adapter model names are omitted.

Two surveys each discard three warmups and record thirty pairs for twelve
cases: (N,run)=(1,1), (256,4), (257,4), (65537,1), (65537,257),
(65537,65537), (1000003,1), (1000003,4), (1000003,257),
(1000003,1000003), (2000003,irregular), (2000003,2000003).
Run=0 denotes cycling lengths 1/2/3/255/256/257/1023/4097. All **72 rows** are
retained, without filtering by interval direction or zero duration.

- Isolated write GPU pass: **18 favorable / 6 inconclusive / 0 adverse**.
- Completed resident compactor: **2 favorable / 22 inconclusive / 0 adverse**.
- Upload/compact/readback: **1 favorable / 23 inconclusive / 0 adverse**.

Adverse pointwise intervals: none.

Adopted for the exact requested-storage reduction and unchanged outputs/stages.
The isolated write results support the change on these fixtures, while neither
completed-call route establishes a broad latency benefit.

At 1,000,003 singleton cells, write medians are 4.544688 -> 4.073184 ms and 1.640800 -> 1.499808 ms;
resident 8.620850 -> 8.647950 ms and 8.836400 -> 8.770450 ms; full 40.441850 -> 40.174250 ms and 43.620250 -> 40.973950 ms.
For 1,000,003 points in one cell, write medians are
0.960992 -> 0.824976 ms and 0.962704 -> 0.825088 ms; resident
8.196350 -> 7.987450 ms and 7.408000 -> 7.259800 ms; full
24.496750 -> 24.105650 ms and 21.682800 -> 21.577600 ms.

Timing boundaries:

- GPU write timing starts with independently prepared valid flags/prefixes,
  separate output groups and warmed parent/current pipelines. Setup, mark/scan,
  buffer requests and CPU verification are outside timestamps. Parent/current
  dispatch in one submission in alternating order; two timestamp pairs collect 32 bytes
  per pair. Guarded keys/starts use logical cell extents, whereas full host-call
  timings use the original point-capacity owned buffers. Current write uses an
  actual four-byte inactive flag binding; neither route resets counter storage.
- Resident wall time starts with sorted entries already uploaded and includes
  the full original/current call, mark/scan/write, resource requests/encoding
  and the original four-byte scalar synchronization. It includes the real
  lifetime reuse. Metadata conversion and correctness checks occur afterward.
- Full wall time starts before entry-buffer upload and ends after actual
  to_voxel_segments readback/conversion, including the existing CPU per-cell
  sorting. Checks and drops are outside timing. Every wall output is verified.
- Wall variants use variant=offset^(round%4). Each route has fifteen measured
  parent-first and fifteen current-first pairs; all four variants visit every
  ordinal position seven/eight times. GPU pairs alternate with fifteen orders
  in each direction. Runtime pool policy and three discarded warmups are shared.
  The frozen parent cache is externally supplied; current retains its runtime
  cache lookup. No output rows or measurement surveys are selectively discarded.

Estimates use paired log(old/new) geometric means and 10,000 paired bootstrap
resamples, seeds `2026101713+1000*survey+10*case+routeIndex` (case zero-based;
routes pass/resident/full indexed 0/1/2), bounds at sorted indices 249/9749.
Ratios above one favor current. These are pointwise percentile 95% intervals
without multiple-comparison correction. Zero durations leave ratios unresolved
and remain counted. All thirty samples are retained; no outlier filtering or
automatic retry. Bootstrap analysis follows both timed GPU surveys.

Pre/post two-second process/GPU observations:

- Survey 1: observed-quiet=False; pre={"busy_processes": [], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": true}; post={"busy_processes": [{"cpu_cores": 0.4147150663488327, "id": 38184, "name": "python"}], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}.

- Survey 2: observed-quiet=False; pre={"busy_processes": [{"cpu_cores": 0.9937085446875701, "id": 38184, "name": "python"}], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}; post={"busy_processes": [], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": true}.

These observations cannot establish isolation during every pair. This is one
shared host/backend/device class and fixed workloads, affected by scheduling,
timestamp granularity, thermal/clock state and transfer/CPU costs. Requested
storage savings are structural; timing benefits remain workload-dependent.
All adverse/inconclusive rows remain visible. The results establish no portable,
resident-wide, transfer-inclusive or application-wide speed guarantee.

| Survey | N | Run | Route | Parent ms | Current ms | Paired old/new | Pointwise 95% interval | Result | Seed | Zero old/new |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | pass | 0.003472 | 0.003552 | 0.979974 | [0.911055, 1.055100] | inconclusive | 2026102713 | 0/0 |
| 1 | 1 | 1 | resident | 1.121500 | 1.109400 | 0.994085 | [0.957350, 1.025442] | inconclusive | 2026102714 | 0/0 |
| 1 | 1 | 1 | full | 4.098300 | 4.117200 | 1.007610 | [0.985489, 1.038849] | inconclusive | 2026102715 | 0/0 |
| 1 | 256 | 4 | pass | 0.003600 | 0.003792 | 0.956195 | [0.886894, 1.035060] | inconclusive | 2026102723 | 0/0 |
| 1 | 256 | 4 | resident | 1.097550 | 1.087200 | 1.005747 | [0.981052, 1.029785] | inconclusive | 2026102724 | 0/0 |
| 1 | 256 | 4 | full | 3.996850 | 4.007250 | 1.001708 | [0.961675, 1.045481] | inconclusive | 2026102725 | 0/0 |
| 1 | 257 | 4 | pass | 0.003632 | 0.003824 | 0.943636 | [0.872618, 1.021087] | inconclusive | 2026102733 | 0/0 |
| 1 | 257 | 4 | resident | 1.113850 | 1.157800 | 0.985626 | [0.953535, 1.019461] | inconclusive | 2026102734 | 0/0 |
| 1 | 257 | 4 | full | 3.990700 | 3.990150 | 0.991426 | [0.972879, 1.009803] | inconclusive | 2026102735 | 0/0 |
| 1 | 65537 | 1 | pass | 0.218560 | 0.205904 | 1.079005 | [1.023173, 1.140233] | favorable | 2026102743 | 0/0 |
| 1 | 65537 | 1 | resident | 2.212200 | 2.038150 | 0.999018 | [0.970758, 1.027883] | inconclusive | 2026102744 | 0/0 |
| 1 | 65537 | 1 | full | 8.272950 | 7.908700 | 1.001069 | [0.971329, 1.032837] | inconclusive | 2026102745 | 0/0 |
| 1 | 65537 | 257 | pass | 0.146176 | 0.091024 | 1.773566 | [1.480134, 2.130433] | favorable | 2026102753 | 0/0 |
| 1 | 65537 | 257 | resident | 2.112450 | 2.061750 | 1.007903 | [0.974164, 1.043279] | inconclusive | 2026102754 | 0/0 |
| 1 | 65537 | 257 | full | 6.463000 | 6.529400 | 1.007393 | [0.991546, 1.023248] | inconclusive | 2026102755 | 0/0 |
| 1 | 65537 | 65537 | pass | 0.154688 | 0.101728 | 1.658516 | [1.349276, 2.054812] | favorable | 2026102763 | 0/0 |
| 1 | 65537 | 65537 | resident | 2.194900 | 2.126500 | 1.029696 | [1.001925, 1.058699] | favorable | 2026102764 | 0/0 |
| 1 | 65537 | 65537 | full | 6.940350 | 6.939600 | 1.000000 | [0.974361, 1.025104] | inconclusive | 2026102765 | 0/0 |
| 1 | 1000003 | 1 | pass | 4.544688 | 4.073184 | 1.143813 | [1.103502, 1.220292] | favorable | 2026102773 | 0/0 |
| 1 | 1000003 | 1 | resident | 8.620850 | 8.647950 | 0.998175 | [0.940046, 1.058644] | inconclusive | 2026102774 | 0/0 |
| 1 | 1000003 | 1 | full | 40.441850 | 40.174250 | 1.010268 | [0.971011, 1.063737] | inconclusive | 2026102775 | 0/0 |
| 1 | 1000003 | 4 | pass | 1.132144 | 1.000736 | 1.136568 | [1.128783, 1.144427] | favorable | 2026102783 | 0/0 |
| 1 | 1000003 | 4 | resident | 7.404950 | 7.171100 | 1.042461 | [0.989700, 1.111310] | inconclusive | 2026102784 | 0/0 |
| 1 | 1000003 | 4 | full | 26.203100 | 25.907150 | 1.016948 | [0.990688, 1.048367] | inconclusive | 2026102785 | 0/0 |
| 1 | 1000003 | 257 | pass | 0.963440 | 0.830192 | 1.162320 | [1.156478, 1.168133] | favorable | 2026102793 | 0/0 |
| 1 | 1000003 | 257 | resident | 7.990800 | 7.939050 | 1.008422 | [0.977592, 1.040773] | inconclusive | 2026102794 | 0/0 |
| 1 | 1000003 | 257 | full | 24.215800 | 24.342500 | 0.986370 | [0.962679, 1.010335] | inconclusive | 2026102795 | 0/0 |
| 1 | 1000003 | 1000003 | pass | 0.960992 | 0.824976 | 1.165564 | [1.158902, 1.172306] | favorable | 2026102803 | 0/0 |
| 1 | 1000003 | 1000003 | resident | 8.196350 | 7.987450 | 0.994648 | [0.952318, 1.034296] | inconclusive | 2026102804 | 0/0 |
| 1 | 1000003 | 1000003 | full | 24.496750 | 24.105650 | 1.027166 | [0.994441, 1.062246] | inconclusive | 2026102805 | 0/0 |
| 1 | 2000003 | 0 | pass | 0.221888 | 0.192848 | 1.151420 | [1.149558, 1.153374] | favorable | 2026102813 | 0/0 |
| 1 | 2000003 | 0 | resident | 5.546000 | 5.499850 | 1.000390 | [0.919908, 1.091725] | inconclusive | 2026102814 | 0/0 |
| 1 | 2000003 | 0 | full | 23.517650 | 22.883250 | 1.022094 | [1.000744, 1.046361] | favorable | 2026102815 | 0/0 |
| 1 | 2000003 | 2000003 | pass | 0.221296 | 0.192704 | 1.148378 | [1.147445, 1.149226] | favorable | 2026102823 | 0/0 |
| 1 | 2000003 | 2000003 | resident | 5.781000 | 5.819350 | 1.002013 | [0.978931, 1.025683] | inconclusive | 2026102824 | 0/0 |
| 1 | 2000003 | 2000003 | full | 23.421900 | 23.667300 | 1.006451 | [0.988193, 1.025101] | inconclusive | 2026102825 | 0/0 |
| 2 | 1 | 1 | pass | 0.003472 | 0.003568 | 0.977473 | [0.911433, 1.049698] | inconclusive | 2026103713 | 0/0 |
| 2 | 1 | 1 | resident | 1.233100 | 1.215850 | 0.973911 | [0.908022, 1.037054] | inconclusive | 2026103714 | 0/0 |
| 2 | 1 | 1 | full | 4.477350 | 4.616050 | 0.995366 | [0.973260, 1.018612] | inconclusive | 2026103715 | 0/0 |
| 2 | 256 | 4 | pass | 0.003696 | 0.003792 | 0.955756 | [0.887681, 1.030599] | inconclusive | 2026103723 | 0/0 |
| 2 | 256 | 4 | resident | 1.102400 | 1.090300 | 1.001212 | [0.977481, 1.024946] | inconclusive | 2026103724 | 0/0 |
| 2 | 256 | 4 | full | 3.975350 | 4.034100 | 0.988146 | [0.959243, 1.013549] | inconclusive | 2026103725 | 0/0 |
| 2 | 257 | 4 | pass | 0.003632 | 0.003840 | 0.944332 | [0.869421, 1.025419] | inconclusive | 2026103733 | 0/0 |
| 2 | 257 | 4 | resident | 1.169450 | 1.134400 | 1.005111 | [0.947781, 1.057347] | inconclusive | 2026103734 | 0/0 |
| 2 | 257 | 4 | full | 4.003200 | 4.163200 | 0.999892 | [0.971790, 1.030872] | inconclusive | 2026103735 | 0/0 |
| 2 | 65537 | 1 | pass | 0.219616 | 0.203984 | 1.118865 | [1.068238, 1.172785] | favorable | 2026103743 | 0/0 |
| 2 | 65537 | 1 | resident | 2.353600 | 2.243750 | 1.017641 | [0.963224, 1.070935] | inconclusive | 2026103744 | 0/0 |
| 2 | 65537 | 1 | full | 8.527150 | 8.647400 | 0.987744 | [0.954988, 1.020743] | inconclusive | 2026103745 | 0/0 |
| 2 | 65537 | 257 | pass | 0.145680 | 0.099120 | 1.690668 | [1.390279, 2.045507] | favorable | 2026103753 | 0/0 |
| 2 | 65537 | 257 | resident | 2.313050 | 2.330000 | 1.012632 | [0.976305, 1.050473] | inconclusive | 2026103754 | 0/0 |
| 2 | 65537 | 257 | full | 7.420150 | 7.372350 | 1.031254 | [0.996011, 1.066600] | inconclusive | 2026103755 | 0/0 |
| 2 | 65537 | 65537 | pass | 0.154640 | 0.109856 | 1.522966 | [1.239238, 1.865140] | favorable | 2026103763 | 0/0 |
| 2 | 65537 | 65537 | resident | 2.397600 | 2.255200 | 1.028829 | [0.995877, 1.062473] | inconclusive | 2026103764 | 0/0 |
| 2 | 65537 | 65537 | full | 7.269700 | 7.131150 | 1.006183 | [0.983831, 1.029120] | inconclusive | 2026103765 | 0/0 |
| 2 | 1000003 | 1 | pass | 1.640800 | 1.499808 | 1.099706 | [1.092329, 1.107561] | favorable | 2026103773 | 0/0 |
| 2 | 1000003 | 1 | resident | 8.836400 | 8.770450 | 1.043114 | [1.005089, 1.095048] | favorable | 2026103774 | 0/0 |
| 2 | 1000003 | 1 | full | 43.620250 | 40.973950 | 1.003360 | [0.942376, 1.063124] | inconclusive | 2026103775 | 0/0 |
| 2 | 1000003 | 4 | pass | 3.157408 | 2.774128 | 1.139533 | [1.127687, 1.151786] | favorable | 2026103783 | 0/0 |
| 2 | 1000003 | 4 | resident | 7.850150 | 7.598300 | 1.027989 | [0.988953, 1.073320] | inconclusive | 2026103784 | 0/0 |
| 2 | 1000003 | 4 | full | 27.990900 | 27.503150 | 1.019955 | [0.948265, 1.094231] | inconclusive | 2026103785 | 0/0 |
| 2 | 1000003 | 257 | pass | 0.966480 | 0.829712 | 1.164878 | [1.157799, 1.172190] | favorable | 2026103793 | 0/0 |
| 2 | 1000003 | 257 | resident | 4.563350 | 4.419800 | 1.017098 | [0.970463, 1.060831] | inconclusive | 2026103794 | 0/0 |
| 2 | 1000003 | 257 | full | 16.605650 | 16.600950 | 0.999188 | [0.971608, 1.028209] | inconclusive | 2026103795 | 0/0 |
| 2 | 1000003 | 1000003 | pass | 0.962704 | 0.825088 | 1.165688 | [1.158851, 1.172635] | favorable | 2026103803 | 0/0 |
| 2 | 1000003 | 1000003 | resident | 7.408000 | 7.259800 | 1.024947 | [0.994526, 1.058857] | inconclusive | 2026103804 | 0/0 |
| 2 | 1000003 | 1000003 | full | 21.682800 | 21.577600 | 1.013014 | [0.992966, 1.034165] | inconclusive | 2026103805 | 0/0 |
| 2 | 2000003 | 0 | pass | 0.219872 | 0.189984 | 1.155668 | [1.150367, 1.159303] | favorable | 2026103813 | 0/0 |
| 2 | 2000003 | 0 | resident | 5.510200 | 5.419850 | 0.994336 | [0.936634, 1.032211] | inconclusive | 2026103814 | 0/0 |
| 2 | 2000003 | 0 | full | 23.607400 | 23.300650 | 1.005991 | [0.981148, 1.039201] | inconclusive | 2026103815 | 0/0 |
| 2 | 2000003 | 2000003 | pass | 1.930784 | 1.643792 | 1.174813 | [1.171751, 1.177857] | favorable | 2026103823 | 0/0 |
| 2 | 2000003 | 2000003 | resident | 5.717800 | 5.555300 | 1.035484 | [0.998383, 1.070645] | inconclusive | 2026103824 | 0/0 |
| 2 | 2000003 | 2000003 | full | 25.437750 | 24.895350 | 1.003042 | [0.977420, 1.029295] | inconclusive | 2026103825 | 0/0 |

## Reproduction and local evidence

Native fixtures are in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/`:
voxel_boundary_reuse_tests.rs, voxel_boundary_reuse_reference.rs,
voxel_boundary_reuse_reference.wgsl and voxel_boundary_reuse_benchmark.rs.
Run the `voxel_compact::boundary_reuse::` correctness filter with all features
in release mode, including ignored tests and excluding benchmark. Run
`benchmark_complete_voxel_boundary_reuse_calls` alone with
`--ignored --nocapture --test-threads=1` for the paired matrix.

Complete raw arrays, machine-specific receipts and logs are ignored under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-boundary-reuse-`:
precheck, validation, provenance, measurements and survey JSON files and logs.
Every aggregate row is published above. Local receipts are not required by the
production library, and published benchmark text omits adapter model names.
