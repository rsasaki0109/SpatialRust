# Unused GPU voxel cell counters

Parent: `cbd334d6af1fc2d101c480b8343f366960c68db0` (merged PR 261).

The compactor previously issued one `atomicAdd` per point into a private
point-sized cell-counter buffer, cleared that buffer on every nonempty call,
then discarded it. `GpuVoxelSegments` owns only keys, point indices and cell
starts. Its existing readback computes cell lengths from adjacent starts and
the point count. Downstream GPU reducers also use starts. The counter has no
consumer and is not returned, read back, or used to construct segment outputs.

Remove that discarded atomic update and explicit buffer clear. Request four
bytes for the unused compatibility binding instead of `4*N` bytes. Keep the
binding number/type, pipeline layout and constructors intact, so legacy scan
entrypoints and existing native parent fixtures retain their binding contracts.
The unused shader variable remains declared. No replacement counting/search
algorithm or new CPU work is needed.

## Exact output and resource reasoning

`write_segments` still reads the same flags and inclusive prefix, copies every
point index, and writes the same boundary key/start. The removed atomic targets
only `cell_counts`; none of the retained shader operations reads it. The full
host call passes only keys/indices/starts to `GpuVoxelSegments::new`. The
unchanged `read_segment_metadata` derives lengths using
`end = next_start` (or N for the final cell), then `count = end - start`.
Thus removing counter values cannot affect owned outputs or derived counts.

The compatibility buffer is never accessed by any production compactor shader
entrypoint. Four bytes satisfy its existing storage binding; stale contents
are permitted and do not need explicit clearing. Each invocation still leases
its own buffer until submission, preserving pool/queue/concurrency ownership.
Wgpu/driver allocation or initialization overhead is outside these source-level
request counts. The existing mark, scan hierarchy and write dispatches, scalar
copy/map/wait and all explicit input/output transfers are retained.

For every nonempty call, requested scratch decreases by **4*N - 4 bytes**;
one explicit `4*N`-byte clear and N discarded atomic read-modify-writes disappear.
The number of buffer leases and compute passes is unchanged. Empty calls keep
the original no-dispatch/zero-readback path. Pool retention remains subject to
the configured budget, previous contents and reuse; fresh-pool comparisons
verify the exact byte difference. Live output buffers and uniform/query/staging
memory are excluded from pool byte statistics as before.

| Source request/work | Parent | Current |
| --- | ---: | ---: |
| Private counter binding bytes for N>0 | 4*N | 4 |
| Explicit counter clear bytes per nonempty call | 4*N | 0 |
| Discarded counter atomic additions | N | 0 |
| Construction scalar readback | 4 bytes | 4 bytes |
| Owned result buffers | keys/indices/starts | keys/indices/starts |

At 1,000,003 points this removes 4,000,008 requested bytes; at 2,000,003,
8,000,008 bytes. These are requested payloads, not physical adapter-memory
measurements or allocation-count guarantees under every reuse/budget state.

## Correctness and integration

- **380 native correctness tests:** 151 image, 192 voxel/readback, 8 RANSAC,
  8 segmentation, 18 filtering, 2 transfer facade and 1 Gaussian facade.
  Ignored GPU correctness tests are included and benchmarks excluded.
  All-feature/all-target Clippy with warnings denied, CPU-only, minimal/CI
  WASM with warnings denied, formatting and whitespace checks pass.
- **50,785,230 raw parent/CPU words and guards** across **55 paired fixtures**
  (110 write dispatches) check physical keys/padding, starts, device index order,
  flags/prefixes and unused scan-output poison. Parent counter words match an
  independent CPU segmentation; the current actual four-byte binding remains
  poisoned and untouched. Output suffix guards and inactive parent count tails
  are checked. Counts 0/1/2/3/255/256/257/511/1025/65537/1,000,003 combine with
  singleton, four-point, 257-point, all-one-cell and irregular run lengths
  (1/2/3/255/256/257/1023/4097). Signed-extreme keys and reversed point indices
  exercise storage layout and original host sorting separately.
- **24,036,610 complete parent/CPU logical values** in 25 paired conditions
  (50 complete compactor calls)
  compare owned cell counts, point counts, scalar receipts, full CPU metadata
  and raw device order for zero/tail/hierarchy/large cases. Earlier outputs
  remain valid across later calls and pool clearing.
- Fresh-pool comparisons for N=1/257/65537 verify **263,168 aggregate bytes**
  saved and unchanged retained resource counts. On timestamp-capable adapters,
  stage names/capacity and query bytes match the parent; other adapters still
  run the storage/output checks without a profiler. Repeated warm calls allocate no new scratch;
  budgets of four/zero bytes preserve exact outputs and bounded retention.
  Existing full-sort/downstream tests also cover concurrent calls, attribute
  gathers/averages, stale scratch and zero/shrinking budgets.
- **536 original GPU/image sources unchanged** and
  **544 source/executable hashes**. The only production changes are
  counter request size, explicit-clear removal and discarded atomic removal.
  All other host and shader tokens match the immediate parent. The full parent
  compactor/helpers, pipeline constructor/layout/helpers and shader are frozen
  with name/visibility/import/shader-reference and external-cache substitutions
  only. The same measured sources/executable pass final broad validation.

## Paired completed-call diagnostics

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=write_gpu_resident_compactor_and_full wall_order=balanced checks_drop_excluded=true`. Adapter model names are omitted.

Two surveys each discard three warmups and measure thirty pairs for twelve
cases: (N,run)=(1,1), (256,4), (257,4), (65537,1), (65537,257),
(65537,65537), (1000003,1), (1000003,4), (1000003,257),
(1000003,1000003), (2000003,irregular), (2000003,2000003).
Irregular is encoded as run=0 and cycles the lengths described above. All
**72 rows** remain below, including adverse/inconclusive/zero-duration results.

- Isolated write GPU pass: **9 favorable / 14 inconclusive / 1 adverse**.
- Completed resident compactor: **6 favorable / 18 inconclusive / 0 adverse**.
- Upload/compact/readback: **3 favorable / 21 inconclusive / 0 adverse**.

Adverse pointwise intervals: survey 1, N=2000003, run=0, pass: 0.994675 [0.991214, 0.998187].

At 1,000,003 singleton cells, write medians are 1.923728 -> 1.649696 ms and 1.909696 -> 1.640272 ms;
resident 8.963350 -> 8.512450 ms and 8.697800 -> 8.204100 ms; full 41.221400 -> 41.374600 ms and 38.524950 -> 38.424650 ms.
At 1,000,003 points in one cell, write medians are
0.981120 -> 0.980208 ms and 0.978496 -> 0.977072 ms; resident
7.889350 -> 7.532550 ms and 7.700450 -> 7.582050 ms; full
24.314450 -> 24.760800 ms and 24.073400 -> 24.034350 ms.

The all-one-cell million-point pass has no resolved benefit in either survey.
The two-million-point irregular pass has a small adverse interval in survey 1
and a favorable interval in survey 2. Absolute durations also vary sharply
between some cases/surveys even when pre/post observations are quiet. Adoption
rests on removing unused work and bounded requested storage, supported by the
repeated singleton write-pass gains; it does not assume universal dense-cloud,
resident or transfer-inclusive latency improvements. All rows remain separate.

Timing boundaries:

- Write-pass timing starts with identical precomputed flags/prefixes and input
  entries, separate output buffers/groups and warmed parent/current pipelines.
  The parent counter is reset once per pair outside timestamps. Parent/current
  dispatch in one submission with alternating order; output verification occurs
  after all rounds. Two query intervals collect 32 bytes per pair. Raw keys and
  starts are allocated to logical cell extents plus suffix guards; host-call
  measurements use the original full point-capacity owned buffers.
- Resident wall time starts with sorted entries already uploaded and includes
  host resource requests/encoding, mark/scan/write and the original four-byte
  scalar synchronization. It is a completed call with a retained input, and
  still includes that scalar transfer. Full metadata conversion and correctness
  checks occur afterward, outside timing.
- Full wall time starts before entry-buffer upload, includes complete compaction
  and ends after actual `to_voxel_segments` readback/conversion, including the
  pre-existing CPU per-cell index sorting. Output/reference comparisons and
  object drops are outside timing. Every wall-call output is checked afterward.
- Wall variants use `variant=offset^(round%4)`. After warmup each route has
  fifteen parent-first and fifteen current-first pairs; each of four variants
  occupies every ordinal position seven/eight times. GPU pairs also use fifteen
  orders in each direction. Both routes share runtime pool policy and warm-up.
  The parent cache is externally supplied; frozen host/planner behavior is
  otherwise identical, while the current call retains its runtime cache lookup.

Estimates use paired log(old/new) geometric means and 10,000 paired bootstrap
resamples with seeds `2026101712+1000*survey+10*case+routeIndex` (case zero-based;
routes pass/resident/full indexed 0/1/2), sorted bounds 249/9749. Ratios above one
favor current. These are pointwise percentile 95% intervals with no multiple
comparison correction. Zero durations leave ratios unresolved and are counted
explicitly. All thirty samples are retained; no outlier filtering, selective
retry or discarded measurement surveys. Bootstrap analysis starts after both
GPU surveys finish.

Pre/post two-second process/GPU observations:

- Survey 1: observed-quiet=False; pre={"busy_processes": [{"cpu_cores": 1.273608349893522, "id": 24044, "name": "python"}], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": false}; post={"busy_processes": [], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": true}.

- Survey 2: observed-quiet=True; pre={"busy_processes": [], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": true}; post={"busy_processes": [], "gpu_samples": [[[0, 0]], [[0, 0]]], "observed_quiet": true}.

These observations cannot establish isolation during every timed pair. This
is one shared host/backend/device class and fixed workloads, with timestamp
granularity, scheduling, thermals and transfer/CPU costs affecting results.
Source request/atomic reductions are structural; timings establish no portable
or application-wide speed guarantee. All adverse/inconclusive rows remain.

| Survey | N | Run | Route | Parent ms | Current ms | Paired old/new | Pointwise 95% interval | Result | Seed | Zero old/new |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | pass | 0.003776 | 0.003440 | 1.070053 | [0.968312, 1.207782] | inconclusive | 2026102712 | 0/0 |
| 1 | 1 | 1 | resident | 1.154800 | 1.186100 | 0.981418 | [0.952365, 1.012906] | inconclusive | 2026102713 | 0/0 |
| 1 | 1 | 1 | full | 4.244800 | 4.343050 | 1.001025 | [0.972810, 1.028521] | inconclusive | 2026102714 | 0/0 |
| 1 | 256 | 4 | pass | 0.003712 | 0.003520 | 1.032654 | [0.964209, 1.104653] | inconclusive | 2026102722 | 0/0 |
| 1 | 256 | 4 | resident | 1.044300 | 1.040400 | 1.008596 | [0.978550, 1.043392] | inconclusive | 2026102723 | 0/0 |
| 1 | 256 | 4 | full | 3.813600 | 3.788150 | 1.001342 | [0.973977, 1.025370] | inconclusive | 2026102724 | 0/0 |
| 1 | 257 | 4 | pass | 0.003712 | 0.003664 | 0.978271 | [0.913240, 1.045570] | inconclusive | 2026102732 | 0/0 |
| 1 | 257 | 4 | resident | 1.146400 | 1.161600 | 0.978126 | [0.919716, 1.022065] | inconclusive | 2026102733 | 0/0 |
| 1 | 257 | 4 | full | 4.133600 | 4.085300 | 1.021803 | [1.000049, 1.045677] | favorable | 2026102734 | 0/0 |
| 1 | 65537 | 1 | pass | 0.297248 | 0.252160 | 1.189449 | [1.133492, 1.250270] | favorable | 2026102742 | 0/0 |
| 1 | 65537 | 1 | resident | 1.363750 | 1.341650 | 1.023828 | [0.988658, 1.061625] | inconclusive | 2026102743 | 0/0 |
| 1 | 65537 | 1 | full | 5.964800 | 5.958750 | 1.002347 | [0.979915, 1.024798] | inconclusive | 2026102744 | 0/0 |
| 1 | 65537 | 257 | pass | 0.160448 | 0.160592 | 1.021025 | [0.967747, 1.078860] | inconclusive | 2026102752 | 0/0 |
| 1 | 65537 | 257 | resident | 2.237550 | 2.143500 | 1.015088 | [0.978159, 1.052684] | inconclusive | 2026102753 | 0/0 |
| 1 | 65537 | 257 | full | 6.691650 | 6.610600 | 1.004701 | [0.979235, 1.030728] | inconclusive | 2026102754 | 0/0 |
| 1 | 65537 | 65537 | pass | 0.167584 | 0.155488 | 1.031006 | [0.978824, 1.086254] | inconclusive | 2026102762 | 0/0 |
| 1 | 65537 | 65537 | resident | 2.045600 | 2.069000 | 1.006896 | [0.985524, 1.029232] | inconclusive | 2026102763 | 0/0 |
| 1 | 65537 | 65537 | full | 6.528000 | 6.586850 | 1.002264 | [0.977286, 1.028961] | inconclusive | 2026102764 | 0/0 |
| 1 | 1000003 | 1 | pass | 1.923728 | 1.649696 | 1.167374 | [1.165387, 1.169602] | favorable | 2026102772 | 0/0 |
| 1 | 1000003 | 1 | resident | 8.963350 | 8.512450 | 1.066720 | [1.029704, 1.121892] | favorable | 2026102773 | 0/0 |
| 1 | 1000003 | 1 | full | 41.221400 | 41.374600 | 0.992962 | [0.915822, 1.079392] | inconclusive | 2026102774 | 0/0 |
| 1 | 1000003 | 4 | pass | 3.373920 | 3.209936 | 1.079903 | [1.053196, 1.130570] | favorable | 2026102782 | 0/0 |
| 1 | 1000003 | 4 | resident | 7.802600 | 7.823650 | 1.003379 | [0.963742, 1.042408] | inconclusive | 2026102783 | 0/0 |
| 1 | 1000003 | 4 | full | 28.439450 | 27.629250 | 1.004409 | [0.919424, 1.098416] | inconclusive | 2026102784 | 0/0 |
| 1 | 1000003 | 257 | pass | 0.980000 | 0.977120 | 0.998543 | [0.993406, 1.003683] | inconclusive | 2026102792 | 0/0 |
| 1 | 1000003 | 257 | resident | 4.994850 | 4.671550 | 1.039306 | [0.999133, 1.079765] | inconclusive | 2026102793 | 0/0 |
| 1 | 1000003 | 257 | full | 19.022050 | 19.173750 | 1.007758 | [0.987259, 1.030282] | inconclusive | 2026102794 | 0/0 |
| 1 | 1000003 | 1000003 | pass | 0.981120 | 0.980208 | 0.997920 | [0.992878, 1.002910] | inconclusive | 2026102802 | 0/0 |
| 1 | 1000003 | 1000003 | resident | 7.889350 | 7.532550 | 1.051833 | [1.020276, 1.085172] | favorable | 2026102803 | 0/0 |
| 1 | 1000003 | 1000003 | full | 24.314450 | 24.760800 | 1.009865 | [0.984815, 1.038743] | inconclusive | 2026102804 | 0/0 |
| 1 | 2000003 | 0 | pass | 1.921504 | 1.933248 | 0.994675 | [0.991214, 0.998187] | adverse | 2026102812 | 0/0 |
| 1 | 2000003 | 0 | resident | 3.999650 | 3.921250 | 1.046898 | [0.981533, 1.140049] | inconclusive | 2026102813 | 0/0 |
| 1 | 2000003 | 0 | full | 21.401700 | 20.994500 | 1.018051 | [0.990298, 1.048052] | inconclusive | 2026102814 | 0/0 |
| 1 | 2000003 | 2000003 | pass | 0.222128 | 0.221072 | 1.004750 | [1.002737, 1.006752] | favorable | 2026102822 | 0/0 |
| 1 | 2000003 | 2000003 | resident | 5.663250 | 5.672400 | 0.994886 | [0.958255, 1.028300] | inconclusive | 2026102823 | 0/0 |
| 1 | 2000003 | 2000003 | full | 23.769250 | 24.070600 | 1.003941 | [0.984758, 1.026197] | inconclusive | 2026102824 | 0/0 |
| 2 | 1 | 1 | pass | 0.003536 | 0.003472 | 1.001871 | [0.927344, 1.080106] | inconclusive | 2026103712 | 0/0 |
| 2 | 1 | 1 | resident | 1.251050 | 1.241150 | 1.008772 | [0.946061, 1.086242] | inconclusive | 2026103713 | 0/0 |
| 2 | 1 | 1 | full | 4.471400 | 4.571000 | 1.007777 | [0.974982, 1.045286] | inconclusive | 2026103714 | 0/0 |
| 2 | 256 | 4 | pass | 0.003648 | 0.003536 | 1.033794 | [0.966604, 1.104776] | inconclusive | 2026103722 | 0/0 |
| 2 | 256 | 4 | resident | 1.111550 | 1.117450 | 1.022524 | [0.984192, 1.066920] | inconclusive | 2026103723 | 0/0 |
| 2 | 256 | 4 | full | 4.036700 | 4.066200 | 0.990043 | [0.954787, 1.021546] | inconclusive | 2026103724 | 0/0 |
| 2 | 257 | 4 | pass | 0.003648 | 0.003664 | 0.975180 | [0.913076, 1.040882] | inconclusive | 2026103732 | 0/0 |
| 2 | 257 | 4 | resident | 1.140350 | 1.139450 | 0.991034 | [0.957548, 1.021966] | inconclusive | 2026103733 | 0/0 |
| 2 | 257 | 4 | full | 4.076800 | 4.073850 | 0.993687 | [0.965857, 1.018408] | inconclusive | 2026103734 | 0/0 |
| 2 | 65537 | 1 | pass | 0.298816 | 0.271472 | 1.151442 | [1.070265, 1.241104] | favorable | 2026103742 | 0/0 |
| 2 | 65537 | 1 | resident | 1.593100 | 1.667700 | 1.067905 | [1.022106, 1.123767] | favorable | 2026103743 | 0/0 |
| 2 | 65537 | 1 | full | 6.783750 | 6.413450 | 1.020460 | [0.993298, 1.053651] | inconclusive | 2026103744 | 0/0 |
| 2 | 65537 | 257 | pass | 0.158320 | 0.161328 | 0.983841 | [0.936030, 1.037945] | inconclusive | 2026103752 | 0/0 |
| 2 | 65537 | 257 | resident | 1.975100 | 1.970900 | 0.989037 | [0.955958, 1.019568] | inconclusive | 2026103753 | 0/0 |
| 2 | 65537 | 257 | full | 6.331900 | 6.220200 | 1.021664 | [1.004197, 1.039772] | favorable | 2026103754 | 0/0 |
| 2 | 65537 | 65537 | pass | 0.160864 | 0.163568 | 0.982006 | [0.936043, 1.033551] | inconclusive | 2026103762 | 0/0 |
| 2 | 65537 | 65537 | resident | 1.958650 | 1.915450 | 1.053090 | [1.023744, 1.086734] | favorable | 2026103763 | 0/0 |
| 2 | 65537 | 65537 | full | 6.162600 | 6.113450 | 1.015163 | [1.000688, 1.031392] | favorable | 2026103764 | 0/0 |
| 2 | 1000003 | 1 | pass | 1.909696 | 1.640272 | 1.165720 | [1.162618, 1.168771] | favorable | 2026103772 | 0/0 |
| 2 | 1000003 | 1 | resident | 8.697800 | 8.204100 | 1.040433 | [1.003828, 1.070533] | favorable | 2026103773 | 0/0 |
| 2 | 1000003 | 1 | full | 38.524950 | 38.424650 | 1.011112 | [0.981908, 1.041511] | inconclusive | 2026103774 | 0/0 |
| 2 | 1000003 | 4 | pass | 1.203424 | 1.135696 | 1.059443 | [1.057737, 1.061163] | favorable | 2026103782 | 0/0 |
| 2 | 1000003 | 4 | resident | 4.922700 | 6.152400 | 1.006206 | [0.957734, 1.049866] | inconclusive | 2026103783 | 0/0 |
| 2 | 1000003 | 4 | full | 22.600800 | 23.044150 | 1.000808 | [0.983518, 1.016377] | inconclusive | 2026103784 | 0/0 |
| 2 | 1000003 | 257 | pass | 0.979936 | 0.982672 | 0.994836 | [0.988493, 1.001177] | inconclusive | 2026103792 | 0/0 |
| 2 | 1000003 | 257 | resident | 7.566550 | 7.520350 | 1.014057 | [0.989891, 1.039368] | inconclusive | 2026103793 | 0/0 |
| 2 | 1000003 | 257 | full | 23.153150 | 22.916700 | 1.002733 | [0.988998, 1.017543] | inconclusive | 2026103794 | 0/0 |
| 2 | 1000003 | 1000003 | pass | 0.978496 | 0.977072 | 0.999137 | [0.994226, 1.003880] | inconclusive | 2026103802 | 0/0 |
| 2 | 1000003 | 1000003 | resident | 7.700450 | 7.582050 | 1.025456 | [0.996289, 1.058633] | inconclusive | 2026103803 | 0/0 |
| 2 | 1000003 | 1000003 | full | 24.073400 | 24.034350 | 0.992489 | [0.971839, 1.011386] | inconclusive | 2026103804 | 0/0 |
| 2 | 2000003 | 0 | pass | 0.230752 | 0.222432 | 1.038059 | [1.035395, 1.040694] | favorable | 2026103812 | 0/0 |
| 2 | 2000003 | 0 | resident | 4.005200 | 3.941450 | 0.998141 | [0.934953, 1.040055] | inconclusive | 2026103813 | 0/0 |
| 2 | 2000003 | 0 | full | 21.692950 | 21.642350 | 0.999825 | [0.985457, 1.015045] | inconclusive | 2026103814 | 0/0 |
| 2 | 2000003 | 2000003 | pass | 1.967968 | 1.959888 | 1.004340 | [1.001364, 1.007361] | favorable | 2026103822 | 0/0 |
| 2 | 2000003 | 2000003 | resident | 6.084800 | 5.714450 | 1.048886 | [1.010118, 1.104875] | favorable | 2026103823 | 0/0 |
| 2 | 2000003 | 2000003 | full | 23.955950 | 24.050200 | 1.002039 | [0.975519, 1.026791] | inconclusive | 2026103824 | 0/0 |

## Reproduction and local evidence

Native fixtures are in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/`:
`voxel_unused_counts_tests.rs`, `voxel_unused_counts_reference.rs`,
`voxel_unused_counts_reference.wgsl` and `voxel_unused_counts_benchmark.rs`.
Run the `voxel_compact::unused_counts::` correctness filter with all features
in release mode, including ignored tests and excluding benchmark. Run
`benchmark_complete_voxel_unused_counts_calls` alone with
`--ignored --nocapture --test-threads=1` for the full paired matrix.

Complete raw arrays and local machine-specific receipts/logs are ignored under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-unused-counts-`:
precheck, validation, provenance, measurements and survey JSON files and logs.
Every aggregate row is published above; these local receipts are not required
by the production library. Published benchmark text omits adapter model names.
