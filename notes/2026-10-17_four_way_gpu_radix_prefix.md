# Deferred four-way GPU radix prefix candidate

Parent: `779ab415b77f56be07ec7bd25c79bf3559393324` (merged PR 258).

**Decision: retain the binary production scan.** The four-way candidate is
correct, but the first two complete paired surveys repeat small prefix and
whole-sort GPU regressions at 2,000,003 points for identical/repeated keys.
In those initial runs, 1,000,003 random/extreme/tied points improve prefix work
about 2% while whole-sort GPU differences are about 0.1%. The final native-only
survey favors all nine cached prefix intervals, but just one whole-sort GPU
interval; none of its cached resident/full intervals establishes an improvement,
and one resident interval is adverse. Completed upload/sort/readback has no
established improvement in any survey. Unconditional adoption has no convincing
overall benefit. Keep the candidate, full parent, CPU checks and benchmark native-only.
No adapter-specific threshold or backend policy is introduced from this data.

All original production GPU/image sources remain identical to the parent. The
existing small path, 2048-workgroup threshold, stable histogram/word ranks,
scatter/sign/digit order, Rust sort body, 24 digits/72 passes/one submission,
cache, scratch budget/labels/sizes, ownership/errors/profiling, explicit
transfers, public builders and CPU paths remain. No public API, dependency,
core, buffer/pass/transfer or unsafe boundary changes.

## Exact candidate arithmetic and synchronization

The cached prefix entry scans 256 vec2<u32> chunk totals for each of 16 digits.
Component x is the current digit's histogram mass; y is cached earlier-digit
mass. Both are bounded by the point count, and their masses are disjoint. The
original load barrier ensures all lanes finish reading their own offset slice
before it is overwritten with final scatter positions.

The candidate's upsweep strides 1/4/16/64 combine four contiguous child totals.
The original root handoff saves total y as bucket_base and zeroes the root.
Downsweep strides 64/16/4/1 distribute the parent exclusive prefix plus earlier
sibling sums to the four child ends. Every active lane owns disjoint slots at
each level and reads its inputs into locals before stores. All 256 lanes join
the following barrier, including lanes whose chunks are empty. No subgroup
assumption or optional feature is introduced.

The resulting x is the exclusive sum of previous chunks for the current digit.
Adding bucket_base and sequential local histogram prefixes reproduces the
original scatter positions. Nonnegative component/subtree totals and combined
offsets are bounded by the u32 point mass, including max-u32 synthetic mass.
Regrouping is exact; independent CPU oracles accumulate u64 and narrow checked.

| Cached tree source work per digit | Binary parent | Four-way candidate |
| --- | ---: | ---: |
| Upsweep stages / barriers | 8 | 4 |
| Downsweep stages / barriers | 8 | 4 |
| Total barriers including load/root handoffs | 18 | 10 |
| vec2 integer additions in the tree | 510 | 510 |

These source counts do not establish generated instructions, transactions or
latency. Shared capacity remains 256 vec2<u32> values (512 u32 words). Outside those two cached
loops, the complete candidate shader matches the frozen parent. Production
continues to use the frozen binary shader.

## Final correctness and provenance

- **371 final native correctness tests:** 145 image, 189 voxel/readback,
  8 RANSAC, 8 segmentation, 18 filtering, 2 image-transfer facade and 1 Gaussian
  facade. Include ignored native GPU correctness tests and exclude benchmarks.
  All-feature/all-target Clippy with warnings denied, CPU-only, minimal/CI WASM
  with warnings denied, formatting and whitespace pass.
- **51,565,932 exact raw offset/histogram/guard words** over 570
  executions: nineteen group counts, five distributions, two specializations
  and parent/production/candidate. Counts cover 1/3/4, 63/64/65, 255/256/257,
  511/512/513, 2047/2048/2049, 4095/4096/4097 and 32769. Distributions are zero,
  first/last single-hot max-u32, balanced max-u32 and uneven max-u32 mass.
  Every offset and unchanged histogram word matches the independent u64
  oracle. Poisoned output/histogram suffixes and unrelated bindings remain.
  Maximum mass is synthetic prefix data, not a full max-u32 point allocation.
- **225,286,624 complete stable-sort/retained words** in 83 complete
  calls plus saved output checks. Parent, unchanged production and candidate
  cover zero/one, 257, 65537, 524287/524288/524289, 1000003 and 2000003 points.
  Random signed/extreme/tied, identical and repeated reverse/modular keys use
  deliberately decreasing point IDs. A stable CPU sort on only signed x/y/z
  verifies original tie order, independent of the fourth word. Zero sentinel,
  original allocation identity, 0/72 timestamp stages and 0/1152 query bytes
  remain. Before the largest same-size group, the test clears cached scratch
  while preserving live outputs because mixed sizes can fill the default
  bounded retention budget. Repeated production/candidate largest sorts then
  allocate no new pooled scratch. Twelve saved outputs survive scratch reuse;
  earlier outputs also survive cache clearing. Production policy is unchanged.
- **522 original GPU/image source files unchanged**, with **530 final source/
  executable hashes**. Only the original native test file appends a module.
  The full immediate-parent shader is frozen; production is explicitly checked
  against it. The complete parent sort body is frozen with only its name and an
  external pipeline selector changed. Its same threshold/layout/scratch/
  dispatch/profiling/control flow is used for native candidate comparisons.
- The initial validated production prototype's **529 source/executable hashes**
  and every corresponding file are archived in an ignored ZIP and checked
  against the preserved manifest. Its shader bytes exactly match the retained
  candidate. Both initial surveys, original raw arrays and final native survey
  remain. Initial validation also passed 371 tests, with 34,377,292 raw words
  and 147,524,412 full-sort words, before restoring production and adding the
  third comparison variant. Counts above describe the final native fixtures.

## Paired measurement phases

Three sequential surveys each use three warmups and thirty measured pairs
across twelve cases and four boundaries. All **144** rows are retained below.
Small controls use random pattern 0 at 257, 65537 and 524288 points. Cached
cases are 524289, 1000003 and 2000003 across patterns 0=random signed/extremes/
ties, 1=identical keys, 2=repeated reverse/modular keys. Point IDs decrease.

Runs 1/2 (`production-prototype`) compare the full frozen parent host body with
the then-live production prototype using the same four-way shader. Its live
runtime cache and the frozen parent's prebuilt cache are warm. These original
observations caused the deferral; they are preserved, not replaced by the final
fixture-only run. Run 3 (`native-only-candidate`) validates the final retained
benchmark: both variants use the complete frozen host body with prebuilt
parent/candidate pipeline selectors. Final production keeps the parent path.
Wall-time cache-lookup details differ between phases; do not pool across phases
or present the retained candidate as a new public resident API.

| Phase | Boundary | All conditions | Cached large path | Unchanged small controls |
| --- | --- | --- | --- | --- |
| production-prototype | prefix | 13 favorable / 7 inconclusive / 4 adverse | 12 favorable / 2 inconclusive / 4 adverse | 1 favorable / 5 inconclusive / 0 adverse |
| production-prototype | sort_gpu | 4 favorable / 16 inconclusive / 4 adverse | 4 favorable / 10 inconclusive / 4 adverse | 0 favorable / 6 inconclusive / 0 adverse |
| production-prototype | resident | 2 favorable / 21 inconclusive / 1 adverse | 1 favorable / 17 inconclusive / 0 adverse | 1 favorable / 4 inconclusive / 1 adverse |
| production-prototype | full | 0 favorable / 21 inconclusive / 3 adverse | 0 favorable / 18 inconclusive / 0 adverse | 0 favorable / 3 inconclusive / 3 adverse |
| native-only-candidate | prefix | 9 favorable / 2 inconclusive / 1 adverse | 9 favorable / 0 inconclusive / 0 adverse | 0 favorable / 2 inconclusive / 1 adverse |
| native-only-candidate | sort_gpu | 1 favorable / 10 inconclusive / 1 adverse | 1 favorable / 8 inconclusive / 0 adverse | 0 favorable / 2 inconclusive / 1 adverse |
| native-only-candidate | resident | 1 favorable / 10 inconclusive / 1 adverse | 0 favorable / 8 inconclusive / 1 adverse | 1 favorable / 2 inconclusive / 0 adverse |
| native-only-candidate | full | 0 favorable / 11 inconclusive / 1 adverse | 0 favorable / 9 inconclusive / 0 adverse | 0 favorable / 2 inconclusive / 1 adverse |

At 1,000,003 random/extreme/tied points, prefix medians are run 1: 0.422080 -> 0.414224 ms; run 2: 0.421408 -> 0.414896 ms; run 3: 0.419072 -> 0.411120 ms;
GPU sort sums are run 1: 7.741344 -> 7.728976 ms; run 2: 7.782608 -> 7.777424 ms; run 3: 7.749472 -> 7.757952 ms. At 2,000,003 identical keys, prefix
medians are run 1: 0.759280 -> 0.761232 ms; run 2: 0.764736 -> 0.765808 ms; run 3: 0.764976 -> 0.754496 ms; GPU sort sums are
run 1: 11.540880 -> 11.545568 ms; run 2: 11.566608 -> 11.570112 ms; run 3: 11.550336 -> 11.543952 ms. For repeated keys, prefix medians are
run 1: 0.759328 -> 0.761248 ms; run 2: 0.764912 -> 0.766544 ms; run 3: 0.765120 -> 0.754448 ms. All adverse/inconclusive conditions, including
unchanged small-control completed calls, remain in the table. Small-control
differences are not attributed to the unchanged production algorithm. The
final fixture survey is a separate measurement phase, not grounds to discard
the repeated initial regressions or claim application/transfer speedups.

- `prefix`: sum of 24 prefix compute-pass timestamps in a complete stable sort,
  using the actual evolving keys and histograms across all 24 digits.
- `sort_gpu`: sum of all 72 histogram/prefix/scatter compute-pass timestamps.
  Both GPU boundaries come from the same separate stage-measurement rounds.
  Parent/candidate use alternating order within one command buffer, separate
  entry/histogram/offset/rank buffers and prebuilt bind groups. A 16*N-byte
  device seed copy resets each sort before its first pass, outside timestamps.
  Query collection returns 2304 bytes per pair; final outputs match the stable
  CPU oracle. GPU sums exclude gaps, host work, allocation, upload, reset,
  query collection, payload readback and CPU waits.
- `resident`: completed internal sort call from uploaded entries and an idle
  queue; includes scratch acquisition/recycling, controls, bind groups, encoding,
  72 passes, submission and explicit wait. Payload readback/assertions follow
  timing. No profiler is attached. Phase-specific cache selection is described
  above; the final candidate uses a native frozen-host fixture.
- `full`: input upload, that complete sort call and the existing native helper's
  owned Vec<u32> readback; includes staging, copy submission, map/wait and CPU
  copy. Transfers are 16*N bytes each way. Key generation, voxel compaction,
  public filters and complete applications are outside this boundary.

Four resident/full variants rotate by round. CPU fixtures, compilation,
assertions and returned-value/resource destruction are outside timing. Resident
upload finishes before its timer. Both variants share the original scratch
pool; GPU-stage instrumentation is separate from unprofiled wall timers.

Ratios are geometric means of paired parent/candidate samples; above one
favors the candidate. Pointwise percentile intervals resample paired log ratios
10,000 times with seed `2026101709 + 1000*run + 10*case + route_index`, route order
prefix/sort_gpu/resident/full, sorted bounds 249/9749. No multiple-comparison
correction is applied. Zero samples would suppress ratios. All raw samples,
phase-specific source/executable manifests and initial archive remain ignored.

Run 1 (production-prototype): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=prefix_gpu_sort_gpu_resident_sort_and_upload_sort_readback checks_drop_excluded=true`; observed-quiet **false**; busy-process counts 0 before / 3 after. GPU/memory utilization `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 2 (production-prototype): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=prefix_gpu_sort_gpu_resident_sort_and_upload_sort_readback checks_drop_excluded=true`; observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 3 (native-only-candidate): `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=prefix_gpu_sort_gpu_resident_sort_and_upload_sort_readback checks_drop_excluded=true`; observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization `[[[2, 0]], [[0, 0]]]` before / `[[[0, 0]], [[5, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all sampled
GPU/memory utilization at most 5%. Snapshots are two seconds apart; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post observations
cannot establish isolation during every pair. These are shared-host diagnostics,
with no portable latency/throughput guarantee. Adapter model names are omitted;
process identities remain in ignored receipts.

| Phase / run | Points / pattern / cached | Boundary | Parent median ms | Candidate median ms | Paired parent/candidate ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| production-prototype / 1 | 257 / 0 / false | prefix | 0.122496 | 0.123456 | 0.995599 [0.987686, 1.005600] |
| production-prototype / 1 | 257 / 0 / false | sort_gpu | 0.303456 | 0.304928 | 0.995268 [0.988525, 1.002207] |
| production-prototype / 1 | 257 / 0 / false | resident | 1.457050 | 1.403950 | 1.034287 [1.016904, 1.051381] |
| production-prototype / 1 | 257 / 0 / false | full | 1.655200 | 1.681900 | 0.956447 [0.908683, 0.995397] |
| production-prototype / 1 | 65537 / 0 / false | prefix | 0.153152 | 0.154112 | 0.995849 [0.987733, 1.003654] |
| production-prototype / 1 | 65537 / 0 / false | sort_gpu | 0.934384 | 0.935408 | 0.999648 [0.996542, 1.002701] |
| production-prototype / 1 | 65537 / 0 / false | resident | 2.674650 | 2.693950 | 0.953996 [0.897727, 0.999559] |
| production-prototype / 1 | 65537 / 0 / false | full | 3.235100 | 3.286750 | 1.034039 [0.979908, 1.122389] |
| production-prototype / 1 | 524288 / 0 / false | prefix | 0.667616 | 0.666752 | 1.001824 [1.000162, 1.003513] |
| production-prototype / 1 | 524288 / 0 / false | sort_gpu | 4.571792 | 4.567312 | 1.000719 [0.999930, 1.001556] |
| production-prototype / 1 | 524288 / 0 / false | resident | 7.266200 | 7.247450 | 1.003123 [0.986982, 1.018355] |
| production-prototype / 1 | 524288 / 0 / false | full | 11.122300 | 11.270950 | 0.980094 [0.962782, 0.994673] |
| production-prototype / 1 | 524289 / 0 / true | prefix | 0.222000 | 0.218592 | 1.024781 [1.016662, 1.034063] |
| production-prototype / 1 | 524289 / 0 / true | sort_gpu | 4.147872 | 4.142032 | 1.001575 [1.000213, 1.003001] |
| production-prototype / 1 | 524289 / 0 / true | resident | 6.760850 | 6.669300 | 1.007057 [0.994996, 1.019493] |
| production-prototype / 1 | 524289 / 0 / true | full | 10.971650 | 10.865200 | 1.000871 [0.985640, 1.015716] |
| production-prototype / 1 | 524289 / 1 / true | prefix | 0.221712 | 0.218528 | 1.015412 [1.010406, 1.020065] |
| production-prototype / 1 | 524289 / 1 / true | sort_gpu | 3.169056 | 3.164992 | 1.002048 [1.000613, 1.003588] |
| production-prototype / 1 | 524289 / 1 / true | resident | 5.748600 | 5.762450 | 1.000388 [0.983691, 1.017866] |
| production-prototype / 1 | 524289 / 1 / true | full | 9.884750 | 9.860650 | 0.990659 [0.971756, 1.009037] |
| production-prototype / 1 | 524289 / 2 / true | prefix | 0.222304 | 0.218928 | 1.014990 [1.008193, 1.021800] |
| production-prototype / 1 | 524289 / 2 / true | sort_gpu | 3.368048 | 3.364592 | 1.000885 [0.999334, 1.002383] |
| production-prototype / 1 | 524289 / 2 / true | resident | 5.911650 | 5.944200 | 1.005176 [0.986140, 1.024200] |
| production-prototype / 1 | 524289 / 2 / true | full | 10.085550 | 10.138650 | 1.003212 [0.972912, 1.042949] |
| production-prototype / 1 | 1000003 / 0 / true | prefix | 0.422080 | 0.414224 | 1.019807 [1.016104, 1.023966] |
| production-prototype / 1 | 1000003 / 0 / true | sort_gpu | 7.741344 | 7.728976 | 1.000948 [1.000340, 1.001596] |
| production-prototype / 1 | 1000003 / 0 / true | resident | 10.975800 | 10.992700 | 0.995692 [0.983891, 1.006901] |
| production-prototype / 1 | 1000003 / 0 / true | full | 18.181950 | 18.245550 | 0.989828 [0.976559, 1.002439] |
| production-prototype / 1 | 1000003 / 1 / true | prefix | 0.420400 | 0.413632 | 1.017153 [1.014592, 1.020550] |
| production-prototype / 1 | 1000003 / 1 / true | sort_gpu | 5.890544 | 5.889232 | 1.000496 [0.999739, 1.001249] |
| production-prototype / 1 | 1000003 / 1 / true | resident | 9.235300 | 9.196600 | 1.010735 [0.996222, 1.025215] |
| production-prototype / 1 | 1000003 / 1 / true | full | 16.332550 | 16.454650 | 1.006348 [0.985136, 1.038194] |
| production-prototype / 1 | 1000003 / 2 / true | prefix | 0.418224 | 0.411648 | 1.016597 [1.013957, 1.019449] |
| production-prototype / 1 | 1000003 / 2 / true | sort_gpu | 6.332208 | 6.332240 | 0.999942 [0.999179, 1.000704] |
| production-prototype / 1 | 1000003 / 2 / true | resident | 9.639150 | 9.427150 | 1.008795 [0.995275, 1.022547] |
| production-prototype / 1 | 1000003 / 2 / true | full | 16.788250 | 16.827200 | 0.995837 [0.982643, 1.008110] |
| production-prototype / 1 | 2000003 / 0 / true | prefix | 0.768672 | 0.769984 | 0.998680 [0.996646, 1.000666] |
| production-prototype / 1 | 2000003 / 0 / true | sort_gpu | 15.345392 | 15.344592 | 0.999807 [0.997998, 1.001637] |
| production-prototype / 1 | 2000003 / 0 / true | resident | 19.274150 | 19.365650 | 0.995259 [0.987791, 1.002792] |
| production-prototype / 1 | 2000003 / 0 / true | full | 34.076700 | 34.350400 | 0.994957 [0.987214, 1.003487] |
| production-prototype / 1 | 2000003 / 1 / true | prefix | 0.759280 | 0.761232 | 0.997474 [0.996390, 0.998562] |
| production-prototype / 1 | 2000003 / 1 / true | sort_gpu | 11.540880 | 11.545568 | 0.999520 [0.999214, 0.999811] |
| production-prototype / 1 | 2000003 / 1 / true | resident | 16.172550 | 16.154300 | 1.005923 [0.998468, 1.013119] |
| production-prototype / 1 | 2000003 / 1 / true | full | 30.844000 | 30.841950 | 0.996829 [0.988260, 1.005782] |
| production-prototype / 1 | 2000003 / 2 / true | prefix | 0.759328 | 0.761248 | 0.997656 [0.996770, 0.998538] |
| production-prototype / 1 | 2000003 / 2 / true | sort_gpu | 12.418096 | 12.426784 | 0.999324 [0.999081, 0.999572] |
| production-prototype / 1 | 2000003 / 2 / true | resident | 16.801150 | 16.855200 | 0.998116 [0.991759, 1.004443] |
| production-prototype / 1 | 2000003 / 2 / true | full | 31.646300 | 31.648300 | 1.003886 [0.994161, 1.012756] |
| production-prototype / 2 | 257 / 0 / false | prefix | 0.120400 | 0.121072 | 0.996225 [0.990522, 1.003471] |
| production-prototype / 2 | 257 / 0 / false | sort_gpu | 0.294592 | 0.296224 | 0.996907 [0.992792, 1.001374] |
| production-prototype / 2 | 257 / 0 / false | resident | 1.558550 | 1.513000 | 1.000694 [0.977946, 1.022080] |
| production-prototype / 2 | 257 / 0 / false | full | 2.104100 | 2.121800 | 1.005555 [0.984945, 1.028099] |
| production-prototype / 2 | 65537 / 0 / false | prefix | 0.150464 | 0.151328 | 0.994245 [0.987720, 1.000590] |
| production-prototype / 2 | 65537 / 0 / false | sort_gpu | 0.925008 | 0.926864 | 0.999541 [0.995510, 1.003707] |
| production-prototype / 2 | 65537 / 0 / false | resident | 2.250100 | 2.219300 | 1.021244 [0.987562, 1.052510] |
| production-prototype / 2 | 65537 / 0 / false | full | 3.007000 | 3.078250 | 0.978810 [0.958590, 0.995803] |
| production-prototype / 2 | 524288 / 0 / false | prefix | 0.671488 | 0.669904 | 1.001834 [0.999965, 1.003450] |
| production-prototype / 2 | 524288 / 0 / false | sort_gpu | 4.600592 | 4.599664 | 1.000835 [0.999841, 1.001815] |
| production-prototype / 2 | 524288 / 0 / false | resident | 6.731200 | 6.765350 | 1.005787 [0.989091, 1.022485] |
| production-prototype / 2 | 524288 / 0 / false | full | 11.043000 | 10.958750 | 1.005943 [0.989644, 1.022074] |
| production-prototype / 2 | 524289 / 0 / true | prefix | 0.221280 | 0.217728 | 1.015631 [1.012320, 1.018966] |
| production-prototype / 2 | 524289 / 0 / true | sort_gpu | 4.167776 | 4.164576 | 1.000853 [0.999850, 1.001850] |
| production-prototype / 2 | 524289 / 0 / true | resident | 6.454050 | 6.498950 | 1.012443 [0.988134, 1.037318] |
| production-prototype / 2 | 524289 / 0 / true | full | 11.162450 | 11.256800 | 0.987881 [0.965488, 1.011712] |
| production-prototype / 2 | 524289 / 1 / true | prefix | 0.221200 | 0.218096 | 1.014490 [1.010345, 1.018511] |
| production-prototype / 2 | 524289 / 1 / true | sort_gpu | 3.168592 | 3.163472 | 1.001092 [0.999727, 1.002441] |
| production-prototype / 2 | 524289 / 1 / true | resident | 5.358950 | 5.371000 | 1.017562 [0.986637, 1.049666] |
| production-prototype / 2 | 524289 / 1 / true | full | 10.038700 | 10.212800 | 0.987290 [0.964674, 1.008981] |
| production-prototype / 2 | 524289 / 2 / true | prefix | 0.221392 | 0.217888 | 1.018075 [1.012804, 1.023647] |
| production-prototype / 2 | 524289 / 2 / true | sort_gpu | 3.369888 | 3.368448 | 1.000943 [0.999488, 1.002360] |
| production-prototype / 2 | 524289 / 2 / true | resident | 5.512450 | 5.454250 | 1.012469 [0.994097, 1.031621] |
| production-prototype / 2 | 524289 / 2 / true | full | 9.894200 | 9.843400 | 1.004906 [0.989281, 1.020687] |
| production-prototype / 2 | 1000003 / 0 / true | prefix | 0.421408 | 0.414896 | 1.016492 [1.013614, 1.019811] |
| production-prototype / 2 | 1000003 / 0 / true | sort_gpu | 7.782608 | 7.777424 | 1.000764 [1.000191, 1.001320] |
| production-prototype / 2 | 1000003 / 0 / true | resident | 10.675750 | 10.492150 | 1.012213 [1.000241, 1.024881] |
| production-prototype / 2 | 1000003 / 0 / true | full | 18.497200 | 18.363350 | 1.005300 [0.992084, 1.018902] |
| production-prototype / 2 | 1000003 / 1 / true | prefix | 0.421072 | 0.414656 | 1.015903 [1.013301, 1.018496] |
| production-prototype / 2 | 1000003 / 1 / true | sort_gpu | 5.899472 | 5.896208 | 1.000413 [0.999628, 1.001162] |
| production-prototype / 2 | 1000003 / 1 / true | resident | 8.563100 | 8.637550 | 0.989821 [0.970499, 1.008327] |
| production-prototype / 2 | 1000003 / 1 / true | full | 16.269900 | 16.292350 | 0.991791 [0.975390, 1.008285] |
| production-prototype / 2 | 1000003 / 2 / true | prefix | 0.421024 | 0.414624 | 1.016409 [1.014664, 1.018238] |
| production-prototype / 2 | 1000003 / 2 / true | sort_gpu | 6.351104 | 6.353344 | 0.999497 [0.998849, 1.000159] |
| production-prototype / 2 | 1000003 / 2 / true | resident | 9.406900 | 9.312250 | 1.009328 [0.994069, 1.024518] |
| production-prototype / 2 | 1000003 / 2 / true | full | 16.836650 | 17.018150 | 0.989140 [0.976933, 1.001844] |
| production-prototype / 2 | 2000003 / 0 / true | prefix | 0.773792 | 0.774512 | 0.998675 [0.996300, 1.000964] |
| production-prototype / 2 | 2000003 / 0 / true | sort_gpu | 15.453328 | 15.438384 | 0.999782 [0.997931, 1.001574] |
| production-prototype / 2 | 2000003 / 0 / true | resident | 19.794550 | 19.734600 | 1.003764 [0.995495, 1.012268] |
| production-prototype / 2 | 2000003 / 0 / true | full | 35.239050 | 34.503900 | 1.014043 [0.999140, 1.029088] |
| production-prototype / 2 | 2000003 / 1 / true | prefix | 0.764736 | 0.765808 | 0.998212 [0.997078, 0.999344] |
| production-prototype / 2 | 2000003 / 1 / true | sort_gpu | 11.566608 | 11.570112 | 0.999662 [0.999407, 0.999921] |
| production-prototype / 2 | 2000003 / 1 / true | resident | 16.140300 | 16.180850 | 1.005211 [0.993903, 1.016320] |
| production-prototype / 2 | 2000003 / 1 / true | full | 31.489800 | 31.052800 | 1.008719 [0.990637, 1.027342] |
| production-prototype / 2 | 2000003 / 2 / true | prefix | 0.764912 | 0.766544 | 0.997850 [0.996669, 0.998998] |
| production-prototype / 2 | 2000003 / 2 / true | sort_gpu | 12.461488 | 12.469248 | 0.999305 [0.998996, 0.999598] |
| production-prototype / 2 | 2000003 / 2 / true | resident | 17.103050 | 17.208600 | 0.993785 [0.984009, 1.003751] |
| production-prototype / 2 | 2000003 / 2 / true | full | 33.311150 | 33.542950 | 0.999425 [0.983979, 1.015185] |
| native-only-candidate / 3 | 257 / 0 / false | prefix | 0.120320 | 0.120960 | 0.988601 [0.976305, 0.999892] |
| native-only-candidate / 3 | 257 / 0 / false | sort_gpu | 0.297296 | 0.298928 | 0.990095 [0.978196, 1.002073] |
| native-only-candidate / 3 | 257 / 0 / false | resident | 1.433900 | 1.381500 | 1.059701 [1.029385, 1.099496] |
| native-only-candidate / 3 | 257 / 0 / false | full | 1.777900 | 1.877150 | 0.924967 [0.889331, 0.959411] |
| native-only-candidate / 3 | 65537 / 0 / false | prefix | 0.149168 | 0.148384 | 0.997552 [0.991280, 1.003905] |
| native-only-candidate / 3 | 65537 / 0 / false | sort_gpu | 0.910256 | 0.915888 | 0.993368 [0.988787, 0.998016] |
| native-only-candidate / 3 | 65537 / 0 / false | resident | 2.146150 | 2.108300 | 1.004481 [0.980929, 1.023663] |
| native-only-candidate / 3 | 65537 / 0 / false | full | 2.861400 | 2.913850 | 0.983193 [0.960744, 1.005604] |
| native-only-candidate / 3 | 524288 / 0 / false | prefix | 0.665872 | 0.665808 | 0.988836 [0.966045, 1.001372] |
| native-only-candidate / 3 | 524288 / 0 / false | sort_gpu | 4.579104 | 4.578352 | 0.998249 [0.994028, 1.001003] |
| native-only-candidate / 3 | 524288 / 0 / false | resident | 6.581300 | 6.614900 | 0.997571 [0.981470, 1.013737] |
| native-only-candidate / 3 | 524288 / 0 / false | full | 10.467700 | 10.486800 | 1.001381 [0.988881, 1.014260] |
| native-only-candidate / 3 | 524289 / 0 / true | prefix | 0.219824 | 0.216752 | 1.023609 [1.011890, 1.043037] |
| native-only-candidate / 3 | 524289 / 0 / true | sort_gpu | 4.141296 | 4.136144 | 1.007500 [0.999352, 1.016233] |
| native-only-candidate / 3 | 524289 / 0 / true | resident | 6.602550 | 6.560300 | 0.997646 [0.981844, 1.013477] |
| native-only-candidate / 3 | 524289 / 0 / true | full | 10.627000 | 10.762050 | 0.994112 [0.984470, 1.003783] |
| native-only-candidate / 3 | 524289 / 1 / true | prefix | 0.220016 | 0.216496 | 1.016058 [1.012502, 1.019630] |
| native-only-candidate / 3 | 524289 / 1 / true | sort_gpu | 3.166880 | 3.161984 | 1.003034 [0.999569, 1.008914] |
| native-only-candidate / 3 | 524289 / 1 / true | resident | 5.648300 | 5.591200 | 1.014068 [0.996503, 1.031956] |
| native-only-candidate / 3 | 524289 / 1 / true | full | 9.806800 | 9.840300 | 0.996797 [0.983853, 1.009826] |
| native-only-candidate / 3 | 524289 / 2 / true | prefix | 0.219792 | 0.216576 | 1.015353 [1.012645, 1.017994] |
| native-only-candidate / 3 | 524289 / 2 / true | sort_gpu | 3.365456 | 3.369280 | 0.997799 [0.989395, 1.005917] |
| native-only-candidate / 3 | 524289 / 2 / true | resident | 5.842100 | 5.859000 | 0.978922 [0.959924, 0.997257] |
| native-only-candidate / 3 | 524289 / 2 / true | full | 10.139200 | 10.118350 | 0.998392 [0.987313, 1.009753] |
| native-only-candidate / 3 | 1000003 / 0 / true | prefix | 0.419072 | 0.411120 | 1.018639 [1.016021, 1.021179] |
| native-only-candidate / 3 | 1000003 / 0 / true | sort_gpu | 7.749472 | 7.757952 | 0.996120 [0.990182, 1.002104] |
| native-only-candidate / 3 | 1000003 / 0 / true | resident | 10.842100 | 10.825800 | 1.003661 [0.993694, 1.013723] |
| native-only-candidate / 3 | 1000003 / 0 / true | full | 18.472150 | 18.488600 | 0.995291 [0.986201, 1.004949] |
| native-only-candidate / 3 | 1000003 / 1 / true | prefix | 0.418560 | 0.410704 | 1.018535 [1.015977, 1.020951] |
| native-only-candidate / 3 | 1000003 / 1 / true | sort_gpu | 5.887056 | 5.885552 | 1.000902 [1.000150, 1.001657] |
| native-only-candidate / 3 | 1000003 / 1 / true | resident | 9.089300 | 9.037750 | 1.003105 [0.987079, 1.019590] |
| native-only-candidate / 3 | 1000003 / 1 / true | full | 16.636250 | 16.689850 | 0.994926 [0.986406, 1.003478] |
| native-only-candidate / 3 | 1000003 / 2 / true | prefix | 0.418800 | 0.410976 | 1.019313 [1.016632, 1.021927] |
| native-only-candidate / 3 | 1000003 / 2 / true | sort_gpu | 6.330688 | 6.332896 | 0.995178 [0.983870, 1.006350] |
| native-only-candidate / 3 | 1000003 / 2 / true | resident | 9.614200 | 9.547400 | 1.013523 [0.998866, 1.028819] |
| native-only-candidate / 3 | 1000003 / 2 / true | full | 17.297950 | 17.320950 | 1.003871 [0.993906, 1.012813] |
| native-only-candidate / 3 | 2000003 / 0 / true | prefix | 0.776544 | 0.765776 | 1.031183 [1.013712, 1.064739] |
| native-only-candidate / 3 | 2000003 / 0 / true | sort_gpu | 15.432000 | 15.401744 | 1.004980 [0.998371, 1.011598] |
| native-only-candidate / 3 | 2000003 / 0 / true | resident | 19.805400 | 19.767750 | 1.003612 [0.997161, 1.009842] |
| native-only-candidate / 3 | 2000003 / 0 / true | full | 34.895200 | 34.929450 | 0.995924 [0.988010, 1.003445] |
| native-only-candidate / 3 | 2000003 / 1 / true | prefix | 0.764976 | 0.754496 | 1.013907 [1.012891, 1.014931] |
| native-only-candidate / 3 | 2000003 / 1 / true | sort_gpu | 11.550336 | 11.543952 | 0.998099 [0.991743, 1.004341] |
| native-only-candidate / 3 | 2000003 / 1 / true | resident | 16.231500 | 16.208300 | 1.000396 [0.993908, 1.006893] |
| native-only-candidate / 3 | 2000003 / 1 / true | full | 31.674250 | 31.458050 | 1.004903 [0.995475, 1.014504] |
| native-only-candidate / 3 | 2000003 / 2 / true | prefix | 0.765120 | 0.754448 | 1.030125 [1.013476, 1.062955] |
| native-only-candidate / 3 | 2000003 / 2 / true | sort_gpu | 12.425760 | 12.421472 | 1.001654 [0.997899, 1.005496] |
| native-only-candidate / 3 | 2000003 / 2 / true | resident | 17.109600 | 17.122400 | 1.000246 [0.993449, 1.007385] |
| native-only-candidate / 3 | 2000003 / 2 / true | full | 32.343950 | 32.428700 | 0.998386 [0.992285, 1.004612] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Native fixtures: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_four_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_four_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_four_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_four_candidate.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_four_benchmark.rs`.
Unchanged production: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/voxel_radix.wgsl`.
Ignored validation/raw/provenance/archive: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/radix-four-*`.
