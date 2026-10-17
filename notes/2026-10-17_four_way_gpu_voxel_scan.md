# Four-way GPU voxel boundary scans

Parent: `df64139fe077a2c3741cfedc3019208a635fc203` (merged PR 257).

The private 256-lane voxel boundary scan now uses a four-way exclusive tree,
then adds the original element to produce its inclusive result. Its host
hierarchy and the internal compactor used by public voxel builders remain
unchanged. Only one production WGSL file changes; separate native fixtures retain the complete parent.

## Exact order and synchronization

Upsweep strides 1/4/16/64 combine four consecutive subtree totals. The existing
root handoff writes the same block total and resets its root to zero. Downsweep
strides 64/16/4/1 give each child its parent's exclusive prefix plus preceding
sibling totals. At the final level this is the exclusive prefix for each lane;
adding the original value produces the same inclusive sum.

Each active lane accesses four distinct tree slots in a disjoint subtree at
that stage. Downsweep reads its inputs into locals before writing the outputs.
Every stage retains a workgroup barrier before any lane can consume the next level.
Inactive tail lanes load zero and still participate in all barriers. No subgroup,
warp-width assumption or early return is added.

| Tree source work per block | Binary parent | Four-way |
| --- | ---: | ---: |
| Upsweep stages / barriers | 8 | 4 |
| Downsweep stages / barriers | 8 | 4 |
| Total barriers including input/root handoffs | 18 | 10 |
| Tree additions, excluding final inclusive output | 510 | 510 |

These are source-level counts, not generated instruction or shared-memory
transaction measurements. The tree still uses 256 u32 words. Its real boundary
inputs are nonnegative flags and hierarchy totals, with total at most the u32
point count. All subtree and prefix sums are bounded by that same mass, so
regrouping introduces neither overflow nor rounding changes, even at max-u32
mass. CPU oracles use direct u64 cumulative sums and checked narrowing.

The original upper-level scan, block-offset shader, pass planning, lazy caches,
scratch sizes/labels, dispatch, compactor mark/write kernels, scalar 4-byte cell
count synchronization, public errors, metadata decoding, index ordering,
ownership and explicit transfers remain. No public API, dependency, core,
allocation, pass, transfer or unsafe boundary is introduced.

## Correctness and integration

- **369 native correctness tests:** 145 image, 187 voxel/readback, 8 RANSAC,
  8 segmentation, 18 filtering, 2 transfer facade and 1 Gaussian facade. Include
  ignored GPU correctness tests and exclude benchmarks. All-feature/all-target
  Clippy with warnings denied, CPU-only, minimal/CI WASM with warnings denied,
  formatting and whitespace pass.
- **12,058,630 exact parent/u64 block-prefix and block-total words** in
  85 paired raw fixtures. Seventeen sizes cover 1/3/4, 63/64/65, 255/256/257,
  tails, 65,535/65,536/65,537 and 1,000,003 elements. Five distributions include
  zero, boundary flags, balanced max-u32 mass, single-hot max-u32 mass and uneven
  weighted max-u32 mass. Poisoned input tails and untouched output/total suffixes
  detect inactive-lane reads and out-of-range writes.
- **12,011,620 exact complete-parent/u64 prefix words** in 90 paired
  hierarchy fixtures, including zero no-op, three hierarchy levels, partial
  blocks, maximum mass and dirty reused scratch/output. Every active prefix and
  every guarded output suffix is checked.
- **21,890,424 parent/CPU logical segment values** in 20 paired complete
  internal compactor fixtures (40 executions), covering zero, single, odd, 65,537 and
  1,000,003 elements with unique/short/cross-block/single-cell runs. Comparisons
  cover key coordinates, cell starts/counts and point indices. Separate raw
  index checks preserve device order; the original host within-cell sort,
  0/4-byte construction receipts and retained outputs remain. Existing sort,
  filtering, pool reuse, validation, profiling and concurrent callers also pass.
- **516 original GPU/image sources unchanged**, with **525 source/executable
  hashes**. The original host scan file differs only by an appended native module.
  The parent shader is frozen byte-for-byte; complete parent scan pipelines and
  hierarchy are frozen with type/name/visibility/external-cache substitutions.
  The full immediate-parent compactor call and its private compactor helpers are frozen,
  changing only its name/visibility and the external scan cache handoff. All
  production compactor, radix, registration and image files compare unchanged.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 routes=scan_gpu_resident_compactor_and_full checks_drop_excluded=true`. Adapter model names are omitted.

Three sequential surveys each use three warm and thirty measured pairs across
eight cases, with zero-based round order rotated to balance old/new variants.
All **72** rows are retained below. The first two surveys produced one adverse
resident-compactor interval at 1,000,003 elements / one-element runs. This
unresolved endpoint result prompted one additional complete eight-case survey,
using unchanged sources/executable and timing boundaries. Both original surveys
are preserved separately; no original condition or result was removed. The
third survey is a diagnostic repeat, not an independent workload expansion.

- GPU scan-pass sums: **24 favorable / 0 inconclusive / 0 adverse**.
- Complete resident compactor: **1 favorable / 22 inconclusive / 1 adverse**.
- Upload/compact/readback chain: **4 favorable / 19 inconclusive / 1 adverse**.

Adverse pointwise intervals are retained: run 2, 1000003 elements / run 1, resident, ratio 0.891018 [0.798779, 0.976702]; run 3, 1000003 elements / run 1, full, ratio 0.962353 [0.929294, 0.992838].

The adverse second-survey resident interval did not recur in the third survey
at that same condition: its paired ratio was 1.047790 [0.977207, 1.127973].
The third-survey full interval was adverse at that condition, while the two
initial full intervals were favorable. These mixed endpoint observations remain
unresolved; no general compactor latency gain is claimed. No further repeat
was performed.

At 1,000,003 elements / 1024-point runs, scan-pass sums are
0.689424 -> 0.576896 ms and 0.733056 -> 0.587104 ms and 0.664304 -> 0.578640 ms; resident compactors are
7.768550 -> 7.627550 ms and 8.719150 -> 8.379300 ms and 7.621850 -> 7.508700 ms; full chains are
22.957800 -> 22.391500 ms and 25.509850 -> 26.393950 ms and 22.944450 -> 21.790850 ms. These observations do not establish a general
transfer-inclusive or application speedup. Small blocks and run distributions
are retained individually rather than inferred from this example.

- `pass`: sum of timestamp durations for the entire private scan hierarchy,
  including its unchanged block-offset passes. Boundary flags are pre-uploaded
  outside timing. Parent/new hierarchies share one command buffer in alternating
  order, with separate outputs and live scratch leases until submission. There
  are 1/3/5 passes per hierarchy, so explicit query collection reads 32/96/160
  bytes after the work. Final outputs match the u64 oracle. This sum excludes
  inter-pass gaps, compactor mark/write, host planning, upload, waits and readback.
  Flags use `(i*37 % 113) < 47` at every count. Run length labels describe the
  compactor fixtures; same-count pass measurements repeat this same synthetic
  flag distribution rather than representing independent run distributions.
- `resident`: the complete compactor call, starting from resident sorted entries.
  Includes original buffer/control/encoder construction, mark/scan/write, scalar
  readback synchronization and result construction. Final metadata readback,
  assertions and output destruction follow timing.
  This compactor is an internal callable body used by the public voxel builders;
  the parent reference does not replace or simulate a new public resident API.
- `full`: upload of sorted entries, complete compactor and original metadata/
  indices readback/host within-cell sort. Assertions and returned-value/resource
  destruction follow timing. Radix sorting and key generation are outside this
  boundary; this is not a full point-cloud filter or application benchmark.

Four resident/full variants rotate by round and stay paired by round. Both
parent/current scan caches are warmed before measured rounds. The frozen parent
uses an explicit external scan cache; the live path retains its original runtime
cache. CPU expectations, pipeline compilation and verification are outside timing.

Ratios are geometric means of paired parent/new samples; above one favors the
new route. Pointwise percentile intervals resample paired log ratios 10,000
times, seed `2026101708 + 100*run + 10*case + route_index`, route order
pass/resident/full, with sorted bounds 249/9749. No multiple-comparison correction
is applied. Zero times would suppress ratios; all raw samples are retained.

Run 1: observed-quiet **false**; busy-process counts 2 before / 1 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 2: observed-quiet **false**; busy-process counts 2 before / 1 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 3: observed-quiet **false**; busy-process counts 2 before / 4 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all GPU/memory
samples at most 5%. Surveys wait two seconds between snapshots; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post samples cannot
prove isolation during every pair. These are shared-host diagnostic observations,
with no portable throughput or application-latency guarantee. Process identities
remain in ignored receipts.

| Run | Elements / run length / scan passes | Boundary | Parent median ms | New median ms | Paired parent/new ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 1 / 1 / 1 | pass | 0.005056 | 0.004608 | 1.113074 [1.060598, 1.166352] |
| 1 | 1 / 1 / 1 | resident | 1.163550 | 1.201100 | 0.983168 [0.948093, 1.022339] |
| 1 | 1 / 1 / 1 | full | 4.225250 | 4.203900 | 0.993875 [0.976022, 1.010947] |
| 1 | 256 / 4 / 1 | pass | 0.004992 | 0.004480 | 1.109915 [1.053266, 1.171037] |
| 1 | 256 / 4 / 1 | resident | 1.217050 | 1.217450 | 0.999107 [0.970136, 1.030656] |
| 1 | 256 / 4 / 1 | full | 4.383700 | 4.317450 | 0.987845 [0.944634, 1.036398] |
| 1 | 257 / 4 / 3 | pass | 0.012512 | 0.011568 | 1.076293 [1.051768, 1.101418] |
| 1 | 257 / 4 / 3 | resident | 1.258350 | 1.255100 | 0.996409 [0.959243, 1.034355] |
| 1 | 257 / 4 / 3 | full | 4.454800 | 4.449000 | 0.985555 [0.949764, 1.021209] |
| 1 | 65536 / 257 / 3 | pass | 0.079680 | 0.058096 | 1.381487 [1.341440, 1.423350] |
| 1 | 65536 / 257 / 3 | resident | 2.211700 | 2.229450 | 1.005513 [0.960424, 1.057510] |
| 1 | 65536 / 257 / 3 | full | 6.921400 | 7.148550 | 0.999560 [0.965019, 1.037365] |
| 1 | 65537 / 257 / 5 | pass | 0.115344 | 0.086800 | 1.324833 [1.305403, 1.344782] |
| 1 | 65537 / 257 / 5 | resident | 2.698600 | 2.611700 | 0.996952 [0.972382, 1.022413] |
| 1 | 65537 / 257 / 5 | full | 7.300200 | 7.181400 | 0.999909 [0.974353, 1.027141] |
| 1 | 1000003 / 1 / 5 | pass | 0.724368 | 0.586976 | 1.236526 [1.232279, 1.240868] |
| 1 | 1000003 / 1 / 5 | resident | 9.296650 | 9.456700 | 0.936049 [0.857886, 1.008002] |
| 1 | 1000003 / 1 / 5 | full | 39.687600 | 39.261200 | 1.051274 [1.011473, 1.100364] |
| 1 | 1000003 / 1024 / 5 | pass | 0.689424 | 0.576896 | 1.196060 [1.193355, 1.198715] |
| 1 | 1000003 / 1024 / 5 | resident | 7.768550 | 7.627550 | 0.995290 [0.971866, 1.017915] |
| 1 | 1000003 / 1024 / 5 | full | 22.957800 | 22.391500 | 1.014546 [0.992823, 1.037575] |
| 1 | 1000003 / 1000003 / 5 | pass | 0.665440 | 0.575584 | 1.158373 [1.154213, 1.162524] |
| 1 | 1000003 / 1000003 / 5 | resident | 8.028600 | 8.089300 | 1.009538 [0.987262, 1.031231] |
| 1 | 1000003 / 1000003 / 5 | full | 23.678100 | 22.954250 | 1.022769 [0.976162, 1.067410] |
| 2 | 1 / 1 / 1 | pass | 0.005024 | 0.004448 | 1.112630 [1.058074, 1.168306] |
| 2 | 1 / 1 / 1 | resident | 1.227150 | 1.251750 | 0.994714 [0.956771, 1.036260] |
| 2 | 1 / 1 / 1 | full | 4.572800 | 4.459000 | 0.995074 [0.967574, 1.025111] |
| 2 | 256 / 4 / 1 | pass | 0.004960 | 0.004560 | 1.105221 [1.050242, 1.161918] |
| 2 | 256 / 4 / 1 | resident | 1.155300 | 1.195850 | 0.968955 [0.934337, 1.005647] |
| 2 | 256 / 4 / 1 | full | 4.172000 | 4.243150 | 0.987867 [0.955488, 1.019243] |
| 2 | 257 / 4 / 3 | pass | 0.012576 | 0.011536 | 1.078677 [1.054684, 1.102569] |
| 2 | 257 / 4 / 3 | resident | 1.189650 | 1.192350 | 0.991589 [0.940543, 1.032147] |
| 2 | 257 / 4 / 3 | full | 4.168150 | 4.149250 | 1.007907 [0.986641, 1.030840] |
| 2 | 65536 / 257 / 3 | pass | 0.025120 | 0.018640 | 1.347102 [1.327568, 1.366937] |
| 2 | 65536 / 257 / 3 | resident | 1.334500 | 1.372600 | 0.973945 [0.931870, 1.017208] |
| 2 | 65536 / 257 / 3 | full | 4.957450 | 4.897800 | 1.014105 [0.982159, 1.045784] |
| 2 | 65537 / 257 / 5 | pass | 0.089504 | 0.069632 | 1.282260 [1.263237, 1.301026] |
| 2 | 65537 / 257 / 5 | resident | 2.408700 | 2.397300 | 1.021777 [0.994399, 1.053206] |
| 2 | 65537 / 257 / 5 | full | 6.599500 | 6.649850 | 1.001938 [0.979117, 1.023743] |
| 2 | 1000003 / 1 / 5 | pass | 1.592256 | 1.523392 | 1.045270 [1.032457, 1.058307] |
| 2 | 1000003 / 1 / 5 | resident | 12.084550 | 16.746550 | 0.891018 [0.798779, 0.976702] |
| 2 | 1000003 / 1 / 5 | full | 56.401200 | 55.236550 | 1.059441 [1.013479, 1.110903] |
| 2 | 1000003 / 1024 / 5 | pass | 0.733056 | 0.587104 | 1.248550 [1.244486, 1.252459] |
| 2 | 1000003 / 1024 / 5 | resident | 8.719150 | 8.379300 | 1.016027 [0.952365, 1.072504] |
| 2 | 1000003 / 1024 / 5 | full | 25.509850 | 26.393950 | 0.982239 [0.944209, 1.019591] |
| 2 | 1000003 / 1000003 / 5 | pass | 0.724480 | 0.585904 | 1.218728 [1.207139, 1.229528] |
| 2 | 1000003 / 1000003 / 5 | resident | 7.969750 | 7.872900 | 1.001754 [0.964832, 1.035541] |
| 2 | 1000003 / 1000003 / 5 | full | 22.313050 | 22.195300 | 0.963130 [0.891152, 1.031769] |
| 3 | 1 / 1 / 1 | pass | 0.004960 | 0.004560 | 1.103146 [1.051452, 1.157007] |
| 3 | 1 / 1 / 1 | resident | 1.402700 | 1.439950 | 0.986792 [0.926083, 1.051495] |
| 3 | 1 / 1 / 1 | full | 5.157200 | 5.097450 | 1.018420 [0.973577, 1.069033] |
| 3 | 256 / 4 / 1 | pass | 0.005120 | 0.004480 | 1.112920 [1.058394, 1.168675] |
| 3 | 256 / 4 / 1 | resident | 1.181050 | 1.180000 | 0.988212 [0.931348, 1.047427] |
| 3 | 256 / 4 / 1 | full | 4.356850 | 4.186500 | 1.035195 [1.000532, 1.068754] |
| 3 | 257 / 4 / 3 | pass | 0.038272 | 0.034288 | 1.082293 [1.027258, 1.137547] |
| 3 | 257 / 4 / 3 | resident | 1.415550 | 1.512650 | 0.973655 [0.906202, 1.043958] |
| 3 | 257 / 4 / 3 | full | 5.297550 | 4.919250 | 1.044273 [0.992994, 1.094471] |
| 3 | 65536 / 257 / 3 | pass | 0.025200 | 0.018736 | 1.345626 [1.323808, 1.367660] |
| 3 | 65536 / 257 / 3 | resident | 2.227400 | 2.223450 | 0.983008 [0.897196, 1.054363] |
| 3 | 65536 / 257 / 3 | full | 6.969500 | 6.953100 | 1.001982 [0.974544, 1.030387] |
| 3 | 65537 / 257 / 5 | pass | 0.032464 | 0.025648 | 1.266745 [1.254558, 1.279307] |
| 3 | 65537 / 257 / 5 | resident | 1.631150 | 1.556650 | 0.980463 [0.903600, 1.048207] |
| 3 | 65537 / 257 / 5 | full | 4.933000 | 4.861400 | 1.004707 [0.978663, 1.031312] |
| 3 | 1000003 / 1 / 5 | pass | 0.715152 | 0.581344 | 1.231798 [1.226548, 1.236868] |
| 3 | 1000003 / 1 / 5 | resident | 9.193800 | 8.730950 | 1.047790 [0.977207, 1.127973] |
| 3 | 1000003 / 1 / 5 | full | 39.826350 | 40.550400 | 0.962353 [0.929294, 0.992838] |
| 3 | 1000003 / 1024 / 5 | pass | 0.664304 | 0.578640 | 1.152791 [1.148843, 1.156804] |
| 3 | 1000003 / 1024 / 5 | resident | 7.621850 | 7.508700 | 1.021158 [1.007695, 1.035448] |
| 3 | 1000003 / 1024 / 5 | full | 22.944450 | 21.790850 | 1.002994 [0.973277, 1.031068] |
| 3 | 1000003 / 1000003 / 5 | pass | 0.657392 | 0.573984 | 1.147924 [1.144166, 1.151883] |
| 3 | 1000003 / 1000003 / 5 | resident | 8.313900 | 8.280350 | 1.010761 [0.961828, 1.052024] |
| 3 | 1000003 / 1000003 / 5 | full | 26.590100 | 25.217950 | 1.061297 [1.027944, 1.096101] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Production: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/inclusive_scan.wgsl`.
Native fixtures: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_scan_four_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_scan_four_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_scan_four_reference.wgsl`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_scan_four_compact_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_scan_four_benchmark.rs`.
Ignored validation/raw measurements/provenance: `C:/Users/rsasa/Workspace/SpatialRust/target/automation/voxel-scan-four-*`.
