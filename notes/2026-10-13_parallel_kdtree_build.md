# Bounded parallel KD-tree construction — 2026-10-13

`KdTree::from_slices_parallel` / `from_point_cloud_parallel` under search
`parallel` (facade `search-parallel`) construct large CPU trees with bounded
scoped subtree workers. GPU GICP's resident reduction/covariance methods use
this host preparation; ordinary tree constructors and ordinary CPU GICP keep
serial placement. GPU algorithms, input staging, equations and receipts are
unchanged. This is CPU parallelism supporting GPU preparation, not device-side
KD-tree construction.

## Structure, exactness and worker boundaries

Canonical axis selection and minimum-pivot partition code are reused. Absolute
preorder node/point ranges are calculated before sibling work. One exact-sized
node array and the point permutation are split into disjoint mutable slices;
there are no per-worker trees, locks, unsafe or node-merging copies. Each
subtree scans/partitions its points in the original order. Local range indices
translate exactly to old global pivot indices, retaining all tie permutations.
Original coordinate arrays are copied once after construction.

For the existing power-of-two leaf size 16, let F be the highest power of two
not exceeding floor(N/16). Levels above F fully split. If floor(N/F)=16,
node count is 2*F-1+2*(N-16*F); otherwise it is 4*F-1. Counts <=16 use one
node, empty inputs zero. This O(1) recurrence gives exact preorder sibling
ranges, including odd boundaries. The planning formula is checked against an
independent recursive-count table for 131,099 sizes and the u32 index boundary.

From **32,768 points**, workers are bounded by available CPU parallelism and
floor(N/16,384). Root selection/partition remains serial, then child budgets
split between a scoped helper and the caller. Every spawning subtree has at
least 32,768 points, so children have at least 16,384. Concurrent helpers are
bounded by the supplied budget; completed calls join them. Below the threshold,
one useful worker or WASM keeps the original serial constructor. Tiny public
calls do not query CPU availability. Bounds apply per call, not process-wide;
multiple callers can still run concurrently. Unequal axes and >u32 original
indices panic, as documented for the explicit constructors.

Resident GICP's internal tree-builder boundary permits a serial control for
verification/measurement. Production selects the explicit parallel constructor
for target and required source trees. The post-preparation pose/evaluation loop
is unchanged. Both preparation paths produce the same tree layout and upload
bytes; no new device copies or downloads are introduced. No production
thread-pool dependency, core model or GPU shader change is added.

## Correctness evidence

- **405** exact full layouts with forced 1/2/6 worker budgets span empty,
  1/15/16/17/31/32/33/257/1025, threshold/odd 32767/32768/32769, 65537 and
  131073 points. Nine morphologies include duplicates/outliers/planes/lines,
  NaN payloads/infinities/signed zeros and distributed finite coordinates.
  Point order, all split/node words and copied coordinate bits match serial.
  Small cases also match the frozen pre-optimization serial builder.
- **30,375** ordered k-NN/radius/gated-nearest results match bit for bit,
  including nonfinite queries, gates and ties. Independent midpoint-tie and
  inclusive-radius expectations, immutable source arrays, malformed axes,
  cloud construction and four concurrent public callers are checked.
- **216** completed native GICP conditions compare the previous serial tree
  preparation and new public path: six sizes, three scales, four covariance
  modes (legacy, resident k-NN, bounded radius, grid) and three stopping modes.
  **1,230,012** compared source points retain exact complete diagnostics:
  poses, fitness, histories, counts, stopping and transfer receipts. The
  largest fixture crosses the parallel threshold. GPU arithmetic is unchanged.

## Completed measurements

Two isolated release surveys per family, three warmups and 20 alternating
paired samples per condition. Construction measures complete APIs, including
node/point allocation, planning, axis/partition work, coordinate copies and
thread creation/joins, excluding output drop and validation snapshots. Kinds
0/3/8 are coincident points, curved patches and distributed finite points.

Registration measures complete resident-grid GICP calls using the same
implementation/configuration except serial versus parallel tree construction.
Runtime initialization/fixture creation is excluded; CPU tree/grid preparation,
uploads, GPU covariance/equation work, waits/readback, CPU 6x6 solves/composition
and final fitness are included. Sizes 4097/32768/65537/262145 use fixed 0.5
spacing on a 32x32 grid; profiles 0/1 have radii 0.51/0.8 and capacities 32/64.
Profile 2 duplicates grid positions, adds 17 targets, radius 1.05 and capacity
128. Binary translation `(0.03125,-0.015625,0.0078125)`, distance gate 0.25,
epsilon 0.001, three updates and zero stopping thresholds are fixed. All warm
and timed outputs match their serial references exactly, including receipts.
No other local builds/tests/statistical work runs during surveys.

Native CPU availability is 12 workers; policy budgets are 1/2/4/12 for the four
sizes. Registration uses Vulkan/DiscreteGpu without recording the adapter
model. These are consecutive local surveys, not cross-device/day evidence.

### construction survey 1

| N | Kind/profile | Serial ms | Parallel ms | Serial/parallel [95%] |
| ---: | ---: | ---: | ---: | --- |
| 4,097 | 0 | 0.219 | 0.223 | 0.99 [0.98–1.00] |
| 4,097 | 3 | 0.521 | 0.527 | 1.00 [0.98–1.01] |
| 4,097 | 8 | 0.567 | 0.569 | 1.00 [0.99–1.01] |
| 32,768 | 0 | 2.241 | 1.565 | 1.43 [1.38–1.47] |
| 32,768 | 3 | 4.033 | 2.620 | 1.47 [1.41–1.54] |
| 32,768 | 8 | 7.091 | 4.350 | 1.60 [1.50–1.69] |
| 65,537 | 0 | 4.862 | 2.184 | 2.15 [2.07–2.22] |
| 65,537 | 3 | 8.692 | 3.459 | 2.55 [2.45–2.64] |
| 65,537 | 8 | 16.529 | 6.060 | 2.48 [2.28–2.68] |
| 262,145 | 0 | 27.686 | 10.520 | 2.72 [2.62–2.82] |
| 262,145 | 3 | 50.530 | 14.552 | 3.32 [3.02–3.64] |
| 262,145 | 8 | 91.752 | 24.697 | 3.37 [3.01–3.74] |

### construction survey 2

| N | Kind/profile | Serial ms | Parallel ms | Serial/parallel [95%] |
| ---: | ---: | ---: | ---: | --- |
| 4,097 | 0 | 0.246 | 0.245 | 1.00 [0.99–1.01] |
| 4,097 | 3 | 0.513 | 0.499 | 1.01 [0.99–1.04] |
| 4,097 | 8 | 0.879 | 0.881 | 0.99 [0.97–1.01] |
| 32,768 | 0 | 2.697 | 1.891 | 1.38 [1.26–1.52] |
| 32,768 | 3 | 5.111 | 3.124 | 1.63 [1.53–1.73] |
| 32,768 | 8 | 8.616 | 5.160 | 1.63 [1.55–1.71] |
| 65,537 | 0 | 6.348 | 3.514 | 1.88 [1.80–1.96] |
| 65,537 | 3 | 9.575 | 4.169 | 2.35 [2.21–2.50] |
| 65,537 | 8 | 18.554 | 7.287 | 2.44 [2.30–2.60] |
| 262,145 | 0 | 26.271 | 9.427 | 2.75 [2.64–2.87] |
| 262,145 | 3 | 43.020 | 11.913 | 3.55 [3.36–3.76] |
| 262,145 | 8 | 80.755 | 21.493 | 3.75 [3.56–3.95] |

### gicp survey 1

| N | Kind/profile | Serial ms | Parallel ms | Serial/parallel [95%] |
| ---: | ---: | ---: | ---: | --- |
| 4,097 | 0 | 6.424 | 6.494 | 1.00 [0.97–1.04] |
| 4,097 | 1 | 6.465 | 6.343 | 1.01 [0.97–1.05] |
| 4,097 | 2 | 7.156 | 7.096 | 1.04 [0.98–1.10] |
| 32,768 | 0 | 21.479 | 19.466 | 1.11 [1.04–1.19] |
| 32,768 | 1 | 21.806 | 19.504 | 1.10 [1.06–1.15] |
| 32,768 | 2 | 23.211 | 19.070 | 1.17 [1.10–1.26] |
| 65,537 | 0 | 33.916 | 28.439 | 1.21 [1.09–1.32] |
| 65,537 | 1 | 44.369 | 30.647 | 1.27 [1.18–1.37] |
| 65,537 | 2 | 41.717 | 33.129 | 1.25 [1.17–1.34] |
| 262,145 | 0 | 139.662 | 95.646 | 1.40 [1.28–1.55] |
| 262,145 | 1 | 117.238 | 84.482 | 1.36 [1.26–1.46] |
| 262,145 | 2 | 138.657 | 90.196 | 1.48 [1.39–1.56] |

### gicp survey 2

| N | Kind/profile | Serial ms | Parallel ms | Serial/parallel [95%] |
| ---: | ---: | ---: | ---: | --- |
| 4,097 | 0 | 7.088 | 7.099 | 0.99 [0.95–1.03] |
| 4,097 | 1 | 6.475 | 6.384 | 1.01 [0.98–1.06] |
| 4,097 | 2 | 6.545 | 6.570 | 0.99 [0.95–1.04] |
| 32,768 | 0 | 16.184 | 13.855 | 1.18 [1.14–1.22] |
| 32,768 | 1 | 17.245 | 14.891 | 1.10 [1.00–1.19] |
| 32,768 | 2 | 19.592 | 17.434 | 1.11 [1.04–1.20] |
| 65,537 | 0 | 31.721 | 25.296 | 1.16 [1.07–1.25] |
| 65,537 | 1 | 29.991 | 26.272 | 1.17 [1.09–1.25] |
| 65,537 | 2 | 31.590 | 24.876 | 1.24 [1.13–1.37] |
| 262,145 | 0 | 110.475 | 75.764 | 1.39 [1.26–1.53] |
| 262,145 | 1 | 114.617 | 76.459 | 1.48 [1.37–1.60] |
| 262,145 | 2 | 121.355 | 81.975 | 1.45 [1.36–1.55] |

Times are medians in milliseconds; ratios are geometric means of paired
serial/parallel latency. >1 favors parallel. Within-survey 95% intervals use
10,000 paired log-ratio bootstrap draws, seed
20261013+24*(family_is_gicp)+12*(survey-1)+condition_index, sorted endpoints
249/9749. All samples, including adverse results, are retained. Intervals do
not establish portable gains, cold GPU/runtime latency or a global crossover.

At 65,537 points, complete construction improves **1.88–2.55x**, and complete
GPU GICP calls **1.16–1.27x**. At 262,145 points, construction improves
**2.72–3.75x**, complete GICP **1.36–1.48x**. The corresponding within-survey
intervals exceed one in both repeats. Threshold-size construction also improves;
registration improvements there are smaller, and one interval lies near one.
4,097-point calls use serial construction in both cases; their near-one ratios
are measurement variability and are not parallel speedup claims.

All complete registration diagnostics remain exact, not merely poses within a
tolerance. H2D receipts retain the resident-grid formula
20*S+36*T+32*target_nodes+4*(source_cells+target_cells+2); D2H remains 1,024
bytes for three updates. The benefit is CPU preparation time. It does not
remove CPU grid construction, target-tree preparation or the 6x6 solver.
No extra GPU phase is claimed.

## Final verification and delivery

- Search all-feature suite: **58 passed, 6 ignored**; registration all-feature
  library: **35 passed, 3 ignored**. The explicit native 216-condition exact
  GICP comparison and all four exact release benchmark runs pass separately.
  New minimal facade smoke passes.
- All-feature/all-target search/registration Clippy, minimal search parallel
  without KD-tree, CPU-only GICP Clippy, WASM search/optional GPU registration
  compile and workspace formatting pass. WASM is compile checked, not browser
  executed; its constructor remains serial.
- Serial construction, axis selection and partition source are unchanged
  against parent 713ba20. CPU GICP and the common resident update loop remain
  unchanged. No GPU shader, core model, dependency or unsafe changes are added.
- Raw timing/interval results are retained in
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\parallel-kdtree-build-measurements.json`.
  Complete construction logs:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\parallel-kdtree-build-construction-1.log`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\parallel-kdtree-build-construction-2.log`.
  Complete GICP logs:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\parallel-kdtree-build-gicp-1.log`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\parallel-kdtree-build-gicp-2.log`.
  Publication files/PR text omit the local adapter model; scans match zero.
  Existing 124 named CI checks remain required before merge.
