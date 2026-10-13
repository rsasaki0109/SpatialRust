# Resident grid GICP covariance preparation — 2026-10-13

The bounded resident-radius preparation removed covariance transfers but added
source KD-tree construction/staging and N*(8+8*capacity) neighborhood storage.
This slice offers explicit `align_with_gpu_resident_grid_covariances` under
`register-gicp-gpu` for point sets whose existing dense radius grids fit.
It reuses the canonical grid covariance kernel on resident original positions,
with no source tree, neighborhood rows or covariance readback/re-upload.
CPU counting-sort grids, target tree and the existing 6x6 pose solve remain.
No default placement or automatic sparse/CPU fallback is added.

## Data, ownership and transfer boundary

`GpuGicpInput::upload_with_gpu_grid_covariances(runtime, (x,y,z), target_tree,
radius, capacity, epsilon)` validates source axes and builds both CPU grids
with the existing checked counting-sort preparation. Original source/target
query buffers supply GPU covariance samples. Sorted indices and CSR offsets
are the only new grid uploads. These private buffers are exclusively leased
through one encoder submission, then safely recycled in queue order. Two
source/target grid dispatches share one compute pass; a separate final pass
zeros BOTH covariance sets on any invalid status. No host wait is introduced.

Covariance storage is owned, 32*(S+T+1) bytes: a 32-byte status prefix followed
by original-order source and target `[xx,yy,zz,xy,xz,yz,0,0]` rows. Status bits
1/4/8 represent invalid grid rows, nonfinite/non-SPD/unbounded covariance or
support overflow. Any failure sets equation status 16 and zeros ALL equation
payload before explicit download rejects it. Outputs survive input drop and
scratch reuse. No extra production diagnostic readback is required.

H2D is **20*S+36*T+32*target_nodes+4*(source_cells+target_cells+2)** bytes.
This includes original positions/tree, sorted indices and grid sentinel
entries. Preparation D2H is zero. Registration downloads only **256*(I+1)**
bytes for I updates; `covariance_batches=2`. Device writes/control uniforms are
excluded consistently with existing receipts. Grid auxiliary storage is
4*(S+T+source_cells+target_cells+2) bytes, replacing both bounded neighborhood
buffers and the additional source-tree staging. Dense grids can still cost
more device memory for sparse extents; the caller selects a route explicitly.

## Math and checked limits

The one canonical radius-grid shader remains unchanged. The new shader adapter
changes only position loads, packed output stores and validation. Raw absolute
f32 means, central sums, inclusive radius predicate, counting-sort traversal
order, 16 sweeps of three Jacobi pivots and eigenvector covariance rebuild are
retained. Fewer than three samples use epsilon*I; coincident larger support
retains the legacy smallest-axis convention. f32 orthogonality rounding of
legacy sweeps is preserved, rather than replacing the rebuild with an ideal
rank-one formula. Static adaptation anchors and exact legacy-word tests guard
future canonical interface changes.

Support capacity includes self/duplicates. The first pass rejects at capacity+1,
without truncating a covariance or allocating capacity-sized rows. Grid starts,
ends and original indices are checked before sample access. Packed f32 Cholesky
pivots and analytic inverse trace validate both covariance sets; global weight
normalization is 6/epsilon. The common equation/world-update loop is unchanged.

Requires nonempty finite clouds, matching source axes, positive capacity,
epsilon as f32 in 1e-4..=1, and positive finite radius with normal f32 square.
Existing grid cell cap, reciprocal/dimension/storage/dispatch validation apply.
Each dataset is restricted to 2^24 points to keep legacy f32 support counts
exact, subject to tighter equation/device limits. 256-thread workgroups and
combined covariance/dispatch offsets are checked before CPU grid allocation.
Dense grid rejection is explicit; callers can use the existing bounded-radius
method for sparse extents. Numerical failure, including overflowing raw means
at huge absolute origins, propagates instead of silently changing arithmetic.
CPU/grid and centered bounded-radius math can still choose different normals,
poses or stopping near gates, repeated spectra and numerical limits.

## Correctness evidence

- Native source/unequal-target comparisons cover 1/2/3/63/64/65/257/1025 points,
  three scales (1e-4/1/1e4) and three radii (0.075/0.25/0.5 times scale), with
  duplicates, tilted planes and sparse support. **22,230 covariance rows /
  133,380 f32 values** match the previous grid API bit for bit; all new padding
  and prefix reserved words are zero. Exact upload receipts match independent
  cell/node counts; there is no covariance/neighborhood readback in production.
- Exact-radius and next-representable positions, sparse epsilon boundaries,
  source/target support overflow, malformed sorted indices/grid offsets and
  raw-mean overflow exercise prefix/global zeroing and equation rejection.
- Covariance-derived equations preserve Euclidean error and physically rescaled
  Hessian/gradient tolerances. Owned equations survive input drops; mismatched
  runtimes reject. Four concurrent input preparations repeatedly exercise
  private grid scratch leases without changing outputs.
- **162** registration size/scale/radius/stopping conditions compare host updates
  and previous GPU reduction, immutable inputs, independent Euclidean fitness
  and exact complete receipts. Maximum scale-normalized host-update coordinate
  difference is **9.536743e-7** across **148,392** points. The host reference
  uses legacy shared-runtime GPU radius covariance, not pure CPU preparation.
- Unequal clouds/outliers and concurrent complete calls retain correct gated
  counts. Dense cell overflow rejects this explicit method; the existing
  bounded-radius method handles the tested sparse extent. Minimal facade smoke
  and invalid configuration/limits are checked.

## Completed-call measurements

The exact native release benchmark is
`completed_resident_grid_covariance_gicp_calls` in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-registration\tests\gpu_gicp_resident_grid_covariances_benchmark.rs`.
Runtime creation is excluded. Three warmups precede 20 cyclic samples for
five stages: ordinary host updates, legacy GPU reduction, bounded resident
radius preparation, new resident grid preparation and prepared equations.
Complete calls include CPU grids/trees, uploads, preparation, all GPU waits,
readbacks, CPU solves/composition and final fitness. Prepared equations time
one completed evaluation on precomputed/uploaded input, excluding preparation,
upload, readback and CPU solve; this is a separate stage, not full-call latency.

Four sizes (257/4097/16384/65537) use fixed 0.5 spacing, a 32x32 coordinate
grid, radii 0.51/0.8/1.05 and capacities 32/64/128. The last profile duplicates
each grid point and adds 17 targets. Sources start at binary translation
`(0.03125,-0.015625,0.0078125)`, distance gate 0.25, epsilon 0.001, three
updates and zero stopping thresholds. All warmed/timed results match their
warmed references; every final evaluation retains N correspondences.

Ordinary **host updates** still use shared-runtime GPU radius covariances;
they are not a pure CPU radius baseline. The **legacy GPU** route reads back
and re-uploads grid covariances. The **bounded radius** route retains the
centered/scaled KD-tree-neighborhood precision path. All three completed-call
baselines are measured, preserving differences in arithmetic and preparation.
Surveys run alone without simultaneous local builds/tests/statistical work.
Backend/type/limits are recorded without the local adapter model.

### Survey 1

| N | Radius / capacity | Host ms | Legacy GPU ms | Bounded radius ms | New grid ms | Prepared ms | Host/new [95%] | Legacy/new [95%] | Bounded/new [95%] |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 257 | 0.51 / 32 | 2.711 | 7.175 | 5.210 | 5.005 | 0.257 | 0.54 [0.53–0.56] | 1.43 [1.40–1.47] | 1.02 [1.00–1.04] |
| 257 | 0.8 / 64 | 2.937 | 7.933 | 5.752 | 5.284 | 0.271 | 0.55 [0.53–0.57] | 1.50 [1.45–1.55] | 1.06 [1.02–1.10] |
| 257 | 1.05 / 128 | 2.758 | 7.665 | 5.375 | 5.235 | 0.249 | 0.51 [0.49–0.53] | 1.39 [1.30–1.47] | 0.98 [0.92–1.05] |
| 4,097 | 0.51 / 32 | 10.539 | 9.144 | 6.595 | 5.648 | 0.723 | 1.86 [1.77–1.94] | 1.60 [1.51–1.69] | 1.20 [1.15–1.24] |
| 4,097 | 0.8 / 64 | 10.797 | 10.538 | 9.297 | 6.138 | 0.887 | 1.73 [1.61–1.83] | 1.62 [1.50–1.72] | 1.52 [1.46–1.59] |
| 4,097 | 1.05 / 128 | 9.519 | 10.713 | 13.002 | 6.404 | 0.849 | 1.46 [1.42–1.50] | 1.62 [1.54–1.70] | 1.92 [1.81–2.04] |
| 16,384 | 0.51 / 32 | 41.818 | 20.641 | 21.481 | 15.122 | 2.196 | 2.64 [2.47–2.79] | 1.34 [1.24–1.44] | 1.45 [1.36–1.53] |
| 16,384 | 0.8 / 64 | 49.736 | 18.893 | 23.217 | 13.831 | 1.844 | 3.47 [3.34–3.61] | 1.36 [1.32–1.39] | 1.67 [1.64–1.71] |
| 16,384 | 1.05 / 128 | 44.878 | 16.087 | 20.483 | 11.324 | 1.051 | 3.81 [3.50–4.14] | 1.36 [1.25–1.47] | 1.88 [1.71–2.11] |
| 65,537 | 0.51 / 32 | 202.410 | 52.115 | 50.756 | 29.753 | 2.169 | 5.80 [5.14–6.48] | 1.47 [1.33–1.63] | 1.60 [1.46–1.78] |
| 65,537 | 0.8 / 64 | 210.300 | 46.296 | 48.961 | 34.220 | 2.261 | 5.62 [4.94–6.37] | 1.31 [1.22–1.43] | 1.58 [1.47–1.71] |
| 65,537 | 1.05 / 128 | 173.850 | 43.560 | 66.350 | 35.464 | 2.445 | 4.42 [3.98–4.86] | 1.18 [1.06–1.31] | 1.84 [1.66–2.07] |

### Survey 2

| N | Radius / capacity | Host ms | Legacy GPU ms | Bounded radius ms | New grid ms | Prepared ms | Host/new [95%] | Legacy/new [95%] | Bounded/new [95%] |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 257 | 0.51 / 32 | 4.768 | 13.285 | 9.253 | 9.208 | 0.411 | 0.54 [0.51–0.57] | 1.49 [1.41–1.59] | 1.08 [1.00–1.18] |
| 257 | 0.8 / 64 | 4.728 | 12.959 | 8.489 | 8.475 | 0.389 | 0.54 [0.52–0.57] | 1.49 [1.40–1.59] | 1.03 [0.98–1.08] |
| 257 | 1.05 / 128 | 3.266 | 8.322 | 6.215 | 6.181 | 0.306 | 0.52 [0.48–0.55] | 1.36 [1.29–1.42] | 1.01 [0.95–1.05] |
| 4,097 | 0.51 / 32 | 15.123 | 13.122 | 9.640 | 8.775 | 0.986 | 1.73 [1.56–1.93] | 1.54 [1.43–1.65] | 1.16 [1.09–1.23] |
| 4,097 | 0.8 / 64 | 21.513 | 17.068 | 14.783 | 11.950 | 1.227 | 1.74 [1.67–1.82] | 1.41 [1.35–1.49] | 1.23 [1.18–1.29] |
| 4,097 | 1.05 / 128 | 18.519 | 17.157 | 17.956 | 10.842 | 1.143 | 1.58 [1.45–1.70] | 1.48 [1.39–1.55] | 1.67 [1.59–1.74] |
| 16,384 | 0.51 / 32 | 67.336 | 28.589 | 30.716 | 21.468 | 3.076 | 3.38 [3.14–3.66] | 1.33 [1.23–1.42] | 1.42 [1.32–1.50] |
| 16,384 | 0.8 / 64 | 60.468 | 22.640 | 27.925 | 18.429 | 2.138 | 3.44 [3.16–3.76] | 1.29 [1.23–1.35] | 1.68 [1.59–1.77] |
| 16,384 | 1.05 / 128 | 56.347 | 19.219 | 24.583 | 13.971 | 1.168 | 3.98 [3.61–4.39] | 1.45 [1.37–1.54] | 2.03 [1.78–2.31] |
| 65,537 | 0.51 / 32 | 250.854 | 63.364 | 74.648 | 49.058 | 6.938 | 5.59 [4.85–6.45] | 1.40 [1.31–1.51] | 1.65 [1.56–1.77] |
| 65,537 | 0.8 / 64 | 234.957 | 59.528 | 70.864 | 47.342 | 2.336 | 5.17 [4.54–5.89] | 1.25 [1.16–1.35] | 1.56 [1.48–1.66] |
| 65,537 | 1.05 / 128 | 181.037 | 43.432 | 63.620 | 34.838 | 2.027 | 5.40 [5.06–5.79] | 1.36 [1.24–1.51] | 1.91 [1.67–2.24] |

Times are medians in milliseconds. Ratios are geometric means of paired
baseline/new latency; >1 favors the new grid method. The 95% intervals use
10,000 paired log-ratio bootstrap draws, seed
20261013+36*(survey-1)+3*condition_index+baseline_index (host=0, legacy=1,
bounded=2), sorted endpoints 249/9749. Intervals describe each survey, not
portable backend/device gains, cold-runtime latency or a universal crossover.
Every sample, including adverse cases, is retained; large within/between-survey
variance is visible even in the separate prepared-equation stage.

For every tested 4,097-or-more condition, new grid improves on legacy GPU by
1.18–1.62x and on bounded radius by 1.16–2.03x; within-survey intervals stay
above one. At 65,537 points, legacy/new is **1.18–1.47x**, bounded/new is
**1.56–1.91x**, and host-update/new is **4.42–5.80x**. This completes an
end-to-end latency improvement over the previous implementations in these
conditions, including CPU preparation and transfers. The measurements do not
separately attribute time to tree removal, neighborhood storage removal,
submission count or the retained grid precision path.

At 257 points, host/new is 0.51–0.55: host updates remain faster. Legacy GPU/new
is 1.36–1.50, while bounded/new is about 0.98–1.08 with several intervals
crossing one. Small-cloud improvement over bounded radius is not established.
No automatic dispatch or default placement change is introduced.

All timed outputs match their warmed references. Every final evaluation
retains N correspondences. Maximum measured host/new pose-coordinate error is
**5.9604645e-8**. Host fitness is zero; new fitness is at most
**1.7646659e-16**. Relative fitness error near zero is not useful. The covariance
values preserve the canonical grid; common weight normalization differs from
the legacy uploaded-covariance route and can change low-order pose rounding.
Both surveys pass (41.68/50.12 seconds excluding compilation), using Vulkan /
DiscreteGpu with storage binding limit 2,147,483,647 bytes and maximum dispatch
dimension 65,535. Eight storage bindings and 256-thread workgroups are required
by the full route. No local adapter model is included.

| N | Radius | Legacy H2D | Bounded H2D | New grid H2D | Legacy D2H | Bounded / new D2H |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 257 | 0.51 | 40,056 | 18,560 | 17,440 | 17,472 | 1,024 |
| 257 | 0.8 | 39,032 | 18,560 | 16,416 | 17,472 | 1,024 |
| 257 | 1.05 | 40,632 | 20,064 | 17,268 | 18,016 | 1,024 |
| 4,097 | 0.51 | 637,144 | 295,040 | 276,608 | 263,232 | 1,024 |
| 4,097 | 0.8 | 615,992 | 295,040 | 255,456 | 263,232 | 1,024 |
| 4,097 | 1.05 | 610,640 | 296,672 | 249,356 | 263,776 | 1,024 |
| 16,384 | 0.51 | 2,540,128 | 1,179,584 | 1,098,336 | 1,049,600 | 1,024 |
| 16,384 | 0.8 | 2,456,808 | 1,179,584 | 1,015,016 | 1,049,600 | 1,024 |
| 16,384 | 1.05 | 2,434,456 | 1,181,216 | 991,916 | 1,050,144 | 1,024 |
| 65,537 | 0.51 | 10,183,856 | 4,718,720 | 4,416,600 | 4,195,392 | 1,024 |
| 65,537 | 0.8 | 9,830,712 | 4,718,720 | 4,063,456 | 4,195,392 | 1,024 |
| 65,537 | 1.05 | 9,730,760 | 4,720,352 | 3,962,756 | 4,195,936 | 1,024 |

Legacy-to-new H2D removes exactly **44*(S+T)** bytes: radius XYZ re-upload and
packed covariance upload. For 65,537 points, new H2D is
3,962,756–4,416,600 bytes versus legacy 9,730,760–10,183,856 (56.63–59.28% less),
and versus bounded 4,718,720–4,720,352 (6.40–16.05% less). D2H is still only
**1,024 bytes** for three updates, versus legacy 4,195,392–4,195,936 bytes
(99.9756% less), matching bounded radius. The duplicated 65,537/65,554 capacity
128 condition no longer allocates its 135,285,912-byte neighborhood payload.
These are logical payload/storage counts, not measured process/device peak
memory; pooling and driver resource lifetime can affect actual allocation.

## Final verification and delivery

- GPU all-feature library: **205 passed, 20 ignored**; integration targets:
  **27 passed, 3 ignored**, **232 passing tests** in total. New grid tests and
  prior resident covariance, independent f64 equations and neighborhood tests
  pass together.
- Registration all-feature library: **35 passed, 1 ignored**. Three new native
  grid tests, three bounded-radius tests and two k-NN tests pass, covering 162
  conditions per preparation family. Minimal grid facade smoke passes.
- GPU/registration all-feature/all-target Clippy, CPU-only GICP Clippy, optional
  WASM registration compile, minimal facade check and workspace formatting pass.
  WASM is compile checked, not browser executed.
- CPU GICP, canonical grid shader/API and the common post-preparation update/
  fitness loop are unchanged against parent 5688fc6. No core/math/search code,
  dependency, unsafe or automatic placement changes are added.
- Measurements and bootstrap results:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-grid-covariances-measurements.json`.
  Survey logs:
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-grid-covariances-bench-1.log`
  and
  `C:\Users\rsasa\Workspace\SpatialRust\target\automation\gpu-gicp-resident-grid-covariances-bench-2.log`.
  Published files and PR text omit the local adapter model; scan matches are zero.
  The existing 124 named CI checks and pinned parent slice remain merge gates.
