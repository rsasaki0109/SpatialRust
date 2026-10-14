# Explicit GPU registration input gathering

Date: 2026-10-15. Parent: `2b888ae73760d6d017c44010bb1aa6fb2e97d27a` (merged PR 202).

## Change and boundary

`GpuIcpInput`, `GpuGicpInput` and `GpuPointToPlaneInput` add explicit
`upload_with_gpu_gather` constructors. Their shared ICP position upload uses
the existing integer target tree/original-order query gather. It transfers
original XYZ SoA axes and indices once, retains traversal-order point records
and original-order padded target queries, then uploads existing CPU-packed
nodes. The source remains an explicitly uploaded padded original-order buffer.
GICP retains its bounded covariance/SPD staging, scale and covariance offset;
point-to-plane retains raw-normal validation, common scaling and padded rows.
Geometry validation, original f64 bounds/anchor/scale math and validation order
precede target gathering. The separate finite gate in shared tree gathering
still runs. Default constructors and high-level algorithm placement remain.

Let S be source count, T target count and M target tree-node count:

| Input | Default H2D bytes | GPU-gather H2D bytes | New D2D output bytes |
| --- | --- | --- | --- |
| ICP | 16*S + 32*T + 32*M | 16*S + 16*T + 32*M | 32*T |
| GICP | 48*S + 64*T + 32*M | 48*S + 48*T + 32*M | 32*T |
| Point-to-plane | 16*S + 48*T + 32*M | 16*S + 32*T + 32*M | 32*T |

There is no construction readback or host completion wait. ICP starts with the
tree receipt (16*T D2D) and records the target-query receipt (16*T D2D) as well
as source/target H2D. GICP and point-to-plane inherit that complete position
receipt and add unchanged covariance/normal host payload. Uniform controls are
excluded. Queue submissions retain temporary buffers and order subsequent
search/reduction passes; owned outputs survive CPU input and GPU input drops.

Two transient gather inputs add 16*T device bytes beside owned outputs. Host
target point/query packing allocations totaling 32*T payload bytes over
construction are removed; that cumulative amount is not a peak host saving.
Backend staging/allocator overhead is additional. Existing registration device
requirements already cover the shared gather. New constructors remain safe,
with no new dependency, unsafe boundary, shader or core change. Small inputs
may be slower; source packing, CPU tree construction and geometry preparation
remain. This does not make the complete registration/pose solve GPU resident.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/icp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/gicp_reduction.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/point_to_plane_reduction.rs`

## Correctness and integration

Four focused native tests passed. Frozen parent constructors and independent
original-coordinate probes compare **174,016,032** uploaded source/target/tree/
node/covariance/normal words. Sizes range from one point to 1,048,577, including
65,537, 262,145 and 524,289; profiles include ordinary coordinates, signed
zeros, subnormals with a representable extent and identical f32::MAX points.
An empty source with a nonempty target preserves empty-source shape and receipts.
Metadata checks cover exact anchor/scale words, normal/weight scale and offset.

The output test compares **6,664** owned result words, including status, for
ordinary/zero/subnormal/extreme inputs and identity/translation/rotation poses.
Untrimmed ICP and both DoubleSingle/CompensatedF32 trimmed moments, GICP and
point-to-plane equations match default uploads exactly. CPU inputs are dropped
before computation, and inputs are dropped before owned-result inspection.
Runtime mismatches reject. Thirteen invalid-input/normal/covariance/extent cases
match frozen feature errors before gather cache initialization. Three cold
concurrent constructors share the runtime's paired gather pipeline and retain
complete correspondence counts.

All **91** selected native release tests passed: nearest/shared gathering,
registration geometry, ICP and both trim precision paths, GICP covariance/
equation preparation and point-to-plane reductions. All-feature/all-target
Clippy with warnings denied, CPU-only and WASM facade builds passed. All **8** public
facade smoke tests and final formatting/diff checks also passed.

Source verification compares frozen constructors with the merged parent,
folds the new route to the CPU branch to check default construction, and
checks geometry/planners/normal/covariance preparation, reduction and receipt
bodies. Shared tree/gather production code is unchanged, as are all **97**
pre-existing GPU shaders. Before/after measurement manifests hash the complete
GPU Rust/WGSL sources, public smoke test and release executable.

## Isolated measurements

Two surveys run sequentially after all builds/tests and a 15-second quiet
window each. A process census during each survey rejects a whole attempt if
another build/test/benchmark overlaps and cancels only the owned benchmark.
Accepted attempts require one passing test, no overlap and empty stderr.
Analysis begins after both surveys and source/executable hash verification.

Each of ten cases has three warm cycles and 30 paired samples per constructor,
alternating old/new order per cycle. The timed region includes the constructor,
`wait_idle`, source-count assertion, receipt retrieval and input destruction;
receipt assertions follow the timer. Runtime creation, fixture allocation and
CPU target-tree construction are excluded. No payload readback, search,
reduction iteration or pose solve is timed. All three APIs include their CPU
geometry/source/covariance/normal preparation costs. Profile 0 is ordinary
bounded coordinates, profile 1 signed zeros, profile 2 subnormal coordinates
plus one representable extent, and profile 4 deterministically shuffled target
input. S equals T in measured cases; asymmetric/empty source shapes are checked
for correctness, with no latency estimate for those shapes.

The full table retains all **60** comparisons, including adverse/inconclusive
rows. Ratios are paired geometric old/new ratios; above one favors GPU gather.
The 95% interval uses 10,000 bootstrap resamples of paired log ratios, sorted
indices 249 and 9,749. Seeds are
`2026101501 + 100*survey + 10*case + JSON-field-index`. Zero-duration counts
are retained and suppress ratio estimates if nonzero.

Both surveys were accepted on attempt 1, with no overlap and empty stderr.
The tests took 57.99 and 56.85 seconds. Conditions were native Windows x86_64,
rustc 1.98.1 (`48a229cea`, 2026-09-01), release optimization, Vulkan discrete
backend, 12 available CPU workers, 2,147,483,647-byte storage binding limit and
65,535 dispatch groups. All zero-duration counts were zero. Final verification
matches **329** source/executable hashes, **318** pre-existing source bodies
and all **97** shaders; the forbidden local adapter-name count is zero.

At 1,048,577 shuffled target points, ICP completed-upload medians fell
**16.4%/21.4%** (41.647200 → 34.822850 ms and 42.322100 → 33.254650 ms),
with paired ratios 1.265625 [1.145704, 1.394057] and 1.313437
[1.216493, 1.419904]. GICP medians fell **18.0%/14.6%** (73.130150 →
59.984250 ms and 72.927400 → 62.266950 ms), with paired ratios 1.187221
[1.117557, 1.250344] and 1.201069 [1.166607, 1.235999]. Ordinary million-point
GICP favors gathering in both surveys, but its median reductions differ
(12.8%/4.8%); ordinary million-point ICP is inconclusive in survey 2.

Point-to-plane has no repeatable speedup claim: its shuffled million-point
interval crosses one in survey 1 and favors gathering in survey 2; ordinary
million-point gathering is inconclusive in survey 1 and **25.8% slower by
median** in survey 2, ratio 0.808891 [0.770562, 0.848582]. Both ordinary and
signed-zero 524,289-point plane cases regress in both surveys. Some ICP
524,289-point cases also regress or are inconclusive. The new explicit route
guarantees the documented payload reduction, not lower latency.

All singleton and 4,097-point cases regress in both surveys, reflecting the
extra gather pass/submission. Of 60 comparisons, **24** intervals are entirely
below one and **14** cross one. All rows are retained below. Use the explicit
route only when its transfer/memory tradeoff is useful; default and high-level
placement are unchanged. No general auto-selection threshold is introduced. No claim about registration iterations,
pose solving or overall registration latency follows from upload measurements.


Local ignored evidence paths:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/registration-gather-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/registration-gather-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/registration-gather-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/registration-gather-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/run-registration-gather-survey.ps1`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-registration-gather.py`

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | icp | 0.020750 | 0.176050 | 0.124779 [0.109999, 0.141677] |
| 1 | 1 | 0 | gicp | 0.031350 | 0.165300 | 0.176019 [0.149634, 0.204039] |
| 1 | 1 | 0 | plane | 0.029900 | 0.161600 | 0.159056 [0.139430, 0.178412] |
| 1 | 4097 | 0 | icp | 0.123550 | 0.345700 | 0.387841 [0.349766, 0.428812] |
| 1 | 4097 | 0 | gicp | 0.434450 | 0.571750 | 0.719252 [0.657324, 0.781749] |
| 1 | 4097 | 0 | plane | 0.215100 | 0.413000 | 0.499526 [0.473519, 0.528452] |
| 1 | 65537 | 0 | icp | 2.886900 | 3.334000 | 0.982100 [0.844739, 1.140721] |
| 1 | 65537 | 0 | gicp | 6.514750 | 6.348100 | 1.034048 [0.992684, 1.075122] |
| 1 | 65537 | 0 | plane | 4.808500 | 5.422150 | 0.914441 [0.868904, 0.960739] |
| 1 | 262145 | 0 | icp | 13.050850 | 9.706350 | 1.341257 [1.263087, 1.416083] |
| 1 | 262145 | 0 | gicp | 22.570350 | 18.027200 | 1.241680 [1.193452, 1.286988] |
| 1 | 262145 | 0 | plane | 19.654550 | 17.788600 | 1.074892 [0.973749, 1.172340] |
| 1 | 524289 | 0 | icp | 21.302350 | 21.107700 | 0.913139 [0.810960, 1.020450] |
| 1 | 524289 | 0 | gicp | 37.471000 | 33.900200 | 1.141409 [1.102800, 1.180541] |
| 1 | 524289 | 0 | plane | 32.353600 | 39.624550 | 0.882085 [0.833405, 0.933074] |
| 1 | 1048577 | 0 | icp | 39.296150 | 34.004800 | 1.189855 [1.105488, 1.272897] |
| 1 | 1048577 | 0 | gicp | 71.028250 | 61.904950 | 1.148529 [1.098163, 1.203500] |
| 1 | 1048577 | 0 | plane | 63.566350 | 62.810900 | 0.991669 [0.937927, 1.045449] |
| 1 | 524289 | 1 | icp | 21.904250 | 24.301950 | 0.915576 [0.815182, 1.019806] |
| 1 | 524289 | 1 | gicp | 36.639600 | 32.569950 | 1.099694 [1.067224, 1.131010] |
| 1 | 524289 | 1 | plane | 32.587600 | 38.369700 | 0.813940 [0.763143, 0.865643] |
| 1 | 524289 | 2 | icp | 22.101100 | 19.512450 | 1.160065 [1.028039, 1.285088] |
| 1 | 524289 | 2 | gicp | 39.041300 | 32.461350 | 1.163922 [1.097552, 1.223718] |
| 1 | 524289 | 2 | plane | 33.292000 | 31.215250 | 1.052557 [1.004661, 1.095508] |
| 1 | 524289 | 4 | icp | 24.435050 | 32.141500 | 0.844392 [0.768341, 0.921207] |
| 1 | 524289 | 4 | gicp | 43.601400 | 39.011900 | 1.107862 [1.055749, 1.159968] |
| 1 | 524289 | 4 | plane | 36.981250 | 43.589350 | 0.854241 [0.800287, 0.906652] |
| 1 | 1048577 | 4 | icp | 41.647200 | 34.822850 | 1.265625 [1.145704, 1.394057] |
| 1 | 1048577 | 4 | gicp | 73.130150 | 59.984250 | 1.187221 [1.117557, 1.250344] |
| 1 | 1048577 | 4 | plane | 66.546050 | 61.338800 | 1.030023 [0.967261, 1.093209] |
| 2 | 1 | 0 | icp | 0.029550 | 0.175050 | 0.163918 [0.150747, 0.178000] |
| 2 | 1 | 0 | gicp | 0.037950 | 0.170850 | 0.228202 [0.219303, 0.237762] |
| 2 | 1 | 0 | plane | 0.037200 | 0.174100 | 0.219193 [0.209070, 0.231490] |
| 2 | 4097 | 0 | icp | 0.132250 | 0.307500 | 0.452042 [0.435123, 0.470136] |
| 2 | 4097 | 0 | gicp | 0.450450 | 0.598900 | 0.767502 [0.743857, 0.794494] |
| 2 | 4097 | 0 | plane | 0.221000 | 0.391750 | 0.570238 [0.547128, 0.595432] |
| 2 | 65537 | 0 | icp | 2.852900 | 3.127000 | 1.013592 [0.880225, 1.171764] |
| 2 | 65537 | 0 | gicp | 6.030150 | 5.883550 | 1.040385 [1.001502, 1.082634] |
| 2 | 65537 | 0 | plane | 4.458750 | 5.135500 | 0.885591 [0.854126, 0.919717] |
| 2 | 262145 | 0 | icp | 13.050700 | 9.486100 | 1.219642 [1.099772, 1.335152] |
| 2 | 262145 | 0 | gicp | 21.576000 | 18.315300 | 1.142810 [1.083236, 1.201634] |
| 2 | 262145 | 0 | plane | 18.713250 | 17.805450 | 1.007371 [0.918763, 1.101768] |
| 2 | 524289 | 0 | icp | 22.180350 | 25.603700 | 0.852452 [0.762414, 0.945143] |
| 2 | 524289 | 0 | gicp | 37.740350 | 35.175900 | 1.056390 [0.994108, 1.115958] |
| 2 | 524289 | 0 | plane | 32.992950 | 40.729850 | 0.803880 [0.758873, 0.843846] |
| 2 | 1048577 | 0 | icp | 37.881400 | 36.721150 | 0.951998 [0.841583, 1.074873] |
| 2 | 1048577 | 0 | gicp | 69.485700 | 66.153000 | 1.052676 [1.011685, 1.092175] |
| 2 | 1048577 | 0 | plane | 62.151700 | 78.179350 | 0.808891 [0.770562, 0.848582] |
| 2 | 524289 | 1 | icp | 21.436000 | 20.628850 | 0.999561 [0.874731, 1.116661] |
| 2 | 524289 | 1 | gicp | 37.981250 | 32.954700 | 1.152360 [1.124859, 1.180725] |
| 2 | 524289 | 1 | plane | 32.957000 | 34.319300 | 0.936246 [0.879742, 0.990080] |
| 2 | 524289 | 2 | icp | 22.454100 | 30.980100 | 0.831319 [0.766238, 0.904969] |
| 2 | 524289 | 2 | gicp | 38.994650 | 35.837100 | 1.086978 [1.036661, 1.134059] |
| 2 | 524289 | 2 | plane | 33.458150 | 41.002500 | 0.793661 [0.748043, 0.832848] |
| 2 | 524289 | 4 | icp | 22.537350 | 21.636050 | 1.035932 [0.906936, 1.167639] |
| 2 | 524289 | 4 | gicp | 38.731450 | 33.130350 | 1.123376 [1.043585, 1.192247] |
| 2 | 524289 | 4 | plane | 34.207100 | 32.907700 | 0.970996 [0.877066, 1.061154] |
| 2 | 1048577 | 4 | icp | 42.322100 | 33.254650 | 1.313437 [1.216493, 1.419904] |
| 2 | 1048577 | 4 | gicp | 72.927400 | 62.266950 | 1.201069 [1.166607, 1.235999] |
| 2 | 1048577 | 4 | plane | 65.524650 | 62.554800 | 1.045765 [1.014461, 1.079467] |
