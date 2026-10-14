# Bounded CPU validation for GPU feature inputs

Date: 2026-10-14. Parent: `db9ea890b6f1f176d7dc154e9f9c73b904b390c6` (merged PR 201).

## Change and boundary

Both constructors of `GpuFpfhInput`, `GpuIssInput` and `GpuMlsInput` use a shared
private CPU finite-coordinate/extent validator. Native inputs with at least
524,288 points use available CPU count, useful 131,072-point bands and at most
four scoped workers. Each worker combines coordinate finite checks and min/max
scans for its original-order SoA band. FPFH unit normals in that same band use
the original f32 multiplication/addition order and inclusive 1e-3 tolerance.
The final extent keeps axis order, f64 differences, squares and sum, followed
by the original finite-f32 gate. Min/max ties can differ in zero sign; squaring
zero differences produces the same positive-zero extent. Input words are never
modified. Small, one-worker and WASM calls retain the prior serial scans and
avoid worker startup; small/WASM calls also avoid the CPU-count query.

Bookkeeping adds only worker-sized handles and extrema; thread stacks/startup
are additional bounded overhead. No point-sized host copy,
new device buffer, compute pass, wait or readback is introduced. All worker
scopes join before success or rejection. Existing device planners and FPFH
normal-length checks keep their order; each caller retains its own error.
Both default and explicit GPU-gather constructors gain validation preparation,
without changing payload packing, ownership or receipts. Shared tree/query
validation still runs afterward; this slice does not remove that separate gate.
GPU feature computation and kernels, core, dependencies and unsafe boundaries
are unchanged.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/feature_validation.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/iss.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mls.rs`

## Correctness and integration

Four pure tests compare a frozen serial predicate against forced available
worker counts 0/1/2/3/4/12/usize::MAX and the production entry point. Sizes cover
empty, singleton, 4,097, 262,145, 524,287/288/289 and 1,048,577. Profiles include
ordinary bounds, mixed signed zeros, subnormals, identical f32::MAX coordinates,
overflowing extents and seeded arbitrary finite words. Accepted extent values
match exact f64 bits. Adjacent f32 values around the extent-overflow boundary
and both unit-normal tolerance edges preserve decisions. Nonfinite coordinates
in every axis/worker and nonfinite/overflowing/zero normals reject; malformed
private shapes and worker caps are checked independently.

Two native tests compare frozen PR 201 constructors against both default and
explicit GPU-gather constructors around the new worker cutoff and existing
FPFH gather cutoff. Integer probes compare **104,858,320** sample/query words,
including zero padding and zero signs; transfer receipts match. Large NaN,
infinite, extent-overflow and invalid-normal inputs retain exact feature errors,
including FPFH normal-length precedence when coordinates are also invalid.
Existing bitwise tree/gather/feature, descriptor/status, owned-result, search,
runtime and concurrency tests also remain in the regression selection.

All **66** selected native release tests and **11** public facade smoke tests
passed, along with all-feature/all-target Clippy with
warnings denied, CPU-only/WASM feature builds and formatting are checked before
measurement. Source verification compares the frozen constructors to the
parent after formatting/comment normalization, permits only each constructor's
validation block to change, and verifies planners/computation/packing/receipt
bodies and all pre-existing GPU shaders. The executable and complete GPU
Rust/WGSL source set are hashed before and after both surveys. The final
manifest verifies **324** source/executable hashes, **310** pre-existing source
bodies and **97** unchanged shaders. The forbidden local adapter-name count is
zero in tracked/nonignored source and the PR body.

## Isolated measurements

Two surveys run sequentially after all builds/tests, each after a 15-second
quiet window. A process census throughout each survey rejects an entire
attempt if another Cargo/rustc/linker/test/benchmark process overlaps; it
cancels only the owned measurement process. Accepted attempts must finish with
one passing test, no stderr and no overlap. Statistical analysis runs after
both surveys and source/executable hash verification.

Each condition uses three warm cycles and 30 paired samples, alternating old
and new order per cycle. `hostpos` measures finite/bounds validation;
`hostnorm` adds FPFH unit normals. Host-only calls below 524,288 points repeat
32 times per sample and report time per call; larger host calls use one call.
`fpfh`, `iss` and `mls` measure the complete explicit GPU-gather constructor,
`wait_idle`, receipt/count assertions and destruction of the returned input.
CPU KD-tree construction, fixture allocation and runtime creation are excluded.
There is no payload readback, neighbor search or feature computation in those
timings. The frozen constructors preserve the parent GPU gather policy,
including the existing large-input two-pass FPFH path. Default uploads are
covered by correctness tests, with no latency claim for that route.

The table reports medians and paired geometric old/new ratios. A ratio above
one favors the new code. Each 95% interval uses 10,000 bootstrap resamples of
paired log ratios, sorted indices 249 and 9,749. Seeds are
`2026101433 + 100*survey + 10*case + JSON-field-index`; zero-duration counts are
retained and suppress ratio estimates if nonzero. Profile 0 uses ordinary
bounded coordinates, profile 1 signed zeros and profile 2 subnormals. All
conditions, including slower or inconclusive results, are retained.

Both surveys were accepted on attempt 1, with no overlap and empty stderr;
measurement tests finished in 55.17 and 52.29 seconds. Conditions were native
Windows x86_64, rustc 1.98.1 (`48a229cea`, 2026-09-01), release optimization,
Vulkan discrete backend, 12 available CPU workers with a four-worker cap,
2,147,483,647-byte storage binding limit and 65,535 dispatch groups. All **80**
comparisons retain 30 paired samples each.

All zero-duration counts were zero. At 1,048,577 ordinary points, position
validation medians fell 34.7%/36.0%, while validation including normals fell
52.3%/56.2%. These are host-only results. Completed explicit GPU upload medians
fell **15.8%/18.0% for FPFH**, **15.0%/10.4% for ISS** and **9.0%/10.1% for
MLS**; the six paired intervals are entirely above one. For example, FPFH
medians were 28.874700 → 24.299900 ms and 26.479350 → 21.704750 ms. Ordinary
524,288/289-point conditions also favor the new path in both surveys; the
524,288-point MLS interval in survey 1 only narrowly excludes one. Signed-zero
ISS at 524,289 points is inconclusive in survey 2. These conditions support
the existing conservative size threshold on this environment; they do not
establish a universal CPU/thread/backend optimum.

Below the threshold, all completed-upload intervals cross one except a slower
4,097-point ISS result in survey 2: paired ratio 0.929385
[0.864953, 0.996200], with medians 0.230000 → 0.236700 ms. Survey 1 for that
condition is inconclusive. Some serial normal-validation conditions improve,
which cannot be attributed to parallelism; source extraction/compiler context
also changes. Across all 80 rows, 25 intervals cross one and one is entirely
below one. All are published below; no small-input upload speedup is claimed.
No claim about descriptor/selection/fitting or complete feature execution
follows from this preparation change.

Local evidence paths (ignored artifacts, no adapter model name):

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/feature-validation-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/feature-validation-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/feature-validation-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/feature-validation-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/run-feature-validation-survey.ps1`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-feature-validation.py`


| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 4097 | 0 | hostpos | 0.020708 | 0.020848 | 1.007394 [0.972302, 1.041938] |
| 1 | 4097 | 0 | hostnorm | 0.032752 | 0.030727 | 1.058241 [1.019117, 1.100485] |
| 1 | 4097 | 0 | fpfh | 0.314400 | 0.301100 | 1.006150 [0.927733, 1.089435] |
| 1 | 4097 | 0 | iss | 0.212200 | 0.211650 | 1.005400 [0.954908, 1.058801] |
| 1 | 4097 | 0 | mls | 0.205550 | 0.206300 | 0.967068 [0.911297, 1.022829] |
| 1 | 262145 | 0 | hostpos | 1.334761 | 1.359750 | 1.011561 [0.971373, 1.052400] |
| 1 | 262145 | 0 | hostnorm | 2.605180 | 2.409195 | 1.066837 [0.998495, 1.141093] |
| 1 | 262145 | 0 | fpfh | 19.591250 | 19.699200 | 0.996345 [0.952299, 1.042403] |
| 1 | 262145 | 0 | iss | 10.085600 | 10.157000 | 0.979584 [0.882688, 1.088280] |
| 1 | 262145 | 0 | mls | 11.308650 | 11.649450 | 0.978778 [0.923332, 1.036765] |
| 1 | 524287 | 0 | hostpos | 3.238178 | 3.208661 | 1.006761 [0.969601, 1.041757] |
| 1 | 524287 | 0 | hostnorm | 5.863645 | 5.246941 | 1.098795 [1.070371, 1.128222] |
| 1 | 524287 | 0 | fpfh | 23.627300 | 24.126150 | 0.976518 [0.938280, 1.014948] |
| 1 | 524287 | 0 | iss | 16.416450 | 16.067800 | 0.984542 [0.944341, 1.020745] |
| 1 | 524287 | 0 | mls | 15.866950 | 15.677700 | 1.022143 [0.958792, 1.088153] |
| 1 | 524288 | 0 | hostpos | 3.300150 | 2.517550 | 1.328942 [1.257592, 1.423067] |
| 1 | 524288 | 0 | hostnorm | 5.792900 | 2.931800 | 1.758872 [1.605122, 1.901717] |
| 1 | 524288 | 0 | fpfh | 15.762650 | 13.774650 | 1.124243 [1.079511, 1.170368] |
| 1 | 524288 | 0 | iss | 9.922900 | 9.051700 | 1.133173 [1.065696, 1.209742] |
| 1 | 524288 | 0 | mls | 9.815250 | 9.430450 | 1.042043 [1.000622, 1.082487] |
| 1 | 524289 | 0 | hostpos | 3.239350 | 2.447250 | 1.318632 [1.281467, 1.363623] |
| 1 | 524289 | 0 | hostnorm | 6.079000 | 2.915950 | 1.899731 [1.780418, 2.015535] |
| 1 | 524289 | 0 | fpfh | 15.777700 | 13.453850 | 1.166322 [1.133965, 1.198476] |
| 1 | 524289 | 0 | iss | 9.331950 | 8.344550 | 1.079897 [1.029338, 1.132028] |
| 1 | 524289 | 0 | mls | 8.476950 | 7.708800 | 1.105504 [1.028396, 1.188821] |
| 1 | 1048577 | 0 | hostpos | 7.122550 | 4.650950 | 1.621169 [1.527520, 1.737492] |
| 1 | 1048577 | 0 | hostnorm | 12.073750 | 5.756750 | 2.131776 [2.036064, 2.221341] |
| 1 | 1048577 | 0 | fpfh | 28.874700 | 24.299900 | 1.190537 [1.149146, 1.235550] |
| 1 | 1048577 | 0 | iss | 21.005350 | 17.862750 | 1.193625 [1.135057, 1.261853] |
| 1 | 1048577 | 0 | mls | 19.686850 | 17.922500 | 1.108887 [1.067830, 1.156217] |
| 1 | 524289 | 1 | hostpos | 3.635500 | 2.852050 | 1.300784 [1.235793, 1.380539] |
| 1 | 524289 | 1 | hostnorm | 6.361050 | 3.405600 | 1.804914 [1.706107, 1.897376] |
| 1 | 524289 | 1 | fpfh | 16.980350 | 14.920500 | 1.142807 [1.105052, 1.182851] |
| 1 | 524289 | 1 | iss | 9.417000 | 8.648800 | 1.112033 [1.030141, 1.202807] |
| 1 | 524289 | 1 | mls | 9.639850 | 9.030950 | 1.048368 [1.015590, 1.081953] |
| 1 | 1048577 | 2 | hostpos | 6.951050 | 4.568250 | 1.556878 [1.514847, 1.597647] |
| 1 | 1048577 | 2 | hostnorm | 12.688000 | 5.360600 | 2.285236 [2.166066, 2.404219] |
| 1 | 1048577 | 2 | fpfh | 28.928400 | 22.887800 | 1.229720 [1.174399, 1.288095] |
| 1 | 1048577 | 2 | iss | 19.813800 | 17.994550 | 1.128768 [1.077625, 1.184284] |
| 1 | 1048577 | 2 | mls | 19.150550 | 17.430050 | 1.095175 [1.063623, 1.125297] |
| 2 | 4097 | 0 | hostpos | 0.022127 | 0.022208 | 0.985581 [0.933799, 1.042619] |
| 2 | 4097 | 0 | hostnorm | 0.034611 | 0.032309 | 1.102121 [1.057167, 1.149509] |
| 2 | 4097 | 0 | fpfh | 0.337600 | 0.347000 | 1.001200 [0.900921, 1.117025] |
| 2 | 4097 | 0 | iss | 0.230000 | 0.236700 | 0.929385 [0.864953, 0.996200] |
| 2 | 4097 | 0 | mls | 0.219200 | 0.218200 | 1.024909 [0.991978, 1.059579] |
| 2 | 262145 | 0 | hostpos | 1.409697 | 1.412323 | 1.011291 [0.969776, 1.059266] |
| 2 | 262145 | 0 | hostnorm | 2.584255 | 2.260456 | 1.107080 [1.050735, 1.166217] |
| 2 | 262145 | 0 | fpfh | 13.221850 | 12.995800 | 1.083604 [0.987212, 1.204246] |
| 2 | 262145 | 0 | iss | 6.614100 | 6.880350 | 0.974444 [0.854729, 1.111120] |
| 2 | 262145 | 0 | mls | 7.953700 | 7.744950 | 1.066829 [0.979116, 1.170584] |
| 2 | 524287 | 0 | hostpos | 3.041800 | 2.987897 | 1.028956 [0.992768, 1.065626] |
| 2 | 524287 | 0 | hostnorm | 5.371338 | 4.680567 | 1.110253 [1.063895, 1.158629] |
| 2 | 524287 | 0 | fpfh | 23.982050 | 23.065900 | 0.999617 [0.920692, 1.082858] |
| 2 | 524287 | 0 | iss | 15.644050 | 15.633900 | 1.047940 [0.980048, 1.128552] |
| 2 | 524287 | 0 | mls | 15.723500 | 15.696800 | 0.958330 [0.901934, 1.010795] |
| 2 | 524288 | 0 | hostpos | 3.252000 | 2.512850 | 1.366904 [1.266032, 1.515313] |
| 2 | 524288 | 0 | hostnorm | 6.004000 | 2.880000 | 1.912398 [1.811545, 2.005214] |
| 2 | 524288 | 0 | fpfh | 15.662100 | 13.475700 | 1.158347 [1.129286, 1.187979] |
| 2 | 524288 | 0 | iss | 9.829900 | 9.076150 | 1.101367 [1.069263, 1.150812] |
| 2 | 524288 | 0 | mls | 9.763750 | 9.086400 | 1.105760 [1.051815, 1.171732] |
| 2 | 524289 | 0 | hostpos | 3.084600 | 2.432500 | 1.257487 [1.223442, 1.290468] |
| 2 | 524289 | 0 | hostnorm | 5.860250 | 2.880950 | 1.916045 [1.806359, 2.037631] |
| 2 | 524289 | 0 | fpfh | 17.016300 | 15.185500 | 1.137781 [1.094049, 1.184694] |
| 2 | 524289 | 0 | iss | 9.712600 | 9.235500 | 1.081013 [1.011518, 1.154644] |
| 2 | 524289 | 0 | mls | 8.268350 | 7.763750 | 1.112416 [1.032736, 1.200360] |
| 2 | 1048577 | 0 | hostpos | 6.528350 | 4.177500 | 1.565102 [1.522351, 1.607559] |
| 2 | 1048577 | 0 | hostnorm | 11.695200 | 5.117400 | 2.156874 [2.028685, 2.276891] |
| 2 | 1048577 | 0 | fpfh | 26.479350 | 21.704750 | 1.203998 [1.170966, 1.239223] |
| 2 | 1048577 | 0 | iss | 18.976800 | 17.006000 | 1.122083 [1.088918, 1.162720] |
| 2 | 1048577 | 0 | mls | 18.764200 | 16.870300 | 1.115671 [1.088290, 1.155653] |
| 2 | 524289 | 1 | hostpos | 3.475050 | 2.639700 | 1.287466 [1.236021, 1.337350] |
| 2 | 524289 | 1 | hostnorm | 6.187000 | 3.079350 | 1.946974 [1.864243, 2.022433] |
| 2 | 524289 | 1 | fpfh | 20.797650 | 17.684850 | 1.121704 [1.080321, 1.168930] |
| 2 | 524289 | 1 | iss | 10.673550 | 9.645550 | 1.080534 [0.967503, 1.206500] |
| 2 | 524289 | 1 | mls | 11.353400 | 10.667800 | 1.061207 [1.005884, 1.118629] |
| 2 | 1048577 | 2 | hostpos | 6.848050 | 4.427000 | 1.531781 [1.468868, 1.600177] |
| 2 | 1048577 | 2 | hostnorm | 12.469500 | 5.409600 | 2.150614 [2.055594, 2.241659] |
| 2 | 1048577 | 2 | fpfh | 28.262600 | 23.964450 | 1.198018 [1.150367, 1.248793] |
| 2 | 1048577 | 2 | iss | 20.802100 | 18.052050 | 1.140972 [1.109071, 1.174293] |
| 2 | 1048577 | 2 | mls | 19.680600 | 17.819050 | 1.115785 [1.082733, 1.161053] |
