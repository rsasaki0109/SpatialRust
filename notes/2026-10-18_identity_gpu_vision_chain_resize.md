# Skip identity resize in the resident GPU Vision chain

Date: 2026-10-18. Named slice: identity GPU Vision chain resize.
Immediate parent: `8b7965639ecb5026e1a60b56c7c45083de1d99f9` (merged PR #276).

## Change and observable behavior

`run_gpu_vision_chain` and its profiled form start grayscale directly from the
borrowed source when both source dimensions match the requested output. The
previous chain always submitted a nearest-neighbor resize, including an identity
copy. Changed dimensions, including an equal-area swapped aspect ratio, still
execute the original resize. Standalone `resize_nearest_gpu` retains its original
identity-copy behavior.

The identity branch explicitly preserves source/runtime validation and the entire
original resize coordinate-product, dispatch and device-limit admission before
doing work. It borrows the uploaded image and recycles only owned intermediates.
All grayscale, blur, Sobel, morphology, normalization and CHW packing arithmetic
and shaders are unchanged. No feature, dependency, unsafe code, core change or
public signature is added. GPU APIs remain provisional under the stability policy.

An identity chain omits one encoder/compute-pass submission, resize uniform and
bind-group setup, and destination texture acquire/write. Its profiler now
requires and records 5/6/7 stages instead of 6/7/8, depending on separable blur and
morphology. Changed-dimension calls still require 6/7/8. Larger existing profiler
capacities remain usable, while collecting an identity chain reads 16 fewer query
bytes. Capacity rejection preserves prior recording; a failed recording remains
invalid for later use or collection.

The output receipt omits this invocation's resize stage and its `4*N` destination
write bytes, where N is output pixels and counters are unsaturated. Incoming
receipt history, including earlier resize/readback labels, is preserved. Each
counter still saturates through the remaining original stage additions, rather
than subtracting `4*N` from a saturated parent total. H2D/D2H image/tensor payload
bytes retain their explicit boundaries; the resident call performs no readback.
One fewer texture acquire request does not establish a physical allocation or
peak-memory reduction because pooling may reuse existing allocations. Bandwidth
and physical memory are unmeasured.

## Correctness and provenance

Four native tests cover 576 complete chain cases: six source shapes from 1x1
through 257x259, seeded and constant inputs, six blur/morphology combinations,
both morphology operations and four output shapes. There are 13,076,640 exact
tensor-word comparisons of the parent and current outputs against an independent
CPU implementation. The CPU oracle independently implements nearest mapping,
u64 BT.601 grayscale, unrounded u64 box prefix sums with one final round, signed
Sobel L1, clipped morphology extrema and f32 normalization.

Additional checks cover exact 5/6/7 and 6/7/8 profiler capacity, accumulated
recordings, unchanged errors and rejection before resource work, a foreign
runtime, strided ROI upload, inherited resize/readback history, dirty unused
physical alpha, saturated receipts, default/tiny/zero pool budgets, concurrent
identity/resized work, caller source ownership and retained tensors after pool
clears and changed-input reuse. The default serial-morphology fixture requests
four image textures instead of five; this is a request count.

The existing 4096x4096 case now verifies five stages, 80 query bytes, 335,544,320
accounted GPU write bytes and all 16,777,216 zero tensor words. Explicit upload
and tensor readback each remain 67,108,864 bytes. Its larger physical working set
is not estimated from receipt totals.

The complete immediate-parent chain functions are frozen in the native fixture.
All shared standalone algorithm bodies and shaders are byte-identical to the
parent; the existing resize validator only gains private sibling visibility.
The production source audit admits only the identity selection/profiler count,
validator visibility and the existing large test's corresponding expectations.

IDENTITY_CHAIN_OUTPUT: verified 13076640 exact parent/CPU/current tensor words across 576 identity/width-change/height-change cases, both morphology operations, serial/separable/wide blur and original source bytes/history/metadata.

IDENTITY_CHAIN_PROFILE: verified exact 5/6/7 identity and 6/7/8 resize capacities, one fewer stage and 16 query bytes, accumulated recordings, preflight rejection and failed-recording safety.

IDENTITY_CHAIN_RESOURCES: verified original admission/errors, foreign devices and metadata-only invalid images, ROI/history/readback preservation, dirty alpha and saturated receipts; five versus four texture requests at normal/tiny/zero budgets.

IDENTITY_CHAIN_CONCURRENT: verified mixed identity/resize and separable first use, shared immutable source and independently owned tensors after source recycling, pool clears and changed-data reuse.

There are 604 unchanged original GPU/image sources and 616 current
source/manifest/executable SHA-256 values, including the ignored lockfile. Hashes agree before measurement and after
both warm/fresh-cache surveys and broad validation. Repository/PR adapter-name
matches are zero; process names/PIDs remain in ignored local diagnostics.

## Validation

| Check | Passed |
| --- | ---: |
| image | 182 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native test executions | 185 |

All-features/all-targets Clippy with warnings denied, CPU-only checks, minimal
and CI-feature WASM with warnings denied, formatting and diff checks pass. Image
tests include ignored correctness cases and exclude explicit benchmarks.

## Measurement method

```text
cargo test -p spatialrust-gpu --all-features --release --lib image::vision_chain::identity_tests::benchmark::benchmark_identity_vision_chain -- --ignored --exact --nocapture --test-threads=1
cargo test -p spatialrust-gpu --all-features --release --lib image::vision_chain::identity_tests::benchmark::benchmark_identity_vision_chain_cold -- --ignored --exact --nocapture --test-threads=1
```

Two consecutive warm surveys use the same release executable, 15 cases, three
warmup pairs and 30 measured pairs per route. Nine cases match source/output
dimensions and six controls resize, including a same-area aspect-ratio change.
Fixtures use seeded RGB data except the marked constant-137 case. Defaults are
Dilate, scale=1/255, mean=0 and std=1. Resident/full order rotates through four
positions using offset XOR (round modulo four); GPU pair order alternates.
Every output is checked against the independent CPU result after timing, and
destruction/recycling are excluded. No workflow build or test overlaps a timed
survey. Shared-host work remains uncontrolled; no case is retried or filtered.

- `gpu`: sum of actual compute-stage timestamps in the complete public profiled
  chain. This excludes submissions, inter-pass gaps, CPU work and payload copies.
- `resident`: public resident chain plus idle wait; upload and checking readback
  are excluded.
- `full`: explicit upload, public chain and tensor readback, including host
  transfer preparation and payload transfers.

Ratios are paired geometric means of parent/current (>1 favors current). Each
95% interval bootstraps 10,000 resamples of 30 paired log ratios, seed
2026101819+1000*survey+10*case_index+route_index, one-based survey, zero-based
case/route indices (gpu, resident, full), sorted draws 249/9749. Intervals above
or below one are favorable or adverse; other intervals are inconclusive. Zeros
remain undefined without epsilon. There is no multiplicity correction. Unchanged
resize controls share changed compiled host code, so unchanged scheduling is
not a zero-overhead guarantee; a favorable control is not attributed to the
omitted identity operation.

| Route | Identity: 18 comparisons | Resize controls: 12 comparisons |
| --- | --- | --- |
| gpu | 18 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1 favorable / 11 inconclusive / 0 adverse / 0 unresolved |
| resident | 15 favorable / 3 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 12 inconclusive / 0 adverse / 0 unresolved |
| full | 8 favorable / 10 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 12 inconclusive / 0 adverse / 0 unresolved |

Identity GPU ratios range from 1.024128 to 1.184947, about
2.4-15.6% less GPU stage time in these conditions. Full-call improvement is supported in only
eight of 18 identity comparisons; the remainder are inconclusive. These are
local paired diagnostics, not isolated portable speed guarantees or a
same-condition competitor benchmark.

## Fresh application-cache calls

Two surveys use four scenarios and 20 alternating parent/current pairs, without
warmup. Before each variant, the retained runtime is idle, its gray/blur/spatial/
AI component caches are cleared and image/scratch pools are cleared; buffer-pool
budget is toggled through zero to its original default. Runtime creation,
upload, CPU oracle, checking and destruction are outside timing. The timed
scope is the public resident call plus completion, including component creation.
The driver cache is not reset; this is not process-cold latency.

Both versions initialize four component cache sets. These are not compute
pipeline counts: the spatial cache still eagerly constructs resize, Sobel and
morphology programs. Skipping the identity dispatch removes its call setup but
does not eliminate compilation of the resize program. Neither compiler work
alone nor physical pipeline memory is measured.

Fresh-cache intervals use 20 paired logs, 10,000 draws and seed
2026101820+1000*survey+scenario_index, sorted draws 249/9749, with the same zero,
multiplicity and shared-host limits. Identity scenarios are small 17x19 (3/3),
VGA 640x480 (3/3), and separable 257x259 (9/9); the resize control is
640x480 to 320x240 (3/3).

| Survey | Scenario | Parent/current component sets | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | small_identity | 4/4 | 2.434900 | 2.322900 | 1.195073 | 1.022903-1.575746 | 0/0 | favorable |
| 1 | vga_identity | 4/4 | 3.106550 | 2.966500 | 1.059579 | 1.013282-1.110706 | 0/0 | favorable |
| 1 | separable_identity | 4/4 | 4.272650 | 3.961600 | 1.067425 | 1.012021-1.128781 | 0/0 | favorable |
| 1 | resize_control | 4/4 | 2.905900 | 2.617150 | 1.032078 | 0.951179-1.108521 | 0/0 | inconclusive |
| 2 | small_identity | 4/4 | 3.223500 | 2.863700 | 1.250186 | 1.056026-1.689113 | 0/0 | favorable |
| 2 | vga_identity | 4/4 | 3.759850 | 3.931700 | 1.033176 | 0.979388-1.097573 | 0/0 | inconclusive |
| 2 | separable_identity | 4/4 | 4.778450 | 4.781100 | 1.021244 | 0.972473-1.078060 | 0/0 | inconclusive |
| 2 | resize_control | 4/4 | 3.181900 | 2.875200 | 1.044411 | 0.980058-1.108487 | 0/0 | inconclusive |

Identity fresh-cache results: 4 favorable / 2 inconclusive / 0 adverse / 0 unresolved. Resize controls: 0 favorable / 2 inconclusive / 0 adverse / 0 unresolved.

## Boundary diagnostics

Two-second observations publish busy process counts and CPU-core use only.
GPU/controller samples are utilization percentages, not allocated memory.
Both warm and both fresh-cache surveys fail the combined observed-quiet check;
these boundary samples do not cover every timed pair.

| Phase | Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Quiet |
| --- | --- | --- | ---: | --- | --- | --- |
| warm | 1 | pre | 0 | none | [[[0, 0]], [[0, 0]]] | True |
| warm | 1 | post | 2 | 0.360, 1.000 | [[[0, 0]], [[0, 0]]] | False |
| warm | 2 | pre | 1 | 0.994 | [[[0, 0]], [[0, 0]]] | False |
| warm | 2 | post | 2 | 1.012, 1.012 | [[[0, 0]], [[0, 0]]] | False |
| fresh-cache | 1 | pre | 1 | 0.485 | [[[0, 0]], [[0, 0]]] | False |
| fresh-cache | 1 | post | 1 | 0.375 | [[[2, 0]], [[0, 0]]] | False |
| fresh-cache | 2 | pre | 1 | 0.264 | [[[0, 0]], [[0, 0]]] | False |
| fresh-cache | 2 | post | 1 | 0.280 | [[[2, 0]], [[0, 0]]] | False |

Environment warm/1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced gpu_order=alternating gpu_scope=sum_of_stages checks_drop_recycle_excluded=true`.

Environment warm/2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced gpu_order=alternating gpu_scope=sum_of_stages checks_drop_recycle_excluded=true`.

Environment fresh-cache/1: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`.

Environment fresh-cache/2: `backend=Vulkan device_type=DiscreteGpu pairs=20 warmups=0 driver_cache_reset=false runtime_creation_and_upload_excluded=true checks_drop_recycle_excluded=true`.

## All warm comparisons

All 90 comparisons follow. Source/output dimensions, kernel sizes and the
constant-input flag identify each fixture. Medians and intervals are in ms and
parent/current ratios respectively; medians are not paired-ratio estimates.

| Survey | Source -> output | Blur/morphology | Constant | Identity | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 17x19 -> 17x19 | 3/3 | False | True | gpu | 0.023264 | 0.019680 | 1.184947 | 1.162538-1.208230 | 0/0 | favorable |
| 1 | 17x19 -> 17x19 | 3/3 | False | True | resident | 0.643650 | 0.575650 | 1.140410 | 1.061758-1.215296 | 0/0 | favorable |
| 1 | 17x19 -> 17x19 | 3/3 | False | True | full | 1.547600 | 1.482350 | 1.072226 | 1.034305-1.121100 | 0/0 | favorable |
| 1 | 257x259 -> 257x259 | 3/3 | False | True | gpu | 0.047936 | 0.042800 | 1.123465 | 1.116782-1.129725 | 0/0 | favorable |
| 1 | 257x259 -> 257x259 | 3/3 | False | True | resident | 0.714350 | 0.607250 | 1.138369 | 1.051027-1.233055 | 0/0 | favorable |
| 1 | 257x259 -> 257x259 | 3/3 | False | True | full | 1.764500 | 1.629100 | 1.047425 | 0.980102-1.109266 | 0/0 | inconclusive |
| 1 | 640x480 -> 640x480 | 3/3 | False | True | gpu | 0.154432 | 0.137840 | 1.121005 | 1.116523-1.125382 | 0/0 | favorable |
| 1 | 640x480 -> 640x480 | 3/3 | False | True | resident | 0.646600 | 0.576150 | 1.050835 | 0.975152-1.133203 | 0/0 | inconclusive |
| 1 | 640x480 -> 640x480 | 3/3 | False | True | full | 2.430850 | 2.419750 | 0.995718 | 0.940598-1.047561 | 0/0 | inconclusive |
| 1 | 1920x1080 -> 1920x1080 | 3/3 | False | True | gpu | 2.564800 | 2.172160 | 1.178822 | 1.176068-1.181614 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 3/3 | False | True | resident | 2.368500 | 2.071800 | 1.115705 | 1.091825-1.138831 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 3/3 | False | True | full | 10.301600 | 10.071550 | 1.030851 | 1.007811-1.052851 | 0/0 | favorable |
| 1 | 3840x2160 -> 3840x2160 | 3/3 | False | True | gpu | 4.739552 | 4.282272 | 1.126690 | 1.026608-1.239676 | 0/0 | favorable |
| 1 | 3840x2160 -> 3840x2160 | 3/3 | False | True | resident | 15.619350 | 11.303700 | 1.361549 | 1.319090-1.406098 | 0/0 | favorable |
| 1 | 3840x2160 -> 3840x2160 | 3/3 | False | True | full | 37.218400 | 35.287900 | 1.038183 | 1.006238-1.066811 | 0/0 | favorable |
| 1 | 640x480 -> 640x480 | 9/9 | False | True | gpu | 0.720320 | 0.653040 | 1.133858 | 1.093794-1.210105 | 0/0 | favorable |
| 1 | 640x480 -> 640x480 | 9/9 | False | True | resident | 1.259100 | 1.060100 | 1.142827 | 1.005840-1.283037 | 0/0 | favorable |
| 1 | 640x480 -> 640x480 | 9/9 | False | True | full | 8.659950 | 8.154150 | 1.060584 | 1.001215-1.122052 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 9/9 | False | True | gpu | 3.598768 | 3.185680 | 1.129631 | 1.127481-1.131711 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 9/9 | False | True | resident | 4.841450 | 4.463050 | 1.077689 | 1.048208-1.101474 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 9/9 | False | True | full | 13.493600 | 12.841950 | 1.065305 | 1.030202-1.122289 | 0/0 | favorable |
| 1 | 257x259 -> 257x259 | 4099/31 | False | True | gpu | 0.734400 | 0.721216 | 1.031822 | 1.016412-1.047792 | 0/0 | favorable |
| 1 | 257x259 -> 257x259 | 4099/31 | False | True | resident | 1.808700 | 1.661250 | 1.065938 | 0.992124-1.149712 | 0/0 | inconclusive |
| 1 | 257x259 -> 257x259 | 4099/31 | False | True | full | 7.364850 | 7.065750 | 1.053668 | 1.014165-1.095398 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 3/3 | True | True | gpu | 4.803072 | 4.129936 | 1.160097 | 1.151612-1.168330 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 3/3 | True | True | resident | 3.699600 | 3.424750 | 1.089727 | 1.078927-1.100807 | 0/0 | favorable |
| 1 | 1920x1080 -> 1920x1080 | 3/3 | True | True | full | 12.619150 | 12.358100 | 1.015803 | 0.981938-1.050604 | 0/0 | inconclusive |
| 1 | 17x19 -> 18x19 | 3/3 | False | False | gpu | 0.069920 | 0.066128 | 1.016926 | 0.990062-1.045897 | 0/0 | inconclusive |
| 1 | 17x19 -> 18x19 | 3/3 | False | False | resident | 1.140750 | 1.127850 | 1.001413 | 0.922509-1.085106 | 0/0 | inconclusive |
| 1 | 17x19 -> 18x19 | 3/3 | False | False | full | 7.116900 | 6.900300 | 1.022357 | 0.984787-1.061302 | 0/0 | inconclusive |
| 1 | 640x480 -> 320x240 | 3/3 | False | False | gpu | 0.305248 | 0.268144 | 1.054824 | 1.017259-1.094703 | 0/0 | favorable |
| 1 | 640x480 -> 320x240 | 3/3 | False | False | resident | 1.557700 | 1.628150 | 0.912404 | 0.809956-1.011166 | 0/0 | inconclusive |
| 1 | 640x480 -> 320x240 | 3/3 | False | False | full | 8.027700 | 7.815050 | 0.972506 | 0.938710-1.008104 | 0/0 | inconclusive |
| 1 | 640x480 -> 480x640 | 3/3 | False | False | gpu | 0.899552 | 0.885392 | 1.005863 | 0.992862-1.018288 | 0/0 | inconclusive |
| 1 | 640x480 -> 480x640 | 3/3 | False | False | resident | 2.299950 | 2.266450 | 1.002232 | 0.950889-1.058974 | 0/0 | inconclusive |
| 1 | 640x480 -> 480x640 | 3/3 | False | False | full | 8.786000 | 8.574100 | 1.008891 | 0.938427-1.069408 | 0/0 | inconclusive |
| 1 | 320x240 -> 640x480 | 9/9 | False | False | gpu | 1.184288 | 1.186848 | 1.000128 | 0.989519-1.011218 | 0/0 | inconclusive |
| 1 | 320x240 -> 640x480 | 9/9 | False | False | resident | 2.629750 | 2.662550 | 0.990401 | 0.968382-1.013018 | 0/0 | inconclusive |
| 1 | 320x240 -> 640x480 | 9/9 | False | False | full | 8.080450 | 7.818450 | 0.996054 | 0.937591-1.057079 | 0/0 | inconclusive |
| 1 | 1920x1080 -> 960x540 | 3/3 | False | False | gpu | 1.605728 | 1.607568 | 0.998965 | 0.987863-1.010421 | 0/0 | inconclusive |
| 1 | 1920x1080 -> 960x540 | 3/3 | False | False | resident | 3.191850 | 3.181650 | 1.007743 | 0.989861-1.026512 | 0/0 | inconclusive |
| 1 | 1920x1080 -> 960x540 | 3/3 | False | False | full | 11.108200 | 11.161250 | 0.998266 | 0.981300-1.014484 | 0/0 | inconclusive |
| 1 | 3840x2160 -> 1920x1080 | 3/3 | False | False | gpu | 2.740896 | 2.741280 | 0.999882 | 0.997812-1.001668 | 0/0 | inconclusive |
| 1 | 3840x2160 -> 1920x1080 | 3/3 | False | False | resident | 3.885500 | 3.864950 | 0.997999 | 0.989829-1.005585 | 0/0 | inconclusive |
| 1 | 3840x2160 -> 1920x1080 | 3/3 | False | False | full | 22.543800 | 22.474050 | 0.999928 | 0.983230-1.016428 | 0/0 | inconclusive |
| 2 | 17x19 -> 17x19 | 3/3 | False | True | gpu | 0.023344 | 0.020016 | 1.169500 | 1.151080-1.187939 | 0/0 | favorable |
| 2 | 17x19 -> 17x19 | 3/3 | False | True | resident | 0.719600 | 0.602050 | 1.106453 | 1.009154-1.209847 | 0/0 | favorable |
| 2 | 17x19 -> 17x19 | 3/3 | False | True | full | 1.723550 | 1.597250 | 1.050177 | 0.992706-1.098397 | 0/0 | inconclusive |
| 2 | 257x259 -> 257x259 | 3/3 | False | True | gpu | 0.048320 | 0.042416 | 1.130534 | 1.122788-1.138238 | 0/0 | favorable |
| 2 | 257x259 -> 257x259 | 3/3 | False | True | resident | 0.711450 | 0.592350 | 1.292641 | 1.218239-1.382545 | 0/0 | favorable |
| 2 | 257x259 -> 257x259 | 3/3 | False | True | full | 1.808600 | 1.727800 | 1.054314 | 1.007002-1.090291 | 0/0 | favorable |
| 2 | 640x480 -> 640x480 | 3/3 | False | True | gpu | 0.154288 | 0.137664 | 1.120798 | 1.117359-1.123986 | 0/0 | favorable |
| 2 | 640x480 -> 640x480 | 3/3 | False | True | resident | 0.866550 | 0.746750 | 1.162138 | 1.079732-1.247453 | 0/0 | favorable |
| 2 | 640x480 -> 640x480 | 3/3 | False | True | full | 3.258250 | 3.105350 | 1.027526 | 0.978295-1.078846 | 0/0 | inconclusive |
| 2 | 1920x1080 -> 1920x1080 | 3/3 | False | True | gpu | 2.552016 | 2.169152 | 1.177091 | 1.172944-1.181886 | 0/0 | favorable |
| 2 | 1920x1080 -> 1920x1080 | 3/3 | False | True | resident | 1.485000 | 1.359350 | 1.101955 | 1.080847-1.121874 | 0/0 | favorable |
| 2 | 1920x1080 -> 1920x1080 | 3/3 | False | True | full | 8.742350 | 8.803700 | 0.980967 | 0.942896-1.011556 | 0/0 | inconclusive |
| 2 | 3840x2160 -> 3840x2160 | 3/3 | False | True | gpu | 5.066560 | 4.621680 | 1.080600 | 1.012026-1.123454 | 0/0 | favorable |
| 2 | 3840x2160 -> 3840x2160 | 3/3 | False | True | resident | 15.738600 | 11.571550 | 1.350656 | 1.267527-1.425471 | 0/0 | favorable |
| 2 | 3840x2160 -> 3840x2160 | 3/3 | False | True | full | 37.251550 | 35.759600 | 1.057736 | 1.028791-1.094525 | 0/0 | favorable |
| 2 | 640x480 -> 640x480 | 9/9 | False | True | gpu | 0.638096 | 0.563472 | 1.118491 | 1.108599-1.128538 | 0/0 | favorable |
| 2 | 640x480 -> 640x480 | 9/9 | False | True | resident | 1.176600 | 1.066800 | 1.095915 | 0.993138-1.188829 | 0/0 | inconclusive |
| 2 | 640x480 -> 640x480 | 9/9 | False | True | full | 7.951050 | 7.799950 | 1.023906 | 0.986525-1.061543 | 0/0 | inconclusive |
| 2 | 1920x1080 -> 1920x1080 | 9/9 | False | True | gpu | 3.715136 | 3.316768 | 1.125913 | 1.121636-1.132218 | 0/0 | favorable |
| 2 | 1920x1080 -> 1920x1080 | 9/9 | False | True | resident | 4.779700 | 4.359350 | 1.065105 | 1.005083-1.103734 | 0/0 | favorable |
| 2 | 1920x1080 -> 1920x1080 | 9/9 | False | True | full | 13.742450 | 13.182200 | 1.020035 | 0.975302-1.057258 | 0/0 | inconclusive |
| 2 | 257x259 -> 257x259 | 4099/31 | False | True | gpu | 0.737856 | 0.708608 | 1.024128 | 1.008529-1.039301 | 0/0 | favorable |
| 2 | 257x259 -> 257x259 | 4099/31 | False | True | resident | 1.865500 | 1.672750 | 1.096410 | 1.027170-1.178079 | 0/0 | favorable |
| 2 | 257x259 -> 257x259 | 4099/31 | False | True | full | 7.789850 | 7.718100 | 1.028918 | 0.990067-1.067985 | 0/0 | inconclusive |
| 2 | 1920x1080 -> 1920x1080 | 3/3 | True | True | gpu | 4.860928 | 4.180144 | 1.161648 | 1.155761-1.166763 | 0/0 | favorable |
| 2 | 1920x1080 -> 1920x1080 | 3/3 | True | True | resident | 3.750300 | 3.464000 | 1.095402 | 1.084237-1.106588 | 0/0 | favorable |
| 2 | 1920x1080 -> 1920x1080 | 3/3 | True | True | full | 15.341850 | 14.561100 | 1.029956 | 0.991624-1.071780 | 0/0 | inconclusive |
| 2 | 17x19 -> 18x19 | 3/3 | False | False | gpu | 0.071760 | 0.070944 | 0.995409 | 0.971051-1.020947 | 0/0 | inconclusive |
| 2 | 17x19 -> 18x19 | 3/3 | False | False | resident | 1.266300 | 1.236000 | 1.044585 | 0.939403-1.158748 | 0/0 | inconclusive |
| 2 | 17x19 -> 18x19 | 3/3 | False | False | full | 6.933150 | 7.239150 | 0.991262 | 0.944564-1.039372 | 0/0 | inconclusive |
| 2 | 640x480 -> 320x240 | 3/3 | False | False | gpu | 0.270048 | 0.267472 | 1.009670 | 0.983027-1.036283 | 0/0 | inconclusive |
| 2 | 640x480 -> 320x240 | 3/3 | False | False | resident | 1.584650 | 1.436850 | 1.057639 | 0.971731-1.139589 | 0/0 | inconclusive |
| 2 | 640x480 -> 320x240 | 3/3 | False | False | full | 8.873900 | 8.648300 | 0.991083 | 0.951369-1.031259 | 0/0 | inconclusive |
| 2 | 640x480 -> 480x640 | 3/3 | False | False | gpu | 0.884768 | 0.884848 | 0.997239 | 0.976072-1.018600 | 0/0 | inconclusive |
| 2 | 640x480 -> 480x640 | 3/3 | False | False | resident | 2.267450 | 2.307350 | 0.982940 | 0.922952-1.039487 | 0/0 | inconclusive |
| 2 | 640x480 -> 480x640 | 3/3 | False | False | full | 8.660600 | 8.958800 | 0.985634 | 0.939210-1.040177 | 0/0 | inconclusive |
| 2 | 320x240 -> 640x480 | 9/9 | False | False | gpu | 1.163648 | 1.163312 | 0.994237 | 0.980319-1.008306 | 0/0 | inconclusive |
| 2 | 320x240 -> 640x480 | 9/9 | False | False | resident | 2.771100 | 2.745500 | 1.016006 | 0.993886-1.039074 | 0/0 | inconclusive |
| 2 | 320x240 -> 640x480 | 9/9 | False | False | full | 8.523300 | 8.483800 | 1.000424 | 0.952528-1.047168 | 0/0 | inconclusive |
| 2 | 1920x1080 -> 960x540 | 3/3 | False | False | gpu | 1.650512 | 1.653264 | 1.001921 | 0.986596-1.017341 | 0/0 | inconclusive |
| 2 | 1920x1080 -> 960x540 | 3/3 | False | False | resident | 3.212250 | 3.225900 | 0.996927 | 0.980390-1.013959 | 0/0 | inconclusive |
| 2 | 1920x1080 -> 960x540 | 3/3 | False | False | full | 11.478400 | 11.491000 | 1.005242 | 0.971464-1.051115 | 0/0 | inconclusive |
| 2 | 3840x2160 -> 1920x1080 | 3/3 | False | False | gpu | 2.778384 | 2.778512 | 0.999462 | 0.997599-1.001305 | 0/0 | inconclusive |
| 2 | 3840x2160 -> 1920x1080 | 3/3 | False | False | resident | 3.937200 | 3.917950 | 1.002814 | 0.990330-1.013411 | 0/0 | inconclusive |
| 2 | 3840x2160 -> 1920x1080 | 3/3 | False | False | full | 23.617250 | 23.507500 | 1.011106 | 0.995734-1.027533 | 0/0 | inconclusive |

## Source and local evidence locations

Source paths for this workspace:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/vision_chain.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/spatial.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/vision_chain_identity_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/vision_chain_identity_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/vision_chain_identity_benchmark.rs`

Ignored local evidence is under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`: the `identity-chain-*`
precheck, warm/fresh-cache raw logs and JSON samples, diagnostics, validation,
source/executable provenance and delivery receipts. These artifacts are not
required at runtime. Raw process identities stay local; all published summaries
above avoid adapter model names. This change does not establish a new competitor
maturity ranking or a release gate on other devices.
