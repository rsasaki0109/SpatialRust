# Constant coordinate digits in CPU-key GPU voxel sorting

Date: 2026-10-18. Named slice: constant CPU voxel radix digits.
Immediate parent: `c4ce95911c049adcce0dba0a0389a97786d633e4` (merged PR #277).

## Change and proof

CPU-key voxel construction still prepares increasing source indices and clamps
coordinates to i32 in the original host pass. It now summarizes which four-bit
coordinate digits differ from the first prepared entry. A digit that is constant
across all entries produces an identity permutation in a stable radix pass and
can be omitted. XOR with the common signed-order sign mask preserves differences:
`(x XOR sign) XOR (first XOR sign) = x XOR first`. All varying digits retain the
original z/y/x, least-significant-digit order, preserving signed lexicographic
keys and original source-index order for ties.

If the number of varying digits is odd, the lowest omitted constant digit is
retained as an identity pass. This makes the selected count even (0, 2, ..., 24),
so ping-pong results finish in the original owned uploaded allocation. No final
copy or new output allocation is introduced. Constant keys run no radix work;
the existing compactor still creates segments and synchronizes its scalar.

The first 256 entries form a bounded probe. If they already establish all 24
varying digits, later entries cannot remove that variation and no full scan or
CPU-count query is needed. Otherwise the complete prepared input is summarized.
At least 524,288 native points use at most four workers, including the caller,
bounded by CPU availability and 131,072 points per useful worker. Worker-local
three-word OR summaries combine against the same global first entry. There is
no per-point planner vector. Small, one-worker and WASM calls scan serially.
This is a second CPU phase after unchanged preparation, with a per-call cap,
not a global thread budget; host scan/startup costs are measured below.

## Contracts and resource accounting

Empty/count/device admission still precedes preparation and planning, retaining
the original full-sort limits even if fewer digits would suffice. Public
signatures, safe Rust, feature/dependency boundaries and original `16*N` entry
upload bytes remain. Each nonempty construction still reads four cell-count
bytes. Explicit later metadata readback retains the same data and ordering.
No host/device transfer or automatic backend selection is added.

For k selected digits, radix work is `3*k` compute passes and k bind groups rather
than 72 passes and 24 groups. When the original batched-control layout fits the
device limits, k>0 still initializes all 24 aligned uniforms; its allocation/size
is not reduced. The existing limit/cap fallback creates individual uniforms for
selected digits. Nonzero sorts retain the same four scratch requests and sizes:
`16*N` ping-pong,
`4*N` ranks and two `64*ceil(N/256)`-byte histogram/offset buffers. At k=0 the sorter
returns before these requests, control creation or its encoder/submission.
Compaction resources and its submission remain. Actual timestamp collection is
`48*k` bytes for the sort alone and zero if there is no sort stage. This private
instrumentation does not add a public CPU-key profiling API or change public
payload receipts. Request/descriptor counts do not establish physical peak
memory or bandwidth; those are unmeasured.

The GPU-key buffer entry point retains all 24 digits, its original borrowed-key
first scatter, profiler preflight and frame contracts. It does not use CPU key
inspection. Original standalone all-digit sorting is retained. The shared host
loop now checks a mask, so identical scheduling is not a zero-overhead promise.
All WGSL, pipeline variants/layouts, original host preparation, radix controls,
scan and compaction code remain unchanged. No core or unsafe changes are made.

## Correctness and provenance

Four new native tests cover every axis/nibble/value including sign bits, exact
mask membership/even parity, stable CPU LSD simulation, probe/worker thresholds,
forced 0/1/2/3/4/12/maximum availability and a last-only varying point. Complete
public parent/current builds are compared with independent BTreeMap grouping
after signed clamp across 120 fixture pairs through 1,000,003 points. Raw sorted
entry words also match stable CPU sorting, and the result is the exact original
buffer handle. Tests cover constant, one-nibble, positive/negative bounded,
mixed-sign, full-width, saturated, noncontiguous-digit and late-change inputs.

Further checks verify exact pass/query groups, zero versus four radix scratch
requests, normal/tiny/zero budgets, unchanged device-key passes and immutable
inputs, original admission errors, and concurrent results after host mutation,
pool clear and changed-input reuse. Complete immediate-parent radix, controls
and CPU/device-key entry-point functions are frozen; only fixture paths/imports
change. Shared unchanged production dependencies are audited separately.

VOXEL_DIGITS_PLAN: verified 804 masks/LSD orders, every axis/nibble/value, sign bits, 256-point probe and worker boundaries, late-only changes, saturation and even owned-output parity.

VOXEL_DIGITS_OUTPUT: verified 212092340 raw parent/current/CPU stable words across 120 complete fixture pairs; all clamped coordinates, signs, constant/sparse/full digits, workgroup/prefix/probe boundaries and original owned output allocation.

VOXEL_DIGITS_RESOURCES: verified exact three-pass/query groups per selected digit, zero radix requests for constant keys, four scratch requests otherwise, default/tiny/zero budgets, untouched GPU-key passes/input and original admission errors.

VOXEL_DIGITS_CONCURRENT: verified simultaneous zero/sparse/full/late-variable calls, immutable shared inputs, independently owned segments after host mutation, pool clears and changed-input reuse.

There are 608 unchanged original GPU/image sources and 622 current
source/manifest/executable SHA-256 values, including the ignored lockfile. Hashes agree before measurement and after both warm/cold
surveys and broad validation. Repository/PR adapter-name matches are zero;
process names/PIDs stay in ignored local diagnostics.

## Validation

| Check | Passed |
| --- | ---: |
| image | 182 |
| voxel | 232 |
| frames | 5 |
| ransac | 8 |
| segmentation | 8 |
| filter | 18 |
| facade | 2 |
| facade-gaussian | 1 |
| Total native test executions | 456 |

All-features/all-targets Clippy with warnings denied, CPU-only, minimal and
CI-feature WASM checks with warnings denied, formatting and diff checks pass.
Native suites include ignored correctness cases and skip explicit benchmarks.

## Measurement method

```text
cargo test -p spatialrust-gpu --all-features --release --lib kernels::voxel_sort::digits::tests::benchmark::benchmark_voxel_constant_digits -- --ignored --exact --nocapture --test-threads=1
cargo test -p spatialrust-gpu --all-features --release --lib kernels::voxel_sort::digits::tests::benchmark::benchmark_voxel_constant_digits_cold -- --ignored --exact --nocapture --test-threads=1
```

Two consecutive warm surveys use the same release executable, 19 cases, three
warmup pairs and 30 measured pairs per route. Complete-build/full order rotates
through four positions with offset XOR (round modulo four); GPU/host pair order
alternates. Results are checked after timing. Destruction is excluded; the full
route includes its required readback/host conversion. No workflow build/test
overlaps a timed survey. No case is retried or filtered.

- `gpu`: sum of actual radix-stage timestamps, excluding planning, upload,
  transfers, CPU work and inter-pass gaps. Constant-key current samples are zero
  because there are no radix stages; their ratios/intervals stay undefined.
- `build`: public CPU-key resident-segment constructor plus completion, including
  preparation, planning, upload, sorting, compaction and scalar synchronization;
  checking metadata readback is excluded.
- `full`: public CPU-key host-returning constructor, including the above and
  explicit metadata readback/host conversion.
- `host`: unchanged CPU entry preparation versus preparation plus digit planning.
  Equality checks and destruction are outside timing.

Ratios are paired geometric means of parent/current (>1 favors current). Each
95% interval uses 10,000 bootstrap resamples of 30 paired log ratios, seed
2026101821+1000*survey+10*case_index+route_index, one-based survey and zero-based
case/route indices (gpu, build, full, host), sorted draws 249/9749. Intervals above
or below one are favorable or adverse; otherwise inconclusive. Zeros remain
undefined without epsilon; there is no multiplicity correction. All 24-digit
controls retain their GPU schedule but share changed compiled host code. No
benefit is attributed to omitted digits in those controls. These shared-host
diagnostics do not establish portable non-regression/application guarantees or
a same-condition competitor benchmark.

| Route | Constant: 8 comparisons | Selected nonzero digits: 22 | All 24 retained: 8 |
| --- | --- | --- | --- |
| gpu | 0 favorable / 0 inconclusive / 0 adverse / 8 unresolved | 22 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 7 inconclusive / 1 adverse / 0 unresolved |
| build | 8 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 22 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 0 favorable / 8 inconclusive / 0 adverse / 0 unresolved |
| full | 8 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 22 favorable / 0 inconclusive / 0 adverse / 0 unresolved | 1 favorable / 7 inconclusive / 0 adverse / 0 unresolved |
| host | 0 favorable / 0 inconclusive / 8 adverse / 0 unresolved | 0 favorable / 0 inconclusive / 22 adverse / 0 unresolved | 0 favorable / 8 inconclusive / 0 adverse / 0 unresolved |

Complete resident-segment builds: target ratios 1.179045-2.022169, about 15.2-50.5% less elapsed time in these initialized-runtime conditions.

Full host-returning calls: target ratios 1.118822-1.578571, about 10.6-36.7% less elapsed time in these initialized-runtime conditions.

Host-only target comparisons are all 30 adverse: planning adds about 14.3-77.2% to original entry-preparation time. Complete-call savings above include that cost. All eight full-width host-control intervals are inconclusive.

An all-digit GPU control is adverse in survey 1, N=1000003, pattern=4: ratio 0.999296, interval 0.998727-0.999722, approximately 0.070% more stage time. All-digit build intervals are inconclusive; the one favorable full-call control is not attributed to digit omission.

Additional CPU planning costs must be assessed against complete builds rather
than inferred from GPU-only gains. All adverse and inconclusive rows remain.

The warm matrix covers selected counts 0/2/4/6/24. Counts 8-22 and broader mixed-sign/range combinations are not comprehensively timed; other devices are unmeasured. Empty/singleton behavior is correctness-covered without a latency claim.

## Fresh runtime/device calls

Two surveys use four scenarios and 20 alternating pairs with no warmup. Each
variant creates a fresh headless runtime/device outside timing. No explicit driver-cache reset is requested; this does not measure process-cold
startup. The timed scope
is the public resident-segment constructor plus completion, including original
preparation, planning, upload and first-use pipelines. Runtime creation, checking
metadata readback and destruction are excluded. Pipeline compilation alone and
physical cache memory are not measured. Intervals use 20 paired logs, 10,000
draws, seed 2026101822+1000*survey+scenario_index and sorted draws 249/9749, with
the same zero/multiplicity/shared-host limits.

| Survey | N | Pattern | Selected digits | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 257 | 0 | 0 | 21.667550 | 18.602550 | 1.068649 | 0.961057-1.182393 | 0/0 | inconclusive |
| 1 | 65537 | 2 | 6 | 21.608300 | 20.841700 | 1.065189 | 0.984358-1.177721 | 0/0 | inconclusive |
| 1 | 65537 | 5 | 24 | 22.700050 | 24.896350 | 0.960559 | 0.881296-1.045545 | 0/0 | inconclusive |
| 1 | 524289 | 2 | 6 | 28.726500 | 28.144300 | 1.000773 | 0.901505-1.110312 | 0/0 | inconclusive |
| 2 | 257 | 0 | 0 | 21.004200 | 18.934950 | 1.111232 | 1.020220-1.192759 | 0/0 | favorable |
| 2 | 65537 | 2 | 6 | 23.193700 | 23.288600 | 0.968334 | 0.878933-1.062957 | 0/0 | inconclusive |
| 2 | 65537 | 5 | 24 | 22.440800 | 23.185200 | 1.000556 | 0.936141-1.076174 | 0/0 | inconclusive |
| 2 | 524289 | 2 | 6 | 29.855600 | 31.800800 | 1.002216 | 0.918847-1.097623 | 0/0 | inconclusive |

Fresh-runtime selected/constant results: 1 favorable / 5 inconclusive / 0 adverse / 0 unresolved. All-digit controls: 0 favorable / 2 inconclusive / 0 adverse / 0 unresolved.

## Boundary diagnostics

Two-second observations publish busy process counts and CPU-core use only.
GPU/controller values are utilization percentages, not allocated memory. These
boundary observations do not cover every timed pair or establish host isolation.

| Phase | Survey | Boundary | Busy processes | CPU cores per busy process | GPU/controller samples | Quiet |
| --- | --- | --- | ---: | --- | --- | --- |
| warm | 1 | pre | 2 | 0.583, 1.417 | [[[0, 0]], [[0, 0]]] | False |
| warm | 1 | post | 3 | 0.269, 0.591, 2.135 | [[[0, 0]], [[0, 0]]] | False |
| warm | 2 | pre | 1 | 0.713 | [[[0, 0]], [[0, 0]]] | False |
| warm | 2 | post | 0 | none | [[[0, 0]], [[0, 0]]] | True |
| fresh-runtime | 1 | pre | 1 | 0.419 | [[[0, 0]], [[0, 0]]] | False |
| fresh-runtime | 1 | post | 1 | 0.512 | [[[2, 1]], [[0, 0]]] | False |
| fresh-runtime | 2 | pre | 1 | 0.644 | [[[0, 0]], [[0, 0]]] | False |
| fresh-runtime | 2 | post | 2 | 0.301, 1.000 | [[[2, 1]], [[0, 0]]] | False |

Environment warm/1: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=30 warmups=3 wall_order=balanced gpu_host_order=alternating checks_drop_excluded=true`.

Environment warm/2: `backend=Vulkan device_type=DiscreteGpu available_workers=12 pairs=30 warmups=3 wall_order=balanced gpu_host_order=alternating checks_drop_excluded=true`.

Environment fresh-runtime/1: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`.

Environment fresh-runtime/2: `pairs=20 warmups=0 fresh_runtime_per_variant=true driver_cache_reset=false runtime_creation_checks_drop_excluded=true`.

## All warm comparisons

Pattern IDs: 0 constant; 1 one varying nibble; 2 bounded positive coordinates;
3 bounded negative coordinates; 4 mixed signs; 5 full-width random; 6 distinct
i64 keys clamping to one cell; 8 noncontiguous varying nibbles; 9 a last-only
change after a constant probe. These identify every measured fixture exactly
with the checked-in generator. Pattern 7 signed/extreme data is correctness-only.
Medians are in ms; paired ratios are not ratios of medians.

| Survey | N | Pattern | Selected digits | Route | Parent median ms | Current median ms | Ratio | 95% interval | Zero counts | Class |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 257 | 0 | 0 | gpu | 0.285312 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 1 | 257 | 0 | 0 | build | 1.639100 | 0.980850 | 1.676819 | 1.643658-1.711470 | 0/0 | favorable |
| 1 | 257 | 0 | 0 | full | 2.510500 | 1.816350 | 1.387774 | 1.359287-1.417508 | 0/0 | favorable |
| 1 | 257 | 0 | 0 | host | 0.000900 | 0.001350 | 0.656345 | 0.516879-0.833303 | 0/0 | adverse |
| 1 | 257 | 1 | 2 | gpu | 0.285536 | 0.024576 | 11.753449 | 11.601906-11.919558 | 0/0 | favorable |
| 1 | 257 | 1 | 2 | build | 1.698000 | 1.135900 | 1.514113 | 1.478209-1.550068 | 0/0 | favorable |
| 1 | 257 | 1 | 2 | full | 2.618550 | 1.977600 | 1.303763 | 1.264361-1.345117 | 0/0 | favorable |
| 1 | 257 | 1 | 2 | host | 0.000900 | 0.001350 | 0.639000 | 0.510461-0.799590 | 0/0 | adverse |
| 1 | 257 | 2 | 6 | gpu | 0.286368 | 0.076320 | 3.728733 | 3.692069-3.764917 | 0/0 | favorable |
| 1 | 257 | 2 | 6 | build | 1.636650 | 1.195350 | 1.380258 | 1.353595-1.406059 | 0/0 | favorable |
| 1 | 257 | 2 | 6 | full | 2.512200 | 2.062000 | 1.233529 | 1.203110-1.271412 | 0/0 | favorable |
| 1 | 257 | 2 | 6 | host | 0.001000 | 0.001400 | 0.655419 | 0.537605-0.798675 | 0/0 | adverse |
| 1 | 257 | 5 | 24 | gpu | 0.290784 | 0.290720 | 1.001498 | 0.995563-1.007495 | 0/0 | inconclusive |
| 1 | 257 | 5 | 24 | build | 1.632500 | 1.638500 | 1.006408 | 0.991553-1.022423 | 0/0 | inconclusive |
| 1 | 257 | 5 | 24 | full | 2.504250 | 2.465050 | 1.024875 | 0.995159-1.067679 | 0/0 | inconclusive |
| 1 | 257 | 5 | 24 | host | 0.001000 | 0.001050 | 0.886204 | 0.684198-1.149886 | 0/0 | inconclusive |
| 1 | 257 | 9 | 2 | gpu | 0.284016 | 0.023968 | 11.998768 | 11.843665-12.182861 | 0/0 | favorable |
| 1 | 257 | 9 | 2 | build | 1.679850 | 1.077700 | 1.523814 | 1.493272-1.554041 | 0/0 | favorable |
| 1 | 257 | 9 | 2 | full | 2.516050 | 1.907450 | 1.309837 | 1.288525-1.331086 | 0/0 | favorable |
| 1 | 257 | 9 | 2 | host | 0.001100 | 0.001600 | 0.656007 | 0.514441-0.836422 | 0/0 | adverse |
| 1 | 65537 | 0 | 0 | gpu | 1.782608 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 1 | 65537 | 0 | 0 | build | 5.026800 | 2.429600 | 2.022169 | 1.945939-2.100279 | 0/0 | favorable |
| 1 | 65537 | 0 | 0 | full | 6.275250 | 3.938300 | 1.578571 | 1.509295-1.635063 | 0/0 | favorable |
| 1 | 65537 | 0 | 0 | host | 0.348750 | 0.399550 | 0.862593 | 0.802824-0.925698 | 0/0 | adverse |
| 1 | 65537 | 2 | 6 | gpu | 1.905744 | 0.555040 | 3.422921 | 3.393675-3.451300 | 0/0 | favorable |
| 1 | 65537 | 2 | 6 | build | 4.980950 | 2.600500 | 1.882308 | 1.772663-1.981404 | 0/0 | favorable |
| 1 | 65537 | 2 | 6 | full | 6.436500 | 4.283600 | 1.542837 | 1.442288-1.663881 | 0/0 | favorable |
| 1 | 65537 | 2 | 6 | host | 0.317100 | 0.372450 | 0.874619 | 0.820449-0.936551 | 0/0 | adverse |
| 1 | 65537 | 3 | 6 | gpu | 0.835328 | 0.242576 | 3.436522 | 3.420905-3.452295 | 0/0 | favorable |
| 1 | 65537 | 3 | 6 | build | 4.933200 | 2.562200 | 1.749022 | 1.597313-1.887483 | 0/0 | favorable |
| 1 | 65537 | 3 | 6 | full | 6.398200 | 4.059900 | 1.448604 | 1.354480-1.542758 | 0/0 | favorable |
| 1 | 65537 | 3 | 6 | host | 0.316800 | 0.378600 | 0.852414 | 0.800277-0.908812 | 0/0 | adverse |
| 1 | 65537 | 5 | 24 | gpu | 1.239984 | 1.239344 | 1.000567 | 0.998486-1.002636 | 0/0 | inconclusive |
| 1 | 65537 | 5 | 24 | build | 3.415550 | 3.452250 | 0.998649 | 0.968460-1.027735 | 0/0 | inconclusive |
| 1 | 65537 | 5 | 24 | full | 5.638650 | 5.588600 | 1.003602 | 0.971095-1.036759 | 0/0 | inconclusive |
| 1 | 65537 | 5 | 24 | host | 0.321450 | 0.321550 | 1.006133 | 0.937423-1.081124 | 0/0 | inconclusive |
| 1 | 65537 | 8 | 4 | gpu | 1.057568 | 0.239888 | 4.407515 | 4.388336-4.425762 | 0/0 | favorable |
| 1 | 65537 | 8 | 4 | build | 3.075900 | 2.084800 | 1.439335 | 1.404542-1.473927 | 0/0 | favorable |
| 1 | 65537 | 8 | 4 | full | 4.512300 | 3.528150 | 1.285827 | 1.251440-1.321995 | 0/0 | favorable |
| 1 | 65537 | 8 | 4 | host | 0.334300 | 0.377150 | 0.859330 | 0.804563-0.919555 | 0/0 | adverse |
| 1 | 524287 | 2 | 6 | gpu | 5.979936 | 1.695744 | 3.531635 | 3.527514-3.535484 | 0/0 | favorable |
| 1 | 524287 | 2 | 6 | build | 13.156050 | 8.110950 | 1.565490 | 1.513054-1.609635 | 0/0 | favorable |
| 1 | 524287 | 2 | 6 | full | 16.049850 | 11.056250 | 1.459253 | 1.430045-1.489805 | 0/0 | favorable |
| 1 | 524287 | 2 | 6 | host | 2.284250 | 2.795450 | 0.821018 | 0.798410-0.844804 | 0/0 | adverse |
| 1 | 524288 | 2 | 6 | gpu | 5.710752 | 1.577040 | 3.619602 | 3.615610-3.623359 | 0/0 | favorable |
| 1 | 524288 | 2 | 6 | build | 12.518300 | 7.453500 | 1.676610 | 1.648377-1.704857 | 0/0 | favorable |
| 1 | 524288 | 2 | 6 | full | 15.867300 | 10.504650 | 1.493162 | 1.468334-1.516574 | 0/0 | favorable |
| 1 | 524288 | 2 | 6 | host | 1.445050 | 2.121850 | 0.681693 | 0.664519-0.698689 | 0/0 | adverse |
| 1 | 524289 | 2 | 6 | gpu | 5.359056 | 1.581712 | 3.382746 | 3.376984-3.388421 | 0/0 | favorable |
| 1 | 524289 | 2 | 6 | build | 12.026200 | 7.863650 | 1.512277 | 1.482787-1.542155 | 0/0 | favorable |
| 1 | 524289 | 2 | 6 | full | 15.537150 | 11.421500 | 1.363946 | 1.339802-1.391351 | 0/0 | favorable |
| 1 | 524289 | 2 | 6 | host | 1.475450 | 2.143800 | 0.681399 | 0.653620-0.705356 | 0/0 | adverse |
| 1 | 1000003 | 0 | 0 | gpu | 9.312096 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 1 | 1000003 | 0 | 0 | build | 20.045550 | 12.063800 | 1.651587 | 1.594338-1.709240 | 0/0 | favorable |
| 1 | 1000003 | 0 | 0 | full | 25.147300 | 17.571200 | 1.403097 | 1.302395-1.492722 | 0/0 | favorable |
| 1 | 1000003 | 0 | 0 | host | 2.409250 | 3.325350 | 0.667001 | 0.566628-0.742540 | 0/0 | adverse |
| 1 | 1000003 | 2 | 6 | gpu | 7.786768 | 2.292592 | 3.386636 | 3.358624-3.402671 | 0/0 | favorable |
| 1 | 1000003 | 2 | 6 | build | 18.367750 | 12.019550 | 1.494590 | 1.419081-1.549812 | 0/0 | favorable |
| 1 | 1000003 | 2 | 6 | full | 23.902250 | 17.714350 | 1.371427 | 1.336553-1.407228 | 0/0 | favorable |
| 1 | 1000003 | 2 | 6 | host | 2.373050 | 3.314900 | 0.743279 | 0.700030-0.817122 | 0/0 | adverse |
| 1 | 1000003 | 4 | 24 | gpu | 6.206320 | 6.208480 | 0.999296 | 0.998727-0.999722 | 0/0 | adverse |
| 1 | 1000003 | 4 | 24 | build | 15.261850 | 15.396650 | 0.999336 | 0.991379-1.007108 | 0/0 | inconclusive |
| 1 | 1000003 | 4 | 24 | full | 21.334450 | 21.032100 | 1.018351 | 1.005261-1.030932 | 0/0 | favorable |
| 1 | 1000003 | 4 | 24 | host | 2.429000 | 2.451400 | 0.994779 | 0.967418-1.022710 | 0/0 | inconclusive |
| 1 | 1000003 | 5 | 24 | gpu | 8.465776 | 8.467712 | 0.996968 | 0.989660-1.001935 | 0/0 | inconclusive |
| 1 | 1000003 | 5 | 24 | build | 20.607350 | 20.976150 | 0.978541 | 0.939241-1.018977 | 0/0 | inconclusive |
| 1 | 1000003 | 5 | 24 | full | 35.685000 | 35.638700 | 1.009847 | 0.973134-1.045937 | 0/0 | inconclusive |
| 1 | 1000003 | 5 | 24 | host | 2.380500 | 2.348500 | 1.006335 | 0.983198-1.030983 | 0/0 | inconclusive |
| 1 | 1000003 | 8 | 4 | gpu | 9.385072 | 2.275728 | 4.097254 | 4.074129-4.119398 | 0/0 | favorable |
| 1 | 1000003 | 8 | 4 | build | 16.966200 | 14.224700 | 1.179045 | 1.144451-1.215594 | 0/0 | favorable |
| 1 | 1000003 | 8 | 4 | full | 24.869050 | 22.020650 | 1.118822 | 1.086666-1.151373 | 0/0 | favorable |
| 1 | 1000003 | 8 | 4 | host | 2.785400 | 3.895750 | 0.707295 | 0.690753-0.724157 | 0/0 | adverse |
| 1 | 1000003 | 6 | 0 | gpu | 13.383856 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 1 | 1000003 | 6 | 0 | build | 21.724700 | 16.548750 | 1.304667 | 1.279794-1.329541 | 0/0 | favorable |
| 1 | 1000003 | 6 | 0 | full | 30.553700 | 25.885200 | 1.186580 | 1.159219-1.214142 | 0/0 | favorable |
| 1 | 1000003 | 6 | 0 | host | 2.775450 | 3.904400 | 0.714167 | 0.698124-0.730299 | 0/0 | adverse |
| 2 | 257 | 0 | 0 | gpu | 0.276800 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 2 | 257 | 0 | 0 | build | 2.098150 | 1.232250 | 1.710703 | 1.644534-1.782774 | 0/0 | favorable |
| 2 | 257 | 0 | 0 | full | 3.195950 | 2.351850 | 1.374493 | 1.311654-1.457288 | 0/0 | favorable |
| 2 | 257 | 0 | 0 | host | 0.001300 | 0.002100 | 0.610139 | 0.475238-0.786290 | 0/0 | adverse |
| 2 | 257 | 1 | 2 | gpu | 0.277824 | 0.024064 | 11.564898 | 11.418546-11.718501 | 0/0 | favorable |
| 2 | 257 | 1 | 2 | build | 2.060550 | 1.384900 | 1.502365 | 1.445737-1.562408 | 0/0 | favorable |
| 2 | 257 | 1 | 2 | full | 3.121350 | 2.522400 | 1.297289 | 1.251143-1.345513 | 0/0 | favorable |
| 2 | 257 | 1 | 2 | host | 0.001150 | 0.002150 | 0.577171 | 0.461432-0.719862 | 0/0 | adverse |
| 2 | 257 | 2 | 6 | gpu | 0.278752 | 0.073136 | 3.827554 | 3.800789-3.855182 | 0/0 | favorable |
| 2 | 257 | 2 | 6 | build | 2.005300 | 1.413050 | 1.379698 | 1.322778-1.439658 | 0/0 | favorable |
| 2 | 257 | 2 | 6 | full | 3.030600 | 2.434550 | 1.225512 | 1.180409-1.268315 | 0/0 | favorable |
| 2 | 257 | 2 | 6 | host | 0.001050 | 0.002050 | 0.564268 | 0.426523-0.735833 | 0/0 | adverse |
| 2 | 257 | 5 | 24 | gpu | 0.286592 | 0.284432 | 1.019032 | 0.994563-1.064775 | 0/0 | inconclusive |
| 2 | 257 | 5 | 24 | build | 2.078700 | 2.047250 | 0.990619 | 0.928978-1.056171 | 0/0 | inconclusive |
| 2 | 257 | 5 | 24 | full | 3.154400 | 3.175000 | 1.005569 | 0.971714-1.044798 | 0/0 | inconclusive |
| 2 | 257 | 5 | 24 | host | 0.001500 | 0.001800 | 0.889342 | 0.719026-1.107801 | 0/0 | inconclusive |
| 2 | 257 | 9 | 2 | gpu | 0.566432 | 0.049104 | 11.648137 | 11.514821-11.771289 | 0/0 | favorable |
| 2 | 257 | 9 | 2 | build | 2.770550 | 1.597750 | 1.670239 | 1.594761-1.745210 | 0/0 | favorable |
| 2 | 257 | 9 | 2 | full | 3.790000 | 2.731100 | 1.385074 | 1.335676-1.435014 | 0/0 | favorable |
| 2 | 257 | 9 | 2 | host | 0.001550 | 0.002350 | 0.624222 | 0.514522-0.757229 | 0/0 | adverse |
| 2 | 65537 | 0 | 0 | gpu | 2.001120 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 2 | 65537 | 0 | 0 | build | 5.830100 | 3.039250 | 1.903158 | 1.831680-1.973378 | 0/0 | favorable |
| 2 | 65537 | 0 | 0 | full | 7.407150 | 4.945350 | 1.546151 | 1.487375-1.599146 | 0/0 | favorable |
| 2 | 65537 | 0 | 0 | host | 0.440150 | 0.521900 | 0.857658 | 0.811470-0.912408 | 0/0 | adverse |
| 2 | 65537 | 2 | 6 | gpu | 2.138832 | 0.629904 | 3.372437 | 3.331961-3.410530 | 0/0 | favorable |
| 2 | 65537 | 2 | 6 | build | 5.934500 | 3.295700 | 1.810116 | 1.767278-1.852847 | 0/0 | favorable |
| 2 | 65537 | 2 | 6 | full | 7.595900 | 5.005250 | 1.488492 | 1.448759-1.529745 | 0/0 | favorable |
| 2 | 65537 | 2 | 6 | host | 0.442800 | 0.504000 | 0.850555 | 0.815733-0.886934 | 0/0 | adverse |
| 2 | 65537 | 3 | 6 | gpu | 2.064128 | 0.613376 | 3.368096 | 3.348760-3.387856 | 0/0 | favorable |
| 2 | 65537 | 3 | 6 | build | 5.923100 | 3.243850 | 1.803073 | 1.746384-1.858186 | 0/0 | favorable |
| 2 | 65537 | 3 | 6 | full | 7.753800 | 5.328450 | 1.493093 | 1.441193-1.548662 | 0/0 | favorable |
| 2 | 65537 | 3 | 6 | host | 0.431900 | 0.517300 | 0.838780 | 0.804771-0.875871 | 0/0 | adverse |
| 2 | 65537 | 5 | 24 | gpu | 2.493136 | 2.495344 | 0.999848 | 0.996926-1.002694 | 0/0 | inconclusive |
| 2 | 65537 | 5 | 24 | build | 6.289000 | 6.206450 | 1.010711 | 0.988424-1.033989 | 0/0 | inconclusive |
| 2 | 65537 | 5 | 24 | full | 8.733100 | 8.918350 | 0.988137 | 0.965933-1.010676 | 0/0 | inconclusive |
| 2 | 65537 | 5 | 24 | host | 0.458550 | 0.443500 | 1.035062 | 0.981169-1.094942 | 0/0 | inconclusive |
| 2 | 65537 | 8 | 4 | gpu | 2.140144 | 0.401504 | 5.273583 | 5.210384-5.336880 | 0/0 | favorable |
| 2 | 65537 | 8 | 4 | build | 5.753700 | 3.128500 | 1.850853 | 1.796351-1.907721 | 0/0 | favorable |
| 2 | 65537 | 8 | 4 | full | 7.620400 | 5.077050 | 1.495496 | 1.443634-1.547462 | 0/0 | favorable |
| 2 | 65537 | 8 | 4 | host | 0.393450 | 0.467700 | 0.835978 | 0.789255-0.887915 | 0/0 | adverse |
| 2 | 524287 | 2 | 6 | gpu | 4.156512 | 1.133424 | 3.656962 | 3.641599-3.674241 | 0/0 | favorable |
| 2 | 524287 | 2 | 6 | build | 10.786550 | 7.913850 | 1.376835 | 1.337409-1.417543 | 0/0 | favorable |
| 2 | 524287 | 2 | 6 | full | 13.752100 | 10.841800 | 1.249887 | 1.216236-1.282592 | 0/0 | favorable |
| 2 | 524287 | 2 | 6 | host | 2.712050 | 3.225200 | 0.815717 | 0.780209-0.852040 | 0/0 | adverse |
| 2 | 524288 | 2 | 6 | gpu | 5.146704 | 1.406848 | 3.644192 | 3.638206-3.650115 | 0/0 | favorable |
| 2 | 524288 | 2 | 6 | build | 10.630550 | 7.369000 | 1.425228 | 1.397125-1.454646 | 0/0 | favorable |
| 2 | 524288 | 2 | 6 | full | 14.021050 | 10.627350 | 1.331543 | 1.302064-1.363820 | 0/0 | favorable |
| 2 | 524288 | 2 | 6 | host | 1.507500 | 2.299100 | 0.664606 | 0.643138-0.686451 | 0/0 | adverse |
| 2 | 524289 | 2 | 6 | gpu | 5.242416 | 1.544640 | 3.403438 | 3.398702-3.408362 | 0/0 | favorable |
| 2 | 524289 | 2 | 6 | build | 11.108000 | 7.805200 | 1.417400 | 1.390694-1.444514 | 0/0 | favorable |
| 2 | 524289 | 2 | 6 | full | 14.730200 | 11.786900 | 1.255817 | 1.234869-1.277656 | 0/0 | favorable |
| 2 | 524289 | 2 | 6 | host | 1.464150 | 2.217050 | 0.673760 | 0.654942-0.692210 | 0/0 | adverse |
| 2 | 1000003 | 0 | 0 | gpu | 10.271792 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 2 | 1000003 | 0 | 0 | build | 19.067150 | 12.143800 | 1.545889 | 1.513960-1.576904 | 0/0 | favorable |
| 2 | 1000003 | 0 | 0 | full | 24.807150 | 17.556200 | 1.371986 | 1.332657-1.411879 | 0/0 | favorable |
| 2 | 1000003 | 0 | 0 | host | 2.435200 | 3.292700 | 0.741518 | 0.720359-0.763679 | 0/0 | adverse |
| 2 | 1000003 | 2 | 6 | gpu | 9.223440 | 2.712672 | 3.418685 | 3.409006-3.428504 | 0/0 | favorable |
| 2 | 1000003 | 2 | 6 | build | 18.908600 | 12.435500 | 1.533134 | 1.512394-1.555182 | 0/0 | favorable |
| 2 | 1000003 | 2 | 6 | full | 24.840700 | 17.926600 | 1.365877 | 1.337515-1.394545 | 0/0 | favorable |
| 2 | 1000003 | 2 | 6 | host | 2.441050 | 3.466450 | 0.718149 | 0.701670-0.734491 | 0/0 | adverse |
| 2 | 1000003 | 4 | 24 | gpu | 6.231168 | 6.230656 | 0.999302 | 0.997770-1.000263 | 0/0 | inconclusive |
| 2 | 1000003 | 4 | 24 | build | 15.818900 | 15.728350 | 1.003483 | 0.989081-1.018922 | 0/0 | inconclusive |
| 2 | 1000003 | 4 | 24 | full | 21.451900 | 21.524900 | 1.015198 | 0.995772-1.035857 | 0/0 | inconclusive |
| 2 | 1000003 | 4 | 24 | host | 2.398950 | 2.389150 | 1.002645 | 0.979376-1.027068 | 0/0 | inconclusive |
| 2 | 1000003 | 5 | 24 | gpu | 8.821760 | 8.820304 | 0.999386 | 0.997879-1.000693 | 0/0 | inconclusive |
| 2 | 1000003 | 5 | 24 | build | 20.914400 | 20.800550 | 1.000919 | 0.989525-1.011861 | 0/0 | inconclusive |
| 2 | 1000003 | 5 | 24 | full | 35.494950 | 35.382200 | 1.014979 | 0.994747-1.035819 | 0/0 | inconclusive |
| 2 | 1000003 | 5 | 24 | host | 2.455750 | 2.487600 | 0.994818 | 0.974556-1.015628 | 0/0 | inconclusive |
| 2 | 1000003 | 8 | 4 | gpu | 8.701632 | 2.188944 | 4.061184 | 4.026503-4.093217 | 0/0 | favorable |
| 2 | 1000003 | 8 | 4 | build | 17.261050 | 13.299850 | 1.284261 | 1.250182-1.316784 | 0/0 | favorable |
| 2 | 1000003 | 8 | 4 | full | 24.579900 | 21.237750 | 1.164573 | 1.144441-1.184387 | 0/0 | favorable |
| 2 | 1000003 | 8 | 4 | host | 2.501850 | 3.443200 | 0.722893 | 0.704871-0.740996 | 0/0 | adverse |
| 2 | 1000003 | 6 | 0 | gpu | 10.682880 | -0.000000 | undefined | undefined-undefined | 0/30 | unresolved |
| 2 | 1000003 | 6 | 0 | build | 19.776500 | 13.972800 | 1.438324 | 1.380972-1.505089 | 0/0 | favorable |
| 2 | 1000003 | 6 | 0 | full | 26.784750 | 21.266650 | 1.271771 | 1.208799-1.345039 | 0/0 | favorable |
| 2 | 1000003 | 6 | 0 | host | 2.447400 | 3.347350 | 0.735987 | 0.708247-0.765807 | 0/0 | adverse |

## Source and local evidence locations

Source paths for this workspace:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_sort.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits_sort_reference.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_radix_digits_controls_reference.rs`

Ignored evidence is under `C:/Users/rsasa/Workspace/SpatialRust/target/automation/`:
the `voxel-digits-*` raw logs/samples, boundary diagnostics, validation,
source/executable hashes and delivery receipts. Raw process identities stay
local; published evidence excludes adapter model names. This change does not
establish a competitor maturity ranking or a release gate on other devices.
