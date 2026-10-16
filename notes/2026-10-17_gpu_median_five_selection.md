# Short exact GPU 5x5 median selection

Parent: `0fd45af805c73a28a49a1b395e03302b65ce6c4c` (merged PR 253).

The resident 5x5 median generates **113 compare-exchange pairs instead of 191**
(40.8% fewer source pairs). A padded odd-even mergesort network replaces the
padded bitonic network. Its comparisons all have ascending, ordered endpoints.
Pairs touching the maximal-byte padding are identities and are removed before
the original backwards dependency pruning retains only median ancestors.
Each retained pair still uses lane-wise u32 min/max for all four physical
components, including alpha. This counts generated source pairs, not machine
instructions or physical memory traffic.

## Correctness argument and scope

Odd-even merging recursively merges the even and odd subsequences and joins
their seams; recursive merging sorts the power-of-two padded input. All seven
padding values are 255, at indices above every real sample. For an ascending
pair with ordered endpoints, a padding endpoint either compares two equal 255s
or leaves a real byte at the lower endpoint and 255 at the upper endpoint.
Those pairs are identities for every u8 lane. Removing comparisons outside the
median's dependency cone preserves its result.

The exhaustive test evaluates all **33,554,432 binary 25-sample windows** against
the frozen network and an independent population-count majority. Each u64 bit
is a separate case: six low input bits enumerate 64 lanes, nineteen high input
bits enumerate 524,288 blocks. AND/OR exactly implement binary min/max.
Thresholding commutes with min/max, so agreement with binary majority for every
threshold establishes the median for arbitrary byte values and ties. The shared
WGSL generator still emits the same min/max operations for each pair.

The 3x3 nineteen-pair network and complete 3x3 shader string remain identical.
Sampling, replicate/constant-zero borders, workgroup and dispatch, planner,
runtime checks, lazy pipeline family, public API, receipts, metadata and output
ownership retain their original code. There is still one device pass, four
physical output bytes per pixel and no image-sized scratch. Upload, readback and
waits remain explicit caller operations. No dependency, core or unsafe changes.

## Validation

- **346 native correctness tests:** 141 image, 184 voxel/readback, 18 filtering,
  2 transfer facade and 1 median facade, including ignored GPU tests and excluding
  benchmarks. All-feature/all-target Clippy with warnings denied, CPU-only,
  minimal and CI WASM builds with warnings denied, formatting and whitespace pass.
- **384 current/complete-parent GPU comparisons / 50,439,680 logical
  bytes**, covering 1..=4 channels, both kernels and borders, single-pixel and
  horizontal/vertical strips, odd dimensions through 1025x257, varied/hot/checker/
  impulse inputs, exact output, metadata and receipts before/after readback.
- Existing independent CPU sorting checks cover 200,000 seeded byte windows and
  512 constant windows. Existing 3x3 checks retain all 512 binary windows and
  362,880 distinct-rank permutations. CPU-image, physical padding, ROI/stride,
  concurrent lazy cache, invalid-limit/runtime, live-output and ownership tests
  remain and pass. The old rank test's 5x5 shader-identity assertions are replaced
  by the new exhaustive 5x5 and unchanged-3x3 checks; its rank loop is unchanged.
- **500 unchanged original GPU/image Rust/WGSL sources**, and **508 validated
  source/executable hashes**. Audit permits the 5x5 network constructor, its two
  private merge helpers and explicit native fixture changes. The complete
  immediate-parent median is frozen with only test visibility, a function name
  and cache-location substitutions. Shared shader generation, planning, public
  call body, resources, transfers and the 3x3 early return compare unchanged.

## Paired diagnostic measurements

Environment: `backend=Vulkan device_type=DiscreteGpu available_workers=12 texture_budget=67108864`. No adapter model name is published.

After all builds/tests, two sequential runs use three warm pairs followed by
thirty measured pairs with alternating old/new order, eleven cases and three
boundaries. All **66** rows, including adverse/inconclusive cases and identical
3x3 controls, are retained. Fixture creation, CPU references, cold pipeline
compilation and assertions are excluded.

- `pass`: GPU pass-boundary timestamps for the exact old/new generated shader
  strings, equivalent layouts/uniform/input/output/dispatch and two owned outputs.
  Both passes share a command buffer with alternating order. Explicit collection
  reads 32 query bytes outside the timestamps; outputs are checked against CPU
  after the rounds. These are compute times, not API/frame latency.
- `resident`: complete public call and `wait_idle` using an uploaded source.
  Readback verification, assertions, output destruction and recycling follow
  timing. Receipts preserve the original explicit upload and one device stage.
- `full`: upload, complete public median and readback. Assertions, destruction
  and recycling occur after timing. This includes host channel conversion and
  transfer/synchronization costs, so it is distinct from resident compute.

The paired geometric mean of original/new sample ratios is above one when the
new path is faster. Pointwise percentile intervals resample paired log ratios
10,000 times with seed `2026101705 + 100*run + 10*case + route_index`, route order
pass/resident/full and bounds 249/9749. They have no multiple-comparison
correction and do not establish causality or host isolation. Zero samples would
suppress ratios; raw records retain every sample.

Both runs favor the 5x5 GPU pass in all **18/18** rows. Resident APIs have
**15 favorable / 3 inconclusive / 0 adverse** intervals. Full calls have
**8 favorable / 9 inconclusive / 1 adverse** interval. Identical 3x3 pass and
resident controls all cross one; one full-call control favors the new route,
showing why full-call changes cannot all be attributed to the network.

For varied 1024x1024 grayscale, GPU pass medians are 0.796448 -> 0.581760 ms and
0.796464 -> 0.581344 ms (about 27% lower). Resident medians are
1.087350 -> 0.852050 ms and 1.073050 -> 0.848900 ms. For varied 1920x1080 RGB,
resident medians are 2.018800 -> 1.591300 ms and 2.241600 -> 1.773350 ms.
The sole adverse full-call row is 257x193 RGB in run two: median
1.686850 -> 1.817100 ms, paired ratio 0.962132 [0.926192, 0.998002]. It is kept
alongside all faster rows. No general transfer-inclusive speedup is claimed.

Run 1: observed-quiet **false**; busy-process counts 1 before / 1 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Run 2: observed-quiet **false**; busy-process counts 3 before / 0 after. GPU/memory utilization samples `[[[0, 0]], [[0, 0]]]` before / `[[[0, 0]], [[0, 0]]]` after.

Quiet requires no other sampled process above 0.25 CPU cores and all GPU/memory
samples at most 5%. Surveys wait two seconds between snapshots; CPU intervals
include snapshot overhead and exclude the coordinator. Pre/post sampling cannot
prove isolation during every timed pair. These are shared-host diagnostic
observations, not portable throughput or application latency guarantees.
Process identities remain in ignored local receipts.

Profiles: 0 varied bytes, 2 constant 255, 4 impulses on 17. Borders: 0 replicate,
1 constant zero. Timed channels are 1/3/4; all four channels receive correctness
coverage. First pipeline compilation remains lazy and is outside warm timing.

| Run | Shape / channels / kernel / profile / border | Boundary | Original median ms | New median ms | Paired original/new ratio [95% interval] |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 65x33 / 1 / 5 / 0 / 0 | pass | 0.012576 | 0.010208 | 1.221140 [1.188499, 1.254376] |
| 1 | 65x33 / 1 / 5 / 0 / 0 | resident | 0.223600 | 0.198100 | 1.060486 [1.029786, 1.093864] |
| 1 | 65x33 / 1 / 5 / 0 / 0 | full | 1.119950 | 1.127650 | 0.981197 [0.951610, 1.005215] |
| 1 | 257x193 / 3 / 5 / 0 / 0 | pass | 0.048576 | 0.035984 | 1.355042 [1.339606, 1.370619] |
| 1 | 257x193 / 3 / 5 / 0 / 0 | resident | 0.274450 | 0.267300 | 1.033137 [0.980568, 1.085732] |
| 1 | 257x193 / 3 / 5 / 0 / 0 | full | 1.296050 | 1.302700 | 0.983216 [0.952529, 1.013387] |
| 1 | 1024x1024 / 1 / 5 / 0 / 0 | pass | 0.796448 | 0.581760 | 1.369293 [1.368343, 1.370188] |
| 1 | 1024x1024 / 1 / 5 / 0 / 0 | resident | 1.087350 | 0.852050 | 1.260039 [1.224591, 1.294815] |
| 1 | 1024x1024 / 1 / 5 / 0 / 0 | full | 4.483900 | 4.360050 | 1.036718 [1.020748, 1.051910] |
| 1 | 1920x1080 / 3 / 5 / 0 / 0 | pass | 1.895968 | 1.299888 | 1.376887 [1.370477, 1.389051] |
| 1 | 1920x1080 / 3 / 5 / 0 / 0 | resident | 2.018800 | 1.591300 | 1.275874 [1.257370, 1.295515] |
| 1 | 1920x1080 / 3 / 5 / 0 / 0 | full | 8.446550 | 8.052450 | 1.047462 [1.028283, 1.067307] |
| 1 | 1920x1080 / 4 / 5 / 0 / 0 | pass | 1.366320 | 0.992080 | 1.376828 [1.376026, 1.377471] |
| 1 | 1920x1080 / 4 / 5 / 0 / 0 | resident | 1.913350 | 1.547650 | 1.238998 [1.215859, 1.262101] |
| 1 | 1920x1080 / 4 / 5 / 0 / 0 | full | 6.311850 | 6.017300 | 1.041126 [1.017298, 1.061873] |
| 1 | 1025x257 / 4 / 5 / 0 / 0 | pass | 0.237776 | 0.172416 | 1.378188 [1.373728, 1.382618] |
| 1 | 1025x257 / 4 / 5 / 0 / 0 | resident | 0.504700 | 0.425800 | 1.177869 [1.122430, 1.248900] |
| 1 | 1025x257 / 4 / 5 / 0 / 0 | full | 2.104250 | 2.090900 | 0.984612 [0.941394, 1.025309] |
| 1 | 4097x17 / 3 / 5 / 0 / 0 | pass | 0.111200 | 0.080464 | 1.375927 [1.366838, 1.384916] |
| 1 | 4097x17 / 3 / 5 / 0 / 0 | resident | 0.299850 | 0.281700 | 1.140592 [1.050575, 1.302011] |
| 1 | 4097x17 / 3 / 5 / 0 / 0 | full | 1.828150 | 1.818650 | 1.002263 [0.984857, 1.020622] |
| 1 | 1024x1024 / 4 / 5 / 2 / 0 | pass | 1.058304 | 0.771968 | 1.370211 [1.369262, 1.371134] |
| 1 | 1024x1024 / 4 / 5 / 2 / 0 | resident | 1.472850 | 1.179400 | 1.256419 [1.211800, 1.324154] |
| 1 | 1024x1024 / 4 / 5 / 2 / 0 | full | 3.881650 | 3.602450 | 1.066167 [1.043406, 1.088569] |
| 1 | 1024x1024 / 3 / 5 / 4 / 1 | pass | 0.927792 | 0.677840 | 1.368677 [1.368047, 1.369241] |
| 1 | 1024x1024 / 3 / 5 / 4 / 1 | resident | 1.347850 | 1.080600 | 1.240807 [1.210301, 1.274404] |
| 1 | 1024x1024 / 3 / 5 / 4 / 1 | full | 5.107800 | 4.887950 | 1.050563 [1.036082, 1.066998] |
| 1 | 257x193 / 4 / 3 / 0 / 0 | pass | 0.012512 | 0.012672 | 1.003113 [0.965736, 1.043018] |
| 1 | 257x193 / 4 / 3 / 0 / 0 | resident | 0.261750 | 0.272250 | 0.961320 [0.906046, 1.019295] |
| 1 | 257x193 / 4 / 3 / 0 / 0 | full | 1.791950 | 1.753750 | 1.023040 [1.003102, 1.043441] |
| 1 | 1024x1024 / 4 / 3 / 0 / 0 | pass | 0.325920 | 0.325472 | 1.001072 [0.990623, 1.011776] |
| 1 | 1024x1024 / 4 / 3 / 0 / 0 | resident | 0.529500 | 0.528350 | 1.000759 [0.962568, 1.041064] |
| 1 | 1024x1024 / 4 / 3 / 0 / 0 | full | 3.631350 | 3.668650 | 1.007743 [0.976680, 1.041734] |
| 2 | 65x33 / 1 / 5 / 0 / 0 | pass | 0.012560 | 0.010224 | 1.217190 [1.184723, 1.249835] |
| 2 | 65x33 / 1 / 5 / 0 / 0 | resident | 0.244750 | 0.238600 | 1.010926 [0.947445, 1.085718] |
| 2 | 65x33 / 1 / 5 / 0 / 0 | full | 1.259350 | 1.285700 | 0.990566 [0.955475, 1.028193] |
| 2 | 257x193 / 3 / 5 / 0 / 0 | pass | 0.048704 | 0.035904 | 1.355116 [1.339889, 1.370513] |
| 2 | 257x193 / 3 / 5 / 0 / 0 | resident | 0.394350 | 0.390100 | 1.024639 [0.982473, 1.067742] |
| 2 | 257x193 / 3 / 5 / 0 / 0 | full | 1.686850 | 1.817100 | 0.962132 [0.926192, 0.998002] |
| 2 | 1024x1024 / 1 / 5 / 0 / 0 | pass | 0.796464 | 0.581344 | 1.370260 [1.369715, 1.370809] |
| 2 | 1024x1024 / 1 / 5 / 0 / 0 | resident | 1.073050 | 0.848900 | 1.252651 [1.231141, 1.272389] |
| 2 | 1024x1024 / 1 / 5 / 0 / 0 | full | 4.450650 | 4.318600 | 1.038660 [1.012805, 1.061479] |
| 2 | 1920x1080 / 3 / 5 / 0 / 0 | pass | 1.894064 | 1.380816 | 1.371075 [1.369779, 1.372145] |
| 2 | 1920x1080 / 3 / 5 / 0 / 0 | resident | 2.241600 | 1.773350 | 1.273855 [1.256561, 1.291221] |
| 2 | 1920x1080 / 3 / 5 / 0 / 0 | full | 8.824350 | 8.163250 | 1.067062 [1.050402, 1.084344] |
| 2 | 1920x1080 / 4 / 5 / 0 / 0 | pass | 1.234384 | 0.895728 | 1.377588 [1.376883, 1.378222] |
| 2 | 1920x1080 / 4 / 5 / 0 / 0 | resident | 1.617200 | 1.317700 | 1.245803 [1.226621, 1.265454] |
| 2 | 1920x1080 / 4 / 5 / 0 / 0 | full | 6.040150 | 5.705250 | 1.062384 [1.042798, 1.090934] |
| 2 | 1025x257 / 4 / 5 / 0 / 0 | pass | 0.177792 | 0.129328 | 1.370785 [1.363007, 1.378346] |
| 2 | 1025x257 / 4 / 5 / 0 / 0 | resident | 0.471300 | 0.407800 | 1.160792 [1.114300, 1.211669] |
| 2 | 1025x257 / 4 / 5 / 0 / 0 | full | 2.333900 | 2.368700 | 0.990632 [0.960933, 1.021241] |
| 2 | 4097x17 / 3 / 5 / 0 / 0 | pass | 0.073776 | 0.053504 | 1.379976 [1.372116, 1.387714] |
| 2 | 4097x17 / 3 / 5 / 0 / 0 | resident | 0.374700 | 0.363850 | 1.110072 [1.007567, 1.292522] |
| 2 | 4097x17 / 3 / 5 / 0 / 0 | full | 2.113000 | 2.158350 | 1.003400 [0.962445, 1.051953] |
| 2 | 1024x1024 / 4 / 5 / 2 / 0 | pass | 0.631968 | 0.460448 | 1.372443 [1.371993, 1.372916] |
| 2 | 1024x1024 / 4 / 5 / 2 / 0 | resident | 0.988300 | 0.814550 | 1.244376 [1.188852, 1.332305] |
| 2 | 1024x1024 / 4 / 5 / 2 / 0 | full | 3.570150 | 3.540900 | 1.012686 [0.986620, 1.040790] |
| 2 | 1024x1024 / 3 / 5 / 4 / 1 | pass | 0.987888 | 0.722016 | 1.368298 [1.367847, 1.368756] |
| 2 | 1024x1024 / 3 / 5 / 4 / 1 | resident | 1.293600 | 1.056350 | 1.184946 [1.133818, 1.224089] |
| 2 | 1024x1024 / 3 / 5 / 4 / 1 | full | 5.452950 | 5.392750 | 1.006660 [0.977458, 1.034429] |
| 2 | 257x193 / 4 / 3 / 0 / 0 | pass | 0.013440 | 0.013504 | 0.999669 [0.959846, 1.039533] |
| 2 | 257x193 / 4 / 3 / 0 / 0 | resident | 0.257600 | 0.268300 | 0.967824 [0.917357, 1.018348] |
| 2 | 257x193 / 4 / 3 / 0 / 0 | full | 1.808900 | 1.764100 | 1.016977 [0.998340, 1.039068] |
| 2 | 1024x1024 / 4 / 3 / 0 / 0 | pass | 0.324960 | 0.325232 | 1.002770 [0.992458, 1.013418] |
| 2 | 1024x1024 / 4 / 3 / 0 / 0 | resident | 0.647750 | 0.625300 | 1.018649 [0.976632, 1.059812] |
| 2 | 1024x1024 / 4 / 3 / 0 / 0 | full | 4.203000 | 4.237400 | 0.988810 [0.962004, 1.015445] |

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Production: `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_five_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_five_reference.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/median_five_benchmark.rs`.
Ignored validation, paired samples and provenance:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/median-five-*`.
