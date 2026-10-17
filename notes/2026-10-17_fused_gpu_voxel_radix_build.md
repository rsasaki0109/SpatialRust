# GPU voxel source indices constructed in the first radix scatter

Parent: `3c4c6c23755d98021c9aab00605ce34aa4ef6c40` (merged PR 264).
Workspace: `C:\Users\rsasa\Workspace\SpatialRust`.

For two or more keys, the first z/shift-zero digit reads the key buffer directly.
Its scatter supplies the increasing source index while writing to the original
distinct ping-pong scratch buffer. This omits the separate entry-build pass and
its intermediate entry write/read. After the first digit, input becomes scratch
and output becomes owned entries. The remaining 23 stable digits are unchanged;
all 24 digits still finish in the original owned allocation.

## Correctness and resource contract

One private, lazily cached shader constructs indices only in this first
scatter. Each active lane reads its own key, replaces the key padding word with
the global index and writes `(x, y, z, global index)` at its original stable
destination. Inactive scatter lanes return under the original count guard;
histogram tail lanes still execute all original workgroup barriers. Digit extraction,
signed sign-bit flip, masks, ranks, group counts, prefixes and scatter indices
retain the immediate parent's exact arithmetic. Stable passes on z/y/x preserve
increasing source indices for identical keys without sorting a fourth word.

The first digit binds distinct read-only keys and writable scratch; later
digits bind distinct owned entries/scratch. The borrowed keys never become a
writable ping-pong target. Dispatch boundaries preserve producer/consumer
ordering. The original small/large cached-prefix threshold (2,048 workgroups)
is unchanged; the first scatter shader is cached lazily once per runtime and
shares both original histogram/prefix variants. The complete original radix
shader is unchanged, including the default scatter's immutable entry load.
Default and prebuilt-entry scatter pipelines retain their exact original source.
A native-only prebuilt-entry wrapper retains the
existing diagnostics; public APIs are unchanged.

Empty input still returns before sorting. Singleton input keeps its original
entry-build pass. Input/storage/dispatch/padded-count checks still run before
work and retain error order and text. CPU keys retain their original clamp and
explicit upload. GPU inputs stay GPU-resident through sorting. Compaction,
caller-owned outputs, scratch requests/leases and four-byte nonempty scalar
readback retain their parent contracts.

| Property for N > 1 | Parent | Current |
| --- | --- | --- |
| Build + radix compute passes | 73 | 72 |
| Build + radix queue submissions | 2 | 1 |
| Build-specific 16-byte uniform requests | 1 | 0 |
| Bind group requests | 25 | 24 |
| Owned entries | 16*N bytes | 16*N bytes |
| Radix scratch, histogram, offsets, ranks | Original | Original |
| Construction scalar readback | 4 bytes | 4 bytes |
| Optional timestamp payload for build + radix | 1,168 bytes | 1,152 bytes |

There is no VRAM allocation or payload-transfer saving claimed. The original
16*N-byte intermediate entry write is omitted. The first histogram/scatter now
read keys directly instead of reading the intermediate entries. Actual component
loads, cache transactions and DRAM bytes depend on compilation and hardware;
no bandwidth figure is inferred from WGSL record size.

Frame preflight accounts for the removed build stage on each of two non-singleton
grids. Exact capacities for 0/1/2/257 points become 0/9/151/155. Existing frame
regressions accept exact capacity, reject one fewer stage before scratch/upload
work, and verify receipts, positions, segments and normal counts.

## Exactness evidence

The complete immediate-parent radix host, shader, entry-build shader, sort API
host and pipeline constructors are frozen in native-test-only references.
Mechanical changes inject those constructors, adjust module paths/visibility,
and suppress warnings for unused reference entry points. The compactor is
unchanged. Source auditing verifies those complete references, enumerates the
production substitutions, and verifies the original digit/rank/prefix/scatter
arithmetic after removing only the enumerated first-input and ping-pong host
substitutions. The private scatter shader is derived from the complete original
structures, bindings, digit function and scatter, changing only the final store
to supply the source index. Existing radix shaders and the legacy whole-shader
identity guard remain exact; subsequent default scatter compilation has no new
mutable local, branch or override.

- VOXEL_RADIX_BUILD_FULL checked 76135932 raw stable sort words and 30 complete parent/CPU API pairs; signed extremes, padding, singleton/empty and retained outputs verified.
- VOXEL_RADIX_BUILD_RAW checked 333184728 entry/first-scatter/histogram/rank/offset/source/guard words across 66 paired fixtures; both cached and original rank paths.
- VOXEL_RADIX_BUILD_CACHE verified one immutable scatter specialization and four concurrent full sorts.
- VOXEL_RADIX_BUILD_STORAGE verified 48 optional query bytes removed, identical scratch storage, warmed reuse, zero budgets and unchanged preflight errors.

Native validation: **397 passing tests** (image=151, voxel=204, frames=5, ransac=8, segmentation=8, filter=18, facade=2, facade-gaussian=1). Clippy all features/targets with warnings denied, CPU-only check, minimal and CI-feature WASM checks with warnings denied, formatting and diff checks pass. 548 original GPU/image Rust/WGSL sources are unchanged after line-ending normalization; 562 source/executable SHA-256 receipts agree before measurements and after broad validation. Repository and PR-body adapter-name audit has zero matches.

## Paired measurements and limitations

Two complete surveys run the same ten cases, with three warmups and thirty
paired samples per case/route. Wall routes use fifteen measured pairs in each
order; first-pass GPU routes alternate fifteen pairs in each order. Pipelines
and both variants are warmed before recorded samples. Correctness assertions,
reference generation and output destruction are outside timed intervals.

`pass` sums GPU timestamps for parent build+first complete digit versus the
direct-key first digit. The sum excludes inter-pass gaps, command setup,
transfers and waits; instrumentation has four parent stages and three current
stages. `resident` measures
an unprofiled completed resident-key sort and compaction, including the existing
four-byte scalar synchronization. `full` measures the unprofiled public CPU-key
call through original host conversion/upload, full GPU sorting/compaction and
explicit metadata readback. Resident result checking/readback is outside its
interval. Thus timestamp improvements alone do not establish full-call gains.

All cases and all samples are retained, including adverse/inconclusive results.
Ratios are parent/current geometric means of paired logs; intervals bootstrap
10,000 paired log resamples, seed `2026101715 + 1000*survey + 10*case + routeIndex`,
sorted endpoints 249/9749. A lower bound above one is favorable; an upper bound
below one is adverse. No filtering or timing-dependent retry is used.

- pass: 20 favorable / 0 inconclusive / 0 adverse.
- resident: 5 favorable / 15 inconclusive / 0 adverse.
- full: 5 favorable / 13 inconclusive / 2 adverse.

- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true
- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true

- survey 1 pre: quiet=False; busy process count=2, CPU core estimates=[0.475, 0.512]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 1 post: quiet=False; busy process count=3, CPU core estimates=[0.463, 0.512, 0.481]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 pre: quiet=False; busy process count=3, CPU core estimates=[0.896, 0.485, 0.515]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 post: quiet=False; busy process count=2, CPU core estimates=[0.522, 0.36]; GPU utilization samples=[[[0, 0]], [[0, 0]]]

Pre/post sampling cannot establish isolation during every pair. Shared-host CPU/GPU load, timing precision and clock/cache state limit these diagnostics. Cold shader compilation, portable speedups and application-wide improvements are unmeasured. Process names/IDs and local adapter names are omitted from published evidence.

| Survey | Points | Pattern | Route | Parent median ms | Current median ms | Paired ratio (95% interval) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 2 | signed random | pass | 0.014832 | 0.012096 | 1.2361 (1.1967, 1.2771) |
| 1 | 2 | signed random | resident | 2.094200 | 2.057950 | 1.0231 (0.9954, 1.0485) |
| 1 | 2 | signed random | full | 5.285550 | 5.206400 | 1.0258 (1.0063, 1.0447) |
| 1 | 256 | repeated reversed keys | pass | 0.015616 | 0.012784 | 1.2481 (1.2089, 1.2882) |
| 1 | 256 | repeated reversed keys | resident | 2.129200 | 2.084750 | 1.0466 (1.0205, 1.0776) |
| 1 | 256 | repeated reversed keys | full | 5.256400 | 5.221400 | 1.0099 (0.9863, 1.0364) |
| 1 | 257 | repeated reversed keys | pass | 0.015776 | 0.012688 | 1.2451 (1.2126, 1.2802) |
| 1 | 257 | repeated reversed keys | resident | 2.130400 | 2.082800 | 1.0329 (1.0028, 1.0656) |
| 1 | 257 | repeated reversed keys | full | 5.244150 | 5.149700 | 1.0308 (1.0009, 1.0613) |
| 1 | 65,537 | signed random | pass | 0.160112 | 0.125552 | 1.2644 (1.2108, 1.3171) |
| 1 | 65,537 | signed random | resident | 5.365900 | 5.357000 | 1.0237 (0.9725, 1.0899) |
| 1 | 65,537 | signed random | full | 10.949800 | 11.086950 | 0.9743 (0.9477, 0.9989) |
| 1 | 65,537 | one key | pass | 0.140624 | 0.121648 | 1.1358 (1.0800, 1.1954) |
| 1 | 65,537 | one key | resident | 4.673900 | 4.383900 | 1.0134 (0.9797, 1.0468) |
| 1 | 65,537 | one key | full | 8.450000 | 8.696550 | 0.9683 (0.9456, 0.9916) |
| 1 | 524,288 | signed random | pass | 0.457168 | 0.368336 | 1.2390 (1.2333, 1.2445) |
| 1 | 524,288 | signed random | resident | 11.462050 | 11.406900 | 1.0058 (0.9904, 1.0214) |
| 1 | 524,288 | signed random | full | 29.600250 | 30.621450 | 0.9800 (0.9608, 1.0002) |
| 1 | 524,289 | signed random | pass | 0.294544 | 0.225984 | 1.3045 (1.3016, 1.3083) |
| 1 | 524,289 | signed random | resident | 8.373050 | 8.338200 | 1.0054 (0.9952, 1.0155) |
| 1 | 524,289 | signed random | full | 32.401000 | 33.017200 | 1.0044 (0.9819, 1.0364) |
| 1 | 1,000,003 | signed random | pass | 0.795648 | 0.637072 | 1.2495 (1.2459, 1.2532) |
| 1 | 1,000,003 | signed random | resident | 17.779650 | 17.724000 | 0.9983 (0.9877, 1.0096) |
| 1 | 1,000,003 | signed random | full | 55.481000 | 56.193650 | 0.9900 (0.9659, 1.0128) |
| 1 | 1,000,003 | one key | pass | 0.504784 | 0.354336 | 1.4257 (1.4176, 1.4340) |
| 1 | 1,000,003 | one key | resident | 11.360950 | 11.299500 | 1.0012 (0.9922, 1.0107) |
| 1 | 1,000,003 | one key | full | 35.040600 | 34.856750 | 1.0005 (0.9867, 1.0132) |
| 1 | 1,000,003 | repeated reversed keys | pass | 0.469136 | 0.337360 | 1.3868 (1.3775, 1.3959) |
| 1 | 1,000,003 | repeated reversed keys | resident | 11.256300 | 11.283700 | 0.9934 (0.9815, 1.0058) |
| 1 | 1,000,003 | repeated reversed keys | full | 35.477100 | 35.210650 | 1.0143 (1.0029, 1.0262) |
| 2 | 2 | signed random | pass | 0.014944 | 0.012016 | 1.2365 (1.1963, 1.2789) |
| 2 | 2 | signed random | resident | 2.100700 | 2.065550 | 1.0377 (1.0129, 1.0627) |
| 2 | 2 | signed random | full | 5.221100 | 5.065150 | 1.0234 (1.0069, 1.0414) |
| 2 | 256 | repeated reversed keys | pass | 0.015856 | 0.012736 | 1.2461 (1.2045, 1.2897) |
| 2 | 256 | repeated reversed keys | resident | 2.155800 | 2.101950 | 1.0172 (0.9783, 1.0456) |
| 2 | 256 | repeated reversed keys | full | 5.196350 | 5.169600 | 0.9908 (0.9594, 1.0197) |
| 2 | 257 | repeated reversed keys | pass | 0.035776 | 0.028608 | 1.2488 (1.2240, 1.2740) |
| 2 | 257 | repeated reversed keys | resident | 2.276150 | 2.210700 | 1.0044 (0.9761, 1.0294) |
| 2 | 257 | repeated reversed keys | full | 5.418200 | 5.397950 | 1.0184 (1.0026, 1.0360) |
| 2 | 65,537 | signed random | pass | 0.162016 | 0.134768 | 1.1928 (1.1627, 1.2279) |
| 2 | 65,537 | signed random | resident | 5.878550 | 5.929250 | 0.9846 (0.9665, 1.0029) |
| 2 | 65,537 | signed random | full | 11.727250 | 11.743300 | 1.0169 (0.9908, 1.0502) |
| 2 | 65,537 | one key | pass | 0.140960 | 0.121152 | 1.1115 (1.0679, 1.1573) |
| 2 | 65,537 | one key | resident | 5.004200 | 5.088700 | 0.9928 (0.9761, 1.0142) |
| 2 | 65,537 | one key | full | 9.508750 | 9.648450 | 0.9844 (0.9591, 1.0115) |
| 2 | 524,288 | signed random | pass | 0.468464 | 0.379248 | 1.2349 (1.2293, 1.2405) |
| 2 | 524,288 | signed random | resident | 10.892150 | 10.851000 | 0.9994 (0.9842, 1.0165) |
| 2 | 524,288 | signed random | full | 30.046900 | 30.505850 | 0.9839 (0.9660, 1.0097) |
| 2 | 524,289 | signed random | pass | 0.624128 | 0.504960 | 1.2369 (1.2327, 1.2410) |
| 2 | 524,289 | signed random | resident | 14.472850 | 14.379550 | 1.0133 (1.0036, 1.0240) |
| 2 | 524,289 | signed random | full | 38.161650 | 38.808300 | 0.9882 (0.9672, 1.0115) |
| 2 | 1,000,003 | signed random | pass | 0.822816 | 0.661600 | 1.2423 (1.2378, 1.2468) |
| 2 | 1,000,003 | signed random | resident | 20.885650 | 21.013550 | 0.9976 (0.9926, 1.0022) |
| 2 | 1,000,003 | signed random | full | 61.496250 | 61.171000 | 0.9895 (0.9645, 1.0126) |
| 2 | 1,000,003 | one key | pass | 0.445712 | 0.314192 | 1.4180 (1.4096, 1.4263) |
| 2 | 1,000,003 | one key | resident | 11.860150 | 11.739750 | 1.0210 (1.0017, 1.0447) |
| 2 | 1,000,003 | one key | full | 36.678800 | 36.131550 | 1.0212 (0.9946, 1.0491) |
| 2 | 1,000,003 | repeated reversed keys | pass | 0.398592 | 0.270160 | 1.4693 (1.4634, 1.4751) |
| 2 | 1,000,003 | repeated reversed keys | resident | 10.007550 | 10.037350 | 1.0062 (0.9935, 1.0192) |
| 2 | 1,000,003 | repeated reversed keys | full | 34.275750 | 33.632150 | 1.0163 (0.9965, 1.0379) |

## Earlier shared-shader direct-scatter surveys (all rows retained)

The earlier direct-scatter candidate first ran two complete surveys before
the broad regression suite reached an old assertion requiring the complete
production shader text to equal its earlier frozen shader. No numeric mismatch
was reported: the assertion stopped before its GPU fixture loop. Update that
guard to undo only the two enumerated first-scatter additions, preserving the
complete old histogram/prefix/default-scatter text comparison. All runtime
fixtures and expected offsets remain unchanged.

That candidate then ran two more complete surveys after the test-only guard
update/rebuild. Both pairs of surveys are preserved below. The shared-shader
version added a default-disabled override and changed the original scatter's
immutable entry load into a mutable value. Repeated completed-call adverse
conditions motivated the final private shader, restoring the complete original
radix shader and its original whole-source guard. This removes that possible
compilation difference from all later scatter passes; the measurements alone
cannot establish the cause of the earlier regressions.

This is not a timing-dependent retry. Every implementation change and complete
survey remains visible. The final version above has two predetermined complete
surveys, followed by broad validation on the identical executable. The earlier
surveys use the same complete-digit GPU and resident/full windows. The first pair had
GPU 20 favorable / 0 inconclusive / 0 adverse, resident 4 favorable /
15 inconclusive / 1 adverse, and full 0 favorable / 19 inconclusive / 1 adverse.
No cross-survey/cross-phase absolute-clock comparison is used for speed claims.

- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true
- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true

- survey 1 pre: quiet=False; busy process count=2, CPU core estimates=[1.359, 0.287]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 1 post: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 pre: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 post: quiet=False; busy process count=3, CPU core estimates=[0.327, 0.256, 0.988]; GPU utilization samples=[[[0, 0]], [[0, 0]]]

| Survey | Points | Pattern | Route | Parent median ms | Current median ms | Paired ratio (95% interval) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 2 | signed random | pass | 0.014720 | 0.012400 | 1.2339 (1.1937, 1.2755) |
| 1 | 2 | signed random | resident | 2.156250 | 2.062450 | 1.0294 (1.0073, 1.0548) |
| 1 | 2 | signed random | full | 5.387650 | 5.225500 | 1.0191 (0.9937, 1.0454) |
| 1 | 256 | repeated reversed keys | pass | 0.015632 | 0.012544 | 1.2570 (1.2211, 1.2938) |
| 1 | 256 | repeated reversed keys | resident | 1.932700 | 1.914650 | 1.0308 (1.0008, 1.0770) |
| 1 | 256 | repeated reversed keys | full | 4.726850 | 4.704000 | 0.9996 (0.9646, 1.0340) |
| 1 | 257 | repeated reversed keys | pass | 0.015632 | 0.012784 | 1.2346 (1.1991, 1.2732) |
| 1 | 257 | repeated reversed keys | resident | 2.054550 | 2.054050 | 1.0092 (0.9884, 1.0323) |
| 1 | 257 | repeated reversed keys | full | 4.971800 | 4.981400 | 0.9981 (0.9709, 1.0226) |
| 1 | 65,537 | signed random | pass | 0.161712 | 0.124064 | 1.2488 (1.1929, 1.3016) |
| 1 | 65,537 | signed random | resident | 5.382900 | 5.408350 | 1.0169 (0.9783, 1.0583) |
| 1 | 65,537 | signed random | full | 10.652350 | 11.341250 | 0.9713 (0.9387, 1.0082) |
| 1 | 65,537 | one key | pass | 0.143536 | 0.118992 | 1.2265 (1.1532, 1.3033) |
| 1 | 65,537 | one key | resident | 4.709000 | 4.969300 | 0.9846 (0.9501, 1.0205) |
| 1 | 65,537 | one key | full | 9.525850 | 9.445650 | 0.9784 (0.9435, 1.0135) |
| 1 | 524,288 | signed random | pass | 0.498304 | 0.406640 | 1.2243 (1.2185, 1.2298) |
| 1 | 524,288 | signed random | resident | 12.002100 | 11.957750 | 0.9994 (0.9842, 1.0143) |
| 1 | 524,288 | signed random | full | 30.837500 | 31.309700 | 0.9702 (0.9511, 0.9897) |
| 1 | 524,289 | signed random | pass | 0.560448 | 0.452496 | 1.2383 (1.2336, 1.2432) |
| 1 | 524,289 | signed random | resident | 14.494450 | 14.284500 | 1.0088 (0.9939, 1.0262) |
| 1 | 524,289 | signed random | full | 37.194450 | 37.378500 | 0.9961 (0.9702, 1.0239) |
| 1 | 1,000,003 | signed random | pass | 0.793824 | 0.637120 | 1.2477 (1.2436, 1.2519) |
| 1 | 1,000,003 | signed random | resident | 19.187050 | 19.007150 | 0.9932 (0.9826, 1.0033) |
| 1 | 1,000,003 | signed random | full | 56.723600 | 56.760500 | 1.0133 (0.9899, 1.0407) |
| 1 | 1,000,003 | one key | pass | 0.442992 | 0.312464 | 1.4136 (1.4040, 1.4230) |
| 1 | 1,000,003 | one key | resident | 10.325200 | 10.441200 | 1.0001 (0.9886, 1.0121) |
| 1 | 1,000,003 | one key | full | 33.403300 | 33.405450 | 0.9934 (0.9767, 1.0099) |
| 1 | 1,000,003 | repeated reversed keys | pass | 0.471520 | 0.339456 | 1.3923 (1.3852, 1.3993) |
| 1 | 1,000,003 | repeated reversed keys | resident | 11.294900 | 11.232700 | 1.0078 (0.9892, 1.0265) |
| 1 | 1,000,003 | repeated reversed keys | full | 36.149800 | 35.859500 | 1.0082 (0.9949, 1.0222) |
| 2 | 2 | signed random | pass | 0.014848 | 0.012048 | 1.2315 (1.1937, 1.2706) |
| 2 | 2 | signed random | resident | 2.208500 | 2.167650 | 1.0314 (1.0051, 1.0578) |
| 2 | 2 | signed random | full | 5.493800 | 5.532600 | 0.9863 (0.9552, 1.0144) |
| 2 | 256 | repeated reversed keys | pass | 0.015488 | 0.012624 | 1.2379 (1.1973, 1.2800) |
| 2 | 256 | repeated reversed keys | resident | 2.260100 | 2.256050 | 0.9996 (0.9511, 1.0428) |
| 2 | 256 | repeated reversed keys | full | 5.414350 | 5.396850 | 1.0169 (0.9853, 1.0522) |
| 2 | 257 | repeated reversed keys | pass | 0.015648 | 0.012608 | 1.2431 (1.2102, 1.2776) |
| 2 | 257 | repeated reversed keys | resident | 2.144250 | 2.064150 | 1.0592 (1.0126, 1.1079) |
| 2 | 257 | repeated reversed keys | full | 5.032550 | 5.085050 | 1.0087 (0.9838, 1.0350) |
| 2 | 65,537 | signed random | pass | 0.163776 | 0.139648 | 1.1690 (1.1262, 1.2154) |
| 2 | 65,537 | signed random | resident | 5.399000 | 5.317300 | 0.9995 (0.9700, 1.0292) |
| 2 | 65,537 | signed random | full | 11.288500 | 11.329500 | 0.9946 (0.9772, 1.0146) |
| 2 | 65,537 | one key | pass | 0.151456 | 0.126688 | 1.1355 (1.0894, 1.1830) |
| 2 | 65,537 | one key | resident | 4.577500 | 4.720900 | 0.9653 (0.9423, 0.9884) |
| 2 | 65,537 | one key | full | 9.186750 | 9.458500 | 0.9812 (0.9562, 1.0064) |
| 2 | 524,288 | signed random | pass | 0.396160 | 0.319184 | 1.2398 (1.2345, 1.2451) |
| 2 | 524,288 | signed random | resident | 8.248300 | 8.298050 | 1.0085 (0.9953, 1.0211) |
| 2 | 524,288 | signed random | full | 26.673400 | 27.155100 | 0.9855 (0.9698, 1.0017) |
| 2 | 524,289 | signed random | pass | 0.543152 | 0.441184 | 1.2392 (1.2345, 1.2436) |
| 2 | 524,289 | signed random | resident | 11.709800 | 11.914350 | 0.9967 (0.9878, 1.0060) |
| 2 | 524,289 | signed random | full | 33.896650 | 33.379900 | 0.9950 (0.9684, 1.0218) |
| 2 | 1,000,003 | signed random | pass | 0.794528 | 0.637152 | 1.2477 (1.2435, 1.2518) |
| 2 | 1,000,003 | signed random | resident | 18.594550 | 18.738650 | 0.9965 (0.9835, 1.0116) |
| 2 | 1,000,003 | signed random | full | 55.160650 | 55.998000 | 0.9801 (0.9515, 1.0048) |
| 2 | 1,000,003 | one key | pass | 0.475936 | 0.335344 | 1.4189 (1.4076, 1.4299) |
| 2 | 1,000,003 | one key | resident | 10.241450 | 10.381750 | 0.9912 (0.9759, 1.0073) |
| 2 | 1,000,003 | one key | full | 32.640200 | 33.537900 | 0.9881 (0.9654, 1.0103) |
| 2 | 1,000,003 | repeated reversed keys | pass | 0.457536 | 0.325408 | 1.4024 (1.3952, 1.4096) |
| 2 | 1,000,003 | repeated reversed keys | resident | 11.372650 | 11.419750 | 0.9958 (0.9821, 1.0095) |
| 2 | 1,000,003 | repeated reversed keys | full | 35.583300 | 35.753850 | 1.0029 (0.9803, 1.0285) |

### Shared-shader candidate after the test-only guard update

These surveys had GPU 20 favorable / 0 inconclusive / 0 adverse, resident 3 favorable / 16 inconclusive / 1 adverse, and full 0 favorable / 19 inconclusive / 1 adverse. The 65,537-point one-key resident and 524,288-point random full-call adverse conditions appeared again across the earlier surveys. Hash receipts prove that the final private-shader version restores the original radix shader/whole-source guard, changes only its cache loader, adds the private shader and rebuilds the executable; sort, reference and benchmark sources are unchanged from this candidate.

- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true
- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true

- survey 1 pre: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 1 post: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 pre: quiet=False; busy process count=1, CPU core estimates=[0.252]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 post: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]

| Survey | Points | Pattern | Route | Parent median ms | Current median ms | Paired ratio (95% interval) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 2 | signed random | pass | 0.014736 | 0.011888 | 1.2419 (1.2017, 1.2840) |
| 1 | 2 | signed random | resident | 2.122600 | 2.082600 | 1.0129 (0.9880, 1.0383) |
| 1 | 2 | signed random | full | 5.226550 | 5.257250 | 1.0044 (0.9762, 1.0287) |
| 1 | 256 | repeated reversed keys | pass | 0.015632 | 0.012544 | 1.2558 (1.2204, 1.2924) |
| 1 | 256 | repeated reversed keys | resident | 2.134550 | 2.045150 | 1.0422 (1.0144, 1.0702) |
| 1 | 256 | repeated reversed keys | full | 5.324400 | 5.303650 | 1.0118 (0.9787, 1.0543) |
| 1 | 257 | repeated reversed keys | pass | 0.015664 | 0.012608 | 1.2508 (1.2167, 1.2880) |
| 1 | 257 | repeated reversed keys | resident | 2.176100 | 1.987650 | 1.0627 (1.0385, 1.0890) |
| 1 | 257 | repeated reversed keys | full | 5.131800 | 5.132600 | 0.9998 (0.9855, 1.0139) |
| 1 | 65,537 | signed random | pass | 0.162336 | 0.142720 | 1.1618 (1.1177, 1.2070) |
| 1 | 65,537 | signed random | resident | 5.266150 | 5.313100 | 0.9876 (0.9661, 1.0110) |
| 1 | 65,537 | signed random | full | 10.487400 | 10.680100 | 0.9956 (0.9701, 1.0246) |
| 1 | 65,537 | one key | pass | 0.154240 | 0.117040 | 1.2790 (1.2182, 1.3405) |
| 1 | 65,537 | one key | resident | 4.495200 | 4.653250 | 0.9696 (0.9403, 0.9994) |
| 1 | 65,537 | one key | full | 9.115100 | 9.208050 | 0.9848 (0.9642, 1.0055) |
| 1 | 524,288 | signed random | pass | 0.483344 | 0.392096 | 1.2310 (1.2254, 1.2363) |
| 1 | 524,288 | signed random | resident | 11.291950 | 11.307900 | 0.9998 (0.9900, 1.0106) |
| 1 | 524,288 | signed random | full | 29.806300 | 30.375000 | 0.9787 (0.9519, 1.0066) |
| 1 | 524,289 | signed random | pass | 0.653456 | 0.530256 | 1.2303 (1.2254, 1.2352) |
| 1 | 524,289 | signed random | resident | 15.662600 | 15.564000 | 1.0080 (0.9947, 1.0210) |
| 1 | 524,289 | signed random | full | 40.625650 | 41.307000 | 0.9796 (0.9593, 1.0003) |
| 1 | 1,000,003 | signed random | pass | 0.834528 | 0.673408 | 1.2399 (1.2354, 1.2444) |
| 1 | 1,000,003 | signed random | resident | 19.319050 | 19.701200 | 0.9943 (0.9835, 1.0053) |
| 1 | 1,000,003 | signed random | full | 59.618850 | 58.257200 | 1.0083 (0.9881, 1.0287) |
| 1 | 1,000,003 | one key | pass | 0.449952 | 0.317824 | 1.4133 (1.4052, 1.4216) |
| 1 | 1,000,003 | one key | resident | 11.516900 | 11.541000 | 1.0024 (0.9889, 1.0167) |
| 1 | 1,000,003 | one key | full | 36.496150 | 36.195400 | 1.0048 (0.9820, 1.0294) |
| 1 | 1,000,003 | repeated reversed keys | pass | 0.519344 | 0.364672 | 1.4208 (1.4122, 1.4294) |
| 1 | 1,000,003 | repeated reversed keys | resident | 11.674150 | 11.724800 | 0.9957 (0.9808, 1.0108) |
| 1 | 1,000,003 | repeated reversed keys | full | 36.076350 | 35.661550 | 1.0053 (0.9878, 1.0235) |
| 2 | 2 | signed random | pass | 0.014704 | 0.011920 | 1.2375 (1.1935, 1.2836) |
| 2 | 2 | signed random | resident | 2.211650 | 2.148750 | 1.0530 (1.0193, 1.0947) |
| 2 | 2 | signed random | full | 5.296400 | 5.327100 | 1.0096 (0.9915, 1.0280) |
| 2 | 256 | repeated reversed keys | pass | 0.015616 | 0.012496 | 1.2580 (1.2226, 1.2946) |
| 2 | 256 | repeated reversed keys | resident | 2.083600 | 2.025750 | 1.0189 (0.9988, 1.0405) |
| 2 | 256 | repeated reversed keys | full | 5.082700 | 4.981300 | 1.0181 (0.9901, 1.0452) |
| 2 | 257 | repeated reversed keys | pass | 0.015616 | 0.012736 | 1.2324 (1.1956, 1.2710) |
| 2 | 257 | repeated reversed keys | resident | 2.187600 | 2.183100 | 1.0006 (0.9537, 1.0494) |
| 2 | 257 | repeated reversed keys | full | 5.099150 | 5.186850 | 1.0000 (0.9714, 1.0295) |
| 2 | 65,537 | signed random | pass | 0.177104 | 0.143824 | 1.2027 (1.1708, 1.2345) |
| 2 | 65,537 | signed random | resident | 6.030150 | 5.946100 | 1.0050 (0.9765, 1.0345) |
| 2 | 65,537 | signed random | full | 11.788950 | 11.762300 | 0.9865 (0.9633, 1.0089) |
| 2 | 65,537 | one key | pass | 0.141872 | 0.115184 | 1.1911 (1.1169, 1.2648) |
| 2 | 65,537 | one key | resident | 4.959800 | 5.138850 | 0.9842 (0.9602, 1.0086) |
| 2 | 65,537 | one key | full | 10.011550 | 9.944950 | 1.0029 (0.9783, 1.0286) |
| 2 | 524,288 | signed random | pass | 0.506880 | 0.412336 | 1.2260 (1.2201, 1.2318) |
| 2 | 524,288 | signed random | resident | 12.284700 | 12.186200 | 0.9972 (0.9810, 1.0137) |
| 2 | 524,288 | signed random | full | 32.020250 | 32.590600 | 0.9796 (0.9590, 0.9990) |
| 2 | 524,289 | signed random | pass | 0.681872 | 0.552784 | 1.2316 (1.2279, 1.2353) |
| 2 | 524,289 | signed random | resident | 14.490050 | 14.490900 | 1.0055 (0.9932, 1.0190) |
| 2 | 524,289 | signed random | full | 39.665150 | 40.328600 | 0.9795 (0.9524, 1.0068) |
| 2 | 1,000,003 | signed random | pass | 0.849600 | 0.685328 | 1.2394 (1.2355, 1.2432) |
| 2 | 1,000,003 | signed random | resident | 20.453600 | 20.055700 | 1.0051 (0.9955, 1.0145) |
| 2 | 1,000,003 | signed random | full | 61.271200 | 60.526000 | 1.0023 (0.9801, 1.0281) |
| 2 | 1,000,003 | one key | pass | 0.557008 | 0.392384 | 1.4117 (1.4024, 1.4210) |
| 2 | 1,000,003 | one key | resident | 12.993850 | 13.041600 | 0.9983 (0.9878, 1.0087) |
| 2 | 1,000,003 | one key | full | 39.370150 | 39.975950 | 1.0128 (0.9848, 1.0408) |
| 2 | 1,000,003 | repeated reversed keys | pass | 0.538528 | 0.383520 | 1.4026 (1.3944, 1.4109) |
| 2 | 1,000,003 | repeated reversed keys | resident | 13.289300 | 13.215750 | 1.0023 (0.9918, 1.0125) |
| 2 | 1,000,003 | repeated reversed keys | full | 39.181700 | 38.745700 | 1.0067 (0.9871, 1.0269) |

## Initial histogram-construction experiment (all rows retained)

Before the direct-scatter implementation, a correct candidate constructed
entries inside the first histogram. It retained the intermediate 16*N entry
write and needed an extra six-binding group for that histogram. Two complete
surveys used the same ten cases and pairing/statistical method. Its `pass`
window was build+histogram versus fused histogram (two versus one timestamped
stages), so these GPU medians must not be compared directly with the final
complete-digit window above. Resident/full windows use the same API definitions.

That first candidate had GPU 20 favorable / 0 inconclusive / 0 adverse,
resident 2 favorable / 17 inconclusive / 1 adverse and full 0 favorable /
15 inconclusive / 5 adverse. The 65,537-point one-key full-call adverse result
recurred in both surveys. Those results motivated removing the intermediate
write and extra group rather than treating the GPU-only improvement as a
completed-call speedup. It is not the final production implementation. Its
source/executable receipts and complete source snapshots are preserved in the
ignored local automation evidence; all diagnostic rows are retained below.

- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true
- backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 wall_order=balanced first_gpu_order=alternating checks_drop_excluded=true

- survey 1 pre: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 1 post: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 pre: quiet=False; busy process count=1, CPU core estimates=[0.272]; GPU utilization samples=[[[0, 0]], [[0, 0]]]
- survey 2 post: quiet=True; busy process count=0, CPU core estimates=[]; GPU utilization samples=[[[0, 0]], [[0, 0]]]

| Survey | Points | Pattern | Route | Parent median ms | Current median ms | Paired ratio (95% interval) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 2 | signed random | pass | 0.007232 | 0.003856 | 1.7786 (1.6586, 1.9046) |
| 1 | 2 | signed random | resident | 2.047550 | 2.001250 | 1.0314 (1.0026, 1.0649) |
| 1 | 2 | signed random | full | 4.835500 | 4.965300 | 0.9862 (0.9649, 1.0077) |
| 1 | 256 | repeated reversed keys | pass | 0.007376 | 0.004192 | 1.7345 (1.6460, 1.8260) |
| 1 | 256 | repeated reversed keys | resident | 1.977350 | 1.943400 | 1.0035 (0.9834, 1.0260) |
| 1 | 256 | repeated reversed keys | full | 4.903900 | 4.833000 | 0.9997 (0.9699, 1.0285) |
| 1 | 257 | repeated reversed keys | pass | 0.007216 | 0.004304 | 1.7363 (1.6384, 1.8452) |
| 1 | 257 | repeated reversed keys | resident | 2.116850 | 2.146950 | 1.0077 (0.9790, 1.0445) |
| 1 | 257 | repeated reversed keys | full | 5.252550 | 5.265350 | 1.0008 (0.9761, 1.0246) |
| 1 | 65,537 | signed random | pass | 0.085920 | 0.076224 | 1.1357 (1.0402, 1.2371) |
| 1 | 65,537 | signed random | resident | 4.395350 | 3.248000 | 1.0272 (0.9811, 1.0861) |
| 1 | 65,537 | signed random | full | 9.180200 | 9.489400 | 0.9654 (0.9415, 0.9901) |
| 1 | 65,537 | one key | pass | 0.084944 | 0.074080 | 1.1500 (1.0414, 1.2652) |
| 1 | 65,537 | one key | resident | 4.304700 | 4.398850 | 0.9611 (0.9301, 0.9879) |
| 1 | 65,537 | one key | full | 8.708700 | 8.902150 | 0.9610 (0.9374, 0.9863) |
| 1 | 524,288 | signed random | pass | 0.171984 | 0.113168 | 1.5213 (1.5137, 1.5283) |
| 1 | 524,288 | signed random | resident | 9.561350 | 9.602400 | 1.0004 (0.9899, 1.0116) |
| 1 | 524,288 | signed random | full | 27.897950 | 28.346850 | 0.9780 (0.9579, 0.9979) |
| 1 | 524,289 | signed random | pass | 0.214736 | 0.153008 | 1.4041 (1.3969, 1.4113) |
| 1 | 524,289 | signed random | resident | 12.425200 | 12.396850 | 0.9905 (0.9801, 1.0008) |
| 1 | 524,289 | signed random | full | 33.923100 | 34.993150 | 0.9776 (0.9525, 1.0050) |
| 1 | 1,000,003 | signed random | pass | 0.312624 | 0.219392 | 1.4259 (1.4198, 1.4321) |
| 1 | 1,000,003 | signed random | resident | 18.272700 | 18.326550 | 1.0036 (0.9966, 1.0107) |
| 1 | 1,000,003 | signed random | full | 56.227900 | 55.993400 | 1.0103 (0.9893, 1.0302) |
| 1 | 1,000,003 | one key | pass | 0.308224 | 0.210304 | 1.4645 (1.4578, 1.4715) |
| 1 | 1,000,003 | one key | resident | 12.069850 | 12.187700 | 0.9956 (0.9792, 1.0120) |
| 1 | 1,000,003 | one key | full | 36.379400 | 37.486300 | 0.9849 (0.9584, 1.0120) |
| 1 | 1,000,003 | repeated reversed keys | pass | 0.277648 | 0.201424 | 1.3786 (1.3726, 1.3848) |
| 1 | 1,000,003 | repeated reversed keys | resident | 11.910100 | 11.947650 | 0.9915 (0.9764, 1.0068) |
| 1 | 1,000,003 | repeated reversed keys | full | 37.709850 | 37.903650 | 0.9902 (0.9631, 1.0179) |
| 2 | 2 | signed random | pass | 0.006944 | 0.003952 | 1.7701 (1.6491, 1.9003) |
| 2 | 2 | signed random | resident | 2.073300 | 2.094900 | 1.0390 (1.0083, 1.0746) |
| 2 | 2 | signed random | full | 5.483150 | 5.323900 | 1.0141 (0.9919, 1.0370) |
| 2 | 256 | repeated reversed keys | pass | 0.007312 | 0.004160 | 1.7443 (1.6545, 1.8418) |
| 2 | 256 | repeated reversed keys | resident | 2.126000 | 2.058400 | 1.0155 (0.9738, 1.0566) |
| 2 | 256 | repeated reversed keys | full | 5.239250 | 5.218150 | 0.9700 (0.8988, 1.0215) |
| 2 | 257 | repeated reversed keys | pass | 0.015872 | 0.008912 | 1.7771 (1.6727, 1.8898) |
| 2 | 257 | repeated reversed keys | resident | 2.259000 | 2.210350 | 0.9728 (0.9130, 1.0272) |
| 2 | 257 | repeated reversed keys | full | 5.558050 | 5.450800 | 0.9969 (0.9468, 1.0355) |
| 2 | 65,537 | signed random | pass | 0.085936 | 0.074096 | 1.1493 (1.0507, 1.2532) |
| 2 | 65,537 | signed random | resident | 5.242850 | 5.291100 | 0.9986 (0.9767, 1.0201) |
| 2 | 65,537 | signed random | full | 10.879300 | 10.984050 | 0.9972 (0.9725, 1.0196) |
| 2 | 65,537 | one key | pass | 0.083488 | 0.072976 | 1.1623 (1.0655, 1.2678) |
| 2 | 65,537 | one key | resident | 4.321800 | 4.407800 | 0.9990 (0.9759, 1.0258) |
| 2 | 65,537 | one key | full | 8.752100 | 9.082850 | 0.9488 (0.9249, 0.9731) |
| 2 | 524,288 | signed random | pass | 0.157008 | 0.104512 | 1.4830 (1.4770, 1.4889) |
| 2 | 524,288 | signed random | resident | 9.256300 | 9.026650 | 0.9974 (0.9757, 1.0219) |
| 2 | 524,288 | signed random | full | 27.645700 | 28.987900 | 0.9842 (0.9595, 1.0099) |
| 2 | 524,289 | signed random | pass | 0.218736 | 0.156192 | 1.3983 (1.3920, 1.4043) |
| 2 | 524,289 | signed random | resident | 12.353150 | 12.479900 | 0.9924 (0.9818, 1.0028) |
| 2 | 524,289 | signed random | full | 34.122100 | 35.202850 | 0.9677 (0.9439, 0.9915) |
| 2 | 1,000,003 | signed random | pass | 0.309664 | 0.217472 | 1.4289 (1.4193, 1.4388) |
| 2 | 1,000,003 | signed random | resident | 17.959000 | 17.911150 | 0.9940 (0.9856, 1.0027) |
| 2 | 1,000,003 | signed random | full | 54.360000 | 54.093300 | 1.0054 (0.9832, 1.0270) |
| 2 | 1,000,003 | one key | pass | 0.279248 | 0.197936 | 1.4173 (1.4098, 1.4250) |
| 2 | 1,000,003 | one key | resident | 10.168850 | 10.155100 | 0.9972 (0.9875, 1.0065) |
| 2 | 1,000,003 | one key | full | 32.296900 | 32.845550 | 0.9962 (0.9770, 1.0163) |
| 2 | 1,000,003 | repeated reversed keys | pass | 0.253280 | 0.175488 | 1.4393 (1.4326, 1.4460) |
| 2 | 1,000,003 | repeated reversed keys | resident | 10.429800 | 10.489900 | 1.0015 (0.9915, 1.0128) |
| 2 | 1,000,003 | repeated reversed keys | full | 33.330650 | 32.434000 | 1.0153 (0.9980, 1.0334) |
