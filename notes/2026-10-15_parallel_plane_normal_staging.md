# Bounded point-to-plane normal staging

Date: 2026-10-15. Parent: `1add8b852c4de9814c6f5e9a3f4323fb6f998ece` (merged PR 203).

## Change and boundary

Both `GpuPointToPlaneInput` constructors share a private CPU normal preparation
helper. From 262,144 native rows, available CPU count, useful 65,536-row bands
and a four-worker cap select bounded scoped workers. Phase one validates finite
components and reduces nonnegative absolute maxima. Every accepted maximum is
finite; merging exact maxima keeps the common scale and the original zero→1
and minimum-normal scale gates. Phase two fills disjoint original-order padded
rows using the original three f32 divisions. Multiplication by a reciprocal or
individual unit normalization is not introduced. Nonzero components whose
scaled magnitude becomes subnormal/zero still reject; signed zeros and positive
zero padding retain their bits.

The same 16*T-byte host payload allocation remains. Parallel packing starts
with a zero-initialized vector, adding a zeroing write and worker-sized handles/
maxima, plus thread stacks/startup. At most four workers run concurrently per
phase. No point-sized temporary besides the existing payload, new GPU buffer,
compute pass, wait, readback or transfer is added. Count/device planning and
normal-length errors keep precedence; normal preparation still completes
before geometry validation or device resources. Errors join all scoped work
and return the caller's original invalid-argument error.

Small/one-worker/WASM inputs retain the complete original serial loop body and
exact-capacity packing. Small calls return before worker selection; small/WASM
calls avoid the CPU-count query. Public APIs, common raw
normal weights, geometry/anchor preparation, both target upload routes,
ownership, receipts and reduction math remain. This change parallelizes host
preparation and does not make the complete registration computation GPU resident.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/point_to_plane_normal_preparation.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/point_to_plane_reduction.rs`

## Correctness and integration

The four new pure tests and two new native tests passed. Frozen PR 203 normal preparation and complete constructors
provide independent serial references. Tests cover empty/small/large inputs,
262,143/144/145 worker boundaries, normal scale extremes and signed zeros,
division-underflow boundaries, arbitrary finite words and nonfinite rows at
worker edges. Both upload routes compare actual GPU normal words, metadata,
receipts, owned equations, runtime rejection and error precedence when source
coordinates are also invalid. Scoped concurrent calls retain independent outputs.

Forced available worker counts 0/1/2/3/4/12/usize::MAX compare **339,479,420**
exact packed words and exact common-scale bits. Native integer probes compare
**25,167,928** uploaded normal words in both constructors, including padding,
scale and unchanged geometry/transfer metadata. Large target/small source
cases exercise parallel preparation before equations; **2,304** owned equation/
status words and **30** errors match frozen constructors, including normal-error
precedence over invalid source coordinates. Existing point-to-plane math,
scratch-reuse and ownership tests also passed in the focused 13-test selection.

All **97** selected native release tests passed, covering shared tree/gather,
registration geometry, both ICP trim precisions, GICP covariance/equation
preparation and point-to-plane math/staging. All-feature/all-target Clippy with
warnings denied, CPU-only and WASM facade builds passed. All **8** public facade
tests and final formatting/diff checks also passed on the final implementation.
Source verification permits only the
normal-preparation block to change, compares both frozen references to the
parent and checks all other GPU sources and all 97 shaders. Before/after survey
manifests hash the complete GPU Rust/WGSL set, public smoke test and executable.
The source manifest covers **334** source/executable files, **326** unchanged
pre-existing sources and **97** unchanged shaders. The forbidden local adapter
name count is zero in tracked/nonignored files and the PR body.

The first measured candidate factored the serial path through the parallel
bound/row helpers. It retained exact values but 65,537-row default uploads
regressed in both surveys (paired ratios 0.953684 [0.915747, 0.992161] and
0.947815 [0.911303, 0.989751], medians 4.056650 → 4.289300 ms and 3.461500 →
3.695850 ms). Singleton host preparation in survey 1 and 262,143-row host
preparation in survey 2 also regressed. The final candidate keeps the complete
original serial body and an early small-input return. Its serial source is
compared directly with both the frozen helper and parent's constructor block.
Compiler/call context can still affect timings; retaining the body is not a
promise that every small-input sample is faster. All first-candidate artifacts,
including the executable/source hashes and all 60 comparisons, are archived at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/plane-staging-candidate1`.

## Isolated measurements

Two surveys run sequentially after all builds/tests and a 15-second quiet
window each. A process census rejects an entire attempt if another build/test/
benchmark overlaps, cancelling only the owned process. Accepted attempts require
one passing test, no overlap and empty stderr. Analysis begins after both surveys
and source/executable hash verification.

Ten cases have three warm cycles and 30 paired samples per route, alternating
old/new order per cycle. `host` times complete validation, scale, allocation,
packing and destruction; calls below 262,144 rows repeat 32 times per sample
and report time per call. `default` and `gather` time each complete constructor,
`wait_idle`, source-count/receipt assertions and destruction. Runtime/fixture/
CPU tree construction are excluded. Source and target counts are equal in
timed calls; asymmetric shapes are tested for correctness. Profile 0 has varied
raw normals, profile 1 signed zeros and profile 2 minimum-normal raw normals.
No neighbor search, payload readback, reduction or pose solve is timed.

Each revision retains all 60 comparisons (**120** rows across both revisions).
Paired geometric old/new ratios above one
favor new staging. Each 95% interval uses 10,000 bootstrap resamples of paired
log ratios, sorted indices 249 and 9,749. Seeds are
`2026101503 + 100*survey + 10*case + JSON-field-index`; zero-duration counts
are retained and suppress estimates if nonzero. Host-only gains are separate
from complete upload latency and do not establish iteration/registration gains.

Both final surveys were accepted on attempt 1, with no overlap and empty stderr;
tests finished in 42.39 and 46.31 seconds. The first candidate's surveys took
45.99 and 42.39 seconds under the same measurement procedure. Conditions were
native Windows x86_64, rustc 1.98.1 (`48a229cea`, 2026-09-01), release
optimization, Vulkan discrete backend, 12 available CPU workers with a
four-worker staging cap, 2,147,483,647-byte storage binding limit and 65,535
dispatch groups. All zero-duration counts were zero. Both final manifests
match all 334 source/executable hashes. Absolute timings differ between survey
sets; conclusions compare paired old/new calls within each survey.

At 1,048,577 varied rows, final host preparation medians fell **60.7%/59.4%**:
20.364350 → 8.003700 ms and 21.704650 → 8.806700 ms, paired ratios 2.472368
[2.379419, 2.566397] and 2.443958 [2.364337, 2.523004]. Complete default
upload medians fell **21.1%/19.8%**: 57.499550 → 45.351750 ms and 61.613250
→ 49.416500 ms, ratios 1.254305 [1.224516, 1.284359] and 1.281508
[1.236797, 1.337740]. Complete GPU-gather upload medians fell **19.8%/23.4%**:
53.546900 → 42.951250 ms and 56.663150 → 43.378650 ms, ratios 1.318192
[1.165533, 1.491262] and 1.298021 [1.199059, 1.408284]. These compare the
previous and final normal preparation on the **same placement route**, not
GPU gather against default upload.

Host preparation and default uploads at/above the cutoff favor the final
implementation in both surveys. GPU-gather 262,144-row survey 2 and both
262,145-row intervals cross one; wider intervals retain submission/backend
variation. All 24 small-input comparisons cross one in the final surveys,
including the formerly slower 65,537-row default condition (survey 1 upper
bound 1.000172 is close to one). No small-input speedup or universal
non-regression is claimed. Of 60 final comparisons, **27** cross one and none
are entirely below one; the first revision retained **4** adverse and **29**
inconclusive comparisons. The cutoff/cap are conservative choices for this
environment, not universal optima. No iteration, pose-solve or end-to-end
registration speedup follows from these upload measurements.

Local ignored evidence paths:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/plane-staging-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/plane-staging-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/plane-staging-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/plane-staging-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/run-plane-staging-survey.ps1`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-plane-staging.py`


## First measured candidate

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | host | 0.000091 | 0.000094 | 0.962190 [0.943592, 0.981764] |
| 1 | 1 | 0 | default | 0.025050 | 0.025250 | 0.966047 [0.909027, 1.023287] |
| 1 | 1 | 0 | gather | 0.168500 | 0.170200 | 0.994960 [0.970588, 1.020779] |
| 1 | 4097 | 0 | host | 0.056180 | 0.057047 | 0.990314 [0.972707, 1.008032] |
| 1 | 4097 | 0 | default | 0.179300 | 0.180350 | 0.992315 [0.940217, 1.046509] |
| 1 | 4097 | 0 | gather | 0.348350 | 0.343750 | 1.008436 [0.923827, 1.099370] |
| 1 | 65537 | 0 | host | 1.142498 | 1.177787 | 1.006914 [0.969037, 1.048031] |
| 1 | 65537 | 0 | default | 4.056650 | 4.289300 | 0.953684 [0.915747, 0.992161] |
| 1 | 65537 | 0 | gather | 4.894300 | 5.146600 | 0.930712 [0.795767, 1.093366] |
| 1 | 262143 | 0 | host | 4.686850 | 4.662991 | 0.996354 [0.958040, 1.035740] |
| 1 | 262143 | 0 | default | 16.712650 | 17.126350 | 0.963176 [0.922431, 1.005497] |
| 1 | 262143 | 0 | gather | 20.093900 | 19.383200 | 1.043429 [0.850842, 1.304225] |
| 1 | 262144 | 0 | host | 4.035250 | 2.819600 | 1.443523 [1.397273, 1.492504] |
| 1 | 262144 | 0 | default | 15.749750 | 13.872950 | 1.127210 [1.099851, 1.156971] |
| 1 | 262144 | 0 | gather | 25.435700 | 23.719150 | 1.069361 [0.860439, 1.328378] |
| 1 | 262145 | 0 | host | 4.471900 | 2.768100 | 1.596613 [1.536913, 1.658951] |
| 1 | 262145 | 0 | default | 15.819900 | 14.626450 | 1.106470 [1.064951, 1.151795] |
| 1 | 262145 | 0 | gather | 27.548200 | 24.564850 | 1.139444 [0.927801, 1.401641] |
| 1 | 524289 | 0 | host | 9.321800 | 4.478500 | 2.061523 [2.008526, 2.117837] |
| 1 | 524289 | 0 | default | 28.349500 | 24.494900 | 1.199705 [1.166562, 1.237640] |
| 1 | 524289 | 0 | gather | 39.099300 | 36.313400 | 1.087333 [0.826448, 1.399579] |
| 1 | 1048577 | 0 | host | 16.009950 | 7.044800 | 2.253734 [2.196935, 2.313579] |
| 1 | 1048577 | 0 | default | 48.864500 | 39.342750 | 1.244389 [1.224756, 1.264535] |
| 1 | 1048577 | 0 | gather | 68.601450 | 59.092000 | 1.237923 [1.037392, 1.487876] |
| 1 | 524289 | 1 | host | 7.567000 | 4.128100 | 1.862141 [1.799561, 1.935443] |
| 1 | 524289 | 1 | default | 25.272900 | 21.280250 | 1.171180 [1.153716, 1.189733] |
| 1 | 524289 | 1 | gather | 33.653000 | 30.175650 | 1.109674 [0.879117, 1.402447] |
| 1 | 524289 | 2 | host | 7.681650 | 4.145100 | 1.869669 [1.826260, 1.916882] |
| 1 | 524289 | 2 | default | 24.956550 | 21.023750 | 1.174656 [1.142333, 1.203876] |
| 1 | 524289 | 2 | gather | 35.370600 | 29.386250 | 1.214503 [0.952898, 1.534903] |
| 2 | 1 | 0 | host | 0.000067 | 0.000072 | 0.967941 [0.924163, 1.015147] |
| 2 | 1 | 0 | default | 0.023350 | 0.022750 | 1.016064 [0.894748, 1.157615] |
| 2 | 1 | 0 | gather | 0.130100 | 0.130850 | 0.978736 [0.926040, 1.033902] |
| 2 | 4097 | 0 | host | 0.047725 | 0.047977 | 0.995292 [0.989672, 1.000150] |
| 2 | 4097 | 0 | default | 0.154650 | 0.157150 | 0.997909 [0.961976, 1.034361] |
| 2 | 4097 | 0 | gather | 0.365500 | 0.377600 | 0.999162 [0.951265, 1.052866] |
| 2 | 65537 | 0 | host | 0.985853 | 0.996020 | 0.986897 [0.967326, 1.005415] |
| 2 | 65537 | 0 | default | 3.461500 | 3.695850 | 0.947815 [0.911303, 0.989751] |
| 2 | 65537 | 0 | gather | 4.554200 | 4.533750 | 0.949774 [0.812344, 1.113737] |
| 2 | 262143 | 0 | host | 3.802559 | 3.937500 | 0.964563 [0.947892, 0.980795] |
| 2 | 262143 | 0 | default | 14.055300 | 13.858150 | 1.028552 [1.008631, 1.052160] |
| 2 | 262143 | 0 | gather | 25.524150 | 31.945900 | 0.971810 [0.782800, 1.207026] |
| 2 | 262144 | 0 | host | 3.930350 | 2.761700 | 1.436090 [1.386418, 1.490080] |
| 2 | 262144 | 0 | default | 14.284950 | 13.250750 | 1.086557 [1.064446, 1.109372] |
| 2 | 262144 | 0 | gather | 32.194000 | 23.793400 | 1.084490 [0.860241, 1.356038] |
| 2 | 262145 | 0 | host | 3.863750 | 2.755400 | 1.471396 [1.407120, 1.556062] |
| 2 | 262145 | 0 | default | 14.132000 | 13.005700 | 1.091497 [1.063977, 1.120210] |
| 2 | 262145 | 0 | gather | 30.800300 | 25.834550 | 1.100074 [0.884675, 1.357093] |
| 2 | 524289 | 0 | host | 7.848700 | 4.122400 | 1.926335 [1.880889, 1.977362] |
| 2 | 524289 | 0 | default | 25.331600 | 21.573850 | 1.177969 [1.160790, 1.197371] |
| 2 | 524289 | 0 | gather | 36.942950 | 33.217100 | 1.119803 [0.914531, 1.360183] |
| 2 | 1048577 | 0 | host | 15.958900 | 7.221900 | 2.266266 [2.214355, 2.322956] |
| 2 | 1048577 | 0 | default | 48.231550 | 39.698750 | 1.221960 [1.198372, 1.245237] |
| 2 | 1048577 | 0 | gather | 68.719450 | 50.368800 | 1.195671 [1.001865, 1.435069] |
| 2 | 524289 | 1 | host | 7.562950 | 4.175400 | 1.834739 [1.786464, 1.881602] |
| 2 | 524289 | 1 | default | 25.246950 | 21.853000 | 1.155621 [1.136029, 1.176786] |
| 2 | 524289 | 1 | gather | 35.840900 | 33.474450 | 1.141512 [0.896382, 1.447404] |
| 2 | 524289 | 2 | host | 7.898400 | 4.192700 | 1.911976 [1.867219, 1.962473] |
| 2 | 524289 | 2 | default | 25.205000 | 21.820300 | 1.175333 [1.149974, 1.202082] |
| 2 | 524289 | 2 | gather | 33.922800 | 30.163350 | 1.174591 [0.910980, 1.518900] |

## Final measurements

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | host | 0.000075 | 0.000073 | 0.930844 [0.779180, 1.041083] |
| 1 | 1 | 0 | default | 0.022200 | 0.022350 | 0.990050 [0.914945, 1.071561] |
| 1 | 1 | 0 | gather | 0.155150 | 0.152950 | 1.028443 [0.986093, 1.072060] |
| 1 | 4097 | 0 | host | 0.058514 | 0.058894 | 0.982842 [0.956569, 1.007470] |
| 1 | 4097 | 0 | default | 0.191050 | 0.184900 | 1.006955 [0.964202, 1.051018] |
| 1 | 4097 | 0 | gather | 0.368100 | 0.362950 | 0.972772 [0.911603, 1.036366] |
| 1 | 65537 | 0 | host | 1.093580 | 1.129589 | 0.992336 [0.965978, 1.021972] |
| 1 | 65537 | 0 | default | 4.024800 | 4.040350 | 0.970244 [0.942529, 1.000172] |
| 1 | 65537 | 0 | gather | 4.501200 | 4.691450 | 0.917828 [0.800758, 1.051711] |
| 1 | 262143 | 0 | host | 4.600181 | 4.665219 | 0.995035 [0.966335, 1.023340] |
| 1 | 262143 | 0 | default | 16.327450 | 17.354300 | 0.978946 [0.936392, 1.022634] |
| 1 | 262143 | 0 | gather | 18.087050 | 17.305100 | 1.064041 [0.881280, 1.303812] |
| 1 | 262144 | 0 | host | 5.187850 | 2.937000 | 1.678393 [1.586834, 1.752368] |
| 1 | 262144 | 0 | default | 17.762200 | 15.497100 | 1.142533 [1.110986, 1.172112] |
| 1 | 262144 | 0 | gather | 17.052850 | 13.649100 | 1.199031 [1.024687, 1.403404] |
| 1 | 262145 | 0 | host | 5.198200 | 2.940300 | 1.714513 [1.672647, 1.755656] |
| 1 | 262145 | 0 | default | 17.723850 | 15.682150 | 1.120286 [1.092496, 1.150474] |
| 1 | 262145 | 0 | gather | 17.825850 | 16.311800 | 1.132757 [0.997459, 1.277795] |
| 1 | 524289 | 0 | host | 10.150500 | 4.461200 | 2.131399 [2.052428, 2.210511] |
| 1 | 524289 | 0 | default | 30.282700 | 24.829400 | 1.216022 [1.198948, 1.231940] |
| 1 | 524289 | 0 | gather | 28.704150 | 25.862600 | 1.232272 [1.089319, 1.398538] |
| 1 | 1048577 | 0 | host | 20.364350 | 8.003700 | 2.472368 [2.379419, 2.566397] |
| 1 | 1048577 | 0 | default | 57.499550 | 45.351750 | 1.254305 [1.224516, 1.284359] |
| 1 | 1048577 | 0 | gather | 53.546900 | 42.951250 | 1.318192 [1.165533, 1.491262] |
| 1 | 524289 | 1 | host | 9.576050 | 4.551000 | 2.092969 [2.020787, 2.163527] |
| 1 | 524289 | 1 | default | 30.170250 | 25.115100 | 1.207065 [1.178639, 1.236596] |
| 1 | 524289 | 1 | gather | 27.319650 | 23.459100 | 1.248796 [1.111348, 1.403683] |
| 1 | 524289 | 2 | host | 10.484800 | 4.517500 | 2.208679 [2.130894, 2.286222] |
| 1 | 524289 | 2 | default | 30.885200 | 24.715550 | 1.241243 [1.217760, 1.267591] |
| 1 | 524289 | 2 | gather | 27.868250 | 23.824500 | 1.275743 [1.071566, 1.519548] |
| 2 | 1 | 0 | host | 0.000103 | 0.000097 | 0.930465 [0.791676, 1.032937] |
| 2 | 1 | 0 | default | 0.035950 | 0.029300 | 1.028220 [0.901993, 1.164858] |
| 2 | 1 | 0 | gather | 0.192600 | 0.190100 | 1.010500 [0.975559, 1.046219] |
| 2 | 4097 | 0 | host | 0.075630 | 0.076356 | 1.030875 [0.968120, 1.118640] |
| 2 | 4097 | 0 | default | 0.313500 | 0.294800 | 0.938497 [0.837387, 1.038070] |
| 2 | 4097 | 0 | gather | 0.556650 | 0.513600 | 1.069311 [0.936855, 1.222203] |
| 2 | 65537 | 0 | host | 1.395472 | 1.375956 | 1.024723 [0.996680, 1.051670] |
| 2 | 65537 | 0 | default | 5.075150 | 5.280000 | 0.990527 [0.949429, 1.035226] |
| 2 | 65537 | 0 | gather | 5.219800 | 5.656550 | 0.929725 [0.827857, 1.053154] |
| 2 | 262143 | 0 | host | 4.877781 | 4.847708 | 1.008360 [0.974324, 1.043408] |
| 2 | 262143 | 0 | default | 16.499250 | 17.322750 | 0.978618 [0.948048, 1.011739] |
| 2 | 262143 | 0 | gather | 17.056850 | 16.391800 | 1.148596 [0.973896, 1.398033] |
| 2 | 262144 | 0 | host | 5.346550 | 3.063300 | 1.655021 [1.583679, 1.722743] |
| 2 | 262144 | 0 | default | 18.322400 | 15.794950 | 1.164481 [1.125820, 1.203855] |
| 2 | 262144 | 0 | gather | 17.883800 | 16.590100 | 1.118983 [0.951533, 1.312183] |
| 2 | 262145 | 0 | host | 5.584650 | 3.402550 | 1.673175 [1.623418, 1.721339] |
| 2 | 262145 | 0 | default | 19.576700 | 17.053350 | 1.142762 [1.102716, 1.184946] |
| 2 | 262145 | 0 | gather | 18.294750 | 17.155850 | 1.111837 [0.992809, 1.238424] |
| 2 | 524289 | 0 | host | 10.668050 | 5.011500 | 2.097285 [1.998579, 2.188930] |
| 2 | 524289 | 0 | default | 32.530450 | 27.054700 | 1.208125 [1.172015, 1.238993] |
| 2 | 524289 | 0 | gather | 32.230700 | 26.077950 | 1.238199 [1.102393, 1.387793] |
| 2 | 1048577 | 0 | host | 21.704650 | 8.806700 | 2.443958 [2.364337, 2.523004] |
| 2 | 1048577 | 0 | default | 61.613250 | 49.416500 | 1.281508 [1.236797, 1.337740] |
| 2 | 1048577 | 0 | gather | 56.663150 | 43.378650 | 1.298021 [1.199059, 1.408284] |
| 2 | 524289 | 1 | host | 10.151000 | 4.734650 | 2.068835 [1.989693, 2.147642] |
| 2 | 524289 | 1 | default | 32.068300 | 26.602000 | 1.206366 [1.172236, 1.240066] |
| 2 | 524289 | 1 | gather | 29.359700 | 22.338750 | 1.287031 [1.169168, 1.413613] |
| 2 | 524289 | 2 | host | 11.710600 | 5.228900 | 2.168381 [2.087671, 2.252147] |
| 2 | 524289 | 2 | default | 36.855300 | 29.514850 | 1.214915 [1.182767, 1.246008] |
| 2 | 524289 | 2 | gather | 29.375350 | 23.564500 | 1.224719 [1.097484, 1.367241] |
