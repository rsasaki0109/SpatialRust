# Fused FPFH GPU input gathering

Date: 2026-10-14. Parent: `980bb6853538be7db43254c8e30312f7a0fdf29f` (merged PR 200).

## Change and boundary

Below 1,048,576 points, `GpuFpfhInput::upload_with_gpu_gather` generates all three device input
outputs in one integer pass: traversal-order tree points, original-order padded
queries and padded position/normal samples. Borrowed XYZ columns followed by
raw original-order normals are written directly into one mapped 24*N-byte GPU
input, beside the existing 4*N-byte original-index permutation. No host
concatenation or point-sized packing vector is added. All coordinate/normal
words and zero padding are copied as u32, preserving signs and subnormals.

Larger inputs retain the previous two-pass path. That path first generated tree points/queries, then uploaded raw normals
and generated samples from resident queries. Fusion removes that second compute
pass, encoder/submission, uniform buffer and query-buffer reads, and reduces
temporary payload inputs from three buffers to two. The sample output is still
owned and 32*N bytes; tree/query output buffers remain 16*N each. All paths keep
32*M-byte nodes packed on CPU.

Host payload remains 28*N + 32*M bytes; device output receipt remains 64*N and
device-to-host bytes remain zero. Tree/query metadata retain their existing
receipts, while the FPFH constructor separately records 12*N normal input and
32*N sample output bytes. Uniform controls are excluded. Fusion changes command
and buffer creation overhead, not payload accounting. There is no explicit host
completion wait, readback or CPU fallback. Queue submission retains both inputs;
CPU trees, coordinate/normal storage and returned GPU inputs can be dropped
before owned results are inspected.

Two fused inputs occupy 28*N device bytes throughout the pass, compared with
up to 28*N total temporary inputs over the previous two submissions. This is
not a claimed reduction in peak GPU memory. Backend staging/allocator overhead
is additional. Previously eliminated host packing allocations totaling 64*N
payload bytes remain eliminated; that cumulative sum is not peak host savings.

CPU partitioning, finite/unit-normal/extent validation, original f64 extent
arithmetic and serial node staging remain. A third lazy pipeline variant uses
the existing FPFH requirement of five storage bindings, six total bindings and
64 lanes. The private sample planner checks count, full query dispatch and
32*N-byte output before allocation; that bound also covers 24*N input and
16*N outputs. Existing public APIs, default FPFH construction, tree-only/paired
entry points, normal error precedence and descriptor algorithms are preserved.
No dependency, unsafe boundary or core change is introduced.

The previous FPFH sample helper and pipeline cache remain for large inputs.
The frozen two-pass constructor is native-test-only. A source comparison folds the parent's GPU
branch and checks this frozen reference, including validation and complete
receipt/ownership construction. The parent's ordinary FPFH branch, public tree
upload/search methods, original tree/query shader bodies and node construction
are also compared after formatting/comment normalization.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/nearest.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/tree_gpu_gather.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/tree_point_gather.wgsl`

## Correctness and integration

New private fused-input probes compare tree points and nodes against default
uploads and independently check original-order queries, samples and padding.
Sizes include empty, one point, 65, 65,537 and 1,048,577; profiles include
bounded varied coordinates, zero signs, subnormals and arbitrary finite words.
Malformed private shapes reject before cache initialization. Empty inputs
allocate no buffers/pipeline; fused-only calls initialize only the third
variant. CPU trees and normals are dropped before complete raw-word readback.

Public FPFH probes compare the fused path against frozen two-pass upload and
independent coordinate/normal/padding bits, through 524,289 points. Unit normals
include axis directions, 0.6/0.8 components, signed zeros and signed subnormals.
Point counts and all transfer directions remain exact. Descriptor/status probes
include capacity overflow, numerical underflow, foreign runtime and input-bound
graph rejection, complete download outcomes and GPU input drop before results.
Nonfinite XYZ, overflowing extent, normal length/unit/finite errors match the
parent before pipeline initialization, in a clean validation scope. Concurrent
cold small callers retain independent buffers; the old sample cache stays uninitialized.
Cutoff-adjacent public tests at 1,048,575/1,048,576/1,048,577 points check actual
sample-cache selection, complete query/sample bits and identical receipts.

Four focused native release tests passed, including cutoff-adjacent public
construction. Raw public probes compare 28,333,248 query/sample words and 33,708
descriptor/status words against frozen two-pass upload. Private fused probes
compare 75,764,352 tree/node/query/sample words, including arbitrary finite bits
through 1,048,577 points and the fused-only lazy variant. CPU shape/size planners,
15 frozen public errors and cold concurrent calls are covered.

Related release groups passed 56 tests: 32 nearest/tree/query, seven previous
feature-input regressions, three FPFH, four ISS, four MLS and six registration
geometry/SoA cases. Eleven public facade tests passed across feature inputs,
high-level FPFH/ISS/MLS, neighborhoods, ICP/trimmed ICP, GICP and point-to-plane.
All-feature/all-target GPU Clippy with warnings denied, CPU-only, WASM,
formatting and diff checks passed. Source receipts pin 317 source/shader/
executable files and verify 306 preexisting GPU source files unchanged,
including 96 shader files; only the existing tree-gather shader was extended.
Default CPU FPFH, public tree upload/search, old tree/query shader entry bodies,
node construction and the frozen two-pass reference match the parent after
formatting/comment normalization.

Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --features gpu-image --release --lib fpfh_fused -- --include-ignored --skip benchmark --nocapture --test-threads=1
cargo test -p spatialrust-gpu --features gpu-image --release --lib kernels::nearest -- --include-ignored --skip benchmark --skip survey_bounded_host_packing --skip survey_bounded_host_tree_packing --skip completed_cpu_and_gpu_tree_uploads --nocapture --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --no-default-features
cargo check -p spatialrust --no-default-features --features register-fpfh-gpu,feature-iss-gpu,filter-mls-gpu,register-icp-gpu,register-gicp-gpu,register-icp-point-to-plane-gpu --target wasm32-unknown-unknown
```

Local source/executable hashes, raw samples and gate receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/fpfh-fused-*`.

## Completed-upload measurement

The ignored `completed_two_pass_and_fused_fpfh_uploads` test compares the
frozen two-pass constructor with the current constructor over ten size/profile
cases. Routes alternate over three warmup and 30 recorded rounds. CPU tree/
fixture creation and runtime setup are outside timing. Each interval includes
limits, finite/unit-normal/extent validation, node staging, all device buffer
creation and compute submissions, `wait_idle`, receipt/count observation and
input drop. Assertions follow timing. Warm-pipeline measurements exclude first
shader compilation, searches, descriptors and complete application latency.

Profiles are ordered grid with small Z variation (0), alternating zero signs
(1), signed subnormals (2), and the same grid with shuffled original order (4).
The xorshift64/Fisher-Yates seed is `0x0123456789abcdef`, with point sets preserved
between ordered/shuffled pairs at 524,289 and 1,048,577 points. All routes use
the same varied unit normals. Unused fixture vectors are dropped before timing.

Two surveys run sequentially with a quiet window and continuous build/test/
benchmark overlap monitoring. Contaminated attempts would be discarded and
only the owned benchmark child stopped. Statistical analysis begins after both
accepted surveys finish. All adverse and inconclusive comparisons are retained.

An unconditional-fusion candidate regressed the shuffled 1,048,577-point case
in both isolated surveys. Its complete 20 comparisons, sources, executable and
receipts are archived at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/fpfh-fused-candidate1/`.
The final policy retains two passes at and above 1,048,576 points. This preserves
the parent's large-input behavior rather than claiming a universal device-size
threshold. The benchmark field/route `fused` identifies the current API: final
million-point cases use retained two-pass construction. Candidate and final
results are recorded separately and are not pooled.


Final environment: native release, `rustc 1.98.1 (48a229cea 2026-09-01)`, Vulkan
discrete-GPU backend, 12 available CPU workers (at most four for shared large
point/query staging/finite scans). Storage binding limit is 2,147,483,647 bytes;
dispatch-axis limit is 65,535 groups. Fused FPFH requires five storage/six total
bindings and 64 lanes, as already required by FPFH. Both final accepted attempts
were number 1, with no overlap and empty stderr; durations were 10.86 and
10.90 seconds. Source/executable hashes matched before and after both surveys.
No local adapter model is recorded.

Ratios are paired geometric means of previous/current duration; values above
one mean shorter current calls. Pointwise 95% intervals use 10,000 paired
log-ratio bootstrap resamples, seed `2026101431 + 100*survey + 10*case + field_index`
(JSON record field index), sorted entries 249/9749. Every comparison has 30
samples. Zero samples would suppress ratios and retain medians/counts; there
were none. Candidate/final results use separate source/executable pairs.

Representative final completed-upload medians:

- survey 1, 1 ordered points: 0.218150 to 0.148500 ms (31.9% median reduction); ratio 1.544323 [1.428166, 1.675028].
- survey 1, 4,097 ordered points: 0.324100 to 0.235550 ms (27.3% median reduction); ratio 1.330093 [1.263077, 1.399773].
- survey 1, 524,289 ordered points: 15.835050 to 14.043550 ms (11.3% median reduction); ratio 1.120604 [1.078376, 1.159607].
- survey 2, 1 ordered points: 0.265450 to 0.170100 ms (35.9% median reduction); ratio 1.645070 [1.536376, 1.757187].
- survey 2, 4,097 ordered points: 0.360750 to 0.227050 ms (37.1% median reduction); ratio 1.563344 [1.470494, 1.658749].
- survey 2, 524,289 ordered points: 15.730750 to 14.901100 ms (5.3% median reduction); ratio 1.065974 [1.030000, 1.101847].

For 4,097 points, both intervals are above one and medians improve about
27–37%. For 524,289 ordered points, both intervals are above one and medians
improve about 5–11%. One-point calls mainly show reduced fixed overhead. At
65,537 points both final intervals cross one; no stable gain is established.
The measured 262,145-point and shuffled 524,289-point cases have intervals above
one in both final surveys, but these results do not establish a universal
threshold or complete FPFH descriptor/application acceleration. Default CPU
input preparation can remain faster for small batches; explicit GPU placement
is unchanged.

Million-point final cases retain the same two-pass commands and payload as the
parent. Ordered cases have intervals crossing one in both surveys; shuffled
cases show a small gain in survey 1 and are inconclusive in survey 2. No large-
input latency gain is claimed. The policy preserves the earlier path after
unconditional fusion regressed this case in both candidate surveys. The
selected cutoff is conservative, not proof that every unmeasured sub-cutoff
size, distribution or backend improves. All 0 final and 2 candidate
comparisons whose entire interval is below one, and every inconclusive result,
are retained below. Payload receipts and result semantics remain exact.

### Final adaptive constructor

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | fused | 0.218150 | 0.148500 | 1.544323 [1.428166, 1.675028] |
| 1 | 4097 | 0 | fused | 0.324100 | 0.235550 | 1.330093 [1.263077, 1.399773] |
| 1 | 65537 | 0 | fused | 1.969400 | 2.023750 | 1.019862 [0.974882, 1.065181] |
| 1 | 262145 | 0 | fused | 9.945200 | 9.078650 | 1.065319 [1.029363, 1.100515] |
| 1 | 524289 | 0 | fused | 15.835050 | 14.043550 | 1.120604 [1.078376, 1.159607] |
| 1 | 524289 | 1 | fused | 17.412050 | 15.403700 | 1.110858 [1.073662, 1.148658] |
| 1 | 524289 | 2 | fused | 17.712550 | 16.963650 | 1.032406 [0.995280, 1.069420] |
| 1 | 1048577 | 0 | fused | 26.179400 | 26.179700 | 0.992783 [0.950734, 1.034729] |
| 1 | 524289 | 4 | fused | 17.485800 | 16.404550 | 1.068464 [1.026777, 1.108746] |
| 1 | 1048577 | 4 | fused | 28.082800 | 27.312450 | 1.033918 [1.003159, 1.066006] |
| 2 | 1 | 0 | fused | 0.265450 | 0.170100 | 1.645070 [1.536376, 1.757187] |
| 2 | 4097 | 0 | fused | 0.360750 | 0.227050 | 1.563344 [1.470494, 1.658749] |
| 2 | 65537 | 0 | fused | 2.050550 | 2.061300 | 1.026735 [0.973962, 1.083951] |
| 2 | 262145 | 0 | fused | 10.198650 | 9.928450 | 1.035525 [1.002585, 1.070486] |
| 2 | 524289 | 0 | fused | 15.730750 | 14.901100 | 1.065974 [1.030000, 1.101847] |
| 2 | 524289 | 1 | fused | 17.470250 | 16.131100 | 1.115798 [1.078938, 1.153492] |
| 2 | 524289 | 2 | fused | 16.207850 | 15.378000 | 1.063239 [1.026738, 1.102043] |
| 2 | 1048577 | 0 | fused | 25.925100 | 26.135850 | 1.002202 [0.970522, 1.035927] |
| 2 | 524289 | 4 | fused | 16.983100 | 15.898650 | 1.048979 [1.019328, 1.077145] |
| 2 | 1048577 | 4 | fused | 26.807750 | 26.451900 | 1.002975 [0.973923, 1.032319] |

### Rejected unconditional-fusion candidate (separate source/executable)

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | fused | 0.233550 | 0.131550 | 1.687254 [1.557194, 1.818476] |
| 1 | 4097 | 0 | fused | 0.305200 | 0.209150 | 1.399760 [1.337712, 1.464468] |
| 1 | 65537 | 0 | fused | 1.970650 | 1.874000 | 0.990046 [0.951518, 1.030114] |
| 1 | 262145 | 0 | fused | 9.433200 | 9.067200 | 1.016350 [0.984649, 1.050285] |
| 1 | 524289 | 0 | fused | 14.550450 | 13.056100 | 1.095377 [1.075725, 1.115461] |
| 1 | 524289 | 1 | fused | 13.296400 | 12.385200 | 1.069533 [1.042383, 1.098392] |
| 1 | 524289 | 2 | fused | 13.112850 | 12.799000 | 1.027893 [1.006232, 1.050288] |
| 1 | 1048577 | 0 | fused | 21.494050 | 21.125250 | 0.986266 [0.932122, 1.029180] |
| 1 | 524289 | 4 | fused | 13.517200 | 13.554550 | 1.004626 [0.983412, 1.026293] |
| 1 | 1048577 | 4 | fused | 22.106850 | 23.118400 | 0.945384 [0.923992, 0.967044] |
| 2 | 1 | 0 | fused | 0.238350 | 0.134250 | 1.794826 [1.710416, 1.878917] |
| 2 | 4097 | 0 | fused | 0.296850 | 0.227550 | 1.325652 [1.277797, 1.375311] |
| 2 | 65537 | 0 | fused | 2.070000 | 2.088450 | 1.007894 [0.955648, 1.062816] |
| 2 | 262145 | 0 | fused | 9.287450 | 8.991700 | 1.041988 [1.018683, 1.065604] |
| 2 | 524289 | 0 | fused | 16.456350 | 14.882750 | 1.106030 [1.074966, 1.137699] |
| 2 | 524289 | 1 | fused | 15.275200 | 14.092900 | 1.097229 [1.065693, 1.129816] |
| 2 | 524289 | 2 | fused | 15.566400 | 14.584950 | 1.041146 [1.010181, 1.073320] |
| 2 | 1048577 | 0 | fused | 23.851500 | 23.993450 | 0.999910 [0.977687, 1.024718] |
| 2 | 524289 | 4 | fused | 15.796800 | 15.506750 | 1.016122 [0.991864, 1.041738] |
| 2 | 1048577 | 4 | fused | 24.852800 | 25.943650 | 0.966294 [0.946989, 0.987555] |
