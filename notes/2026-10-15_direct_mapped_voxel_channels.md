# Direct mapped voxel channel readback

Standalone multi-channel first-value gather and average APIs used to read
all output planes into one flat float vector, then split off later planes.
The first returned channel retained the complete batch capacity, and every
other channel allocated and copied its plane again.

The two-/four-channel gather dispatchers and aggregate average call now use
one private mapped-plane decoder and the existing unmap guard. Each plane
copies directly into its final owned vector. A nonempty K-channel batch with
N output cells requests 4*K*N payload bytes instead of 4*(2*K-1)*N; the
4*(K-1)*N difference removes both repeated-copy bytes and excess retained
first-channel capacity. There are still K float-vector allocations, plus the
outer vector. GPU staging/output and complete payload transfers remain.

Gather retains its adapter-selected batch width. Four channels at 1,048,576
cells remove 12,582,912 bytes (12 MiB) if gathered as one four-channel batch,
or 8,388,608 bytes (8 MiB) across two two-channel batches. The four-channel
average has one aggregate readback and removes 12 MiB independently of gather
capabilities. A one-channel batch retains its existing payload requirement.
These requested allocation/capacity counts are not RSS or peak-memory claims.

Public signatures, channel order, empty returns, ordered length validation,
host inputs, input uploads, device buffers, uniforms, compute/copy commands,
submissions, map/wait, error messages and shader math are preserved. The
mapped view lives through final output copies and drops before unmapping on
return or unwind. No mapped reference escapes. Combined XYZ/attribute APIs,
u8 channels and scalar/XYZ-only decoders retain their existing implementations.
No dependency, unsafe code, GPU pass, transfer or synchronization is added.

## Verification method

The complete parent public gather/average calls, gather dispatcher/two/four
channel paths, dummy-buffer constructor, float mapper and channel splitter
are frozen from merge `0982ff93fe33776a2e771f7c89d758349ef8a0f3`. Their uniform
structures and dispatch constants are frozen as well. Only names and test
visibility change. Production differences are limited to the new decoder,
three final readback call sites/imports and a native test module declaration.
All other GPU/image sources and WGSL files must match the parent.

Raw staged storage covers 0/1/2/3/4/7/11 channels and 0/1/17/65,537/262,145
cells, plus trailing padding. Signed zeros, NaN payloads, infinities,
subnormals, extrema and deterministic mixed u32 encodings must retain their
exact bits through direct and frozen decoding. Tests check lengths, requested
capacities, repeated maps, independently mutated live outputs and dropped GPU
staging. An intentional short-buffer decoder panic must unmap and permit
subsequent direct and original mappings.

Complete public calls cover 0/1/2/3/4/7 channels, 0/1/17/257/65,537 cells
and one/four scattered input points per cell. Finite dyadic values support
an independent exact CPU first/mean oracle. The 257-cell fixtures also build
segments with the public GPU sorter; other fixtures explicitly upload valid
group metadata. Frozen results, channel ordering and backend-dependent batch
capacities are checked. Empty inputs, empty-segment validation precedence,
exact length errors before resource use, repeated calls, changed host inputs,
dropped inputs, pool clears and concurrent calls retain owned outputs.

## Measurement method

The benchmark times the complete host-input public gather/average API,
including per-call input uploads, output/staging/uniform creation,
compute/copy submissions, map/wait, host decoding and owned return. CPU
source/expected-value generation, the slice-reference array, segment setup,
output assertions and final output drop are outside timing. It is not GPU
pass time or complete voxel downsampling with key generation/segmentation.

Eight fixed conditions use (cells, channels): (16,2), (16,4), (4096,4),
(262144,2), (262144,4), (262144,7), (1048576,2), (1048576,4), all with four
source points per cell. Both routes use three warmups and 30 alternating
paired samples. Every output word and returned capacity must match the
independent expected values. All favorable, adverse and inconclusive rows
remain in the report.

Two isolated surveys use bounded three-minute quiet scheduling windows with
before/during/after competing-work checks. Overlapping/incomplete trials are
discarded entirely, and other projects' work is left running. If a scheduling
window expires, one fresh bounded retry may be used to obtain replication.
Source/executable hashes are captured after tests and checked after attempts.
Paired log old/new ratios use 10,000 bootstrap samples, sorted indices 249/9749
for 95% intervals, with seed
2026101521 + 100*survey + 10*condition_index + JSON_field_index. Ratios above
one favor direct decoding; intervals crossing one are inconclusive. A zero
sample suppresses that row's ratio. Geometric paired ratios differ from ratios
of reported medians. Raw validation/timing/hash receipts remain local under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/` with the
`voxel-channels-` prefix. Published environment descriptions omit adapter names.

## Completed checks and measurements

All 40 native GPU voxel regressions and 18 voxel-filter regressions passed
(58 distinct local tests). The new matrices made 27,526,800 raw float-word
and 13,425,648 frozen/analytic/repeated public-output comparisons. Additional
checks covered updated host inputs, dropped sources, empty/length validation,
channel order, backend-batched capacities, live outputs, pool clears,
concurrent calls and caught panic/remap. Both prior XYZ regressions and the
new channel regressions remain in the native selection.

All-feature/all-target GPU Clippy with warnings denied, CPU-only builds,
minimal `gpu-wgpu` and CI `gpu-image,gpu-aoso-staging` WASM builds with
warnings denied, formatting and diff checks passed.

358 source/executable hashes were pinned after validation and verified after
measurement attempts. Eight frozen complete parent functions, two uniforms
and dispatch constants match their originals. 350 other existing GPU/image
sources and all 97 WGSL files are unchanged. The shared unmap guard and the
original map/wait/diagnostic path are preserved. Privacy scans found no
forbidden adapter-name matches.

Isolated latency unavailable due competing work; no speed claim.

One additional non-isolated functional invocation completed all eight fixed conditions and both public routes, including 1,048,576 cells/four channels. Every warmup and paired output/capacity assertion passed. Its timing arrays are excluded entirely from performance evidence; it validates large outputs and the requested-capacity reduction only. This is separate from the 58 regression tests above.

Survey 1 had no accepted trial across 1 three-minute scheduling window(s); 0 overlapping trial(s) are discarded in full. None of its timings are used.

Survey 2 had no accepted trial across 1 three-minute scheduling window(s); 0 overlapping trial(s) are discarded in full. None of its timings are used.

No isolated completed timing evidence is available. Source/capacity savings are established independently; no speedup is claimed.

