# Direct parallel GPU neighborhood row decoding

Date: 2026-10-19. Named slice family: direct parallel neighborhood rows.
Workspace: `C:/Users/rsasa/Workspace/SpatialRust`.
Parent: `331adc0f776aa3b4fbcbda72d42a428147a2f269` (merged PR #286).

## Decision and limitations

Defer production adoption. The candidate moves metadata validation and owned
neighbor collection into each existing parallel band's row loop, removing the
private per-row Result-returning helper. The initial two surveys did not establish
repeatable complete upload/search/download improvement: both favorable full-call
conditions in survey one were inconclusive in survey two. CPU controls also
showed adverse cases. Production therefore retains the complete parent decoder.
The candidate and full-download wrapper are reachable only from native test
modules and remain available for future measurement with the regressions below.
This is CPU work after explicit GPU readback, not an additional device algorithm.
Measurements do not establish a compiler/inlining explanation for differences.

| Phase | Scope | Policy | Favorable | Inconclusive | Adverse | Unresolved | Old/new ratio range |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| initial | cpu | parallel rows | 4 | 13 | 1 | 0 | 0.9427–1.1362 |
| initial | cpu | original control | 0 | 7 | 5 | 0 | 0.8883–1.0280 |
| initial | download | parallel rows | 4 | 14 | 0 | 0 | 0.9743–1.0633 |
| initial | download | original control | 0 | 12 | 0 | 0 | 0.9556–1.0242 |
| initial | full | parallel rows | 2 | 12 | 0 | 0 | 0.9765–1.0603 |
| initial | full | original control | 0 | 6 | 0 | 0 | 0.9837–1.0065 |
| final-test-only | cpu | parallel rows | 4 | 14 | 0 | 0 | 0.9604–1.0643 |
| final-test-only | cpu | original control | 1 | 11 | 0 | 0 | 0.8734–1.0994 |
| final-test-only | download | parallel rows | 1 | 17 | 0 | 0 | 0.9781–1.0167 |
| final-test-only | download | original control | 1 | 11 | 0 | 0 | 0.9606–1.0920 |
| final-test-only | full | parallel rows | 1 | 13 | 0 | 0 | 0.9752–1.0349 |
| final-test-only | full | original control | 0 | 6 | 0 | 0 | 0.9798–1.0091 |

initial, cpu: 5 adverse / 12 controls; worst paired old/new 0.8883 (12.6% longer
time).

initial, download: 0 adverse / 12 controls; worst paired old/new 0.9556 (4.6% longer
time).

initial, full: 0 adverse / 6 controls; worst paired old/new 0.9837 (1.7% longer
time).

final-test-only, cpu: 0 adverse / 12 controls; worst paired old/new 0.8734 (14.5%
longer time).

final-test-only, download: 0 adverse / 12 controls; worst paired old/new 0.9606
(4.1% longer time).

final-test-only, full: 0 adverse / 6 controls; worst paired old/new 0.9798 (2.1%
longer time).

Every initial and final native comparison is retained below, including adverse
and inconclusive results. Individual intervals have no multiple-comparison
adjustment. These are
synthetic shared-host observations, not a portable application speed guarantee
or a competitor performance comparison. Unchanged controls can still vary with
compiler layout, surrounding code and host contention.

## Schema and resource proof

Each row retains `(stored_count,total_count,(index,distance_bits)*capacity)`.
Before allocation or output assignment, reject `stored_count > capacity` or
`total_count < stored_count`; nearest-k also rejects unequal counts. The error
type and message are unchanged. Valid rows collect exactly the stored prefix,
convert u32 indices to usize and reconstruct each f32 from its original bits,
then assign the exact total. Unused padded words remain unread. Row order,
NaNs/infinities/signed zeros, arbitrary index bits, radius completeness and
overflow behavior therefore retain their original semantics.

In both candidate and parent, the same safely initialized final row/count slots
are divided into disjoint bands. At most four workers participate per call,
including the caller; this
does not impose a process-wide concurrency limit. All scoped workers join before
the mapped borrow ends, including on malformed rows. No extra worker, allocation,
intermediate row vector, synchronization or unsafe boundary is introduced.
Serial small/sparse/one-worker and WASM behavior is unchanged, as is the existing
large zero-capacity count-only decoder. No new runtime configuration is added.

GPU uploads/search/shaders, staging/copy/submit/map/wait/unmap, explicit transfer
receipts, public APIs, manifests and dependencies are unchanged. Owned CPU
outputs survive source input drops and pool clears. Runtime mismatch and map
error handling retain the public downloader's original boundaries.

## Correctness and validation

Three new focused tests compare the complete frozen parent with the test-only
candidate and independent packed-word oracles. They cover empty/small/cutoff/odd/large
shapes, sparse and concentrated rows, capacities through 65, all forced worker
counts including unavailable and oversized requests, arbitrary index/distance
bits and poisoned padding. Malformed stored/total headers are placed at each
worker boundary and at the first/middle/last rows; failed GPU maps are repaired
and downloaded again. Concurrent downloads, runtime mismatch, ownership after
source drops/pool clears, exact receipt bytes and unchanged completeness are
checked. Existing count-only, public search, overflow and CPU-oracle tests remain.

Final related coverage has 61 native test executions: 35 neighborhood-family,
8 covariance, 4 ISS, 4 MLS, 3 FPFH and 7 facade executions. Focused precheck and
benchmark invocations are separate. CPU-only and both WASM configurations compile;
WASM GPU execution is not claimed. Clippy denies warnings for all features and
targets. Both WASM warning checks run before the native release precheck.

| Check | Result | Executions or scope |
| --- | --- | ---: |
| clippy | pass | build/lint |
| nearest | pass | 35 |
| gicp | pass | 8 |
| iss | pass | 4 |
| mls | pass | 4 |
| fpfh | pass | 3 |
| facade | pass | 7 |
| cpu-only | pass | build/lint |
| wasm-minimal | pass | build/lint |
| wasm-ci | pass | build/lint |
| fmt | pass | build/lint |
| diff | pass | build/lint |

The production decoder matches the merged parent byte for byte after line-ending
normalization. The only change to an original Rust source is the focused native
test registration. The frozen complete parent decoder excludes its original
test registration.
The full parent downloader differs only in receiver binding and delegation to
that frozen decoder. The test-only candidate matches the archived initial
production candidate except for comments and visibility of its forced-worker
entry point. The candidate downloader differs only in decoder delegation.
The initial audit reconstructs the candidate by deleting only the old private
row helper and inserting its unchanged body into the parallel loop.
All 696 other original Rust/WGSL sources in GPU/search/image/core
match after line-ending normalization. All 736 source/manifest/executable
SHA-256 values match across final measurements and validation. Initial source,
executable and all evidence are retained separately with checked hashes; source
changed once between the initial phase and the final test-only phase.
Measured native executable SHA-256: `4aba4ca613e8c0479fcc4e5aa1721707853c35236f7da9a7eb26b907a52dec18`.
Tracked-source and PR-body scans found zero local-adapter-name matches.

## Native measurement method

Toolchain: `rustc 1.98.1 (48a229cea 2026-09-01)`; `cargo 1.98.1 (797e8a9bc 2026-08-05)`;
host `Windows/AMD64`, native release profile with thin LTO and one codegen unit.

Each of the initial and final phases has two surveys that measure 15 packed
payloads through CPU conversion and complete explicit download, plus 10 complete
tree/query upload, GPU search and download cases: all 160 route comparisons
across both phases. No selected reruns or dropped rows. Each variant has three
warmups and 30 paired samples. Four CPU/download variants use balanced
four-position order; the two full variants alternate. Selectors, validation,
worker creation/join, host allocations and unmap are timed. Full calls also
include both uploads and GPU search; CPU tree building, fixture generation,
result checking and output drops are outside both timers. Both full variants
use the same unchanged production upload/search code and differ only in their
parent/candidate decoder delegation through complete download implementations.
The initial phase compares frozen parent code against the production candidate.
After deferral, final measurements compare full frozen parent and candidate
test-only implementations in the final executable; they do not time a new
production optimization. Phase results are reported separately and not pooled.

Packed inputs include dense parallel nearest/radius rows, capacity-65 padding,
large capacity-one rows, cutoff neighbors and small/sparse/count-only controls.
Full calls use synthetic targets of 17 and 4,097 unique grid points with repeated
exact queries, nearest-k capacities eight/sixteen and infinite-radius rows.
The zero-capacity infinite-radius control counts every target point with the
existing unchanged count-only fast path. These are not sensor/application data.

Ratios are exponential means of paired log old/new times. Individual 95% paired
bootstrap intervals use 10,000 resamples, sorted indices 249/9749, and seeds
`seed + 1000*survey + 100*(full_job) + 3*case + route_index`, with seed 2026101995
for initial and 2026101997 for final. No trimming,
filtering, epsilon for zero samples or multiple-comparison adjustment is used.
Zero samples leave a comparison unresolved. An interval wholly above/below one
is favorable/adverse; other resolved intervals are inconclusive.

- initial, survey 1, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 worker_creation_and_join_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- initial, survey 1, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- initial, survey 2, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 worker_creation_and_join_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- initial, survey 2, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- final-test-only, survey 1, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 test_only_candidate=true worker_creation_and_join_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- final-test-only, survey 1, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 test_only_candidate=true tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- final-test-only, survey 2, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 test_only_candidate=true worker_creation_and_join_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- final-test-only, survey 2, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 test_only_candidate=true tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.

Pre/post process CPU and anonymized GPU utilization samples cannot prove isolation
during every pair. Own builds/tests and bootstrap calculations did not overlap
timed measurements. Backend/device type and available workers are recorded;
local adapter name is excluded. No process/driver cache or clock reset is claimed.

Run the retained benchmark family alone with
`cargo test -p spatialrust-gpu --all-features --release --lib benchmark_neighborhood_row_ -- --ignored --nocapture --test-threads=1`.
This executes both packed and complete-search cases and prints every paired
sample; the published tables retain both phases rather than selecting a rerun.

## Every native comparison

| Phase | Survey | Scope | Points | Queries | Capacity | Stored | Kind | Policy | Old median ms | New median ms | Ratio | Individual paired 95% interval |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| initial | 1 | cpu | payload | 1025 | 8 | 8 | nearest | original | 0.062100 | 0.068000 | 0.888305 | 0.824447–0.952941 |
| initial | 1 | download | payload | 1025 | 8 | 8 | nearest | original | 1.742550 | 1.809150 | 0.980687 | 0.935113–1.033103 |
| initial | 1 | cpu | payload | 65535 | 8 | 8 | nearest | original | 9.214800 | 10.124600 | 0.925276 | 0.857426–0.997479 |
| initial | 1 | download | payload | 65535 | 8 | 8 | nearest | original | 13.685150 | 13.300950 | 0.979812 | 0.929942–1.033602 |
| initial | 1 | cpu | payload | 65536 | 8 | 8 | nearest | parallel | 4.989150 | 5.189400 | 0.962339 | 0.923225–0.998087 |
| initial | 1 | download | payload | 65536 | 8 | 8 | nearest | parallel | 9.190550 | 8.896500 | 1.015452 | 0.970243–1.058441 |
| initial | 1 | cpu | payload | 65537 | 8 | 8 | nearest | parallel | 4.460750 | 4.395800 | 1.018148 | 0.977011–1.064083 |
| initial | 1 | download | payload | 65537 | 8 | 8 | nearest | parallel | 8.475300 | 7.928050 | 1.063328 | 1.023025–1.106257 |
| initial | 1 | cpu | payload | 65537 | 16 | 16 | nearest | parallel | 5.091750 | 5.263550 | 0.983859 | 0.926292–1.045788 |
| initial | 1 | download | payload | 65537 | 16 | 16 | nearest | parallel | 10.984750 | 10.522500 | 1.057090 | 1.015637–1.102609 |
| initial | 1 | cpu | payload | 65537 | 64 | 64 | nearest | parallel | 10.112150 | 9.698600 | 1.072356 | 0.996236–1.152974 |
| initial | 1 | download | payload | 65537 | 64 | 64 | nearest | parallel | 20.138250 | 20.876450 | 1.004096 | 0.937722–1.080294 |
| initial | 1 | cpu | payload | 131073 | 8 | 8 | nearest | parallel | 6.514250 | 5.827750 | 1.084583 | 1.024170–1.155452 |
| initial | 1 | download | payload | 131073 | 8 | 8 | nearest | parallel | 11.775650 | 11.986800 | 1.011510 | 0.970839–1.064349 |
| initial | 1 | cpu | payload | 524289 | 1 | 1 | nearest | parallel | 18.838750 | 19.002550 | 1.000646 | 0.962558–1.036679 |
| initial | 1 | download | payload | 524289 | 1 | 1 | nearest | parallel | 25.133050 | 23.737700 | 1.056635 | 1.004291–1.111150 |
| initial | 1 | cpu | payload | 65537 | 65 | 9 | radius | parallel | 4.819250 | 4.385150 | 1.092016 | 1.030498–1.151062 |
| initial | 1 | download | payload | 65537 | 65 | 9 | radius | parallel | 15.600350 | 15.478850 | 1.021391 | 0.977605–1.068720 |
| initial | 1 | cpu | payload | 65537 | 65 | 65 | radius | parallel | 10.145750 | 10.132250 | 1.022342 | 0.981110–1.065684 |
| initial | 1 | download | payload | 65537 | 65 | 65 | radius | parallel | 21.718000 | 21.422200 | 1.003813 | 0.954519–1.054355 |
| initial | 1 | cpu | payload | 65537 | 16 | 16 | radius | parallel | 4.163850 | 4.314750 | 0.997385 | 0.949569–1.044038 |
| initial | 1 | download | payload | 65537 | 16 | 16 | radius | parallel | 10.190500 | 10.445750 | 1.021391 | 0.986063–1.057123 |
| initial | 1 | cpu | payload | 65537 | 8 | 0 | radius | original | 0.967750 | 1.043550 | 0.945439 | 0.905461–0.989494 |
| initial | 1 | download | payload | 65537 | 8 | 0 | radius | original | 6.454500 | 6.766550 | 0.967535 | 0.929061–1.005842 |
| initial | 1 | cpu | payload | 65537 | 8 | 1 | radius | original | 4.115300 | 5.181300 | 0.948326 | 0.842895–1.066160 |
| initial | 1 | download | payload | 65537 | 8 | 1 | radius | original | 11.076000 | 11.271000 | 0.990378 | 0.944405–1.040554 |
| initial | 1 | cpu | payload | 65537 | 65 | 0 | radius | original | 1.718100 | 1.540650 | 0.993275 | 0.923919–1.068096 |
| initial | 1 | download | payload | 65537 | 65 | 0 | radius | original | 12.558850 | 12.998100 | 0.992018 | 0.952615–1.026637 |
| initial | 1 | cpu | payload | 65537 | 0 | 0 | radius | original | 0.572150 | 0.585700 | 1.027976 | 0.984404–1.076405 |
| initial | 1 | download | payload | 65537 | 0 | 0 | radius | original | 5.228600 | 4.876050 | 1.024205 | 0.963718–1.083255 |
| initial | 1 | full | 17 | 1025 | 8 | search | nearest | original | 1.635000 | 1.624850 | 1.000969 | 0.962085–1.040727 |
| initial | 1 | full | 17 | 65535 | 8 | search | nearest | original | 10.962750 | 10.843700 | 0.998277 | 0.975336–1.020645 |
| initial | 1 | full | 17 | 65536 | 8 | search | nearest | parallel | 9.990750 | 10.202350 | 0.976453 | 0.938852–1.015160 |
| initial | 1 | full | 17 | 65537 | 8 | search | nearest | parallel | 10.448350 | 10.341950 | 1.004323 | 0.960547–1.052558 |
| initial | 1 | full | 17 | 65537 | 16 | search | nearest | parallel | 12.529000 | 12.505250 | 1.005262 | 0.972182–1.034282 |
| initial | 1 | full | 4097 | 65537 | 8 | search | nearest | parallel | 10.199300 | 9.866500 | 1.038945 | 1.003810–1.075322 |
| initial | 1 | full | 4097 | 65537 | 16 | search | nearest | parallel | 13.986800 | 13.540000 | 1.060296 | 1.004602–1.121135 |
| initial | 1 | full | 17 | 131073 | 8 | search | nearest | parallel | 15.237250 | 15.304450 | 0.986989 | 0.950870–1.023070 |
| initial | 1 | full | 17 | 65537 | 8 | search | radius | parallel | 10.222800 | 10.591050 | 0.979876 | 0.934882–1.027222 |
| initial | 1 | full | 17 | 65537 | 0 | search | radius | original | 8.022700 | 7.704050 | 1.006455 | 0.966795–1.048480 |
| initial | 2 | cpu | payload | 1025 | 8 | 8 | nearest | original | 0.068800 | 0.073150 | 0.939661 | 0.908828–0.967537 |
| initial | 2 | download | payload | 1025 | 8 | 8 | nearest | original | 1.868250 | 1.999150 | 0.955586 | 0.905658–1.008255 |
| initial | 2 | cpu | payload | 65535 | 8 | 8 | nearest | original | 10.202350 | 10.755000 | 0.986276 | 0.918252–1.063390 |
| initial | 2 | download | payload | 65535 | 8 | 8 | nearest | original | 14.606350 | 14.036200 | 1.014566 | 0.952062–1.082285 |
| initial | 2 | cpu | payload | 65536 | 8 | 8 | nearest | parallel | 4.090750 | 4.227750 | 0.987876 | 0.950691–1.029289 |
| initial | 2 | download | payload | 65536 | 8 | 8 | nearest | parallel | 8.110850 | 7.801450 | 1.000447 | 0.968825–1.031343 |
| initial | 2 | cpu | payload | 65537 | 8 | 8 | nearest | parallel | 4.302050 | 4.265350 | 1.024188 | 0.984255–1.064759 |
| initial | 2 | download | payload | 65537 | 8 | 8 | nearest | parallel | 8.264300 | 7.814450 | 1.015886 | 0.978371–1.055144 |
| initial | 2 | cpu | payload | 65537 | 16 | 16 | nearest | parallel | 6.709300 | 6.254950 | 0.942660 | 0.881129–1.002880 |
| initial | 2 | download | payload | 65537 | 16 | 16 | nearest | parallel | 13.633950 | 12.631100 | 1.027838 | 0.971676–1.084412 |
| initial | 2 | cpu | payload | 65537 | 64 | 64 | nearest | parallel | 11.067600 | 10.333550 | 1.037966 | 0.942850–1.143026 |
| initial | 2 | download | payload | 65537 | 64 | 64 | nearest | parallel | 21.586050 | 20.841700 | 1.041527 | 0.994564–1.095076 |
| initial | 2 | cpu | payload | 131073 | 8 | 8 | nearest | parallel | 6.060400 | 6.161550 | 1.015532 | 0.968449–1.058389 |
| initial | 2 | download | payload | 131073 | 8 | 8 | nearest | parallel | 11.486150 | 11.230950 | 1.003009 | 0.973790–1.032827 |
| initial | 2 | cpu | payload | 524289 | 1 | 1 | nearest | parallel | 18.767050 | 18.666800 | 1.006211 | 0.953778–1.060681 |
| initial | 2 | download | payload | 524289 | 1 | 1 | nearest | parallel | 22.876050 | 24.119550 | 0.974332 | 0.933172–1.015417 |
| initial | 2 | cpu | payload | 65537 | 65 | 9 | radius | parallel | 4.859150 | 4.360050 | 1.136195 | 1.108464–1.162743 |
| initial | 2 | download | payload | 65537 | 65 | 9 | radius | parallel | 19.785000 | 19.417600 | 1.026905 | 1.013388–1.040619 |
| initial | 2 | cpu | payload | 65537 | 65 | 65 | radius | parallel | 8.325200 | 8.086650 | 1.026109 | 0.991388–1.067768 |
| initial | 2 | download | payload | 65537 | 65 | 65 | radius | parallel | 22.952850 | 23.008950 | 0.989809 | 0.978304–1.001517 |
| initial | 2 | cpu | payload | 65537 | 16 | 16 | radius | parallel | 3.577300 | 3.488400 | 1.051500 | 1.017197–1.086555 |
| initial | 2 | download | payload | 65537 | 16 | 16 | radius | parallel | 10.387350 | 10.072300 | 1.006140 | 0.956038–1.044172 |
| initial | 2 | cpu | payload | 65537 | 8 | 0 | radius | original | 0.833800 | 0.894650 | 0.943908 | 0.898776–0.995212 |
| initial | 2 | download | payload | 65537 | 8 | 0 | radius | original | 6.952000 | 6.882550 | 1.013353 | 0.987884–1.042202 |
| initial | 2 | cpu | payload | 65537 | 8 | 1 | radius | original | 3.718300 | 4.763900 | 0.935380 | 0.847460–1.027311 |
| initial | 2 | download | payload | 65537 | 8 | 1 | radius | original | 10.661200 | 10.452500 | 1.016458 | 0.957009–1.082182 |
| initial | 2 | cpu | payload | 65537 | 65 | 0 | radius | original | 1.427100 | 1.475000 | 0.957102 | 0.887111–1.037436 |
| initial | 2 | download | payload | 65537 | 65 | 0 | radius | original | 10.861750 | 10.888600 | 0.999274 | 0.984700–1.014426 |
| initial | 2 | cpu | payload | 65537 | 0 | 0 | radius | original | 0.429800 | 0.424250 | 1.013400 | 0.974502–1.054493 |
| initial | 2 | download | payload | 65537 | 0 | 0 | radius | original | 4.117250 | 4.122000 | 0.999922 | 0.956867–1.045044 |
| initial | 2 | full | 17 | 1025 | 8 | search | nearest | original | 1.540150 | 1.555850 | 0.989077 | 0.962015–1.017366 |
| initial | 2 | full | 17 | 65535 | 8 | search | nearest | original | 10.342700 | 10.506450 | 0.983710 | 0.952471–1.016799 |
| initial | 2 | full | 17 | 65536 | 8 | search | nearest | parallel | 7.439050 | 6.988250 | 1.030020 | 0.993842–1.066206 |
| initial | 2 | full | 17 | 65537 | 8 | search | nearest | parallel | 9.539450 | 9.437650 | 1.007067 | 0.983291–1.030978 |
| initial | 2 | full | 17 | 65537 | 16 | search | nearest | parallel | 20.648100 | 20.595650 | 1.021221 | 0.995260–1.061946 |
| initial | 2 | full | 4097 | 65537 | 8 | search | nearest | parallel | 9.302500 | 8.924450 | 1.006194 | 0.972765–1.041115 |
| initial | 2 | full | 4097 | 65537 | 16 | search | nearest | parallel | 20.569450 | 20.755100 | 0.997305 | 0.984395–1.009614 |
| initial | 2 | full | 17 | 131073 | 8 | search | nearest | parallel | 15.960250 | 15.634150 | 1.016286 | 0.990980–1.041185 |
| initial | 2 | full | 17 | 65537 | 8 | search | radius | parallel | 9.712000 | 9.889000 | 1.004796 | 0.962585–1.060684 |
| initial | 2 | full | 17 | 65537 | 0 | search | radius | original | 5.272700 | 5.303750 | 1.004662 | 0.970728–1.038641 |
| final-test-only | 1 | cpu | payload | 1025 | 8 | 8 | nearest | original | 0.050850 | 0.050600 | 1.007829 | 0.958496–1.048915 |
| final-test-only | 1 | download | payload | 1025 | 8 | 8 | nearest | original | 1.286800 | 1.244300 | 1.020580 | 0.988241–1.053420 |
| final-test-only | 1 | cpu | payload | 65535 | 8 | 8 | nearest | original | 5.545400 | 5.475000 | 1.021271 | 0.993502–1.052573 |
| final-test-only | 1 | download | payload | 65535 | 8 | 8 | nearest | original | 7.941850 | 8.297450 | 1.002284 | 0.974789–1.028685 |
| final-test-only | 1 | cpu | payload | 65536 | 8 | 8 | nearest | parallel | 3.532300 | 3.430200 | 1.047757 | 1.006020–1.097734 |
| final-test-only | 1 | download | payload | 65536 | 8 | 8 | nearest | parallel | 5.790800 | 5.928450 | 0.978087 | 0.950580–1.005691 |
| final-test-only | 1 | cpu | payload | 65537 | 8 | 8 | nearest | parallel | 3.303550 | 3.395450 | 0.978691 | 0.946387–1.012420 |
| final-test-only | 1 | download | payload | 65537 | 8 | 8 | nearest | parallel | 7.353350 | 7.501250 | 0.989676 | 0.965342–1.014960 |
| final-test-only | 1 | cpu | payload | 65537 | 16 | 16 | nearest | parallel | 4.736850 | 4.584800 | 1.021100 | 0.981315–1.061990 |
| final-test-only | 1 | download | payload | 65537 | 16 | 16 | nearest | parallel | 11.323400 | 11.411200 | 0.997258 | 0.970427–1.025941 |
| final-test-only | 1 | cpu | payload | 65537 | 64 | 64 | nearest | parallel | 7.654650 | 7.395600 | 1.043763 | 0.995161–1.103039 |
| final-test-only | 1 | download | payload | 65537 | 64 | 64 | nearest | parallel | 22.403250 | 22.327900 | 1.006390 | 0.974797–1.038695 |
| final-test-only | 1 | cpu | payload | 131073 | 8 | 8 | nearest | parallel | 4.534250 | 4.643250 | 0.984369 | 0.944784–1.025527 |
| final-test-only | 1 | download | payload | 131073 | 8 | 8 | nearest | parallel | 11.418800 | 11.508600 | 0.987700 | 0.969293–1.006703 |
| final-test-only | 1 | cpu | payload | 524289 | 1 | 1 | nearest | parallel | 13.918100 | 13.541000 | 1.019605 | 0.967979–1.071450 |
| final-test-only | 1 | download | payload | 524289 | 1 | 1 | nearest | parallel | 20.344400 | 19.793200 | 1.012255 | 0.978364–1.046703 |
| final-test-only | 1 | cpu | payload | 65537 | 65 | 9 | radius | parallel | 3.338750 | 3.084350 | 1.061776 | 1.027451–1.098863 |
| final-test-only | 1 | download | payload | 65537 | 65 | 9 | radius | parallel | 17.845000 | 17.718050 | 1.006241 | 0.997883–1.014752 |
| final-test-only | 1 | cpu | payload | 65537 | 65 | 65 | radius | parallel | 7.616250 | 7.460000 | 1.006748 | 0.980753–1.033440 |
| final-test-only | 1 | download | payload | 65537 | 65 | 65 | radius | parallel | 22.219000 | 22.473900 | 0.989993 | 0.969865–1.008141 |
| final-test-only | 1 | cpu | payload | 65537 | 16 | 16 | radius | parallel | 2.878850 | 2.896400 | 0.997219 | 0.967210–1.028097 |
| final-test-only | 1 | download | payload | 65537 | 16 | 16 | radius | parallel | 9.559950 | 9.699900 | 0.997252 | 0.979497–1.016083 |
| final-test-only | 1 | cpu | payload | 65537 | 8 | 0 | radius | original | 0.751900 | 0.731300 | 1.012609 | 0.963243–1.063345 |
| final-test-only | 1 | download | payload | 65537 | 8 | 0 | radius | original | 6.078550 | 6.028450 | 1.022484 | 0.999924–1.047151 |
| final-test-only | 1 | cpu | payload | 65537 | 8 | 1 | radius | original | 2.839800 | 3.326550 | 0.909071 | 0.811749–1.022272 |
| final-test-only | 1 | download | payload | 65537 | 8 | 1 | radius | original | 10.542900 | 10.929600 | 0.974160 | 0.919027–1.033523 |
| final-test-only | 1 | cpu | payload | 65537 | 65 | 0 | radius | original | 1.454850 | 1.351900 | 1.017851 | 0.964636–1.075269 |
| final-test-only | 1 | download | payload | 65537 | 65 | 0 | radius | original | 10.798900 | 10.713650 | 1.020695 | 0.995965–1.054689 |
| final-test-only | 1 | cpu | payload | 65537 | 0 | 0 | radius | original | 0.387450 | 0.391400 | 1.009945 | 0.966447–1.053927 |
| final-test-only | 1 | download | payload | 65537 | 0 | 0 | radius | original | 3.848300 | 3.830100 | 0.986327 | 0.948367–1.021271 |
| final-test-only | 1 | full | 17 | 1025 | 8 | search | nearest | original | 1.270100 | 1.272050 | 0.998224 | 0.961974–1.037464 |
| final-test-only | 1 | full | 17 | 65535 | 8 | search | nearest | original | 8.847650 | 8.891400 | 1.002855 | 0.980545–1.024726 |
| final-test-only | 1 | full | 17 | 65536 | 8 | search | nearest | parallel | 5.360700 | 5.403800 | 1.001257 | 0.970775–1.031976 |
| final-test-only | 1 | full | 17 | 65537 | 8 | search | nearest | parallel | 7.397850 | 7.520200 | 0.975217 | 0.947284–1.003039 |
| final-test-only | 1 | full | 17 | 65537 | 16 | search | nearest | parallel | 12.964500 | 12.899700 | 1.012692 | 0.982985–1.053339 |
| final-test-only | 1 | full | 4097 | 65537 | 8 | search | nearest | parallel | 8.548600 | 8.997550 | 0.977227 | 0.955536–1.000117 |
| final-test-only | 1 | full | 4097 | 65537 | 16 | search | nearest | parallel | 8.641550 | 8.736650 | 0.994684 | 0.965331–1.024409 |
| final-test-only | 1 | full | 17 | 131073 | 8 | search | nearest | parallel | 12.871700 | 13.140100 | 1.004312 | 0.978352–1.032591 |
| final-test-only | 1 | full | 17 | 65537 | 8 | search | radius | parallel | 7.120650 | 7.270500 | 0.999894 | 0.964161–1.036643 |
| final-test-only | 1 | full | 17 | 65537 | 0 | search | radius | original | 5.106350 | 5.431300 | 0.979844 | 0.931491–1.028340 |
| final-test-only | 2 | cpu | payload | 1025 | 8 | 8 | nearest | original | 0.051750 | 0.050750 | 1.009980 | 0.989463–1.025726 |
| final-test-only | 2 | download | payload | 1025 | 8 | 8 | nearest | original | 1.175950 | 1.186650 | 1.017142 | 0.983946–1.052214 |
| final-test-only | 2 | cpu | payload | 65535 | 8 | 8 | nearest | original | 6.814850 | 6.726400 | 0.943114 | 0.844109–1.027543 |
| final-test-only | 2 | download | payload | 65535 | 8 | 8 | nearest | original | 9.637750 | 9.686700 | 0.960588 | 0.914952–1.002955 |
| final-test-only | 2 | cpu | payload | 65536 | 8 | 8 | nearest | parallel | 4.103900 | 3.797200 | 1.064274 | 1.013333–1.124873 |
| final-test-only | 2 | download | payload | 65536 | 8 | 8 | nearest | parallel | 7.142450 | 7.044700 | 1.016266 | 0.978001–1.060779 |
| final-test-only | 2 | cpu | payload | 65537 | 8 | 8 | nearest | parallel | 4.004650 | 4.194850 | 0.966218 | 0.929119–1.005511 |
| final-test-only | 2 | download | payload | 65537 | 8 | 8 | nearest | parallel | 8.446900 | 8.386200 | 1.004695 | 0.983973–1.023616 |
| final-test-only | 2 | cpu | payload | 65537 | 16 | 16 | nearest | parallel | 4.045850 | 4.017900 | 0.987519 | 0.940392–1.032748 |
| final-test-only | 2 | download | payload | 65537 | 16 | 16 | nearest | parallel | 10.807650 | 11.042350 | 0.987555 | 0.966805–1.006927 |
| final-test-only | 2 | cpu | payload | 65537 | 64 | 64 | nearest | parallel | 7.753750 | 7.939450 | 0.960386 | 0.900052–1.024397 |
| final-test-only | 2 | download | payload | 65537 | 64 | 64 | nearest | parallel | 22.305250 | 22.692750 | 0.989454 | 0.964194–1.014219 |
| final-test-only | 2 | cpu | payload | 131073 | 8 | 8 | nearest | parallel | 4.631600 | 4.800900 | 0.968119 | 0.922706–1.016167 |
| final-test-only | 2 | download | payload | 131073 | 8 | 8 | nearest | parallel | 11.385900 | 11.706900 | 0.982862 | 0.964446–1.001222 |
| final-test-only | 2 | cpu | payload | 524289 | 1 | 1 | nearest | parallel | 12.915500 | 12.711400 | 1.038679 | 0.989205–1.090247 |
| final-test-only | 2 | download | payload | 524289 | 1 | 1 | nearest | parallel | 19.228550 | 19.166150 | 0.995237 | 0.965904–1.026573 |
| final-test-only | 2 | cpu | payload | 65537 | 65 | 9 | radius | parallel | 3.430950 | 3.193500 | 1.062627 | 1.030025–1.095548 |
| final-test-only | 2 | download | payload | 65537 | 65 | 9 | radius | parallel | 18.060150 | 17.749500 | 1.016671 | 1.004502–1.028739 |
| final-test-only | 2 | cpu | payload | 65537 | 65 | 65 | radius | parallel | 7.714800 | 7.603950 | 0.986268 | 0.943084–1.024354 |
| final-test-only | 2 | download | payload | 65537 | 65 | 65 | radius | parallel | 22.298500 | 22.404200 | 0.996466 | 0.982543–1.010476 |
| final-test-only | 2 | cpu | payload | 65537 | 16 | 16 | radius | parallel | 2.875300 | 2.858400 | 1.008064 | 0.978233–1.038154 |
| final-test-only | 2 | download | payload | 65537 | 16 | 16 | radius | parallel | 9.619600 | 9.555900 | 1.008895 | 0.990907–1.029347 |
| final-test-only | 2 | cpu | payload | 65537 | 8 | 0 | radius | original | 0.792050 | 0.710200 | 1.099436 | 1.039468–1.162787 |
| final-test-only | 2 | download | payload | 65537 | 8 | 0 | radius | original | 6.070950 | 6.110650 | 1.000521 | 0.982710–1.018888 |
| final-test-only | 2 | cpu | payload | 65537 | 8 | 1 | radius | original | 3.034350 | 3.636250 | 0.873373 | 0.757597–1.006280 |
| final-test-only | 2 | download | payload | 65537 | 8 | 1 | radius | original | 11.180950 | 9.622250 | 1.092010 | 1.027589–1.159637 |
| final-test-only | 2 | cpu | payload | 65537 | 65 | 0 | radius | original | 1.138750 | 1.266250 | 0.926880 | 0.825267–1.042061 |
| final-test-only | 2 | download | payload | 65537 | 65 | 0 | radius | original | 10.442450 | 10.766550 | 0.999526 | 0.987222–1.011519 |
| final-test-only | 2 | cpu | payload | 65537 | 0 | 0 | radius | original | 0.378850 | 0.376900 | 0.991619 | 0.956363–1.027993 |
| final-test-only | 2 | download | payload | 65537 | 0 | 0 | radius | original | 3.587700 | 3.515550 | 1.002021 | 0.975567–1.030891 |
| final-test-only | 2 | full | 17 | 1025 | 8 | search | nearest | original | 1.252100 | 1.247750 | 1.009094 | 0.981532–1.037696 |
| final-test-only | 2 | full | 17 | 65535 | 8 | search | nearest | original | 8.320150 | 8.298200 | 1.006543 | 0.987386–1.025677 |
| final-test-only | 2 | full | 17 | 65536 | 8 | search | nearest | parallel | 5.371600 | 5.467100 | 0.987103 | 0.951916–1.022041 |
| final-test-only | 2 | full | 17 | 65537 | 8 | search | nearest | parallel | 5.954950 | 5.771950 | 1.034908 | 1.007342–1.066996 |
| final-test-only | 2 | full | 17 | 65537 | 16 | search | nearest | parallel | 11.953500 | 12.173900 | 0.984922 | 0.955557–1.008373 |
| final-test-only | 2 | full | 4097 | 65537 | 8 | search | nearest | parallel | 8.272500 | 8.346850 | 0.999309 | 0.974908–1.028421 |
| final-test-only | 2 | full | 4097 | 65537 | 16 | search | nearest | parallel | 8.790200 | 8.723650 | 0.993237 | 0.978409–1.008290 |
| final-test-only | 2 | full | 17 | 131073 | 8 | search | nearest | parallel | 13.176250 | 13.121700 | 1.020614 | 0.992560–1.050506 |
| final-test-only | 2 | full | 17 | 65537 | 8 | search | radius | parallel | 7.480900 | 7.456350 | 1.008026 | 0.974969–1.042446 |
| final-test-only | 2 | full | 17 | 65537 | 0 | search | radius | original | 4.975400 | 4.975000 | 0.997943 | 0.938821–1.055420 |

## CPU-only exploration, retained separately

One complete CPU harness compares the actual parent and candidate production
decoder sources with existing core/search libraries and a matching local
two-variant neighborhood-kind enum. It uses three warmups and 20 alternating
pairs, includes worker creation/join and checks every output index/distance bit
and count outside timing. The six complete median rows are retained below.
This preliminary run omits WGPU and host-load surveys and is not pooled into
native intervals. Frozen sources, executable, raw output and receipt hashes
remain in ignored local provenance. Cargo formatting preceded the native freeze.

| Queries | Capacity | Stored | Kind | Old median ms | New median ms |
| ---: | ---: | ---: | --- | ---: | ---: |
| 65535 | 8 | 8 | nearest | 7.068650 | 7.137900 |
| 65537 | 8 | 8 | nearest | 4.741800 | 4.736850 |
| 65537 | 16 | 16 | nearest | 6.374550 | 6.053200 |
| 65537 | 64 | 64 | nearest | 15.208400 | 15.042600 |
| 131073 | 8 | 8 | nearest | 6.689300 | 6.546700 |
| 65537 | 65 | 9 | radius | 3.488400 | 3.079650 |

## Local provenance

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-validation.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-initial-retention.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-final-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-final-validation.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-final-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-exploration-hashes.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-row-exploration.json`

These ignored files are local provenance. Complete tables and benchmark sources
provide the published evidence. No same-condition PCL/Open3D/OpenCV performance
comparison is made.
