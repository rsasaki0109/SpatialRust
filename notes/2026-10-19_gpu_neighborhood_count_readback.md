# Count-only GPU neighborhood readback decoding

Date: 2026-10-19. Named slice family: count-only neighborhood readback.
Workspace: `C:/Users/rsasa/Workspace/SpatialRust`.
Parent: `8f9a704608687d744dfd22f81639b6b7d5cf9452` (merged PR #285).

## Result and limitations

For zero-capacity results with at least 65,536 queries, decode the fixed two-word
headers with a complete integer-OR validation, then construct the final empty
rows and exact count plane in bulk. Smaller and nonzero-capacity results retain
the parent decoder, including its original bounded parallel policy. This change
is CPU work after an explicit GPU readback; it adds no GPU acceleration claim.
All local results and controls are below; intervals are individual, not adjusted
for multiple comparisons. Shared-host synthetic results are conditional evidence,
not portable application performance or a competitor speed comparison.

| Scope | Policy | Favorable | Inconclusive | Adverse | Unresolved | Old/new ratio range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu | count-only | 16 | 0 | 0 | 0 | 1.3606–1.7149 |
| cpu | original control | 0 | 18 | 2 | 0 | 0.9480–1.0329 |
| download | count-only | 14 | 2 | 0 | 0 | 1.0384–1.3562 |
| download | original control | 0 | 19 | 1 | 0 | 0.9539–1.0406 |
| full | count-only | 10 | 2 | 0 | 0 | 1.0277–1.1483 |
| full | original control | 0 | 8 | 0 | 0 | 0.9520–1.0128 |

cpu: 2 adverse / 20 controls; worst paired old/new 0.9480 (5.5% longer time); download:
1 adverse / 20 controls; worst paired old/new 0.9539 (4.8% longer time); full: 0 adverse
/ 8 controls; worst paired old/new 0.9520 (5.0% longer time).

Adverse and inconclusive controls are retained. Even a source-identical fallback
can differ through compiler layout and surrounding branch costs; shared-host
samples cannot establish a causal explanation for every observed difference.

## Schema and resource proof

A zero-capacity row is exactly `(stored_count,total_count)`. Every valid stored
count is zero. Nearest-k also requires total_count zero; radius accepts every
u32 total, including its maximum. Bitwise OR is zero exactly when every examined
word is zero. Nearest-k ORs both words; radius ORs only the stored-count words.
This proves the same original metadata predicate at this capacity. Every header
is examined, even after an invalid early word, and errors retain the same type
and message. The public admission fixes the stride to two words at capacity zero.

Only after validation are final empty rows and exact totals allocated. No
neighbors are allocated, and no new thread or intermediate point-sized vector
is introduced. Radius totals are gathered from the second words; nearest-k totals
are initialized to their proven zeros. Output vector sizes and ownership retain
the original schema. The validation and bulk-output passes change host access
order and may add a header scan; error inputs now finish the bounded full scan
before returning instead of stopping at the first invalid row. There is no
constant-cost or zero-copy claim. Small results retain the original collector
because exploratory measurements showed overhead around 4,096 rows.

GPU uploads/search/shaders, staging/copy/submit/map/wait/unmap, explicit receipts,
public APIs, manifests and dependencies are unchanged. Nonempty zero-capacity
readback still transfers `8 * queries` bytes; zero queries still transfer none.
All mapped borrows end before unmapping; final owned outputs outlive source
buffers. Radius completeness and the first overflow error retain their semantics.
No new unsafe code, runtime configuration or worker-availability query is added.

## Correctness and validation

Three focused tests cover empty/small/cutoff/large shapes, all count bits and
maximum u32 totals, every corruption bit in stored counts and nearest totals,
first/middle/last malformed GPU headers, repeated failed and successful maps,
runtime mismatches, CPU radius/nearest count oracles, exact receipts, radius
overflow, pool clears, input drops and concurrent owned downloads. Existing
bitwise/nonzero-row/parallel decoding tests remain controls.

Final related coverage has 58 native test executions: 32 neighborhood-family,
8 covariance, 4 ISS, 4 MLS, 3 FPFH and 7 facade executions. Focused precheck and
benchmark invocations are separate. CPU-only and both WASM configurations compile;
WASM GPU execution is not claimed. Clippy denies warnings for all features and
targets. Both WASM warning checks run before the native release precheck.

| Check | Result | Executions or scope |
| --- | --- | ---: |
| clippy | pass | build/lint |
| nearest | pass | 32 |
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

The frozen full parent decoder excludes only its original test registration.
The full parent downloader differs only in receiver binding and delegation to
that frozen decoder. An audit reverses only the added count-only constant,
selector and helper plus focused test registration; 691 other original Rust/WGSL
sources in GPU/search/image/core match after line-ending normalization.
All 730 source/manifest/executable SHA-256 values match across measurements and
validation. Native measured executable SHA-256: `129f36211e73d3fc5959dc713d8bb33f854bcabccb93fb76edc046df65ac3261`.
Tracked-source and PR-body scans found zero local-adapter-name matches.

## Native measurement method

Two surveys each measure 18 packed payloads through CPU conversion and complete
explicit download, plus 10 complete tree/query upload, GPU search and download
cases: all 92 route comparisons. No selected reruns or dropped rows. Each variant
has three warmups and 30 paired samples. Four CPU/download variants use balanced
four-position order; two full variants alternate. Selectors, validation, host
allocations and unmap are timed. Full calls also include both uploads and GPU
search; CPU tree building, fixture generation, result checking and output drops
are outside both timers. The frozen parent decoder/downloader is the baseline;
both full variants use the same unchanged GPU search and upload implementation.

Packed inputs include nearest/radius count-only rows, arbitrary u32 totals,
small and threshold-adjacent controls, and unchanged nonzero/sparse/parallel
paths. Full calls use synthetic targets of 17 and 4,097 unique grid points with
repeated exact queries. Zero-radius counts equal one, infinite-radius counts
equal target size, and nearest-k with capacity zero has zero counts. These
fixtures do not represent sensor or application workloads. Nearest-k with k=0
retains its original no-search-dispatch return; its full route times uploads
and readback without a neighborhood search dispatch.

Ratios are exponential means of paired log old/new times. Individual 95% paired
bootstrap intervals use 10,000 resamples, sorted indices 249/9749, and seeds
`2026101993 + 1000*survey + 100*(full_job) + 3*case + route_index`. No trimming,
filtering, epsilon for zero samples or multiple-comparison adjustment is used.
Zero samples leave a comparison unresolved. An interval wholly above/below one
is favorable/adverse; other resolved intervals are inconclusive.

- Survey 1, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 worker_creation_unchanged=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- Survey 1, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- Survey 2, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 worker_creation_unchanged=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- Survey 2, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.

Pre/post process CPU and anonymized GPU utilization samples cannot prove isolation
during every pair. Own builds/tests and bootstrap calculations did not overlap
timed measurements. Backend/device type and available workers are recorded;
local adapter name is excluded. No process/driver cache or clock reset is claimed.

## Every native comparison

| Survey | Scope | Points | Queries | Capacity | Stored | Kind | Policy | Old median ms | New median ms | Ratio | Individual paired 95% interval |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | cpu | payload | 1025 | 0 | 0 | nearest | original | 0.004500 | 0.005000 | 0.955930 | 0.910153–1.007535 |
| 1 | download | payload | 1025 | 0 | 0 | nearest | original | 1.159400 | 1.138800 | 0.989190 | 0.961472–1.015872 |
| 1 | cpu | payload | 4096 | 0 | 0 | nearest | original | 0.018200 | 0.018500 | 0.984206 | 0.963767–1.003091 |
| 1 | download | payload | 4096 | 0 | 0 | nearest | original | 1.174050 | 1.196900 | 0.953865 | 0.908634–0.997417 |
| 1 | cpu | payload | 65535 | 0 | 0 | nearest | original | 0.559450 | 0.557200 | 0.988274 | 0.931777–1.047210 |
| 1 | download | payload | 65535 | 0 | 0 | nearest | original | 1.803700 | 1.796800 | 1.002339 | 0.981057–1.023889 |
| 1 | cpu | payload | 65536 | 0 | 0 | nearest | count-only | 0.506650 | 0.381700 | 1.382891 | 1.320717–1.447241 |
| 1 | download | payload | 65536 | 0 | 0 | nearest | count-only | 1.705250 | 1.532250 | 1.115853 | 1.084530–1.147656 |
| 1 | cpu | payload | 65537 | 0 | 0 | nearest | count-only | 0.531000 | 0.371100 | 1.418300 | 1.351003–1.484183 |
| 1 | download | payload | 65537 | 0 | 0 | nearest | count-only | 1.988450 | 1.782050 | 1.116105 | 1.060771–1.169200 |
| 1 | cpu | payload | 524289 | 0 | 0 | nearest | count-only | 4.359050 | 2.759900 | 1.553098 | 1.505471–1.599337 |
| 1 | download | payload | 524289 | 0 | 0 | nearest | count-only | 6.938150 | 5.163750 | 1.355385 | 1.310367–1.406000 |
| 1 | cpu | payload | 1048577 | 0 | 0 | nearest | count-only | 11.429400 | 6.644650 | 1.714862 | 1.611436–1.822043 |
| 1 | download | payload | 1048577 | 0 | 0 | nearest | count-only | 17.813900 | 13.194850 | 1.356176 | 1.285057–1.433511 |
| 1 | cpu | payload | 1025 | 0 | 0 | radius | original | 0.005850 | 0.006200 | 0.956338 | 0.909655–1.008020 |
| 1 | download | payload | 1025 | 0 | 0 | radius | original | 3.992900 | 4.000650 | 0.968618 | 0.907156–1.032428 |
| 1 | cpu | payload | 4096 | 0 | 0 | radius | original | 0.019650 | 0.020400 | 0.976685 | 0.909648–1.031523 |
| 1 | download | payload | 4096 | 0 | 0 | radius | original | 4.015750 | 4.084750 | 0.967471 | 0.918519–1.017658 |
| 1 | cpu | payload | 65535 | 0 | 0 | radius | original | 0.676250 | 0.709200 | 0.991434 | 0.953543–1.034552 |
| 1 | download | payload | 65535 | 0 | 0 | radius | original | 5.076300 | 5.081800 | 1.008819 | 0.966374–1.055088 |
| 1 | cpu | payload | 65536 | 0 | 0 | radius | count-only | 0.592200 | 0.457700 | 1.391613 | 1.289031–1.502610 |
| 1 | download | payload | 65536 | 0 | 0 | radius | count-only | 5.442900 | 5.042850 | 1.057553 | 0.995193–1.119013 |
| 1 | cpu | payload | 65537 | 0 | 0 | radius | count-only | 0.584550 | 0.423550 | 1.426849 | 1.330030–1.520851 |
| 1 | download | payload | 65537 | 0 | 0 | radius | count-only | 4.663250 | 4.387400 | 1.063090 | 1.010137–1.118699 |
| 1 | cpu | payload | 524289 | 0 | 0 | radius | count-only | 5.199450 | 3.706950 | 1.427820 | 1.329881–1.527788 |
| 1 | download | payload | 524289 | 0 | 0 | radius | count-only | 11.156950 | 9.301700 | 1.201228 | 1.143157–1.261387 |
| 1 | cpu | payload | 1048577 | 0 | 0 | radius | count-only | 12.561850 | 8.600550 | 1.452358 | 1.340463–1.573258 |
| 1 | download | payload | 1048577 | 0 | 0 | radius | count-only | 18.319750 | 14.872900 | 1.282120 | 1.206858–1.365804 |
| 1 | cpu | payload | 65537 | 8 | 0 | radius | original | 0.947150 | 0.981750 | 0.948047 | 0.900111–0.993035 |
| 1 | download | payload | 65537 | 8 | 0 | radius | original | 5.306300 | 5.330200 | 0.969308 | 0.919570–1.018429 |
| 1 | cpu | payload | 65537 | 8 | 1 | radius | original | 4.821050 | 4.633700 | 0.993356 | 0.894104–1.094603 |
| 1 | download | payload | 65537 | 8 | 1 | radius | original | 9.422800 | 9.790450 | 0.979749 | 0.912680–1.057174 |
| 1 | cpu | payload | 65537 | 8 | 8 | nearest | original | 4.049450 | 4.116400 | 0.981867 | 0.946912–1.019485 |
| 1 | download | payload | 65537 | 8 | 8 | nearest | original | 8.077500 | 8.209200 | 1.023018 | 0.982909–1.065131 |
| 1 | cpu | payload | 65537 | 65 | 0 | radius | original | 1.682250 | 1.608900 | 1.002749 | 0.921087–1.089496 |
| 1 | download | payload | 65537 | 65 | 0 | radius | original | 11.020950 | 11.083300 | 0.979130 | 0.943716–1.008157 |
| 1 | full | 17 | 65535 | 0 | search | radius | original | 2.965850 | 3.161750 | 0.968148 | 0.914515–1.016989 |
| 1 | full | 17 | 65536 | 0 | search | radius | count-only | 3.455750 | 3.297900 | 1.042736 | 0.977871–1.101044 |
| 1 | full | 17 | 65537 | 0 | search | radius | count-only | 3.204100 | 2.991550 | 1.070867 | 1.029554–1.115209 |
| 1 | full | 17 | 524289 | 0 | search | radius | count-only | 11.706000 | 10.359900 | 1.148321 | 1.107743–1.192601 |
| 1 | full | 4097 | 65537 | 0 | search | radius | count-only | 3.221400 | 3.135000 | 1.027667 | 0.994503–1.059242 |
| 1 | full | 4097 | 131073 | 0 | search | radius | count-only | 5.889000 | 5.332800 | 1.097018 | 1.070570–1.124325 |
| 1 | full | 17 | 65537 | 0 | search | nearest | count-only | 3.632600 | 3.404900 | 1.087953 | 1.049691–1.124053 |
| 1 | full | 17 | 65537 | 8 | search | nearest | original | 10.058650 | 10.014750 | 0.996311 | 0.951863–1.042902 |
| 1 | full | 4097 | 65537 | 8 | search | nearest | original | 12.451100 | 12.426250 | 1.004715 | 0.976915–1.037658 |
| 1 | full | 17 | 4096 | 0 | search | radius | original | 3.110250 | 3.124150 | 0.951975 | 0.885552–1.011931 |
| 2 | cpu | payload | 1025 | 0 | 0 | nearest | original | 0.004600 | 0.004950 | 0.959420 | 0.920557–1.004100 |
| 2 | download | payload | 1025 | 0 | 0 | nearest | original | 1.155950 | 1.146500 | 0.998043 | 0.968777–1.026048 |
| 2 | cpu | payload | 4096 | 0 | 0 | nearest | original | 0.019050 | 0.019500 | 0.990374 | 0.980141–1.001532 |
| 2 | download | payload | 4096 | 0 | 0 | nearest | original | 1.241900 | 1.231500 | 0.993766 | 0.957677–1.033288 |
| 2 | cpu | payload | 65535 | 0 | 0 | nearest | original | 0.547300 | 0.548950 | 0.999556 | 0.971045–1.027276 |
| 2 | download | payload | 65535 | 0 | 0 | nearest | original | 1.760950 | 1.753550 | 1.040568 | 0.989685–1.103003 |
| 2 | cpu | payload | 65536 | 0 | 0 | nearest | count-only | 0.553500 | 0.388600 | 1.403567 | 1.352476–1.451864 |
| 2 | download | payload | 65536 | 0 | 0 | nearest | count-only | 1.761200 | 1.609600 | 1.079027 | 1.051022–1.107024 |
| 2 | cpu | payload | 65537 | 0 | 0 | nearest | count-only | 0.540050 | 0.383600 | 1.425585 | 1.382504–1.470822 |
| 2 | download | payload | 65537 | 0 | 0 | nearest | count-only | 1.813250 | 1.604600 | 1.118879 | 1.083020–1.159881 |
| 2 | cpu | payload | 524289 | 0 | 0 | nearest | count-only | 4.350550 | 2.720700 | 1.588659 | 1.518746–1.663736 |
| 2 | download | payload | 524289 | 0 | 0 | nearest | count-only | 6.724350 | 4.992900 | 1.335460 | 1.291297–1.379882 |
| 2 | cpu | payload | 1048577 | 0 | 0 | nearest | count-only | 10.930300 | 6.160550 | 1.685496 | 1.583168–1.794299 |
| 2 | download | payload | 1048577 | 0 | 0 | nearest | count-only | 16.045900 | 12.375150 | 1.356067 | 1.255855–1.457844 |
| 2 | cpu | payload | 1025 | 0 | 0 | radius | original | 0.005400 | 0.005600 | 0.970738 | 0.918920–1.032832 |
| 2 | download | payload | 1025 | 0 | 0 | radius | original | 3.099050 | 3.164500 | 0.968002 | 0.931674–1.001814 |
| 2 | cpu | payload | 4096 | 0 | 0 | radius | original | 0.020000 | 0.017600 | 1.032946 | 0.967418–1.120112 |
| 2 | download | payload | 4096 | 0 | 0 | radius | original | 3.054000 | 3.004050 | 1.021937 | 0.992280–1.055003 |
| 2 | cpu | payload | 65535 | 0 | 0 | radius | original | 0.663800 | 0.648450 | 0.997689 | 0.955259–1.043831 |
| 2 | download | payload | 65535 | 0 | 0 | radius | original | 4.102800 | 3.956050 | 1.012448 | 0.962018–1.068573 |
| 2 | cpu | payload | 65536 | 0 | 0 | radius | count-only | 0.662950 | 0.479250 | 1.369137 | 1.307123–1.434172 |
| 2 | download | payload | 65536 | 0 | 0 | radius | count-only | 4.125500 | 3.973250 | 1.038379 | 0.991924–1.089078 |
| 2 | cpu | payload | 65537 | 0 | 0 | radius | count-only | 0.609950 | 0.425450 | 1.402829 | 1.344788–1.466816 |
| 2 | download | payload | 65537 | 0 | 0 | radius | count-only | 3.894900 | 3.683100 | 1.071933 | 1.040441–1.102400 |
| 2 | cpu | payload | 524289 | 0 | 0 | radius | count-only | 4.900800 | 3.623750 | 1.360582 | 1.297459–1.424262 |
| 2 | download | payload | 524289 | 0 | 0 | radius | count-only | 8.541550 | 7.491150 | 1.173939 | 1.126444–1.224085 |
| 2 | cpu | payload | 1048577 | 0 | 0 | radius | count-only | 8.590350 | 6.165500 | 1.395065 | 1.338740–1.454099 |
| 2 | download | payload | 1048577 | 0 | 0 | radius | count-only | 14.466500 | 12.659300 | 1.145825 | 1.122074–1.170007 |
| 2 | cpu | payload | 65537 | 8 | 0 | radius | original | 0.678550 | 0.715450 | 0.948766 | 0.900134–0.996584 |
| 2 | download | payload | 65537 | 8 | 0 | radius | original | 5.901750 | 5.825600 | 0.991634 | 0.952053–1.025304 |
| 2 | cpu | payload | 65537 | 8 | 1 | radius | original | 4.218700 | 4.287300 | 1.004118 | 0.928487–1.089778 |
| 2 | download | payload | 65537 | 8 | 1 | radius | original | 9.075300 | 8.960000 | 0.995719 | 0.944392–1.050169 |
| 2 | cpu | payload | 65537 | 8 | 8 | nearest | original | 4.265100 | 4.261850 | 0.999252 | 0.947323–1.057406 |
| 2 | download | payload | 65537 | 8 | 8 | nearest | original | 10.088800 | 10.069800 | 1.010636 | 0.975115–1.042959 |
| 2 | cpu | payload | 65537 | 65 | 0 | radius | original | 1.420900 | 1.291800 | 0.993546 | 0.922944–1.069914 |
| 2 | download | payload | 65537 | 65 | 0 | radius | original | 15.668750 | 15.679300 | 0.995589 | 0.980223–1.011648 |
| 2 | full | 17 | 65535 | 0 | search | radius | original | 2.433750 | 2.459450 | 1.007483 | 0.982829–1.033567 |
| 2 | full | 17 | 65536 | 0 | search | radius | count-only | 2.500900 | 2.273550 | 1.067405 | 1.036887–1.095754 |
| 2 | full | 17 | 65537 | 0 | search | radius | count-only | 2.479400 | 2.310350 | 1.078912 | 1.053011–1.105785 |
| 2 | full | 17 | 524289 | 0 | search | radius | count-only | 10.681800 | 9.670500 | 1.121673 | 1.091957–1.152080 |
| 2 | full | 4097 | 65537 | 0 | search | radius | count-only | 3.163150 | 3.041950 | 1.046948 | 1.017396–1.077396 |
| 2 | full | 4097 | 131073 | 0 | search | radius | count-only | 5.535600 | 5.127800 | 1.058265 | 1.006021–1.110061 |
| 2 | full | 17 | 65537 | 0 | search | nearest | count-only | 3.023050 | 2.810450 | 1.085012 | 1.060308–1.109435 |
| 2 | full | 17 | 65537 | 8 | search | nearest | original | 7.442300 | 7.320050 | 0.993533 | 0.969350–1.021449 |
| 2 | full | 4097 | 65537 | 8 | search | nearest | original | 10.201800 | 10.493050 | 0.982640 | 0.965263–1.001186 |
| 2 | full | 17 | 4096 | 0 | search | radius | original | 2.487500 | 2.444900 | 1.012793 | 0.962754–1.064252 |

## CPU-only exploration, retained separately

The first candidate used separate header validation with an early-exit predicate
and a 4,096-row cutoff. It regressed around that cutoff. The revised candidate
uses a complete OR reduction, parent allocation order and the 65,536-row cutoff.
Both complete CPU harness runs are retained below. They use actual production
and parent decoder source plus existing core/search Rust libraries, with a
matching local two-variant neighborhood-kind enum. They omit WGPU and are
exploratory medians, not native full-call or application measurements; they are
not pooled into native estimates. Timings use three warmups and 30 alternating
pairs, and validate every resulting row and count outside timing. Initial
source, executable, raw output and receipt hashes are preserved locally.

| Candidate | Rows | Kind | Old median ms | New median ms |
| --- | ---: | --- | ---: | ---: |
| initial-4096 | 65 | nearest | 0.000400 | 0.000300 |
| initial-4096 | 65 | radius | 0.000400 | 0.000400 |
| initial-4096 | 1025 | nearest | 0.003600 | 0.003000 |
| initial-4096 | 1025 | radius | 0.003600 | 0.003000 |
| initial-4096 | 4095 | nearest | 0.044700 | 0.044500 |
| initial-4096 | 4095 | radius | 0.044000 | 0.043700 |
| initial-4096 | 4096 | nearest | 0.044700 | 0.061750 |
| initial-4096 | 4096 | radius | 0.045050 | 0.059850 |
| initial-4096 | 4097 | nearest | 0.042950 | 0.059200 |
| initial-4096 | 4097 | radius | 0.040350 | 0.053650 |
| initial-4096 | 65537 | nearest | 0.473450 | 0.416800 |
| initial-4096 | 65537 | radius | 0.479550 | 0.426550 |
| initial-4096 | 524289 | nearest | 3.176850 | 2.627750 |
| initial-4096 | 524289 | radius | 3.838800 | 3.513300 |
| revised-65536 | 65 | nearest | 0.000200 | 0.000200 |
| revised-65536 | 65 | radius | 0.000300 | 0.000300 |
| revised-65536 | 1025 | nearest | 0.002300 | 0.002300 |
| revised-65536 | 1025 | radius | 0.002300 | 0.002300 |
| revised-65536 | 4095 | nearest | 0.035150 | 0.035100 |
| revised-65536 | 4095 | radius | 0.035100 | 0.035150 |
| revised-65536 | 4096 | nearest | 0.034200 | 0.034200 |
| revised-65536 | 4096 | radius | 0.034050 | 0.033900 |
| revised-65536 | 4097 | nearest | 0.045500 | 0.045350 |
| revised-65536 | 4097 | radius | 0.033900 | 0.034100 |
| revised-65536 | 65537 | nearest | 0.368150 | 0.303550 |
| revised-65536 | 65537 | radius | 0.365500 | 0.310050 |
| revised-65536 | 524289 | nearest | 3.415000 | 2.561400 |
| revised-65536 | 524289 | radius | 3.233750 | 2.981800 |

## Local provenance

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-count-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-count-validation.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-count-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-count-initial-exploration-hashes.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-count-exploration.json`

These ignored files are local provenance. The complete tables and benchmark
sources provide the published evidence. No same-condition PCL/Open3D/OpenCV
performance comparison is made.
