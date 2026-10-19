# Bounded CPU decoding of explicit GPU neighborhood readback

Date: 2026-10-19. Named slice family: parallel neighborhood readback.
Workspace: `C:/Users/rsasa/Workspace/SpatialRust`.
Parent: `bf8e60831366a4e6f3ebb80f4b79734bdab46f8a` (merged PR #284).

## Result and limits

Large, sufficiently populated neighborhood readbacks can decode on at most
four native CPU threads per call, including the caller. GPU search, shaders,
queue synchronization, output schema and payload transfers retain their original
contracts. The complete local results below include serial controls and adverse
results; they are synthetic measurements on one shared host, not a general
application speed guarantee or a comparison against another library.

| Route | Policy | Favorable | Inconclusive | Adverse | Unresolved | Paired old/new ratio range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu | parallel policy | 18 | 0 | 0 | 0 | 1.6589–2.5759 |
| cpu | serial control | 3 | 5 | 8 | 0 | 0.8541–1.1572 |
| download | parallel policy | 18 | 0 | 0 | 0 | 1.2262–2.0591 |
| download | serial control | 0 | 14 | 2 | 0 | 0.9672–1.0092 |
| full | parallel policy | 10 | 0 | 0 | 0 | 1.3115–1.6362 |
| full | serial control | 0 | 7 | 1 | 0 | 0.9367–1.0461 |

Serial controls still include 8 adverse CPU, 2 adverse download and 1 adverse full-call
intervals. Their worst paired old/new ratios are 0.8541, 0.9672 and 0.9367 respectively
(about 17.1%, 3.4% and 6.8% longer times). The first-quarter concentrated control has
adverse CPU intervals in both surveys. Restoring the original collector mitigates the
initial candidate but does not establish equal fallback cost; the probe, surrounding
branches, compiler layout and shared host can matter. These are retained limits, with no
established causal attribution for every control result.

## Execution policy and correctness

Below 65,536 rows, at zero capacity, and on WASM, the original in-loop serial
collector is retained without querying host worker availability. For other calls,
64 evenly spaced row headers estimate stored-neighbor density. The integer
average must reach `ceil(524288 / rows)`. Each of four sampled contiguous regions
must also reach at least half that required average, rounded up. Sampling only
selects execution policy: it is not a validation shortcut or a proof of total
work or balance. Periodic or otherwise unrepresentative distributions can still
trigger unhelpful parallel work, and uneven work within a region can reduce gain.

Native worker availability is queried only after this eligibility test. At most
four available workers participate; one-worker and unavailable-worker calls use
the original serial collector. Each extra worker owns disjoint final row/count
slots and only borrows the already completed mapped u32 payload. Every worker is
joined before the mapped view is dropped and the staging buffer is unmapped,
including when one band rejects a header. Every header retains the original
stored-count/capacity, total-count and nearest-count equality checks and error.

Row order, neighbor order, u32-to-usize indices, every returned f32 bit pattern,
exact radius totals, first-overflow reporting, ignored unused slots, runtime
identity, empty results and owned-output lifetimes are unchanged. Radius
capacity remains arbitrary within original admission; the CPU converter does
not impose the nearest-k limit. No new public API, dependency, unsafe boundary,
GPU dispatch, device payload copy or hidden transfer is introduced.

Parallel conversion safely initializes final top-level row/count slots before
filling them, adding host writes of those slot sizes compared with serial pushes.
It creates at most three scoped worker threads plus a small handle vector per
call; thread creation, joining, scheduling, allocator contention and OS stack
resources have costs. There is no new point-sized worker payload copy or global
thread pool. Simultaneous callers have separate per-call bounds, not a global
four-thread budget. Low availability, shared-host load and unusual distributions
can remove gains. Device-to-host bytes remain `queries * (8 + 8 * capacity)` for
nonempty queries, including padding; search itself still records no payload copy.

## Correctness and build validation

Five focused tests cover serial/parallel cutoffs, availability 0/1/2/3/4/large,
partial bands and large query counts, zero and arbitrary radius capacities,
uniform/mixed/concentrated counts, unused poison slots, signed zeros,
subnormals/extrema/infinities/quiet and signaling NaN payloads, u32 maximum indices,
all-band malformed headers, repeated failed/successful maps, runtime mismatches,
concurrent downloads, owned results after pool clears and buffer drops, radius
overflow and complete CPU nearest/radius comparisons around the 65,536 cutoff.

The focused neighborhood family (29 tests), downstream GPU families and facade
provide 55 native test executions in the final validation, excluding the focused precheck and measured
benchmark invocations. The initial broader search validation ran 80 native test
executions; unchanged tree-building benchmark families are not repeated in the
final focused scope. CPU-only, minimal WASM, CI WASM feature configuration,
Clippy with warnings denied, formatting and diff checks also pass. GPU-required
coverage was run natively; WASM coverage is compilation, not GPU execution.

| Check | Result | Test executions / scope |
| --- | --- | ---: |
| clippy | pass | build/lint |
| nearest | pass | 29 |
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

Source audit retains 599 original GPU/search/image source files
byte-for-byte after text line-ending normalization. Only the original download's
CPU collector is delegated and a private decoder module is registered. The full
frozen parent downloader differs only in receiver binding and formatting.
710 source/manifest/executable SHA-256 values match from measurement through
final validation; manifests and original shader sources remain unchanged.
Final measured release executable SHA-256: `0d50e6e4ffa3526ebf6c25686c10ed882fda380635d2c48731e5d7d0a2738226`.
Privacy scans found zero local-adapter-name matches in tracked material and PR text.

## Measurement method

Two complete surveys each run 17 packed-payload cases through both CPU conversion
and complete explicit download, plus nine complete upload/search/download cases.
This gives all 86 final route comparisons, with no selected reruns or exclusions.
Both variants get three warmups then 30 paired samples. CPU/download variants use
balanced four-position ordering; complete search variants alternate order.
Worker selection, spawning, joining and safe output initialization are timed.
GPU download timings include staging creation, copy, submit, map, wait, decode
and unmap; complete calls additionally upload the tree and queries and execute
search. CPU tree construction, fixture generation, checking and output drops
are outside both timers. CPU-only conversion uses the original collector;
GPU download uses the complete frozen original function. Both full variants use
the same unchanged current GPU search programs and upload paths.

Packed fixtures vary query count, capacity, stored count and radius truncation,
including capacities beyond nearest-k, empty/count-only and sparse controls.
Distribution 0 is uniform stored counts, 1 varies between 8 and 16, and 2 stores
neighbors only in the first quarter, which the region policy rejects. Complete
search fixtures use a synthetic 17-point target with repeated exact queries;
this isolates query/readback throughput and does not represent large-target
search or application performance. Zero-radius controls contain one exact match,
not empty rows. The initial raw full-call flag named `empty` meant zero radius;
the delivered benchmark and analyzed metadata use `zero_radius`.

Each paired old/new ratio is the exponential mean log ratio. Individual 95%
paired bootstrap intervals use 10,000 resamples and indices 249/9749, with no
multiple-comparison adjustment, filtering, trimming or epsilon for zero samples.
Final seeds are `2026101991 + 1000*survey + 100*(full_job) + 3*case + route_index`.
Ratios above one favor this change. An interval entirely above/below one is
labeled favorable/adverse; other resolved intervals are inconclusive.

Environment and pre/post diagnostics:

- Survey 1, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 max_workers=4 probe_rows=64 warmups=3 pairs=30 checks_drops_excluded=true worker_spawn_and_join_included=true`; pre/post observed-quiet `False`.
- Survey 1, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.
- Survey 2, packed: `backend=Vulkan device_type=DiscreteGpu available_workers=12 max_workers=4 probe_rows=64 warmups=3 pairs=30 checks_drops_excluded=true worker_spawn_and_join_included=true`; pre/post observed-quiet `False`.
- Survey 2, full: `backend=Vulkan device_type=DiscreteGpu available_workers=12 warmups=3 pairs=30 tree_build_excluded=true uploads_search_map_decode_included=true checks_drops_excluded=true`; pre/post observed-quiet `False`.

All pre/post surveys observed a shared host. These samples cannot establish
isolation during every pair. Own builds/tests and bootstrap analysis did not
overlap timed measurements. No local GPU model is recorded. Backend/device type,
available workers, timing scopes, process CPU observations and anonymized GPU
utilization are retained in ignored receipts. GPU clocks/driver caches were not
reset. There is no same-condition PCL, Open3D or OpenCV speed comparison.

## All final comparisons

| Survey | Route | Queries | Capacity | Stored / distribution | Kind | Policy | Old median ms | New median ms | Ratio | Paired 95% interval |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | cpu | 1025 | 8 | 8/uniform | nearest | serial | 0.047450 | 0.052050 | 0.920185 | 0.896712–0.946452 |
| 1 | download | 1025 | 8 | 8/uniform | nearest | serial | 1.205450 | 1.223850 | 1.001623 | 0.971383–1.033098 |
| 1 | cpu | 65535 | 8 | 8/uniform | nearest | serial | 6.659000 | 7.011300 | 0.967125 | 0.935870–1.002608 |
| 1 | download | 65535 | 8 | 8/uniform | nearest | serial | 9.447300 | 9.431900 | 0.980442 | 0.949292–1.011051 |
| 1 | cpu | 65536 | 8 | 8/uniform | nearest | parallel | 8.198900 | 3.954100 | 2.099959 | 1.949390–2.259078 |
| 1 | download | 65536 | 8 | 8/uniform | nearest | parallel | 13.639650 | 8.486100 | 1.581559 | 1.505998–1.661334 |
| 1 | cpu | 65537 | 8 | 8/uniform | nearest | parallel | 7.851400 | 3.717200 | 2.035298 | 1.882869–2.195603 |
| 1 | download | 65537 | 8 | 8/uniform | nearest | parallel | 12.327450 | 8.610350 | 1.454941 | 1.391343–1.525248 |
| 1 | cpu | 65537 | 16 | 16/uniform | nearest | parallel | 8.102200 | 4.562950 | 1.883482 | 1.736238–2.048640 |
| 1 | download | 65537 | 16 | 16/uniform | nearest | parallel | 18.067200 | 11.444050 | 1.550030 | 1.498169–1.601789 |
| 1 | cpu | 65537 | 64 | 64/uniform | nearest | parallel | 20.366300 | 8.248150 | 1.976781 | 1.779530–2.211126 |
| 1 | download | 65537 | 64 | 64/uniform | nearest | parallel | 33.426900 | 24.022000 | 1.343808 | 1.275373–1.417506 |
| 1 | cpu | 131073 | 8 | 8/uniform | nearest | parallel | 11.951100 | 5.527600 | 2.120913 | 1.942981–2.311542 |
| 1 | download | 131073 | 8 | 8/uniform | nearest | parallel | 20.066400 | 12.616350 | 1.545706 | 1.482545–1.613490 |
| 1 | cpu | 524288 | 1 | 1/uniform | nearest | parallel | 33.552450 | 14.106450 | 2.316358 | 2.144838–2.495599 |
| 1 | download | 524288 | 1 | 1/uniform | nearest | parallel | 44.007650 | 20.713050 | 2.059143 | 1.926205–2.189368 |
| 1 | cpu | 65537 | 8 | 0/uniform | radius | serial | 0.787250 | 0.810750 | 0.946051 | 0.906583–0.991422 |
| 1 | download | 65537 | 8 | 0/uniform | radius | serial | 6.560050 | 6.610350 | 1.005787 | 0.978313–1.035239 |
| 1 | cpu | 65537 | 8 | 1/uniform | radius | serial | 5.895200 | 4.960450 | 1.132433 | 0.979948–1.297645 |
| 1 | download | 65537 | 8 | 1/uniform | radius | serial | 11.473950 | 11.305750 | 1.009205 | 0.927703–1.092818 |
| 1 | cpu | 65537 | 0 | 0/uniform | radius | serial | 0.698500 | 0.706700 | 0.977785 | 0.930617–1.024877 |
| 1 | download | 65537 | 0 | 0/uniform | radius | serial | 4.997050 | 5.080050 | 0.994429 | 0.965193–1.025128 |
| 1 | cpu | 65537 | 65 | 9/uniform | radius | parallel | 7.621450 | 3.818250 | 1.958628 | 1.782779–2.166536 |
| 1 | download | 65537 | 65 | 9/uniform | radius | parallel | 23.073950 | 18.538600 | 1.226237 | 1.188284–1.264498 |
| 1 | cpu | 65537 | 65 | 1/uniform | radius | serial | 5.253550 | 4.711700 | 1.103018 | 1.027177–1.188762 |
| 1 | download | 65537 | 65 | 1/uniform | radius | serial | 21.288450 | 21.373750 | 0.987809 | 0.963900–1.011053 |
| 1 | cpu | 65537 | 65 | 0/uniform | radius | serial | 1.093850 | 1.205100 | 0.860981 | 0.783177–0.944233 |
| 1 | download | 65537 | 65 | 0/uniform | radius | serial | 10.919450 | 11.895750 | 0.975195 | 0.951202–1.001712 |
| 1 | cpu | 65537 | 65 | 65/uniform | radius | parallel | 12.931750 | 7.598950 | 1.749267 | 1.660094–1.844361 |
| 1 | download | 65537 | 65 | 65/uniform | radius | parallel | 26.906850 | 21.640250 | 1.289234 | 1.251035–1.329847 |
| 1 | cpu | 65537 | 16 | 16/mixed8to16 | radius | parallel | 5.163100 | 3.313050 | 1.723456 | 1.592132–1.879625 |
| 1 | download | 65537 | 16 | 16/mixed8to16 | radius | parallel | 13.016300 | 9.672550 | 1.353883 | 1.306913–1.404048 |
| 1 | cpu | 65537 | 64 | 64/first-quarter | radius | serial | 3.223000 | 3.651700 | 0.913138 | 0.853786–0.973954 |
| 1 | download | 65537 | 64 | 64/first-quarter | radius | serial | 18.447150 | 18.864950 | 0.967193 | 0.948924–0.985693 |
| 1 | full | 1025 | 8 | search/uniform | nearest | serial | 1.259500 | 1.270150 | 1.005163 | 0.980000–1.031894 |
| 1 | full | 65535 | 8 | search/uniform | nearest | serial | 8.801100 | 9.049700 | 1.004998 | 0.975167–1.035585 |
| 1 | full | 65536 | 8 | search/uniform | nearest | parallel | 9.436600 | 6.456250 | 1.521467 | 1.430764–1.607342 |
| 1 | full | 65537 | 8 | search/uniform | nearest | parallel | 13.725100 | 8.822300 | 1.591301 | 1.507680–1.677280 |
| 1 | full | 65537 | 16 | search/uniform | nearest | parallel | 27.691550 | 22.635050 | 1.319307 | 1.279205–1.363265 |
| 1 | full | 131073 | 8 | search/uniform | nearest | parallel | 27.279650 | 18.560100 | 1.516661 | 1.454090–1.581325 |
| 1 | full | 65537 | 8 | search/uniform | radius | parallel | 13.051850 | 9.042000 | 1.376383 | 1.306440–1.447104 |
| 1 | full | 65537 | 8 | search/uniform (zero radius) | radius | serial | 11.270850 | 11.182700 | 1.032617 | 0.989565–1.076780 |
| 1 | full | 65537 | 0 | search/uniform | radius | serial | 5.262200 | 5.626250 | 0.936747 | 0.909094–0.965632 |
| 2 | cpu | 1025 | 8 | 8/uniform | nearest | serial | 0.048400 | 0.048350 | 0.995968 | 0.984969–1.007180 |
| 2 | download | 1025 | 8 | 8/uniform | nearest | serial | 1.131350 | 1.160950 | 0.976257 | 0.912206–1.027282 |
| 2 | cpu | 65535 | 8 | 8/uniform | nearest | serial | 6.824550 | 6.312800 | 1.068252 | 1.027830–1.112114 |
| 2 | download | 65535 | 8 | 8/uniform | nearest | serial | 8.378800 | 8.747850 | 0.968611 | 0.947337–0.989402 |
| 2 | cpu | 65536 | 8 | 8/uniform | nearest | parallel | 7.216300 | 3.532700 | 2.018108 | 1.879213–2.158487 |
| 2 | download | 65536 | 8 | 8/uniform | nearest | parallel | 10.231600 | 6.284700 | 1.582542 | 1.508297–1.663086 |
| 2 | cpu | 65537 | 8 | 8/uniform | nearest | parallel | 5.772100 | 3.545000 | 1.658892 | 1.565704–1.768951 |
| 2 | download | 65537 | 8 | 8/uniform | nearest | parallel | 12.490150 | 7.464450 | 1.548098 | 1.465191–1.636196 |
| 2 | cpu | 65537 | 16 | 16/uniform | nearest | parallel | 8.334750 | 4.469750 | 1.886489 | 1.752648–2.034841 |
| 2 | download | 65537 | 16 | 16/uniform | nearest | parallel | 15.731350 | 11.191050 | 1.414175 | 1.354369–1.478542 |
| 2 | cpu | 65537 | 64 | 64/uniform | nearest | parallel | 22.444900 | 8.284400 | 2.222623 | 2.000809–2.463977 |
| 2 | download | 65537 | 64 | 64/uniform | nearest | parallel | 30.212150 | 22.924800 | 1.359721 | 1.295737–1.428748 |
| 2 | cpu | 131073 | 8 | 8/uniform | nearest | parallel | 11.328600 | 5.418150 | 2.042177 | 1.864537–2.242676 |
| 2 | download | 131073 | 8 | 8/uniform | nearest | parallel | 18.814800 | 12.538250 | 1.471754 | 1.393094–1.553257 |
| 2 | cpu | 524288 | 1 | 1/uniform | nearest | parallel | 39.148800 | 15.590900 | 2.575911 | 2.369806–2.800679 |
| 2 | download | 524288 | 1 | 1/uniform | nearest | parallel | 42.344400 | 22.177950 | 1.985643 | 1.832906–2.155540 |
| 2 | cpu | 65537 | 8 | 0/uniform | radius | serial | 0.699900 | 0.801300 | 0.929194 | 0.873520–0.986294 |
| 2 | download | 65537 | 8 | 0/uniform | radius | serial | 6.429150 | 6.628500 | 0.973350 | 0.945953–1.004427 |
| 2 | cpu | 65537 | 8 | 1/uniform | radius | serial | 4.672400 | 4.478550 | 1.030577 | 0.929109–1.142815 |
| 2 | download | 65537 | 8 | 1/uniform | radius | serial | 11.008100 | 11.623800 | 0.974020 | 0.906502–1.050075 |
| 2 | cpu | 65537 | 0 | 0/uniform | radius | serial | 0.599800 | 0.658700 | 0.954775 | 0.911961–0.992224 |
| 2 | download | 65537 | 0 | 0/uniform | radius | serial | 4.894800 | 4.917400 | 0.994972 | 0.953770–1.054571 |
| 2 | cpu | 65537 | 65 | 9/uniform | radius | parallel | 6.802200 | 3.833900 | 1.920949 | 1.785376–2.074764 |
| 2 | download | 65537 | 65 | 9/uniform | radius | parallel | 23.872550 | 18.458500 | 1.249591 | 1.217989–1.279748 |
| 2 | cpu | 65537 | 65 | 1/uniform | radius | serial | 7.031550 | 5.285600 | 1.157192 | 1.065147–1.252831 |
| 2 | download | 65537 | 65 | 1/uniform | radius | serial | 22.286100 | 22.567050 | 1.003598 | 0.971070–1.040445 |
| 2 | cpu | 65537 | 65 | 0/uniform | radius | serial | 1.154650 | 1.302900 | 0.854105 | 0.784045–0.930045 |
| 2 | download | 65537 | 65 | 0/uniform | radius | serial | 11.000900 | 10.864450 | 1.005137 | 0.987275–1.025003 |
| 2 | cpu | 65537 | 65 | 65/uniform | radius | parallel | 15.438950 | 8.320200 | 1.837962 | 1.748826–1.933930 |
| 2 | download | 65537 | 65 | 65/uniform | radius | parallel | 30.453650 | 22.895000 | 1.413632 | 1.353129–1.478050 |
| 2 | cpu | 65537 | 16 | 16/mixed8to16 | radius | parallel | 6.908200 | 3.692050 | 1.849758 | 1.729960–1.981671 |
| 2 | download | 65537 | 16 | 16/mixed8to16 | radius | parallel | 14.548000 | 10.510200 | 1.381376 | 1.326545–1.437510 |
| 2 | cpu | 65537 | 64 | 64/first-quarter | radius | serial | 4.210250 | 4.782500 | 0.884999 | 0.818957–0.956399 |
| 2 | download | 65537 | 64 | 64/first-quarter | radius | serial | 19.804750 | 20.190600 | 0.979598 | 0.956912–1.001873 |
| 2 | full | 1025 | 8 | search/uniform | nearest | serial | 1.463800 | 1.427650 | 1.046050 | 0.997353–1.105464 |
| 2 | full | 65535 | 8 | search/uniform | nearest | serial | 10.187650 | 10.334400 | 0.963409 | 0.920695–1.006606 |
| 2 | full | 65536 | 8 | search/uniform | nearest | parallel | 9.995500 | 6.519300 | 1.636224 | 1.568529–1.714763 |
| 2 | full | 65537 | 8 | search/uniform | nearest | parallel | 13.365900 | 8.619350 | 1.501780 | 1.435944–1.572997 |
| 2 | full | 65537 | 16 | search/uniform | nearest | parallel | 29.590350 | 23.477650 | 1.311450 | 1.277633–1.349414 |
| 2 | full | 131073 | 8 | search/uniform | nearest | parallel | 27.993800 | 18.693500 | 1.560344 | 1.489384–1.633617 |
| 2 | full | 65537 | 8 | search/uniform | radius | parallel | 13.865300 | 9.513200 | 1.415970 | 1.343764–1.486449 |
| 2 | full | 65537 | 8 | search/uniform (zero radius) | radius | serial | 12.318550 | 12.488850 | 1.017941 | 0.943862–1.095466 |
| 2 | full | 65537 | 0 | search/uniform | radius | serial | 5.627700 | 5.732300 | 1.004524 | 0.962338–1.052332 |

## Initial candidate, retained in full

The first candidate used a shared per-row `Result<Vec<Neighbor>>` helper even in
serial decoding. Its selected parallel cases improved, but serial controls
regressed. The delivered serial path restores the original in-loop collector;
the delivered policy additionally rejects concentrated sampled work. Final mixed
and concentrated fixtures were added without dropping original cases. Initial
results are exploratory evidence and use a different executable; they are not
pooled into final estimates. During final preparation, the native-only per-row
helper was also excluded from WASM after its unused-function warning was
confirmed with warnings denied. The task-owned native build was stopped before
that source edit; both WASM configurations now run before the final native build.
The interrupted build produced no measurement and is not counted as validation.
The initial source/executable snapshot and all logs
and receipts are preserved and hash-checked in the audit.

| Route | Policy | Favorable | Inconclusive | Adverse | Unresolved | Paired old/new ratio range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu | parallel policy | 16 | 0 | 0 | 0 | 1.6136–2.2591 |
| cpu | serial control | 0 | 3 | 11 | 0 | 0.6251–1.0050 |
| download | parallel policy | 16 | 0 | 0 | 0 | 1.2340–1.8911 |
| download | serial control | 0 | 6 | 8 | 0 | 0.9154–1.0030 |
| full | parallel policy | 10 | 0 | 0 | 0 | 1.2743–1.6177 |
| full | serial control | 1 | 5 | 2 | 0 | 0.9111–1.0393 |

| Survey | Route | Queries | Capacity | Stored / distribution | Kind | Policy | Old median ms | New median ms | Ratio | Paired 95% interval |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | cpu | 1025 | 8 | 8/uniform | nearest | serial | 0.051900 | 0.055900 | 0.934828 | 0.914477–0.965973 |
| 1 | download | 1025 | 8 | 8/uniform | nearest | serial | 1.242000 | 1.208300 | 1.003004 | 0.971834–1.032456 |
| 1 | cpu | 65535 | 8 | 8/uniform | nearest | serial | 6.285950 | 6.546300 | 0.953854 | 0.910395–0.987727 |
| 1 | download | 65535 | 8 | 8/uniform | nearest | serial | 8.852450 | 9.068650 | 0.980359 | 0.936676–1.034526 |
| 1 | cpu | 65536 | 8 | 8/uniform | nearest | parallel | 8.761700 | 3.799450 | 2.109061 | 1.920418–2.323020 |
| 1 | download | 65536 | 8 | 8/uniform | nearest | parallel | 11.784450 | 7.792850 | 1.550604 | 1.465584–1.641577 |
| 1 | cpu | 65537 | 8 | 8/uniform | nearest | parallel | 7.579200 | 3.904750 | 2.017018 | 1.857490–2.197377 |
| 1 | download | 65537 | 8 | 8/uniform | nearest | parallel | 12.637750 | 7.975800 | 1.505778 | 1.421485–1.594007 |
| 1 | cpu | 65537 | 16 | 16/uniform | nearest | parallel | 10.905100 | 4.566650 | 2.192174 | 2.002137–2.389037 |
| 1 | download | 65537 | 16 | 16/uniform | nearest | parallel | 17.771050 | 11.403100 | 1.489659 | 1.425770–1.554717 |
| 1 | cpu | 65537 | 64 | 64/uniform | nearest | parallel | 16.054600 | 7.938150 | 2.047628 | 1.884754–2.235144 |
| 1 | download | 65537 | 64 | 64/uniform | nearest | parallel | 37.682700 | 22.614600 | 1.520852 | 1.435688–1.608514 |
| 1 | cpu | 131073 | 8 | 8/uniform | nearest | parallel | 11.938050 | 5.352600 | 2.126652 | 1.939292–2.332606 |
| 1 | download | 131073 | 8 | 8/uniform | nearest | parallel | 18.271900 | 12.433850 | 1.452761 | 1.382552–1.528673 |
| 1 | cpu | 524288 | 1 | 1/uniform | nearest | parallel | 34.601500 | 16.022650 | 2.259078 | 2.044784–2.499967 |
| 1 | download | 524288 | 1 | 1/uniform | nearest | parallel | 43.235300 | 22.491550 | 1.891118 | 1.788662–2.002560 |
| 1 | cpu | 65537 | 8 | 0/uniform | radius | serial | 0.815100 | 1.188750 | 0.688470 | 0.660713–0.719520 |
| 1 | download | 65537 | 8 | 0/uniform | radius | serial | 7.013350 | 6.963000 | 0.971554 | 0.945855–0.996514 |
| 1 | cpu | 65537 | 8 | 1/uniform | radius | serial | 5.013700 | 5.388900 | 0.916412 | 0.829200–1.010784 |
| 1 | download | 65537 | 8 | 1/uniform | radius | serial | 11.698500 | 12.060350 | 0.977009 | 0.918917–1.041871 |
| 1 | cpu | 65537 | 0 | 0/uniform | radius | serial | 0.643600 | 1.079150 | 0.634677 | 0.605526–0.666737 |
| 1 | download | 65537 | 0 | 0/uniform | radius | serial | 5.322400 | 5.711450 | 0.917799 | 0.892536–0.942632 |
| 1 | cpu | 65537 | 65 | 9/uniform | radius | parallel | 7.271350 | 4.075700 | 1.825136 | 1.686260–1.981492 |
| 1 | download | 65537 | 65 | 9/uniform | radius | parallel | 23.656250 | 18.799300 | 1.248495 | 1.222939–1.273567 |
| 1 | cpu | 65537 | 65 | 1/uniform | radius | serial | 7.625500 | 9.087300 | 0.844274 | 0.786335–0.905525 |
| 1 | download | 65537 | 65 | 1/uniform | radius | serial | 21.688050 | 23.520400 | 0.932808 | 0.904196–0.962257 |
| 1 | cpu | 65537 | 65 | 0/uniform | radius | serial | 1.065300 | 1.616400 | 0.665350 | 0.602400–0.740192 |
| 1 | download | 65537 | 65 | 0/uniform | radius | serial | 15.843450 | 16.509800 | 0.942724 | 0.910792–0.968523 |
| 1 | cpu | 65537 | 65 | 65/uniform | radius | parallel | 13.764850 | 7.553500 | 1.771782 | 1.684244–1.867053 |
| 1 | download | 65537 | 65 | 65/uniform | radius | parallel | 26.587300 | 21.426250 | 1.324097 | 1.283449–1.365454 |
| 1 | full | 1025 | 8 | search/uniform | nearest | serial | 1.287800 | 1.288800 | 0.993097 | 0.959442–1.030661 |
| 1 | full | 65535 | 8 | search/uniform | nearest | serial | 8.714950 | 8.775500 | 0.980150 | 0.951010–1.008393 |
| 1 | full | 65536 | 8 | search/uniform | nearest | parallel | 8.514100 | 5.719600 | 1.479759 | 1.439434–1.519164 |
| 1 | full | 65537 | 8 | search/uniform | nearest | parallel | 10.649000 | 6.029700 | 1.613067 | 1.517526–1.713254 |
| 1 | full | 65537 | 16 | search/uniform | nearest | parallel | 17.279550 | 12.930100 | 1.319182 | 1.247672–1.384361 |
| 1 | full | 131073 | 8 | search/uniform | nearest | parallel | 27.132350 | 18.496850 | 1.471724 | 1.410568–1.535327 |
| 1 | full | 65537 | 8 | search/uniform | radius | parallel | 13.196150 | 9.286000 | 1.385193 | 1.324166–1.446948 |
| 1 | full | 65537 | 8 | search/uniform (zero radius) | radius | serial | 11.037650 | 11.177450 | 0.985396 | 0.936727–1.034092 |
| 1 | full | 65537 | 0 | search/uniform | radius | serial | 5.981850 | 6.499950 | 0.911085 | 0.874507–0.951064 |
| 2 | cpu | 1025 | 8 | 8/uniform | nearest | serial | 0.047900 | 0.052400 | 0.908743 | 0.888288–0.924997 |
| 2 | download | 1025 | 8 | 8/uniform | nearest | serial | 1.192450 | 1.198000 | 0.977804 | 0.948665–1.004660 |
| 2 | cpu | 65535 | 8 | 8/uniform | nearest | serial | 6.113250 | 6.413600 | 1.004973 | 0.962231–1.059682 |
| 2 | download | 65535 | 8 | 8/uniform | nearest | serial | 8.542800 | 8.734100 | 0.968285 | 0.931559–1.008047 |
| 2 | cpu | 65536 | 8 | 8/uniform | nearest | parallel | 5.669000 | 3.315500 | 1.807717 | 1.659194–1.983882 |
| 2 | download | 65536 | 8 | 8/uniform | nearest | parallel | 10.063850 | 6.154850 | 1.653582 | 1.563505–1.739665 |
| 2 | cpu | 65537 | 8 | 8/uniform | nearest | parallel | 5.158800 | 3.318000 | 1.613612 | 1.515703–1.726027 |
| 2 | download | 65537 | 8 | 8/uniform | nearest | parallel | 11.639300 | 7.378650 | 1.529112 | 1.457633–1.603051 |
| 2 | cpu | 65537 | 16 | 16/uniform | nearest | parallel | 7.580300 | 4.131850 | 1.957738 | 1.819257–2.111505 |
| 2 | download | 65537 | 16 | 16/uniform | nearest | parallel | 16.082100 | 10.568750 | 1.474780 | 1.407541–1.540189 |
| 2 | cpu | 65537 | 64 | 64/uniform | nearest | parallel | 15.673600 | 8.498650 | 1.950888 | 1.740533–2.182842 |
| 2 | download | 65537 | 64 | 64/uniform | nearest | parallel | 30.678650 | 22.866750 | 1.337659 | 1.278596–1.400353 |
| 2 | cpu | 131073 | 8 | 8/uniform | nearest | parallel | 9.079350 | 5.326600 | 1.803479 | 1.679515–1.947368 |
| 2 | download | 131073 | 8 | 8/uniform | nearest | parallel | 19.996150 | 12.340650 | 1.572025 | 1.488467–1.658367 |
| 2 | cpu | 524288 | 1 | 1/uniform | nearest | parallel | 32.995900 | 14.592950 | 2.194821 | 2.040703–2.360325 |
| 2 | download | 524288 | 1 | 1/uniform | nearest | parallel | 40.513700 | 22.164750 | 1.819630 | 1.730012–1.915643 |
| 2 | cpu | 65537 | 8 | 0/uniform | radius | serial | 0.714600 | 1.037900 | 0.677663 | 0.646957–0.709226 |
| 2 | download | 65537 | 8 | 0/uniform | radius | serial | 6.403000 | 6.733400 | 0.969612 | 0.946395–0.996942 |
| 2 | cpu | 65537 | 8 | 1/uniform | radius | serial | 3.640400 | 4.109850 | 0.910437 | 0.805212–1.023120 |
| 2 | download | 65537 | 8 | 1/uniform | radius | serial | 10.716300 | 10.344150 | 0.968412 | 0.910728–1.029149 |
| 2 | cpu | 65537 | 0 | 0/uniform | radius | serial | 0.698400 | 1.122250 | 0.625077 | 0.592494–0.659188 |
| 2 | download | 65537 | 0 | 0/uniform | radius | serial | 5.506000 | 6.275100 | 0.915391 | 0.882318–0.947599 |
| 2 | cpu | 65537 | 65 | 9/uniform | radius | parallel | 8.736900 | 4.349250 | 1.985064 | 1.856524–2.123461 |
| 2 | download | 65537 | 65 | 9/uniform | radius | parallel | 23.981550 | 19.382300 | 1.233983 | 1.207820–1.262892 |
| 2 | cpu | 65537 | 65 | 1/uniform | radius | serial | 7.432050 | 8.902550 | 0.836134 | 0.776890–0.895632 |
| 2 | download | 65537 | 65 | 1/uniform | radius | serial | 22.960650 | 24.307850 | 0.937742 | 0.912154–0.965247 |
| 2 | cpu | 65537 | 65 | 0/uniform | radius | serial | 1.617700 | 2.340650 | 0.639272 | 0.589713–0.692640 |
| 2 | download | 65537 | 65 | 0/uniform | radius | serial | 15.695550 | 16.832850 | 0.939442 | 0.920630–0.957763 |
| 2 | cpu | 65537 | 65 | 65/uniform | radius | parallel | 19.176450 | 9.407250 | 2.072443 | 1.938684–2.189341 |
| 2 | download | 65537 | 65 | 65/uniform | radius | parallel | 35.164900 | 24.463950 | 1.437751 | 1.380834–1.501540 |
| 2 | full | 1025 | 8 | search/uniform | nearest | serial | 1.720350 | 1.653550 | 1.039311 | 1.002547–1.077994 |
| 2 | full | 65535 | 8 | search/uniform | nearest | serial | 11.620100 | 11.864000 | 0.990318 | 0.951897–1.027873 |
| 2 | full | 65536 | 8 | search/uniform | nearest | parallel | 12.943350 | 7.972700 | 1.617657 | 1.524928–1.717185 |
| 2 | full | 65537 | 8 | search/uniform | nearest | parallel | 15.084750 | 9.389700 | 1.586330 | 1.517165–1.655514 |
| 2 | full | 65537 | 16 | search/uniform | nearest | parallel | 30.485750 | 24.292750 | 1.274259 | 1.243268–1.307296 |
| 2 | full | 131073 | 8 | search/uniform | nearest | parallel | 30.755350 | 20.737650 | 1.535540 | 1.468041–1.605212 |
| 2 | full | 65537 | 8 | search/uniform | radius | parallel | 16.418450 | 11.708200 | 1.365282 | 1.291960–1.442491 |
| 2 | full | 65537 | 8 | search/uniform (zero radius) | radius | serial | 14.889650 | 14.896000 | 1.016921 | 0.955878–1.075684 |
| 2 | full | 65537 | 0 | search/uniform | radius | serial | 6.851000 | 7.268850 | 0.927189 | 0.881167–0.974294 |

## Local evidence paths

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-validation.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-provenance.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-initial-snapshot/files.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-initial-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-packed-survey-1.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-packed-survey-2.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-full-survey-1.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/neighbor-readback-full-survey-2.json`

These ignored paths are local provenance. The complete tables above are the
published evidence; reproducible fixture and benchmark sources are in the PR.
