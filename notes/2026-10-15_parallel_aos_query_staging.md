# Bounded AoS GPU query staging

Date: 2026-10-15. Parent: `0093b8ef31edca43047c2fec903aa44c35ce8f1b` (merged PR 204).

## Change and boundary

`GpuQueryBatch::upload` shares private native CPU preparation for original-order
AoS queries. At least 524,288 points use available CPU count, useful
262,144-point bands and a four-worker cap. Scoped workers validate finite XYZ
and write disjoint padded rows. Coordinate bits, source order and positive-zero
padding remain exact. Small/one-worker/WASM calls retain the complete original
finite scan and exact-capacity collector. Small calls return before CPU-count
queries; WASM never starts workers.

The same 16*N-byte host payload remains, initially zeroed on the parallel path.
Worker handles and thread stacks add bounded overhead; no additional point-sized
copy is allocated. Large invalid inputs may allocate/zero that payload before
rejection, unlike the serial early rejection. Device buffer/dispatch limits
still precede host work. All scoped work joins successfully before GPU resource
creation, with the same invalid-argument errors. No new dependency, unsafe code,
GPU buffer, compute pass, wait, readback or transfer is introduced. Empty uploads
still allocate no GPU buffer. Upload receipts remain 16*N host-to-device bytes
and zero device-to-device/device-to-host payload bytes.

Registration source uploads share this preparation in both existing upload
routes. Public APIs, SoA packing, device limits, tree construction, normal and
covariance preparation, search/reduction shaders, and placement remain. This is
host preparation parallelism; it is not a whole-registration GPU residency or
end-to-end registration speed claim.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/aos_query_staging.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/neighborhoods.rs`

## Correctness and integration

Three pure and three native tests compare frozen PR 204 serial preparation and
complete public query upload against the new path. Empty, small, cutoff and
large sizes include 524,287/288/289 and 1,048,577 points. Forced available worker
counts 0/1/2/3/4/12/usize::MAX, signed zeros, subnormals, varied coordinates and
seeded arbitrary finite words check exact XYZ and positive-zero padding.
Nonfinite components at original-order worker edges retain exact errors.
Concurrent calls own disjoint outputs.

Integer GPU probes read actual STORAGE-only query inputs without changing
production buffer usages. Host input is dropped before probes. Nearest-k and
inclusive radius outputs compare words, decoded rows, total match counts,
truncation and wrong-runtime rejection. Large-source/small-target ICP compares
owned moment words after source/input destruction; the baseline constructor's
source is replaced with the frozen serial query upload, isolating this change.

Exact comparisons cover **300,943,552** packed host words,
**42,991,936** actual GPU query words, **22,022,924** search/owned-moment
output words and **516** frozen errors (480 host and 36 native). These are
word/error comparison counts, not distinct input points. The six new tests pass.

All **103** selected native release tests (42 shared search/staging, 24 ICP,
24 GICP, 13 point-to-plane) and **8** public facade tests passed. All-feature/
all-target Clippy with warnings denied, CPU-only and WASM facade checks,
formatting and diff checks passed. No new algorithmic tests are skipped in
these selections; measurement tests run separately.

Source verification allows only the query upload staging block and module
declarations to change in pre-existing GPU sources. The complete frozen upload
and serial helper are checked against PR 204. All **331** other pre-existing
GPU sources, including **97** shaders, remain unchanged. Survey manifests hash
**339** GPU source/WGSL, public smoke-test and executable files. Tracked/
nonignored files and the PR body have zero forbidden local adapter-name matches.

## Isolated measurements

Two sequential surveys run after all builds/tests and a 15-second quiet window.
A process census rejects an entire attempt if a build/test/other benchmark
overlaps, cancelling only the owned measurement child. Accepted runs require
one passing test, no overlap and empty stderr. Analysis starts after both runs
and verification of exact source/executable hashes.

Ten cases use three warm cycles and 30 paired samples per route with alternating
old/new order. `host` includes finite validation, allocation, packing and drop;
small calls below 524,288 points repeat 32 times per sample and report per-call
time. `upload` includes the complete public query upload, `wait_idle`, count/
receipt assertions and destruction. Runtime and fixture creation are excluded.
No neighborhood search, payload readback, moment reduction or pose solve is
timed. Profile 0 is varied bounded coordinates, 1 signed zeros, 2 signed
subnormals and 3 seeded arbitrary finite words.

Environment: native Windows x86_64, rustc 1.98.1 (`48a229cea`, 2026-09-01),
Vulkan discrete-device backend, 12 available
CPU workers with at most four selected; storage binding limit 2,147,483,647 bytes
and dispatch limit 65,535 groups. Timings are local measurements, not portable
throughput guarantees.

Ratios are geometric means of paired old/new sample ratios; greater than one
favors the new route. Percentile 95% intervals resample paired log ratios 10,000
times, seed `2026101505 + 100*survey + 10*case + JSON_field_index`, sorted bounds
249/9749. An interval crossing one is inconclusive. All adverse/inconclusive
comparisons are retained. Zero samples suppress ratios.

Both surveys were accepted on attempt 1 with no overlap and empty stderr
(test durations 13.16 and 14.51 seconds). All 40 comparisons have 30 nonzero
paired samples: 21 favorable intervals, 17 crossing one and two adverse.
At 1,048,577 varied points:

- Host preparation: medians 5.893550 → 3.345150 ms and 6.643600 → 3.492050 ms
  (43.2% and 47.4% lower); paired ratios 1.808529 [1.750693, 1.867420] and
  1.878226 [1.801093, 1.963647].
- Completed public query upload: 8.394450 → 5.504350 ms and 8.829000 →
  5.642700 ms (34.4% and 36.1% lower); paired ratios 1.810093
  [1.548042, 2.230406] and 1.640693 [1.488187, 1.875614].

At the parallel cutoff, host preparation improves in both surveys; completed
upload improvements are noisier and several intervals cross one. Signed-zero,
subnormal and arbitrary finite inputs remain in the table. Small calls retain
the original serial body, so their differences must not be attributed to worker
parallelism. Two host comparisons are adverse: 65,537 points in survey 1
(0.979173 [0.963170, 0.996584], medians 0.446181 → 0.458056 ms) and singleton
in survey 2 (0.973779 [0.964447, 0.983100], rounded medians 0.000055 →
0.000056 ms). Neither is adverse in the other survey. Compiler/call context,
measurement resolution and local conditions can still affect unchanged serial
loops. These small costs are retained explicitly. No whole registration,
search or reduction speedup is inferred from query-upload timings.

## Raw receipts

Ignored local evidence lives under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/`: `aos-query-*.log`,
`aos-query-provenance.json`, `aos-query-measurements.json`, `aos-query-table.md`,
and the measurement/analysis/CI guard scripts. These retain raw paired samples,
process-isolation acceptance, exact source/executable hashes and delivery state.

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | host | 0.000070 | 0.000072 | 0.992681 [0.981169, 1.003031] |
| 1 | 1 | 0 | upload | 0.009000 | 0.008800 | 1.026509 [0.900443, 1.170486] |
| 1 | 4097 | 0 | host | 0.011372 | 0.011339 | 0.966837 [0.887617, 1.030608] |
| 1 | 4097 | 0 | upload | 0.032500 | 0.030700 | 1.114612 [0.980784, 1.345889] |
| 1 | 65537 | 0 | host | 0.446181 | 0.458056 | 0.979173 [0.963170, 0.996584] |
| 1 | 65537 | 0 | upload | 0.787300 | 0.610100 | 1.261819 [1.188267, 1.340826] |
| 1 | 524287 | 0 | host | 2.719670 | 2.765634 | 1.028964 [0.973343, 1.085818] |
| 1 | 524287 | 0 | upload | 4.439000 | 4.264600 | 1.040928 [0.987168, 1.096897] |
| 1 | 524288 | 0 | host | 2.917050 | 2.208350 | 1.366958 [1.308453, 1.432484] |
| 1 | 524288 | 0 | upload | 4.059550 | 3.316700 | 1.145433 [0.901771, 1.318496] |
| 1 | 524289 | 0 | host | 2.981700 | 2.245150 | 1.303050 [1.254041, 1.356422] |
| 1 | 524289 | 0 | upload | 4.327150 | 3.409350 | 1.120100 [0.893479, 1.286761] |
| 1 | 1048577 | 0 | host | 5.893550 | 3.345150 | 1.808529 [1.750693, 1.867420] |
| 1 | 1048577 | 0 | upload | 8.394450 | 5.504350 | 1.810093 [1.548042, 2.230406] |
| 1 | 524289 | 1 | host | 2.923900 | 2.219350 | 1.307738 [1.267076, 1.353366] |
| 1 | 524289 | 1 | upload | 4.174450 | 3.264400 | 1.371445 [1.223345, 1.621423] |
| 1 | 524289 | 2 | host | 3.347300 | 2.488100 | 1.305049 [1.162795, 1.415288] |
| 1 | 524289 | 2 | upload | 6.999450 | 5.856150 | 1.044860 [0.852257, 1.199851] |
| 1 | 524289 | 3 | host | 3.305200 | 2.509350 | 1.341117 [1.292643, 1.395884] |
| 1 | 524289 | 3 | upload | 7.117900 | 6.132750 | 1.219030 [1.106383, 1.430914] |
| 2 | 1 | 0 | host | 0.000055 | 0.000056 | 0.973779 [0.964447, 0.983100] |
| 2 | 1 | 0 | upload | 0.007000 | 0.007100 | 1.062712 [0.908945, 1.241408] |
| 2 | 4097 | 0 | host | 0.007786 | 0.007802 | 1.005265 [0.962482, 1.055817] |
| 2 | 4097 | 0 | upload | 0.022050 | 0.020150 | 1.149492 [0.995439, 1.387945] |
| 2 | 65537 | 0 | host | 0.348391 | 0.342656 | 0.999833 [0.972382, 1.024744] |
| 2 | 65537 | 0 | upload | 0.731900 | 0.459050 | 1.354307 [1.242283, 1.478306] |
| 2 | 524287 | 0 | host | 2.910111 | 2.839486 | 1.029462 [0.975188, 1.088345] |
| 2 | 524287 | 0 | upload | 4.223850 | 4.003300 | 1.047760 [0.983247, 1.114169] |
| 2 | 524288 | 0 | host | 3.605350 | 2.553750 | 1.457346 [1.376645, 1.551962] |
| 2 | 524288 | 0 | upload | 4.687300 | 3.824900 | 1.180203 [1.007099, 1.314134] |
| 2 | 524289 | 0 | host | 3.125400 | 2.419350 | 1.281384 [1.242267, 1.330141] |
| 2 | 524289 | 0 | upload | 4.355250 | 3.553400 | 1.109407 [0.924753, 1.237064] |
| 2 | 1048577 | 0 | host | 6.643600 | 3.492050 | 1.878226 [1.801093, 1.963647] |
| 2 | 1048577 | 0 | upload | 8.829000 | 5.642700 | 1.640693 [1.488187, 1.875614] |
| 2 | 524289 | 1 | host | 3.341550 | 2.488000 | 1.319109 [1.267717, 1.376689] |
| 2 | 524289 | 1 | upload | 4.479300 | 3.473650 | 1.347203 [1.241754, 1.516638] |
| 2 | 524289 | 2 | host | 3.423000 | 2.454100 | 1.393872 [1.338308, 1.449889] |
| 2 | 524289 | 2 | upload | 6.436400 | 5.500300 | 1.101525 [0.942200, 1.221791] |
| 2 | 524289 | 3 | host | 3.011550 | 2.404200 | 1.271860 [1.231674, 1.315297] |
| 2 | 524289 | 3 | upload | 6.683150 | 5.772550 | 1.223835 [1.123402, 1.389472] |

