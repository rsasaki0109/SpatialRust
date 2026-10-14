# Explicit GPU feature input gathering

Date: 2026-10-14. Parent: `0efb7b3f70ff71d9611069c036aa2722dc0cc1db` (merged PR 199).

## Change and boundary

`GpuFpfhInput::upload_with_gpu_gather`, `GpuIssInput::upload_with_gpu_gather`
and `GpuMlsInput::upload_with_gpu_gather` explicitly select shared GPU tree/query
construction. Original SoA axes and the CPU tree's point permutation are sent
once; an integer GPU pass produces traversal-order tree points and original-order
padded queries. ISS and MLS reuse queries as their position buffer, as before.

FPFH also uploads borrowed original-order raw XYZ normals and uses a second
integer GPU pass to assemble its existing 32-byte position/normal samples from
resident queries and normals. This avoids uploading duplicated positions and
normal padding. All coordinate/normal bits and zero padding are preserved;
neither preparation shader performs floating-point arithmetic.

For N points and M nodes, logical payload and staging sizes are:

| Input | Default host upload | Gather host upload | Gather device output bytes | Removed host packing allocation payload (sum) | Extra transient GPU input bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPFH | 64*N + 32*M | 28*N + 32*M | 64*N | 64*N | Up to 28*N |
| ISS / MLS | 32*N + 32*M | 16*N + 32*M | 32*N | 32*N | 16*N |

FPFH point payload falls by 56.25%, while ISS/MLS point payload halves. Node
payload is unchanged, so these percentages do not apply to total upload bytes.
Removed host packing bytes sum the eliminated point-sized allocations over
construction; they are not a peak-memory reduction. Default point, query and
sample staging happen at different times. Node staging remains on both paths.
Device output accounting counts tree points/queries and, for FPFH, samples.
Uniform control data is excluded. Owned output sizes and buffer usage are
unchanged. Backend staging and allocator overhead are additional to the logical
buffer sizes above. FPFH's two separately submitted passes may retain both sets
of temporary inputs simultaneously, hence the upper bound of 28*N device bytes.

No readback, host completion wait or automatic CPU fallback occurs. Queue order
makes generated queries available to the sample pass and subsequent feature
kernels. Outputs are owned and reusable; callers may drop CPU trees, coordinate
storage and normals after construction. The sample pipeline initializes lazily
per runtime, including simultaneous first callers. Empty inputs allocate no
buffers or pipelines. CPU tree construction/partitioning, finite coordinates,
unit normals, finite extent checks and node packing remain on CPU.

Default constructors delegate to a shared private body with the GPU option
disabled. A source comparison reconstructs that disabled branch and checks the
entire default body against the parent after formatting/comment normalization.
Validation order, feature device planners and all post-upload computation bodies
are checked unchanged. The FPFH planner's 132*N output bound also bounds 32*N
sample bytes, 12*N raw normal bytes and 3*N shader indexing; its dispatch check
matches the new sample pass. No new device requirement, dependency, unsafe
boundary or core type is introduced. Existing high-level feature placement is
preserved; small inputs can be slower and the new APIs require explicit choice.

Production paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/iss.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/mls.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/fpfh_sample_gather.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/fpfh_sample_gather.wgsl`

## Correctness and integration

Native integer probes compare generated original-order queries and FPFH samples
against the default constructors and independent coordinate/normal/padding-bit
expectations. Cases include empty/small/partial dispatch and CPU worker bands
through 524,289 points, bounded finite coordinates, signed zeros, subnormals and
constant maximum finite coordinates. Unit normals include axis directions,
0.6/0.8 components, signed zero and signed subnormal components. Receipts,
counts, empty pipeline allocation and CPU input drop are checked.

Resident output comparisons cover FPFH descriptors, ISS saliency/selection and
MLS orders 0/1/2. Raw result and status words, download errors/diagnostics and
transfers remain exact, including capacity overflow and numerical underflow.
Foreign runtimes and neighborhoods from another input reject. GPU inputs are
dropped before owned outputs are inspected. Nonfinite XYZ, overflowing extent,
normal length/unit/finite errors match default errors before either preparation
pipeline initializes, inside a clean device-validation scope. Concurrent cold
FPFH constructors retain independent sample outputs on one runtime.

Seven feature-input native release tests passed, including the three existing
SoA regressions. New probes verify 68,195,040 query/sample words, 60 resident
output comparisons / 51,432 words, 35 exact default errors and cold concurrent
sample generation. Related native groups passed 47 tests: 30 nearest/tree/query,
three FPFH, four ISS, four MLS and six registration geometry/SoA. Eleven public
facade tests passed, including the new shared input/output/receipt test and
existing high-level FPFH/ISS/MLS, neighborhood, ICP/trimmed ICP, GICP and
point-to-plane tests. All-feature/all-target GPU Clippy with warnings denied,
CPU-only, WASM, formatting and diff checks passed. Only the benchmark harness
changed after correctness checks; its expanded release build and Clippy passed.

The source receipt pins 314 source/shader/executable files, checks 303
preexisting GPU source files unchanged (including all 96 existing shaders), and
verifies default upload, device-planning and post-upload computation bodies
against the parent. New sample packing is the only production shader addition.

Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --features gpu-image --release --lib feature_soa_upload_tests -- --include-ignored --skip benchmark --nocapture --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo check -p spatialrust-gpu --no-default-features
cargo check -p spatialrust --no-default-features --features register-fpfh-gpu,feature-iss-gpu,filter-mls-gpu,register-icp-gpu,register-gicp-gpu,register-icp-point-to-plane-gpu --target wasm32-unknown-unknown
```

Local logs and source/executable receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/feature-gpu-*`.

## Completed-upload measurement

The ignored `completed_default_and_gpu_feature_uploads` test compares default
and explicit GPU constructors for all three inputs over ten size/profile
cases. Six routes rotate cyclically for three warmup and 30 recorded rounds.
Fixture creation, CPU KD-tree construction and runtime setup are outside timing.
Each interval includes limits, finite/normal/extent validation, applicable host
packing, node staging, every device buffer creation and submission, `wait_idle`,
receipt/count observation and output drop. Assertions follow timing. These are
warm-pipeline input-construction measurements; first shader compilation, searches,
descriptor/selection/fitting kernels and complete algorithms are outside timing.

Profiles are a bounded ordered grid with small Z variation (0), alternating
zero signs (1), signed subnormals (2) and the same grid with deterministically
shuffled original input order (4), with the same varied valid normals in every
route. Profile 4 uses xorshift64/Fisher-Yates with initial state
`0x0123456789abcdef`; it preserves the point set at each matched size. The
524,289/1,048,577-point ordered/shuffled pairs help assess input-order
dependence. Unused fixture point/normal vectors are dropped before timing.
Two sequential surveys run with no overlapping build/test/benchmark processes.
The controller requires a quiet window, monitors overlap and discards the whole
attempt while stopping only its own child if overlap appears. Analysis starts
only after both accepted surveys finish. All adverse and inconclusive cases
are retained.

The first seven-case survey pair did not establish repeatable latency gains
for the ordered grid, despite deterministic transfer reductions. That motivated
larger and shuffled comparisons. Its complete 42 comparisons, original sources,
executable and receipts are archived at
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/feature-gpu-initial/`.
The expanded harness additionally releases unused fixture vectors before
timing; production implementation, correctness tests and feature computations
are unchanged. Initial results are retained separately from the final expanded
source/executable measurements rather than pooled with them.


Final environment: native release, `rustc 1.98.1 (48a229cea 2026-09-01)`,
Vulkan discrete-GPU backend, 12 available CPU workers (at most four for shared
large point/query finite scans/CPU staging). Storage binding limit is
2,147,483,647 bytes, dispatch-axis limit 65,535 groups, and the existing feature
planners require five/six storage bindings for FPFH/ISS/MLS respectively. Both
expanded surveys were accepted on attempt 1 with no overlap and empty stderr;
durations were 39.90 and 31.92 seconds. Source/executable hashes matched before
and after both surveys. No local adapter model is recorded.

Ratios are paired geometric means of old/new duration; values above one mean
shorter GPU input calls. Pointwise 95% intervals use 10,000 paired log-ratio
bootstrap resamples with seed `2026101429 + 100*survey + 10*case + field_index`
(JSON record field index), sorted entries 249/9749. Every comparison has 30
samples. A zero sample would suppress its ratio and retain medians/zero counts;
no initial or expanded comparison contains zero samples. Initial and expanded
tables use their own within-survey pairs and are not pooled.

For 1,048,577 points with shuffled input order (profile 4):

- survey 1, fpfh_gpu: 64.817800 to 45.453100 ms (29.9% median reduction); ratio 1.407537 [1.313372, 1.504870].
- survey 1, iss_gpu: 42.776750 to 31.424850 ms (26.5% median reduction); ratio 1.322806 [1.216430, 1.435614].
- survey 1, mls_gpu: 42.283750 to 29.759350 ms (29.6% median reduction); ratio 1.324162 [1.223935, 1.428700].
- survey 2, fpfh_gpu: 48.000700 to 34.779250 ms (27.5% median reduction); ratio 1.371223 [1.242041, 1.509547].
- survey 2, iss_gpu: 31.024100 to 22.657200 ms (27.0% median reduction); ratio 1.337395 [1.213187, 1.474683].
- survey 2, mls_gpu: 29.695050 to 23.660000 ms (20.3% median reduction); ratio 1.269251 [1.142985, 1.390109].

All six intervals for this case are above one. Observed median input-call
reductions are roughly 28–30% for FPFH, 27% for ISS and 20–30% for MLS. These
specific shuffled-input results establish a useful measured case for the new
constructors, not universal feature or end-to-end algorithm acceleration.

Small 1/4,097-point calls regress for all three inputs in both final surveys.
At 524,289 ordered points all three improve in expanded survey 1 but regress
in expanded survey 2; shuffled 524,289-point results also reverse. Ordered
1,048,577-point calls improve in survey 1 but remain inconclusive in survey 2.
The initial ordered-grid pair likewise did not establish stable gains. This
variation prevents a general size threshold or a causal claim that shuffling
alone guarantees faster calls. Synthetic zero/subnormal profiles are not
evidence of typical application-wide performance. All 22 expanded and
20 initial comparisons whose full interval is below one are retained,
alongside every inconclusive comparison. Default constructors and high-level
placement remain unchanged. The deterministic host-payload reductions and
documented temporary GPU-memory cost hold regardless of latency variation.

### Final expanded comparisons

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | fpfh_gpu | 0.022400 | 0.250550 | 0.088730 [0.083481, 0.094449] |
| 1 | 1 | 0 | iss_gpu | 0.023150 | 0.140400 | 0.156938 [0.146503, 0.168740] |
| 1 | 1 | 0 | mls_gpu | 0.019450 | 0.140250 | 0.138959 [0.127794, 0.151837] |
| 1 | 4097 | 0 | fpfh_gpu | 0.134350 | 0.333950 | 0.413193 [0.388087, 0.442058] |
| 1 | 4097 | 0 | iss_gpu | 0.092050 | 0.218900 | 0.392522 [0.364718, 0.418443] |
| 1 | 4097 | 0 | mls_gpu | 0.091900 | 0.226800 | 0.386858 [0.364962, 0.409581] |
| 1 | 65537 | 0 | fpfh_gpu | 3.918400 | 3.637600 | 1.103045 [1.005642, 1.207701] |
| 1 | 65537 | 0 | iss_gpu | 2.140250 | 2.697450 | 0.907793 [0.805734, 1.041206] |
| 1 | 65537 | 0 | mls_gpu | 2.180350 | 2.354100 | 1.000950 [0.889163, 1.130825] |
| 1 | 262145 | 0 | fpfh_gpu | 16.097050 | 12.836800 | 1.205221 [1.132575, 1.279817] |
| 1 | 262145 | 0 | iss_gpu | 9.775200 | 7.991850 | 1.144388 [1.062244, 1.234585] |
| 1 | 262145 | 0 | mls_gpu | 9.145900 | 9.154700 | 1.082586 [0.963476, 1.224608] |
| 1 | 524289 | 0 | fpfh_gpu | 30.830100 | 20.042300 | 1.502741 [1.414755, 1.590459] |
| 1 | 524289 | 0 | iss_gpu | 18.521950 | 14.333000 | 1.284077 [1.145019, 1.430224] |
| 1 | 524289 | 0 | mls_gpu | 18.012600 | 13.565600 | 1.420677 [1.301254, 1.565297] |
| 1 | 524289 | 1 | fpfh_gpu | 32.399900 | 22.505700 | 1.357320 [1.287065, 1.428927] |
| 1 | 524289 | 1 | iss_gpu | 19.947100 | 15.142600 | 1.282004 [1.173120, 1.397561] |
| 1 | 524289 | 1 | mls_gpu | 18.235150 | 14.627450 | 1.284005 [1.196995, 1.380211] |
| 1 | 524289 | 2 | fpfh_gpu | 37.347500 | 24.213050 | 1.511400 [1.403942, 1.621930] |
| 1 | 524289 | 2 | iss_gpu | 21.608700 | 15.842050 | 1.332776 [1.221088, 1.460324] |
| 1 | 524289 | 2 | mls_gpu | 21.210650 | 16.116700 | 1.360634 [1.225458, 1.513575] |
| 1 | 1048577 | 0 | fpfh_gpu | 62.108000 | 40.294200 | 1.471667 [1.312902, 1.620958] |
| 1 | 1048577 | 0 | iss_gpu | 35.303850 | 25.142950 | 1.311644 [1.224068, 1.403292] |
| 1 | 1048577 | 0 | mls_gpu | 34.646750 | 26.811050 | 1.275764 [1.197457, 1.368418] |
| 1 | 524289 | 4 | fpfh_gpu | 31.333900 | 21.952350 | 1.468026 [1.381864, 1.559621] |
| 1 | 524289 | 4 | iss_gpu | 20.415300 | 16.037700 | 1.320330 [1.196285, 1.470700] |
| 1 | 524289 | 4 | mls_gpu | 19.902100 | 14.726650 | 1.424911 [1.302363, 1.560709] |
| 1 | 1048577 | 4 | fpfh_gpu | 64.817800 | 45.453100 | 1.407537 [1.313372, 1.504870] |
| 1 | 1048577 | 4 | iss_gpu | 42.776750 | 31.424850 | 1.322806 [1.216430, 1.435614] |
| 1 | 1048577 | 4 | mls_gpu | 42.283750 | 29.759350 | 1.324162 [1.223935, 1.428700] |
| 2 | 1 | 0 | fpfh_gpu | 0.028900 | 0.238250 | 0.113567 [0.103844, 0.124227] |
| 2 | 1 | 0 | iss_gpu | 0.028250 | 0.162500 | 0.160809 [0.146676, 0.175824] |
| 2 | 1 | 0 | mls_gpu | 0.025950 | 0.161300 | 0.156536 [0.143137, 0.172550] |
| 2 | 4097 | 0 | fpfh_gpu | 0.111600 | 0.289250 | 0.405662 [0.386034, 0.425776] |
| 2 | 4097 | 0 | iss_gpu | 0.078950 | 0.235350 | 0.355126 [0.328557, 0.383985] |
| 2 | 4097 | 0 | mls_gpu | 0.075400 | 0.236700 | 0.340021 [0.322634, 0.357742] |
| 2 | 65537 | 0 | fpfh_gpu | 3.027600 | 2.867400 | 1.005448 [0.930865, 1.081038] |
| 2 | 65537 | 0 | iss_gpu | 1.696150 | 2.351550 | 0.829742 [0.729703, 0.958421] |
| 2 | 65537 | 0 | mls_gpu | 1.614750 | 1.922250 | 1.010312 [0.881263, 1.165843] |
| 2 | 262145 | 0 | fpfh_gpu | 10.441600 | 9.836900 | 1.080336 [1.012779, 1.149909] |
| 2 | 262145 | 0 | iss_gpu | 6.424300 | 6.547100 | 0.970704 [0.905006, 1.040954] |
| 2 | 262145 | 0 | mls_gpu | 6.028100 | 7.192200 | 0.952211 [0.860011, 1.069924] |
| 2 | 524289 | 0 | fpfh_gpu | 19.547500 | 22.631650 | 0.882241 [0.828440, 0.936564] |
| 2 | 524289 | 0 | iss_gpu | 11.847550 | 15.704850 | 0.775665 [0.705560, 0.855680] |
| 2 | 524289 | 0 | mls_gpu | 11.527750 | 15.813700 | 0.825123 [0.753659, 0.912636] |
| 2 | 524289 | 1 | fpfh_gpu | 20.298650 | 23.565850 | 0.878736 [0.835385, 0.922762] |
| 2 | 524289 | 1 | iss_gpu | 12.538700 | 17.124500 | 0.738335 [0.683859, 0.793554] |
| 2 | 524289 | 1 | mls_gpu | 11.727950 | 16.813450 | 0.806830 [0.734531, 0.899866] |
| 2 | 524289 | 2 | fpfh_gpu | 23.116150 | 19.762300 | 1.170423 [1.060248, 1.284987] |
| 2 | 524289 | 2 | iss_gpu | 14.627200 | 15.492300 | 1.028035 [0.929482, 1.133897] |
| 2 | 524289 | 2 | mls_gpu | 14.072500 | 12.657150 | 1.068042 [0.952289, 1.212469] |
| 2 | 1048577 | 0 | fpfh_gpu | 38.641650 | 41.515750 | 1.016735 [0.923367, 1.121905] |
| 2 | 1048577 | 0 | iss_gpu | 22.972900 | 25.912650 | 0.933296 [0.850248, 1.021977] |
| 2 | 1048577 | 0 | mls_gpu | 23.027800 | 26.275150 | 0.941428 [0.850838, 1.046729] |
| 2 | 524289 | 4 | fpfh_gpu | 23.042550 | 26.022050 | 0.895404 [0.845665, 0.952654] |
| 2 | 524289 | 4 | iss_gpu | 15.243500 | 20.606350 | 0.782296 [0.722981, 0.847016] |
| 2 | 524289 | 4 | mls_gpu | 14.895450 | 20.300200 | 0.828420 [0.759152, 0.908192] |
| 2 | 1048577 | 4 | fpfh_gpu | 48.000700 | 34.779250 | 1.371223 [1.242041, 1.509547] |
| 2 | 1048577 | 4 | iss_gpu | 31.024100 | 22.657200 | 1.337395 [1.213187, 1.474683] |
| 2 | 1048577 | 4 | mls_gpu | 29.695050 | 23.660000 | 1.269251 [1.142985, 1.390109] |

### Initial seven-case comparisons (separate source/executable)

| Survey | Points | Profile | Route | Old median ms | New median ms | Old/new paired ratio [95% CI] |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 1 | 0 | fpfh_gpu | 0.030250 | 0.250600 | 0.120995 [0.112389, 0.129767] |
| 1 | 1 | 0 | iss_gpu | 0.029350 | 0.165500 | 0.177282 [0.159989, 0.195280] |
| 1 | 1 | 0 | mls_gpu | 0.027200 | 0.173350 | 0.161186 [0.149159, 0.173707] |
| 1 | 4097 | 0 | fpfh_gpu | 0.189250 | 0.483500 | 0.403218 [0.381422, 0.426463] |
| 1 | 4097 | 0 | iss_gpu | 0.139250 | 0.325250 | 0.439500 [0.417168, 0.461813] |
| 1 | 4097 | 0 | mls_gpu | 0.132150 | 0.315850 | 0.405991 [0.384196, 0.428036] |
| 1 | 65537 | 0 | fpfh_gpu | 3.619150 | 3.243850 | 1.062764 [0.997693, 1.133341] |
| 1 | 65537 | 0 | iss_gpu | 1.904250 | 2.458150 | 0.880147 [0.788753, 0.997540] |
| 1 | 65537 | 0 | mls_gpu | 2.009050 | 2.186950 | 1.034762 [0.914277, 1.183512] |
| 1 | 262145 | 0 | fpfh_gpu | 14.635050 | 12.475950 | 1.069181 [0.982352, 1.152064] |
| 1 | 262145 | 0 | iss_gpu | 9.208950 | 9.471100 | 1.035702 [0.944243, 1.134093] |
| 1 | 262145 | 0 | mls_gpu | 8.870400 | 8.943550 | 1.044419 [0.915466, 1.220309] |
| 1 | 524289 | 0 | fpfh_gpu | 25.824800 | 25.892250 | 0.965567 [0.908968, 1.017500] |
| 1 | 524289 | 0 | iss_gpu | 15.326000 | 18.385850 | 0.813059 [0.726032, 0.905031] |
| 1 | 524289 | 0 | mls_gpu | 14.811700 | 17.501650 | 0.944718 [0.810758, 1.130576] |
| 1 | 524289 | 1 | fpfh_gpu | 24.100900 | 24.989700 | 0.946887 [0.893217, 1.003952] |
| 1 | 524289 | 1 | iss_gpu | 15.113500 | 18.834050 | 0.803567 [0.705914, 0.916545] |
| 1 | 524289 | 1 | mls_gpu | 14.300100 | 17.961650 | 0.893859 [0.805726, 0.996358] |
| 1 | 524289 | 2 | fpfh_gpu | 26.266200 | 27.817850 | 0.915450 [0.851958, 0.975965] |
| 1 | 524289 | 2 | iss_gpu | 17.637200 | 20.819150 | 0.826838 [0.757700, 0.899947] |
| 1 | 524289 | 2 | mls_gpu | 15.521750 | 20.405550 | 0.869406 [0.777012, 0.988683] |
| 2 | 1 | 0 | fpfh_gpu | 0.022050 | 0.224750 | 0.099763 [0.095197, 0.104377] |
| 2 | 1 | 0 | iss_gpu | 0.020600 | 0.127300 | 0.146716 [0.135083, 0.158877] |
| 2 | 1 | 0 | mls_gpu | 0.019950 | 0.134050 | 0.153752 [0.140603, 0.168222] |
| 2 | 4097 | 0 | fpfh_gpu | 0.125600 | 0.333350 | 0.384074 [0.368709, 0.402237] |
| 2 | 4097 | 0 | iss_gpu | 0.089900 | 0.221550 | 0.371268 [0.343868, 0.398753] |
| 2 | 4097 | 0 | mls_gpu | 0.088000 | 0.234600 | 0.364726 [0.342737, 0.388266] |
| 2 | 65537 | 0 | fpfh_gpu | 3.058300 | 3.053650 | 1.063634 [0.998437, 1.133209] |
| 2 | 65537 | 0 | iss_gpu | 1.854500 | 2.333000 | 0.881680 [0.778919, 1.011743] |
| 2 | 65537 | 0 | mls_gpu | 1.787650 | 1.849300 | 1.033292 [0.915605, 1.177780] |
| 2 | 262145 | 0 | fpfh_gpu | 11.708800 | 9.843000 | 1.196581 [1.151394, 1.243794] |
| 2 | 262145 | 0 | iss_gpu | 7.376100 | 6.777100 | 1.086899 [1.004468, 1.174737] |
| 2 | 262145 | 0 | mls_gpu | 6.981000 | 7.136050 | 1.105149 [0.992294, 1.253870] |
| 2 | 524289 | 0 | fpfh_gpu | 22.125300 | 20.821900 | 1.080659 [1.024020, 1.135769] |
| 2 | 524289 | 0 | iss_gpu | 13.782150 | 15.775800 | 0.865537 [0.805007, 0.932480] |
| 2 | 524289 | 0 | mls_gpu | 13.582150 | 15.763550 | 0.943007 [0.879642, 1.020321] |
| 2 | 524289 | 1 | fpfh_gpu | 27.366150 | 24.216100 | 1.118189 [1.011450, 1.245216] |
| 2 | 524289 | 1 | iss_gpu | 16.968750 | 18.374000 | 0.917042 [0.808133, 1.057555] |
| 2 | 524289 | 1 | mls_gpu | 15.936200 | 17.835950 | 0.979720 [0.881704, 1.092000] |
| 2 | 524289 | 2 | fpfh_gpu | 28.324700 | 27.610150 | 1.010689 [0.932712, 1.097883] |
| 2 | 524289 | 2 | iss_gpu | 17.077100 | 20.428900 | 0.918991 [0.813880, 1.057891] |
| 2 | 524289 | 2 | mls_gpu | 16.997000 | 20.458200 | 0.945166 [0.854846, 1.052172] |
