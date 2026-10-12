# Resident GPU MLS smoothing

Date: 2026-10-13. Named slice: constant, linear and quadratic Moving Least
Squares surface fits on GPU, following the resident radius-neighborhood API.

## Execution and transfer contract

`filter-mls-gpu` exposes
`MlsSmoothing::smoothed_positions_with_gpu(runtime, cloud, neighbor_capacity)`
and `filter_with_gpu` in filtering and the facade. Ordinary smoothing retains
its existing bounded CPU path. The explicit wrapper builds one CPU KD-tree,
uploads the tree and original queries, runs radius search and MLS on GPU, then
downloads positions/modes/status. Schema, metadata and all non-position fields
are reconstructed on CPU using the existing output builder. There is no
intermediate neighborhood/frame/system readback or automatic CPU fallback.

`GpuMlsInput::upload` under `gpu-wgpu` owns reusable input on one runtime. Query
positions also serve as the frame samples, without another sample upload.
For N original points and M tree nodes, H2D payload is `32*N + 32*M` bytes:
16*N tree positions, 16*N queries and 32*M nodes. CPU tree construction is
external to this resident API. `neighborhoods_resident` submits radius search
without readback or host completion wait. Rows are tied to that exact input;
passing a different input's neighborhoods or a different runtime rejects.

Capacity and minimum support include self. Every radius match must fit every
row, even rows below minimum support; truncated prefixes reject. After that
support test, fitting excludes only the query index, retaining coincident
points at other indices. Distances use an inclusive radius gate and Gaussian
weights `exp(-distance_squared / max((radius/2)^2, 1e-12))`.

`smooth_resident` submits three ordered device passes in one submission:

1. Validate counts, indices and distance gates; prepare weighted centroid,
   covariance, smallest-eigenvector normal and orthonormal local frame.
2. Fit and evaluate the requested constant/linear/quadratic height field.
3. Finalize validity, zeroing ALL output if any row or numerical error occurred.

Frames use query-relative coordinates and actual local extent for scaling,
compensated f32 weight/centroid/covariance sums, and max-pivot Jacobi with
32 iterations and the physical covariance stopping rule of 1e-12. Centroids
are reconstructed as absolute f32 XYZ before covariance/fitting. The constant
fit is a scalar weighted mean height. Linear and quadratic fits use normalized
bases `[1,s,t]` and `[1,s,t,s*s,s*t,t*t]`, direct f32 Gram/RHS accumulation and
in-place Cholesky. A relative Schur-complement threshold of
`1e-6 * largest_absolute_Gram_entry` selects the plane on rank failure.

Sparse/zero-weight/zero-extent points retain exact original coordinates.
Insufficient polynomial support or failed rank tests produce plane projections
on device. No point is fitted by a CPU fallback. Input XYZ and global extent
squares must be finite; radius must be positive with a finite nonzero f32
square. An evaluated nonzero local extent with a subnormal/zero f32 square
sets global numerical-error status. These checks are stricter than ordinary
CPU MLS. Huge usize support requirements map to u32::MAX.

Per-runtime lazy pipelines share six storage bindings and one 32-byte uniform,
with 64 invocations per workgroup. Private scratch leases contain 64*N frame
bytes and 192*N system bytes; both are allocated for every nonempty order.
Consumed scratch rows are initialized before use. Leases remain exclusive
through submission, then return to the existing byte-bounded pool. Result
buffers are owned separately and never pooled. All sizes, u32 indexing, host
bounds, shader bindings and dispatch limits are checked before allocation.

`GpuMlsPositions` owns original-order vec4<f32> rows `(XYZ,mode)`, where mode
0 means unchanged, 1 plane fallback and 2 the requested polynomial. Empty
physical position storage has 16 unused bytes. Eight u32 status words contain
bad rows, required capacity, malformed rows, last bad row plus one, numerical
errors, unchanged count, plane count and polynomial count. The FIRST FIVE
words must all be zero. The last three are outcome diagnostics, not errors;
their sum must equal N. All XYZ/mode words are zeroed on global errors. Custom
GPU consumers must inspect status and treat exposed storage as immutable.

`download` explicitly copies position/mode/status in one staging submission,
waits, validates status before decoding and verifies finite XYZ, valid modes
and matching counts. D2H payload is `16*N + 32` bytes. Empty H2D/D2H transfers
are both zero. Uniform controls and private scratch are excluded from payload
receipts. Native Vulkan execution is verified; WASM evidence is compilation
only, without a browser synchronous-download claim. No core dependency or
unsafe code is added.

## Precision and correctness evidence

GPU MLS is opt-in and DOES NOT promise CPU coordinate or fallback parity.
CPU MLS uses unnormalized f64 Gram/RHS sums and Gaussian elimination with an
absolute pivot threshold of 1e-12; the GPU route uses normalized f32 Cholesky
with the relative threshold above. Eigenvectors, near-rank fits and fallback
decisions can differ. Per-point modes/counts expose the actual GPU outcome.
Use the existing CPU API when its exact decisions are required.

Four explicitly run adapter integration tests cover 0/1/2/17/63/64/65/257/4097
points, five noisy/curved/coincident/planar/skinny fixture kinds and orders
0/1/2: 135 conditions and 205,470 coordinates. Each coordinate is checked
against independent existing CPU math for its REPORTED outcome: exact original
XYZ for unchanged, CPU order-0 weighted plane for plane fallback, and the CPU
requested order for polynomial fits. The selected-fixture coordinate bound is
`5e-5 + search_radius*1e-3 + 8*ulp(reference_coordinate)`; maximum observed error
against that mode reference is 0.0001719594. This is fixture evidence, not a
general numerical guarantee or a bound on requested-order CPU differences.

The raw difference from the requested CPU order is also retained. Its maximum
is **0.97287786** on the 65-point curved/noisy fixture with quadratic order:
GPU chose 52 plane fallbacks and 13 polynomial fits. This adverse, nearly
singular case is not equivalent to CPU quadratic smoothing; the per-mode
bound does not erase that difference. For the 4,097-point well-supported
noisy, curved and coincident quadratic fixtures, all 4,097 points fit
polynomials and the raw CPU difference is at most 0.0000004768.

An independent symmetric paraboloid with a 0.02 center spike distinguishes
linear and quadratic execution: linear center height exceeds 0.002 while
quadratic center height is below 1e-6. A deterministic noisy plane's height
RMS decreases from 0.01141399 to 0.0021848176 (about 81%). These tests establish
actual polynomial fitting and smoothing beyond the mode-conditioned comparisons.

Other checks cover custom position names, exact schema/metadata preservation,
all scalar attribute types including NaN payloads/signed zero, exact unchanged
XYZ bits, full transfer receipts, invalid config/coordinates/extents,
self-inclusive overflow, poisoned indices, numerical underflow, translated
planes, empty inputs, and fresh output after invalid work. Four concurrent
callers reuse scratch while keeping earlier output alive; results remain
usable after input/tree/neighborhood drops and reject mismatched ownership.

Local validation passed: GPU all-features tests 207 passed / 23 ignored,
filtering all-features library tests 40 passed / 1 ignored, four explicitly run
adapter integration tests, and one explicitly run facade GPU test. Clippy
passed for GPU/filtering all features and the CPU-only MLS configuration.
GPU no-default-features, filtering MLS-GPU wasm32-unknown-unknown and workspace
Rustfmt checks passed. Removing the added GPU result types/methods leaves CPU
MLS source exactly equal to baseline commit
`053e84fac38831cf33f43a0847e1a34d61b6cbd9`.

## Completed-call measurements

Release builds on a native Vulkan discrete-device backend. Three warmups per
stage precede 20 paired rounds with cyclic CPU/full-GPU/resident stage order.
Two surveys run separately with no other local build, test or statistical
analysis during timing. All samples are retained, including slow observations.
Runtime/pipeline creation is warmed outside timers.

CPU timings use current public `MlsSmoothing::filter`, including its bounded
worker policy: four workers at 65,537 points and one at smaller selected sizes.
Full GPU timings use `filter_with_gpu`, including validation, CPU tree build,
tree/query upload, radius search, frames/fits/finalization, position/mode/status
download, completion wait and schema/attribute reconstruction. Result
verification/drop is outside timers; internal preparation/staging destruction
is included in the completed call.

Resident timings include owned result allocation, private scratch acquisition,
all three MLS passes, submission and `wait_idle`. They exclude CPU preparation,
upload, neighborhood generation, download and cloud reconstruction. These are
prepared fitting-phase measurements, not GPU timestamps or full-call speedups.

P is a noisy planar 128-column grid with spacing 0.05 and order 1; C adds
curved height and deterministic noise with order 2; D duplicates consecutive C
points and uses order 2. Search radius is 0.16 throughout, minimum support 6,
capacity 64 for P/C and 96 for D. Before timing, each GPU output is checked
against its mode reference and its raw requested-order CPU difference is
recorded. Repeated CPU and GPU calls match their respective references exactly.
Ratios on rows with different CPU/GPU fitting decisions do not imply
accuracy-equivalent output.

Times below are medians in milliseconds. Ratios are geometric means of 20
paired CPU/full-GPU timings. Percentile 95% bootstrap intervals use 10,000
resamples of paired log ratios, seeds 20261013 plus zero-based record index,
and sorted entries 249/9749. These selected fixtures do not establish a
universal deployment crossover; condition geometry and scheduling matter.

| Survey | Points | Fixture/order | Workers | Unchanged/plane/poly | Mode error | Requested CPU error | CPU ms | Full GPU ms | Resident ms | Paired full-call ratio (95% interval) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 257 | P/1 | 1 | 0/0/257 | 2.98e-08 | 2.98e-08 | 1.757 | 34.259 | 6.090 | 0.09x (0.06-0.14) |
| 1 | 257 | C/2 | 1 | 0/138/119 | 0.000172 | 0.00676 | 3.034 | 5.726 | 2.249 | 0.44x (0.35-0.55) |
| 1 | 257 | D/2 | 1 | 0/35/222 | 4.42e-07 | 4.42e-07 | 2.356 | 4.909 | 1.986 | 0.52x (0.46-0.60) |
| 1 | 4,097 | P/1 | 1 | 0/0/4097 | 1.19e-07 | 1.19e-07 | 77.003 | 15.063 | 4.116 | 5.31x (4.84-5.77) |
| 1 | 4,097 | C/2 | 1 | 0/0/4097 | 4.77e-07 | 4.77e-07 | 94.565 | 25.771 | 14.504 | 3.84x (3.51-4.18) |
| 1 | 4,097 | D/2 | 1 | 0/0/4097 | 2.38e-07 | 2.38e-07 | 80.714 | 15.875 | 11.662 | 5.13x (4.81-5.49) |
| 1 | 16,384 | P/1 | 1 | 0/0/16384 | 5.96e-08 | 5.96e-08 | 304.223 | 24.767 | 9.304 | 9.43x (7.68-11.22) |
| 1 | 16,384 | C/2 | 1 | 0/0/16384 | 4.77e-07 | 4.77e-07 | 279.677 | 33.858 | 25.175 | 7.78x (6.94-8.68) |
| 1 | 16,384 | D/2 | 1 | 0/0/16384 | 4.77e-07 | 4.77e-07 | 605.405 | 68.258 | 52.525 | 8.57x (7.91-9.20) |
| 1 | 65,537 | P/1 | 4 | 0/0/65537 | 4.77e-07 | 4.77e-07 | 209.168 | 58.353 | 38.299 | 3.45x (3.13-3.71) |
| 1 | 65,537 | C/2 | 4 | 0/1/65536 | 1.91e-06 | 1.91e-06 | 210.302 | 111.005 | 93.274 | 1.92x (1.83-2.02) |
| 1 | 65,537 | D/2 | 4 | 0/0/65537 | 9.54e-07 | 9.54e-07 | 371.191 | 226.562 | 204.646 | 1.61x (1.48-1.76) |
| 2 | 257 | P/1 | 1 | 0/0/257 | 2.98e-08 | 2.98e-08 | 0.756 | 1.435 | 0.259 | 0.56x (0.53-0.60) |
| 2 | 257 | C/2 | 1 | 0/138/119 | 0.000172 | 0.00676 | 0.907 | 1.231 | 0.288 | 0.72x (0.70-0.75) |
| 2 | 257 | D/2 | 1 | 0/35/222 | 4.42e-07 | 4.42e-07 | 1.066 | 1.626 | 0.352 | 0.67x (0.65-0.68) |
| 2 | 4,097 | P/1 | 1 | 0/0/4097 | 1.19e-07 | 1.19e-07 | 27.877 | 3.067 | 0.720 | 9.08x (8.48-9.81) |
| 2 | 4,097 | C/2 | 1 | 0/0/4097 | 4.77e-07 | 4.77e-07 | 38.621 | 3.780 | 1.559 | 10.05x (8.90-11.36) |
| 2 | 4,097 | D/2 | 1 | 0/0/4097 | 2.38e-07 | 2.38e-07 | 50.834 | 11.088 | 7.999 | 4.58x (4.21-5.01) |
| 2 | 16,384 | P/1 | 1 | 0/0/16384 | 5.96e-08 | 5.96e-08 | 301.913 | 44.647 | 10.117 | 7.79x (6.26-9.68) |
| 2 | 16,384 | C/2 | 1 | 0/0/16384 | 4.77e-07 | 4.77e-07 | 387.395 | 42.101 | 26.952 | 8.45x (7.24-9.38) |
| 2 | 16,384 | D/2 | 1 | 0/0/16384 | 4.77e-07 | 4.77e-07 | 294.013 | 56.855 | 50.388 | 5.01x (4.74-5.29) |
| 2 | 65,537 | P/1 | 4 | 0/0/65537 | 4.77e-07 | 4.77e-07 | 203.909 | 59.630 | 38.248 | 3.43x (3.26-3.59) |
| 2 | 65,537 | C/2 | 4 | 0/1/65536 | 1.91e-06 | 1.91e-06 | 224.250 | 112.656 | 93.240 | 2.00x (1.92-2.09) |
| 2 | 65,537 | D/2 | 4 | 0/0/65537 | 9.54e-07 | 9.54e-07 | 429.181 | 226.458 | 204.603 | 1.94x (1.73-2.21) |

All 257-point full GPU calls are slower in both surveys (paired ratios
0.09-0.72x). At 4,097 and 16,384 points, selected ratios are 3.84-10.05x
and 5.01-9.43x respectively. Against four CPU workers at 65,537 points,
ratios are 1.61-3.45x, with raw requested-order coordinate differences no
larger than 0.00000190735 and at most one GPU plane fallback. These measured
gains support an explicit GPU option on these fixtures, without changing
placement defaults. Substantial run-to-run timing variation is retained in
the table, especially for small calls and the first survey; no samples or
conditions are excluded. The 257-point C/2 row retains its 0.006760955 raw
CPU difference and 138 plane fallbacks, independently of timing.

Reproduction from `C:/Users/rsasa/Workspace/SpatialRust`:

```powershell
cargo test -p spatialrust-gpu --all-features
cargo test -p spatialrust-filtering --all-features --test gpu_mls -- --include-ignored --nocapture
cargo test -p spatialrust --no-default-features --features filter-mls-gpu --test gpu_mls_smoke -- --include-ignored
cargo test -p spatialrust-filtering --all-features --release --test gpu_mls_benchmark completed_mls_calls -- --include-ignored --nocapture --exact --test-threads=1
```

The benchmark's exact filter is required: its imported reference module also
contains adapter integration tests. Benchmark runs are isolated from local
builds, tests and statistics, and intentionally ignored in normal CI. Explicit
adapter tests fail if no native adapter exists instead of silently skipping.

Local validation logs, raw timing surveys and measurement JSON are under
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-mls-*.log` and
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-mls-measurements.json`.
These ignored local artifacts do not name the local adapter.
