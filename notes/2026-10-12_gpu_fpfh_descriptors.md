# Resident GPU FPFH descriptors

Date: 2026-10-12. Named slice: GPU radius neighborhoods, SPFH and weighted
FPFH descriptors, following the resident neighborhood API in PR #166.

## Execution and transfer contract

The optional `register-fpfh-gpu` feature exposes
`fpfh_descriptors_gpu(runtime, cloud, radius, neighbor_capacity)` in registration
and the facade. The caller chooses GPU execution explicitly. This wrapper uses
the existing CPU input validation, semantic position/normal access and KD-tree
construction, then uploads, computes and downloads. Ordinary FPFH, descriptor
matching, RANSAC/global registration, ISS and MLS retain their current CPU paths.

`GpuFpfhInput::upload` uploads original-order XYZ/unit normals together with a
GPU KD-tree and reusable original-order query batch. Positions and normals must
be finite, normals must satisfy the existing unit-length tolerance, and global
coordinate extents must have a finite f32 squared length. Counts, host sizes,
device buffer sizes, dispatch limits and bindings are checked before allocation.
The descriptor shader requires five storage bindings, one uniform binding and
64 invocations per workgroup. No new dependency, core API or unsafe code is added.

`radius_neighborhoods_resident` computes radius rows once without readback or a
completion wait. Its typed result is tied to that exact uploaded input; another
input's neighborhoods are rejected even on the same runtime. Radius capacity
includes self and must accommodate ALL matches. An incomplete row is an error,
not an approximate descriptor computed from a truncated prefix. Radius must be
positive with a finite, nonzero f32 square.

`descriptors_resident` consumes those rows in two ordered compute passes in one
submission. Each original point builds three normalized eleven-bin SPFH
histograms. The next pass adds ordered inverse-distance-weighted neighboring
SPFH values. Self, coincident points and degenerate Darboux frames follow the
existing CPU exclusion rules. Signed-zero `atan2` cases are handled explicitly.
SPFH storage is private scratch, completely written before reuse; returned
descriptor/status buffers are owned independently and never pooled. Calls can
reuse uploaded inputs/neighborhoods, retain earlier outputs and run concurrently.

GPU status contains four u32 words: bad row count, last bad row plus one,
maximum required radius capacity, and malformed row count. SPFH validates row
counts and target indices before accessing samples. If any row is incomplete or
malformed, the second pass zeros ALL descriptor rows. Custom GPU consumers must
inspect status before accepting results and treat exposed storage as immutable
application data. Explicit download rejects bad status before decoding rows.

For N points and M CPU-tree nodes, input upload is `64*N + 32*M` H2D payload
bytes: 32*N samples, 16*N queries and 16*N tree positions plus 32*M nodes.
Resident neighborhood/descriptor execution transfers zero payload bytes.
Explicit download copies `132*N + 16` D2H bytes in one submission, including
33-f32 descriptors and status, and waits for completion. Empty input has zero
payload transfers. Uniform controls are excluded from payload receipts. There
is no neighborhood/SPFH readback or automatic CPU fallback.

GPU angular arithmetic, `atan2` approximation and bin selection may differ from
CPU near histogram boundaries. This API does not promise bitwise parity or a
universal small numerical bound across inputs/backends. The observed error below
applies to the selected fixtures. WASM validation is compilation only; browser
execution of the synchronous download path was not tested.

## Correctness and regression evidence

- Three new GPU primitive tests cover limits, invalid inputs/empty receipts,
  overflow and an externally poisoned target index. Exact status and complete
  zeroed output are checked, followed by successful scratch reuse.
- Four explicitly run adapter integration tests compare 1,205,424 descriptor
  values across 0/1/2/17/63/64/65/257/4097 points, two radii, planar/curved/
  coincident-pair fixtures and alternating signed axis normals. Absolute
  tolerance is 0.002; maximum observed CPU difference is 0.000045776367.
- Independent planar-pair descriptors have exactly 200 in each central bin;
  isolated/coincident/parallel-normal degeneracies are zero. Tests also verify
  CPU validation errors, capacity errors, exact transfer receipts, concurrent
  calls, retained outputs and runtime/input ownership guards.
- All-feature GPU regression tests: 199 passed, 23 ignored. Registration library
  regression tests: 35 passed, one ignored. The facade GPU integration test
  passed. After the WASM readback portability fix, the four adapter integration
  tests and facade test passed again.
- All-feature GPU/registration Clippy with warnings denied, CPU-only registration
  Clippy, minimal GPU compilation, WASM GPU-FPFH compilation and workspace
  formatting passed. Source comparison confirms existing CPU FPFH math and
  registration code are unchanged. CI adds isolated registration/facade feature
  profiles; hosted CI compiles adapter tests rather than proving GPU execution.

## Completed-call measurements

Release builds on a native Vulkan discrete-device backend. Adapter model names
are deliberately omitted. Three warmups for each stage precede 20 paired rounds;
CPU/full-GPU/resident stage order rotates each round. Two surveys ran separately
without another local build, test or statistical analysis during measurement.
All samples are retained, including adverse and variable results.

The CPU baseline is the current public `fpfh_descriptors`, including its existing
bounded CPU parallelism: four workers at 65,537 points, one at the smaller sizes.
Full GPU calls include CPU validation/lifting/tree construction, all input
uploads, GPU radius search, SPFH, weighted FPFH and descriptor/status download
with completion wait. Resident timings include descriptor allocation, both
descriptor passes, submission and `wait_idle`; they exclude CPU preparation,
input upload, neighborhood generation and final download. Correctness checks
and dropping returned descriptor values happen outside timers. Runtime/pipeline
creation is warmed outside timing. Resident columns are not complete-call
speedups, GPU timestamp measurements or global-registration timings.

Fixtures use radius 0.35 and capacity 64: P is a planar grid with common +Z
normals; C is a curved grid with analytic normals; A uses the curved positions
with alternating signed axis normals. Each fixture uses 32 columns and spacing
0.125. Reported times are medians in milliseconds. Speedup is the geometric
mean of the 20 paired CPU/full-GPU ratios, with a percentile 95% bootstrap
interval from 10,000 resamples of paired log ratios (deterministic seeds
20261012 plus the zero-based record index). It is not a ratio of medians and
does not establish deployment performance or a crossover threshold.

| Survey | Points | Fixture | CPU workers | CPU ms | Full GPU ms | Resident ms | Paired full-call speedup (95% interval) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | --- |
| 1 | 257 | P | 1 | 0.565 | 1.570 | 0.239 | 0.39x (0.35-0.42) |
| 1 | 257 | C | 1 | 0.633 | 1.265 | 0.256 | 0.50x (0.49-0.52) |
| 1 | 257 | A | 1 | 0.704 | 1.284 | 0.258 | 0.56x (0.53-0.58) |
| 1 | 4,097 | P | 1 | 13.434 | 2.906 | 0.430 | 4.74x (4.55-4.94) |
| 1 | 4,097 | C | 1 | 13.649 | 3.139 | 0.435 | 4.33x (4.08-4.60) |
| 1 | 4,097 | A | 1 | 13.712 | 3.057 | 0.423 | 4.81x (4.49-5.16) |
| 1 | 16,384 | P | 1 | 52.217 | 9.765 | 4.880 | 6.85x (5.85-8.07) |
| 1 | 16,384 | C | 1 | 47.275 | 25.295 | 13.840 | 1.88x (1.76-2.02) |
| 1 | 16,384 | A | 1 | 50.727 | 25.147 | 13.767 | 2.01x (1.89-2.13) |
| 1 | 65,537 | P | 4 | 87.486 | 29.268 | 4.425 | 2.43x (2.04-2.90) |
| 1 | 65,537 | C | 4 | 62.074 | 23.937 | 3.032 | 2.44x (2.11-2.76) |
| 1 | 65,537 | A | 4 | 62.031 | 45.835 | 22.891 | 1.73x (1.49-2.02) |
| 2 | 257 | P | 1 | 0.586 | 1.113 | 0.224 | 0.52x (0.51-0.53) |
| 2 | 257 | C | 1 | 0.636 | 1.266 | 0.275 | 0.50x (0.49-0.52) |
| 2 | 257 | A | 1 | 0.647 | 1.102 | 0.224 | 0.57x (0.56-0.59) |
| 2 | 4,097 | P | 1 | 11.668 | 2.575 | 0.380 | 4.70x (4.50-4.92) |
| 2 | 4,097 | C | 1 | 11.257 | 2.466 | 0.368 | 4.62x (4.47-4.78) |
| 2 | 4,097 | A | 1 | 11.844 | 2.601 | 0.373 | 4.69x (4.44-4.96) |
| 2 | 16,384 | P | 1 | 48.882 | 4.940 | 1.154 | 10.00x (9.59-10.43) |
| 2 | 16,384 | C | 1 | 50.195 | 24.807 | 13.671 | 2.56x (2.29-2.87) |
| 2 | 16,384 | A | 1 | 46.026 | 25.883 | 13.921 | 1.91x (1.79-2.04) |
| 2 | 65,537 | P | 4 | 88.117 | 23.325 | 4.438 | 3.31x (2.85-3.77) |
| 2 | 65,537 | C | 4 | 71.337 | 50.672 | 28.314 | 1.61x (1.39-1.91) |
| 2 | 65,537 | A | 4 | 64.949 | 21.837 | 3.024 | 2.86x (2.59-3.08) |

Timings vary substantially between surveys and stages, and resident results
are not monotonic in point count. The data does not establish the cause of
that variation. Small fixtures can favor CPU; warmed resident results must not
be used to imply transfer-inclusive gains on arbitrary inputs.

## Reproduction

From `C:/Users/rsasa/Workspace/SpatialRust`, run adapter correctness explicitly:

```powershell
cargo test -p spatialrust-registration --no-default-features --features register-fpfh-gpu --test gpu_fpfh -- --include-ignored --nocapture
cargo test -p spatialrust --no-default-features --features register-fpfh-gpu --test gpu_fpfh_smoke -- --include-ignored
```

Run this exact benchmark filter alone twice. The filter excludes the included
reference module's four adapter tests from timing runs:

```powershell
cargo test -p spatialrust-registration --no-default-features --features register-fpfh-gpu --test gpu_fpfh_benchmark --release completed_fpfh_calls -- --include-ignored --nocapture
```

Local untracked receipts:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-final-integration-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-facade-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-native-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-registration-tests.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-fpfh-measurements.json`
