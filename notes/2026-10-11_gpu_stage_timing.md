# Shared GPU compute-pass timing — 2026-10-11

Image preprocessing and point-cloud frames now share optional, bounded GPU
timestamp instrumentation. Default runtime constructors and existing operation
signatures remain unchanged; they request no timestamp features or queries.
The implementation lives in
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/profiling.rs`.

## Contract

- `WgpuRuntime::new_headless_with_timestamps` and its async counterpart request
  only `TIMESTAMP_QUERY` when the selected adapter supports it. An unsupported
  adapter still initializes normally; inspect `timestamp_queries_enabled()`.
- `GpuStageProfiler::new(runtime, max_stages)` accepts 1..=2048 stages and
  allocates two queries per stage. Profiled operation entry points check runtime
  identity, recording validity and remaining capacity before any work.
- A dedicated retained runtime identity protects against wgpu handle equality
  collisions across separate instances and remains valid when the runtime moves.
- `run_gpu_vision_chain_profiled` measures resize, gray, blur, Sobel, morphology
  and CHW packing. Intermediate images remain resident and are recycled through
  the existing texture pool.
- `run_aoso_voxel_normal_frame_profiled` measures both voxel/radius partitions:
  key generation, entry construction, all 24 radix digits (histogram, prefix,
  scatter), boundary marking, prefix initialization/scans, segment writing and
  radius normals. Repeated stages retain individual samples in submission order.
- Native `profiler.collect(runtime)` is an explicit synchronization and readback
  boundary. It consumes the recording, resolves only written queries, converts
  tick differences using the queue's timestamp period, and returns a separate
  `GpuTimingReport`. Empty recordings have zero transfer bytes.
- Query readback is 16 bytes per pass. Image and point payload receipts do not
  absorb those bytes: a resident image chain still reports zero payload readback,
  and a nonempty point frame still reports its two four-byte cell counts.
- Capacity rejection preserves earlier successful samples. An operation failure
  invalidates the recording so abandoned query slots cannot become timings.
  A new profiler is required after such a failure.
- Pass durations exclude uploads, copies, CPU preparation, scalar readback and
  gaps between passes. They are measurements under instrumentation, not an
  unprofiled performance baseline. Precision depends on the backend; zero is
  valid. Browser construction/recording remains async-compatible, but this slice
  provides timestamp collection on native hosts only.

## Validation

On the local GPU / Vulkan / driver 596.36:

```powershell
cargo test -p spatialrust-gpu --features gpu-aoso-staging,gpu-image -- --test-threads=1
cargo test -p spatialrust-filtering -p spatialrust-segmentation --features spatialrust-filtering/filter-voxel-gpu,spatialrust-segmentation/segment-euclidean-gpu -- --test-threads=1
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-wgpu --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-image --all-targets -- -D warnings
cargo check -p spatialrust-gpu
cargo check -p spatialrust-gpu --all-features --target wasm32-unknown-unknown --lib
cargo fmt --all --check
git diff --check
```

55 GPU-crate unit tests, seven profiling integration tests and 28 downstream
filtering/segmentation tests passed (90 total). These include the existing seven
image regressions. New checks cover image output/receipt parity, joint image and
point recordings, source-position/segment/planar-normal parity, empty/singleton
and non-power-of-two inputs, exact query byte counts, bounded capacity, partial
failure, separate-instance rejection, disabled default features, unsupported
adapter feature selection, and tick conversion including zero/wraparound.

Hardware-dependent profiling integration tests skip when timestamp queries are
unavailable. Unsupported selection and tick conversion also have pure tests.
The local GPU example confirms the hardware timestamp path ran locally.

CI additionally exposed a wgpu-only WebAssembly configuration where neither
native collection nor the optional image/point profiled entry points existed.
The unused runtime identity and validation helpers are now gated to the
configurations that use them. Reproduced and verified with `RUSTFLAGS=-D warnings`
for `spatialrust-web --target wasm32-unknown-unknown --features wasm,webgpu` and
the image/AoSoA GPU WASM configuration. The browser CI now builds both surfaces.

## Reproduction and observations

The next CI run passed all 685 Python/ONNX tests, then encountered
`FileExistsError` while creating `target/icp-convergence-study`: that output
directory had been restored from the Rust target cache. Controlled ICP and
trimmed-ICP studies now write under runner temporary storage, in a directory
named by run ID and attempt, and artifact upload reads those same fresh paths.
This preserves the studies' refusal to overwrite existing output and prevents
an old cached report from being mistaken for a current execution.

The runnable example is
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/examples/gpu_stage_timing.rs`:

```powershell
cargo run -p spatialrust-gpu --release --features gpu-aoso-staging,gpu-image --example gpu_stage_timing
```

After three warm-up executions, it averages ten instrumented executions. Wall
time includes `wait_idle()` but excludes profiler allocation and collection.
Image source upload is excluded; point upload and scalar cell-count readbacks
are included. This observation was collected after GPU correctness tests had
finished; an earlier overlapping run was discarded.

| Workload | Completed wall mean | GPU pass sum mean | Query readback/run |
| --- | ---: | ---: | ---: |
| RGB 640×480 → resize 320×240 → gray → blur → Sobel → dilation → CHW | 1.017 ms | 0.055 ms | 96 B |
| 65,536 XYZ → voxel partition → radius grid → normals | 12.485 ms | 3.711 ms | 2,992 B |

Point stages, aggregating both partitions and repeated radix/scan passes:

| Stage | Mean ms/run |
| --- | ---: |
| Radius normals | 2.188 |
| Radix histogram | 0.464 |
| Radix prefix | 0.282 |
| Radix scatter | 0.534 |
| Compaction mark/init/scan/write | 0.198 |
| Voxel keys | 0.020 |
| Sort entry construction | 0.023 |

Numbers are a single-device diagnostic sample, rounded independently, not a
claim of improvement or an Auto-policy threshold recommendation. In this point
fixture normals and radix dominate measured compute time; compaction scans are
0.147 ms. Hierarchical compaction remains a scaling follow-up requiring larger
inputs and completed-work comparisons. For this small image output the wall/pass
gap warrants inspecting dispatch/submission and resource setup before assuming
shader arithmetic is the main limit. This report does not classify that gap as
CPU time: transfers, driver work and synchronization are also excluded from pass
samples. Bounded scratch retention remains the subsequent memory follow-up.

The subsequent same-day parallel-execution slice implements hierarchical
compaction and documents completed segment-build comparisons in
`C:/Users/rsasa/Workspace/SpatialRust/notes/2026-10-11_parallel_image_point_execution.md`.
Its profiled 65,536-point frame uses 159 passes and 2,544 query bytes; the
187-pass/2,992-byte observation above describes the preceding implementation.
