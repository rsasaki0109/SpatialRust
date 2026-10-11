# Parallel image and point-cloud execution — 2026-10-11

This slice combines GPU point-cloud compaction and CPU image-analysis
parallelism. Public operation signatures, automatic GPU placement, core
dependencies and explicit transfer boundaries remain unchanged.

## GPU compaction

Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/kernels/voxel_scan.rs`
and
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/shaders/inclusive_scan.wgsl`.

Boundary flags now pass through a work-efficient Blelloch scan within each
256-lane workgroup. Padded lanes contribute zero and participate in every
barrier. Block totals are scanned recursively, then globally inclusive offsets
are propagated down the hierarchy. Work is linear in the input size plus its
shrinking hierarchy, replacing full-buffer Hillis–Steele scans at every stride.
Boundary marking, all scan levels, offset propagation and segment writing are
encoded in one command buffer/submission. No hierarchy data is read by the CPU.

The old public pipeline-cache fields remain available. Their unused binding
retains a four-byte placeholder, replacing the full-sized ping-pong scan
buffer. For 1,048,576 elements, hierarchy buffers total 32,900 bytes instead of
the old 4,194,304-byte scan scratch (plus the four-byte placeholder). Input flags
and the final inclusive prefix still each occupy four bytes per element.
Buffer handles cloned during encoding retain ownership without device copies.
Temporary hierarchy buffers are not yet retained for reuse between calls.

At 65,536 elements the scan uses three passes instead of initialization plus
16 full-array scans. At 65,537 elements it uses five passes. A complete profiled
65,536-point voxel/radius/normal frame now records 159 passes and explicitly
reads back 2,544 query bytes, versus 187 passes and 2,992 bytes before this slice.
Frame payload receipts still report eight scalar cell-count bytes, and a
standalone nonempty segment build still reports four. Empty builds report zero.
Profiler capacity uses the hierarchy's actual number of levels before work.

GPU tests compare all inclusive values against a serial prefix for sizes
0, 1, 2, 255, 256, 257, 511, 65,535, 65,536, 65,537 and 1,000,003. Segment tests
cover distinct keys, runs crossing blocks and hierarchy levels, and one-cell
inputs, with reversed point indices. Raw device order is checked independently
of the existing host conversion, which sorts indices within each cell.

## CPU image analysis

Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/src/analysis.rs`.

`imgproc-analysis` now enables the existing optional Rayon dependency.
Thresholding dispatches at 100,000 components; grayscale u8 histograms dispatch
at 1,000,000 pixels. Workers are bounded by logical rows and the active Rayon
pool. Each task owns a contiguous row band. Images below those limits, empty
images, single-row inputs and single-worker pools use the scalar path.

Thresholding writes disjoint output bands using the same component transform
as the scalar path. The parallel path selects its mode and converts constant
outputs once, then iterates logical row slices excluding source padding.
The scalar fallback retains the original per-pixel implementation. Small and
empty inputs avoid initializing the global thread pool. Histogram tasks accumulate
256 exact `u64` counts in private arrays (2 KiB per band) and merge integer
counts after joining, without atomics or shared counters. Otsu and equalization
also use this histogram routine. Generic histograms and the remaining analysis
algorithms retain their existing implementations.

Tests compare one, two and six workers for all five threshold modes on a
multichannel u16 ROI, and one/six workers bit-for-bit on padded f32 RGBA input
containing NaN, infinities and signed zero. Histogram tests exclude padding and
check exact counts and totals across one/two/six workers. Empty, singleton,
small and large single-row inputs exercise scalar fallbacks. Metadata and
dimensions remain unchanged.

## Measurement method

Reference revision: `7c79854ba80b91fcbe8f5cfa2b166e3b5885570c`, before this slice.
Identical benchmark sources and lockfiles are copied into a detached temporary
worktree for the reference. GPU packages are explicitly cleaned/rebuilt before
each revision: the shared Cargo target initially reused the reference artifact,
so those initial comparisons were discarded. The accepted new executable was
checked for the hierarchy shader. Runs are sequential after GPU correctness
tests finish, with no concurrent agent builds or correctness workloads during
sampling. Other desktop applications and user Python processes remain active;
CPU samples showed changing background load. These are observations on a shared
desktop, not isolated hardware performance guarantees.
The GPU is NVIDIA GeForce GTX 1660 Ti, Vulkan, driver 596.36; the CPU is Intel
Core i7-9750H (six physical/twelve logical cores). CPU parallel runs set
`RAYON_NUM_THREADS=6`; a new one-worker run compares the scalar fallback with
threading. All benchmark artifacts remain under the ignored target tree.

The GPU benchmark lives at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/benches/voxel_segments.rs`.
It uses pre-uploaded synthetic signed XYZ voxel keys and includes entry
construction, all radix passes, compaction, scalar cell-count readback, resource
allocation and `wait_idle` on every iteration. It excludes input upload, normal
estimation and full segment-array readback. It measures the complete segment
build rather than just a shader scan.

The CPU benchmark lives at
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-vision/benches/analysis.rs`.
It measures packed u8 grayscale thresholding, including output allocation, and
256-bin histograms at 1920×1080 and 3840×2160. The deterministic input cycles
through all 256 intensities. These measurements are workload/device-specific;
they do not establish a new automatic GPU crossover threshold.

## GPU observations

Criterion time estimates in milliseconds, with 95% confidence intervals:

| Resident keys | Reference | Hierarchical scan | Estimate ratio |
| --- | ---: | ---: | ---: |
| 65,536 | 2.852 (2.805–2.893) | 2.261 (2.219–2.319) | 1.26× |
| 1,048,576 | 13.716 (13.573–13.935) | 12.829 (12.758–12.900) | 1.07× |

These are completed, unprofiled segment builds. The time estimate decreases
20.7% and 6.5%, respectively; radix sorting and resource setup remain included.
The accepted GPU comparison uses 20 samples, three seconds of warm-up and five
seconds of requested measurement per case. CPU analysis groups use ten samples
with the same warm-up/measurement duration.

```powershell
# In each worktree, with the same CARGO_TARGET_DIR:
cargo clean -p spatialrust-gpu --release
cargo bench -p spatialrust-gpu --features gpu-wgpu --bench voxel_segments -- --sample-size 20 --warm-up-time 3 --measurement-time 5 --save-baseline before
# After cleaning/rebuilding the new revision, replace --save-baseline before
# with --baseline before. Inspect the executable for the new hierarchy shader.

$env:RAYON_NUM_THREADS = '6'
cargo bench -p spatialrust-vision --features imgproc-analysis --bench analysis -- '^(threshold|histogram_u8)/(1080p|4k)$' --warm-up-time 3 --measurement-time 5 --baseline before
$env:RAYON_NUM_THREADS = '1'
cargo bench -p spatialrust-vision --features imgproc-analysis --bench analysis -- '^(threshold|histogram_u8)/(1080p|4k)$' --warm-up-time 3 --measurement-time 5 --save-baseline sequential
```

## CPU observations

Criterion time estimates in milliseconds. Parentheses show 95% confidence
intervals for the final new one/six-worker measurements:

| Operation | Reference | New: one worker | New: six workers | One/six ratio |
| --- | ---: | ---: | ---: | ---: |
| Threshold 1080p | 2.282 | 2.375 (2.324–2.442) | 1.472 (1.465–1.477) | 1.61× |
| Threshold 4K | 23.959 | 9.614 (9.289–9.942) | 5.123 (5.061–5.185) | 1.88× |
| Histogram 1080p | 2.702 | 1.777 (1.463–2.123) | 0.484 (0.481–0.487) | 3.67× |
| Histogram 4K | 6.889 | 4.455 (4.350–4.692) | 1.698 (1.674–1.724) | 2.62× |

Reference and new executables were built before sampling, then executed
sequentially. The original 4K threshold was 9.904 ms in the initial reference
round and 23.959 ms in the later round, demonstrating substantial background
load changes. Consequently, the reference/new differences are not reliable
estimates of threading alone. The final one/six-worker ratios are the more
useful observation, still subject to shared-desktop load; the one-worker 1080p
histogram also has a wide interval. No Python overhead or GPU transfers enter
these native image measurements.

## Validation

57 GPU unit tests, seven profiling integration tests, 28 downstream filtering/
segmentation tests, and 146 vision unit plus 13 property tests passed (251
tests across the final relevant suites). The analysis-only configuration also
passed its 16 tests before the final scalar-fallback adjustment; the full vision
suite revalidated all analysis tests afterward. GPU tests ran serially on the
physical adapter.

```powershell
cargo test -p spatialrust-gpu --features gpu-aoso-staging,gpu-image -- --test-threads=1
cargo test -p spatialrust-filtering -p spatialrust-segmentation --features spatialrust-filtering/filter-voxel-gpu,spatialrust-segmentation/segment-euclidean-gpu -- --test-threads=1
cargo test -p spatialrust-vision --features full
cargo clippy -p spatialrust-gpu --all-features --all-targets -- -D warnings
cargo clippy -p spatialrust-gpu --features gpu-wgpu --all-targets -- -D warnings
cargo clippy -p spatialrust-vision --features full --all-targets -- -D warnings
cargo clippy -p spatialrust-vision --features imgproc-analysis --all-targets -- -D warnings
cargo check -p spatialrust-gpu -p spatialrust-vision
$env:RUSTFLAGS = '-D warnings'
cargo check -p spatialrust-web --target wasm32-unknown-unknown --features wasm,webgpu
cargo check -p spatialrust-gpu --target wasm32-unknown-unknown --features gpu-image,gpu-aoso-staging
cargo check -p spatialrust-vision --target wasm32-unknown-unknown --features imgproc-analysis
cargo fmt --all --check
git diff --check
```

Browser CI additionally builds the new analysis-only WASM surface. The detached
benchmark worktree was removed after verifying that its changes consisted only
of copied benchmark sources/Cargo configuration and the copied ignored lockfile.
CPU scalar fallback was retained after initial threshold variants regressed;
the rejected intermediate measurements are not used for the final comparison.
