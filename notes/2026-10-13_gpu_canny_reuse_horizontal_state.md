# GPU Canny horizontal/state storage reuse

Date: 2026-10-13. Named slice: reuse the dead horizontal-row lifetime for later
classification/root flags in the existing resident Canny API.

## Execution and requested memory

The 5/7 horizontal pass packs the same unrounded signed derivative/smoothing
halves into the atomic state binding. The vertical pass reconstructs those
integer values and writes complete eight-byte gradients. Only after every
vertical invocation has completed does classification overwrite every state word
with its low classification bits. The existing weak-union/root-flag/output passes
then use those words as before. Ordered compute passes separate both lifetimes:
no row is overwritten while any vertical invocation can still read it. The
production route adds no copy, clear, pass, submission or host synchronization.
The row phase uses integer atomic store/load on the same word type; classification
and flags keep their existing atomic operations and full-width parent indices.

Remove the separate horizontal lease and binding 3, including the eight-byte
unused aperture-3 dummy. Three storage buffers remain: gradients (8*N), shared
rows/states/flags (4*N), and parents (4*N). Existing index/dimension/workgroup/
control/texture/largest-buffer/profiler checks remain; required storage-buffer
count falls from four to three. A two-buffer limit still rejects before work.
The largest binding stays 8*N and controls stay 96 bytes. GPU stage names and
five/six passes remain. Horizontal stage destinations still occupy 4*N for 5/7;
classification subsequently writes the same allocation. Receipts stay 32*N/36*N
for stage destination footprints, with unchanged image upload/padded readback.
These counts do not measure repeated in-place atomic byte traffic.

| Aperture | PR #185 private bytes | Reused private bytes | Previous 4K bytes | Reused 4K bytes | Saved 4K bytes |
| --- | --- | --- | ---: | ---: | ---: |
| 3 | 16*N+8 | 16*N | 132,710,408 | 132,710,400 | 8 |
| 5/7 | 20*N | 16*N | 165,888,000 | 132,710,400 | 33,177,600 |

N=width*height; 4K means 3840x2160. These requested private buffer bytes exclude
input/output textures, controls, driver overhead and retained pool resources.
Against the initial PR #180 unpacked route, cumulative private savings at 4K are
99,532,808 bytes for aperture 3 and 165,888,000 for 5/7. No automatic CPU/GPU
placement, implicit blur, fallback or point/image readback is added.

## Correctness and execution references

The complete PR #185 shader is frozen from
`2ddb905882cc5bb22e85dad3a36b7fb4a628a7b9`. Native references compile its actual
separate horizontal buffer plus shared state/flags (four storage bindings). Older
PR #180/#181/#183 references retain both separate row and separate root-flag
buffers. Baseline controls, layout, allocation/submission and receipt footprints
match those respective routes. The shared-state historical benchmark explicitly
compares frozen PR #183 and #185, preserving its meaning independently of current
production execution.

- All 864 aperture/norm/threshold/profile/strip edge conditions produce 367,056
  exact CPU bytes and match four frozen GPU generations.
- 288 complete snapshots span six shapes, four profiles, three apertures, both
  norms and single-group versus ordinary dispatch. A test-only copy captures
  unrounded rows after vertical gradients and before classification overwrites
  them, while scratch remains exclusively leased. Rows match frozen GPU and
  independent integer convolution. All 4,018,560 final state words, gradients,
  decoded classifications, minimum roots, flags and edges match frozen PR #185;
  edge images also match CPU. State scratch starts poisoned with all-one words.
  Degenerate strips, 513x129 and 257x65 inputs exercise boundary and grid-stride
  reuse without leaving stale row bits in final flags.
- Actual native leases for 1x1, 17x19 and 3840x2160 across every aperture verify
  exactly three new allocations, distinct state/parent buffers and total buffer
  sizes equal to 16*N, including removal of the aperture-3 dummy.
- Existing independent f64 gradients/NMS/BFS, long weak snakes/diagonals, contended
  markers, 433,320 shared-state words, signed rounding/packing extrema, ownership,
  ROI/metadata, profiler, invalid/pre-allocation/capability and dirty-scratch
  reuse/chain checks remain.

Native Canny family: 15 passed, zero failed, five measurements explicitly ignored.

GPU image regression: 94 passed, zero failed, 18 measurements explicitly ignored.
All-feature/all-target GPU Clippy, feature-free GPU, GPU-image WASM, minimal
GPU-image facade and formatting checks pass. WASM was compile checked; browser
execution was not measured.

## Completed-call measurement

Two isolated release surveys completed in 22.48 and 22.28 seconds. Survey 2's
first completed attempt (24.69 seconds) overlapped foreign benchmark processes
and was discarded in full; its second attempt was accepted. The historical
PR #183/#185 completed-call test also passed; its first overlapping run (24.44
seconds) was excluded and its repeat completed alone in 23.35 seconds. Historical
timings do not enter the new table. No overlapping local build/test/benchmark
was detected during accepted runs. Statistical analysis followed all surveys.
Backend is Vulkan, device class DiscreteGpu, global Rayon pool has 12 workers.
No local adapter model is recorded.

Separate runtimes retain independent default bounded pools. Each geometry
clears scratch/texture retention before setup; each route then uses its own
normal 64 MiB scratch retention policy. Pipeline compilation, CPU references,
fixture creation, completed warmups and post-timer comparisons are excluded.
Three warmups precede twenty paired rounds rotating old full, reused full, old
resident and reused resident stages. Ten cases cover 256x192, 1920x1080 and
3840x2160, 3/5/7 apertures, both norms, deterministic noise/bar/low-contrast
images and thresholds 100/200.

Full calls include image upload, controls, scratch/output acquisition, all
passes, completed readback/unpacking and recycling owned images. Resident calls
start with an uploaded source and time the complete public call plus wait_idle;
readback/checks/recycling occur outside timing. Every edge byte and all explicit
transfer/destination receipt fields agree with CPU and the frozen route. No
intermediate image/label collection occurs in the production call.

## Paired results

Times are medians in milliseconds. Ratios are geometric means of twenty paired
old/reused times; values above one favor reuse. Intervals use 10,000 paired
log-ratio bootstrap draws, sorted entries 249/9749, with seed
20261013 + 2*record_index + comparison_index. These unadjusted intervals describe
within-survey variation, not repeat-run or hardware-wide bounds. All adverse
and tail samples in accepted runs are retained.


| Survey | Image/aperture/norm | Profile | Old full ms | Reused full ms | Old resident ms | Reused resident ms | Old/reused full (95%) | Old/reused resident (95%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 256x192/A5/L1 | noise | 1.137 | 1.113 | 0.385 | 0.353 | 1.026x (0.977-1.093) | 1.064x (0.950-1.193) |
| 1 | 256x192/A7/L2 | bars | 1.167 | 1.125 | 0.381 | 0.355 | 1.036x (0.990-1.100) | 1.073x (0.975-1.182) |
| 1 | 1920x1080/A3/L1 | bars | 9.218 | 9.340 | 1.112 | 1.097 | 0.965x (0.909-1.007) | 1.052x (0.988-1.128) |
| 1 | 1920x1080/A5/L1 | bars | 9.624 | 9.375 | 1.253 | 1.223 | 1.041x (1.010-1.085) | 1.027x (0.992-1.067) |
| 1 | 1920x1080/A5/L2 | noise | 9.717 | 9.647 | 1.526 | 1.567 | 1.005x (0.981-1.031) | 0.926x (0.800-1.011) |
| 1 | 1920x1080/A7/L2 | noise | 9.878 | 9.884 | 1.694 | 1.727 | 0.981x (0.947-1.007) | 1.027x (0.965-1.114) |
| 1 | 3840x2160/A3/L1 | bars | 37.594 | 35.010 | 10.012 | 8.593 | 1.042x (0.966-1.126) | 1.136x (1.085-1.184) |
| 1 | 3840x2160/A5/L2 | bars | 35.514 | 35.939 | 9.897 | 8.789 | 0.970x (0.875-1.072) | 1.149x (1.107-1.191) |
| 1 | 3840x2160/A7/L2 | noise | 36.586 | 36.960 | 9.588 | 7.847 | 0.990x (0.968-1.012) | 1.210x (1.167-1.257) |
| 1 | 3840x2160/A7/L2 | low contrast | 36.070 | 36.789 | 11.884 | 10.488 | 0.991x (0.925-1.067) | 1.082x (1.038-1.132) |
| 2 | 256x192/A5/L1 | noise | 1.132 | 1.121 | 0.373 | 0.354 | 0.995x (0.970-1.021) | 1.063x (0.963-1.171) |
| 2 | 256x192/A7/L2 | bars | 1.186 | 1.172 | 0.384 | 0.366 | 0.990x (0.970-1.010) | 1.049x (0.956-1.160) |
| 2 | 1920x1080/A3/L1 | bars | 8.873 | 9.030 | 1.081 | 1.128 | 0.998x (0.974-1.029) | 1.018x (0.970-1.072) |
| 2 | 1920x1080/A5/L1 | bars | 9.214 | 9.248 | 1.339 | 1.292 | 1.006x (0.993-1.020) | 1.031x (0.979-1.084) |
| 2 | 1920x1080/A5/L2 | noise | 9.914 | 9.569 | 1.446 | 1.344 | 0.978x (0.924-1.025) | 1.062x (0.971-1.206) |
| 2 | 1920x1080/A7/L2 | noise | 9.860 | 9.992 | 1.655 | 1.581 | 0.968x (0.925-0.998) | 1.020x (0.972-1.070) |
| 2 | 3840x2160/A3/L1 | bars | 33.773 | 34.529 | 9.979 | 8.596 | 0.996x (0.882-1.127) | 1.182x (1.124-1.252) |
| 2 | 3840x2160/A5/L2 | bars | 34.382 | 35.607 | 8.296 | 7.777 | 0.985x (0.882-1.100) | 1.040x (1.016-1.066) |
| 2 | 3840x2160/A7/L2 | noise | 34.852 | 34.382 | 9.253 | 8.006 | 1.016x (0.947-1.081) | 1.215x (1.165-1.268) |
| 2 | 3840x2160/A7/L2 | low contrast | 35.436 | 35.908 | 10.531 | 9.583 | 0.971x (0.903-1.037) | 1.101x (1.014-1.180) |

All eight 4K resident intervals exclude one. Aperture-3 bars give 1.136x and
1.182x; aperture-5 bars give 1.149x and 1.040x; aperture-7 noise gives 1.210x and
1.215x; aperture-7 low contrast gives 1.082x and 1.101x. Completed resident calls
include controls, binding/encoding, scratch acquisition and device completion.
Ratios are not attributed solely to shader throughput or memory bandwidth.
Aperture 3 removes only an eight-byte dummy, so its result especially cannot be
explained as bandwidth improvement from a large row-memory saving.

At 4K, both routes retain 66,355,200 bytes for 5/7 under the unchanged 64 MiB
policy; aperture 3 changes from 66,355,208 to 66,355,200 cached bytes. The budget
does not retain additional full-size 33,177,600-byte pixel buffers. The new route
removes one such live allocation/lease for 5/7 and one small buffer/binding for
3. Driver/resource validation and completion behavior are part of these public
call timings. Changing retention policy or device can change results. Requested
private bytes are not a claim of equivalent total process VRAM or cache savings.

Every 4K full-call interval includes one; no repeated full-call improvement is
established. Full HD aperture-7 noise has adverse full estimates 0.981x and
0.968x, with survey 2's interval below one (0.925-0.998). Full HD aperture-5 noise
has an adverse survey-1 resident estimate 0.926x (0.800-1.011), then an inconclusive
1.062x repeat. Small/Full HD resident intervals include one throughout. The lone
Full HD aperture-5 bar full interval above one does not repeat. These adverse
cases remain documented; no blanket full-call, small-image or general-device
speedup is claimed. The exact working-memory saving is independent of timing.

Private raw/derived artifacts remain outside Git:

- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-reuse-horizontal-state-benchmark-1.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-reuse-horizontal-state-benchmark-2.log`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-reuse-horizontal-state-measurements.json`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/analyze-gpu-canny-reuse-horizontal-state.py`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/run-gpu-canny-reuse-horizontal-state-survey.ps1`
- `C:/Users/rsasa/Workspace/SpatialRust/target/automation/gpu-canny-state-flags-historical-recheck-benchmark-1.log`

Source paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny.wgsl`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_frozen.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_reuse_rows_tests.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/canny_reuse_rows_benchmark.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/tests/fixtures/canny_reuse_rows_previous.wgsl`

From `C:/Users/rsasa/Workspace/SpatialRust`, compile the release binary with
--no-run first, then run this exact test alone twice sequentially:

```text
cargo test -p spatialrust-gpu --features gpu-image --release --lib image::canny::tests::reuse_rows_tests::benchmark::completed_reuse_row_calls -- --include-ignored --exact --nocapture --test-threads=1
```
