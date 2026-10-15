# Borrow packed RGBA pixels during explicit GPU uploads

This named slice removes redundant framework host packing for already packed
four-channel images. It changes input preparation, not algorithm placement.

`ImageView::as_packed_slice(&self)` returns exactly the logical elements with
the backing storage lifetime, without requiring scalar `Copy` or `Clone`.
Empty views return an empty slice before dimension multiplication. Single-row
views qualify regardless of stride; multiple rows require `width * CHANNELS`
stride. Checked arithmetic and slice bounds keep the accessor safe. Dense
height crops qualify; narrow multi-row crops with parent-row gaps do not.
Trailing allocation capacity, row padding and neighboring ROI pixels are
excluded.

`GpuImage::upload_u8` keeps the parent planner and creates the texture in the
same order. For qualifying four-channel views a borrowed `Cow<[u8]>` replaces
the previous zero-filled host vector and row copy. Other views call the same
`pack_rgba8` function. The existing queue write consumes the bytes before the
call returns; uploaded outputs remain independent of later input mutation or
drop. No unsafe code, dependency, GPU pass, wait, readback or device transfer
is added. Backend staging still exists; this is not a device zero-copy API.
The four physical bytes per pixel, metadata, ownership, texture pools and
explicit transfer receipts are unchanged.

The removed framework payload is exactly `4 * width * height` bytes per
qualifying call (31.640625 MiB at 3840x2160). This is an allocation removed
over construction, not a measured process peak-memory reduction.

## Validation and measurement boundary

The test fixture freezes the complete uploader from parent merge
`671c76f478cf67ad5453afc98a3fd0484c84548b`; only its test method name changes.
Both upload routes use the same current runtime, planner, pool and packing
implementation. Generic accessor tests check pointer identity, exact logical
length, scalar bit patterns, trailing storage, row padding, ROI neighbors,
one-row maximum stride, empty maximum dimensions and non-Copy scalars.
Native tests compare logical bytes, full physical RGBA bytes, unused zero
lanes, metadata and repeated transfer receipts across all logical channel
counts. Input mutation/drop precede download, and concurrent packed uploads
keep independent live outputs before recycling.

The paired benchmark compares host payload preparation separately from
completed public uploads (API call, empty submit to flush writes, and
`wait_idle`). Payload drop is outside the host timer. Upload readback,
correctness assertions and recycling are outside the upload timer. Host
allocation, texture acquisition, backend staging and completion remain inside
the upload timer. There is no GPU kernel in this measurement and no claim
about complete image-processing pipeline latency.

The additional native checks passed 117,638,192 physical/logical byte
comparisons across 36 fixtures (1x1, odd 65x3, 1024x1025; packed, strided ROI,
and one-row ROI; channels 1..=4), with repeated readbacks. Four concurrent
callers each retain two independent uploaded outputs after dropping their
host storage. The payload selection test also checks borrowed pointer identity
and maximum single-row stride without creating a giant allocation.

The benchmark fixtures are dense RGBA at 64x64, 257x129, 1024x1024,
1920x1080, 3840x2160 and 1023x1025; an 8191x1 row ROI; a strided
1920x1080 RGBA ROI; dense 1920x1080 RGB and 1024x1024 gray controls.
Each route has three warmups and 30 paired samples with alternating order.
The isolated-run controller required three quiet samples five seconds apart,
then checked for competing build/test/benchmark and identifiable Python
computation before, during and after the child. Idle HTTP servers were
excluded; unrelated jobs were never terminated. The first two attempts
started but detected overlap and cancelled only their own child. Neither
completed and neither contributes timing evidence. Subsequent attempts could
not obtain a quiet window before the 25-minute deadline. Accepted surveys: 0.
The second survey and statistical analysis were not run. Raw interrupted
attempts and pinned hashes remain in ignored local automation artifacts.

**No latency improvement is claimed.** In particular, neither host preparation
ratios nor completed-upload speedups are established by this change. Pointer
identity in the borrowed path and source comparison against the frozen
complete parent establish removal of the framework payload allocation/copy;
correctness tests establish byte/metadata/receipt and input-lifetime behavior.
This scope remains useful without attributing any speed to noisy or incomplete
measurements. Backend staging, transfer bytes and process peak memory remain
outside the allocation claim.

The reusable fixture is configured for two sequential surveys, each with the
10 shapes/layouts above and separate host/completed-upload routes. Planned
analysis uses paired log old/new latency ratios and 10,000 bootstrap resamples,
sorted indices 249/9749 for the 95% interval, with seeds
2026101509 + 100*survey + 10*record_index + JSON_field_index. Ratios above one
favor the borrowed path; intervals crossing one are inconclusive. Any zero
sample suppresses the ratio for that row. All rows, including controls,
adverse results and inconclusive results, must be retained. This methodology
is provided for reproduction, not as evidence that those surveys completed.

## Completed validation

- 13 image-model tests, including the five new generic packed-slice cases.
- 110 native image regression tests, including the three new upload tests.
- Four public facade tests (median, bilateral and two image-transfer cases).
- Image-model Clippy and GPU all-feature/all-target Clippy with warnings denied.
- CPU-only GPU crate and `gpu-image` facade WASM builds; formatting and diff checks.
- 347 source/executable hashes pinned after validation and verified after the
  measurement attempt. Against the parent, 339 other preexisting sources,
  including all 97 WGSL files, are unchanged. The frozen complete uploader
  differs only in test visibility/name; production planner, write and receipt
  body differ only at payload selection. The image-view addition leaves all
  other preexisting data-model behavior unchanged.
- Native validation used Windows x86_64 and rustc 1.98.1
  (48a229cea, 2026-09-01). No accepted latency samples are published.
