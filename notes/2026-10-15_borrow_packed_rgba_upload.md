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
