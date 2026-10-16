# Direct mapped histogram and CDF readback

Parent: `8da1fd736afc1fe4fa66e62d494eb9793c97990a` (merged PR 246).

The shared native histogram/CDF reader previously mapped the 1,024-byte staging
buffer, copied its bytes into an owned vector, unmapped, then decoded that vector
into `[u32; 256]`. Decode the borrowed mapped bytes directly into the same fixed
array with the existing `with_staging_bytes` lifetime helper. The original
`u32::from_le_bytes` loop remains; there is no alignment cast or float conversion.

The mapped view drops before the existing unmap guard on return or unwinding.
The new result is owned before the mapping is released. The original byte-copy
wrapper now has only test users and is compiled only for native tests; its body
and the shared map/unmap implementation are unchanged.

## Deterministic request and copy scope

| One successful native readback | Before | After |
| --- | --- | --- |
| Intermediate framework byte vectors | 1 | 0 |
| Bytes copied from the mapped view into that vector | 1,024 | 0 |
| Returned count array | 256 u32 | 256 u32 |
| Logical device-to-host receipt | 1,024 bytes | 1,024 bytes |
| Staging buffers / copy commands / submissions / map requests / `Maintain::Wait` polls | 1 each | 1 each |

Reading both a histogram and its CDF removes two intermediate host vectors and
2,048 intermediate host-copy bytes. This does not establish that the whole call
is allocation-free: wgpu, mapping channels and other backend/framework work can
allocate. The fixed result array still needs to be written and may be moved.
No device transfer, physical allocation, peak-memory, bandwidth or latency
improvement is claimed. No latency benchmark was performed for this cleanup.

## Preserved behavior

Both public methods retain their original runtime rejection, success-only
receipt updates, stage names, buffer ownership and recycling. Histogram and
inclusive CDF construction, shader bodies, dispatch choice, profiler paths and
the 256-bin integer layout are unchanged. Native synchronous readback remains
native-only. Existing GPU image/tensor and point-cloud paths are unchanged.

## Validation

- **335 native tests:** 135 image, 180 voxel/readback, 18 filtering and 2 facade,
  including ignored GPU correctness tests and excluding benchmarks.
- **81,920 original/CPU raw u32 word comparisons** cover 64 data
  presets with zero, `u32::MAX`, byte-order markers, high-bit and arbitrary words.
  Sixteen extra source words remain outside the readback prefix. Repeated reads
  after source updates agree, and earlier arrays remain owned after source drop
  and pool clearing.
- **153,600 original/CPU histogram/CDF word comparisons** cover
  all logical channels for one- through four-channel images, three dimensions
  and two strided view offsets. Repeated public reads preserve exact receipts,
  count/channel metadata, stage names and owned results after recycling. Wrong
  runtimes return the original errors without updating receipts or pools.
- Four concurrent callers retain **6,144 histogram/CDF words**
  across repeated readbacks after source/result recycling and runtime drop.
- Existing exact CDF scans through `u32::MAX`, ROI/alpha behavior, dirty-output
  reuse, map/unmap success/error/panic-remap and GPU profiler regressions pass.
- All-features/all-targets Clippy with warnings denied; CPU-only and minimal/CI
  WASM builds with WASM warnings denied; workspace formatting and whitespace.
- **486 unchanged original GPU/image Rust/WGSL files** and **492 validated
  source/executable hashes**. The complete original shared reader and both public
  readback methods are frozen in native tests. Source comparison verifies the
  same decode loop, staging/copy/submit/map/wait behavior, public methods,
  receipt updates, shaders and mapping lifetime helper. The only additional
  production compilation change makes the unused byte-copy wrapper test-only.

Repository: `C:/Users/rsasa/Workspace/SpatialRust`.
Implementation:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram.rs`.
Shared lifetime helper:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/gpu_image.rs`.
Native fixtures:
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram/mapped_readback_tests.rs`,
`C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/histogram/mapped_readback_reference.rs`.
Ignored receipts:
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/histogram-readback-*`.
The local adapter name is omitted from tracked files and delivery text.
