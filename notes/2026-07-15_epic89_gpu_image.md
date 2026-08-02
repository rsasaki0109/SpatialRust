# Epic 89 GpuImage / wgpu vision completion record

Date: 2026-07-15 (Asia/Tokyo)

## Delivered contracts

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\gpu_image.rs`
  owns `GpuImage` and `GpuImageReceipt` with explicit upload/readback/recycle.
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\kernels\`
  implements device-resident `copy_gpu_image`, BT.601 `rgb_to_gray_gpu`, and
  gray `box_blur_gpu`.
- Feature flag: `spatialrust-gpu/gpu-image` and facade `gpu-image`.
- Storage layout v1 uses one `u32` word per interleaved component; textures are
  intentionally deferred.

## Verification

- `cargo test -p spatialrust-gpu --features gpu-image --lib`: image chain tests
  assert mid-pipeline `device_to_host_bytes == 0`, gray matches CPU BT.601, and
  upload/copy/readback round-trips packed RGB.
- Criterion bench `gpu_image_pipeline` covers 640p/1080p/4K upload and
  gray→box-blur chains.

## Notes

CPU `spatialrust-vision` remains the numerical baseline. Kernel APIs never take
only host images unless the call is an explicit upload.
