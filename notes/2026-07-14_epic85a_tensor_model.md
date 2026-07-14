# Epic 85A tensor data model

Date: 2026-07-14

## Artifacts

- Crate: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-tensor`
- Architecture: `C:\Users\rsasa\Workspace\SpatialRust\docs\ARCHITECTURE.md`
- Roadmap: `C:\Users\rsasa\Workspace\SpatialRust\docs\ROADMAP.md`

## Decisions

The generic tensor crate is separate from
`spatialrust-core::SpatialTensor`, whose established responsibility is chunked
point-cloud iteration. `TensorDescriptor` represents dtype, shape, optional
signed element strides, byte offset, and device. `TensorView` is lifetime-bound
and zero-copy; `TensorBuffer` owns CPU storage; `to_owned_copy` is intentionally
named so a copy cannot be hidden.

Host slices accept CPU and explicitly pinned host device categories. CUDA,
ROCm, Vulkan, Metal, WebGPU, and other device allocations require future
backend-specific handles and explicit transfer APIs.

The model follows the official DLPack major-version 1 ABI concepts: native-endian
dtype code/bits/lanes, element rather than byte strides, byte offset, device
identity, and scalar rank zero. The official header is currently 1.3; DLPack
FFI will remain a separate audited feature, enforce major-version compatibility,
and reject unsupported later-minor dtype/device codes rather than guessing.

## Verification

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" test -p spatialrust-tensor --all-features
& "$env:USERPROFILE\.cargo\bin\cargo.exe" clippy -p spatialrust-tensor --all-features --all-targets -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo.exe" check -p spatialrust --no-default-features --features tensor
```

## 85B image and spatial bridges

Packed `Image<T, C>` storage is borrowed as HWC and packed `PlanarImage<T, C>`
storage as CHW without changing the allocation pointer. Padded and ROI views
must call `pack_interleaved_image` or `pack_planar_image`, making allocation and
copying visible at the call site. Existing `spatialrust-core::SpatialTensor`
exposes individual `f32` Schema-SoA fields without silently interleaving them.

The packing benchmark is
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-tensor\benches\image_bridge.rs`
and covers 640×480, 1920×1080, and 3840×2160.
