# Visual migration policy (`visual-1`)

This guide moves ad-hoc visualization code to the SpatialRust Visual contracts
without introducing hidden ownership or device transfers.

## 1. Separate data from presentation

Keep application point clouds and algorithm outputs in their owning crates.
Expose borrowed `PositionColumns3`, optional RGB/scalar columns, line lists, or
indexed triangles through `spatialrust-viz`. Do not add renderer handles,
window types, or interleaved presentation buffers to `spatialrust-core`.

Replace monolithic visualization point structs with capability-backed or
column-backed views. Validate lengths and indices when constructing the view.

## 2. Make device crossings explicit

Replace renderer constructors that silently consume host arrays with:

1. caller-created `WgpuRuntime`;
2. named `WgpuRenderer::upload`;
3. retained `GpuGeometry`;
4. device-resident render calls; and
5. optional, caller-requested screenshot or picking readback.

Store and audit every `TransferReceipt`. A render path that cannot explain its
host-to-device and device-to-host bytes does not conform to `visual-1`.

## 3. Move UI state to the portable model

Represent camera, viewport, layer presentation, selection, and revision with
`ViewerState`. Apply interactions through `ViewerController`/`BrowserInput`.
Persist the versioned Web envelope rather than serializing native window or GPU
objects. Reject unknown state versions; do not guess migrations.

Native window creation belongs behind `viewer-native`. Scene, mapping, RGB-D,
semantic, LOD, Web, and Python support each remain separately feature-gated.

## 4. Bound large and remote inputs

Do not materialize an entire COPC or remote source for display. Build a
validated `LodIndex`, set all `LodBudgets`, admit requests before IO, retain
host leases while borrowed, and report upload completion. For browsers, admit
`ByteRange` values through `RangePlanner` before fetch and cache admission.
Cancellation must release reservations.

## 5. Choose Python ownership intentionally

For zero-copy Python geometry, supply contiguous one-dimensional `float32`
X/Y/Z arrays and retain the returned `ViewerPointSource`; it keeps the NumPy
owners alive. For isolation from later mutation, use the explicit owned-copy
constructor and retain its byte receipt. Do not infer zero-copy from array
syntax alone.

Notebook messages must use the `spatialrust-jupyter` transport version and an
exact configured origin. State changes are accepted only after Rust validation.

## Compatibility promise

The backend-independent `spatialrust-viz` geometry, camera, style, layer, and
transfer-receipt contracts are Stable for Visual 1. Renderer, viewer, LOD, Web,
Python, and Jupyter adapters are Provisional and may evolve with notice.
Breaking Stable changes require the normal SpatialRust major-version policy.

Release evidence acknowledges this document with migration policy identifier
`visual-1`.
