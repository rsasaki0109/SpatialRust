# Visualization guide

SpatialRust Visual is an opt-in inspection stack for point clouds, algorithm
results, reconstructed scenes, synchronized RGB-D data, and bounded remote
datasets. It does not add rendering or UI dependencies to `spatialrust-core`.

## Choose a surface

| Need | Feature | Primary API |
| --- | --- | --- |
| Borrowed geometry, cameras, styles, layers | `viz` | `spatialrust::viz` |
| Explicit wgpu upload and headless rendering | `render-wgpu` | `spatialrust::render_wgpu` |
| Portable viewer state and controls | `viewer` | `spatialrust::viewer` |
| Native winit shell | `viewer-native` | `NativeViewer` |
| Scene/mapping/RGB-D/semantic adapters | `viewer-full` | viewer adapter functions |
| Bounded hierarchy and COPC LOD | `viewer-lod` | `spatialrust::lod` |
| Browser state and bounded ranges | `web` / `web-wasm` / `web-webgpu` | `spatialrust::web` |
| CPython native viewer and NumPy sources | Python wheel | `spatialrust.ViewerState` |
| Notebook Web viewer | `spatialrust-jupyter` | `ViewerWidget` |

Default builds enable none of these features. Native windows, WebAssembly,
Python, codecs, ROS 2, ONNX, and CUDA remain independent.

## Rust quick start

Use borrowed structure-of-arrays columns to describe geometry. Creating a
`PointCloudView` does not allocate or upload:

```rust
use spatialrust::viz::{PointCloudView, PositionColumns3, VisualPrimitive};

let x = [0.0_f32, 1.0];
let y = [0.0_f32, 0.0];
let z = [0.0_f32, 0.0];
let positions = PositionColumns3::try_new(&x, &y, &z)?;
let primitive = VisualPrimitive::Points(PointCloudView::positions_only(positions));
```

With `render-wgpu`, the caller creates a `WgpuRuntime`, calls
`WgpuRenderer::upload`, and retains the returned `GpuGeometry`. Upload,
render-uniform upload, screenshot readback, and picking each return separate
byte-exact transfer receipts. Rendering to a device target does not read it
back.

Run the fail-closed canonical image check:

```text
cargo run -p spatialrust --no-default-features --features render-wgpu \
  --example visual_headless_conformance
```

The fixture requires an adapter, compares the entire 64×64 RGBA image by a
stable hash, and verifies the exact 12-byte geometry upload, 112-byte frame
upload, and 16,384-byte requested readback. Absence of an adapter is a failure.

## Native inspection and debug overlays

`viewer` contains deterministic state, controls, layer presentation, attribute
inspection, RGB-D timelines, and owned debug overlays. `viewer-native` adds the
winit event shell. It does not choose a renderer or upload geometry. Normal,
voxel, plane, cluster, correspondence, bounds, and search-radius overlays have
stable layer identities and expose generated-byte receipts.

Scene features adapt meshes, surfels, Gaussians, trajectories, pose graphs,
camera frusta, and semantic centroids. Adapters either preserve a borrowed
source or report every byte generated while transposing or synthesizing
geometry.

## Bounded large-cloud viewing

`spatialrust-lod` plans from a validated hierarchy and camera. Configure hard
point, host-memory, GPU-memory, per-frame upload, and in-flight request limits.
Admission fails before a limit is exceeded. Host chunks use drop-scoped leases;
GPU chunks use protected deterministic LRU eviction. The optional COPC adapter
only creates a bounded query—the caller still performs IO and reports upload
completion.

## Web, Python, and Jupyter

All adapters carry the same versioned viewer-state envelope.

- Web requests require an admitted exact byte range, abort support, a `206`
  response, exact body length, and bounded cache admission.
- Python zero-copy point sources retain contiguous `float32` X/Y/Z owners and
  preserve pointer identity. The owned path is explicitly named and returns a
  copy receipt.
- `spatialrust-jupyter` uses an AnyWidget iframe transport with exact origin,
  source-window, and protocol-version checks. Frontend state is revalidated by
  the Rust binding.

See [VISUAL_MIGRATION.md](VISUAL_MIGRATION.md) for adoption and
[VISUAL_RELEASE_RECEIPT.md](VISUAL_RELEASE_RECEIPT.md) for the aggregate gate.
