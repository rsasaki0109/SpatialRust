# Epic 104: texture-backed GPU Image v2

`GpuImage` now owns a pooled `rgba8uint` 2D texture while retaining a logical
1–4 channel count and semantic image metadata. Physical storage is four bytes
per pixel instead of four bytes per component. Host/device movement remains
limited to the explicitly named `upload_u8` and `readback_u8` calls.

Texture kernels cover copy, RGB-to-gray, mean box filtering, nearest resize,
Sobel L1 magnitude, erosion, and dilation. The integration chain
`upload → resize → gray → blur → Sobel → dilate → readback` records no D2H bytes
until the final readback and preserves ordered stage names. Known-pixel tests
exercise resize, flat-image Sobel, and impulse morphology on a real headless
wgpu adapter.

The runtime records adapter name/backend/device class/driver, exposes an
explicit `wait_idle` synchronization boundary, and pools up to eight textures
per resolution. Image pipelines are cached per runtime/device rather than in a
process-global slot, so multiple adapters cannot cross-bind GPU resources.
Pooling was validated under thousands of Criterion iterations;
it prevents the deferred-destruction VRAM growth observed during the first
allocation-heavy benchmark design.

On the reference low-power adapter, synchronized medians were:

| Workload | VGA | 1080p | 4K |
| --- | ---: | ---: | ---: |
| RGBA texture upload | 1.186 ms | 8.030 ms | 28.934 ms |
| upload + gray + 5x5 box blur | 2.648 ms | 14.081 ms | 60.774 ms |
| resident resize + gray + blur + Sobel + dilate | 0.963 ms | 3.504 ms | 13.523 ms |

The resident chain resizes to half width/height before later stages. These are
machine-local Criterion results, not a blanket cross-adapter performance claim.
