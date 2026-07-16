# Epic 119 GPU-resident vision chain — 2026-07-16

## Scope

Epic 119 adds an explicitly GPU-resident Vision 2 chain:

`upload → nearest resize → RGB-to-gray → box blur → Sobel → morphology → normalized CHW f32`

The public entry point is `run_gpu_vision_chain`. The caller owns the uploaded
`GpuImage`; the returned `GpuAiTensor` remains device-resident until the caller
requests `readback_f32`. CPU APIs are unchanged and never select the GPU
implicitly.

Production files:

- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\vision_chain.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\ai_tensor.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\image\gpu_image.rs`
- `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-gpu\src\runtime.rs`

## Correctness and transfer gate

The focused real-GPU test records these seven ordered stages:

1. `upload_u8_texture`
2. `resize_nearest_gpu`
3. `rgb_to_gray_gpu`
4. `box_blur_gpu`
5. `sobel_gpu`
6. `morphology_gpu`
7. `pack_ai_chw_gpu`

For a 32×24 RGB input, physical upload accounting is exactly 3,072 bytes
(RGBA8 texture storage), with zero device-to-host bytes before explicit
readback. `GpuImageReceipt::validate_resident_chain` rejects an incorrect
upload budget and rejects the same receipt after an explicit tensor readback.
The returned 1×12×16 tensor contains finite normalized values.

The chain composes the existing individually tested GPU kernels. Resize,
fixed-point BT.601 gray, rounded box blur, clamped L1 Sobel, and morphology
retain their existing known-pixel/CPU-reference tests. Public code remains safe
Rust; no `unsafe` block was added.

## Steady-state reuse

Four per-runtime pipeline families are initialized: gray, box blur, spatial
(resize/Sobel/morphology), and AI packing. Intermediates are returned to the
texture pool immediately after their consuming submission. After two warm-up
chains, a third chain leaves both cached texture and storage-buffer counts
unchanged. Tensor storage is acquired from and returned to the existing
size-keyed GPU buffer pool.

## Synchronized benchmark receipt

Command:

```powershell
$env:PATH = "C:\Users\rsasa\.cargo\bin;$env:PATH"
cargo bench -p spatialrust-gpu --features gpu-image --bench gpu_image_pipeline -- --quick
```

The 2026-07-16 host was Windows NT 10.0.26300.0 with an Intel Core i7-9750H.
The low-power headless adapter selected by `WgpuRuntime` was Intel UHD Graphics
630 through Vulkan (`IntegratedGpu`, Intel driver). Criterion used the
optimized build, one explicit `wait_idle` per GPU-resident iteration, and
packed deterministic RGB8 input. Output dimensions were half the input width
and height. CPU is the matching scalar reference; GPU round-trip includes
upload and final CHW readback; GPU resident excludes both from the timed loop
after one caller upload.

| Profile | CPU chain | GPU round-trip | GPU resident | Resident vs CPU |
| --- | ---: | ---: | ---: | ---: |
| VGA 640×480 | 4.179 ms | 3.949 ms | 1.696 ms | 2.46× |
| 1080p | 32.197 ms | 15.618 ms | 7.423 ms | 4.34× |
| 4K | 125.610 ms | 59.864 ms | 16.005 ms | 7.85× |

These are short same-host Criterion `--quick` measurements, not portable
latency guarantees. They demonstrate the cost separation required by 119D:
at these three profiles the resident loop measured 2.33×, 2.10×, and 3.74×
faster than the corresponding GPU round-trip loop.

## Verification

```powershell
cargo test -p spatialrust-gpu --features gpu-image --lib
cargo clippy -p spatialrust-gpu --features gpu-image --all-targets -- -D warnings
cargo check -p spatialrust-gpu --features gpu-image --bench gpu_image_pipeline
```

Observed result: 32 unit tests passed, including the real-GPU transfer and
steady-state denial test; clippy and benchmark compilation passed.
