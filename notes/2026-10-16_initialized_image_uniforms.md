# Initialize image compute uniforms at creation

Parent: `f00d12e6e7cdff134ab8767e5837cd00bc79f2b0` (merged PR 241).

The shared resize/Sobel executor and CHW tensor packer now use
`DeviceExt::create_buffer_init` with `UNIFORM` usage. Previously each created
an unmapped `UNIFORM | COPY_DST` buffer and then called `queue.write_buffer`.
The same parameter bytes are initialized during creation and unmapped before
the original bind/encode/submit sequence. This matches the existing blur and
morphology uniform initialization pattern.

## Source-level request ledger

These are source API requests, rather than physical allocations, backend copies
or measured latency. Parameter delivery still occurs; it has moved to initial
mapped contents. The locked wgpu 24.0.5 `DeviceExt` implementation rounds buffer
sizes to four-byte alignment, creates a mapped buffer, copies the contents and
unmaps it. Both original structs are already aligned to four-byte lengths.

| Boundary | Uniform bytes | Explicit queue writes before / after | Uniform buffers before / after | Initial mapped writes before / after |
| --- | --- | --- | --- | --- |
| Resize | 32 | 1 / 0 | 1 / 1 | 0 / 1 |
| Sobel | 32 | 1 / 0 | 1 / 1 | 0 / 1 |
| CHW pack | 64 | 1 / 0 | 1 / 1 | 0 / 1 |
| Resize -> Sobel -> CHW | 128 total | 3 / 0 | 3 / 3 | 0 / 3 |

Every uniform remains private to its dispatch. Original struct definitions,
zero padding, parameter construction, capability/overflow/runtime/profiler
validation and error ordering are unchanged. The public APIs, kernels, image
textures, tensor buffers, pools, output ownership, receipts and pipeline caches
are unchanged. The example chain retains three compute passes and three
explicit submissions. This introduces no explicit device readback or host wait.
Image and tensor payload transfer accounting is unchanged; existing receipts
exclude these internal control uniforms. Backend initialization can still use
internal staging/copies. No reduction in physical transfer, peak memory or
GPU execution time is established.

## Correctness and validation

The added native regression submits 96 complete public chains: eight source /
destination shape pairs, all four channel counts and three different exact
dyadic normalization presets. Grayscale chains include Sobel; two-, three- and
four-channel chains pack resized images directly. Shapes cover single rows /
columns, odd edges, 16-wide workgroups and the AI packer's 256x256 selection
boundary. The independent CPU oracle uses integer nearest-neighbor sampling,
replicate-border Sobel masks and channel-major scalar normalization. All
4,005,210 f32 words match by raw bits.

Twenty-four tensors per channel count remain alive while subsequent calls vary
dimensions and coefficients and recycle source/intermediate textures. Reverse
readback after idle pool clearing checks parameter isolation, metadata, exact
shape, stages and transfer receipts. Owned results remain valid after tensor
recycling and buffer-pool clearing. Existing image regressions additionally
cover arbitrary normalization/overflow, strided ROI, invalid limits, runtime
identity, profiler capacity, timestamps and complete vision-chain behavior.

Validation in `C:/Users/rsasa/Workspace/SpatialRust`:

- All-features/all-targets Clippy with `-D warnings`.
- 132 image, 168 voxel, 18 filtering and 2 facade tests: **320 native tests**.
  Native GPU tests ran on a headless adapter, including ignored
  correctness tests; benchmark tests were excluded.
- CPU-only build, minimal `gpu-wgpu` WASM build and CI
  `gpu-image,gpu-aoso-staging` WASM build, with WASM warnings denied.
- Workspace formatting and diff whitespace checks.
- Source audit: 475 original GPU/image Rust/WGSL files unchanged. The other
  three files differ only in the two uniform initializers, the `DeviceExt`
  import and native regression registration. Shaders, production validation,
  output/pool/profiler/receipt/submit/readback bodies remain exact.
- 480 source/executable hashes pin the validated inputs. No local adapter name
  appears in tracked files or delivery text.

No latency benchmark was performed for this small initialization cleanup.
Removing explicit queue-write requests is a source-level result; speed,
physical allocation and backend transfer changes remain unmeasured.

Implementation paths:

- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/kernels/spatial.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/ai_tensor.rs`
- `C:/Users/rsasa/Workspace/SpatialRust/crates/spatialrust-gpu/src/image/uniform_tests.rs`

Ignored local receipts and logs reside in
`C:/Users/rsasa/Workspace/SpatialRust/target/automation/image-uniforms-*`.
