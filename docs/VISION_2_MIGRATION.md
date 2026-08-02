# Migrating performance-sensitive vision pipelines to Vision 2

Vision 2 is an additive performance and evidence milestone over the stable
Vision 1 ownership model. Existing CPU entry points remain CPU-only. New plan,
workspace, caller-output, fused preprocessing, and GPU-resident APIs make reuse
and transfer costs explicit without changing the semantics of the generic
fallbacks.

## Prefer reusable CPU execution

- Construct `BilinearResizeU8Plan`, `NearestResizeU8Plan`, or
  `AreaResizeU8Plan` when dimensions repeat.
- Use `*_into` entry points with caller-owned output for steady-state loops.
- Retain Gaussian, morphology, and Canny workspaces across frames.
- Use fused resize-to-gray or resize-normalize-CHW only when that exact
  downstream representation is required. Standalone APIs remain valid.
- Packed specializations may dispatch to SIMD or bounded row parallelism.
  Strided views and unsupported shapes retain safe generic fallbacks.

## Keep device placement explicit

Start a GPU chain with `GpuImage::upload_u8`. Use
`run_gpu_vision_chain` when resize, grayscale, blur, Sobel, morphology, and CHW
packing should remain resident. `GpuAiTensor::readback_f32` is the explicit
optional host boundary. Validate a no-readback path with
`GpuImageReceipt::validate_resident_chain`.

No CPU API selects wgpu implicitly, and production APIs do not hide host/device
copies.

## Accuracy and performance claims

Vision 2 does not promise one universal OpenCV win. Each claim is scoped to its
documented input shape, allocation mode, host, thread policy, and correctness
gate. Resize, color, and Gaussian use their documented error bounds; Sobel and
morphology retain exact gates; Canny retains exact binary metrics on the named
workloads.

Before adopting a specialization, run the focused receipt harness for the
production shapes. Treat `docs/ROADMAP.md` and dated notes as the authoritative
scope of measured claims.

## Release evidence

`Vision2ReleaseGate` requires:

- Rust/Python accuracy and Linux/Windows/macOS conformance;
- native and Python allocate/reuse performance measurements;
- explicit peak-memory, allocation-count, worker-thread, and GPU-transfer
  measurements;
- the dated component/kernel/GPU receipts;
- Pages documentation, unsafe audit, this migration policy, and the runnable
  `vision_2_release_gate` example.

Missing, skipped, duplicated, or over-budget evidence denies the release and
returns all applicable reasons.
